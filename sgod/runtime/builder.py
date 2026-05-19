"""Config-driven assembly of (Backbone, Oracle, Policy) into a HallucinationDecoder.

The YAML schema:

    backbone:
      name: <registry-name>            # e.g. "llava-1.5-7b"
      <kwargs forwarded to constructor>

    oracle:
      name: <registry-name>            # e.g. "reltr" or "grounding-dino"
      <kwargs forwarded to constructor>

    policy:
      name: <registry-name>            # "dt-sgod" or "sgod-v1"
      config: {...}                    # policy-specific dataclass kwargs
      label_embedder:                  # DT-SGOD only; ignored for SGOD v1
        type: "hash" | "clip"
        <type-specific kwargs>
      clip_factory:                    # SGOD v1 only
        model_name: ...
        vocab_cache_path: ...

    runtime:
      max_new_tokens: int
      temperature: float
      hidden_buffer_size: int

The builder is dispatched per-policy because each policy has different
construction needs (DT-SGOD wants dims from the backbone; SGOD v1 wants the
backbone's tokenizer + a CLIP factory). The dispatch is contained here so the
rest of the runtime stays policy-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sgod.core.interfaces import Backbone, Oracle, Policy
from sgod.core.registry import build as registry_build
from sgod.runtime.decoder import HallucinationDecoder


# ── public API ───────────────────────────────────────────────────────────────

def build_from_config(config: dict, *, lazy: bool = True) -> HallucinationDecoder:
    """Build a HallucinationDecoder from a parsed YAML config dict.

    Args:
        config: Parsed config. See module docstring for schema.
        lazy:   Forwarded to backbones that support lazy model loading.
                When True, heavy model weights are loaded on first forward,
                not at construction — keeps tests and dry-runs fast.
    """
    _ensure_components_registered()
    backbone = _build_backbone(config["backbone"], lazy=lazy)
    oracle = _build_oracle(config["oracle"])
    policy = _build_policy(config["policy"], backbone=backbone)
    runtime_cfg = config.get("runtime", {}) or {}
    return HallucinationDecoder(backbone=backbone, oracle=oracle, policy=policy, **runtime_cfg)


def _ensure_components_registered() -> None:
    """Import concrete component packages so their @register decorators fire.

    Lazy-imported (not at module load) so fast unit tests that don't use
    build_from_config skip the torch/transformers import cost. Failures are
    swallowed because some components are optional — the actual KeyError
    raised by the registry lookup is the user-visible signal.
    """
    import importlib

    for mod in ("sgod.backbones", "sgod.oracles", "sgod.policies"):
        try:
            importlib.import_module(mod)
        except Exception:  # noqa: BLE001 — best-effort registration
            pass


def load_config(path: str | Path) -> dict:
    """Load a YAML config file."""
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


# ── component builders ───────────────────────────────────────────────────────

def _build_backbone(cfg: dict, *, lazy: bool) -> Backbone:
    cfg = dict(cfg)  # don't mutate caller
    name = cfg.pop("name")
    # Backbones that accept `lazy` (e.g. LLaVAv15Backbone) get the flag; the
    # registry-level kwarg pass-through ignores unknown keys via the class
    # signature, so we set it conservatively.
    cfg.setdefault("lazy", lazy)
    return registry_build("backbone", name, **cfg)


def _build_oracle(cfg: dict) -> Oracle:
    cfg = dict(cfg)
    name = cfg.pop("name")
    # Hybrid mode: RelTROracle + nested grounding_dino block builds a
    # GroundingDinoModule and threads it in as `gd_module`.
    gd_block = cfg.pop("grounding_dino", None)
    if gd_block:
        if not gd_block.get("enabled", True):
            gd_block = None
    if gd_block is not None and name == "reltr":
        cfg["gd_module"] = _build_grounding_dino_module(gd_block)
    return registry_build("oracle", name, **cfg)


def _build_grounding_dino_module(cfg: dict) -> Any:
    """Lazily import GroundingDinoModule so unit tests of unrelated configs
    don't require the transformers dep tree to be importable."""
    from sgod.sgg.grounding_dino_module import (
        GroundingDinoModule,
        default_vocabulary,
    )

    cfg = dict(cfg)
    cfg.pop("enabled", None)
    cfg.setdefault("vocab", default_vocabulary())
    return GroundingDinoModule(**cfg)


def _build_policy(cfg: dict, *, backbone: Backbone) -> Policy:
    cfg = dict(cfg)
    name = cfg.pop("name")
    if name == "dt-sgod":
        return _build_dt_sgod(cfg, backbone=backbone)
    if name == "sgod-v1":
        return _build_sgod_v1(cfg, backbone=backbone)
    # Fallback: registry-style construction. Used when a future policy is
    # added that doesn't need backbone-dependent wiring.
    return registry_build("policy", name, **cfg)


def _build_dt_sgod(cfg: dict, *, backbone: Backbone) -> Policy:
    from sgod.policies.dt_sgod import DTSGODConfig, DTSGODPolicy

    dt_config = DTSGODConfig(**(cfg.get("config") or {}))
    label_embedder = _build_label_embedder(cfg.get("label_embedder"))
    return DTSGODPolicy(
        hidden_dim=backbone.hidden_dim,
        vocab_size=backbone.vocab_size,
        config=dt_config,
        label_embedder=label_embedder,
    )


def _build_sgod_v1(cfg: dict, *, backbone: Backbone) -> Policy:
    from sgod.policies import SGODv1Policy

    clip_factory_cfg = cfg.pop("clip_factory", None) or {}
    cfg.pop("config", None)
    cfg.pop("label_embedder", None)
    clip_factory = _build_clip_factory(clip_factory_cfg)
    # `backbone.tokenizer()` may trigger lazy load; OK — SGOD v1 needs tokenizer
    # for top-K decoding anyway.
    tokenizer = backbone.tokenizer()
    return SGODv1Policy(tokenizer=tokenizer, clip_factory=clip_factory, **cfg)


# ── helpers ──────────────────────────────────────────────────────────────────

def _build_label_embedder(cfg: dict | None):
    """Build a LabelEmbedder per config block, or return None for default."""
    if not cfg:
        return None
    cfg = dict(cfg)
    kind = cfg.pop("type", "hash")
    if kind == "hash":
        from sgod.policies.dt_sgod import HashLabelEmbedder

        return HashLabelEmbedder(**cfg)
    if kind == "clip":
        from sgod.policies.dt_sgod import CLIPTextLabelEmbedder

        return CLIPTextLabelEmbedder(**cfg)
    raise ValueError(f"Unknown label embedder type: {kind!r}")


def _build_clip_factory(cfg: dict):
    """Return a callable image → CLIPScorer matching the legacy interface.

    When `cfg` is empty/None, defers to `sgod.utils.model_loader.load_clip_factory`
    with all defaults. This is the path SGOD v1 expects.
    """
    from sgod.utils.model_loader import load_clip_factory

    return load_clip_factory(**(cfg or {}))


__all__ = ["build_from_config", "load_config"]
