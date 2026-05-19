"""Pluggable label embedders for the Grounding Planner.

The slow module needs a vector per object-class label. Two implementations:

  - `HashLabelEmbedder`     md5-seeded random vec. Deterministic, no extra deps.
                            Used by default for fast unit tests and as the
                            Day-2 placeholder. Captures *nothing* about
                            semantic similarity ("dog"/"puppy" stay orthogonal).

  - `CLIPTextLabelEmbedder` CLIP text-encoder embedding. Brings open-vocab
                            generalization: semantically close labels share a
                            neighbourhood. Larger embed_dim (e.g. 512 for
                            ViT-B-32, 768 for ViT-L-14); the GP's `node_proj`
                            layer absorbs the dim change automatically.

Both implement the `LabelEmbedder` protocol so the GP only depends on the
abstract surface — swap is one constructor argument.
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor


class LabelEmbedder(ABC):
    """Per-label vector lookup. Implementations are expected to cache."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of vectors returned by `embed`."""

    @abstractmethod
    def embed(self, label: str) -> Tensor:
        """Return a CPU tensor of shape [embed_dim] for `label`.

        Must be deterministic: repeated calls with the same label return the
        same tensor (call sites rely on this for stable featurization).
        """


# ── deterministic-hash baseline (the Day-2 default) ──────────────────────────

class HashLabelEmbedder(LabelEmbedder):
    """Md5-seeded pseudo-random embedding. Stable across processes.

    Used as the default so unit tests stay CPU-only and dependency-free.
    Carries no semantic structure — every label is orthogonal in expectation.
    """

    def __init__(self, embed_dim: int = 128, scale: float = 0.02) -> None:
        self._embed_dim = embed_dim
        self._scale = scale
        self._cache: dict[str, Tensor] = {}

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @torch.no_grad()
    def embed(self, label: str) -> Tensor:
        cached = self._cache.get(label)
        if cached is not None:
            return cached
        digest = hashlib.md5(label.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big")
        g = torch.Generator()
        g.manual_seed(seed)
        feat = torch.randn(self._embed_dim, generator=g) * self._scale
        self._cache[label] = feat
        return feat


# ── CLIP-text-encoder embedder ───────────────────────────────────────────────

class CLIPTextLabelEmbedder(LabelEmbedder):
    """Encodes labels with CLIP's text encoder; caches per label.

    The CLIP model is loaded lazily on the first `embed()` call. For tests,
    pass a pre-loaded `model` + `tokenizer` (any objects exposing the
    open_clip API) via the `_preloaded` kwarg to bypass the network load.

    Output is L2-normalized in the CLIP text space so downstream cosine /
    dot-product operations are well-scaled.
    """

    # Known embed dims for the common open_clip pretrained models. Lets us
    # answer `embed_dim` without loading the actual model when the user
    # specifies a recognized configuration. Defer to a probe load only when
    # the model is unknown and no explicit override is supplied.
    _KNOWN_DIMS: dict[str, int] = {
        "ViT-B-32": 512,
        "ViT-B-16": 512,
        "ViT-L-14": 768,
        "ViT-L-14-336": 768,
        "ViT-H-14": 1024,
    }

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str | None = None,
        embed_dim: int | None = None,
        _preloaded: dict[str, Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._pretrained = pretrained
        self._device = device
        self._preloaded = _preloaded
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        # Resolve embed_dim eagerly when possible: explicit > _preloaded probe
        # via a single dummy encode > known-model lookup > defer to load-time probe.
        self._embed_dim: int | None = self._resolve_embed_dim_lazy(embed_dim)
        self._cache: dict[str, Tensor] = {}

    @property
    def embed_dim(self) -> int:
        if self._embed_dim is None:
            self._ensure_loaded()
        return self._embed_dim  # type: ignore[return-value]

    def _resolve_embed_dim_lazy(self, override: int | None) -> int | None:
        """Try to determine embed_dim without triggering a model load.

        Returns None when the dim cannot be settled — `_ensure_loaded` will
        then probe at first use.
        """
        if override is not None:
            return int(override)
        if self._preloaded is not None:
            # `_preloaded["model"].encode_text` of a 1-token input gives us dim
            # without hitting the open_clip loader. We don't run encode_text
            # here either — keep it lazy. None ⇒ load-time probe.
            return None
        return self._KNOWN_DIMS.get(self._model_name)

    @torch.no_grad()
    def embed(self, label: str) -> Tensor:
        cached = self._cache.get(label)
        if cached is not None:
            return cached
        self._ensure_loaded()
        # open_clip tokenizers accept a list of strings.
        tokens = self._tokenizer([label])
        device = self._resolve_device()
        if hasattr(tokens, "to"):
            tokens = tokens.to(device)
        feats = self._model.encode_text(tokens)
        if feats.dim() == 2:
            feats = feats[0]
        feats = feats.detach().to(dtype=torch.float32, device="cpu")
        # L2-normalize so dot-products are bounded in [-1, 1].
        feats = torch.nn.functional.normalize(feats, dim=-1)
        self._cache[label] = feats
        return feats

    # ── internals ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._preloaded is not None:
            self._model = self._preloaded["model"]
            self._tokenizer = self._preloaded["tokenizer"]
        else:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(
                self._model_name, pretrained=self._pretrained,
            )
            self._model = model.to(self._resolve_device()).eval()
            self._tokenizer = open_clip.get_tokenizer(self._model_name)
        # Probe the embed dim only when not already known (unknown model + no override).
        if self._embed_dim is None:
            with torch.no_grad():
                probe_tokens = self._tokenizer(["a"])
                if hasattr(probe_tokens, "to"):
                    probe_tokens = probe_tokens.to(self._resolve_device())
                probe = self._model.encode_text(probe_tokens)
                self._embed_dim = int(probe.shape[-1])

    def _resolve_device(self) -> str:
        if self._device is not None:
            return self._device
        return "cuda" if torch.cuda.is_available() else "cpu"


__all__ = ["CLIPTextLabelEmbedder", "HashLabelEmbedder", "LabelEmbedder"]
