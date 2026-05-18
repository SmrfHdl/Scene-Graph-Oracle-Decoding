"""Lightweight string→class registry for backbones, oracles, and policies.

Lets configs reference components by short string names (`"llava-1.5-7b"`,
`"reltr"`, `"dt-sgod"`) without importing the implementing modules.

Usage:
    @register("backbone", "llava-1.5-7b")
    class LLaVAv15Backbone(Backbone): ...

    backbone = build("backbone", "llava-1.5-7b", checkpoint="...")
"""
from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")

_REGISTRY: dict[str, dict[str, type]] = {
    "backbone": {},
    "oracle": {},
    "policy": {},
}


def register(kind: str, name: str) -> Callable[[type[T]], type[T]]:
    """Decorator: register `cls` under (kind, name)."""
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown kind {kind!r}; expected one of {list(_REGISTRY)}")

    def _decorator(cls: type[T]) -> type[T]:
        if name in _REGISTRY[kind]:
            existing = _REGISTRY[kind][name]
            if existing is cls:
                return cls  # Idempotent: re-import of same module.
            raise KeyError(
                f"({kind!r}, {name!r}) already registered to {existing!r}; "
                f"refusing to overwrite with {cls!r}"
            )
        _REGISTRY[kind][name] = cls
        return cls

    return _decorator


def build(kind: str, name: str, **kwargs: Any) -> Any:
    """Instantiate the class registered under (kind, name)."""
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown kind {kind!r}")
    if name not in _REGISTRY[kind]:
        available = sorted(_REGISTRY[kind])
        raise KeyError(f"{kind!r}/{name!r} not registered. Available: {available}")
    return _REGISTRY[kind][name](**kwargs)


def list_registered(kind: str) -> list[str]:
    """List all registered names for a given kind. Useful for CLIs and debugging."""
    if kind not in _REGISTRY:
        raise KeyError(f"Unknown kind {kind!r}")
    return sorted(_REGISTRY[kind])


__all__ = ["build", "list_registered", "register"]
