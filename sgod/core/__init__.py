"""sgod.core — shared abstractions and types.

Three pluggable layers (see docs/dt_sgod_implementation_plan.md §2):
- Backbone: wraps a VLM, exposes hidden states + logits per step
- Oracle:   image → structured belief (SceneGraph)
- Policy:   (Backbone state, Oracle evidence) → logit adjustment Δ

All three are ABCs in `interfaces`; concrete implementations live under
`sgod.backbones`, `sgod.oracles`, `sgod.policies`.
"""
from sgod.core.interfaces import Backbone, Oracle, Policy
from sgod.core.registry import build, list_registered, register
from sgod.core.types import GenerationState, OracleEvidence, SceneGraph

__all__ = [
    "Backbone",
    "Oracle",
    "Policy",
    "GenerationState",
    "OracleEvidence",
    "SceneGraph",
    "build",
    "list_registered",
    "register",
]
