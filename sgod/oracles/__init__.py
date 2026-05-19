"""sgod.oracles — concrete Oracle implementations.

Each Oracle wraps a scene-graph or open-vocab detection model and returns an
`OracleEvidence` packaging a `SceneGraph` plus optional extras. Oracles must
be *independent* of the VLM Backbone — see the design_independence_principle
memory.

Implemented:
- RelTROracle           — closed-vocab SGG (RelTR, 151 entities + 51 relations).
                          Supports hybrid mode where Grounding DINO supplies
                          open-vocab objects and RelTR keeps relations.
- GroundingDinoOracle   — open-vocab object detection only (no relations).

Note on `sgod.oracle` vs `sgod.oracles`: the singular `sgod.oracle` package
holds the legacy `VisualOracle` + CLIP scorers (per-token soft scoring); this
plural `sgod.oracles` package holds the framework-level `Oracle` ABC
implementations. The two coexist because SGODv1Policy still consumes
VisualOracle internally.
"""
from sgod.oracles.grounding_dino_oracle import GroundingDinoOracle
from sgod.oracles.reltr_oracle import RelTROracle

__all__ = ["GroundingDinoOracle", "RelTROracle"]
