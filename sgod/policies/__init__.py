"""sgod.policies — decoding-time intervention policies.

A Policy implements the `sgod.core.Policy` interface: given a `GenerationState`
(backbone hidden, LM logits, oracle evidence, prior policy state), return a
logit adjustment Δ to be added to the LM logits before sampling.

Implemented:
- DTSGODPolicy   — dual-timescale slot-recurrent (this proposal, paper target).
- SGODv1Policy   — training-free linear mixing (the predecessor, used as ablation
                   baseline + Stage-0 distillation teacher).
"""
from sgod.policies.dt_sgod.policy import DTSGODPolicy
from sgod.policies.sgod_v1 import SGODv1Policy

__all__ = ["DTSGODPolicy", "SGODv1Policy"]
