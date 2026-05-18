"""sgod.policies — decoding-time intervention policies.

A Policy implements the `sgod.core.Policy` interface: given a `GenerationState`
(backbone hidden, LM logits, oracle evidence, prior policy state), return a
logit adjustment Δ to be added to the LM logits before sampling.

Implemented:
- DTSGODPolicy   — dual-timescale slot-recurrent (this proposal, paper target).

Planned:
- SGODv1Policy   — training-free linear mixing (the predecessor, used as ablation
                   baseline + Stage-0 distillation teacher).
"""
from sgod.policies.dt_sgod.policy import DTSGODPolicy

__all__ = ["DTSGODPolicy"]
