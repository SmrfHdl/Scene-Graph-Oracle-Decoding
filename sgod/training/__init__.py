"""sgod.training — adapter training utilities.

Stage 0:  Distill SGOD v1's training-free Δ into the DT-SGOD SpeakerAdapter.
          Unlocks Gate G2 (γ off 0). See `distill_stage0`.

Future stages live alongside as separate modules so the imports of each
stage are explicit and self-contained.
"""
from sgod.training.distill_stage0 import distill_step, prepare_for_stage0

__all__ = ["distill_step", "prepare_for_stage0"]
