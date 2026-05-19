"""Stage-0 distillation: SGOD v1 teacher → DT-SGOD SpeakerAdapter student.

Pipeline (offline, after traces are collected):

    1. SGOD v1 generates with the orchestrator + TraceCollector attached.
       Result: list[StepTrace] capturing (h_t, lm_logits, δ_teacher, evidence)
       per step.
    2. `prepare_for_stage0(policy)` warms γ off zero so out_proj has a non-zero
       gradient on the first step (the double-zero init is a flat point — see
       the docstring for the breakdown).
    3. For each trace, `distill_step` runs the student's SpeakerAdapter + ATG
       and applies:
          L = w_delta · MSE(δ_student, δ_teacher)
            + w_atg   · BCE(fire_prob, indicator(δ_teacher non-zero))

       The GP runs once per trace to produce h_slow. For Stage 0 the GP is
       frozen (only SpeakerAdapter + ATG learn).

Gate G2 (proven by subsequent eval, not here):
    After Stage 0, γ > 0 and the student's Δ tracks the teacher closely on a
    held-out trace set. Stage 1 then opens DPO + LoRA training on top.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

from sgod.policies.dt_sgod import DTSGODPolicy
from sgod.runtime.trace_collector import StepTrace


# ── Stage-0 warm start ───────────────────────────────────────────────────────

@torch.no_grad()
def prepare_for_stage0(
    policy: DTSGODPolicy,
    gate_init: float = 0.1,
    out_proj_std: float = 0.0,
) -> None:
    """Break the double-zero init so distillation gradients can flow.

    Why this exists:
        At init the SpeakerAdapter has γ=0 AND out_proj=0. Then Δ ≡ 0 and:
            dΔ/d(out_proj.weight) ∝ tanh(γ) = 0
            dΔ/d(γ)               ∝ out_proj(attn_out) = 0
        Every parameter's gradient is zero → no learning. To bootstrap, we
        nudge γ off zero (out_proj stays at zero so Δ still starts near 0).

    With γ = 0.1, out_proj = 0:
        dΔ/d(out_proj.weight) ∝ tanh(0.1) ≈ 0.0997 ≠ 0  ✓ out_proj can learn
        dΔ/d(γ)               ∝ out_proj(attn_out) = 0   (waits for out_proj to move)

    After the first few updates out_proj is non-zero and γ also starts to learn.

    Set `out_proj_std > 0` to additionally perturb out_proj — useful only if
    you want symmetric early gradients. The default (zero out_proj + γ=0.1)
    is the cleaner unlock and keeps |Δ| small at training start.
    """
    policy.speaker_adapter.gate.data.fill_(float(gate_init))
    if out_proj_std > 0:
        policy.speaker_adapter.out_proj.weight.data.normal_(mean=0.0, std=out_proj_std)


# ── Single distillation step ─────────────────────────────────────────────────

def distill_step(
    policy: DTSGODPolicy,
    trace: StepTrace,
    optimizer: torch.optim.Optimizer,
    *,
    w_delta: float = 1.0,
    w_atg: float = 0.1,
    fire_gp: bool = True,
) -> dict:
    """Run one distillation step on a single trace. Returns loss components.

    Args:
        policy:    Student DTSGODPolicy. Must have been `prepare_for_stage0`-ed.
        trace:     One StepTrace from the teacher.
        optimizer: Wraps the params being trained — typically SpeakerAdapter
                   + AnchorTriggerGate. The GP is frozen for Stage 0.
        w_delta:   Weight on the MSE loss against teacher Δ.
        w_atg:     Weight on the ATG BCE loss against the teacher-fired
                   indicator. Set 0 to skip (e.g. ATG ablation).
        fire_gp:   If True, run the GP once on this trace to get a non-trivial
                   h_slow before SpeakerAdapter cross-attention. If False,
                   use the freshly-initialized h_slow (slot embeddings only).

    Returns:
        dict with keys 'loss', 'loss_delta', 'loss_atg', 'fire_prob_mean',
        'gate' — for logging.
    """
    # Traces are captured from the teacher backbone (typically fp16 on CUDA);
    # the student policy is fp32 on whatever device prepare_for_stage0 placed
    # it on. Cast trace tensors to match the policy — they're inputs with no
    # autograd graph, so casting is free.
    policy_dtype = next(policy.speaker_adapter.parameters()).dtype
    policy_device = next(policy.speaker_adapter.parameters()).device
    h = trace.hidden_states.to(device=policy_device, dtype=policy_dtype)
    lm_logits = trace.lm_logits.to(device=policy_device, dtype=policy_dtype)
    delta_teacher = trace.delta_teacher.to(device=policy_device, dtype=policy_dtype)
    evidence = trace.evidence
    device = h.device
    B = h.shape[0]

    # 1) Slow module: init h_slow, optionally fire GP once so the cross-attn
    #    sees content. For Stage 0 this is a per-trace single-fire — the
    #    inter-step recurrence is exercised at Stage 1 when full generation
    #    is rolled out.
    with torch.no_grad():
        h_slow = policy.grounding_planner.init_state(evidence, batch_size=B, device=device)
    if fire_gp:
        # GP frozen but we still let it forward; gradient won't flow into GP
        # params because they're typically excluded from the optimizer for
        # Stage 0 (see scripts/distill_sgod_v1.py for the param-grouping).
        h_slow = policy.grounding_planner(h_slow, h, evidence)

    # 2) Student Δ.
    delta_student = policy.speaker_adapter(h, h_slow)

    # 3) ATG with grad (the inference path is no_grad; we recompute here so
    #    BCE has somewhere to land).
    h_slow_pooled = h_slow.mean(dim=1)
    lm_entropy = _entropy(lm_logits)
    fire_prob = policy.anchor_gate(h, h_slow_pooled, lm_entropy)  # [B]

    # 4) Losses.
    loss_delta = F.mse_loss(delta_student, delta_teacher)

    teacher_fired = (delta_teacher.abs().sum(dim=-1) > 0).to(fire_prob.dtype)  # [B]
    if w_atg > 0.0:
        loss_atg = F.binary_cross_entropy(fire_prob, teacher_fired)
    else:
        loss_atg = torch.zeros((), device=device)

    loss = w_delta * loss_delta + w_atg * loss_atg

    # 5) Step.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
        "loss_delta": float(loss_delta.item()),
        "loss_atg": float(loss_atg.item()),
        "fire_prob_mean": float(fire_prob.mean().item()),
        "gate": float(policy.speaker_adapter.gate.item()),
    }


# ── Param-group helpers ──────────────────────────────────────────────────────

def stage0_trainable_params(policy: DTSGODPolicy) -> Iterable[torch.nn.Parameter]:
    """Yield the parameters trained in Stage 0: SpeakerAdapter + AnchorTriggerGate.

    Convenience for constructing the optimizer:
        opt = torch.optim.AdamW(stage0_trainable_params(policy), lr=1e-4)
    The GroundingPlanner is intentionally excluded — it is frozen at the
    Day-2 init for Stage 0 and starts updating in Stage 1.
    """
    yield from policy.speaker_adapter.parameters()
    yield from policy.anchor_gate.parameters()


# ── helpers ──────────────────────────────────────────────────────────────────

def _entropy(logits: Tensor) -> Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


__all__ = ["distill_step", "prepare_for_stage0", "stage0_trainable_params"]
