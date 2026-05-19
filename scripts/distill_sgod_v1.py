"""Stage-0 distillation entry point: SGOD v1 teacher → DT-SGOD student.

Two-phase script:

    phase 1 (collect):  Run SGOD v1 through the orchestrator on a calibration
                        dataset and dump per-step traces to disk.

    phase 2 (train):    Load the traces, instantiate a fresh DT-SGOD policy,
                        warm-start γ via `prepare_for_stage0`, and run
                        `distill_step` for `epochs` passes over the trace set.

Usage (one phase at a time, so partial runs are restartable):

    # Phase 1
    python scripts/distill_sgod_v1.py collect \\
        --teacher-config configs/sgod_v1_llava15.yaml \\
        --dataset jsonl:data/stage0/coco_calib.jsonl \\
        --out-traces outputs/stage0/traces.pt

    # Phase 2
    python scripts/distill_sgod_v1.py train \\
        --student-config configs/dt_sgod_llava15.yaml \\
        --traces outputs/stage0/traces.pt \\
        --out-ckpt outputs/stage0/dtsgod_stage0.pt \\
        --epochs 3 --lr 1e-4

Dataset format (jsonl): one example per line, each like
    {"image": "/abs/path/to/img.jpg", "question": "Is there a dog?"}

This is a skeleton: it wires up the pipeline correctly but the dataset/IO
side is deliberately minimal. Stage 1 will extend with batching, validation
splits, and DPO traces.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from PIL import Image

from sgod.policies.dt_sgod import DTSGODPolicy
from sgod.runtime import (
    HallucinationDecoder,
    TraceCollector,
    build_from_config,
    load_config,
)
from sgod.runtime.trace_collector import StepTrace
from sgod.training import distill_step, prepare_for_stage0
from sgod.training.distill_stage0 import stage0_trainable_params

log = logging.getLogger("distill_sgod_v1")


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="phase", required=True)

    pc = sub.add_parser("collect", help="Run teacher to dump traces.")
    pc.add_argument("--teacher-config", type=Path, required=True)
    pc.add_argument("--dataset", type=str, required=True,
                    help="Dataset spec: jsonl:<path>")
    pc.add_argument("--out-traces", type=Path, required=True)
    pc.add_argument("--limit", type=int, default=None,
                    help="Max examples to process (debug).")

    pt = sub.add_parser("train", help="Run distillation on collected traces.")
    pt.add_argument("--student-config", type=Path, required=True)
    pt.add_argument("--traces", type=Path, required=True)
    pt.add_argument("--out-ckpt", type=Path, required=True)
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--lr", type=float, default=1e-4)
    pt.add_argument("--w-delta", type=float, default=1.0)
    pt.add_argument("--w-atg", type=float, default=0.1)
    pt.add_argument("--gate-init", type=float, default=0.1)

    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args(argv)


# ── Phase 1: collect ─────────────────────────────────────────────────────────

def _iter_dataset(spec: str):
    """Yield (image_path, question) pairs from a dataset spec."""
    kind, path = spec.split(":", 1)
    if kind != "jsonl":
        raise ValueError(f"Unsupported dataset kind: {kind!r}; only 'jsonl' supported")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            yield rec["image"], rec["question"]


def _phase_collect(args: argparse.Namespace) -> int:
    log.info("Building teacher from %s", args.teacher_config)
    teacher_cfg = load_config(args.teacher_config)
    teacher = build_from_config(teacher_cfg, lazy=False)

    collector = TraceCollector()
    teacher.step_hook = collector

    n = 0
    args.out_traces.parent.mkdir(parents=True, exist_ok=True)
    for image_path, question in _iter_dataset(args.dataset):
        if args.limit is not None and n >= args.limit:
            break
        img = Image.open(image_path)
        prompt = f"USER: <image>\n{question} ASSISTANT:"
        teacher.generate(image=img, prompt=prompt, question=question)
        n += 1
        log.info("[%d] %s :: %d cumulative traces", n, Path(image_path).name, len(collector.traces))

    log.info("Saving %d traces → %s", len(collector.traces), args.out_traces)
    collector.save(args.out_traces)
    return 0


# ── Phase 2: train ───────────────────────────────────────────────────────────

def _phase_train(args: argparse.Namespace) -> int:
    log.info("Loading traces from %s", args.traces)
    traces: list[StepTrace] = TraceCollector.load(args.traces)
    log.info("Loaded %d traces", len(traces))

    log.info("Building student from %s", args.student_config)
    student_cfg = load_config(args.student_config)
    # Lazy backbone load is fine — the student doesn't drive the backbone during
    # Stage-0 (it learns from cached traces). But we DO need backbone.hidden_dim
    # and .vocab_size, which is what build_from_config reads — lazy LLaVA exposes
    # those as static constants.
    decoder: HallucinationDecoder = build_from_config(student_cfg, lazy=True)
    policy = decoder.policy
    assert isinstance(policy, DTSGODPolicy), "Stage-0 expects a DTSGODPolicy student"

    prepare_for_stage0(policy, gate_init=args.gate_init)

    opt = torch.optim.AdamW(stage0_trainable_params(policy), lr=args.lr)
    log.info("Starting Stage-0 distillation: epochs=%d lr=%g w_delta=%g w_atg=%g",
             args.epochs, args.lr, args.w_delta, args.w_atg)

    for epoch in range(args.epochs):
        total_loss, total_delta, total_atg, n_steps = 0.0, 0.0, 0.0, 0
        for t in traces:
            info = distill_step(policy, t, opt, w_delta=args.w_delta, w_atg=args.w_atg)
            total_loss  += info["loss"]
            total_delta += info["loss_delta"]
            total_atg   += info["loss_atg"]
            n_steps     += 1
        log.info(
            "epoch %d  loss=%.4g  L_delta=%.4g  L_atg=%.4g  gate=%.3f",
            epoch, total_loss / n_steps, total_delta / n_steps, total_atg / n_steps,
            policy.speaker_adapter.gate.item(),
        )

    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "speaker_adapter": policy.speaker_adapter.state_dict(),
        "anchor_gate": policy.anchor_gate.state_dict(),
        "gate_init": args.gate_init,
        "epochs": args.epochs,
    }, args.out_ckpt)
    log.info("Saved student adapter checkpoint → %s", args.out_ckpt)
    return 0


# ── Main ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    if args.phase == "collect":
        return _phase_collect(args)
    if args.phase == "train":
        return _phase_train(args)
    raise SystemExit(f"Unknown phase: {args.phase!r}")


if __name__ == "__main__":
    sys.exit(main())
