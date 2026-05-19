"""Sanity-check a Stage-0 traces.pt file before kicking off distillation.

What we want to confirm:

  1. Volume: ≥ ~10k steps total (500 examples × ~20-50 tokens each).
  2. Anchor signal: some non-trivial fraction of steps have a rule anchor
     (noun/relation/attr), not 100% neutral.
  3. Δ-teacher non-degenerate: not all zero (would mean oracle never
     intervened) and not blowing up (would mean instability).
  4. Scene graph evidence: most examples produced non-empty SceneGraphs —
     otherwise the oracle is silent and Stage-0 learns nothing useful.

If any of these fail, training will produce a no-op student. Re-run collect
with adjusted oracle thresholds or a different dataset.

Usage:
    python scripts/inspect_stage0_traces.py outputs/stage0/traces.pt
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from sgod.runtime.trace_collector import TraceCollector


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", type=Path, help="Path to traces.pt")
    ap.add_argument("--quantile", type=float, default=0.99,
                    help="Quantile to report for |Δ| (default 0.99)")
    args = ap.parse_args()

    print(f"Loading {args.traces} ...")
    traces = TraceCollector.load(args.traces)
    n = len(traces)
    if n == 0:
        print("ERROR: no traces in file")
        return 1

    # 1. Volume
    examples = len({id(t.evidence) for t in traces})
    print("\n=== Volume ===")
    print(f"  total steps:    {n}")
    print(f"  approx examples (unique evidence objs): {examples}")
    print(f"  steps/example:  {n/examples:.1f}")

    # 2. Anchor type distribution
    anchor_counts = Counter(t.rule_anchor_type or "neutral" for t in traces)
    print("\n=== Anchor types (rule-based) ===")
    for kind, c in anchor_counts.most_common():
        pct = 100.0 * c / n
        print(f"  {kind:10s}: {c:6d}  ({pct:5.1f}%)")
    fire_rate = 1 - anchor_counts.get("neutral", 0) / n
    print(f"  anchor fire rate (non-neutral): {fire_rate:.3f}")

    # 3. Δ-teacher distribution
    abs_max_per_step = torch.tensor([t.delta_teacher.abs().max().item() for t in traces])
    nonzero_per_step = torch.tensor([(t.delta_teacher != 0).any().item() for t in traces], dtype=torch.float)
    print("\n=== Δ-teacher (per-step max |Δ|) ===")
    print(f"  mean: {abs_max_per_step.mean().item():.4f}")
    print(f"  median: {abs_max_per_step.median().item():.4f}")
    print(f"  q{int(args.quantile*100)}: {torch.quantile(abs_max_per_step, args.quantile).item():.4f}")
    print(f"  max: {abs_max_per_step.max().item():.4f}")
    print(f"  fraction of steps with any non-zero Δ: {nonzero_per_step.mean().item():.3f}")

    # On anchor steps only (where δ is supposed to fire).
    anchor_mask = torch.tensor([t.rule_anchor_type not in (None, "neutral") for t in traces])
    if anchor_mask.any():
        anchor_dmax = abs_max_per_step[anchor_mask]
        print(f"\n  On anchor steps only ({int(anchor_mask.sum())} steps):")
        print(f"    mean max|Δ|: {anchor_dmax.mean().item():.4f}")
        print(f"    fraction non-zero: {nonzero_per_step[anchor_mask].mean().item():.3f}")

    # 4. Scene graph richness
    unique_evidence = {id(t.evidence): t.evidence for t in traces}
    n_objs = [len(e.scene_graph.objects) for e in unique_evidence.values()]
    n_rels = [len(e.scene_graph.relations) for e in unique_evidence.values()]
    empty = sum(1 for o in n_objs if o == 0)
    n_objs_t = torch.tensor(n_objs, dtype=torch.float)
    n_rels_t = torch.tensor(n_rels, dtype=torch.float)
    print(f"\n=== Scene-graph richness (over {len(n_objs)} examples) ===")
    print(f"  objects   mean={n_objs_t.mean().item():.1f}  median={int(n_objs_t.median().item())}  max={int(n_objs_t.max().item())}")
    print(f"  relations mean={n_rels_t.mean().item():.1f}  median={int(n_rels_t.median().item())}  max={int(n_rels_t.max().item())}")
    print(f"  examples with EMPTY scene graph: {empty}/{len(n_objs)} ({100*empty/len(n_objs):.1f}%)")

    # 5. Verdict
    #
    # We treat the *effective* anchor rate as the share of steps where the
    # teacher actually injected a non-zero Δ — this is the real Stage-0
    # supervision signal. `rule_anchor_type` is a separate (currently
    # unwired) ATG warmup label; absence does NOT mean the oracle is dead.
    print("\n=== Verdict ===")
    issues = []
    effective_rate = nonzero_per_step.mean().item()
    if n < 5000:
        issues.append(f"low volume ({n} steps) — recommend ≥ 10k")
    if effective_rate < 0.05:
        issues.append(f"Δ non-zero rate too low ({effective_rate:.3f}) — oracle barely fires")
    if empty / max(len(n_objs), 1) > 0.3:
        issues.append(f"too many empty scene graphs ({empty}/{len(n_objs)})")
    if abs_max_per_step.max().item() > 100:
        issues.append(f"Δ magnitude exploded (max {abs_max_per_step.max():.1f}) — instability")
    if fire_rate < 0.01 and effective_rate > 0.05:
        print("  NOTE: rule_anchor_type is unwired in the orchestrator — using Δ non-zero rate")
        print(f"        as the effective anchor signal ({effective_rate:.3f}).")

    if not issues:
        print("  OK — traces look healthy. Proceed to `distill_sgod_v1.py train`.")
        return 0
    print("  Issues:")
    for it in issues:
        print(f"    - {it}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
