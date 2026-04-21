"""
Analysis and visualization for pilot experiment results.

Reads the results.json produced by run_pilot.py and generates:
  - Overall comparison table (baseline vs GT SG vs predicted SG)
  - Per relation-detail-type breakdown
  - Confidence interval on the delta
  - Optional matplotlib plots (--plot)

Usage:
    python experiments/pilot/analyze_pilot.py \\
        --results outputs/pilot/run_01/results.json

    python experiments/pilot/analyze_pilot.py \\
        --results outputs/pilot/run_01/results.json --plot
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Stats helpers ─────────────────────────────────────────────────────────────

def accuracy(results: list[dict], key: str) -> float:
    vals = [r[key] for r in results if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else 0.0


def wilson_ci(n: int, k: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, centre - margin), min(1, centre + margin)


def paired_t_pvalue(diffs: list[float]) -> float:
    """One-sample t-test p-value for H0: mean(diffs) == 0."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    mean_d = sum(diffs) / n
    var_d  = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se     = math.sqrt(var_d / n)
    if se == 0:
        return 0.0 if mean_d != 0 else 1.0
    t_stat = mean_d / se
    # Approximate two-tailed p-value using normal approximation (good for n≥30)
    # For smaller n, this is a rough estimate.
    z = abs(t_stat)
    p = 2 * (1 - _norm_cdf(z))
    return p


def _norm_cdf(x: float) -> float:
    """Approximate standard normal CDF via Horner's method."""
    a = 0.2316419
    k = 1 / (1 + a * x)
    poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937
           + k * (-1.821255978 + k * 1.330274429))))
    return 1 - (1 / math.sqrt(2 * math.pi)) * math.exp(-x**2 / 2) * poly


# ── Table printing ────────────────────────────────────────────────────────────

def print_table(rows: list[tuple], headers: list[str]) -> None:
    widths = [max(len(str(r[i])) for r in ([headers] + rows)) for i in range(len(headers))]
    sep  = "  ".join("─" * w for w in widths)
    fmt  = "  ".join(f"{{:<{w}}}" for w in widths)
    print("  " + fmt.format(*headers))
    print("  " + sep)
    for row in rows:
        print("  " + fmt.format(*[str(x) for x in row]))


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze(results_path: Path, threshold_pp: float = 3.0, plot: bool = False) -> None:
    with open(results_path) as f:
        results = json.load(f)

    n = len(results)
    if n == 0:
        print("Empty results file.")
        return

    has_c = any(r.get("correct_c") is not None for r in results)

    # ── Overall metrics ───────────────────────────────────────────────────────
    acc_a = accuracy(results, "correct_a")
    acc_b = accuracy(results, "correct_b")
    delta = acc_b - acc_a

    # Confidence intervals
    k_a      = sum(r["correct_a"] for r in results)
    k_b      = sum(r["correct_b"] for r in results)
    lo_a, hi_a = wilson_ci(n, k_a)
    lo_b, hi_b = wilson_ci(n, k_b)

    # Paired test: per-sample diff (b_correct - a_correct)
    diffs  = [r["correct_b"] - r["correct_a"] for r in results]
    p_val  = paired_t_pvalue(diffs)

    threshold_met = delta * 100 >= threshold_pp

    print("\n" + "=" * 64)
    print("  PILOT EXPERIMENT RESULTS")
    print("=" * 64)
    print(f"  N = {n} questions\n")

    rows = [
        ("A  Baseline",
         f"{acc_a*100:.1f}%",
         f"[{lo_a*100:.1f}%, {hi_a*100:.1f}%]",
         "—"),
        ("B  GT scene graph",
         f"{acc_b*100:.1f}%",
         f"[{lo_b*100:.1f}%, {hi_b*100:.1f}%]",
         f"{delta*100:+.1f} pp"),
    ]
    if has_c:
        acc_c       = accuracy(results, "correct_c")
        k_c         = sum(r["correct_c"] for r in results if r.get("correct_c") is not None)
        n_c         = sum(1 for r in results if r.get("correct_c") is not None)
        lo_c, hi_c  = wilson_ci(n_c, k_c)
        rows.append((
            "C  RelTR scene graph",
            f"{acc_c*100:.1f}%",
            f"[{lo_c*100:.1f}%, {hi_c*100:.1f}%]",
            f"{(acc_c-acc_a)*100:+.1f} pp",
        ))

    print_table(rows, ["Variant", "Accuracy", "95% CI", "Δ vs A"])
    print()
    print(f"  Paired t-test p-value (B vs A): {p_val:.4f}"
          + (" ***" if p_val < 0.001 else " **" if p_val < 0.01
             else " *" if p_val < 0.05 else " (not significant)"))
    print()
    print(f"  Gate threshold:  +{threshold_pp:.0f} pp")
    print(f"  Gate PASSED:     {'YES ✓  → proceed to main eval' if threshold_met else 'NO ✗  → reassess direction'}")

    # ── Per-type breakdown ────────────────────────────────────────────────────
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        t = r.get("detailed_type") or "unknown"
        by_type[t].append(r)

    if len(by_type) > 1:
        print("\n" + "─" * 64)
        print("  Per detailed-type breakdown (top types)")
        print("─" * 64)
        type_rows = []
        for dtype, recs in sorted(by_type.items(), key=lambda x: -len(x[1])):
            a = accuracy(recs, "correct_a")
            b = accuracy(recs, "correct_b")
            type_rows.append((dtype, len(recs), f"{a*100:.1f}%", f"{b*100:.1f}%",
                               f"{(b-a)*100:+.1f} pp"))
        print_table(type_rows[:15], ["Type", "N", "Acc A", "Acc B", "Δ"])

    # ── Error analysis ────────────────────────────────────────────────────────
    flipped_right = [r for r in results if not r["correct_a"] and r["correct_b"]]
    flipped_wrong = [r for r in results if r["correct_a"] and not r["correct_b"]]

    print(f"\n  B fixed A's errors:   {len(flipped_right):3d} questions")
    print(f"  B introduced errors:  {len(flipped_wrong):3d} questions")

    if flipped_right:
        print("\n  Example B-corrected cases (first 3):")
        for r in flipped_right[:3]:
            print(f"    Q:  {r['question']}")
            print(f"    GT: {r['gt_answer']}  |  A: {r['pred_a']}  |  B: {r['pred_b']}")

    # ── Optional plots ────────────────────────────────────────────────────────
    if plot:
        _make_plots(results, results_path.parent, has_c)


def _make_plots(results: list[dict], out_dir: Path, has_c: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [WARN] matplotlib not installed — skipping plots. pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: overall bar chart
    labels = ["A: Baseline", "B: GT SG"]
    accs   = [accuracy(results, "correct_a"), accuracy(results, "correct_b")]
    colors = ["#4e79a7", "#f28e2b"]
    if has_c:
        labels.append("C: RelTR SG")
        accs.append(accuracy(results, "correct_c"))
        colors.append("#59a14f")
    ax = axes[0]
    bars = ax.bar(labels, [a * 100 for a in accs], color=colors, width=0.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Pilot: Overall Accuracy by Variant")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc*100:.1f}%", ha="center", va="bottom", fontsize=11)

    # Plot 2: per-type delta (B - A)
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r.get("detailed_type") or "unknown"].append(r)
    types   = sorted(by_type, key=lambda t: -len(by_type[t]))[:10]
    deltas  = [(accuracy(by_type[t], "correct_b") - accuracy(by_type[t], "correct_a")) * 100
               for t in types]
    ax2 = axes[1]
    bar_colors = ["#59a14f" if d >= 0 else "#e15759" for d in deltas]
    ax2.barh(types, deltas, color=bar_colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Δ Accuracy (B − A, pp)")
    ax2.set_title("Delta by Relation Type (GT SG vs Baseline)")

    plt.tight_layout()
    plot_path = out_dir / "pilot_results.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {plot_path}")
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze SGOD pilot experiment results")
    p.add_argument("--results", required=True,
                   help="Path to results.json from run_pilot.py")
    p.add_argument("--threshold", type=float, default=3.0,
                   help="Gate threshold in percentage points (default: 3.0)")
    p.add_argument("--plot", action="store_true",
                   help="Generate matplotlib plots")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    analyze(Path(args.results), threshold_pp=args.threshold, plot=args.plot)


if __name__ == "__main__":
    main()
