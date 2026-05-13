"""
analysis.py
===========
Statistical analysis of the simulation results.

Reads the per-seed and per-episode CSVs written by `simulation.py` and produces:
  1. A clean Table II equivalent (mean ± std across seeds for each method).
  2. Pairwise two-sample t-tests over independent seed-level means against
     DRL-MADRL, with Bonferroni correction at alpha = 0.05 (Section V-A).
  3. An ablation table comparing the full DRL-MADRL against NGC / NRS / NER /
     NPS variants, if those methods are present in the input file.

Usage:
    python analysis.py                              # uses results/seed_means.csv
    python analysis.py --csv results/seed_means.csv # explicit path
    python analysis.py --reference DRL-MADRL --alpha 0.05
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


METRICS = ["atct", "energy_kwh", "sla_pct", "tasks_done"]
METRIC_LABELS = {
    "atct":       "ATCT (s)",
    "energy_kwh": "Energy (kWh)",
    "sla_pct":    "SLA (%)",
    "tasks_done": "Tasks Done",
}
# For ATCT and Energy, lower = better. For SLA and Tasks Done, higher = better.
HIGHER_IS_BETTER = {"sla_pct": True, "tasks_done": True,
                    "atct": False, "energy_kwh": False}


def load_seed_means(path: str) -> Dict[str, Dict[str, List[float]]]:
    """Returns {method: {metric: [values per seed]}} from seed_means.csv."""
    data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = row["method"]
            for k in METRICS:
                data[m][k].append(float(row[k]))
    return {m: dict(v) for m, v in data.items()}


def descriptive_table(data: Dict[str, Dict[str, List[float]]]) -> List[Dict]:
    rows = []
    for method, metrics in data.items():
        row = {"method": method, "n_seeds": len(next(iter(metrics.values())))}
        for k in METRICS:
            arr = np.array(metrics[k], dtype=np.float64)
            row[k + "_mean"] = float(arr.mean())
            row[k + "_std"]  = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        rows.append(row)
    return rows


def print_descriptive(rows: List[Dict]):
    print("\n" + "=" * 92)
    print(f"{'Method':<22} {'ATCT (s)':>14} {'Energy (kWh)':>16} "
          f"{'SLA (%)':>12} {'Tasks':>10} {'Seeds':>8}")
    print("-" * 92)
    for r in rows:
        print(f"{r['method']:<22} "
              f"{r['atct_mean']:>8.2f} ± {r['atct_std']:<4.2f} "
              f"{r['energy_kwh_mean']:>10.1f} ± {r['energy_kwh_std']:<4.1f} "
              f"{r['sla_pct_mean']:>7.2f} ± {r['sla_pct_std']:<4.2f} "
              f"{r['tasks_done_mean']:>9.0f} "
              f"{r['n_seeds']:>8d}")
    print("=" * 92)


def pairwise_ttests(data: Dict[str, Dict[str, List[float]]],
                    reference: str = "DRL-MADRL",
                    alpha: float = 0.05) -> List[Dict]:
    """Two-sample Welch t-test of `reference` vs each other method, per metric.

    Bonferroni: corrected alpha = alpha / num_comparisons, where
    num_comparisons = (#methods - 1) * (#metrics).
    """
    if reference not in data:
        raise KeyError(f"Reference method '{reference}' not found in data.")
    ref_metrics = data[reference]
    others = [m for m in data if m != reference]
    n_comparisons = max(1, len(others) * len(METRICS))
    corrected = alpha / n_comparisons

    rows = []
    for other in others:
        for k in METRICS:
            ref_vals = np.asarray(ref_metrics[k], dtype=np.float64)
            cmp_vals = np.asarray(data[other][k], dtype=np.float64)
            if len(ref_vals) < 2 or len(cmp_vals) < 2:
                t_stat, p_val = float("nan"), float("nan")
            else:
                t_stat, p_val = stats.ttest_ind(ref_vals, cmp_vals,
                                                equal_var=False)
            higher_better = HIGHER_IS_BETTER[k]
            ref_better = (ref_vals.mean() > cmp_vals.mean()) if higher_better \
                          else (ref_vals.mean() < cmp_vals.mean())
            rows.append({
                "metric":          k,
                "reference":       reference,
                "compared_to":     other,
                "ref_mean":        float(ref_vals.mean()),
                "cmp_mean":        float(cmp_vals.mean()),
                "ref_better":      bool(ref_better),
                "t":               float(t_stat) if t_stat == t_stat else None,
                "p":               float(p_val)  if p_val  == p_val  else None,
                "alpha_uncorr":    alpha,
                "alpha_bonf":      corrected,
                "sig_uncorrected": (p_val == p_val) and (p_val < alpha),
                "sig_bonferroni":  (p_val == p_val) and (p_val < corrected),
            })
    return rows


def print_significance(rows: List[Dict], alpha: float):
    if not rows:
        print("\n(No comparison rows.)")
        return
    n = len(rows)
    bonf = rows[0]["alpha_bonf"]
    print("\n" + "=" * 100)
    print(f"Pairwise Welch t-tests vs {rows[0]['reference']}  "
          f"(alpha = {alpha}, Bonferroni-corrected = {bonf:.4f}, "
          f"comparisons = {n})")
    print("-" * 100)
    print(f"{'Metric':<14} {'vs':<22} {'ref mean':>10} {'cmp mean':>10} "
          f"{'t':>8} {'p':>10} {'sig':>10} {'better':>8}")
    print("-" * 100)
    for r in rows:
        sig = "**" if r["sig_bonferroni"] else ("*" if r["sig_uncorrected"] else "ns")
        better = "ref" if r["ref_better"] else "cmp"
        p_str = f"{r['p']:.4f}" if r["p"] is not None else "  n/a"
        t_str = f"{r['t']:.2f}"  if r["t"] is not None else " n/a"
        print(f"{METRIC_LABELS[r['metric']]:<14} "
              f"{r['compared_to']:<22} "
              f"{r['ref_mean']:>10.2f} "
              f"{r['cmp_mean']:>10.2f} "
              f"{t_str:>8} "
              f"{p_str:>10} "
              f"{sig:>10} "
              f"{better:>8}")
    print("=" * 100)
    print("Legend: **=significant after Bonferroni, *=p<alpha uncorrected, ns=not significant.")


def write_significance_csv(rows: List[Dict], path: str):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    keys = ["metric", "reference", "compared_to", "ref_mean", "cmp_mean",
            "ref_better", "t", "p", "alpha_uncorr", "alpha_bonf",
            "sig_uncorrected", "sig_bonferroni"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in keys})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRL-MADRL statistical analysis")
    parser.add_argument("--csv",       default="results/seed_means.csv",
                        help="Per-seed CSV from simulation.py")
    parser.add_argument("--reference", default="DRL-MADRL",
                        help="Reference method for pairwise tests")
    parser.add_argument("--alpha",     type=float, default=0.05)
    parser.add_argument("--out",       default="results/significance.csv")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Missing input CSV: {args.csv}\n"
                         f"Run: python simulation.py first.")

    data = load_seed_means(args.csv)
    rows = descriptive_table(data)
    print_descriptive(rows)

    if args.reference in data:
        sig_rows = pairwise_ttests(data, reference=args.reference,
                                   alpha=args.alpha)
        print_significance(sig_rows, args.alpha)
        write_significance_csv(sig_rows, args.out)
        print(f"\nSignificance table written to {args.out}")
    else:
        print(f"\nReference method '{args.reference}' not present; "
              f"skipping significance tests.")
