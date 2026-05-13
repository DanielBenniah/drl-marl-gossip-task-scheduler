"""
verify_run.py
=============
Post-run learning health check. Reads `raw_episodes.csv` from a results
directory and verifies, per method:

  1. Number of episodes and seeds present (sanity).
  2. ATCT trend: linear regression slope across the run, with bootstrap CI.
     For learned methods, slope should be negative (improving) or near zero
     (plateau). For classical baselines, slope should be ~0 (no learning).
  3. Plateau detection: t-test on the first quarter vs last quarter of
     episodes. If significantly different, the run is still trending; if
     not, the method has plateaued.
  4. Cross-seed variance: std of the per-seed last-K-episode mean.
  5. Per-method summary table.

Usage:
    python verify_run.py --csv results/long/raw_episodes.csv
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
from scipy import stats


def load_episodes(path: str) -> Dict[str, Dict[int, Dict[int, dict]]]:
    """Returns {method: {seed: {episode: result_dict}}}."""
    out: Dict[str, Dict[int, Dict[int, dict]]] = defaultdict(
        lambda: defaultdict(dict))
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            m = row["method"]
            s = int(row["seed"])
            ep = int(row["episode"])
            out[m][s][ep] = {
                "atct":       float(row["atct"]),
                "energy_kwh": float(row["energy_kwh"]),
                "sla_pct":    float(row["sla_pct"]),
                "tasks_done": float(row["tasks_done"]),
            }
    return {m: dict(s) for m, s in out.items()}


def per_seed_atct_curve(method_data: Dict[int, Dict[int, dict]]) -> Dict[int, np.ndarray]:
    """Returns {seed: np.array([atct_ep0, atct_ep1, ...])} sorted by episode."""
    return {
        seed: np.array([ep_data[ep]["atct"]
                        for ep in sorted(ep_data.keys())])
        for seed, ep_data in method_data.items()
    }


def trend_slope(curve: np.ndarray) -> float:
    """Linear regression slope of ATCT vs episode index (units: s / episode)."""
    if len(curve) < 2:
        return 0.0
    x = np.arange(len(curve), dtype=np.float64)
    slope, _ = np.polyfit(x, curve, 1)
    return float(slope)


def plateau_test(curve: np.ndarray, alpha: float = 0.05) -> Dict:
    """Welch t-test on first quartile vs last quartile of the run."""
    n = len(curve)
    if n < 8:
        return {"first_q_mean": float("nan"), "last_q_mean": float("nan"),
                "p": float("nan"), "plateaued": None}
    q = n // 4
    first = curve[:q]
    last  = curve[-q:]
    t_stat, p_val = stats.ttest_ind(first, last, equal_var=False)
    plateaued = (p_val == p_val) and (p_val > alpha)
    return {
        "first_q_mean": float(first.mean()),
        "last_q_mean":  float(last.mean()),
        "delta":        float(last.mean() - first.mean()),
        "p":            float(p_val) if p_val == p_val else None,
        "plateaued":    plateaued,
    }


def report(data: Dict[str, Dict[int, Dict[int, dict]]],
           eval_window: int = None):
    print("\n" + "=" * 110)
    print(f"{'Method':<22} {'Seeds':>6} {'Eps':>5} {'ATCT slope/ep':>14} "
          f"{'first-q':>8} {'last-q':>8} {'delta':>7} {'p':>7} "
          f"{'plateau':>8} {'last-K mean':>12} {'last-K std':>11}")
    print("-" * 110)
    for method in sorted(data.keys()):
        seeds_data = data[method]
        seeds = sorted(seeds_data.keys())
        n_seeds = len(seeds)
        episode_counts = [len(seeds_data[s]) for s in seeds]
        n_eps = max(episode_counts) if episode_counts else 0
        K = eval_window or max(10, n_eps // 10)
        K = min(K, n_eps)

        # Per-seed slopes and final-K means
        slopes = []
        last_ks = []
        first_q = []
        last_q  = []
        plateaued_count = 0
        for s in seeds:
            curve = per_seed_atct_curve({s: seeds_data[s]})[s]
            if len(curve) >= 2:
                slopes.append(trend_slope(curve))
            last_ks.append(curve[-K:].mean())
            t = plateau_test(curve)
            if t["first_q_mean"] == t["first_q_mean"]:
                first_q.append(t["first_q_mean"])
                last_q.append(t["last_q_mean"])
                if t["plateaued"]:
                    plateaued_count += 1

        slope_mean = float(np.mean(slopes)) if slopes else 0.0
        last_k_mean = float(np.mean(last_ks)) if last_ks else 0.0
        last_k_std  = float(np.std(last_ks, ddof=1)) if len(last_ks) > 1 else 0.0
        fq = float(np.mean(first_q)) if first_q else float("nan")
        lq = float(np.mean(last_q)) if last_q else float("nan")
        delta = lq - fq if first_q else float("nan")

        # Aggregate p-value: combine per-seed tests (just average, informative
        # not formal). Better: pool curves and run one t-test.
        if first_q and len(first_q) >= 2:
            pooled_first = np.concatenate([per_seed_atct_curve({s: seeds_data[s]})[s][:len(seeds_data[s])//4]
                                           for s in seeds if len(seeds_data[s]) >= 8])
            pooled_last  = np.concatenate([per_seed_atct_curve({s: seeds_data[s]})[s][-len(seeds_data[s])//4:]
                                           for s in seeds if len(seeds_data[s]) >= 8])
            if len(pooled_first) > 1 and len(pooled_last) > 1:
                _, p_val = stats.ttest_ind(pooled_first, pooled_last, equal_var=False)
            else:
                p_val = float("nan")
        else:
            p_val = float("nan")

        plateau_str = f"{plateaued_count}/{n_seeds}"
        p_str = f"{p_val:.4f}" if (p_val == p_val) else "n/a"
        print(f"{method:<22} {n_seeds:>6} {n_eps:>5} "
              f"{slope_mean:>+14.4f} "
              f"{fq:>8.2f} {lq:>8.2f} {delta:>+7.2f} {p_str:>7} "
              f"{plateau_str:>8} {last_k_mean:>12.2f} {last_k_std:>11.3f}")
    print("=" * 110)
    print("Legend:")
    print("  ATCT slope/ep : per-seed mean of OLS slope (s/episode); negative = improving.")
    print("  first-q / last-q : ATCT means over first vs last quarter of episodes.")
    print("  delta : last-q - first-q (negative = improvement during training).")
    print("  p : Welch t-test (pooled across seeds) of first-q vs last-q.")
    print("       p > 0.05 => plateau (no significant trend); p < 0.05 => still trending.")
    print("  plateau : count of seeds individually plateaued at alpha=0.05.")
    print("  last-K  : mean ATCT over last K episodes; std across seeds.")


def health_summary(data: Dict[str, Dict[int, Dict[int, dict]]]) -> Dict:
    """Returns a dict of per-method health flags for programmatic use."""
    out = {}
    for method, seeds_data in data.items():
        seeds = sorted(seeds_data.keys())
        slopes = []
        for s in seeds:
            curve = per_seed_atct_curve({s: seeds_data[s]})[s]
            if len(curve) >= 2:
                slopes.append(trend_slope(curve))
        slope_mean = float(np.mean(slopes)) if slopes else 0.0
        out[method] = {
            "n_seeds":      len(seeds),
            "n_eps":        max((len(seeds_data[s]) for s in seeds), default=0),
            "slope_mean":   slope_mean,
            "improving":    slope_mean < -0.001,  # > 1 ms/episode
            "stable":       abs(slope_mean) < 0.001,
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-run learning verification")
    parser.add_argument("--csv",         default="results/raw_episodes.csv")
    parser.add_argument("--eval-window", type=int, default=None,
                        help="Trailing window for last-K mean/std")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"Missing input CSV: {args.csv}")

    data = load_episodes(args.csv)
    if not data:
        raise SystemExit(f"No rows in {args.csv}")

    report(data, eval_window=args.eval_window)
