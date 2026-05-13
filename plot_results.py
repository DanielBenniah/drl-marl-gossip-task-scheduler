"""
plot_results.py
===============
Generate the main result figures from experiment CSV outputs:
  Figure 1: Learning dynamics (ATCT vs episode for each DRL/MARL method).
  Figure 2: Performance comparison across four metrics (bar chart).
  Figure 3: Ablation study (DRL-MADRL vs NGC / NRS / NER / NPS).

Usage:
    python plot_results.py
    python plot_results.py --episodes results/raw_episodes.csv \
                           --seeds results/seed_means.csv
"""

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PAPER_METHODS = ["Random", "WeightedRR", "PriorityMinMin", "PCH",
                 "PPO", "MADDPG", "MAPPO", "DRL-MADRL"]
ABLATION_METHODS = ["DRL-MADRL", "DRL-MADRL-NGC", "DRL-MADRL-NRS",
                    "DRL-MADRL-NER", "DRL-MADRL-NPS", "PCH"]


def _read_episodes(path: str) -> Dict[str, Dict[int, List[float]]]:
    """Returns {method: {episode: [atct values across seeds]}}."""
    data: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            data[row["method"]][int(row["episode"])].append(float(row["atct"]))
    return {m: dict(v) for m, v in data.items()}


def _read_seed_means(path: str) -> Dict[str, Dict[str, List[float]]]:
    metrics = ["atct", "energy_kwh", "sla_pct", "tasks_done"]
    data: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    with open(path, "r") as f:
        for row in csv.DictReader(f):
            for k in metrics:
                data[row["method"]][k].append(float(row[k]))
    return {m: dict(v) for m, v in data.items()}


def figure1_learning(episodes_csv: str, out_path: str,
                     methods=None):
    if methods is None:
        methods = ["PPO", "MADDPG", "MAPPO", "DRL-MADRL"]
    data = _read_episodes(episodes_csv)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for m in methods:
        if m not in data:
            continue
        eps = sorted(data[m].keys())
        means = [np.mean(data[m][e]) for e in eps]
        stds  = [np.std(data[m][e], ddof=1) if len(data[m][e]) > 1 else 0.0
                 for e in eps]
        ax.plot(eps, means, label=m, linewidth=2)
        ax.fill_between(eps,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.15)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ATCT (s)")
    ax.set_title("(a) Learning dynamics: ATCT vs. episode")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    # Right panel: improvement of DRL-MADRL over Random by episode
    ax = axes[1]
    if "DRL-MADRL" in data and "Random" in data:
        eps = sorted(set(data["DRL-MADRL"].keys()) & set(data["Random"].keys()))
        impr = []
        for e in eps:
            d = np.mean(data["DRL-MADRL"][e])
            r = np.mean(data["Random"][e])
            impr.append(100.0 * (r - d) / max(r, 1e-9))
        ax.plot(eps, impr, color="tab:green", linewidth=2)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("ATCT improvement vs. Random (%)")
        ax.set_title("(b) DRL-MADRL improvement over Random")
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "Need 'DRL-MADRL' and 'Random' rows in episodes CSV",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def figure2_bars(seed_csv: str, out_path: str, methods=None):
    if methods is None:
        methods = PAPER_METHODS
    data = _read_seed_means(seed_csv)
    methods = [m for m in methods if m in data]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    panels = [
        ("atct",       "(a) Average task completion time", "ATCT (s)"),
        ("energy_kwh", "(b) Energy consumption",           "Energy (kWh)"),
        ("sla_pct",    "(c) SLA compliance",               "SLA (%)"),
        ("tasks_done", "(d) Tasks completed",              "Tasks"),
    ]
    for ax, (key, title, ylabel) in zip(axes, panels):
        means = [np.mean(data[m][key]) for m in methods]
        stds  = [np.std(data[m][key], ddof=1) if len(data[m][key]) > 1 else 0.0
                 for m in methods]
        colors = ["#888"] * len(methods)
        if "DRL-MADRL" in methods:
            colors[methods.index("DRL-MADRL")] = "tab:green"
        ax.bar(range(len(methods)), means, yerr=stds, capsize=4, color=colors)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def figure3_ablation(seed_csv: str, out_path: str):
    data = _read_seed_means(seed_csv)
    methods = [m for m in ABLATION_METHODS if m in data]
    if "DRL-MADRL" not in methods:
        print(f"Skipping {out_path}: no DRL-MADRL row in {seed_csv}.")
        return

    atct_means = [np.mean(data[m]["atct"]) for m in methods]
    sla_means  = [np.mean(data[m]["sla_pct"]) for m in methods]
    atct_stds  = [np.std(data[m]["atct"], ddof=1) if len(data[m]["atct"]) > 1 else 0.0
                  for m in methods]
    sla_stds   = [np.std(data[m]["sla_pct"], ddof=1) if len(data[m]["sla_pct"]) > 1 else 0.0
                  for m in methods]

    x = np.arange(len(methods))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax2 = ax.twinx()
    ax.bar(x - w/2, atct_means, w, yerr=atct_stds, capsize=4,
           label="ATCT (s)", color="tab:blue")
    ax2.bar(x + w/2, sla_means, w, yerr=sla_stds, capsize=4,
            label="SLA (%)", color="tab:orange")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right")
    ax.set_ylabel("ATCT (s)", color="tab:blue")
    ax2.set_ylabel("SLA (%)", color="tab:orange")
    ax.set_title("Ablation study (Section V-E)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--episodes", default="results/raw_episodes.csv")
    parser.add_argument("--seeds",    default="results/seed_means.csv")
    parser.add_argument("--out-dir",  default="results/figures")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if os.path.exists(args.episodes):
        figure1_learning(args.episodes,
                         os.path.join(args.out_dir, "figure1_learning.png"))
    else:
        print(f"Skipping Figure 1: {args.episodes} not found.")

    if os.path.exists(args.seeds):
        figure2_bars(args.seeds,
                     os.path.join(args.out_dir, "figure2_bars.png"))
        figure3_ablation(args.seeds,
                         os.path.join(args.out_dir, "figure3_ablation.png"))
    else:
        print(f"Skipping Figures 2 / 3: {args.seeds} not found.")
