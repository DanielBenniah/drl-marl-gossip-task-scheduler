"""
run_sensitivity.py
==================
Hyperparameter sensitivity scan for the contention-adaptive blend (Section
IV-D.1). Tests whether DRL-MADRL-Adaptive's behavior is robust to the
choice of threshold tau and width w in:

    alpha = clip((c - tau) / w, 0, 1)

We test 6 combinations spanning conservative-to-aggressive blending,
plus the original (tau=0.05, w=0.15) for reference. Run at lambda=20
(high contention, where adaptive should engage) for 30 episodes x 3 seeds.

If results are within +/- 0.5 s ATCT and +/- 1 pp SLA across the scan,
the adaptive blend is robust to hyperparameter choice (defensible against
"did you tune?" reviewer concerns).

Usage:
    python run_sensitivity.py
"""

import os
import csv
import time
import numpy as np

from marl_scheduler import (
    InfrastructureGenerator, WorkloadGenerator, DRLMADRLScheduler,
)
from simulation import Simulator


SETTINGS = [
    ("conservative",  0.10, 0.30),
    ("reported",      0.05, 0.15),
    ("moderate-A",    0.05, 0.20),
    ("moderate-B",    0.10, 0.20),
    ("aggressive",    0.02, 0.10),
    ("very-aggressive", 0.02, 0.05),
]

SEEDS = [42, 43, 44]
NUM_TASKS = 1000
NUM_EPS = 30
LAMBDA = 20.0
EVAL_K = 10


def run_one(tau, w, seed):
    nodes = InfrastructureGenerator.create_nodes(num_nodes=100, seed=seed)
    wgen  = WorkloadGenerator(seed=seed)
    sched = DRLMADRLScheduler(nodes, seed=seed,
                              contention_adaptive=True,
                              adaptive_threshold=tau,
                              adaptive_width=w)
    sim   = Simulator(nodes, sched, time_step=5.0)

    last_k_atct = []
    last_k_sla  = []
    last_k_e    = []
    for ep in range(NUM_EPS):
        ep_seed = seed * 1_000_000 + ep
        tasks = wgen.generate_tasks(num_tasks=NUM_TASKS,
                                    arrival_rate=LAMBDA,
                                    episode_seed=ep_seed)
        result = sim.run_episode(tasks)
        if ep >= NUM_EPS - EVAL_K:
            last_k_atct.append(result["atct"])
            last_k_sla.append(result["sla_pct"])
            last_k_e.append(result["energy_kwh"])
    return (float(np.mean(last_k_atct)),
            float(np.mean(last_k_sla)),
            float(np.mean(last_k_e)))


def main():
    out_dir = "results/sensitivity"
    os.makedirs(out_dir, exist_ok=True)

    print(f"Hyperparameter sensitivity at lambda={LAMBDA}, "
          f"{NUM_EPS} ep x {len(SEEDS)} seeds, last-{EVAL_K} mean.")
    print(f"{'Setting':<18} {'tau':>6} {'w':>6} "
          f"{'ATCT':>14} {'SLA':>14} {'Energy':>14}")
    print("-" * 80)

    rows = []
    t0 = time.time()
    for label, tau, w in SETTINGS:
        atcts, slas, energies = [], [], []
        for seed in SEEDS:
            t1 = time.time()
            atct, sla, e = run_one(tau, w, seed)
            atcts.append(atct); slas.append(sla); energies.append(e)
            print(f"  ({label} seed={seed} done in {time.time()-t1:.0f}s)",
                  flush=True)
        atct_m = float(np.mean(atcts)); atct_s = float(np.std(atcts, ddof=1)) if len(atcts) > 1 else 0.0
        sla_m  = float(np.mean(slas));  sla_s  = float(np.std(slas, ddof=1))  if len(slas)  > 1 else 0.0
        e_m    = float(np.mean(energies)); e_s = float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0
        print(f"{label:<18} {tau:>6.3f} {w:>6.3f} "
              f"{atct_m:>7.2f}±{atct_s:.2f} "
              f"{sla_m:>7.2f}±{sla_s:.2f} "
              f"{e_m:>7.2f}±{e_s:.2f}", flush=True)
        rows.append({"setting": label, "tau": tau, "w": w,
                     "atct_mean": atct_m, "atct_std": atct_s,
                     "sla_mean":  sla_m,  "sla_std":  sla_s,
                     "energy_mean": e_m,  "energy_std": e_s})

    print(f"\nTotal time: {time.time()-t0:.0f}s")

    csv_path = os.path.join(out_dir, "sensitivity.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {csv_path}")

    # Verdict
    atct_range = max(r["atct_mean"] for r in rows) - min(r["atct_mean"] for r in rows)
    sla_range  = max(r["sla_mean"] for r in rows) - min(r["sla_mean"] for r in rows)
    print(f"\n=== Verdict ===")
    print(f"ATCT range across settings: {atct_range:.2f} s")
    print(f"SLA  range across settings: {sla_range:.2f} pp")
    if atct_range <= 0.5 and sla_range <= 1.0:
        print("VERDICT: Adaptive blend is ROBUST to hyperparameter choice.")
        print("Defensible against 'did you tune?' reviewer concerns.")
    else:
        print("VERDICT: Adaptive blend has noticeable hyperparameter sensitivity.")
        print("Suggest reporting the full sensitivity scan in V-E.")


if __name__ == "__main__":
    main()
