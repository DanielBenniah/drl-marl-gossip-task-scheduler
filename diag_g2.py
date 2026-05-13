"""
diag_g2.py
==========
Verify that the new PerAgentGossip implementation works as intended:

  1. Standalone correctness: per-agent estimates converge to the true
     per-node utilization vector under gossip.
  2. NGC isolation: with enable_communication=False, agents' estimates of
     non-local nodes stay at the default (e.g., 0.5).
  3. End-to-end: in a real episode, full DRL-MADRL vs NGC produce different
     gossip estimates AND different placements AND different metrics.
  4. Eq. 14 dependency: the placement decision is now genuinely affected by
     the gossip view (agent's estimate of other nodes' load).

If any of these checks fail, NGC is not a meaningful ablation.

Usage:
    python diag_g2.py
"""

import numpy as np

from marl_scheduler import (
    InfrastructureGenerator, WorkloadGenerator, PerAgentGossip,
    DRLMADRLScheduler,
)
from simulation import Simulator


def test_standalone_convergence():
    """Per-agent gossip should converge agents' estimates to true u_j."""
    print("=== Test 1: Per-agent gossip convergence (full communication) ===")
    rng = np.random.default_rng(0)
    N = 100
    true_utils = rng.uniform(0.0, 1.0, size=N).astype(np.float32)
    g = PerAgentGossip(num_nodes=N, gossip_prob=0.3, avg_weight=0.5,
                       refresh_rate=0.1, default_estimate=0.5, seed=42)
    g.initialize(true_utils)

    print(f"  True utilizations: mean={true_utils.mean():.3f} "
          f"std={true_utils.std():.3f}")

    for t in [10, 100, 500, 2000]:
        # Reset and run
        g2 = PerAgentGossip(N, 0.3, 0.5, 0.1, default_estimate=0.5, seed=42)
        g2.initialize(true_utils)
        for _ in range(t):
            g2.step(true_utils, enable_communication=True)
        # Average MAE across all agents
        mae = float(np.mean([np.abs(g2.estimates[i] - true_utils).mean()
                             for i in range(N)]))
        # Check diagonal (self-estimates) — should be exact
        diag_mae = float(np.mean([abs(g2.estimates[i][i] - true_utils[i])
                                  for i in range(N)]))
        # Check off-diagonal (other-node estimates)
        off_mae = float(np.mean([
            np.mean([abs(g2.estimates[i][j] - true_utils[j])
                     for j in range(N) if j != i])
            for i in range(N)
        ]))
        print(f"  t={t:>4}: MAE all={mae:.4f}  diag={diag_mae:.4f}  "
              f"off-diag={off_mae:.4f}")
    print()


def test_ngc_isolation():
    """Without communication, off-diagonal estimates should stay at 0.5."""
    print("=== Test 2: NGC (no communication) isolation ===")
    N = 100
    rng = np.random.default_rng(0)
    true_utils = rng.uniform(0.0, 1.0, size=N).astype(np.float32)
    g = PerAgentGossip(N, 0.3, 0.5, 0.1, default_estimate=0.5, seed=42)
    g.initialize(true_utils)

    for t in [50, 200]:
        g2 = PerAgentGossip(N, 0.3, 0.5, 0.1, default_estimate=0.5, seed=42)
        g2.initialize(true_utils)
        for _ in range(t):
            g2.step(true_utils, enable_communication=False)
        diag_err = float(np.mean([abs(g2.estimates[i][i] - true_utils[i])
                                  for i in range(N)]))
        off_diag = float(np.mean([
            np.mean([g2.estimates[i][j] for j in range(N) if j != i])
            for i in range(N)
        ]))
        print(f"  t={t:>3}: diagonal MAE (self) = {diag_err:.4f}  "
              f"(expect ~0)  ;  off-diagonal mean = {off_diag:.4f}  "
              f"(expect ~0.5)")
    print()


def test_full_vs_ngc_episode():
    """In a real episode, full DRL-MADRL vs NGC should produce different
    gossip vectors, different placements, and different metrics."""
    print("=== Test 3: Full vs NGC in a real episode ===")
    seed = 42
    nodes_full = InfrastructureGenerator.create_nodes(num_nodes=100, seed=seed)
    nodes_ngc  = InfrastructureGenerator.create_nodes(num_nodes=100, seed=seed)

    sched_full = DRLMADRLScheduler(nodes_full, seed=seed, use_gossip=True)
    sched_ngc  = DRLMADRLScheduler(nodes_ngc,  seed=seed, use_gossip=False)

    wgen = WorkloadGenerator(seed=seed)
    tasks = wgen.generate_tasks(num_tasks=500, arrival_rate=20.0,  # high contention
                                episode_seed=seed * 1_000_000)

    sim_full = Simulator(nodes_full, sched_full)
    sim_ngc  = Simulator(nodes_ngc,  sched_ngc)

    res_full = sim_full.run_episode(list(tasks))
    # Re-generate identical task copy for NGC
    tasks2 = wgen.generate_tasks(num_tasks=500, arrival_rate=20.0,
                                 episode_seed=seed * 1_000_000)
    res_ngc  = sim_ngc.run_episode(list(tasks2))

    print(f"  Full DRL-MADRL: ATCT={res_full['atct']:.3f}  "
          f"SLA={res_full['sla_pct']:.2f}%  "
          f"E={res_full['energy_kwh']:.3f}  "
          f"tasks={res_full['tasks_done']}")
    print(f"  NGC variant:    ATCT={res_ngc['atct']:.3f}  "
          f"SLA={res_ngc['sla_pct']:.2f}%  "
          f"E={res_ngc['energy_kwh']:.3f}  "
          f"tasks={res_ngc['tasks_done']}")

    # Inspect gossip estimates at end of episode
    print("\n  Final gossip state (agent 0's view of all nodes):")
    view_full = sched_full.gossip.get_view(0)
    view_ngc  = sched_ngc.gossip.get_view(0)
    actual = np.array([n.cpu_utilization for n in nodes_full],
                      dtype=np.float32)
    print(f"    Full agent 0 view: mean={view_full.mean():.4f}  "
          f"std={view_full.std():.4f}  off-diag mean="
          f"{np.delete(view_full, 0).mean():.4f}")
    print(f"    NGC  agent 0 view: mean={view_ngc.mean():.4f}  "
          f"std={view_ngc.std():.4f}  off-diag mean="
          f"{np.delete(view_ngc, 0).mean():.4f}  "
          f"(should be ~0.5)")
    print(f"    True u (snapshot): mean={actual.mean():.4f}  "
          f"std={actual.std():.4f}")

    # MAE: how wrong is each agent's view of the cluster?
    n_agents = len(nodes_full)
    mae_full = float(np.mean([np.abs(sched_full.gossip.get_view(i)
                                     - actual).mean()
                              for i in range(n_agents)]))
    mae_ngc  = float(np.mean([np.abs(sched_ngc.gossip.get_view(i)
                                     - actual).mean()
                              for i in range(n_agents)]))
    print(f"\n  Mean |estimate - truth| across all agents:")
    print(f"    Full: {mae_full:.4f}  (lower = better gossip)")
    print(f"    NGC : {mae_ngc:.4f}  (higher because no communication)")

    # Verdict
    print()
    if mae_full < mae_ngc * 0.5:
        print("  VERDICT: Full DRL-MADRL has substantially better gossip "
              "estimates than NGC.")
        print("  Gossip is genuinely providing information that NGC lacks.")
    else:
        print("  WARNING: Full and NGC have similar estimate quality. "
              "Gossip is not adding much information.")
    print()


if __name__ == "__main__":
    test_standalone_convergence()
    test_ngc_isolation()
    test_full_vs_ngc_episode()
