# DRL-MADRL: Decentralized Multi-Agent RL for Distributed Task Scheduling

Code and experimental data accompanying the paper:

> **"Decentralized Task Scheduling in Distributed Systems: A Lightweight,
> Contention-Adaptive Multi-Agent Deep Reinforcement Learning Approach with
> Gossip-Based Coordination"**
> *Daniel Benniah John* — IEEE Access, 2026.

A pure-NumPy implementation of a decentralized multi-agent scheduler for
heterogeneous distributed systems. Each agent maintains a per-node gossip-
derived utilization estimate vector and uses a contention-adaptive blend
of priority-aware heuristic scoring and learned policy to make placement
decisions. ~80 KB per agent, sub-10 ms decision latency, no deep-learning
framework dependency.

## Quick Start

```bash
git clone https://github.com/DanielBenniah/drl-marl-gossip-task-scheduler.git
cd drl-marl-gossip-task-scheduler
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Headline Table II at the default contention regime (lambda=0.5)
# 8 methods, 5 seeds, 30 episodes per seed
python simulation.py --arrival-rate 0.5 --results-dir results/lam05

# At higher contention (where DRL-MADRL's advantage materializes)
python simulation.py --arrival-rate 20 --results-dir results/lam20
python simulation.py --arrival-rate 50 --results-dir results/lam50

# Section V-E ablation (5 ablation variants + PCH baseline)
python simulation.py --arrival-rate 20 --scheduler ablation \
                     --results-dir results/ablation_lam20

# Bonferroni-corrected pairwise significance vs DRL-MADRL
python analysis.py --csv results/lam20/seed_means.csv \
                   --out results/lam20/significance.csv

# Per-method plateau verification (Welch t-test, first vs last quartile)
python verify_run.py --csv results/lam20/raw_episodes.csv

# Generate Figures 1, 2, 3 from a results directory
python plot_results.py --episodes results/lam20/raw_episodes.csv \
                       --seeds   results/lam20/seed_means.csv \
                       --out-dir results/lam20/figures
```

## What this codebase contains

The table below maps every component of the system to the source location
where it is implemented. Section labels (`Sec. III-D`, etc.) are
cross-references to the corresponding write-up in the paper.

| Component | Implemented in |
|---|---|
| Heterogeneous 100 nodes (20/50/30 tiers, Sec. III-B) | `InfrastructureGenerator.create_nodes` |
| Pareto duration α=1.5, log-normal CPU/mem, Poisson arrivals (Sec. III-C) | `WorkloadGenerator.generate_tasks` |
| Effective execution time *t<sub>exec</sub>* = *t<sub>j</sub>* · (*C*<sub>ref</sub>/*C<sub>i</sub>*), C<sub>ref</sub>=16 (Sec. III-D) | `Node.assign` |
| Linear power model (Sec. III-E) | `Node.instantaneous_power` |
| Per-agent vector gossip with O(N) payload, O(1) events (Sec. IV-C) | `PerAgentGossip` |
| Actor-critic, 128-ReLU hidden, softmax over N candidates, 50-D obs (Sec. IV-B) | `ActorCriticNetwork`, `build_observation` |
| Priority + capacity-weighted assignment score (Sec. IV-D, Eq. 14) | `priority_score`, `assignment_score`, `select_best_node` |
| Contention-adaptive score blend (Sec. IV-D.1) | `DRLMADRLScheduler` with `contention_adaptive=True` |
| Adaptive reward shaping w<sub>SLA</sub>, w<sub>energy</sub> (Sec. IV-E) | `AdaptiveRewardShaper` |
| Prioritized experience replay (Sec. IV-F) | `PrioritizedReplayBuffer` |
| PPO-clipped actor update + entropy bonus | `ActorCriticNetwork.update`, `PolicyValueNetwork.update` |
| Capacity-proportional Weighted Round-Robin baseline (Sec. V-A.2) | `WeightedRoundRobinScheduler` (deficit-weighted) |
| Priority-aware Min-Min baseline (Sec. V-A.3) | `PriorityMinMinScheduler` |
| Priority-Capacity Heuristic baseline (PCH, Sec. V-A) | `PriorityCapacityHeuristicScheduler` |
| Single-agent PPO with shaped reward (Sec. V-A.4) | `PPOScheduler` |
| Canonical CTDE MADDPG with target Q-network (τ=0.005, Sec. V-A.5) | `MADDPGScheduler` |
| Canonical CTDE MAPPO with shared centralized critic (Sec. V-A.6) | `MAPPOScheduler` |
| 30 episodes, last 10 averaged, 5 seeds 42–46 | `run_experiment` |
| Bonferroni-corrected pairwise t-tests on seed-level means | `analysis.py` |
| Section V-E ablations (NGC / NRS / NER / NPS) | `DRLMADRLScheduler` flags + registry keys |
| Plateau verification (Welch t-test, first vs last quartile) | `verify_run.py` |
| Hyperparameter sensitivity for the contention-adaptive blend | `run_sensitivity.py` |

## Repository layout

```
drl-marl-gossip-task-scheduler/
├── README.md                       # this file
├── LICENSE                         # MIT
├── requirements.txt                # NumPy, SciPy, Matplotlib (no PyTorch/TF)
├── RESULTS_SUMMARY.md              # supplementary results documentation
│
├── marl_scheduler.py               # DRL-MADRL core (Sections III-IV)
├── simulation.py                   # Discrete-event simulator + all baselines
├── analysis.py                     # Bonferroni significance testing
├── verify_run.py                   # Plateau detection / training health
├── plot_results.py                 # Figures 1, 2, 3 generator
├── run_sensitivity.py              # Hyperparameter sensitivity (Table VI)
├── diag_g2.py                      # Per-agent gossip mechanism diagnostic
│
├── results/                        # All experiment outputs
│   ├── g2_table2_lam05/            # Table II(a) source (lambda=0.5)
│   ├── g2_table2_lam20/            # Table II(b) source (lambda=20)
│   ├── g2_lam50/                   # Table II(c) source (lambda=50)
│   ├── g2_ablation_lam20/          # Table V (ablation) source
│   ├── sensitivity/                # Table VI (sensitivity scan) source
│   ├── long/                       # Figure 1 source (200-ep learning curves)
│   ├── adaptive_lam05/             # Adaptive-blend comparison (lambda=0.5)
│   ├── adaptive_lam20/             # Adaptive-blend comparison (lambda=20)
│   └── adaptive_lam50/             # Adaptive-blend comparison (lambda=50)
└── logs/                           # Run logs for each results directory
```

Each subdirectory of `results/` contains:
- `results.json` — aggregated mean/std per method
- `seed_means.csv` — one row per `(method, seed)` of last-K-episode means
  (input for `analysis.py`)
- `raw_episodes.csv` — one row per `(method, seed, episode)` (input for
  `plot_results.py` learning curves)
- `significance.csv` — Bonferroni-corrected pairwise t-test results
- `figures/figure{1,2,3}_*.png` — generated figures

## Architecture

DRL-MADRL agent network (per-agent, 100 agents in a 100-node cluster):

```
Input (50-dim observation: local node, gossip-derived cluster view, task)
        │
   Hidden (128 ReLU)
   ┌────┴────┐
Actor       Critic
softmax(N)  scalar V
```

~20K parameters per agent → ~80 KB float32. Total system footprint is
under 8 MB for all 100 agents. Per-agent gossip estimate vector adds
~400 bytes per agent.

Gossip (Section IV-C): each agent *i* maintains a vector
**z**<sub>*i*</sub> ∈ ℝ<sup>*N*</sup> of estimates of all *N* nodes'
utilization. With probability *p<sub>g</sub>* = 0.3 per step, agent *i*
selects a random neighbor *j*, and the two exchange and average their
estimate vectors with averaging weight *w* = 0.5. Diagonal entry
**z**<sub>*i*</sub>[*i*] is always pinned to true local utilization
*u<sub>i</sub>* (each agent knows its own state exactly). This is the
randomized pairwise gossip-averaging protocol of Boyd et al. 2006, which
converges to the consensus mean in O(log *N*) rounds.

## Command-line reference

### `simulation.py`

| Flag | Default | Notes |
|---|---|---|
| `--episodes` | 30 | Training episodes per seed |
| `--tasks` | 1000 | Tasks per episode |
| `--arrival-rate` | 0.5 | Poisson λ; raise for contention. The default λ=0.5 leaves the cluster lightly loaded; λ ∈ {20, 50} test high-contention regimes. |
| `--scheduler` | `all` | `all` = 8 methods (classical + heuristic + learned baselines + DRL-MADRL); `marl` = 4 learned methods only; `ablation` = DRL-MADRL + 4 V-E variants + PCH; `adaptive` = DRL-MADRL + Adaptive + NPS + PCH; or any single method name from the registry. |
| `--seeds` | `42,43,44,45,46` | Comma-separated seed list |
| `--eval-window` | auto | Trailing episodes averaged for final metrics. Defaults to `max(10, episodes // 10)`. |
| `--results-dir` | `results` | Output directory |

Schedulers in the registry: `Random`, `WeightedRR`, `PriorityMinMin`,
`PCH`, `PPO`, `MADDPG`, `MAPPO`, `DRL-MADRL`, `DRL-MADRL-Adaptive`,
`DRL-MADRL-NGC`, `DRL-MADRL-NRS`, `DRL-MADRL-NER`, `DRL-MADRL-NPS`.

### `analysis.py`

Reads `seed_means.csv` and writes `significance.csv`:
- Descriptive mean ± std across seeds
- Welch two-sample t-tests of every method vs. the `--reference` method
- Bonferroni correction at `--alpha 0.05` over (#methods − 1) × (#metrics)

```bash
python analysis.py --csv <DIR>/seed_means.csv \
                   --reference DRL-MADRL \
                   --alpha 0.05 \
                   --out <DIR>/significance.csv
```

### `verify_run.py`

Plateau verification using Welch t-tests on per-seed first vs last quartile
of training episodes. Reports a method as plateaued if no significant
trend is detected at α=0.05.

```bash
python verify_run.py --csv <DIR>/raw_episodes.csv
```

### `plot_results.py`

Generates Figures 1 (learning curves), 2 (bar comparison), and 3 (ablation)
from a results directory:

```bash
python plot_results.py --episodes <DIR>/raw_episodes.csv \
                       --seeds   <DIR>/seed_means.csv \
                       --out-dir <DIR>/figures
```

### `run_sensitivity.py`

Hyperparameter sensitivity scan for the contention-adaptive blend
(Section IV-D.1). Tests six (τ, w) settings at λ=20 with 3 seeds. Used
to generate Table VI.

```bash
python run_sensitivity.py
```

### `diag_g2.py`

Standalone diagnostic that verifies the per-agent gossip mechanism:
- Convergence test: synthetic utilization vector → mean absolute error
  across agents over time
- NGC isolation test: with communication disabled, off-diagonal estimates
  should remain at the uninformed prior (0.5)
- End-to-end test: full vs NGC scheduler in a real episode

```bash
python diag_g2.py
```

## Reproducing the reported results

Each reported table and figure has a corresponding directory in this
repository (see the "Repository layout" section above). To reproduce
from scratch:

```bash
# Table II — three contention regimes
for L in 0.5 20 50; do
  python simulation.py --arrival-rate $L \
                       --results-dir "results/g2_table2_lam${L/./}"
  python analysis.py --csv "results/g2_table2_lam${L/./}/seed_means.csv" \
                     --out "results/g2_table2_lam${L/./}/significance.csv"
done

# Table III — plateau verification (200 episodes)
python simulation.py --episodes 200 --eval-window 20 \
                     --arrival-rate 0.5 --results-dir results/long
python verify_run.py --csv results/long/raw_episodes.csv

# Table V — ablation under high contention
python simulation.py --arrival-rate 20 --scheduler ablation \
                     --results-dir results/g2_ablation_lam20
python analysis.py --csv results/g2_ablation_lam20/seed_means.csv

# Table VI — adaptive-blend hyperparameter sensitivity
python run_sensitivity.py

# Figure 1 — learning curves
python plot_results.py --episodes results/long/raw_episodes.csv \
                       --seeds   results/long/seed_means.csv \
                       --out-dir results/long/figures
```

Wall-clock estimates on commodity laptop hardware (Apple M-series, no GPU):
- Table II (one λ regime, 8 methods × 5 seeds × 30 ep): ~10–20 minutes
- Table V (ablation, 6 variants × 5 seeds × 30 ep): ~50 minutes
- Table VI (sensitivity, 6 settings × 3 seeds × 30 ep): ~35 minutes
- Figure 1 (200 ep × 5 seeds × 4 methods): ~80 minutes

## Notes on numerical stability

The reward energy term uses `r_energy = -0.3 · E_t` with `E_t` measured
in **kilojoules** (not raw Joules) so that the four reward components
operate on comparable scales. All learned networks additionally use:
- Value clamping to ±10³
- Gradient norm clipping at 5.0
- Skipping of updates with non-finite advantage / target

PPO-style clipped surrogate (ε = 0.2) and entropy regularization
(β = 0.01) are applied to all learned actor updates as standard MARL
stabilization practice. MADDPG's centralized critic uses target
Q-networks with Polyak-averaged soft updates (τ = 0.005) — required to
prevent off-policy bootstrapping divergence.

These engineering choices are reflected in the paper's algorithm
description. See `RESULTS_SUMMARY.md` for a fuller discussion of each
choice and the rationale behind it.

## Citation

```bibtex
@article{benniah2026drlmadrl,
  title   = {Decentralized Task Scheduling in Distributed Systems:
             A Lightweight, Contention-Adaptive Multi-Agent Deep
             Reinforcement Learning Approach with Gossip-Based Coordination},
  author  = {Daniel Benniah John},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
