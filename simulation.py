"""
simulation.py
=============
Discrete-event simulation runner for DRL-MADRL.
Generates the data behind Table II and the V-E ablation:
  - DRL-MADRL vs Random / Weighted-Round-Robin / Priority-Min-Min /
    PCH / PPO / MADDPG / MAPPO

Implementation details:
  * WRR is capacity-proportional (matches Sec. V-A.2).
  * PPO / MADDPG / MAPPO share the same shaped reward as DRL-MADRL with frozen
    weights at their initial values (Sec. V-A.4: 'Same reward structure
    as DRL-MADRL, minus gossip and adaptive components').
  * PPO / MAPPO use proper clipped-ratio updates with entropy coefficient 0.01
    (Table I).
  * MADDPG uses a 10,000-capacity replay buffer with batch size 64 (Table I)
    and Polyak-averaged target networks (tau = 0.005).
  * Ablation variants of DRL-MADRL (NGC / NRS / NER / NPS) are exposed via
    `SCHEDULERS` keys for Sec. V-E reproduction.
  * Per-episode metrics are logged to `results/raw_episodes.csv` for the
    learning-curve and significance analysis in `analysis.py`.

Usage:
    python simulation.py
"""

import os
import csv
import time
import json
import argparse
import numpy as np
from typing import List, Dict, Optional

from marl_scheduler import (
    Task, Node, InfrastructureGenerator, WorkloadGenerator,
    DRLMADRLScheduler, build_observation, GossipConsensus, PerAgentGossip,
    AdaptiveRewardShaper, DEFAULT_REWARD_WEIGHTS,
    reward_components, weighted_reward, task_features,
    priority_score, set_assignment_weights,
)


# ---------------------------------------------------------------------------
# BASELINE SCHEDULERS
# ---------------------------------------------------------------------------

class RandomScheduler:
    def __init__(self, nodes, seed=42):
        self.nodes = nodes
        self.rng   = np.random.default_rng(seed)

    def schedule_task(self, task, current_time):
        feasible = [n for n in self.nodes if n.can_accept(task)]
        if not feasible:
            return None
        return self.rng.choice(feasible)

    def on_task_complete(self, *args, **kwargs): pass
    def gossip_step(self): pass


class WeightedRoundRobinScheduler:
    """Capacity-proportional weighted round robin (Sec. V-A.2).

    Each node is repeated in the dispatch sequence in proportion to its CPU
    capacity. We use deficit-weighted scheduling to avoid building a giant
    sequence: each node carries a `deficit` counter that increases by its
    weight on every dispatch attempt; we pick the feasible node with the
    largest deficit and decrement by the total weight.
    """
    def __init__(self, nodes):
        self.nodes   = nodes
        self.weights = np.array([n.cpu_capacity for n in nodes], dtype=np.float64)
        self.weights = self.weights / self.weights.sum()
        self.deficit = np.zeros(len(nodes), dtype=np.float64)

    def schedule_task(self, task, current_time):
        self.deficit += self.weights
        feasible_ids = [n.node_id for n in self.nodes if n.can_accept(task)]
        if not feasible_ids:
            return None
        # Pick feasible node with max deficit
        best = max(feasible_ids, key=lambda i: self.deficit[i])
        self.deficit[best] -= 1.0
        return self.nodes[best]

    def on_task_complete(self, *args, **kwargs): pass
    def gossip_step(self): pass


class PriorityMinMinScheduler:
    """Priority-aware Min-Min (Sec. V-A.3).

    Tasks are dispatched in priority order by the simulator (see Simulator);
    this scheduler then assigns to the least-loaded feasible node.
    """
    def __init__(self, nodes):
        self.nodes = nodes

    def schedule_task(self, task, current_time):
        feasible = [n for n in self.nodes if n.can_accept(task)]
        if not feasible:
            return None
        return min(feasible, key=lambda n: n.cpu_utilization)

    def on_task_complete(self, *args, **kwargs): pass
    def gossip_step(self): pass


class PriorityCapacityHeuristicScheduler:
    """Pure heuristic: Eq. 13-14 with NO learned policy term (w_pi = 0).

    This baseline isolates 'what does the priority-capacity heuristic alone
    achieve' — i.e., what fraction of DRL-MADRL's performance comes from
    the heuristic vs the RL component. If full DRL-MADRL beats this PCH
    baseline, the RL adds value; if they're statistically tied, the
    framework is essentially a heuristic with an RL wrapper.

    Uses the same Eq. 13 priority score and Eq. 14 assignment score as
    DRL-MADRL but with policy weight w_pi forced to 0 and a uniform
    `pi` vector (so the policy term contributes nothing). No actor-critic,
    no gossip, no replay, no learning.
    """
    def __init__(self, nodes):
        self.nodes = nodes
        N = len(nodes)
        # Uniform pi makes the policy term constant across candidates.
        self._uniform_pi = np.full(N, 1.0/N, dtype=np.float32)

    def schedule_task(self, task, current_time):
        feasible = [n for n in self.nodes if n.can_accept(task)]
        if not feasible:
            return None
        # Save and override Eq. 14 weights to zero out the policy term.
        from marl_scheduler import (get_assignment_weights,
                                     set_assignment_weights, select_best_node)
        saved = get_assignment_weights()
        set_assignment_weights(pi=0.0,
                               u=saved["u"], m=saved["m"],
                               c=saved["c"], p=saved["p"])
        try:
            chosen = select_best_node(task, self.nodes, self._uniform_pi,
                                      current_time)
        finally:
            set_assignment_weights(**saved)
        return chosen

    def on_task_complete(self, *args, **kwargs): pass
    def gossip_step(self): pass


# ---------------------------------------------------------------------------
# LIGHTWEIGHT PPO / MAPPO ACTOR NETWORK (with proper clip + entropy bonus)
# ---------------------------------------------------------------------------

def _clip_norm(g, max_norm=5.0):
    """In-place gradient norm clipping for stability."""
    n = float(np.linalg.norm(g))
    if n > max_norm and n > 0:
        g *= (max_norm / n)
    return g


class PolicyValueNetwork:
    """Shared trunk feedforward network with policy and value heads.

    Implements PPO-style clipped surrogate update with entropy bonus.
    Used as the actor for PPO baseline and MADDPG/MAPPO actors.
    Includes value clamping and gradient norm clipping for numerical safety
    when reward magnitudes are large.
    """

    def __init__(self, input_dim, hidden_dim, output_dim, lr=1e-3,
                 clip_eps=0.2, entropy_coeff=0.01, seed=42,
                 value_clip=1e3, grad_clip=5.0):
        rng = np.random.default_rng(seed)
        s1  = np.sqrt(2.0 / input_dim)
        s2  = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.normal(0, s1, (hidden_dim, input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, s2, (output_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        self.Wv = rng.normal(0, s2, (1, hidden_dim)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)
        self.lr            = lr
        self.clip_eps      = clip_eps
        self.entropy_coeff = entropy_coeff
        self.value_clip    = value_clip
        self.grad_clip     = grad_clip

    def _h(self, obs):
        return np.maximum(0.0, self.W1 @ obs + self.b1)

    def policy(self, obs):
        h = self._h(obs)
        logits = self.W2 @ h + self.b2
        logits -= logits.max()
        e = np.exp(logits)
        s = e.sum()
        if s == 0 or not np.isfinite(s):
            return np.ones_like(e) / len(e)
        return e / s

    def value(self, obs):
        v = float((self.Wv @ self._h(obs) + self.bv).item())
        return float(np.clip(v, -self.value_clip, self.value_clip))

    def update(self, obs, action, advantage, value_target, pi_old_action):
        """PPO clipped surrogate gradient + entropy bonus + value MSE."""
        if not (np.isfinite(value_target) and np.isfinite(advantage)):
            return

        h  = self._h(obs)
        pi = self.policy(obs)
        v  = float(np.clip((self.Wv @ h + self.bv).item(),
                           -self.value_clip, self.value_clip))

        # Clipped ratio gradient: d/dtheta min(r*A, clip(r,1-eps,1+eps)*A)
        ratio = pi[action] / max(pi_old_action, 1e-9)
        if (advantage >= 0 and ratio > 1.0 + self.clip_eps) or \
           (advantage <  0 and ratio < 1.0 - self.clip_eps):
            surrogate_coeff = 0.0
        else:
            surrogate_coeff = advantage * ratio

        dlog = -pi.copy(); dlog[action] += 1.0

        log_pi   = np.log(pi + 1e-12)
        H        = float(-np.sum(pi * log_pi))
        dH_dlog  = -pi * (log_pi + H)

        dlogits = surrogate_coeff * dlog + self.entropy_coeff * dH_dlog

        # Critic gradients (gradient of MSE loss L = (V - target)^2).
        # Output-layer gradients (dLv_dWv, dLv_dbv) are subtracted in the
        # parameter update for proper gradient descent on L.
        # For the shared hidden layer, the combined update ASCENDS
        # (d_pre*obs etc.); to make the critic's contribution descend L,
        # we use +2*td*Wv (ascent of -L) instead of -2*td*Wv (ascent of L).
        td      = float(np.clip(value_target - v, -self.value_clip, self.value_clip))
        dLv_dbv = np.array([-2*td], dtype=np.float32)
        dLv_dWv = -2*td * h.reshape(1, -1)
        dLv_dh_descent = +2*td * self.Wv.flatten()  # ascent direction for -L

        dJ_dh = self.W2.T @ dlogits
        d_pre = (dLv_dh_descent + dJ_dh) * (h > 0).astype(np.float32)
        dW1   = np.outer(d_pre, obs)
        db1   = d_pre
        dW2   = np.outer(dlogits, h)
        db2   = dlogits

        for g in (dW1, db1, dW2, db2, dLv_dWv, dLv_dbv):
            _clip_norm(g, self.grad_clip)

        self.W2 += self.lr * dW2
        self.b2 += self.lr * db2
        self.Wv -= self.lr * dLv_dWv
        self.bv -= self.lr * dLv_dbv
        self.W1 += self.lr * dW1
        self.b1 += self.lr * db1


# ---------------------------------------------------------------------------
# REWARD HELPER (shared by all DRL/MARL baselines)
# ---------------------------------------------------------------------------

def baseline_reward(task, finish_time, energy_step, utilizations):
    """Eq. (16) with frozen default weights — used by PPO / MADDPG / MAPPO."""
    r_sla, r_compl, r_energy, r_balance, _ = reward_components(
        task, finish_time, energy_step, utilizations)
    return weighted_reward(DEFAULT_REWARD_WEIGHTS,
                           r_sla, r_compl, r_energy, r_balance)


# ---------------------------------------------------------------------------
# PPO BASELINE  (Sec. V-A.4)
# ---------------------------------------------------------------------------

class PPOScheduler:
    """Single-agent PPO with full system state + task features (Path-alpha
    extension), shaped reward minus gossip/adaptive."""

    def __init__(self, nodes, seed=42):
        self.nodes = nodes
        N = len(nodes)
        # Input: per-node CPU utilization (N) concatenated with the 4 task
        # features (cpu_req, mem_req, priority, deadline_slack).
        self.net    = PolicyValueNetwork(input_dim=N + 4, hidden_dim=128,
                                         output_dim=N,
                                         lr=1e-3, clip_eps=0.2, entropy_coeff=0.01,
                                         seed=seed)
        self.gamma  = 0.95
        self._last  = None  # (obs, action, pi_old_action)

    def _obs(self, task=None, current_time=0.0):
        utils = np.array([n.cpu_utilization for n in self.nodes], dtype=np.float32)
        return np.concatenate([utils, task_features(task, current_time)])

    def schedule_task(self, task, current_time):
        obs      = self._obs(task, current_time)
        pi       = self.net.policy(obs)
        feasible = [n for n in self.nodes if n.can_accept(task)]
        if not feasible:
            return None
        mask = np.zeros(len(self.nodes), dtype=np.float32)
        for n in feasible:
            mask[n.node_id] = 1.0
        pi_masked = pi * mask
        if pi_masked.sum() < 1e-9:
            chosen = feasible[0]
        else:
            pi_masked /= pi_masked.sum()
            chosen = self.nodes[int(np.argmax(pi_masked))]
        self._last = (obs, chosen.node_id, float(pi[chosen.node_id]))
        return chosen

    def on_task_complete(self, task, node, current_time, energy_step):
        if self._last is None:
            return
        obs, action, pi_old = self._last
        utils  = np.array([n.cpu_utilization for n in self.nodes], dtype=np.float32)
        reward = baseline_reward(task, current_time, energy_step, utils)
        v      = self.net.value(obs)
        adv    = reward - v
        self.net.update(obs, action, adv, reward, pi_old)
        self._last = None

    def gossip_step(self): pass


# ---------------------------------------------------------------------------
# MADDPG  (Sec. V-A.5, Table I: replay 10,000 / batch 64)
# ---------------------------------------------------------------------------

class CentralizedCriticNetwork:
    """Centralized critic taking joint state (and optionally action) for
    MADDPG / MAPPO.

    Supports an optional target network with Polyak-averaged soft updates.
    Used by MADDPG to stabilize off-policy learning (without target
    networks the critic bootstraps from itself, which can cause divergence
    over long training horizons; this is the canonical issue addressed by
    target networks in Lillicrap et al. 2015 / Lowe et al. 2017).
    """

    def __init__(self, state_dim, hidden_dim=128, lr=1e-3, seed=42,
                 value_clip=1e3, grad_clip=5.0, use_target=False, tau=0.005):
        rng = np.random.default_rng(seed)
        s   = np.sqrt(2.0 / state_dim)
        self.W1 = rng.normal(0, s, (hidden_dim, state_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0/hidden_dim), (1, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self.lr         = lr
        self.value_clip = value_clip
        self.grad_clip  = grad_clip
        self.use_target = use_target
        self.tau        = float(tau)
        if use_target:
            # Target network = exact copy of online network at init.
            self.target_W1 = self.W1.copy()
            self.target_b1 = self.b1.copy()
            self.target_W2 = self.W2.copy()
            self.target_b2 = self.b2.copy()

    def target_value(self, joint_state):
        """Value estimate using TARGET network (used for Bellman target).
        Falls back to online network if no target is configured."""
        if not self.use_target:
            return self.value(joint_state)
        h = np.maximum(0.0, self.target_W1 @ joint_state + self.target_b1)
        v = float((self.target_W2 @ h + self.target_b2).item())
        return float(np.clip(v, -self.value_clip, self.value_clip))

    def soft_update_target(self):
        """Polyak update: target <- tau * online + (1 - tau) * target."""
        if not self.use_target:
            return
        self.target_W1 = self.tau * self.W1 + (1.0 - self.tau) * self.target_W1
        self.target_b1 = self.tau * self.b1 + (1.0 - self.tau) * self.target_b1
        self.target_W2 = self.tau * self.W2 + (1.0 - self.tau) * self.target_W2
        self.target_b2 = self.tau * self.b2 + (1.0 - self.tau) * self.target_b2

    def value(self, joint_state):
        h = np.maximum(0.0, self.W1 @ joint_state + self.b1)
        v = float((self.W2 @ h + self.b2).item())
        return float(np.clip(v, -self.value_clip, self.value_clip))

    def update(self, joint_state, target):
        if not np.isfinite(target):
            return
        h  = np.maximum(0.0, self.W1 @ joint_state + self.b1)
        v  = float(np.clip((self.W2 @ h + self.b2).item(),
                           -self.value_clip, self.value_clip))
        td = float(np.clip(target - v, -self.value_clip, self.value_clip))
        dW2 = -2*td * h.reshape(1, -1)
        db2 = np.array([-2*td], dtype=np.float32)
        d_h = (-2*td * self.W2.flatten()) * (h > 0)
        dW1 = np.outer(d_h, joint_state)
        for g in (dW1, d_h, dW2, db2):
            _clip_norm(g, self.grad_clip)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 += self.lr * dW1
        self.b1 += self.lr * d_h


class _ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.capacity = capacity
        self.buffer   = []
        self.pos      = 0

    def push(self, item):
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.pos] = item
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, rng):
        n = min(batch_size, len(self.buffer))
        idx = rng.choice(len(self.buffer), size=n, replace=False)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class MADDPGScheduler:
    """MADDPG: centralized critic, decentralized actors with local + gossip
    observation.

    Canonical CTDE: each actor sees only its own local state (4 features)
    plus task features (4 features) plus gossip-derived estimates of
    cluster mean utilization (1 feature). Centralized critic during training
    sees full joint state. Replay buffer 10,000, batch 64 (Table I).

    Each actor's input is 4 + 4 + 1 = 9-D, NOT the full N-D system state.
    This matches canonical MADDPG decentralization at execution.
    """

    LOCAL_OBS_DIM = 4 + 4 + 1   # local node state (4) + task (4) + gossip mean (1)

    def __init__(self, nodes, seed=42):
        self.nodes = nodes
        N = len(nodes)
        self.actors = [PolicyValueNetwork(self.LOCAL_OBS_DIM, 128, N, lr=1e-3,
                                          clip_eps=0.2, entropy_coeff=0.01,
                                          seed=seed+i) for i in range(N)]
        # Centralized critic with TARGET network (Lillicrap 2015 / Lowe 2017
        # style soft updates) for off-policy stability. Without target,
        # the critic bootstraps from itself and can degrade over long
        # training horizons.
        self.critic = CentralizedCriticNetwork(N*2 + N + 4, 128, lr=1e-3,
                                               seed=seed,
                                               use_target=True, tau=0.005)
        self.gamma  = 0.95
        self.buffer = _ReplayBuffer(capacity=10_000)
        self.batch_size = 64
        self.rng    = np.random.default_rng(seed)
        # Same gossip mechanism as DRL-MADRL for fair comparison.
        self.gossip = PerAgentGossip(N, gossip_prob=0.3, avg_weight=0.5,
                                     refresh_rate=0.1, default_estimate=0.5,
                                     seed=seed)
        utils = np.array([n.cpu_utilization for n in nodes], dtype=np.float32)
        self.gossip.initialize(utils)
        self._last  = None

    def _local_obs(self, agent_id, task=None, current_time=0.0):
        node = self.nodes[agent_id]
        local = np.array([
            node.cpu_utilization,
            node.mem_utilization,
            min(node.queue_length / 20.0, 1.0),
            node.cpu_capacity / 32.0,
        ], dtype=np.float32)
        gossip_mean = np.array([self.gossip.get_view(agent_id).mean()],
                               dtype=np.float32)
        return np.concatenate([local, task_features(task, current_time),
                               gossip_mean])

    def _joint_state(self):
        utils = [n.cpu_utilization for n in self.nodes]
        mems  = [n.mem_utilization for n in self.nodes]
        return np.array(utils + mems, dtype=np.float32)

    def _critic_input(self, joint_state, action, tfeat):
        N = len(self.nodes)
        a = np.zeros(N, dtype=np.float32); a[action] = 1.0
        return np.concatenate([joint_state, a, tfeat])

    def schedule_task(self, task, current_time):
        feasible  = [n for n in self.nodes if n.can_accept(task)]
        if not feasible:
            return None
        ingress = min(feasible, key=lambda n: n.cpu_utilization)
        local_obs = self._local_obs(ingress.node_id, task, current_time)
        joint     = self._joint_state()
        pi        = self.actors[ingress.node_id].policy(local_obs)
        mask      = np.zeros(len(self.nodes), dtype=np.float32)
        for n in feasible:
            mask[n.node_id] = 1.0
        pim = pi * mask
        if pim.sum() < 1e-9:
            chosen = feasible[0]
        else:
            pim /= pim.sum()
            chosen = self.nodes[int(np.argmax(pim))]
        tfeat = task_features(task, current_time)
        self._last = (joint, local_obs, chosen.node_id,
                      float(pi[chosen.node_id]), ingress.node_id, tfeat)
        return chosen

    def on_task_complete(self, task, node, current_time, energy_step):
        if self._last is None:
            return
        joint, local_obs, action, pi_old, agent_id, tfeat = self._last
        utils  = np.array([n.cpu_utilization for n in self.nodes], dtype=np.float32)
        reward = baseline_reward(task, current_time, energy_step, utils)
        next_joint = self._joint_state()
        next_local = self._local_obs(agent_id, task=None, current_time=current_time)
        next_tfeat = task_features(None, current_time)
        self.buffer.push((joint, local_obs, action, pi_old,
                          agent_id, reward, next_joint, next_local,
                          tfeat, next_tfeat))
        self._last = None

        if len(self.buffer) < self.batch_size:
            return
        batch = self.buffer.sample(self.batch_size, self.rng)
        for js, lobs, a, p_old, aid, r, njs, nlobs, tf, ntf in batch:
            # Bellman target uses TARGET critic for stability (Lowe 2017).
            pi_next = self.actors[aid].policy(nlobs)
            a_next  = int(np.argmax(pi_next))
            target  = r + self.gamma * self.critic.target_value(
                self._critic_input(njs, a_next, ntf))
            cin     = self._critic_input(js, a, tf)
            self.critic.update(cin, target)
            adv = target - self.critic.value(cin)
            self.actors[aid].update(lobs, a, adv, target, p_old)
        # Polyak soft-update target network after each batch of updates.
        self.critic.soft_update_target()

    def gossip_step(self):
        utils = np.array([n.cpu_utilization for n in self.nodes],
                         dtype=np.float32)
        self.gossip.step(utils, enable_communication=True)


# ---------------------------------------------------------------------------
# MAPPO  (Sec. V-A.6, Table I: shared critic, no replay)
# ---------------------------------------------------------------------------

class MAPPOScheduler:
    """MAPPO: multi-agent PPO with a shared centralized critic, no replay.

    Canonical CTDE: each actor sees only its own local state (4 features) +
    task features (4 features) + gossip-derived cluster mean utilization
    (1 feature), NOT the full system state. The critic sees the full system
    state during training. Same gossip mechanism as DRL-MADRL for fair
    comparison.
    """

    LOCAL_OBS_DIM = 4 + 4 + 1   # local node state (4) + task (4) + gossip mean (1)

    def __init__(self, nodes, seed=42):
        self.nodes = nodes
        N = len(nodes)
        self.actors = [PolicyValueNetwork(self.LOCAL_OBS_DIM, 128, N, lr=1e-3,
                                          clip_eps=0.2, entropy_coeff=0.01,
                                          seed=seed+i) for i in range(N)]
        # Centralized critic sees the full system state (CPU utilizations).
        self.critic = CentralizedCriticNetwork(N + 4, 128, lr=1e-3, seed=seed)
        self.gamma  = 0.95
        self.gossip = PerAgentGossip(N, gossip_prob=0.3, avg_weight=0.5,
                                     refresh_rate=0.1, default_estimate=0.5,
                                     seed=seed)
        utils = np.array([n.cpu_utilization for n in nodes], dtype=np.float32)
        self.gossip.initialize(utils)
        self._last  = None

    def _local_obs(self, agent_id, task=None, current_time=0.0):
        node = self.nodes[agent_id]
        local = np.array([
            node.cpu_utilization,
            node.mem_utilization,
            min(node.queue_length / 20.0, 1.0),
            node.cpu_capacity / 32.0,
        ], dtype=np.float32)
        gossip_mean = np.array([self.gossip.get_view(agent_id).mean()],
                               dtype=np.float32)
        return np.concatenate([local, task_features(task, current_time),
                               gossip_mean])

    def _critic_state(self, task=None, current_time=0.0):
        # Critic sees full true utilization vector (CTDE: centralized training)
        utils = np.array([n.cpu_utilization for n in self.nodes], dtype=np.float32)
        return np.concatenate([utils, task_features(task, current_time)])

    def schedule_task(self, task, current_time):
        feasible = [n for n in self.nodes if n.can_accept(task)]
        if not feasible:
            return None
        ingress = min(feasible, key=lambda n: n.cpu_utilization)
        local_obs    = self._local_obs(ingress.node_id, task, current_time)
        critic_state = self._critic_state(task, current_time)
        pi = self.actors[ingress.node_id].policy(local_obs)
        mask = np.zeros(len(self.nodes), dtype=np.float32)
        for n in feasible:
            mask[n.node_id] = 1.0
        pim = pi * mask
        if pim.sum() < 1e-9:
            chosen = feasible[0]
        else:
            pim /= pim.sum()
            chosen = self.nodes[int(np.argmax(pim))]
        self._last = (local_obs, critic_state, chosen.node_id,
                      float(pi[chosen.node_id]), ingress.node_id)
        return chosen

    def on_task_complete(self, task, node, current_time, energy_step):
        if self._last is None:
            return
        local_obs, critic_state, action, pi_old, agent_id = self._last
        utils  = np.array([n.cpu_utilization for n in self.nodes], dtype=np.float32)
        reward = baseline_reward(task, current_time, energy_step, utils)
        v      = self.critic.value(critic_state)
        adv    = reward - v
        self.critic.update(critic_state, reward)
        self.actors[agent_id].update(local_obs, action, adv, reward, pi_old)
        self._last = None

    def gossip_step(self):
        utils = np.array([n.cpu_utilization for n in self.nodes],
                         dtype=np.float32)
        self.gossip.step(utils, enable_communication=True)


# ---------------------------------------------------------------------------
# SIMULATOR
# ---------------------------------------------------------------------------

class Simulator:
    """
    Discrete-event simulation. Time step = 5 s.
    Tracks ATCT, energy (kWh), SLA satisfaction, throughput.
    """

    def __init__(self, nodes: List[Node], scheduler, time_step=5.0):
        self.nodes      = nodes
        self.scheduler  = scheduler
        self.time_step  = time_step

    def _reset_nodes(self):
        for n in self.nodes:
            n.cpu_used = 0.0
            n.mem_used = 0.0
            n.running_tasks.clear()
            n.queue.clear()

    def run_episode(self, tasks: List[Task]) -> Dict:
        self._reset_nodes()

        pending    = sorted(tasks, key=lambda t: t.arrival_time)
        completed  = []
        current_time = 0.0
        total_energy = 0.0
        max_time   = pending[-1].arrival_time + 3600.0 if pending else 3600.0

        while (pending or any(n.running_tasks for n in self.nodes)) and current_time < max_time:
            arrived = [t for t in pending if t.arrival_time <= current_time]
            arrived.sort(key=lambda t: -(3 - t.priority))
            for task in arrived:
                pending.remove(task)
                chosen = self.scheduler.schedule_task(task, current_time)
                if chosen:
                    chosen.assign(task, current_time)
                else:
                    pending.append(task)

            for node in self.nodes:
                total_energy += node.instantaneous_power() * self.time_step

            self.scheduler.gossip_step()

            current_time += self.time_step

            for node in self.nodes:
                done = [t for t in node.running_tasks
                        if t.start_time is not None and
                           current_time >= t.start_time + t.effective_duration]
                for task in done:
                    node.release(task, current_time)
                    energy_step = node.instantaneous_power() * self.time_step
                    self.scheduler.on_task_complete(task, node, current_time, energy_step)
                    completed.append(task)

        n_done     = len(completed)
        atct       = float(np.mean([t.finish_time - t.arrival_time
                                    for t in completed])) if completed else 0.0
        sla_rate   = float(np.mean([t.met_sla for t in completed])) * 100 if completed else 0.0
        energy_kwh = total_energy / 3_600_000.0

        return {
            "atct":       atct,
            "energy_kwh": energy_kwh,
            "sla_pct":    sla_rate,
            "tasks_done": n_done,
        }


# ---------------------------------------------------------------------------
# SCHEDULER REGISTRY (all baselines + ablation variants)
# ---------------------------------------------------------------------------

SEEDS = [42, 43, 44, 45, 46]

SCHEDULERS = {
    "Random":           lambda nodes, seed: RandomScheduler(nodes, seed),
    "WeightedRR":       lambda nodes, seed: WeightedRoundRobinScheduler(nodes),
    "PriorityMinMin":   lambda nodes, seed: PriorityMinMinScheduler(nodes),
    "PCH":              lambda nodes, seed: PriorityCapacityHeuristicScheduler(nodes),
    "PPO":              lambda nodes, seed: PPOScheduler(nodes, seed),
    "MADDPG":           lambda nodes, seed: MADDPGScheduler(nodes, seed),
    "MAPPO":            lambda nodes, seed: MAPPOScheduler(nodes, seed),
    "DRL-MADRL":        lambda nodes, seed: DRLMADRLScheduler(nodes, seed=seed),
    # Adaptive-w_pi variant: blends Eq. 14 (heuristic) with policy
    # argmax (NPS) based on observed contention level. Should match
    # full DRL-MADRL at low contention and NPS at high contention.
    "DRL-MADRL-Adaptive": lambda nodes, seed: DRLMADRLScheduler(
        nodes, seed=seed, contention_adaptive=True),
    # Sec. V-E ablations
    "DRL-MADRL-NGC":    lambda nodes, seed: DRLMADRLScheduler(nodes, seed=seed, use_gossip=False),
    "DRL-MADRL-NRS":    lambda nodes, seed: DRLMADRLScheduler(nodes, seed=seed, use_adaptive_reward=False),
    "DRL-MADRL-NER":    lambda nodes, seed: DRLMADRLScheduler(nodes, seed=seed, use_replay=False),
    "DRL-MADRL-NPS":    lambda nodes, seed: DRLMADRLScheduler(nodes, seed=seed, use_priority_scoring=False),
}

PAPER_METHODS = ["Random", "WeightedRR", "PriorityMinMin", "PCH",
                 "PPO", "MADDPG", "MAPPO", "DRL-MADRL"]
ABLATION_METHODS = ["DRL-MADRL", "DRL-MADRL-NGC", "DRL-MADRL-NRS",
                    "DRL-MADRL-NER", "DRL-MADRL-NPS", "PCH"]
ABLATION_FAST = ["DRL-MADRL", "DRL-MADRL-NGC", "DRL-MADRL-NRS"]
ADAPTIVE_COMPARISON = ["DRL-MADRL", "DRL-MADRL-Adaptive", "DRL-MADRL-NPS", "PCH"]


# ---------------------------------------------------------------------------
# EXPERIMENT RUNNER (with per-episode logging)
# ---------------------------------------------------------------------------

def run_experiment(scheduler_name: str, num_episodes=30, num_tasks=1000,
                   seeds=None, episode_log: Optional[list] = None,
                   arrival_rate: float = 0.5,
                   eval_window: Optional[int] = None) -> Dict:
    """Run `num_episodes` per seed, average final `eval_window` episodes per
    seed, aggregate mean/std across seeds. Optionally append per-episode rows
    to `episode_log` for learning-curve analysis.

    `eval_window`: number of trailing episodes to average. Default = max(10,
    num_episodes // 10), i.e. ~last 10% of training, with a floor of 10.

    `arrival_rate`: Poisson lambda for task arrivals (default 0.5)."""
    if seeds is None:
        seeds = SEEDS
    if eval_window is None:
        eval_window = max(10, num_episodes // 10)
    eval_window = min(eval_window, num_episodes)
    seed_runs = []

    for seed in seeds:
        nodes   = InfrastructureGenerator.create_nodes(num_nodes=100, seed=seed)
        wgen    = WorkloadGenerator(seed=seed)
        sched   = SCHEDULERS[scheduler_name](nodes, seed)
        sim     = Simulator(nodes, sched, time_step=5.0)

        ep_results = []
        for ep in range(num_episodes):
            # Per-episode reproducible workload (independent of cumulative RNG state)
            episode_seed = seed * 1_000_000 + ep
            tasks  = wgen.generate_tasks(num_tasks=num_tasks,
                                         arrival_rate=arrival_rate,
                                         episode_seed=episode_seed)
            result = sim.run_episode(tasks)
            ep_results.append(result)
            if episode_log is not None:
                episode_log.append({
                    "method":  scheduler_name,
                    "seed":    seed,
                    "episode": ep,
                    **result,
                })

        eval_eps = ep_results[-eval_window:]
        run_mean = {k: float(np.mean([e[k] for e in eval_eps]))
                    for k in eval_eps[0]}
        run_mean["seed"] = seed
        seed_runs.append(run_mean)

    metric_keys = [k for k in seed_runs[0] if k != "seed"]
    agg = {
        k: {
            "mean": float(np.mean([r[k] for r in seed_runs])),
            "std":  float(np.std( [r[k] for r in seed_runs], ddof=1))
                    if len(seed_runs) > 1 else 0.0,
            "values": [r[k] for r in seed_runs],
        }
        for k in metric_keys
    }
    return agg


def run_all(num_episodes=30, num_tasks=1000, methods=None,
            episode_log: Optional[list] = None,
            arrival_rate: float = 0.5,
            eval_window: Optional[int] = None,
            seeds=None,
            checkpoint_dir: Optional[str] = None):
    """Run all methods, optionally writing per-method CSV/JSON checkpoints
    so that a crash mid-run does not lose completed-method data.

    `checkpoint_dir`: if given, after each method finishes we (re)write
    `<dir>/raw_episodes.csv`, `<dir>/seed_means.csv`, `<dir>/results.json`."""
    if methods is None:
        methods = PAPER_METHODS
    results = {}
    t0 = time.time()
    for name in methods:
        print(f"  Running {name} ...", end=" ", flush=True)
        t1 = time.time()
        results[name] = run_experiment(name, num_episodes, num_tasks,
                                       seeds=seeds,
                                       episode_log=episode_log,
                                       arrival_rate=arrival_rate,
                                       eval_window=eval_window)
        dt = time.time() - t1
        r  = results[name]
        print(f"done ({dt:.0f}s)  "
              f"ATCT={r['atct']['mean']:.2f}s±{r['atct']['std']:.2f}  "
              f"SLA={r['sla_pct']['mean']:.2f}%  "
              f"Energy={r['energy_kwh']['mean']:.2f} kWh", flush=True)
        if checkpoint_dir is not None:
            os.makedirs(checkpoint_dir, exist_ok=True)
            with open(os.path.join(checkpoint_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
            if episode_log is not None:
                write_episode_csv(episode_log,
                                  os.path.join(checkpoint_dir, "raw_episodes.csv"))
            write_seed_csv(results,
                           os.path.join(checkpoint_dir, "seed_means.csv"),
                           seeds=seeds)
            print(f"    [checkpoint saved -> {checkpoint_dir}/]", flush=True)
    print(f"\nTotal time: {time.time()-t0:.0f}s")
    return results


def print_table(results: Dict):
    print("\n" + "="*88)
    print(f"{'Method':<22} {'ATCT(s)':>10} {'Std':>8} {'Energy(kWh)':>14} "
          f"{'SLA(%)':>10} {'Tasks':>8}")
    print("-"*88)
    for name, r in results.items():
        print(f"{name:<22} "
              f"{r['atct']['mean']:>10.1f} "
              f"{r['atct']['std']:>8.2f} "
              f"{r['energy_kwh']['mean']:>14.1f} "
              f"{r['sla_pct']['mean']:>10.1f} "
              f"{r['tasks_done']['mean']:>8.0f}")
    print("="*88)


def write_episode_csv(rows, path):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["method", "seed", "episode", "atct", "energy_kwh", "sla_pct", "tasks_done"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in keys})


def write_seed_csv(results: Dict, path, seeds=None):
    """Write per-seed last-K-episode means. `seeds` overrides the default
    SEEDS list when the run used a custom seed set."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = ["method", "seed", "atct", "energy_kwh", "sla_pct", "tasks_done"]
    seeds = seeds if seeds is not None else SEEDS
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for name, r in results.items():
            n = len(r["atct"]["values"])
            seed_iter = seeds[:n] if len(seeds) >= n else list(range(n))
            for i, s in enumerate(seed_iter):
                w.writerow({
                    "method":     name,
                    "seed":       s,
                    "atct":       r["atct"]["values"][i],
                    "energy_kwh": r["energy_kwh"]["values"][i],
                    "sla_pct":    r["sla_pct"]["values"][i],
                    "tasks_done": r["tasks_done"]["values"][i],
                })


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DRL-MADRL Simulation Runner")
    parser.add_argument("--episodes",   type=int,   default=30,   help="Training episodes")
    parser.add_argument("--tasks",      type=int,   default=1000, help="Tasks per episode")
    parser.add_argument("--arrival-rate", type=float, default=0.5,
                        help="Poisson lambda for task arrivals "
                             "(default 0.5; raise for contention)")
    parser.add_argument("--scheduler",  type=str,   default="all",
                        choices=list(SCHEDULERS.keys()) + ["all", "ablation",
                                                            "ablation-fast", "marl",
                                                            "adaptive"],
                        help="'all' = all baselines + DRL-MADRL; "
                             "'marl' = only PPO/MADDPG/MAPPO/DRL-MADRL; "
                             "'ablation' = DRL-MADRL + 4 ablation variants; "
                             "'ablation-fast' = DRL-MADRL + NGC + NRS (3 variants); "
                             "'adaptive' = Full vs Adaptive vs NPS vs PCH (4 variants)")
    parser.add_argument("--seeds",      type=str,   default=None,
                        help="Comma-separated seeds (e.g. '42,43,44,45,46'). "
                             "Default: 42..46 (5 seeds).")
    parser.add_argument("--eval-window", type=int,  default=None,
                        help="Trailing episodes to average for final metrics. "
                             "Default: max(10, episodes // 10).")
    parser.add_argument("--results-dir", type=str,  default="results",
                        help="Directory to write CSV / JSON outputs")
    parser.add_argument("--output",     type=str,   default="results.json")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    eval_window = args.eval_window
    print(f"DRL-MADRL Simulation  |  episodes={args.episodes}  tasks={args.tasks}  "
          f"lambda={args.arrival_rate}  seeds={seeds or SEEDS}  "
          f"eval_window={eval_window or 'auto'}\n")

    episode_log = []

    if args.scheduler == "all":
        method_set = PAPER_METHODS
    elif args.scheduler == "ablation":
        method_set = ABLATION_METHODS
    elif args.scheduler == "ablation-fast":
        method_set = ABLATION_FAST
    elif args.scheduler == "adaptive":
        method_set = ADAPTIVE_COMPARISON
    elif args.scheduler == "marl":
        method_set = ["PPO", "MADDPG", "MAPPO", "DRL-MADRL"]
    else:
        method_set = [args.scheduler]

    if len(method_set) == 1:
        print(f"  Running {method_set[0]} ...")
        results = {method_set[0]: run_experiment(method_set[0],
                                                  args.episodes, args.tasks,
                                                  seeds=seeds,
                                                  episode_log=episode_log,
                                                  arrival_rate=args.arrival_rate,
                                                  eval_window=eval_window)}
    else:
        results = run_all(args.episodes, args.tasks,
                          methods=method_set,
                          episode_log=episode_log,
                          arrival_rate=args.arrival_rate,
                          eval_window=eval_window,
                          seeds=seeds,
                          checkpoint_dir=args.results_dir)

    print_table(results)

    os.makedirs(args.results_dir, exist_ok=True)
    json_path = os.path.join(args.results_dir, args.output)
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    write_episode_csv(episode_log, os.path.join(args.results_dir, "raw_episodes.csv"))
    write_seed_csv(results,        os.path.join(args.results_dir, "seed_means.csv"),
                   seeds=seeds)
    print(f"\nResults saved to {args.results_dir}/")
