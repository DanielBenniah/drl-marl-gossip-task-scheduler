"""
marl_scheduler.py
=================
DRL-MADRL: Decentralized Multi-Agent Deep Reinforcement Learning Scheduler
with Adaptive Decentralized Consensus via Gossip-Based Neighbor Coordination (ADC-GNC).

Paper: "Decentralized Task Scheduling in Distributed Systems:
        A Lightweight Multi-Agent Deep Reinforcement Learning Approach
        with Gossip-Based Consensus" (IEEE Access, 2026)
Author: Daniel Benniah John
"""

import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# 1. DATA STRUCTURES
# ---------------------------------------------------------------------------

class Task:
    def __init__(self, task_id, cpu_req, mem_req, duration,
                 arrival_time, priority, deadline):
        self.task_id      = task_id
        self.cpu_req      = cpu_req
        self.mem_req      = mem_req
        self.duration     = duration   # base duration on a reference node
        self.arrival_time = arrival_time
        self.priority     = priority   # 0=Production, 1=Batch, 2=Best-effort
        self.deadline     = deadline
        self.start_time   = None
        self.finish_time  = None
        self.assigned_node = None
        # Set when assigned: duration * Node.speed_factor (heterogeneous exec).
        self.effective_duration = duration

    @property
    def completed(self):
        return self.finish_time is not None

    @property
    def met_sla(self):
        return self.completed and self.finish_time <= self.deadline


# Reference capacity used by the heterogeneous execution-time model
# (Section III "Effective Execution Time"): t_exec = t_j * (C_ref / C_i).
# C_ref = mean of the medium tier (8..16 cores).
NODE_SPEED_REF_CAPACITY = 16.0


class Node:
    def __init__(self, node_id, cpu_capacity, mem_capacity,
                 idle_power, dynamic_power):
        self.node_id       = node_id
        self.cpu_capacity  = cpu_capacity
        self.mem_capacity  = mem_capacity
        self.idle_power    = idle_power
        self.dynamic_power = dynamic_power
        self.cpu_used      = 0.0
        self.mem_used      = 0.0
        self.running_tasks = []
        self.queue         = []

    @property
    def cpu_utilization(self):
        return self.cpu_used / max(self.cpu_capacity, 1e-9)

    @property
    def mem_utilization(self):
        return self.mem_used / max(self.mem_capacity, 1e-9)

    @property
    def queue_length(self):
        return len(self.queue)

    @property
    def speed_factor(self):
        """Heterogeneous execution-time multiplier (Section III, III-C').

        Wall-clock execution time of a task on this node is
            t_exec = base_duration * speed_factor
        with speed_factor = C_ref / C_i. High-tier nodes (e.g. 28 cores) are
        faster (factor ~0.57); low-tier nodes (e.g. 5 cores) are slower
        (factor ~3.2). This captures Dell R750 vs. Pi 4 / Jetson IPC and
        core-count gaps, which the workload's Pareto base duration alone does
        not encode.
        """
        return NODE_SPEED_REF_CAPACITY / max(self.cpu_capacity, 1.0)

    def can_accept(self, task):
        return (self.cpu_used + task.cpu_req <= self.cpu_capacity and
                self.mem_used + task.mem_req <= self.mem_capacity)

    def assign(self, task, current_time):
        self.cpu_used += task.cpu_req
        self.mem_used += task.mem_req
        task.assigned_node = self.node_id
        task.start_time    = current_time
        # Effective wall-clock duration on this specific node.
        task.effective_duration = task.duration * self.speed_factor
        self.running_tasks.append(task)

    def release(self, task, current_time):
        self.cpu_used -= task.cpu_req
        self.mem_used -= task.mem_req
        task.finish_time  = current_time
        if task in self.running_tasks:
            self.running_tasks.remove(task)

    def instantaneous_power(self):
        """Linear power model: P_i(t) = P_idle + P_dyn * u_i(t)  [Eq.3]"""
        return self.idle_power + self.dynamic_power * self.cpu_utilization


# ---------------------------------------------------------------------------
# 2. INFRASTRUCTURE GENERATOR  (Section III-A)
# ---------------------------------------------------------------------------

class InfrastructureGenerator:
    @staticmethod
    def create_nodes(num_nodes=100, seed=42):
        rng = np.random.default_rng(seed)
        nodes = []
        n_high   = int(0.20 * num_nodes)
        n_medium = int(0.50 * num_nodes)
        n_low    = num_nodes - n_high - n_medium

        specs = [
            (n_high,   (24,32),  (96,128),  (150,200), (200,300)),
            (n_medium, (8, 16),  (32, 64),  (50,  80), (60, 120)),
            (n_low,    (2,  8),  (8,  32),  (15,  30), (20,  50)),
        ]
        node_id = 0
        for count, cpu_r, mem_r, idle_r, dyn_r in specs:
            for _ in range(count):
                nodes.append(Node(
                    node_id=node_id,
                    cpu_capacity=float(rng.integers(*cpu_r, endpoint=True)),
                    mem_capacity=float(rng.integers(*mem_r, endpoint=True)),
                    idle_power=float(rng.uniform(*idle_r)),
                    dynamic_power=float(rng.uniform(*dyn_r)),
                ))
                node_id += 1
        return nodes


# ---------------------------------------------------------------------------
# 3. WORKLOAD GENERATOR  (Section III-B / III-C)
# ---------------------------------------------------------------------------

class WorkloadGenerator:
    PRIORITY_DEADLINE_MUL = {0: 1.5, 1: 3.0, 2: 5.0}
    PRIORITY_WEIGHTS      = [0.25, 0.60, 0.15]

    def __init__(self, seed=42):
        self.seed = seed
        self.rng  = np.random.default_rng(seed)

    def generate_tasks(self, num_tasks=1000, arrival_rate=0.5,
                       episode_seed=None, priority_weights=None):
        """Generate one episode of tasks. If `episode_seed` is provided,
        a fresh RNG is used so each episode is independently reproducible
        (preferred for long runs where you may need to debug specific eps).
        Otherwise, the generator uses its persistent RNG.

        `priority_weights`: per-class probabilities [p_prod, p_batch, p_be].
        Default is `WorkloadGenerator.PRIORITY_WEIGHTS`. Override per-episode
        to inject a workload shift for non-stationary experiments."""
        rng = np.random.default_rng(episode_seed) if episode_seed is not None \
              else self.rng
        pw = priority_weights if priority_weights is not None \
             else self.PRIORITY_WEIGHTS
        tasks = []
        current_time = 0.0
        for i in range(num_tasks):
            current_time += rng.exponential(1.0 / arrival_rate)
            # Pareto duration  alpha=1.5, t_min=5s  [Eq.1]
            duration = 5.0 * (1.0 - rng.random()) ** (-1.0 / 1.5)
            # Log-normal cpu/mem  [Eq.2]
            cpu_req  = float(np.clip(rng.lognormal(0.5, 0.8), 0.5, 16.0))
            mem_req  = float(np.clip(rng.lognormal(2.0, 1.0), 1.0, 64.0))
            priority = int(rng.choice([0,1,2], p=pw))
            deadline = current_time + self.PRIORITY_DEADLINE_MUL[priority] * duration
            tasks.append(Task(i, cpu_req, mem_req, duration,
                              current_time, priority, deadline))
        return tasks


# ---------------------------------------------------------------------------
# 4. ACTOR-CRITIC NETWORK  (Section IV-B)
# ---------------------------------------------------------------------------

def _clip_norm(g, max_norm=5.0):
    n = float(np.linalg.norm(g))
    if n > max_norm and n > 0:
        g *= (max_norm / n)
    return g


class ActorCriticNetwork:
    """
    NumPy-only actor-critic.
    ~19 557 params  =>  ~78 KB float32.

    Includes value clamping + gradient norm clipping for stability under the
    multi-component shaped reward (which can produce large transient targets).
    Also includes PPO-style clipped-ratio surrogate (clip_eps) and entropy
    bonus (entropy_coeff) to stabilize on-policy / off-policy training over
    long horizons. Set clip_eps=None / entropy_coeff=0 to recover vanilla PG.
    """

    def __init__(self, input_dim=50, hidden_dim=128, output_dim=100,
                 learning_rate=1e-3, lr_decay=1.0, seed=42,
                 value_clip=1e3, grad_clip=5.0,
                 clip_eps=0.2, entropy_coeff=0.01):
        # We default to no per-step lr decay. With prioritized replay each
        # task completion triggers ~64 sub-update steps, so any small per-step
        # decay (e.g. 0.9995^step) collapses lr below numerical precision
        # within one episode. Per-episode decay can be applied externally by
        # the scheduler if desired.
        rng = np.random.default_rng(seed)
        s1  = np.sqrt(2.0 / input_dim)
        s2  = np.sqrt(2.0 / hidden_dim)
        # Shared hidden layer
        self.W1 = rng.normal(0, s1, (hidden_dim, input_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        # Actor head
        self.W2 = rng.normal(0, s2, (output_dim, hidden_dim)).astype(np.float32)
        self.b2 = np.zeros(output_dim, dtype=np.float32)
        # Critic head
        self.Wv = rng.normal(0, s2, (1, hidden_dim)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)
        self.lr            = learning_rate
        self.lr_decay      = lr_decay
        self.step          = 0
        self.value_clip    = value_clip
        self.grad_clip     = grad_clip
        self.clip_eps      = clip_eps
        self.entropy_coeff = entropy_coeff

    def _hidden(self, obs):
        """h = ReLU(W1*o + b1)  [Eq.6]"""
        return np.maximum(0.0, self.W1 @ obs + self.b1)

    def policy(self, obs):
        """pi(a|o) = softmax(W2*h + b2)  [Eq.7]"""
        h = self._hidden(obs)
        logits = self.W2 @ h + self.b2
        logits -= logits.max()
        e = np.exp(logits)
        s = e.sum()
        if s == 0 or not np.isfinite(s):
            return np.ones_like(e) / len(e)
        return e / s

    def value(self, obs):
        """V(o) = Wv*h + bv  [Eq.8]"""
        v = float((self.Wv @ self._hidden(obs) + self.bv).item())
        return float(np.clip(v, -self.value_clip, self.value_clip))

    def update(self, obs, action, advantage, value_target, pi_old_action=None):
        """Policy gradient (PPO-clipped surrogate, Eq.18 + clip) + entropy
        bonus + value MSE (Eq.19).

        If `pi_old_action` is None or `clip_eps` is None, falls back to
        vanilla policy gradient (advantage * dlog).
        """
        if not (np.isfinite(value_target) and np.isfinite(advantage)):
            return
        self.step += 1
        lr = self.lr * (self.lr_decay ** self.step)
        h  = self._hidden(obs)
        pi = self.policy(obs)
        v  = float(np.clip((self.Wv @ h + self.bv).item(),
                           -self.value_clip, self.value_clip))

        # Critic gradients (gradient of MSE loss L = (V - target)^2).
        # dLv_dWv and dLv_dbv are the gradients of L w.r.t. output-layer
        # parameters; these are subtracted directly (gradient descent on L)
        # in the parameter update below.
        # dLv_dh is the gradient of L w.r.t. the hidden layer; since the
        # combined update for W1/b1 below ASCENDS the combined direction
        # (d_pre * obs etc.), we negate dLv_dh so its contribution to the
        # hidden-layer update reduces L (descent direction). Without this
        # sign flip, the hidden layer would be pushed to maximize critic
        # loss, harming critic representation learning.
        td   = float(np.clip(value_target - v, -self.value_clip, self.value_clip))
        dLv_dbv = np.array([-2*td], dtype=np.float32)
        dLv_dWv = -2*td * h.reshape(1,-1)
        dLv_dh_descent = +2*td * self.Wv.flatten()  # ascent direction for -L

        # Actor gradient: PPO clipped surrogate if pi_old is given, else vanilla.
        # dlog = d log pi(a) / d logits = onehot(a) - pi
        dlog = -pi.copy(); dlog[action] += 1.0

        if pi_old_action is not None and self.clip_eps is not None:
            ratio = float(pi[action]) / max(float(pi_old_action), 1e-9)
            if (advantage >= 0 and ratio > 1.0 + self.clip_eps) or \
               (advantage <  0 and ratio < 1.0 - self.clip_eps):
                surrogate_coeff = 0.0
            else:
                surrogate_coeff = float(advantage) * ratio
        else:
            surrogate_coeff = float(advantage)

        # Entropy bonus on logits: dH/dlogit_l = -pi_l * (log pi_l + H)
        if self.entropy_coeff > 0.0:
            log_pi  = np.log(pi + 1e-12)
            H       = float(-np.sum(pi * log_pi))
            dH_dlog = -pi * (log_pi + H)
        else:
            dH_dlog = 0.0

        dlogits = surrogate_coeff * dlog + self.entropy_coeff * dH_dlog

        dJ_db2 = dlogits
        dJ_dW2 = np.outer(dlogits, h)
        dJ_dh  = self.W2.T @ dlogits

        # Hidden layer: combined ASCENT direction = dJ/dh (actor objective)
        # + (-dL/dh) (critic descent expressed as ascent of -L).
        d_pre = (dLv_dh_descent + dJ_dh) * (h > 0).astype(np.float32)
        dW1 = np.outer(d_pre, obs)
        db1 = d_pre

        for g in (dJ_dW2, dJ_db2, dLv_dWv, dLv_dbv, dW1, db1):
            _clip_norm(g, self.grad_clip)

        self.W2 += lr * dJ_dW2;   self.b2 += lr * dJ_db2
        self.Wv -= lr * dLv_dWv;  self.bv -= lr * dLv_dbv
        self.W1 += lr * dW1;      self.b1 += lr * db1


# ---------------------------------------------------------------------------
# 5. PRIORITIZED EXPERIENCE REPLAY  (Section IV-F)
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """pi = |delta_i| + epsilon  [Eq.17]"""

    def __init__(self, capacity=10_000, epsilon=1e-6):
        self.capacity   = capacity
        self.epsilon    = epsilon
        self.buffer     = []
        self.priorities = []
        self.pos        = 0

    def push(self, obs, action, reward, next_obs, td_error, pi_old_action=None):
        entry    = (obs, action, reward, next_obs, pi_old_action)
        priority = abs(td_error) + self.epsilon
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos]     = entry
            self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=64):
        probs = np.array(self.priorities, dtype=np.float64)
        probs /= probs.sum()
        idxs  = np.random.choice(len(self.buffer),
                                 size=min(batch_size, len(self.buffer)),
                                 p=probs, replace=False)
        return [self.buffer[i] for i in idxs]

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# 6. ADC-GNC GOSSIP CONSENSUS  (Section IV-C)
# ---------------------------------------------------------------------------

class GossipConsensus:
    """
    Randomised pairwise gossip — scalar variant.

    Each agent maintains a single scalar z_i, an estimate of mean network
    utilization. This is the formulation in Eq. 9-11 of the paper.

    Communication: O(1) per step per agent.
    Convergence of the averaging component: O(log N) under standard
    random-neighbour assumptions [35,36].

    NOTE: This scalar form is retained for backward compatibility. The
    primary algorithm in Section IV-D (assignment_score / select_best_node)
    requires per-node utilization estimates, which the scalar form cannot
    provide. The PerAgentGossip class below extends the mechanism to
    per-node estimates while preserving O(1) per-step communication.
    """

    def __init__(self, num_nodes, gossip_prob=0.3, avg_weight=0.5,
                 refresh_rate=0.1, seed=42):
        self.N   = num_nodes
        self.p_g = gossip_prob
        self.w   = avg_weight          # must be in (0, 0.5]
        self.eta = refresh_rate
        self.rng = np.random.default_rng(seed)
        self.z   = np.zeros(num_nodes, dtype=np.float32)

    def initialize(self, utilizations):
        """z_i(0) = u_i(0)"""
        self.z = utilizations.copy().astype(np.float32)

    def step(self, utilizations):
        for i in range(self.N):
            if self.rng.random() < self.p_g:
                # Pairwise exchange  [Eq.9, Eq.10]
                j = int(self.rng.integers(0, self.N))
                while j == i:
                    j = int(self.rng.integers(0, self.N))
                zi_new = (1-self.w)*self.z[i] + self.w*self.z[j]
                zj_new = (1-self.w)*self.z[j] + self.w*self.z[i]
                self.z[i] = zi_new
                self.z[j] = zj_new
            else:
                # Self-refresh  [Eq.11]
                self.z[i] = self.eta*utilizations[i] + (1-self.eta)*self.z[i]

    def get_estimate(self, node_id):
        return float(self.z[node_id])


# ---------------------------------------------------------------------------
# 6'. PER-AGENT GOSSIP (Section IV-C extension for placement decisions)
# ---------------------------------------------------------------------------

class PerAgentGossip:
    """Per-agent gossip-based decentralized state estimation.

    Each agent i maintains a vector estimates[i] in R^N of estimates of all
    nodes' CPU utilization. estimates[i][i] is always agent i's true local
    utilization (it knows itself). estimates[i][j] for j != i is updated
    only via gossip exchanges with neighbors.

    Per-step communication: O(1) gossip events per agent (with probability
    p_g per agent per step, a single random neighbor j is contacted). Each
    event exchanges the agent's full N-dimensional estimate vector
    (vector-payload exchange, the standard Boyd et al. 2006 randomized
    gossip-averaging protocol). Per-event payload is therefore O(N) floats.
    For N=100 nodes this is ~400 bytes per gossip event — modest in
    absolute terms and far below full broadcast or centralized state
    queries. Convergence to the true mean utilization vector occurs in
    O(log N) gossip rounds.

    Without communication (NGC ablation, set enable_communication=False in
    step()), estimates of non-local nodes stay at the uninformed default
    (default_estimate, typically 0.5). This forces placement decisions to
    rely only on the agent's local view, exposing the coordination value
    of gossip.

    Memory: O(N) per agent, O(N^2) total system. For N=100, this is
    400 bytes per agent and 40 KB total — comparable to the per-agent
    network footprint.

    Note: the `refresh_rate` parameter is accepted for backward
    compatibility with the scalar GossipConsensus class but is unused in
    PerAgentGossip — the diagonal entries are refreshed exactly each step
    (the agent always knows its own state directly).
    """

    def __init__(self, num_nodes, gossip_prob=0.3, avg_weight=0.5,
                 refresh_rate=0.1, default_estimate=0.5, seed=42):
        self.N = num_nodes
        self.p_g = gossip_prob
        self.w = avg_weight
        self.eta = refresh_rate  # unused; kept for API parity with GossipConsensus
        self.default_estimate = float(default_estimate)
        self.rng = np.random.default_rng(seed)
        # estimates[i][j] = agent i's estimate of node j's utilization
        self.estimates = np.full((num_nodes, num_nodes),
                                 self.default_estimate, dtype=np.float32)

    def initialize(self, utilizations):
        """Each agent knows its own initial utilization; non-local stays at
        default."""
        for i in range(self.N):
            self.estimates[i][i] = float(utilizations[i])

    def step(self, utilizations, enable_communication=True):
        """Update each agent's self-estimate; optionally do pairwise gossip.

        If enable_communication is False, only self-estimates are refreshed
        (NGC ablation). All non-local estimates remain at default.

        On each gossip event between agents i and j, they exchange and
        average their full estimate vectors. This is the standard gossip
        averaging protocol [Boyd et al. 2006] which converges in O(log N)
        rounds, vs. O(N log N) for single-entry exchange. Per-step
        communication is O(1) gossip events per agent, with each event
        exchanging the agent's current N-dimensional view (a fixed-size
        vector of utilization estimates).
        """
        # Self-refresh: each agent's diagonal entry tracks its own true u_i.
        # Use full update (eta=1 effectively for diagonal) since the agent
        # always knows its own state directly.
        for i in range(self.N):
            self.estimates[i][i] = float(utilizations[i])

        if not enable_communication:
            return

        # Pairwise gossip: vector exchange and averaging.
        for i in range(self.N):
            if self.rng.random() < self.p_g:
                j = int(self.rng.integers(0, self.N))
                while j == i:
                    j = int(self.rng.integers(0, self.N))
                # Symmetric averaging on the entire estimate vector.
                # This is a single gossip event per agent; the message
                # carries the agent's current state estimates.
                avg_i = (1.0 - self.w) * self.estimates[i] + self.w * self.estimates[j]
                avg_j = (1.0 - self.w) * self.estimates[j] + self.w * self.estimates[i]
                self.estimates[i] = avg_i
                self.estimates[j] = avg_j
                # Re-pin diagonals: each agent always knows its own state
                self.estimates[i][i] = float(utilizations[i])
                self.estimates[j][j] = float(utilizations[j])

    def estimate_at(self, agent_id, node_id):
        """Agent `agent_id`'s estimate of node `node_id`'s utilization."""
        return float(self.estimates[agent_id][node_id])

    def get_view(self, agent_id):
        """Agent's full estimate vector (read-only copy)."""
        return self.estimates[agent_id].copy()

    def get_mean_estimate(self, agent_id):
        """Agent's estimate of mean cluster utilization (z_i in original
        paper formulation)."""
        return float(self.estimates[agent_id].mean())


# ---------------------------------------------------------------------------
# 7. REWARD COMPONENTS AND ADAPTIVE REWARD SHAPING  (Section IV-E)
# ---------------------------------------------------------------------------

DEFAULT_REWARD_WEIGHTS = {
    "w_sla":     0.40,
    "w_compl":   0.30,
    "w_energy":  0.20,
    "w_balance": 0.10,
}


def reward_components(task, finish_time, energy_step, utilizations):
    """Compute the four reward components from Section IV-E (paragraph after Eq.16).

    These are independent of the weighting scheme so PPO / MADDPG / MAPPO baselines
    can reuse them with fixed weights ('same reward structure as DRL-MADRL,
    minus gossip and adaptive components').

    Note on r_energy units: r_energy = -0.3 * E_t with E_t expressed in
    kilojoules (i.e. energy_step divided by 1000). The simulator's
    `energy_step` is computed in raw Joules (power[W] * time_step[s]),
    which under default parameters can reach ~2500 J -- if used directly
    in r_energy this would swamp r_sla/r_compl by an order of magnitude.
    Scaling to kJ keeps the four reward components on comparable scales
    (Eq. 16 components fall in roughly [-100, 100]).
    """
    p = task.priority
    violated  = finish_time > task.deadline
    r_sla     = (-20.0 if violated else 15.0) * (4 - p)
    r_compl   = max(0.0, 100.0 - 0.5*(finish_time - task.arrival_time))
    r_energy  = -0.3 * (energy_step / 1000.0)        # E_t in kJ; see docstring
    r_balance = -200.0 * float(np.var(utilizations))
    return r_sla, r_compl, r_energy, r_balance, violated


def weighted_reward(weights, r_sla, r_compl, r_energy, r_balance):
    return (weights["w_sla"]    * r_sla
          + weights["w_compl"]  * r_compl
          + weights["w_energy"] * r_energy
          + weights["w_balance"]* r_balance)


class AdaptiveRewardShaper:
    """
    Online weight adaptation [Eq.15], full reward [Eq.16].
    When `adaptive=False`, weights are frozen at initialization (used for the
    NRS ablation and as a fixed-weight reward source for non-DRL-MADRL baselines).
    """

    def __init__(self, w_sla=0.40, w_compl=0.30, w_energy=0.20, w_balance=0.10,
                 target_viol=0.15, kappa=0.05, history_len=100, adaptive=True):
        self.weights = {
            "w_sla":     w_sla,
            "w_compl":   w_compl,
            "w_energy":  w_energy,
            "w_balance": w_balance,
        }
        self.rho_star = target_viol
        self.kappa    = kappa
        self._hist    = []
        self._hlen    = history_len
        self.adaptive = adaptive

    def _record(self, violated):
        self._hist.append(int(violated))
        if len(self._hist) > self._hlen:
            self._hist.pop(0)

    def adapt_weights(self, mean_util):
        """Update w_SLA [Eq.15] and w_energy, then renormalize. No-op if frozen."""
        if not self.adaptive:
            return
        if len(self._hist) >= 10:
            rho_obs = float(np.mean(self._hist))
            self.weights["w_sla"] = float(np.clip(
                self.weights["w_sla"] + self.kappa*(rho_obs - self.rho_star),
                0.05, 0.80))
        if mean_util > 0.75:
            self.weights["w_energy"] = min(self.weights["w_energy"] + 0.01, 0.40)
        elif mean_util < 0.40:
            self.weights["w_energy"] = max(self.weights["w_energy"] - 0.01, 0.05)
        total = sum(self.weights.values())
        for k in self.weights:
            self.weights[k] /= total

    def compute_reward(self, task, finish_time, energy_step, utilizations):
        """r_t = w_SLA*r_SLA + w_compl*r_compl + w_e*r_energy + w_b*r_balance  [Eq.16]"""
        r_sla, r_compl, r_energy, r_balance, violated = reward_components(
            task, finish_time, energy_step, utilizations)
        self._record(violated)
        return weighted_reward(self.weights, r_sla, r_compl, r_energy, r_balance)

    # Back-compat properties
    @property
    def w_sla(self):     return self.weights["w_sla"]
    @property
    def w_compl(self):   return self.weights["w_compl"]
    @property
    def w_energy(self):  return self.weights["w_energy"]
    @property
    def w_balance(self): return self.weights["w_balance"]


# ---------------------------------------------------------------------------
# 8. OBSERVATION BUILDER  (Eq.12)
# ---------------------------------------------------------------------------

def task_features(task, current_time):
    """Four-dimensional task descriptor used to extend Eq. 12's observation.

    Allows the policy to condition placement on what task is being scheduled.
    Returns zeros if `task` is None (used for next_obs after completion when
    no task is pending from this agent).
    """
    if task is None:
        return np.zeros(4, dtype=np.float32)
    span  = max(task.deadline - task.arrival_time, 1e-9)
    slack = (task.deadline - current_time) / span
    return np.array([
        min(task.cpu_req / 16.0, 1.0),
        min(task.mem_req / 64.0, 1.0),
        task.priority / 2.0,
        float(np.clip(slack, 0.0, 1.0)),
    ], dtype=np.float32)


def build_observation(node, all_nodes, gossip, task=None, current_time=0.0,
                      obs_dim=50):
    """o_i = [local_state, gossip_view, neighbor_estimates, aggregates, task].

    Eq. 12 of the paper, extended (Section IV-A) to include task descriptor
    features at indices 30..33 so the policy can condition placement on the
    current task's CPU, memory, priority, and deadline-slack ratio.

    The agent perspective: `node` is the agent's own node. The agent
    directly knows its own state (lines 0-3, 25-28) and memory/queue
    information (which we treat as locally observable). For cluster-wide
    information about other nodes (lines 4, 5-14, 15-24, 29), the agent
    relies on its `gossip` view if `gossip` is a PerAgentGossip; otherwise
    it falls back to the centralized scalar estimator (legacy behavior).
    """
    obs = np.zeros(obs_dim, dtype=np.float32)
    N   = len(all_nodes)
    aid = node.node_id

    # 0-3: agent's own local state — directly observable
    obs[0] = node.cpu_utilization
    obs[1] = node.mem_utilization
    obs[2] = min(node.queue_length / 20.0, 1.0)
    obs[3] = node.cpu_capacity / 32.0

    # 4: agent's estimate of mean cluster utilization
    if isinstance(gossip, PerAgentGossip):
        view = gossip.get_view(aid)
        obs[4] = float(view.mean())
    else:
        obs[4] = gossip.get_estimate(aid)
        view = None

    # 5-14: neighbor utilization estimates (from gossip view, not direct read)
    for k in range(10):
        nb_id = (aid + k + 1) % N
        if view is not None:
            obs[5+k] = float(view[nb_id])
        else:
            obs[5+k] = all_nodes[nb_id].cpu_utilization

    # 15-24: cluster-wide aggregate statistics from agent's gossip view
    if view is not None:
        utils_view = view
    else:
        utils_view = np.array([n.cpu_utilization for n in all_nodes],
                              dtype=np.float32)
    # Memory aggregates: keep direct reads (memory tracked locally per node;
    # gossip extension to memory is straightforward future work)
    mem_utils = np.array([n.mem_utilization for n in all_nodes],
                         dtype=np.float32)
    obs[15] = float(utils_view.mean())
    obs[16] = float(utils_view.std())
    obs[17] = float(utils_view.min())
    obs[18] = float(utils_view.max())
    obs[19] = float(mem_utils.mean())
    obs[20] = float(mem_utils.std())
    obs[21] = min(sum(n.queue_length for n in all_nodes) / 200.0, 1.0)
    obs[22] = float(np.var(utils_view))
    obs[23] = float(np.percentile(utils_view, 75))
    obs[24] = float(np.percentile(utils_view, 25))

    # 25-28: agent's own derived state — directly observable
    obs[25] = node.instantaneous_power() / max(node.idle_power + node.dynamic_power, 1e-9)
    obs[26] = len(node.running_tasks) / max(node.cpu_capacity, 1.0)
    obs[27] = node.cpu_used / max(node.cpu_capacity, 1e-9)
    obs[28] = node.mem_used / max(node.mem_capacity, 1e-9)

    # 29: deviation of agent's self-estimate from its estimated cluster mean
    if view is not None:
        obs[29] = float(view[aid] - view.mean())
    else:
        obs[29] = gossip.get_estimate(aid) - utils_view.mean()

    # 30-33: task-conditioned features
    obs[30:34] = task_features(task, current_time)
    return obs


# ---------------------------------------------------------------------------
# 9. DRL-MADRL AGENT
# ---------------------------------------------------------------------------

class DRLMADRLAgent:
    def __init__(self, node_id, num_nodes, obs_dim=50, hidden_dim=128,
                 lr=1e-3, buffer_capacity=10_000, batch_size=64,
                 gamma=0.95, seed=42, use_replay=True):
        self.node_id    = node_id
        self.num_nodes  = num_nodes
        self.batch_size = batch_size
        self.gamma      = gamma
        self.use_replay = use_replay
        self.network    = ActorCriticNetwork(obs_dim, hidden_dim, num_nodes,
                                             lr, seed=seed+node_id)
        self.buffer     = PrioritizedReplayBuffer(buffer_capacity) if use_replay else None

    def select_action(self, obs):
        pi     = self.network.policy(obs)
        action = int(np.random.choice(self.num_nodes, p=pi))
        return action, pi

    def store(self, obs, action, reward, next_obs, pi_old_action=None):
        v      = self.network.value(obs)
        v_next = self.network.value(next_obs)
        td_err = reward + self.gamma * v_next - v
        if self.use_replay:
            self.buffer.push(obs, action, reward, next_obs, td_err,
                             pi_old_action=pi_old_action)
        else:
            target = reward + self.gamma * v_next
            adv    = target - v
            self.network.update(obs, action, adv, target,
                                pi_old_action=pi_old_action)

    def learn(self):
        if not self.use_replay:
            return
        if len(self.buffer) < self.batch_size:
            return
        for entry in self.buffer.sample(self.batch_size):
            obs, action, reward, next_obs, pi_old_action = entry
            v_next  = self.network.value(next_obs)
            target  = reward + self.gamma * v_next
            adv     = target - self.network.value(obs)
            self.network.update(obs, action, adv, target,
                                pi_old_action=pi_old_action)


# ---------------------------------------------------------------------------
# 10. PRIORITY-AWARE ACTION SELECTION  (Section IV-D)
# ---------------------------------------------------------------------------

def priority_score(task, current_time):
    """score_j = 0.4*(3-p) + 0.3*slack + 0.3*urgency  [Eq.13]"""
    span        = max(task.deadline - task.arrival_time, 1e-9)
    slack_ratio = (task.deadline - current_time) / span
    urgency     = 1.0 / (1.0 + max(task.deadline - current_time, 0.0))
    return 0.4*(3 - task.priority) + 0.3*slack_ratio + 0.3*urgency


# Default Eq. 14 weights (paper). Can be overridden globally via
# `set_assignment_weights(...)` for ablation / experimental sweeps.
_ASSIGNMENT_WEIGHTS = {"pi": 0.25, "u": 0.30, "m": 0.20, "c": 0.15, "p": 0.10}


def set_assignment_weights(pi=0.25, u=0.30, m=0.20, c=0.15, p=0.10):
    """Override global Eq. 14 weights. Used by experiments that test how
    much performance depends on the heuristic vs the learned policy.
    Weights need not sum to 1 (no renormalization)."""
    global _ASSIGNMENT_WEIGHTS
    _ASSIGNMENT_WEIGHTS = {"pi": pi, "u": u, "m": m, "c": c, "p": p}


def get_assignment_weights():
    return dict(_ASSIGNMENT_WEIGHTS)


def assignment_score(pi, node, task, pscore, c_max=1.0,
                     ingress_id=None, gossip=None):
    """S_ij = w_pi*pi + w_u*(1-u_est) + w_m*(1-m/M) + w_c*c + w_p*p  [Eq.14]

    When `ingress_id` and `gossip` are provided, the load term (1-u) uses
    the ingress agent's gossip-derived estimate of node j's utilization
    rather than node j's true cpu_utilization. This makes the placement
    decision properly decentralized — the ingress agent only knows its own
    state directly and relies on gossip for non-local information.

    Without `gossip` (backward-compatible path), the score reads
    node.cpu_utilization directly, which is the centralized-state form
    used in the original paper (Eq. 14 as written in [Author 2026]).

    Memory term (1 - m_i/M_i) currently always uses true mem_utilization;
    extending gossip to memory is a straightforward extension noted as
    future work.
    """
    w = _ASSIGNMENT_WEIGHTS
    cap_norm = node.cpu_capacity / max(c_max, 1e-9)
    c_ij = cap_norm * (1.0 - abs(task.cpu_req / max(node.cpu_capacity, 1e-9) - 0.5))

    if gossip is not None and ingress_id is not None:
        # Decentralized form: use the ingress agent's gossip estimate of
        # node j's utilization.  For the agent's own node this returns
        # the local truth (estimates[i][i] == u_i).
        u_est = gossip.estimate_at(ingress_id, node.node_id)
    else:
        # Backward-compat path: read true utilization (centralized state).
        u_est = node.cpu_utilization

    return (w["pi"] * float(pi[node.node_id])
          + w["u"]  * (1.0 - u_est)
          + w["m"]  * (1.0 - node.mem_utilization)
          + w["c"]  * c_ij
          + w["p"]  * pscore)


def select_best_node(task, nodes, pi, current_time,
                     ingress_id=None, gossip=None):
    """Compute Eq. 14 score over feasible nodes and pick the highest.

    When `ingress_id` and `gossip` are provided, the scoring is
    decentralized: the ingress agent uses its gossip estimates of non-local
    nodes' utilization. Without them, the score reads true cpu_utilization
    (centralized state, used for backward compatibility with single-agent
    baselines).
    """
    pscore = priority_score(task, current_time)
    c_max = max((n.cpu_capacity for n in nodes), default=1.0)
    best_node, best_score = None, -np.inf
    for node in nodes:
        if node.can_accept(task):
            s = assignment_score(pi, node, task, pscore, c_max=c_max,
                                 ingress_id=ingress_id, gossip=gossip)
            if s > best_score:
                best_score = s
                best_node  = node
    return best_node


# ---------------------------------------------------------------------------
# 11. FULL DRL-MADRL SCHEDULER
# ---------------------------------------------------------------------------

class DRLMADRLScheduler:
    """
    One agent per node; decentralized at execution time —
    no centralized controller or centralized critic during deployment.

    The placement decision (Eq. 14, select_best_node) uses each agent's
    PerAgentGossip estimates of non-local nodes' utilization, NOT direct
    queries. This makes the scheduler genuinely decentralized: without
    gossip exchanges, an agent has no information about other nodes.

    Ablation flags (Section V-E):
      use_gossip            (NGC: inter-agent gossip exchange disabled;
                             agents only know their own local utilization)
      use_adaptive_reward   (NRS: reward weights frozen at initial values)
      use_replay            (NER: prioritized replay disabled)
      use_priority_scoring  (NPS: priority + assignment score replaced by
                             policy argmax over feasible nodes)
    """

    def __init__(self, nodes, obs_dim=50, hidden_dim=128, lr=1e-3,
                 gamma=0.95, gossip_prob=0.3, avg_weight=0.5,
                 refresh_rate=0.1, buffer_capacity=10_000,
                 batch_size=64, seed=42,
                 use_gossip=True, use_adaptive_reward=True,
                 use_replay=True, use_priority_scoring=True,
                 default_gossip_estimate=0.5,
                 contention_adaptive=False,
                 adaptive_threshold=0.05, adaptive_width=0.15):
        self.nodes = nodes
        N = len(nodes)
        self.use_gossip           = use_gossip
        self.use_adaptive_reward  = use_adaptive_reward
        self.use_replay           = use_replay
        self.use_priority_scoring = use_priority_scoring
        self.contention_adaptive  = contention_adaptive
        self.adaptive_threshold   = adaptive_threshold
        self.adaptive_width       = adaptive_width

        # Per-agent gossip: each agent maintains its own N-vector of estimates.
        # Without communication, non-local estimates stay at default_gossip_estimate.
        self.gossip = PerAgentGossip(N, gossip_prob, avg_weight, refresh_rate,
                                     default_estimate=default_gossip_estimate,
                                     seed=seed)
        self.reward_shaper = AdaptiveRewardShaper(adaptive=use_adaptive_reward)
        self.agents = [
            DRLMADRLAgent(i, N, obs_dim, hidden_dim, lr,
                          buffer_capacity, batch_size, gamma, seed,
                          use_replay=use_replay)
            for i in range(N)
        ]
        utils = np.array([n.cpu_utilization for n in nodes], dtype=np.float32)
        self.gossip.initialize(utils)
        # task_id -> (ingress_agent_id, ingress_obs_at_decision_time, pi_old)
        self._pending = {}

    def schedule_task(self, task, current_time):
        feasible = [n for n in self.nodes if n.can_accept(task)]
        if not feasible:
            return None
        # Section IV-A: the ingress agent makes the decision.
        ingress = min(feasible, key=lambda n: n.cpu_utilization)
        ingress_obs = build_observation(ingress, self.nodes, self.gossip,
                                        task=task, current_time=current_time)
        _, pi = self.agents[ingress.node_id].select_action(ingress_obs)

        if self.use_priority_scoring:
            if self.contention_adaptive:
                # Blend Eq. 14 (heuristic-driven) with policy argmax
                # (NPS-like behavior) based on observed contention level.
                # Contention signal: fraction of cluster nodes that cannot
                # currently accept the incoming task. When contention is
                # low, most nodes are feasible and the heuristic's
                # capacity-priority ranking picks the right one. When
                # contention is high, few nodes are feasible and the
                # heuristic's bias toward high-tier nodes over-concentrates
                # placements; deferring to the learned policy spreads
                # decisions better. This signal requires only knowledge
                # of own local state per agent in deployment (each agent
                # checks its own can_accept and shares feasibility via
                # gossip; here we compute it directly for the simulator).
                feasible_count = sum(1 for n in self.nodes
                                     if n.can_accept(task))
                contention = 1.0 - (feasible_count / max(len(self.nodes), 1))
                alpha = float(np.clip(
                    (contention - self.adaptive_threshold) / self.adaptive_width,
                    0.0, 1.0
                ))
                chosen = self._adaptive_blend_select(
                    task, pi, current_time, ingress.node_id, alpha
                )
            else:
                # Pass ingress_id and gossip so Eq. 14 uses gossip estimates
                # for non-local node utilizations.
                chosen = select_best_node(task, self.nodes, pi, current_time,
                                          ingress_id=ingress.node_id,
                                          gossip=self.gossip)
        else:
            mask = np.zeros(len(self.nodes), dtype=np.float32)
            for n in feasible:
                mask[n.node_id] = 1.0
            masked = pi * mask
            chosen = (self.nodes[int(np.argmax(masked))]
                      if masked.sum() >= 1e-9 else feasible[0])

        if chosen is not None:
            pi_old_action = float(pi[chosen.node_id])
            self._pending[task.task_id] = (ingress.node_id,
                                           ingress_obs.copy(),
                                           pi_old_action)
        return chosen

    def on_task_complete(self, task, node, current_time, energy_step):
        utils  = np.array([n.cpu_utilization for n in self.nodes], dtype=np.float32)
        reward = self.reward_shaper.compute_reward(task, current_time, energy_step, utils)

        decision = self._pending.pop(task.task_id, None)
        if decision is not None:
            ingress_id, ingress_obs, pi_old_action = decision
            agent     = self.agents[ingress_id]
            next_obs  = build_observation(self.nodes[ingress_id],
                                          self.nodes, self.gossip,
                                          task=None,
                                          current_time=current_time)
            agent.store(ingress_obs, node.node_id, reward, next_obs,
                        pi_old_action=pi_old_action)
            agent.learn()

        # Adaptive shaper uses gossip estimate of mean cluster utilization
        # (consistent with decentralized-execution framing). Under NGC,
        # gossip exchange is disabled and only self-estimates are accurate;
        # the agent's mean-estimate becomes biased toward its own load.
        # We average across all agents' mean estimates as a system-wide proxy.
        mean_util_estimate = float(self.gossip.estimates.mean())
        self.reward_shaper.adapt_weights(mean_util_estimate)

    def gossip_step(self):
        # PerAgentGossip handles enable/disable internally via the
        # enable_communication flag. Under NGC, only self-refresh occurs;
        # estimates of non-local nodes stay at the uninformed default.
        utils = np.array([n.cpu_utilization for n in self.nodes],
                         dtype=np.float32)
        self.gossip.step(utils, enable_communication=self.use_gossip)

    def _adaptive_blend_select(self, task, pi, current_time, ingress_id, alpha):
        """Contention-adaptive placement: blend Eq. 14 score with policy score.

        For each feasible candidate node j, compute:
          combined_score(j) = (1 - alpha) * eq14_score(j) + alpha * pi[j]_norm

        where eq14_score follows Section IV-D (using gossip estimates for
        non-local nodes) and pi[j]_norm is the policy probability normalized
        over feasible nodes. Pick argmax.

        At alpha=0, this reduces to standard Eq. 14 (full DRL-MADRL).
        At alpha=1, this reduces to policy argmax (NPS-equivalent).
        For intermediate alpha, the heuristic and policy contributions are
        smoothly blended. The caller (schedule_task) sets alpha as a
        clipped linear function of an observed contention signal, defined
        as the fraction of cluster nodes currently infeasible for the
        incoming task: alpha = clip((c - tau) / w, 0, 1) with c = (1 -
        feasible_count/N). Under low contention (most nodes feasible,
        c ~ 0), alpha ~ 0 and the heuristic dominates. Under high
        contention (few feasible nodes, c -> 1), alpha -> 1 and the
        policy carries the decision.

        This addresses the empirical finding (Section V-E) that priority-
        aware action selection (Eq. 13-14) is beneficial under low
        contention but constraining under high contention; the contention-
        adaptive blend captures the best of both regimes.
        """
        feasible = [n for n in self.nodes if n.can_accept(task)]
        if not feasible:
            return None
        # Normalize policy over feasible only, so pi values are comparable to
        # heuristic scores in [0, ~1].
        mask = np.zeros(len(self.nodes), dtype=np.float32)
        for n in feasible:
            mask[n.node_id] = 1.0
        pi_masked = pi * mask
        if pi_masked.sum() >= 1e-9:
            pi_norm = pi_masked / pi_masked.sum()
        else:
            pi_norm = mask / max(mask.sum(), 1.0)

        pscore = priority_score(task, current_time)
        c_max = max((n.cpu_capacity for n in self.nodes), default=1.0)

        best_node, best_score = None, -np.inf
        for node in feasible:
            heur_score = assignment_score(pi, node, task, pscore, c_max=c_max,
                                          ingress_id=ingress_id,
                                          gossip=self.gossip)
            policy_score = float(pi_norm[node.node_id])
            combined = (1.0 - alpha) * heur_score + alpha * policy_score
            if combined > best_score:
                best_score = combined
                best_node = node
        return best_node
