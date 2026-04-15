#!/usr/bin/env python3
"""
defender2.py
Deep Q-Network adaptive defender.
Linear function approximation DQN with experience replay,
soft target network updates, EMA anomaly detection,
per-source reputation scoring, and dynamic nftables rule generation.
"""

import subprocess
import time
import json
import os
import re
import sys
import threading
import collections
import numpy as np
from datetime import datetime

# ── Constants ────────────────────────────────────────────────────────────────
STATE_FILE   = "/home/kaexb/sim/defender_state.json"
LOG_FILE     = "/home/kaexb/sim/defender.log"
SHARED_FILE  = "/home/kaexb/sim/shared_state.json"
TRAFFIC_LOG  = "/home/kaexb/sim/traffic.log"
HISTORY_FILE = "/home/kaexb/sim/defender_history.jsonl"

NETNS  = "right"
IFACE  = "gre1"
SRC_IP = "192.168.100.1"  # attacker IP

# ── DQN Hyperparameters ──────────────────────────────────────────────────────
LR           = 0.08    # learning rate for weight updates
GAMMA        = 0.93    # discount factor
EPSILON_INIT = 0.35    # initial exploration
EPSILON_MIN  = 0.03    # minimum exploration
EPSILON_DECAY= 0.9985  # per-step decay
BATCH_SIZE   = 64      # experience replay batch
REPLAY_MAX   = 2000    # max replay buffer size
TARGET_UPDATE = 25     # soft update target network every N steps

# ── State Feature Definitions ────────────────────────────────────────────────
# 12 normalized features extracted from traffic window
# [0] conn_rate_norm       - connection rate normalized to baseline*20
# [1] port_diversity       - unique ports / total events
# [2] icmp_ratio           - ICMP events / total
# [3] udp_ratio            - UDP events / total
# [4] syn_ratio            - SYN events / total
# [5] frag_ratio           - IP fragment events / total
# [6] large_pkt_ratio      - packets >500 bytes / total
# [7] reputation           - source reputation score 0-1
# [8] alert_level_norm     - current alert level / 3
# [9] pps_norm             - packets/sec normalized
# [10] bytes_norm          - bytes/sec normalized
# [11] consecutive_attacks  - consecutive high-rate windows / 10

STATE_DIM = 12

# ── Actions ──────────────────────────────────────────────────────────────────
# 9 actions ranging from passive monitoring to aggressive blocking
ACTIONS = [
    "monitor",           # 0: watch only, no rules
    "log_alert",         # 1: log to file + alert but no block
    "rate_limit_soft",   # 2: 50 conn/s - gentle rate limiting
    "rate_limit_medium", # 3: 20 conn/s - moderate rate limiting
    "rate_limit_hard",   # 4: 5 conn/s - aggressive rate limiting
    "block_icmp",        # 5: block ICMP only
    "block_udp_flood",   # 6: block high-rate UDP
    "block_temp",        # 7: full block for 30s
    "block_long",        # 8: full block for 120s
]
N_ACTIONS = len(ACTIONS)

# Action aggressiveness level (used for false positive estimation)
ACTION_AGGRESSION = {
    "monitor":           0.0,
    "log_alert":         0.1,
    "rate_limit_soft":   0.2,
    "rate_limit_medium": 0.35,
    "rate_limit_hard":   0.55,
    "block_icmp":        0.4,
    "block_udp_flood":   0.45,
    "block_temp":        0.8,
    "block_long":        0.95,
}

# ── Alert Level Thresholds ───────────────────────────────────────────────────
ALERT_THRESHOLDS = [0.25, 0.50, 0.75]  # ELEVATED, HIGH, CRITICAL

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}][{level:7s}] DEF: {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def run_cmd(cmd, timeout: int = 5) -> tuple:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          timeout=timeout,
                          shell=isinstance(cmd, str))
        return r.returncode, r.stdout + r.stderr
    except Exception as e:
        return -1, str(e)


def run_in_ns(cmd: list, timeout: int = 5) -> tuple:
    return run_cmd(["sudo", "ip", "netns", "exec", NETNS] + cmd, timeout)


# ═══════════════════════════════════════════════════════════════════════════════
# DEEP Q-NETWORK
# Linear function approximation: Q(s,a) = s · W[:,a]
# ═══════════════════════════════════════════════════════════════════════════════

class DQNetwork:
    """
    Linear DQN using numpy.
    Architecture: Q(s,a) = s_vec @ W[:,a] + b[a]
    Two networks: online (trained every step) and target (soft-updated periodically).
    Experience replay buffer for off-policy learning.
    """

    def __init__(self, state_dim: int = STATE_DIM, n_actions: int = N_ACTIONS):
        self.state_dim = state_dim
        self.n_actions = n_actions

        # Initialize weights with small random values
        # Use He initialization scaled for our state size
        scale = np.sqrt(2.0 / state_dim)
        self.W = np.random.randn(state_dim, n_actions) * scale * 0.1
        self.b = np.zeros(n_actions)

        # Target network (copy of online network, updated slowly)
        self.W_target = self.W.copy()
        self.b_target = self.b.copy()

        # Replay buffer: each entry is (state, action, reward, next_state, done)
        self.replay_buffer = collections.deque(maxlen=REPLAY_MAX)

        # Training metadata
        self.epsilon = EPSILON_INIT
        self.update_count = 0
        self.loss_history = collections.deque(maxlen=500)
        self.reward_history = collections.deque(maxlen=500)

        # Per-action statistics
        self.action_counts = np.zeros(n_actions, dtype=int)
        self.action_rewards = np.zeros(n_actions)

    def q_values(self, state_vec: np.ndarray,
                 use_target: bool = False) -> np.ndarray:
        """Compute Q-values for all actions given state vector."""
        if use_target:
            return state_vec @ self.W_target + self.b_target
        return state_vec @ self.W + self.b

    def select_action(self, state_vec: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_values(state_vec)))

    def store_transition(self, state: np.ndarray, action: int,
                         reward: float, next_state: np.ndarray,
                         done: bool):
        """Add experience to replay buffer."""
        self.replay_buffer.append((
            state.copy(), action, reward, next_state.copy(), done
        ))
        self.action_counts[action] += 1
        self.action_rewards[action] += reward
        self.reward_history.append(reward)

    def train_step(self) -> float:
        """
        Sample a minibatch from replay buffer and perform one gradient step.
        Uses TD target: y = r + gamma * max_a Q_target(s', a)
        Gradient: dL/dW = -(y - Q(s,a)) * s
        """
        if len(self.replay_buffer) < BATCH_SIZE:
            return 0.0

        # Sample random minibatch
        indices = np.random.choice(len(self.replay_buffer),
                                   BATCH_SIZE, replace=False)
        batch = [self.replay_buffer[i] for i in indices]

        states      = np.array([b[0] for b in batch], dtype=np.float32)
        actions     = np.array([b[1] for b in batch], dtype=int)
        rewards     = np.array([b[2] for b in batch], dtype=np.float32)
        next_states = np.array([b[3] for b in batch], dtype=np.float32)
        dones       = np.array([b[4] for b in batch], dtype=np.float32)

        # Compute TD targets using target network
        q_next = next_states @ self.W_target + self.b_target  # (batch, n_actions)
        max_q_next = np.max(q_next, axis=1)                    # (batch,)
        targets = rewards + GAMMA * max_q_next * (1.0 - dones) # (batch,)

        # Compute current Q-values
        q_current = states @ self.W + self.b  # (batch, n_actions)
        q_chosen  = q_current[np.arange(BATCH_SIZE), actions]  # (batch,)

        # TD errors
        td_errors = targets - q_chosen  # (batch,)
        loss = float(np.mean(td_errors ** 2))

        # Gradient update: W[:,a] += lr * td_error * s
        for i in range(BATCH_SIZE):
            a = actions[i]
            err = td_errors[i]
            self.W[:, a] += LR * err * states[i]
            self.b[a]    += LR * err

        # Clip weights to prevent explosion
        np.clip(self.W, -10.0, 10.0, out=self.W)
        np.clip(self.b, -5.0, 5.0,  out=self.b)

        self.loss_history.append(loss)
        self.update_count += 1

        # Soft update target network
        if self.update_count % TARGET_UPDATE == 0:
            tau = 0.1  # soft update coefficient
            self.W_target = tau * self.W + (1 - tau) * self.W_target
            self.b_target = tau * self.b + (1 - tau) * self.b_target

        # Decay epsilon
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        return loss

    def policy_summary(self) -> dict:
        """
        Compute which action the policy prefers for different threat levels.
        Evaluates the policy on canonical threat states.
        """
        threat_states = {
            "low_threat":    np.array([0.1, 0.1, 0.0, 0.1, 0.1, 0.0,
                                       0.0, 0.2, 0.0, 0.1, 0.1, 0.0]),
            "medium_threat": np.array([0.4, 0.4, 0.1, 0.2, 0.3, 0.0,
                                       0.1, 0.5, 0.3, 0.4, 0.3, 0.2]),
            "high_threat":   np.array([0.8, 0.7, 0.3, 0.4, 0.6, 0.2,
                                       0.3, 0.8, 0.7, 0.7, 0.6, 0.5]),
            "ddos_state":    np.array([1.0, 0.6, 0.5, 0.6, 0.8, 0.4,
                                       0.6, 0.95, 0.9, 0.9, 0.8, 0.8]),
        }
        result = {}
        for name, state in threat_states.items():
            qs = self.q_values(state)
            best_action = ACTIONS[int(np.argmax(qs))]
            result[name] = {
                "best_action": best_action,
                "q_values": {ACTIONS[i]: round(float(qs[i]), 3)
                             for i in range(N_ACTIONS)},
            }
        return result

    def get_action_stats(self) -> dict:
        """Return per-action selection count and average reward."""
        stats = {}
        for i, action in enumerate(ACTIONS):
            count = int(self.action_counts[i])
            avg_r = (self.action_rewards[i] / count
                     if count > 0 else 0.0)
            stats[action] = {
                "count": count,
                "avg_reward": round(float(avg_r), 3),
            }
        return stats

    def serialize(self) -> dict:
        return {
            "W":             self.W.tolist(),
            "b":             self.b.tolist(),
            "W_target":      self.W_target.tolist(),
            "b_target":      self.b_target.tolist(),
            "epsilon":       self.epsilon,
            "update_count":  self.update_count,
            "loss_history":  list(self.loss_history)[-100:],
            "action_counts": self.action_counts.tolist(),
            "action_rewards":self.action_rewards.tolist(),
        }

    def deserialize(self, data: dict):
        self.W             = np.array(data["W"], dtype=np.float32)
        self.b             = np.array(data["b"], dtype=np.float32)
        self.W_target      = np.array(data["W_target"], dtype=np.float32)
        self.b_target      = np.array(data["b_target"], dtype=np.float32)
        self.epsilon       = data["epsilon"]
        self.update_count  = data["update_count"]
        self.loss_history  = collections.deque(
            data.get("loss_history", []), maxlen=500)
        self.action_counts = np.array(
            data.get("action_counts", [0]*N_ACTIONS), dtype=int)
        self.action_rewards= np.array(
            data.get("action_rewards", [0.0]*N_ACTIONS))


# ═══════════════════════════════════════════════════════════════════════════════
# TRAFFIC ANALYSIS WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

class TrafficEvent:
    __slots__ = ["timestamp", "proto", "src_ip", "dst_port",
                 "pkt_size", "flags"]

    def __init__(self, timestamp: float, proto: str, src_ip: str,
                 dst_port: int, pkt_size: int = 60, flags: str = ""):
        self.timestamp = timestamp
        self.proto     = proto
        self.src_ip    = src_ip
        self.dst_port  = dst_port
        self.pkt_size  = pkt_size
        self.flags     = flags


class TrafficWindow:
    """
    Sliding window of traffic events.
    Extracts normalized feature vector for DQN state representation.
    """

    def __init__(self, window_secs: float = 5.0):
        self.window = window_secs
        self.events: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def add(self, event: TrafficEvent):
        with self._lock:
            self.events.append(event)
            self._prune()

    def _prune(self):
        cutoff = time.time() - self.window
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()

    def snapshot(self) -> dict:
        """Extract traffic statistics from current window."""
        with self._lock:
            self._prune()
            events = list(self.events)

        if not events:
            return {
                "total": 0, "rate": 0.0, "port_diversity": 0.0,
                "icmp_ratio": 0.0, "udp_ratio": 0.0, "syn_ratio": 0.0,
                "frag_ratio": 0.0, "large_pkt_ratio": 0.0,
                "total_bytes": 0, "avg_size": 0.0,
                "unique_ports": 0, "sources": set(),
            }

        total  = len(events)
        rate   = total / self.window

        # Protocol breakdown
        proto_counts = collections.Counter(e.proto for e in events)
        unique_ports = len(set(e.dst_port for e in events))
        sizes        = [e.pkt_size for e in events]
        total_bytes  = sum(sizes)
        avg_size     = total_bytes / total

        return {
            "total":           total,
            "rate":            rate,
            "port_diversity":  unique_ports / max(total, 1),
            "unique_ports":    unique_ports,
            "icmp_ratio":      proto_counts.get("icmp", 0) / total,
            "udp_ratio":       proto_counts.get("udp",  0) / total,
            "syn_ratio":       proto_counts.get("syn",  0) / total,
            "frag_ratio":      proto_counts.get("frag", 0) / total,
            "large_pkt_ratio": sum(1 for s in sizes if s > 500) / total,
            "total_bytes":     total_bytes,
            "avg_size":        avg_size,
            "sources":         set(e.src_ip for e in events),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REPUTATION SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

class ReputationTracker:
    """
    Per-source IP reputation scoring system.
    Score range: 0.0 (benign) to 1.0 (confirmed attacker).
    Uses exponential decay for old attacks and fast increase for new ones.
    """

    def __init__(self):
        self.scores: dict  = {}
        self.event_counts: dict = collections.defaultdict(int)
        self.first_seen: dict  = {}
        self.last_seen: dict   = {}
        self.attack_types: dict = collections.defaultdict(set)

    def update(self, src_ip: str, is_attack: bool,
               attack_severity: float = 0.5):
        """Update reputation score for a source IP."""
        if src_ip not in self.scores:
            self.scores[src_ip]    = 0.5  # start neutral
            self.first_seen[src_ip] = time.time()

        self.last_seen[src_ip] = time.time()
        self.event_counts[src_ip] += 1

        old_score = self.scores[src_ip]

        if is_attack:
            # Increase reputation score (more dangerous = faster increase)
            delta = 0.08 + attack_severity * 0.12
            new_score = min(1.0, old_score + delta)
        else:
            # Slow decay for clean traffic
            new_score = max(0.0, old_score - 0.01)

        self.scores[src_ip] = round(new_score, 4)

    def get(self, src_ip: str) -> float:
        return self.scores.get(src_ip, 0.5)

    def is_known_attacker(self, src_ip: str,
                          threshold: float = 0.7) -> bool:
        return self.get(src_ip) >= threshold

    def decay_all(self):
        """Passive time-based decay for all scores."""
        for ip in list(self.scores.keys()):
            self.scores[ip] = max(0.0, self.scores[ip] - 0.005)

    def to_dict(self) -> dict:
        return {ip: round(v, 3) for ip, v in self.scores.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    EMA-based anomaly detector.
    Maintains moving averages of normal traffic characteristics.
    Computes anomaly scores as deviation from baseline.
    """

    def __init__(self, alpha: float = 0.15):
        self.alpha = alpha  # EMA smoothing factor

        # EMA baselines
        self.ema_rate       = 0.0
        self.ema_port_div   = 0.0
        self.ema_icmp_ratio = 0.0
        self.ema_udp_ratio  = 0.0
        self.ema_syn_ratio  = 0.0
        self.ema_pkt_size   = 60.0

        # Baseline (established during warmup)
        self.baseline_rate  = 0.0
        self.baseline_set   = False
        self.baseline_samples = []

        # Consecutive high-anomaly window counter
        self.consecutive_high = 0
        self.max_consecutive  = 0

    def update_ema(self, snap: dict):
        """Update all EMA values with new snapshot."""
        a = self.alpha
        self.ema_rate       = a * snap.get("rate", 0)       + (1-a) * self.ema_rate
        self.ema_port_div   = a * snap.get("port_diversity", 0) + (1-a) * self.ema_port_div
        self.ema_icmp_ratio = a * snap.get("icmp_ratio", 0) + (1-a) * self.ema_icmp_ratio
        self.ema_udp_ratio  = a * snap.get("udp_ratio",  0) + (1-a) * self.ema_udp_ratio
        self.ema_syn_ratio  = a * snap.get("syn_ratio",  0) + (1-a) * self.ema_syn_ratio
        self.ema_pkt_size   = a * snap.get("avg_size", 60)  + (1-a) * self.ema_pkt_size

    def add_baseline_sample(self, snap: dict):
        """Accumulate baseline samples during warmup period."""
        self.baseline_samples.append(snap.get("rate", 0))

    def establish_baseline(self):
        """Compute baseline from accumulated samples."""
        if self.baseline_samples:
            self.baseline_rate = sum(self.baseline_samples) / len(self.baseline_samples)
            self.baseline_set  = True
            log(f"Baseline established: {self.baseline_rate:.3f} pkt/s "
                f"from {len(self.baseline_samples)} samples")
        self.baseline_samples = []

    def compute_threat_score(self, snap: dict, reputation: float) -> float:
        """
        Compute composite threat score 0.0-1.0.

        Components:
        - Rate anomaly: how much above baseline is current rate
        - Port diversity: scanning pattern detection
        - Protocol ratios: ICMP/UDP/SYN floods
        - Packet size: large packets = bandwidth attacks
        - Reputation: known attacker bonus
        - Consecutive high windows: sustained attack bonus
        """
        base = max(self.baseline_rate, 0.1)
        rate = snap.get("rate", 0)

        # Rate anomaly (normalized 0-1)
        rate_anom = min(rate / (base * 15), 1.0)

        # Port diversity (scanning)
        port_div = min(snap.get("port_diversity", 0) * 3, 1.0)

        # Protocol-specific anomalies
        icmp_anom = min(snap.get("icmp_ratio", 0) * 5, 1.0)
        udp_anom  = min(snap.get("udp_ratio",  0) * 4, 1.0)
        syn_anom  = min(snap.get("syn_ratio",  0) * 4, 1.0)
        frag_anom = min(snap.get("frag_ratio", 0) * 8, 1.0)

        # Large packets
        large_anom = min(snap.get("large_pkt_ratio", 0) * 2, 1.0)

        # Sustained attack bonus
        consec_bonus = min(self.consecutive_high / 10.0, 0.2)

        # Weighted combination
        threat = (
            rate_anom  * 0.30 +
            port_div   * 0.12 +
            icmp_anom  * 0.12 +
            udp_anom   * 0.10 +
            syn_anom   * 0.14 +
            frag_anom  * 0.08 +
            large_anom * 0.06 +
            reputation * 0.05 +
            consec_bonus
        )

        threat = min(threat, 1.0)

        # Update consecutive counter
        if threat > 0.5:
            self.consecutive_high += 1
            self.max_consecutive = max(self.max_consecutive,
                                       self.consecutive_high)
        else:
            self.consecutive_high = max(0, self.consecutive_high - 1)

        return round(threat, 4)

    def get_alert_level(self, threat_score: float) -> int:
        """Convert threat score to 0-3 alert level."""
        for i, threshold in enumerate(ALERT_THRESHOLDS):
            if threat_score <= threshold:
                return i
        return 3


# ═══════════════════════════════════════════════════════════════════════════════
# NFTABLES RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class NFTablesEngine:
    """
    Dynamic nftables rule management.
    Applies, tracks, and expires firewall rules based on DQN decisions.
    Maintains rule history and supports automatic expiry.
    """

    def __init__(self, netns: str = NETNS):
        self.netns = netns
        self.active_rules: dict = {}   # ip -> {action, expires, rule_handle}
        self.rule_log: list = []
        self.total_blocks = 0
        self.total_rate_limits = 0
        self.total_false_pos_est = 0
        self._lock = threading.Lock()

    def init(self):
        """Initialize nftables table and chain."""
        cmds = [
            ["sudo", "ip", "netns", "exec", self.netns,
             "nft", "add", "table", "inet", "defender"],
            ["sudo", "ip", "netns", "exec", self.netns,
             "nft", "add", "chain", "inet", "defender", "input",
             "{", "type", "filter", "hook", "input",
             "priority", "0", ";", "}"],
        ]
        for cmd in cmds:
            run_cmd(cmd)
        log("nftables table and chain initialized")

    def flush(self):
        """Remove all rules from the defender chain."""
        run_cmd(["sudo", "ip", "netns", "exec", self.netns,
                 "nft", "flush", "chain", "inet", "defender", "input"])

    def _add_rule(self, rule_str: str) -> bool:
        rc, out = run_cmd(rule_str)
        if rc != 0:
            log(f"nft rule failed: {out}", "WARN")
            return False
        return True

    def apply_rate_limit(self, src_ip: str, rate: int,
                         burst: int, duration: int) -> bool:
        """Apply connection rate limit rule for source IP."""
        rule = (
            f"sudo ip netns exec {self.netns} nft add rule inet defender "
            f"input ip saddr {src_ip} ct state new "
            f"limit rate {rate}/second burst {burst} packets accept"
        )
        if self._add_rule(rule):
            with self._lock:
                self.active_rules[src_ip] = {
                    "action":  f"rate_limit_{rate}/s",
                    "expires": time.time() + duration,
                    "type":    "rate_limit",
                }
                self.total_rate_limits += 1
                ts = datetime.now().strftime("%H:%M:%S")
                self.rule_log.append(
                    f"RATELIMIT {src_ip} {rate}/s for {duration}s @ {ts}")
            return True
        return False

    def apply_block(self, src_ip: str, duration: int) -> bool:
        """Apply hard block rule for source IP."""
        rule = (
            f"sudo ip netns exec {self.netns} nft add rule inet defender "
            f"input ip saddr {src_ip} drop"
        )
        if self._add_rule(rule):
            with self._lock:
                self.active_rules[src_ip] = {
                    "action":  f"block_{duration}s",
                    "expires": time.time() + duration,
                    "type":    "block",
                }
                self.total_blocks += 1
                ts = datetime.now().strftime("%H:%M:%S")
                self.rule_log.append(
                    f"BLOCK {src_ip} for {duration}s @ {ts}")
            return True
        return False

    def apply_block_icmp(self, src_ip: str, duration: int) -> bool:
        """Block only ICMP from source."""
        rule = (
            f"sudo ip netns exec {self.netns} nft add rule inet defender "
            f"input ip saddr {src_ip} ip protocol icmp drop"
        )
        if self._add_rule(rule):
            with self._lock:
                self.active_rules[src_ip] = {
                    "action":  "block_icmp",
                    "expires": time.time() + duration,
                    "type":    "block_icmp",
                }
                ts = datetime.now().strftime("%H:%M:%S")
                self.rule_log.append(f"BLOCK_ICMP {src_ip} @ {ts}")
            return True
        return False

    def apply_block_udp(self, src_ip: str, duration: int) -> bool:
        """Block high-rate UDP from source."""
        rule = (
            f"sudo ip netns exec {self.netns} nft add rule inet defender "
            f"input ip saddr {src_ip} ip protocol udp "
            f"limit rate over 10/second drop"
        )
        if self._add_rule(rule):
            with self._lock:
                self.active_rules[src_ip] = {
                    "action":  "block_udp_flood",
                    "expires": time.time() + duration,
                    "type":    "block_udp",
                }
                ts = datetime.now().strftime("%H:%M:%S")
                self.rule_log.append(f"BLOCK_UDP {src_ip} @ {ts}")
            return True
        return False

    def expire_rules(self):
        """Remove expired rules and flush/rebuild active ones."""
        now = time.time()
        expired = []
        with self._lock:
            for ip, rule in self.active_rules.items():
                if now > rule["expires"]:
                    expired.append(ip)
            for ip in expired:
                del self.active_rules[ip]
                ts = datetime.now().strftime("%H:%M:%S")
                self.rule_log.append(f"EXPIRED {ip} @ {ts}")

        if expired:
            # Flush all rules and re-apply active ones
            self.flush()
            with self._lock:
                active = dict(self.active_rules)
            for ip, rule in active.items():
                if rule["type"] == "block":
                    remaining = int(rule["expires"] - now)
                    self.apply_block(ip, remaining)
                elif rule["type"] == "rate_limit":
                    remaining = int(rule["expires"] - now)
                    self.apply_rate_limit(ip, 10, 20, remaining)
                elif rule["type"] == "block_icmp":
                    remaining = int(rule["expires"] - now)
                    self.apply_block_icmp(ip, remaining)
            for ip in expired:
                log(f"Rule expired for {ip}", "EXPIRE")

    def is_blocked(self, src_ip: str) -> bool:
        with self._lock:
            return src_ip in self.active_rules

    def get_active_rules(self) -> dict:
        with self._lock:
            return dict(self.active_rules)

    def get_log(self, n: int = 10) -> list:
        return self.rule_log[-n:]


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_reward(
    action_idx: int,
    threat_score: float,
    is_attack: bool,
    prev_threat: float,
    reputation: float,
) -> float:
    """
    Shaped reward function for defender DQN.

    True positive rewards:
    - Blocking/rate-limiting during real attack: positive reward
    - Higher reward for catching higher-severity attacks early

    False positive penalties:
    - Blocking/rate-limiting during benign traffic: negative reward
    - Harder blocks during low threat = higher penalty

    Adaptive rewards:
    - Reward de-escalating when threat decreases (good judgement)
    - Reward proportional escalation to threat level
    """
    action = ACTIONS[action_idx]
    aggression = ACTION_AGGRESSION[action]

    if is_attack:
        # True positive: we correctly identified an attack
        if aggression >= 0.7:
            r = 2.0 + threat_score * 1.0   # big block during real attack
        elif aggression >= 0.3:
            r = 1.2 + threat_score * 0.5   # rate limit during real attack
        elif aggression >= 0.1:
            r = 0.5                          # alert during real attack
        else:
            r = -0.5 * threat_score         # monitor = missed detection penalty

        # Bonus for catching early (when threat just escalated)
        if threat_score > prev_threat + 0.2:
            r += 0.3  # caught escalation quickly

    else:
        # Not an attack (low threat traffic)
        if aggression == 0.0:
            r = 0.4                          # correct: just monitoring
        elif aggression <= 0.2:
            r = 0.1                          # mild over-reaction, mostly ok
        elif aggression <= 0.4:
            r = -0.3 - threat_score * 0.2   # moderate false positive
        elif aggression <= 0.7:
            r = -0.8 - threat_score * 0.3   # significant false positive
        else:
            r = -1.5 - threat_score * 0.5   # severe false positive (hard block)

    # Scale by threat intensity
    r *= (0.5 + threat_score * 0.5)

    # Clamp to reasonable range
    return round(max(-3.0, min(3.0, r)), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# STATE VECTOR BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_state_vector(
    snap: dict,
    reputation: float,
    alert_level: int,
    baseline_rate: float,
    consecutive_high: int,
) -> np.ndarray:
    """
    Build normalized 12-dimensional state vector for DQN input.
    All values normalized to [0, 1] range.
    """
    base = max(baseline_rate, 0.1)
    rate = snap.get("rate", 0)

    return np.array([
        min(rate / (base * 20), 1.0),              # [0] rate anomaly
        min(snap.get("port_diversity", 0), 1.0),   # [1] port diversity
        snap.get("icmp_ratio",       0.0),          # [2] ICMP ratio
        snap.get("udp_ratio",        0.0),          # [3] UDP ratio
        snap.get("syn_ratio",        0.0),          # [4] SYN ratio
        snap.get("frag_ratio",       0.0),          # [5] fragment ratio
        snap.get("large_pkt_ratio",  0.0),          # [6] large packet ratio
        min(reputation, 1.0),                       # [7] source reputation
        min(alert_level / 3.0, 1.0),               # [8] alert level norm
        min(rate / 500.0, 1.0),                     # [9] raw PPS norm
        min(snap.get("total_bytes", 0) / 500_000, 1.0),  # [10] bytes norm
        min(consecutive_high / 10.0, 1.0),         # [11] sustained attack
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAFFIC MONITOR (tcpdump parser)
# ═══════════════════════════════════════════════════════════════════════════════

def parse_tcpdump_line(line: str) -> TrafficEvent:
    """
    Parse a tcpdump output line into a TrafficEvent.
    Handles TCP (with flags), UDP, ICMP, and IP fragments.
    """
    now = time.time()

    # Detect protocol
    proto = "tcp"
    flags = ""

    line_upper = line.upper()
    if "ICMP" in line_upper:
        proto = "icmp"
    elif " UDP " in line_upper or "UDP" in line_upper:
        proto = "udp"

    # Check for IP fragments
    if "FRAG" in line_upper or "FLAGS [+]" in line or "> 0+" in line:
        proto = "frag"

    # Check for TCP flags
    syn_match = "FLAGS [S]" in line or " S " in line.split(":")[0]
    if proto == "tcp" and syn_match:
        proto = "syn"
        flags = "S"

    # Extract src_ip and dst_port
    src_ip, dst_port = None, 0
    pkt_size = 60

    # Pattern: src_ip.sport > dst_ip.dport
    m = re.search(r'(\d+\.\d+\.\d+\.\d+)\.(\d+) > (\d+\.\d+\.\d+\.\d+)\.(\d+)',
                  line)
    if m:
        src_ip = m.group(1)
        try:
            dst_port = int(m.group(4))
        except:
            dst_port = 0

    # Extract packet length
    lm = re.search(r'length (\d+)', line)
    if lm:
        try:
            pkt_size = int(lm.group(1))
        except:
            pass

    if src_ip:
        return TrafficEvent(now, proto, src_ip, dst_port, pkt_size, flags)
    return None


def monitor_traffic(window: TrafficWindow, stop_event: threading.Event):
    """
    Run tcpdump in the right namespace, parse output, feed to window.
    Runs in a daemon thread.
    """
    log("Starting tcpdump monitor on gre1")
    while not stop_event.is_set():
        try:
            proc = subprocess.Popen(
                ["sudo", "ip", "netns", "exec", NETNS,
                 "tcpdump", "-i", IFACE, "-n", "-l",
                 "--immediate-mode", "-v", "-e"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )

            with open(TRAFFIC_LOG, "a") as tf:
                while not stop_event.is_set():
                    line = proc.stdout.readline()
                    if not line:
                        break
                    tf.write(line)
                    event = parse_tcpdump_line(line)
                    if event:
                        window.add(event)

            proc.terminate()
            proc.wait(timeout=3)

        except Exception as e:
            log(f"tcpdump error: {e}, restarting in 2s", "WARN")
            time.sleep(2)


def start_listeners():
    """Start ncat listeners so attacker has real services to hit."""
    ports = [22, 80, 443, 8080]
    for port in ports:
        subprocess.Popen(
            ["sudo", "ip", "netns", "exec", NETNS,
             "ncat", "-lk", "-p", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    log(f"Listeners started on ports {ports}")


# ═══════════════════════════════════════════════════════════════════════════════
# STATE PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_state() -> tuple:
    """Load defender state and DQN from disk. Returns (state_dict, dqn)."""
    dqn = DQNetwork()

    default = {
        "alert_level":      0,
        "total_detections": 0,
        "total_blocks":     0,
        "total_fp_est":     0,
        "adaptation_count": 0,
        "reward_history":   [],
        "last_reward":      0.0,
        "cumul_reward":     0.0,
        "prev_threat":      0.0,
        "cycle":            0,
    }

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            state = default.copy()
            state.update({k: v for k, v in saved.items() if k != "dqn"})
            if "dqn" in saved and saved["dqn"]:
                dqn.deserialize(saved["dqn"])
                log(f"DQN loaded: {dqn.update_count} updates, "
                    f"eps={dqn.epsilon:.3f}, "
                    f"replay={len(dqn.replay_buffer)}")
            return state, dqn
        except Exception as e:
            log(f"State load error: {e}, starting fresh", "WARN")

    return default, dqn


def save_state(state: dict, dqn: DQNetwork,
               nft: NFTablesEngine, rep: ReputationTracker):
    """Save all defender state to disk."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    d = dict(state)
    d["reward_history"] = state["reward_history"][-200:]
    d["dqn"] = dqn.serialize()
    d["reputation"] = rep.to_dict()
    d["rules_log"] = nft.get_log(50)
    with open(STATE_FILE, "w") as f:
        json.dump(d, f, indent=2)


def log_history(event: dict):
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def update_shared(state: dict, dqn: DQNetwork, nft: NFTablesEngine,
                  rep: ReputationTracker, detector: AnomalyDetector,
                  action: str, reward: float, snap: dict,
                  threat_score: float):
    """Write defender state to shared file for dashboard."""
    shared = {}
    if os.path.exists(SHARED_FILE):
        try:
            with open(SHARED_FILE) as f:
                shared = json.load(f)
        except:
            pass

    alert_labels = ["NORMAL", "ELEVATED", "HIGH", "CRITICAL"]
    al = state["alert_level"]

    # Recent loss
    recent_loss = float(dqn.loss_history[-1]) if dqn.loss_history else 0.0

    shared["defender"] = {
        "alert_level":       al,
        "alert_label":       alert_labels[min(al, 3)],
        "threat_score":      round(threat_score, 3),
        "total_detections":  state["total_detections"],
        "total_blocks":      state["total_blocks"],
        "total_fp_est":      state["total_fp_est"],
        "adaptation_count":  state["adaptation_count"],
        "current_action":    action,
        "last_reward":       round(reward, 3),
        "cumul_reward":      round(state["cumul_reward"], 2),
        "reward_history":    state["reward_history"][-30:],
        "baseline_rate":     round(detector.baseline_rate, 3),
        "ema_rate":          round(detector.ema_rate, 3),
        "traffic_rate":      round(snap.get("rate", 0), 2),
        "port_diversity":    round(snap.get("port_diversity", 0), 3),
        "icmp_ratio":        round(snap.get("icmp_ratio", 0), 3),
        "udp_ratio":         round(snap.get("udp_ratio", 0), 3),
        "syn_ratio":         round(snap.get("syn_ratio", 0), 3),
        "consecutive_high":  detector.consecutive_high,
        "max_consecutive":   detector.max_consecutive,
        "reputation":        rep.to_dict(),
        "active_rules":      {ip: r["action"]
                              for ip, r in nft.get_active_rules().items()},
        "rules_log":         nft.get_log(8),
        "dqn_epsilon":       round(dqn.epsilon, 4),
        "dqn_updates":       dqn.update_count,
        "dqn_loss":          round(recent_loss, 5),
        "dqn_policy":        dqn.policy_summary(),
        "dqn_action_stats":  dqn.get_action_stats(),
        "cycle":             state["cycle"],
        "timestamp":         datetime.now().isoformat(),
    }

    with open(SHARED_FILE, "w") as f:
        json.dump(shared, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("Adaptive Defender v2 Starting (DQN)")
    log(f"State dim: {STATE_DIM}, Actions: {N_ACTIONS}")
    log(f"DQN: lr={LR}, gamma={GAMMA}, replay={REPLAY_MAX}")
    log("=" * 60)

    os.makedirs("/home/kaexb/sim", exist_ok=True)

    # Initialize components
    state, dqn = load_state()
    detector = AnomalyDetector(alpha=0.2)
    reputation = ReputationTracker()
    nft = NFTablesEngine(netns=NETNS)
    window = TrafficWindow(window_secs=5.0)
    stop_event = threading.Event()

    # Setup
    nft.init()
    start_listeners()

    # Start traffic monitor thread
    monitor_thread = threading.Thread(
        target=monitor_traffic,
        args=(window, stop_event),
        daemon=True,
    )
    monitor_thread.start()
    log("Traffic monitor thread started")

    # ── WARMUP / BASELINE PHASE ──────────────────────────────────────────────
    log("Warming up baseline (20s)...")
    for i in range(20):
        snap = window.snapshot()
        detector.add_baseline_sample(snap)
        detector.update_ema(snap)
        time.sleep(1)
        if i % 5 == 0:
            log(f"  Warmup {i+1}/20 rate={snap.get('rate',0):.2f}/s")

    detector.establish_baseline()

    # ── MAIN DEFENSE LOOP ────────────────────────────────────────────────────
    prev_state_vec = None
    prev_action_idx = 0
    prev_threat = 0.0
    last_save = time.time()
    last_flush = time.time()
    last_rep_decay = time.time()

    try:
        while True:
            state["cycle"] += 1
            cycle = state["cycle"]

            # Get traffic snapshot
            snap = window.snapshot()
            detector.update_ema(snap)

            # Compute reputation for primary attacker
            rep_score = reputation.get(SRC_IP)

            # Compute threat score
            threat_score = detector.compute_threat_score(snap, rep_score)
            alert_level  = detector.get_alert_level(threat_score)
            state["alert_level"] = alert_level

            # Build state vector for DQN
            state_vec = build_state_vector(
                snap, rep_score, alert_level,
                detector.baseline_rate,
                detector.consecutive_high,
            )

            # DQN action selection
            action_idx  = dqn.select_action(state_vec)
            action_name = ACTIONS[action_idx]

            # Determine if this is a real attack
            is_attack = threat_score > 0.35

            # Update reputation
            reputation.update(SRC_IP, is_attack,
                              attack_severity=threat_score)

            # Execute action
            blocked = nft.is_blocked(SRC_IP)

            if action_name == "monitor":
                pass  # just watch

            elif action_name == "log_alert" and not blocked:
                ts = datetime.now().strftime("%H:%M:%S")
                nft.rule_log.append(f"ALERT {SRC_IP} threat={threat_score:.2f} @ {ts}")

            elif action_name == "rate_limit_soft" and not blocked:
                nft.apply_rate_limit(SRC_IP, 50, 100, 60)

            elif action_name == "rate_limit_medium" and not blocked:
                nft.apply_rate_limit(SRC_IP, 20, 40, 60)

            elif action_name == "rate_limit_hard" and not blocked:
                nft.apply_rate_limit(SRC_IP, 5, 10, 60)

            elif action_name == "block_icmp" and not blocked:
                nft.apply_block_icmp(SRC_IP, 45)

            elif action_name == "block_udp_flood" and not blocked:
                nft.apply_block_udp(SRC_IP, 45)

            elif action_name == "block_temp" and not blocked:
                nft.apply_block(SRC_IP, 30)
                state["total_blocks"] += 1

            elif action_name == "block_long" and not blocked:
                nft.apply_block(SRC_IP, 120)
                state["total_blocks"] += 1

            # Track detection
            if is_attack:
                state["total_detections"] += 1

            # Estimate false positives
            if not is_attack and ACTION_AGGRESSION[action_name] >= 0.5:
                state["total_fp_est"] += 1

            # Compute reward
            reward = compute_reward(
                action_idx, threat_score, is_attack,
                prev_threat, rep_score,
            )
            state["last_reward"]   = reward
            state["cumul_reward"] += reward
            state["reward_history"].append(round(reward, 3))
            state["prev_threat"]   = threat_score

            # Store transition in replay buffer and train
            if prev_state_vec is not None:
                done = False
                dqn.store_transition(
                    prev_state_vec, prev_action_idx,
                    reward, state_vec, done,
                )
                if cycle % 3 == 0:
                    loss = dqn.train_step()
                    if loss > 0:
                        state["adaptation_count"] += 1

            prev_state_vec  = state_vec.copy()
            prev_action_idx = action_idx
            prev_threat     = threat_score

            # Expire old rules
            nft.expire_rules()

            # Periodic reputation decay
            if time.time() - last_rep_decay > 30:
                reputation.decay_all()
                last_rep_decay = time.time()

            # Periodic full flush + rebuild (prevent rule accumulation)
            if time.time() - last_flush > 90:
                nft.flush()
                last_flush = time.time()
                active = nft.get_active_rules()
                for ip, rule in active.items():
                    rem = int(rule["expires"] - time.time())
                    if rem > 0:
                        if rule["type"] == "block":
                            nft.apply_block(ip, rem)
                        elif rule["type"] == "rate_limit":
                            nft.apply_rate_limit(ip, 10, 20, rem)

            # Log structured history every 10 cycles
            if cycle % 10 == 0:
                log_history({
                    "ts": datetime.now().isoformat(),
                    "cycle": cycle,
                    "threat": threat_score,
                    "alert": alert_level,
                    "action": action_name,
                    "reward": reward,
                    "is_attack": is_attack,
                    "rate": snap.get("rate", 0),
                    "rep": rep_score,
                    "dqn_eps": dqn.epsilon,
                    "dqn_loss": float(dqn.loss_history[-1]) if dqn.loss_history else 0,
                })

            # Update shared state for dashboard
            update_shared(state, dqn, nft, reputation, detector,
                          action_name, reward, snap, threat_score)

            # Periodic save
            if time.time() - last_save > 30:
                save_state(state, dqn, nft, reputation)
                last_save = time.time()
                loss = float(dqn.loss_history[-1]) if dqn.loss_history else 0.0
                log(f"Cycle {cycle:5d} | "
                    f"threat={threat_score:.3f} [{['NORM','ELEV','HIGH','CRIT'][min(alert_level,3)]}] | "
                    f"action={action_name:20s} | "
                    f"reward={reward:+.3f} | "
                    f"eps={dqn.epsilon:.3f} | "
                    f"loss={loss:.5f} | "
                    f"blk={state['total_blocks']} "
                    f"det={state['total_detections']}")

            time.sleep(1)

    except KeyboardInterrupt:
        log("Defender shutting down...")
        stop_event.set()
        nft.flush()
        save_state(state, dqn, nft, reputation)
        log("State saved. Goodbye.")


if __name__ == "__main__":
    main()