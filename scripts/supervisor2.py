#!/usr/bin/env python3
"""
supervisor2.py
Episode manager, process supervisor, health monitor, and live terminal dashboard.
Launches defender first, waits for baseline, then launches attacker.
Monitors both processes and restarts if they die.
Renders a live terminal dashboard updating every 2 seconds.
"""

import subprocess
import time
import json
import os
import sys
import signal
import threading
import shutil
from datetime import datetime, timedelta
from collections import deque

# ── Constants ────────────────────────────────────────────────────────────────
SIMDIR       = "/home/kaexb/sim"
DEFENDER_PY  = "/home/kaexb/defender2.py"
ATTACKER_PY  = "/home/kaexb/attacker2.py"
SHARED_FILE  = f"{SIMDIR}/shared_state.json"
BASELINE_WAIT = 25    # seconds to wait for defender baseline
MAX_RESTARTS  = 5     # max restarts before giving up
DASH_INTERVAL = 2.0   # dashboard refresh interval seconds

# ── ANSI Colors ──────────────────────────────────────────────────────────────
class C:
    RED      = '\033[0;31m'
    LRED     = '\033[1;31m'
    GREEN    = '\033[0;32m'
    LGREEN   = '\033[1;32m'
    YELLOW   = '\033[1;33m'
    BLUE     = '\033[0;34m'
    CYAN     = '\033[0;36m'
    LCYAN    = '\033[1;36m'
    MAGENTA  = '\033[0;35m'
    LMAGENTA = '\033[1;35m'
    BOLD     = '\033[1m'
    DIM      = '\033[2m'
    RESET    = '\033[0m'
    # Alert level colors
    ALERT    = [GREEN, YELLOW, RED, LRED]

def colored(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"

def clear_screen():
    print("\033[2J\033[H", end="", flush=True)

def move_to(row: int, col: int = 1):
    print(f"\033[{row};{col}H", end="", flush=True)

def hide_cursor():
    print("\033[?25l", end="", flush=True)

def show_cursor():
    print("\033[?25h", end="", flush=True)

def get_terminal_size() -> tuple:
    try:
        size = shutil.get_terminal_size()
        return size.columns, size.lines
    except:
        return 120, 40

# ── SPARKLINE ────────────────────────────────────────────────────────────────
SPARK_CHARS = "▁▂▃▄▅▆▇█"

def sparkline(values: list, width: int = 24,
              color: str = C.CYAN) -> str:
    """Render a sparkline from a list of floats."""
    if not values:
        return colored("─" * width, C.DIM)
    recent = values[-width:]
    mn, mx = min(recent), max(recent)
    if mn == mx:
        return colored("─" * width, C.DIM)
    result = ""
    for v in recent:
        idx = int((v - mn) / (mx - mn) * (len(SPARK_CHARS) - 1))
        idx = max(0, min(idx, len(SPARK_CHARS) - 1))
        result += SPARK_CHARS[idx]
    # Pad with dashes if shorter than width
    result = result.ljust(width, "─")
    return colored(result, color)

# ── BAR ──────────────────────────────────────────────────────────────────────
def bar(val: float, max_val: float = 1.0, width: int = 20,
        color: str = C.GREEN, empty_color: str = C.DIM) -> str:
    """Render a horizontal bar."""
    if max_val <= 0:
        max_val = 1.0
    filled = int(min(val / max_val, 1.0) * width)
    filled = max(0, min(filled, width))
    b = (colored("█" * filled, color) +
         colored("░" * (width - filled), empty_color))
    return b

# ── TABLE ROW ────────────────────────────────────────────────────────────────
def padl(text: str, width: int) -> str:
    """Left-pad with spaces, strip ANSI for width calculation."""
    ansi_escape = __import__('re').compile(r'\033\[[0-9;]*m')
    visible_len = len(ansi_escape.sub('', text))
    padding = max(0, width - visible_len)
    return text + " " * padding

def padr(text: str, width: int) -> str:
    ansi_escape = __import__('re').compile(r'\033\[[0-9;]*m')
    visible_len = len(ansi_escape.sub('', text))
    padding = max(0, width - visible_len)
    return " " * padding + text


# ═══════════════════════════════════════════════════════════════════════════════
# PROCESS SUPERVISOR
# ═══════════════════════════════════════════════════════════════════════════════

class ManagedProcess:
    """Wrapper for a subprocess with restart capability."""

    def __init__(self, name: str, cmd: list, log_file: str):
        self.name      = name
        self.cmd       = cmd
        self.log_file  = log_file
        self.proc      = None
        self.pid       = None
        self.restarts  = 0
        self.start_time = None
        self.status    = "stopped"  # stopped, starting, running, dead

    def start(self) -> bool:
        try:
            log_fh = open(self.log_file, "a")
            self.proc = subprocess.Popen(
                self.cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
            )
            self.pid        = self.proc.pid
            self.start_time = time.time()
            self.status     = "running"
            return True
        except Exception as e:
            self.status = "dead"
            return False

    def is_alive(self) -> bool:
        if self.proc is None:
            return False
        return self.proc.poll() is None

    def stop(self):
        if self.proc and self.is_alive():
            try:
                subprocess.run(["sudo", "kill", str(self.pid)],
                               capture_output=True)
                self.proc.wait(timeout=5)
            except:
                try:
                    self.proc.kill()
                except:
                    pass
        self.status = "stopped"

    def restart(self) -> bool:
        self.stop()
        self.restarts += 1
        time.sleep(2)
        return self.start()

    def uptime(self) -> str:
        if self.start_time is None:
            return "─"
        elapsed = int(time.time() - self.start_time)
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        if h > 0:
            return f"{h}h{m:02d}m{s:02d}s"
        return f"{m}m{s:02d}s"


class Supervisor:
    """
    Manages the defender and attacker processes.
    Monitors health, restarts on failure, tracks episode stats.
    """

    def __init__(self):
        os.makedirs(SIMDIR, exist_ok=True)

        self.defender = ManagedProcess(
            name     = "defender",
            cmd      = ["sudo", "python3", DEFENDER_PY],
            log_file = f"{SIMDIR}/defender_stdout.log",
        )
        self.attacker = ManagedProcess(
            name     = "attacker",
            cmd      = ["sudo", "python3", ATTACKER_PY],
            log_file = f"{SIMDIR}/attacker_stdout.log",
        )

        self.start_time     = time.time()
        self.attacker_started = False
        self.episode        = 1
        self.episode_start  = time.time()
        self._stop          = threading.Event()

        # Health check history
        self.health_log     = deque(maxlen=100)
        self.event_log      = deque(maxlen=50)

        # Stats history for dashboard sparklines
        self.atk_success_history = deque(maxlen=120)
        self.dfn_block_history   = deque(maxlen=120)
        self.threat_history      = deque(maxlen=120)
        self.pps_history         = deque(maxlen=120)

    def add_event(self, msg: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.event_log.append(f"[{ts}][{level}] {msg}")
        # Also print to log file
        log_path = f"{SIMDIR}/supervisor.log"
        with open(log_path, "a") as f:
            f.write(f"[{ts}][{level}] {msg}\n")

    def start_defender(self) -> bool:
        self.add_event("Starting defender (DQN)...")
        if self.defender.start():
            self.add_event(f"Defender started PID={self.defender.pid}")
            return True
        self.add_event("Defender failed to start!", "ERROR")
        return False

    def start_attacker(self) -> bool:
        self.add_event("Starting attacker (Q-Learning)...")
        if self.attacker.start():
            self.add_event(f"Attacker started PID={self.attacker.pid}")
            self.attacker_started = True
            return True
        self.add_event("Attacker failed to start!", "ERROR")
        return False

    def health_check(self):
        """Check process health, restart if needed."""
        if not self.defender.is_alive():
            if self.defender.restarts < MAX_RESTARTS:
                self.add_event(f"Defender died! Restarting "
                               f"(attempt {self.defender.restarts+1})", "WARN")
                self.defender.restart()
            else:
                self.add_event("Defender exceeded max restarts!", "ERROR")

        if self.attacker_started and not self.attacker.is_alive():
            if self.attacker.restarts < MAX_RESTARTS:
                self.add_event(f"Attacker died! Restarting "
                               f"(attempt {self.attacker.restarts+1})", "WARN")
                self.attacker.restart()
            else:
                self.add_event("Attacker exceeded max restarts!", "ERROR")

    def read_shared(self) -> dict:
        try:
            with open(SHARED_FILE) as f:
                return json.load(f)
        except:
            return {}

    def update_history(self, shared: dict):
        """Update sparkline history from shared state."""
        atk = shared.get("attacker", {})
        dfn = shared.get("defender", {})

        total = max(atk.get("total_attacks", 0), 1)
        succ  = atk.get("total_success", 0)
        self.atk_success_history.append(succ / total)
        self.dfn_block_history.append(dfn.get("total_blocks", 0))
        self.threat_history.append(dfn.get("threat_score", 0))
        self.pps_history.append(atk.get("pps", 0))

    def cleanup(self):
        """Stop all processes and clean up."""
        self.add_event("Supervisor shutting down...")
        self.attacker.stop()
        self.defender.stop()
        # Flush nftables
        subprocess.run(
            ["sudo", "ip", "netns", "exec", "right",
             "nft", "flush", "table", "inet", "defender"],
            capture_output=True,
        )
        # Kill stray ncat listeners
        subprocess.run(["sudo", "pkill", "-f", "ncat -lk"],
                       capture_output=True)
        self.add_event("Cleanup complete.")
        show_cursor()


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD RENDERER
# ═══════════════════════════════════════════════════════════════════════════════

STAGE_NAMES = {0: "RECON", 1: "PROBE", 2: "FLOOD", 3: "DDOS"}
STAGE_COLORS = {
    0: C.CYAN,
    1: C.GREEN,
    2: C.YELLOW,
    3: C.LRED,
}


def render_header(cols: int):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    title = "⚔  ADAPTIVE ATTACK vs DEFENSE SIMULATION  🛡"
    sub   = f"TECO Lab · rhel-kaexb.idm.tec.net · {now}"
    print(colored("═" * cols, C.CYAN))
    print(colored(title.center(cols), C.LCYAN + C.BOLD))
    print(colored(sub.center(cols), C.DIM))
    print(colored("═" * cols, C.CYAN))


def render_alert_banner(dfn: dict, cols: int):
    alert = dfn.get("alert_label", "NORMAL")
    threat = dfn.get("threat_score", 0)
    al    = dfn.get("alert_level", 0)
    acolor = C.ALERT[min(al, 3)]

    det   = dfn.get("total_detections", 0)
    blk   = dfn.get("total_blocks", 0)
    fp    = dfn.get("total_fp_est", 0)
    adapt = dfn.get("adaptation_count", 0)

    left  = f"  THREAT: {colored(f'{alert:10s}', acolor + C.BOLD)}  " \
            f"Score: {colored(f'{threat:.3f}', acolor)}  " \
            f"{bar(threat, 1.0, 20, acolor)}"
    right = (f"Det:{colored(str(det), C.GREEN)}  "
             f"Blk:{colored(str(blk), C.RED)}  "
             f"FP~:{colored(str(fp), C.YELLOW)}  "
             f"Adapt:{colored(str(adapt), C.CYAN)}")
    print(left + "  " + right)
    print(colored("─" * cols, C.DIM))


def render_two_column(atk: dict, dfn: dict, cols: int):
    """Render attacker and defender stats side by side."""
    half = (cols - 3) // 2

    # Stage info
    stage = atk.get("current_stage", 0)
    sname = STAGE_NAMES.get(stage, "?")
    scol  = STAGE_COLORS.get(stage, C.RESET)

    # Attacker column content
    total  = max(atk.get("total_attacks", 0), 1)
    succ   = atk.get("total_success", 0)
    blkd   = atk.get("total_blocked", 0)
    srate  = succ / total
    atkeps = atk.get("epsilon", 0)
    atkpps = atk.get("pps", 0)
    _atk_kb    = atk.get("pkt_bytes", 0) // 1024
    _atk_ema   = atk.get("timing_ema", 0)
    _atk_evas  = atk.get("evasion_mode", False)
    _atk_cumul = atk.get("cumulative_reward", 0)
    _atk_rh    = atk.get("reward_history", [])
    atk_lines = [
        colored("⚔  ATTACKER — Q-Learning + UCB1", C.LRED),
        "",
        f"  Stage:    {colored(f'{stage} ({sname})', scol + C.BOLD)}   "
        f"Campaign #{atk.get('campaign_num', 0)}",
        f"  Attacks:  {colored(str(total), C.BOLD)}  "
        f"✓{colored(str(succ), C.GREEN)}  "
        f"✗{colored(str(blkd), C.RED)}",
        f"  Succ Rate:{bar(srate, 1.0, 16, C.GREEN)}  "
        f"{colored(f'{srate:.1%}', C.GREEN)}",
        f"  Epsilon:  {bar(atkeps, 0.5, 16, C.YELLOW)}  "
        f"{colored(f'{atkeps:.4f}', C.YELLOW)}",
        f"  PPS:      {colored(f'{atkpps:.0f}', C.CYAN)}  "
        f"Bytes: {colored(f'{_atk_kb:.0f}KB', C.CYAN)}",
        f"  Timing:   {colored(f'{_atk_ema:.2f}s EMA', C.MAGENTA)}  "
        f"Evasion: {colored(str(_atk_evas), C.LRED if _atk_evas else C.DIM)}",
        f"  Attack:   {colored(atk.get('current_attack','─')[:30], C.BOLD)}",
        f"  Best Q:   {colored(atk.get('best_attack','─')[:28], C.GREEN)}",
        f"  Reward:   {sparkline(atk.get('reward_history',[]), 24, C.RED)}",
        f"  Cumul:    {colored(f'{_atk_cumul:+.2f}', C.LRED)}",
    ]

    # Defender column content
    dfneps  = dfn.get("dqn_epsilon", 0)
    updates = dfn.get("dqn_updates", 0)
    loss    = dfn.get("dqn_loss", 0)
    cur_act = dfn.get("current_action", "─")
    tr      = dfn.get("traffic_rate", 0)
    base    = dfn.get("baseline_rate", 0)

    _dfn_portdiv = dfn.get("port_diversity", 0)
    _dfn_rep_list = list(dfn.get("reputation", {}).values())
    _dfn_rep = _dfn_rep_list[0] if _dfn_rep_list else 0.0
    _dfn_icmp = dfn.get("icmp_ratio", 0)
    _dfn_udp  = dfn.get("udp_ratio", 0)
    _dfn_syn  = dfn.get("syn_ratio", 0)
    _dfn_cumul = dfn.get("cumul_reward", 0)
    _dfn_rh   = dfn.get("reward_history", [])
    dfn_lines = [
        colored("🛡  DEFENDER — Deep Q-Network", C.LGREEN),
        "",
        f"  Updates:  {colored(str(updates), C.BOLD)}  "
        f"Loss: {colored(f'{loss:.5f}', C.CYAN)}",
        f"  Epsilon:  {bar(dfneps, 0.4, 16, C.LGREEN)}  "
        f"{colored(f'{dfneps:.4f}', C.LGREEN)}",
        f"  Action:   {colored(cur_act[:30], C.BOLD)}",
        f"  Traffic:  {colored(f'{tr:.1f}/s', C.CYAN)}  "
        f"Base: {colored(f'{base:.2f}/s', C.DIM)}",
        f"  Ratio:    {bar(tr, max(base*15,0.1), 16, C.YELLOW)}",
        f"  Port Div: {colored(f'{_dfn_portdiv:.3f}', C.MAGENTA)}  "
        f"Rep: {colored(f'{_dfn_rep:.2f}' if _dfn_rep_list else '─', C.RED)}",
        f"  ICMP/UDP: {colored(f'{_dfn_icmp:.2f}', C.YELLOW)}/"
        f"{colored(f'{_dfn_udp:.2f}', C.CYAN)}  "
        f"SYN: {colored(f'{_dfn_syn:.2f}', C.RED)}",
        f"  Consec:   {colored(str(dfn.get('consecutive_high',0)), C.RED)} high windows  "
        f"Max: {colored(str(dfn.get('max_consecutive',0)), C.LRED)}",
        f"  Reward:   {sparkline(dfn.get('reward_history',[]), 24, C.GREEN)}",
        f"  Cumul:    {colored(f'{_dfn_cumul:+.2f}', C.LGREEN)}",
    ]

    # Pad both to same length
    max_lines = max(len(atk_lines), len(dfn_lines))
    atk_lines += [""] * (max_lines - len(atk_lines))
    dfn_lines += [""] * (max_lines - len(dfn_lines))

    print(colored("┌" + "─"*half + "┬" + "─"*half + "┐", C.DIM))
    for a, d in zip(atk_lines, dfn_lines):
        left_str  = padl(a, half)
        right_str = padl(d, half)
        print(colored("│", C.DIM) + left_str + colored("│", C.DIM) + right_str + colored("│", C.DIM))
    print(colored("└" + "─"*half + "┴" + "─"*half + "┘", C.DIM))


def render_q_tables(atk: dict, dfn: dict, cols: int):
    """Render Q-value tables for attacker and DQN policy for defender."""
    half = (cols - 3) // 2

    q_atk = atk.get("q_table", {})
    top_atk = sorted(q_atk.items(), key=lambda x: x[1], reverse=True)[:6]

    policy = dfn.get("dqn_policy", {})
    high_threat = policy.get("high_threat", {})
    ddos_state  = policy.get("ddos_state", {})

    print(colored("┌" + "─"*half + "┬" + "─"*half + "┐", C.DIM))
    print(colored("│", C.DIM) +
          padl(colored("  Attack Q-Values (UCB1 Policy)", C.LRED), half) +
          colored("│", C.DIM) +
          padl(colored("  Defender Policy (High-Threat State)", C.LGREEN), half) +
          colored("│", C.DIM))
    print(colored("│", C.DIM) +
          colored("─"*half, C.DIM) +
          colored("│", C.DIM) +
          colored("─"*half, C.DIM) +
          colored("│", C.DIM))

    # Build rows
    dfn_policy_items = []
    if isinstance(high_threat, dict):
        q_vals = high_threat.get("q_values", {})
        sorted_q = sorted(q_vals.items(), key=lambda x: x[1], reverse=True)[:6]
        for action, qv in sorted_q:
            col = C.GREEN if qv > 0 else C.DIM
            dfn_policy_items.append(
                f"  {colored(f'{action:<22}', C.RESET)} "
                f"{colored(f'{qv:+.3f}', col)}"
            )

    max_rows = max(len(top_atk), len(dfn_policy_items), 1)
    for i in range(max_rows):
        if i < len(top_atk):
            name, val = top_atk[i]
            vcol = C.GREEN if val > 0 else (C.RED if val < 0 else C.DIM)
            atk_str = (f"  {colored(f'{name:<28}', C.RESET)} "
                       f"{colored(f'{val:+.3f}', vcol)} "
                       f"{bar(max(val,-1)+1, 2.0, 8, vcol)}")
        else:
            atk_str = ""

        dfn_str = dfn_policy_items[i] if i < len(dfn_policy_items) else ""

        print(colored("│", C.DIM) +
              padl(atk_str, half) +
              colored("│", C.DIM) +
              padl(dfn_str, half) +
              colored("│", C.DIM))

    print(colored("└" + "─"*half + "┴" + "─"*half + "┘", C.DIM))


def render_stage_progression(atk: dict, cols: int):
    """Render campaign stage progression bars."""
    sa = atk.get("stage_attempts", [0,0,0,0])
    ss = atk.get("stage_success",  [0,0,0,0])
    current = atk.get("current_stage", 0)

    print(colored("  Campaign Stage Progression:", C.BOLD))
    for i in range(4):
        att = sa[i] if i < len(sa) else 0
        suc = ss[i] if i < len(ss) else 0
        r   = suc / max(att, 1)
        scol = STAGE_COLORS.get(i, C.RESET)
        marker = colored(" ◄ ACTIVE", C.YELLOW + C.BOLD) if i == current else ""
        barcol = C.GREEN if r > 0.5 else (C.YELLOW if r > 0.25 else C.RED)
        print(f"  {colored(STAGE_NAMES[i]+':', scol):<20} "
              f"{bar(r, 1.0, 20, barcol)} "
              f"{colored(f'{r:.0%}', barcol)} "
              f"({suc}/{att}){marker}")


def render_rules_and_actions(dfn: dict, cols: int):
    """Render recent nftables actions and defender action distribution."""
    half = (cols - 3) // 2

    rules = dfn.get("rules_log", [])[-6:]
    action_stats = dfn.get("dqn_action_stats", {})

    print(colored("┌" + "─"*half + "┬" + "─"*half + "┐", C.DIM))
    print(colored("│", C.DIM) +
          padl(colored("  Recent Defender Actions", C.LGREEN), half) +
          colored("│", C.DIM) +
          padl(colored("  Action Distribution (DQN)", C.LGREEN), half) +
          colored("│", C.DIM))
    print(colored("│", C.DIM) +
          colored("─"*half, C.DIM) +
          colored("│", C.DIM) +
          colored("─"*half, C.DIM) +
          colored("│", C.DIM))

    # Action distribution sorted by count
    total_acts = sum(v.get("count", 0) for v in action_stats.values())
    total_acts = max(total_acts, 1)
    sorted_acts = sorted(action_stats.items(),
                         key=lambda x: x[1].get("count", 0), reverse=True)[:6]

    max_rows = max(len(rules), len(sorted_acts))
    for i in range(max_rows):
        # Rules column
        if i < len(rules):
            r = rules[-(i+1)]  # most recent first
            rcol = (C.LRED   if "BLOCK"    in r else
                    C.YELLOW if "RATE"     in r else
                    C.CYAN   if "ALERT"    in r else
                    C.DIM    if "EXPIRED"  in r else C.RESET)
            rule_str = f"  {colored(r[:half-4], rcol)}"
        else:
            rule_str = ""

        # Action distribution column
        if i < len(sorted_acts):
            act, stats = sorted_acts[i]
            cnt = stats.get("count", 0)
            avg_r = stats.get("avg_reward", 0)
            pct = cnt / total_acts
            acol = C.GREEN if avg_r > 0 else C.RED
            act_str = (f"  {colored(f'{act:<20}', C.RESET)} "
                       f"{bar(pct, 1.0, 10, acol)} "
                       f"{colored(f'{cnt:4d}', C.BOLD)}")
        else:
            act_str = ""

        print(colored("│", C.DIM) +
              padl(rule_str, half) +
              colored("│", C.DIM) +
              padl(act_str, half) +
              colored("│", C.DIM))

    print(colored("└" + "─"*half + "┴" + "─"*half + "┘", C.DIM))


def render_process_status(sup: Supervisor, cols: int):
    """Render process health status."""
    def status_str(proc: ManagedProcess) -> str:
        if proc.is_alive():
            return colored("● RUNNING", C.GREEN)
        elif proc.status == "stopped":
            return colored("○ STOPPED", C.DIM)
        else:
            return colored("✗ DEAD", C.LRED)

    total_elapsed = int(time.time() - sup.start_time)
    h = total_elapsed // 3600
    m = (total_elapsed % 3600) // 60
    s = total_elapsed % 60
    uptime = f"{h}h{m:02d}m{s:02d}s" if h > 0 else f"{m}m{s:02d}s"

    print(colored("  Process Status:", C.BOLD))
    print(f"  Defender: {status_str(sup.defender)}  "
          f"PID:{colored(str(sup.defender.pid or '─'), C.CYAN)}  "
          f"Up:{colored(sup.defender.uptime(), C.GREEN)}  "
          f"Restarts:{colored(str(sup.defender.restarts), C.YELLOW)}")
    print(f"  Attacker: {status_str(sup.attacker)}  "
          f"PID:{colored(str(sup.attacker.pid or '─'), C.CYAN)}  "
          f"Up:{colored(sup.attacker.uptime(), C.GREEN)}  "
          f"Restarts:{colored(str(sup.attacker.restarts), C.YELLOW)}")
    print(f"  Session:  {colored(uptime, C.CYAN)}  "
          f"Episode: {colored(str(sup.episode), C.BOLD)}  "
          f"Logs: {colored(SIMDIR, C.DIM)}")

    # Recent supervisor events
    recent_events = list(sup.event_log)[-3:]
    for evt in recent_events:
        level = "WARN" if "WARN" in evt else "ERROR" if "ERROR" in evt else "INFO"
        ecol = C.YELLOW if level=="WARN" else C.LRED if level=="ERROR" else C.DIM
        print(f"  {colored(evt, ecol)}")


def render_live_engagement(atk: dict, dfn: dict, cols: int):
    """Render the current attack vs defense matchup."""
    cur_atk = atk.get("current_attack", "─")
    cur_dfn = dfn.get("current_action", "─")
    atk_r   = atk.get("last_reward", 0)
    dfn_r   = dfn.get("last_reward", 0)
    atk_col = C.GREEN if atk_r >= 0 else C.RED
    dfn_col = C.GREEN if dfn_r >= 0 else C.RED

    print(colored("─" * cols, C.DIM))
    print(f"  {colored('LIVE:', C.BOLD)}  "
          f"{colored('⚔', C.LRED)} {colored(cur_atk[:35], C.RED + C.BOLD)}  "
          f"{colored('vs', C.DIM)}  "
          f"{colored('🛡', C.LGREEN)} {colored(cur_dfn[:30], C.GREEN + C.BOLD)}")
    print(f"          "
          f"Q-reward: {colored(f'{atk_r:+.3f}', atk_col)}  "
          f"{'·'*20}  "
          f"Q-reward: {colored(f'{dfn_r:+.3f}', dfn_col)}")
    print(colored("─" * cols, C.DIM))


def render_dashboard(sup: Supervisor, shared: dict):
    """Full dashboard render."""
    cols, rows = get_terminal_size()
    atk = shared.get("attacker", {})
    dfn = shared.get("defender", {})

    clear_screen()
    render_header(cols)
    render_alert_banner(dfn, cols)
    render_two_column(atk, dfn, cols)
    render_q_tables(atk, dfn, cols)
    print()
    render_stage_progression(atk, cols)
    print()
    render_rules_and_actions(dfn, cols)
    print()
    render_live_engagement(atk, dfn, cols)
    render_process_status(sup, cols)
    print()
    print(colored(f"  Ctrl+C to stop  │  Dashboard: http://10.74.6.16:8888  │  "
                  f"Refresh: {DASH_INTERVAL}s", C.DIM))


def render_waiting(sup: Supervisor, message: str, elapsed: float,
                   total: float):
    """Render a waiting/countdown screen."""
    cols, _ = get_terminal_size()
    clear_screen()
    print(colored("═" * cols, C.CYAN))
    print(colored("  ADAPTIVE ATTACK/DEFENSE SIMULATION v2".center(cols), C.LCYAN))
    print(colored("═" * cols, C.CYAN))
    print()

    pct = elapsed / max(total, 1)
    print(f"  {colored(message, C.BOLD)}")
    print(f"  {bar(pct, 1.0, 40, C.GREEN)} "
          f"{colored(f'{int(elapsed):.0f}/{int(total):.0f}s', C.CYAN)}")
    print()

    events = list(sup.event_log)[-8:]
    for evt in events:
        ecol = (C.YELLOW if "WARN" in evt else
                C.LRED  if "ERROR" in evt else C.DIM)
        print(f"  {colored(evt, ecol)}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def check_prerequisites() -> bool:
    """Verify namespaces exist and required files are present."""
    errors = []

    # Check namespaces
    result = subprocess.run(["sudo", "ip", "netns", "list"],
                           capture_output=True, text=True)
    if "left" not in result.stdout or "right" not in result.stdout:
        errors.append("Network namespaces 'left' and 'right' not found. "
                      "Run gre-ipsec-setup.sh first.")

    # Check required files
    for path in [DEFENDER_PY, ATTACKER_PY,
                 "/home/kaexb/packet_engine.py"]:
        if not os.path.exists(path):
            errors.append(f"Required file not found: {path}")

    if errors:
        print(colored("PREREQUISITES FAILED:", C.LRED))
        for e in errors:
            print(f"  {colored('✗', C.LRED)} {e}")
        return False

    return True


def main():
    if not check_prerequisites():
        sys.exit(1)

    sup = Supervisor()
    stop_event = threading.Event()

    def handle_signal(sig, frame):
        print()
        sup.add_event("SIGINT received, shutting down...", "WARN")
        stop_event.set()
        sup.cleanup()
        show_cursor()
        print(colored(f"\nSimulation stopped. All logs in {SIMDIR}", C.GREEN))
        sys.exit(0)

    signal.signal(signal.SIGINT,  handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    hide_cursor()

    # Clear old state files for fresh start
    sup.add_event("Clearing previous simulation state...")
    for fname in ["attacker_state.json", "defender_state.json",
                  "shared_state.json"]:
        fpath = os.path.join(SIMDIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
            sup.add_event(f"Cleared {fname}")

    # Start defender
    if not sup.start_defender():
        show_cursor()
        print(colored("Failed to start defender!", C.LRED))
        sys.exit(1)

    # Wait for defender baseline
    baseline_start = time.time()
    sup.add_event(f"Waiting {BASELINE_WAIT}s for defender baseline...")

    while time.time() - baseline_start < BASELINE_WAIT:
        elapsed = time.time() - baseline_start
        render_waiting(sup, "Defender establishing traffic baseline...",
                       elapsed, BASELINE_WAIT)

        # Health check
        if not sup.defender.is_alive():
            sup.add_event("Defender died during baseline!", "ERROR")
            sup.defender.restart()

        time.sleep(1)

    # Start attacker
    if not sup.start_attacker():
        show_cursor()
        print(colored("Failed to start attacker!", C.LRED))
        sys.exit(1)

    sup.add_event("Simulation fully started. Dashboard active.")

    # ── MAIN LOOP ────────────────────────────────────────────────────────────
    last_health_check = time.time()
    last_save         = time.time()

    while not stop_event.is_set():
        # Read shared state
        shared = sup.read_shared()

        # Update history for sparklines
        if shared:
            sup.update_history(shared)

        # Render dashboard
        try:
            render_dashboard(sup, shared)
        except Exception as e:
            # Don't crash dashboard on render errors
            pass

        # Health check every 10 seconds
        if time.time() - last_health_check > 10:
            sup.health_check()
            last_health_check = time.time()

        time.sleep(DASH_INTERVAL)

    # Should not reach here normally (handled by signal)
    sup.cleanup()
    show_cursor()


if __name__ == "__main__":
    main()