#!/usr/bin/env python3
"""
server.py
HTTP server for the attack/defense simulation dashboard.
Serves dashboard.html and provides a /state JSON endpoint.
Also serves /logs endpoint for raw log streaming,
and /history endpoint for JSONL history files.
"""

import http.server
import socketserver
import json
import os
import sys
import time
import threading
from datetime import datetime

# ── Config ───────────────────────────────────────────────────────────────────
PORT         = 8888
HOST         = "0.0.0.0"
SIMDIR       = "/home/kaexb/sim"
DASH_FILE    = "/home/kaexb/dashboard.html"
SHARED_FILE  = f"{SIMDIR}/shared_state.json"
ATK_LOG      = f"{SIMDIR}/attacker.log"
DFN_LOG      = f"{SIMDIR}/defender.log"
ATK_HISTORY  = f"{SIMDIR}/attacker_history.jsonl"
DFN_HISTORY  = f"{SIMDIR}/defender_history.jsonl"
SUPERVISOR_LOG = f"{SIMDIR}/supervisor.log"

# ── CORS headers ─────────────────────────────────────────────────────────────
CORS_HEADERS = {
    "Access-Control-Allow-Origin":  "*",
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
}


def read_file_safe(path: str, default: str = "") -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except:
        return default


def read_json_safe(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except:
        return {}


def read_jsonl_tail(path: str, n: int = 100) -> list:
    """Read last N lines from a JSONL file."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        result = []
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    result.append(json.loads(line))
                except:
                    pass
        return result
    except:
        return []


def read_log_tail(path: str, n: int = 50) -> list:
    """Read last N lines from a log file."""
    try:
        with open(path, "r") as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except:
        return []


def build_state_response() -> dict:
    """
    Build complete state response for the /state endpoint.
    Combines shared state with computed summary statistics.
    """
    state = read_json_safe(SHARED_FILE)

    atk = state.get("attacker", {})
    dfn = state.get("defender", {})

    # Compute some derived stats for the dashboard
    total_atk = max(atk.get("total_attacks", 0), 1)
    success_rate = atk.get("total_success", 0) / total_atk

    dfn_total  = max(dfn.get("total_detections", 0) +
                     dfn.get("total_blocks", 0), 1)

    # Q-table summary: top 5 by Q-value and bottom 3
    q_table = atk.get("q_table", {})
    if q_table:
        sorted_q = sorted(q_table.items(), key=lambda x: x[1], reverse=True)
        state["q_summary"] = {
            "top5":    [(n, round(v, 3)) for n, v in sorted_q[:5]],
            "bottom3": [(n, round(v, 3)) for n, v in sorted_q[-3:]],
            "mean":    round(sum(q_table.values()) / len(q_table), 3),
            "max":     round(sorted_q[0][1], 3) if sorted_q else 0,
            "min":     round(sorted_q[-1][1], 3) if sorted_q else 0,
        }

    # DQN convergence indicator
    dqn_loss_hist = dfn.get("dqn_loss", 0)
    state["meta"] = {
        "timestamp":        datetime.now().isoformat(),
        "success_rate":     round(success_rate, 4),
        "simulation_active": (
            os.path.exists(SHARED_FILE) and
            os.path.getmtime(SHARED_FILE) > time.time() - 15
        ),
    }

    return state


class SimHandler(http.server.BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        # Only log errors, suppress normal access log
        if args and len(args) >= 2:
            code = str(args[1])
            if code.startswith("4") or code.startswith("5"):
                super().log_message(format, *args)

    def send_cors_headers(self):
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)

    def send_json(self, data: dict, code: int = 200):
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_text(self, text: str, code: int = 200):
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, html: str, code: int = 200):
        body = html.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]  # strip query string

        # ── Main dashboard ───────────────────────────────────────────────────
        if path == "/" or path == "/index.html":
            html = read_file_safe(DASH_FILE, self._not_found_html())
            self.send_html(html)

        # ── Simulation state JSON ────────────────────────────────────────────
        elif path == "/state":
            state = build_state_response()
            self.send_json(state)

        # ── Attacker log tail ────────────────────────────────────────────────
        elif path == "/logs/attacker":
            lines = read_log_tail(ATK_LOG, 100)
            self.send_json({"lines": lines, "file": ATK_LOG})

        # ── Defender log tail ────────────────────────────────────────────────
        elif path == "/logs/defender":
            lines = read_log_tail(DFN_LOG, 100)
            self.send_json({"lines": lines, "file": DFN_LOG})

        # ── Supervisor log tail ──────────────────────────────────────────────
        elif path == "/logs/supervisor":
            lines = read_log_tail(SUPERVISOR_LOG, 50)
            self.send_json({"lines": lines, "file": SUPERVISOR_LOG})

        # ── Attacker history JSONL ───────────────────────────────────────────
        elif path == "/history/attacker":
            events = read_jsonl_tail(ATK_HISTORY, 200)
            self.send_json({
                "events": events,
                "count":  len(events),
                "file":   ATK_HISTORY,
            })

        # ── Defender history JSONL ───────────────────────────────────────────
        elif path == "/history/defender":
            events = read_jsonl_tail(DFN_HISTORY, 200)
            self.send_json({
                "events": events,
                "count":  len(events),
                "file":   DFN_HISTORY,
            })

        # ── Combined history for charts ──────────────────────────────────────
        elif path == "/history/combined":
            atk_h = read_jsonl_tail(ATK_HISTORY, 100)
            dfn_h = read_jsonl_tail(DFN_HISTORY, 100)
            self.send_json({
                "attacker": atk_h,
                "defender": dfn_h,
            })

        # ── Health check ─────────────────────────────────────────────────────
        elif path == "/health":
            shared_age = 0
            if os.path.exists(SHARED_FILE):
                shared_age = int(time.time() - os.path.getmtime(SHARED_FILE))
            self.send_json({
                "status":       "ok",
                "server_time":  datetime.now().isoformat(),
                "shared_age_s": shared_age,
                "sim_active":   shared_age < 15,
            })

        # ── Simple status text page ──────────────────────────────────────────
        elif path == "/status":
            state = read_json_safe(SHARED_FILE)
            atk = state.get("attacker", {})
            dfn = state.get("defender", {})
            lines = [
                f"Simulation Status @ {datetime.now().strftime('%H:%M:%S')}",
                "=" * 50,
                f"Attacker:",
                f"  Total attacks:  {atk.get('total_attacks', 0)}",
                f"  Success rate:   {atk.get('success_rate', 0):.1%}",
                f"  Current stage:  {atk.get('current_stage', '─')}",
                f"  Current attack: {atk.get('current_attack', '─')}",
                f"  Epsilon:        {atk.get('epsilon', 0):.4f}",
                f"",
                f"Defender:",
                f"  Alert level:    {dfn.get('alert_label', 'NORMAL')}",
                f"  Threat score:   {dfn.get('threat_score', 0):.3f}",
                f"  Total blocks:   {dfn.get('total_blocks', 0)}",
                f"  DQN updates:    {dfn.get('dqn_updates', 0)}",
                f"  Current action: {dfn.get('current_action', '─')}",
                f"  DQN loss:       {dfn.get('dqn_loss', 0):.5f}",
            ]
            self.send_text("\n".join(lines))

        # ── 404 ──────────────────────────────────────────────────────────────
        else:
            self.send_json({"error": "not found", "path": path}, code=404)

    def _not_found_html(self) -> str:
        return """<!DOCTYPE html>
<html><head><title>Sim Dashboard</title>
<style>body{background:#0d1117;color:#e6edf3;font-family:monospace;
text-align:center;padding:50px}</style></head>
<body>
<h1 style="color:#f59e0b">⚠ dashboard.html not found</h1>
<p>Copy dashboard.html to /home/kaexb/dashboard.html</p>
<p>API endpoints: <a href="/state" style="color:#06b6d4">/state</a>
   <a href="/health" style="color:#06b6d4">/health</a>
   <a href="/status" style="color:#06b6d4">/status</a></p>
</body></html>"""


class ThreadedHTTPServer(socketserver.ThreadingMixIn,
                         http.server.HTTPServer):
    """Handle each request in a separate thread."""
    daemon_threads = True
    allow_reuse_address = True


def print_startup_banner(port: int):
    vm_ip = "10.74.6.16"
    print("=" * 60)
    print("  Simulation Dashboard Server")
    print("=" * 60)
    print(f"  Listening: http://0.0.0.0:{port}")
    print(f"  Access:    http://{vm_ip}:{port}")
    print()
    print("  Endpoints:")
    print(f"    /              - Main dashboard")
    print(f"    /state         - Full simulation state JSON")
    print(f"    /health        - Server health check")
    print(f"    /status        - Plain text status")
    print(f"    /logs/attacker - Last 100 attacker log lines")
    print(f"    /logs/defender - Last 100 defender log lines")
    print(f"    /history/attacker - Attacker event history JSONL")
    print(f"    /history/defender - Defender event history JSONL")
    print(f"    /history/combined - Both histories")
    print()
    print(f"  Sim files: {SIMDIR}")
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print()


def main():
    os.makedirs(SIMDIR, exist_ok=True)
    print_startup_banner(PORT)

    try:
        server = ThreadedHTTPServer((HOST, PORT), SimHandler)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"Port {PORT} already in use. Kill the old process first:")
            print(f"  sudo pkill -f server.py")
        else:
            print(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()