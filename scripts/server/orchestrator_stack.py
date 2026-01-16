#!/usr/bin/env python3
"""Unified orchestrator stack launcher.

Launches all models + orchestrator with granular reload support.

Usage:
    orchestrator_stack.py start [--hot-only] [--include-warm ROLE...] [--dev]
    orchestrator_stack.py stop [--all | COMPONENT...]
    orchestrator_stack.py reload COMPONENT...
    orchestrator_stack.py status

Examples:
    # Start HOT models only
    ./orchestrator_stack.py start --hot-only

    # Start with warm architect
    ./orchestrator_stack.py start --include-warm architect_general

    # Dev mode (single 0.5B model)
    ./orchestrator_stack.py start --dev

    # Reload orchestrator API after code changes
    ./orchestrator_stack.py reload orchestrator

    # Check status
    ./orchestrator_stack.py status
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.registry_loader import RegistryLoader

# =============================================================================
# Configuration
# =============================================================================

STATE_FILE = Path("/mnt/raid0/llm/claude/logs/orchestrator_state.json")
LLAMA_SERVER = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-server")
LOG_DIR = Path("/mnt/raid0/llm/claude/logs")

# Port assignments by role
PORT_MAP = {
    "frontdoor": 8080,
    "coder_primary": 8080,  # Shares with frontdoor (same model)
    "coder_escalation": 8081,
    "worker_general": 8082,
    "worker_math": 8082,  # Shares with worker_general
    "worker_vision": 8082,  # Shares with worker_general
    "architect_general": 8083,
    "architect_coding": 8084,
    "ingest_long_context": 8085,
    "orchestrator": 8000,
}

# HOT roles (always started)
HOT_ROLES = {"frontdoor", "coder_escalation", "worker_general"}

# Servers to start (unique ports only)
HOT_SERVERS = [
    {"port": 8080, "roles": ["frontdoor", "coder_primary"]},
    {"port": 8081, "roles": ["coder_escalation"]},
    {"port": 8082, "roles": ["worker_general", "worker_math", "worker_vision"]},
]

WARM_SERVERS = [
    {"port": 8083, "roles": ["architect_general"]},
    {"port": 8084, "roles": ["architect_coding"]},
    {"port": 8085, "roles": ["ingest_long_context"]},
]

DEV_MODEL = "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"
DEV_MODEL_PATH = "/mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"

# =============================================================================
# State Management
# =============================================================================


@dataclass
class ProcessInfo:
    """Information about a running process."""
    role: str
    pid: int
    port: int
    started_at: str
    model_path: str
    log_file: str


def load_state() -> dict[str, ProcessInfo]:
    """Load state from file."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return {k: ProcessInfo(**v) for k, v in data.items()}
    except (json.JSONDecodeError, TypeError):
        return {}


def save_state(state: dict[str, ProcessInfo]) -> None:
    """Save state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({k: asdict(v) for k, v in state.items()}, f, indent=2)


# =============================================================================
# Process Management
# =============================================================================


def check_free_memory() -> int:
    """Return free memory in GB."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb // (1024 * 1024)
    return 0


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process(pid: int, timeout: int = 5) -> bool:
    """Kill a process gracefully, then forcefully."""
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(timeout):
            time.sleep(1)
            try:
                os.kill(pid, 0)  # Check if still alive
            except ProcessLookupError:
                return True
        # Force kill
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        print(f"  [!] Permission denied killing PID {pid}")
        return False


def wait_for_health(port: int, timeout: int = 120) -> bool:
    """Wait for server health endpoint."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    return False


# =============================================================================
# Server Launching
# =============================================================================


def build_server_command(
    role_config: Any,
    port: int,
    dev_mode: bool = False,
) -> list[str]:
    """Build llama-server command from role config."""
    if dev_mode:
        return [
            str(LLAMA_SERVER),
            "-m", DEV_MODEL_PATH,
            "--host", "0.0.0.0",
            "--port", str(port),
            "-np", "4",
            "-c", "4096",
            "-t", "16",
        ]

    model_path = role_config.model.full_path
    accel = role_config.acceleration

    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "--host", "0.0.0.0",
        "--port", str(port),
        "-np", "4",  # Parallel slots
        "-c", "8192",  # Context size
        "-t", "96",  # Threads
    ]

    # Add acceleration based on type
    if accel.type == "moe_expert_reduction" and accel.experts:
        cmd.extend([
            "--override-kv",
            f"{accel.override_key}=int:{accel.experts}",
        ])
    elif accel.type == "speculative_decoding" and accel.draft_role:
        # Get draft model path from registry
        registry = RegistryLoader()
        draft_config = registry.get_role(accel.draft_role)
        if draft_config:
            cmd.extend([
                "-md", draft_config.model.full_path,
                "--draft-max", str(accel.k or 16),
            ])

    return cmd


def start_server(
    port: int,
    roles: list[str],
    registry: RegistryLoader,
    dev_mode: bool = False,
) -> ProcessInfo | None:
    """Start a llama-server for the given roles."""
    # Use first role's config for the server
    primary_role = roles[0]
    role_config = registry.get_role(primary_role)

    if not role_config and not dev_mode:
        print(f"  [!] Role {primary_role} not found in registry")
        return None

    log_file = LOG_DIR / f"llama-server-{port}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = build_server_command(role_config, port, dev_mode)

    model_name = DEV_MODEL if dev_mode else role_config.model.name

    print(f"  Starting port {port}: {model_name}")
    print(f"    Roles: {', '.join(roles)}")
    print(f"    Command: {' '.join(cmd[:5])}...")

    # Start process
    with open(log_file, "w") as log:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        proc = subprocess.Popen(
            ["numactl", "--interleave=all"] + cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f"    PID: {proc.pid}")

    # Wait for health
    print(f"    Waiting for health...")
    if wait_for_health(port, timeout=180):
        print(f"    [OK] Server ready")
        return ProcessInfo(
            role=primary_role,
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path=DEV_MODEL_PATH if dev_mode else role_config.model.full_path,
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] Server did not become healthy")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def start_orchestrator() -> ProcessInfo | None:
    """Start the orchestrator API."""
    log_file = LOG_DIR / "orchestrator.log"

    print("  Starting orchestrator API on port 8000")

    # Set environment
    env = os.environ.copy()
    env["HF_HOME"] = "/mnt/raid0/llm/cache/huggingface"
    env["TMPDIR"] = "/mnt/raid0/llm/tmp"

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.api:app",
                "--host", "0.0.0.0",
                "--port", "8000",
            ],
            cwd="/mnt/raid0/llm/claude",
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f"    PID: {proc.pid}")
    print(f"    Waiting for health...")

    if wait_for_health(8000, timeout=60):
        print(f"    [OK] Orchestrator ready")
        return ProcessInfo(
            role="orchestrator",
            pid=proc.pid,
            port=8000,
            started_at=datetime.now().isoformat(),
            model_path="uvicorn",
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] Orchestrator did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


# =============================================================================
# Commands
# =============================================================================


def cmd_start(args: argparse.Namespace) -> int:
    """Start the orchestrator stack."""
    print("=" * 60)
    print("ORCHESTRATOR STACK STARTUP")
    print("=" * 60)
    print()

    # Check memory
    free_gb = check_free_memory()
    print(f"[i] Free memory: {free_gb} GB")
    if free_gb < 100 and not args.dev:
        print("[!] WARNING: Less than 100GB free. Consider --dev mode.")
        if input("Continue? (y/N) ").lower() != "y":
            return 1
    print()

    # Load registry
    registry = RegistryLoader()
    state: dict[str, ProcessInfo] = {}

    # Kill existing processes on target ports
    print("[1] Cleaning up existing processes...")
    for server in HOT_SERVERS + WARM_SERVERS:
        port = server["port"]
        if is_port_in_use(port):
            print(f"  Port {port} in use, attempting cleanup...")
            # Find PID from lsof
            try:
                result = subprocess.run(
                    ["lsof", "-t", f"-i:{port}"],
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    for pid_str in result.stdout.strip().split("\n"):
                        pid = int(pid_str)
                        kill_process(pid)
            except Exception as e:
                print(f"  [!] Error cleaning port {port}: {e}")
    print()

    # Determine which servers to start
    servers_to_start = []

    if args.dev:
        print("[2] Starting in DEV mode (single 0.5B model)...")
        servers_to_start = [{"port": 8080, "roles": ["dev"]}]
    else:
        print("[2] Starting HOT servers...")
        servers_to_start = HOT_SERVERS.copy()

        # Add warm servers if requested
        if args.include_warm:
            for warm_server in WARM_SERVERS:
                for role in warm_server["roles"]:
                    if role in args.include_warm:
                        servers_to_start.append(warm_server)
                        print(f"  Including WARM server: port {warm_server['port']} ({role})")
                        break

    print()

    # Start servers sequentially
    print("[3] Starting llama-servers...")
    for i, server in enumerate(servers_to_start):
        port = server["port"]
        roles = server["roles"]

        info = start_server(port, roles, registry, args.dev)
        if info:
            state[f"server_{port}"] = info
            # Also map all roles to this server
            for role in roles:
                if role not in state:
                    state[role] = info
        else:
            print(f"  [!] Failed to start server on port {port}")
            if not args.dev:
                # In production, this is fatal
                return 1

        # Cooldown between large models
        if i < len(servers_to_start) - 1 and not args.dev:
            print("  Cooldown (30s) for tensor repack...")
            time.sleep(30)

    print()

    # Start orchestrator
    print("[4] Starting orchestrator API...")
    info = start_orchestrator()
    if info:
        state["orchestrator"] = info
    else:
        print("  [!] Failed to start orchestrator")
        return 1

    print()

    # Save state
    save_state(state)
    print(f"[i] State saved to {STATE_FILE}")
    print()

    # Final status
    print("=" * 60)
    print("STACK READY")
    print("=" * 60)
    cmd_status(args)

    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop components."""
    state = load_state()

    if not state:
        print("No running components found")
        return 0

    targets = []
    if args.all:
        targets = list(state.keys())
    elif args.components:
        targets = args.components
    else:
        print("Specify --all or component names")
        return 1

    for name in targets:
        if name in state:
            info = state[name]
            print(f"Stopping {name} (PID {info.pid})...")
            if kill_process(info.pid):
                del state[name]
                print(f"  [OK] Stopped")
            else:
                print(f"  [!] Failed to stop")
        else:
            print(f"  [?] {name} not found in state")

    save_state(state)
    return 0


def cmd_reload(args: argparse.Namespace) -> int:
    """Reload components."""
    state = load_state()
    registry = RegistryLoader()

    for component in args.components:
        print(f"Reloading {component}...")

        if component == "orchestrator":
            # Stop existing
            if "orchestrator" in state:
                kill_process(state["orchestrator"].pid)
                time.sleep(2)

            # Start new
            info = start_orchestrator()
            if info:
                state["orchestrator"] = info
            else:
                print(f"  [!] Failed to restart orchestrator")
                return 1

        elif component in PORT_MAP:
            port = PORT_MAP[component]
            key = f"server_{port}"

            # Find roles for this port
            roles = [component]
            for server in HOT_SERVERS + WARM_SERVERS:
                if server["port"] == port:
                    roles = server["roles"]
                    break

            # Stop existing
            if key in state:
                kill_process(state[key].pid)
                time.sleep(2)

            # Start new
            info = start_server(port, roles, registry, dev_mode=False)
            if info:
                state[key] = info
                for role in roles:
                    state[role] = info
            else:
                print(f"  [!] Failed to restart {component}")
                return 1

        else:
            print(f"  [?] Unknown component: {component}")

    save_state(state)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of all components."""
    state = load_state()

    if not state:
        print("No components running")
        return 0

    print()
    print(f"{'COMPONENT':<25} {'PORT':<8} {'PID':<10} {'STATUS':<10} {'MODEL'}")
    print("-" * 80)

    seen_pids = set()
    for name, info in sorted(state.items()):
        if info.pid in seen_pids:
            continue  # Skip duplicates (roles sharing servers)
        seen_pids.add(info.pid)

        # Check if process is alive
        try:
            os.kill(info.pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False

        # Check health endpoint
        healthy = False
        if alive:
            healthy = wait_for_health(info.port, timeout=3)

        status = "healthy" if healthy else ("running" if alive else "dead")
        model = Path(info.model_path).stem if info.model_path != "uvicorn" else "uvicorn"

        print(f"{name:<25} {info.port:<8} {info.pid:<10} {status:<10} {model[:30]}")

    print()
    print(f"State file: {STATE_FILE}")
    return 0


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator stack manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the stack")
    start_parser.add_argument("--hot-only", action="store_true", help="Start HOT models only")
    start_parser.add_argument("--include-warm", nargs="+", metavar="ROLE", help="Include WARM models")
    start_parser.add_argument("--dev", action="store_true", help="Dev mode (single 0.5B model)")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop components")
    stop_parser.add_argument("--all", action="store_true", help="Stop all components")
    stop_parser.add_argument("components", nargs="*", help="Components to stop")

    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload components")
    reload_parser.add_argument("components", nargs="+", help="Components to reload")

    # Status command
    subparsers.add_parser("status", help="Show status")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "reload":
        return cmd_reload(args)
    elif args.command == "status":
        return cmd_status(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
