#!/usr/bin/env python3
"""Unified orchestration CLI.

Routes to various subcommand modules:
  orch sessions ...  -> Session management
  orch run ...       -> Run tasks
  orch stack ...     -> Server stack management
  orch status        -> Quick status

Usage:
    orch sessions list [--status STATUS]
    orch sessions search QUERY
    orch sessions resume SESSION_ID
    orch run "task description"
    orch status
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> int:
    """Main entry point for orch CLI."""
    parser = argparse.ArgumentParser(
        prog="orch",
        description="Hierarchical orchestration CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Subcommands:
  sessions    Manage conversation sessions
  run         Execute orchestration tasks
  stack       Manage server stack
  status      Show system status

Examples:
  orch sessions list --status active
  orch sessions resume abc123
  orch run "Write a Python function"
  orch status
        """,
    )

    parser.add_argument(
        "subcommand",
        nargs="?",
        choices=["sessions", "run", "stack", "status", "help"],
        help="Subcommand to run",
    )

    parser.add_argument(
        "args",
        nargs="*",
        help="Arguments for subcommand",
    )

    # Parse just the first argument to determine subcommand
    args, remaining = parser.parse_known_args()

    if not args.subcommand or args.subcommand == "help":
        parser.print_help()
        return 0

    # Combine args.args and remaining (for different parsing behaviors)
    subcommand_args = (args.args or []) + remaining

    # Route to subcommand
    if args.subcommand == "sessions":
        from src.cli_sessions import main as sessions_main

        # Reconstruct sys.argv for the subcommand
        sys.argv = ["orch sessions"] + subcommand_args
        return sessions_main()

    elif args.subcommand == "run":
        from src.cli import main as cli_main

        sys.argv = ["orch run"] + subcommand_args
        return cli_main()

    elif args.subcommand == "stack":
        # Delegate to orchestrator_stack.py
        script_path = Path(__file__).parent.parent / "scripts" / "server" / "orchestrator_stack.py"
        if script_path.exists():
            cmd = [sys.executable, str(script_path)] + subcommand_args
            return subprocess.run(cmd).returncode
        else:
            print(f"Stack script not found: {script_path}", file=sys.stderr)
            return 1

    elif args.subcommand == "status":
        return cmd_status()

    return 1


def cmd_status() -> int:
    """Show quick system status."""
    import urllib.request
    import urllib.error

    print()
    print("=" * 50)
    print("ORCHESTRATOR STATUS")
    print("=" * 50)
    print()

    # Check orchestrator API
    try:
        with urllib.request.urlopen("http://localhost:8000/health", timeout=3) as resp:
            if resp.status == 200:
                print("  Orchestrator API:  \033[92m● Running\033[0m (port 8000)")
            else:
                print("  Orchestrator API:  \033[91m✗ Error\033[0m")
    except (urllib.error.URLError, TimeoutError):
        print("  Orchestrator API:  \033[90m○ Offline\033[0m")

    # Check llama-server ports
    ports = [8080, 8081, 8082, 8083, 8084, 8085]
    port_names = {
        8080: "frontdoor",
        8081: "coder_escalation",
        8082: "worker",
        8083: "architect_general",
        8084: "architect_coding",
        8085: "ingest",
    }

    for port in ports:
        name = port_names.get(port, str(port))
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as resp:
                if resp.status == 200:
                    print(f"  {name:<20} \033[92m● Running\033[0m (port {port})")
        except (urllib.error.URLError, TimeoutError):
            pass  # Don't show offline servers

    print()

    # Check session store
    try:
        from src.session import SQLiteSessionStore

        store = SQLiteSessionStore()
        sessions = store.list_sessions(limit=5)
        active = sum(1 for s in sessions if s.status.value == "active")
        total = len(sessions)

        print(f"  Sessions:          {active} active / {total} recent")
    except Exception as e:
        logger.debug("Session store error: %s", e)
        print(f"  Sessions:          \033[91m✗ Error\033[0m ({e})")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
