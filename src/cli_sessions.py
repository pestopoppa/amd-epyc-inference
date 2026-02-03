#!/usr/bin/env python3
"""Session management CLI for orchestration.

Provides commands for listing, searching, resuming, and managing sessions.
Works both via API (when orchestrator is running) and directly with SQLite.

Usage:
    python -m src.cli_sessions list [--status STATUS] [--project PROJECT]
    python -m src.cli_sessions search QUERY
    python -m src.cli_sessions show SESSION_ID
    python -m src.cli_sessions resume SESSION_ID [--output FORMAT]
    python -m src.cli_sessions archive SESSION_ID
    python -m src.cli_sessions findings SESSION_ID
    python -m src.cli_sessions delete SESSION_ID [--force]

Examples:
    # List all active sessions
    orch sessions list --status active

    # Search sessions by topic
    orch sessions search "document analysis"

    # Resume a session (shows context injection)
    orch sessions resume abc123

    # Show session details
    orch sessions show abc123 --findings --checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# =============================================================================
# Output Formatters
# =============================================================================


def format_datetime(dt_str: str) -> str:
    """Format ISO datetime for display."""
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        now = datetime.utcnow()
        delta = now - dt.replace(tzinfo=None)

        if delta.days == 0:
            if delta.seconds < 60:
                return "just now"
            elif delta.seconds < 3600:
                return f"{delta.seconds // 60}m ago"
            else:
                return f"{delta.seconds // 3600}h ago"
        elif delta.days == 1:
            return "yesterday"
        elif delta.days < 7:
            return f"{delta.days}d ago"
        else:
            return dt.strftime("%Y-%m-%d")
    except (ValueError, AttributeError):
        return dt_str[:10] if dt_str else "unknown"


def truncate(text: str | None, length: int = 40) -> str:
    """Truncate text with ellipsis."""
    if not text:
        return ""
    if len(text) <= length:
        return text
    return text[: length - 3] + "..."


def format_status_icon(status: str) -> str:
    """Get icon for status."""
    icons = {
        "active": "\033[92m●\033[0m",  # Green
        "idle": "\033[93m○\033[0m",  # Yellow
        "stale": "\033[91m◌\033[0m",  # Red
        "archived": "\033[90m◦\033[0m",  # Gray
    }
    return icons.get(status, "?")


# =============================================================================
# Direct Store Access (for offline use)
# =============================================================================


def get_store():
    """Get session store (direct access, not via API)."""
    from src.session import SQLiteSessionStore

    return SQLiteSessionStore()


# =============================================================================
# Commands
# =============================================================================


def cmd_list(args: argparse.Namespace) -> int:
    """List sessions with optional filtering."""
    store = get_store()

    # Build filter
    where = {}
    if args.status:
        where["status"] = args.status
    if args.project:
        where["project"] = args.project

    sessions = store.list_sessions(
        where=where if where else None,
        limit=args.limit,
    )

    if not sessions:
        print("No sessions found")
        return 0

    # Table header
    print()
    print(
        f"{'ST':<3} {'ID':<12} {'NAME':<25} {'PROJECT':<15} {'LAST ACTIVE':<12} {'MSGS':<5} {'TOPIC'}"
    )
    print("-" * 100)

    for s in sessions:
        status_icon = format_status_icon(s.status.value)
        session_id = s.id[:10]
        name = truncate(s.name, 23) or "(unnamed)"
        project = truncate(s.project, 13) or "-"
        last_active = format_datetime(s.last_active.isoformat())
        msgs = s.message_count
        topic = truncate(s.last_topic, 30) or "-"

        print(
            f"{status_icon:<3} {session_id:<12} {name:<25} {project:<15} {last_active:<12} {msgs:<5} {topic}"
        )

    print()
    print(f"Total: {len(sessions)} session(s)")

    if args.status is None:
        # Show status summary
        by_status = {}
        for s in sessions:
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
        parts = [f"{v} {k}" for k, v in sorted(by_status.items())]
        print(f"  ({', '.join(parts)})")

    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Search sessions by query."""
    store = get_store()

    sessions = store.search_sessions(args.query, limit=args.limit)

    if not sessions:
        print(f"No sessions found matching '{args.query}'")
        return 0

    print()
    print(f"Search results for: '{args.query}'")
    print()
    print(f"{'ST':<3} {'ID':<12} {'NAME':<25} {'SUMMARY/TOPIC'}")
    print("-" * 80)

    for s in sessions:
        status_icon = format_status_icon(s.status.value)
        session_id = s.id[:10]
        name = truncate(s.name, 23) or "(unnamed)"
        summary = truncate(s.summary or s.last_topic, 35) or "-"

        print(f"{status_icon:<3} {session_id:<12} {name:<25} {summary}")

    print()
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show detailed session information."""
    store = get_store()

    # Find session (support partial ID match)
    session = store.get_session(args.session_id)
    if not session:
        # Try partial match
        all_sessions = store.list_sessions(limit=1000)
        matches = [s for s in all_sessions if s.id.startswith(args.session_id)]
        if len(matches) == 1:
            session = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous ID '{args.session_id}', matches: {[s.id[:10] for s in matches]}")
            return 1
        else:
            print(f"Session '{args.session_id}' not found")
            return 1

    # Basic info
    print()
    print("=" * 60)
    print(f"SESSION: {session.id}")
    print("=" * 60)
    print()
    print(f"  Name:           {session.name or '(unnamed)'}")
    print(f"  Project:        {session.project or '-'}")
    print(f"  Status:         {session.status.value}")
    print(f"  Task ID:        {session.task_id}")
    print()
    print(f"  Created:        {session.created_at.strftime('%Y-%m-%d %H:%M')}")
    print(
        f"  Last Active:    {session.last_active.strftime('%Y-%m-%d %H:%M')} ({format_datetime(session.last_active.isoformat())})"
    )
    print(f"  Resume Count:   {session.resume_count}")
    print(f"  Message Count:  {session.message_count}")
    print()

    if session.tags:
        print(f"  Tags:           {', '.join(session.tags)}")
        print()

    if session.last_topic:
        print(f"  Last Topic:     {session.last_topic}")
        print()

    if session.summary:
        print("  Summary:")
        for line in session.summary.split("\n"):
            print(f"    {line}")
        print()

    # Findings
    if args.findings:
        findings = store.get_findings(session.id)
        if findings:
            print("-" * 60)
            print(f"KEY FINDINGS ({len(findings)})")
            print("-" * 60)
            for f in findings:
                icon = "✓" if f.confirmed else "?"
                tags = f" [{', '.join(f.tags)}]" if f.tags else ""
                print(f"  {icon} {f.content[:60]}{tags}")
            print()

    # Checkpoints
    if args.checkpoints:
        checkpoints = store.get_checkpoints(session.id, limit=5)
        if checkpoints:
            print("-" * 60)
            print(f"RECENT CHECKPOINTS ({len(checkpoints)})")
            print("-" * 60)
            for c in checkpoints:
                print(
                    f"  {c.created_at.strftime('%Y-%m-%d %H:%M')} - {c.trigger} (msgs: {c.message_count})"
                )
            print()

    # Documents (if any)
    docs = store.get_documents(session.id)
    if docs:
        print("-" * 60)
        print(f"DOCUMENTS ({len(docs)})")
        print("-" * 60)
        for d in docs:
            path = Path(d.file_path).name
            print(
                f"  {path} ({d.total_pages or '?'} pages) - {format_datetime(d.processed_at.isoformat())}"
            )
        print()

    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume a session and show context for injection."""
    store = get_store()

    # Find session
    session = store.get_session(args.session_id)
    if not session:
        all_sessions = store.list_sessions(limit=1000)
        matches = [s for s in all_sessions if s.id.startswith(args.session_id)]
        if len(matches) == 1:
            session = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous ID '{args.session_id}', matches: {[s.id[:10] for s in matches]}")
            return 1
        else:
            print(f"Session '{args.session_id}' not found")
            return 1

    # Build resume context
    context = store.build_resume_context(session.id)
    if not context:
        print(f"Failed to build resume context for '{session.id}'")
        return 1

    # Fork task ID
    old_task_id = session.task_id
    new_task_id = session.fork_task_id()
    session.update_activity()
    store.update_session(session)

    if args.output == "json":
        # JSON output for programmatic use
        output = {
            "session_id": session.id,
            "old_task_id": old_task_id,
            "new_task_id": new_task_id,
            "resume_count": session.resume_count,
            "context": context.format_for_injection(),
            "findings": [f.content for f in context.findings],
            "warnings": context.warnings,
            "document_changes": [
                {
                    "file": c.file_path,
                    "changed": c.new_hash is not None,
                    "exists": c.exists,
                }
                for c in context.document_changes
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        # Human-readable output
        print()
        print("=" * 60)
        print("SESSION RESUMED")
        print("=" * 60)
        print()
        print(f"  Session:      {session.id[:10]}...")
        print(f"  Name:         {session.name or '(unnamed)'}")
        print(f"  Resume Count: {session.resume_count}")
        print(f"  Task ID:      {old_task_id[:8]}... -> {new_task_id[:12]}...")
        print()

        # Warnings
        if context.warnings:
            print("-" * 60)
            print("WARNINGS")
            print("-" * 60)
            for w in context.warnings:
                print(f"  ⚠ {w}")
            print()

        # Document changes
        if context.document_changes:
            print("-" * 60)
            print("DOCUMENT STATUS")
            print("-" * 60)
            for c in context.document_changes:
                path = Path(c.file_path).name
                if not c.exists:
                    status = "✗ MISSING"
                elif c.new_hash:
                    status = "⚠ CHANGED"
                else:
                    status = "✓ unchanged"
                print(f"  {status:<12} {path}")
            print()

        # Key findings
        if context.findings:
            print("-" * 60)
            print("KEY FINDINGS TO REMEMBER")
            print("-" * 60)
            for f in context.findings[:10]:
                print(f"  • {f.content[:70]}")
            if len(context.findings) > 10:
                print(f"  ... and {len(context.findings) - 10} more")
            print()

        # Context injection block
        print("-" * 60)
        print("CONTEXT INJECTION (for LLM)")
        print("-" * 60)
        print()
        print(context.format_for_injection())
        print()

    return 0


def cmd_archive(args: argparse.Namespace) -> int:
    """Archive a session."""
    store = get_store()

    # Find session (support partial ID match)
    session_id = args.session_id
    if not store.get_session(session_id):
        all_sessions = store.list_sessions(limit=1000)
        matches = [s for s in all_sessions if s.id.startswith(session_id)]
        if len(matches) == 1:
            session_id = matches[0].id
        elif len(matches) > 1:
            print(f"Ambiguous ID '{args.session_id}', matches: {[s.id[:10] for s in matches]}")
            return 1
        else:
            print(f"Session '{args.session_id}' not found")
            return 1

    if not store.archive_session(session_id):
        print(f"Session '{session_id}' not found")
        return 1

    print(f"Session '{session_id[:10]}' archived")
    return 0


def cmd_findings(args: argparse.Namespace) -> int:
    """List findings for a session."""
    store = get_store()

    # Find session
    session = store.get_session(args.session_id)
    if not session:
        all_sessions = store.list_sessions(limit=1000)
        matches = [s for s in all_sessions if s.id.startswith(args.session_id)]
        if len(matches) == 1:
            session = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous ID '{args.session_id}'")
            return 1
        else:
            print(f"Session '{args.session_id}' not found")
            return 1

    findings = store.get_findings(session.id)

    if not findings:
        print(f"No findings for session '{session.id[:10]}'")
        return 0

    if args.output == "json":
        output = [
            {
                "id": f.id,
                "content": f.content,
                "source": f.source.value,
                "confidence": f.confidence,
                "confirmed": f.confirmed,
                "tags": f.tags,
                "created_at": f.created_at.isoformat(),
            }
            for f in findings
        ]
        print(json.dumps(output, indent=2))
    else:
        print()
        print(f"FINDINGS for {session.name or session.id[:10]}")
        print("=" * 60)
        print()

        for f in findings:
            icon = "✓" if f.confirmed else "?"
            conf = f"({f.confidence:.0%})" if f.confidence < 1.0 else ""
            tags = f" [{', '.join(f.tags)}]" if f.tags else ""
            source = f.source.value.replace("_", " ")

            print(f"{icon} {f.content}")
            print(f"  {source} {conf}{tags} - {format_datetime(f.created_at.isoformat())}")
            print()

    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a session."""
    store = get_store()

    # Find session (support partial ID match)
    session = store.get_session(args.session_id)
    if not session:
        all_sessions = store.list_sessions(limit=1000)
        matches = [s for s in all_sessions if s.id.startswith(args.session_id)]
        if len(matches) == 1:
            session = matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous ID '{args.session_id}', matches: {[s.id[:10] for s in matches]}")
            return 1
        else:
            print(f"Session '{args.session_id}' not found")
            return 1

    # Confirm deletion
    if not args.force:
        print(f"Delete session '{session.name or session.id[:10]}'?")
        print(f"  Messages: {session.message_count}")
        print(f"  Findings: {len(store.get_findings(session.id))}")
        print()
        response = input("Type 'yes' to confirm: ")
        if response.lower() != "yes":
            print("Cancelled")
            return 0

    if store.delete_session(session.id):
        print(f"Session '{session.id[:10]}' deleted")
        return 0
    else:
        print("Failed to delete session")
        return 1


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="orch sessions",
        description="Session management for orchestration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # List command
    list_parser = subparsers.add_parser("list", help="List sessions")
    list_parser.add_argument(
        "--status", choices=["active", "idle", "stale", "archived"], help="Filter by status"
    )
    list_parser.add_argument("--project", help="Filter by project")
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum results")

    # Search command
    search_parser = subparsers.add_parser("search", help="Search sessions")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--limit", type=int, default=20, help="Maximum results")

    # Show command
    show_parser = subparsers.add_parser("show", help="Show session details")
    show_parser.add_argument("session_id", help="Session ID (or prefix)")
    show_parser.add_argument("--findings", action="store_true", help="Include findings")
    show_parser.add_argument("--checkpoints", action="store_true", help="Include checkpoints")

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume session with context")
    resume_parser.add_argument("session_id", help="Session ID (or prefix)")
    resume_parser.add_argument(
        "--output", choices=["text", "json"], default="text", help="Output format"
    )

    # Archive command
    archive_parser = subparsers.add_parser("archive", help="Archive a session")
    archive_parser.add_argument("session_id", help="Session ID")

    # Findings command
    findings_parser = subparsers.add_parser("findings", help="List session findings")
    findings_parser.add_argument("session_id", help="Session ID (or prefix)")
    findings_parser.add_argument(
        "--output", choices=["text", "json"], default="text", help="Output format"
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a session")
    delete_parser.add_argument("session_id", help="Session ID")
    delete_parser.add_argument("--force", "-f", action="store_true", help="Skip confirmation")

    args = parser.parse_args()

    # Dispatch to command
    commands = {
        "list": cmd_list,
        "search": cmd_search,
        "show": cmd_show,
        "resume": cmd_resume,
        "archive": cmd_archive,
        "findings": cmd_findings,
        "delete": cmd_delete,
    }

    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        print("\nCancelled")
        return 130
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
