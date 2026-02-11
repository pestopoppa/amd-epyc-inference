#!/usr/bin/env python3
"""Incrementally re-index files changed since last indexing into NextPLAID.

Designed for `make gates` — runs quickly when few files changed, skips entirely
when no changes detected. Uses git diff against a stored commit hash.

Usage:
    python scripts/nextplaid/reindex_changed.py          # Incremental
    python scripts/nextplaid/reindex_changed.py --full    # Force full reindex
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/mnt/raid0/llm/claude")
CLIENT_URL = "http://localhost:8088"
STAMP_FILE = PROJECT_ROOT / "cache" / "next-plaid" / ".last_indexed_commit"

# File patterns (must match index_codebase.py)
CODE_EXTENSIONS = {".py"}
DOC_EXTENSIONS = {".md", ".yaml", ".yml", ".json"}

CODE_PREFIXES = ("src/", "orchestration/", "scripts/", "tests/")
DOC_PREFIXES = ("docs/", "handoffs/", "orchestration/")
DOC_ROOT_FILES = {"CLAUDE.md", "CHANGELOG.md"}

SKIP_FRAGMENTS = {"__pycache__", ".pyc", "node_modules", ".git", "cache/"}


def get_last_commit() -> str | None:
    """Read the commit hash from last successful index."""
    if STAMP_FILE.exists():
        return STAMP_FILE.read_text().strip()
    return None


def save_commit(commit: str) -> None:
    STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(commit + "\n")


def current_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    return result.stdout.strip()


def changed_files_since(base_commit: str) -> list[str]:
    """Get files changed between base_commit and HEAD (including untracked)."""
    # Tracked changes
    result = subprocess.run(
        ["git", "diff", "--name-only", base_commit, "HEAD"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    tracked = set(result.stdout.strip().split("\n")) if result.stdout.strip() else set()

    # Unstaged changes (modified but not committed)
    result2 = subprocess.run(
        ["git", "diff", "--name-only"],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )
    unstaged = set(result2.stdout.strip().split("\n")) if result2.stdout.strip() else set()

    return sorted(tracked | unstaged - {""})


def classify_file(rel_path: str) -> str | None:
    """Classify a file into 'code', 'docs', or None (skip)."""
    if any(skip in rel_path for skip in SKIP_FRAGMENTS):
        return None

    p = Path(rel_path)
    ext = p.suffix.lower()

    # Root-level doc files
    if rel_path in DOC_ROOT_FILES and ext in DOC_EXTENSIONS:
        return "docs"

    # Code files
    if ext in CODE_EXTENSIONS and any(rel_path.startswith(pre) for pre in CODE_PREFIXES):
        return "code"

    # Doc files
    if ext in DOC_EXTENSIONS and any(rel_path.startswith(pre) for pre in DOC_PREFIXES):
        return "docs"

    return None


def chunk_file(path: Path, max_chars: int = 1800) -> list[dict]:
    """Split file into chunks (same logic as index_codebase.py)."""
    try:
        text = path.read_text(errors="replace")
    except Exception:
        return []

    if not text.strip():
        return []

    lines = text.split("\n")
    chunks: list[dict] = []
    chunk_lines: list[str] = []
    char_count = 0
    start_line = 1

    for i, line in enumerate(lines, 1):
        chunk_lines.append(line)
        char_count += len(line) + 1

        at_boundary = (line.strip() == "" and char_count >= max_chars * 0.7)
        at_limit = char_count >= max_chars

        if at_boundary or at_limit:
            chunks.append({
                "text": "\n".join(chunk_lines),
                "file": str(path.relative_to(PROJECT_ROOT)),
                "start_line": start_line,
                "end_line": i,
            })
            overlap = chunk_lines[-3:] if len(chunk_lines) >= 3 else chunk_lines[-1:]
            chunk_lines = list(overlap)
            char_count = sum(len(ln) + 1 for ln in chunk_lines)
            start_line = max(1, i - len(overlap) + 1)

    if chunk_lines and char_count > 10:
        chunks.append({
            "text": "\n".join(chunk_lines),
            "file": str(path.relative_to(PROJECT_ROOT)),
            "start_line": start_line,
            "end_line": len(lines),
        })

    return chunks


def reindex_files(client, files_by_index: dict[str, list[str]]) -> int:
    """Re-index changed files into their respective indices."""
    total = 0

    for index_name, rel_paths in files_by_index.items():
        if not rel_paths:
            continue

        all_texts: list[str] = []
        all_metadata: list[dict] = []

        for rel in rel_paths:
            full = PROJECT_ROOT / rel
            if not full.exists():
                # File was deleted — NextPLAID doesn't support per-doc deletion easily,
                # so we skip. The stale chunks will have low relevance scores.
                continue
            for chunk in chunk_file(full):
                all_texts.append(chunk["text"])
                all_metadata.append({
                    "file": chunk["file"],
                    "start_line": str(chunk["start_line"]),
                    "end_line": str(chunk["end_line"]),
                })

        if not all_texts:
            continue

        # Batch update
        BATCH = 100
        for i in range(0, len(all_texts), BATCH):
            client.update_documents_with_encoding(
                index_name,
                documents=all_texts[i : i + BATCH],
                metadata=all_metadata[i : i + BATCH],
            )

        total += len(all_texts)
        print(f"  [{index_name}] Updated {len(all_texts)} chunks from {len(rel_paths)} files")

    return total


def main():
    parser = argparse.ArgumentParser(description="Incremental NextPLAID reindex")
    parser.add_argument("--full", action="store_true", help="Force full reindex via index_codebase.py")
    parser.add_argument("--url", default=CLIENT_URL, help="NextPLAID server URL")
    args = parser.parse_args()

    if args.full:
        # Delegate to full indexer
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "nextplaid" / "index_codebase.py"),
             "--reindex", "--url", args.url],
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            save_commit(current_head())
        return

    try:
        from next_plaid_client import NextPlaidClient
    except ImportError:
        print("Warning: next-plaid-client not installed, skipping reindex", file=sys.stderr)
        return

    client = NextPlaidClient(args.url)

    # Health check
    try:
        client.health()
    except Exception:
        print("Warning: NextPLAID not reachable, skipping reindex", file=sys.stderr)
        return

    last_commit = get_last_commit()
    head = current_head()

    if last_commit == head:
        print("  NextPLAID index up-to-date (no new commits)")
        return

    if last_commit is None:
        print("  No previous index stamp — running full index")
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "nextplaid" / "index_codebase.py"),
             "--reindex", "--url", args.url],
            cwd=PROJECT_ROOT,
        )
        if result.returncode == 0:
            save_commit(head)
        return

    # Incremental: find changed files
    changed = changed_files_since(last_commit)
    if not changed:
        print("  No files changed since last index")
        save_commit(head)
        return

    # Classify changed files
    files_by_index: dict[str, list[str]] = {"code": [], "docs": []}
    for f in changed:
        idx = classify_file(f)
        if idx:
            files_by_index[idx].append(f)

    code_count = len(files_by_index["code"])
    doc_count = len(files_by_index["docs"])

    if code_count == 0 and doc_count == 0:
        print("  No indexable files changed")
        save_commit(head)
        return

    print(f"  {code_count} code + {doc_count} doc files changed since {last_commit[:8]}")
    t0 = time.time()

    total = reindex_files(client, files_by_index)

    elapsed = time.time() - t0
    print(f"  Reindexed {total} chunks in {elapsed:.1f}s")
    save_commit(head)


if __name__ == "__main__":
    main()
