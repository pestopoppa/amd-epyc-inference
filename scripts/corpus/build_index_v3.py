#!/usr/bin/env python3
"""Parallel sharded corpus index builder (v3).

Downloads languages in parallel, shards ngrams across N SQLite databases
(hash-based), builds indexes in parallel. Total build time ~1.5-2.5h vs 5+h
for v2's monolithic approach.

Architecture:
  Phase 1: Parallel download → temp/{lang}.db (no index)
  Phase 2: Merge snippets → snippets.db, shard ngrams → shard_{00..15}.db
  Phase 3: Parallel CREATE INDEX on each shard
  Phase 4: Cleanup temp DBs, write meta.json

Usage:
    # Full build (all 6 languages, 16 shards)
    python scripts/corpus/build_index_v3.py --output /mnt/raid0/llm/cache/corpus/v3_sharded

    # Python only (fast validation)
    python scripts/corpus/build_index_v3.py --languages python --max-files-per-lang 10000

    # Resume after interruption
    python scripts/corpus/build_index_v3.py --resume

    # Custom shard count
    python scripts/corpus/build_index_v3.py --shards 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(processName)s] %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "/mnt/raid0/llm/cache/corpus/v3_sharded"

LANGUAGE_MAP = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "typescript",
    "rust": "rust",
    "go": "go",
    "c++": "c++",
}

# Snippet extraction config (same as v2 for compatibility)
MIN_SNIPPET_LINES = 5
MAX_SNIPPET_LINES = 50
MAX_SNIPPET_CHARS = 2000
NGRAM_SIZE = 4
MIN_FILE_BYTES = 100
MAX_FILE_BYTES = 100_000
BATCH_SIZE = 5000
PROGRESS_INTERVAL = 10_000

SPLIT_PATTERNS = {
    "python": re.compile(r"^(def |class |async def )", re.MULTILINE),
    "javascript": re.compile(
        r"^(function |const |let |var |class |export |module\.exports)", re.MULTILINE
    ),
    "typescript": re.compile(
        r"^(function |const |let |var |class |export |interface |type )", re.MULTILINE
    ),
    "rust": re.compile(r"^(fn |pub fn |impl |struct |enum |trait |mod )", re.MULTILINE),
    "go": re.compile(r"^(func |type |var |const )", re.MULTILINE),
    "c++": re.compile(
        r"^(class |struct |void |int |bool |auto |template |namespace )", re.MULTILINE
    ),
}


# ── Snippet extraction (shared with v2) ─────────────────────────────────


def extract_snippets_from_content(
    content: str, source: str, language: str,
) -> list[dict]:
    """Extract code snippets from file content."""
    lines = content.split("\n")
    if len(lines) < MIN_SNIPPET_LINES:
        return []

    split_re = SPLIT_PATTERNS.get(language)
    snippets = []
    current: list[str] = []

    def flush():
        if len(current) >= MIN_SNIPPET_LINES:
            body = "\n".join(current)[:MAX_SNIPPET_CHARS]
            snippets.append({
                "code": body,
                "source": source,
                "hash": hashlib.md5(body.encode()).hexdigest()[:12],
                "language": language,
            })

    for line in lines:
        if split_re and split_re.match(line) and not line.startswith((" ", "\t")):
            flush()
            current = [line]
        else:
            current.append(line)
            if len(current) >= MAX_SNIPPET_LINES:
                flush()
                current = []

    flush()
    return snippets


def _normalize_token(token: str) -> str:
    return re.sub(r"[^a-z0-9_]", "", token.lower())


def extract_ngrams(text: str, n: int = NGRAM_SIZE) -> list[str]:
    raw_words = text.lower().split()
    words = [_normalize_token(w) for w in raw_words]
    words = [w for w in words if w]
    if len(words) < n:
        return []
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def gram_to_shard(gram: str, num_shards: int) -> int:
    """Deterministic shard assignment via hash."""
    h = hashlib.md5(gram.encode()).digest()
    return int.from_bytes(h[:4], "little") % num_shards


# ── Phase 1: Download one language to temp DB ────────────────────────────


def _create_temp_db(db_path: str) -> sqlite3.Connection:
    """Create temp per-language DB (no ngram index yet)."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-256000")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            source TEXT DEFAULT '',
            hash TEXT NOT NULL,
            language TEXT DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS ngrams (
            gram TEXT NOT NULL,
            snippet_id INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_snippets_hash ON snippets(hash);
    """)
    return conn


def download_language(
    language: str,
    output_dir: str,
    max_files: int = 0,
) -> dict:
    """Download and process one language into a temp DB. Runs in subprocess."""
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    db_path = os.path.join(temp_dir, f"{language}.db")

    # Check if already complete (resume support)
    done_marker = db_path + ".done"
    if os.path.exists(done_marker):
        log.info("[%s] Already downloaded (found .done marker), skipping", language)
        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
        conn.close()
        return {"language": language, "files": 0, "snippets": count, "skipped": 0, "resumed": True}

    # Remove partial DB if exists
    if os.path.exists(db_path):
        os.remove(db_path)
        for wal_file in [db_path + "-wal", db_path + "-shm"]:
            if os.path.exists(wal_file):
                os.remove(wal_file)

    conn = _create_temp_db(db_path)
    seen_hashes: set[str] = set()

    from datasets import load_dataset

    data_dir = LANGUAGE_MAP.get(language, language)
    log.info("[%s] Streaming bigcode/the-stack data_dir=%s ...", language, data_dir)

    try:
        ds = load_dataset(
            "bigcode/the-stack",
            data_dir=f"data/{data_dir}",
            split="train",
            streaming=True,
        )
    except Exception as e:
        log.error("[%s] Failed to load: %s", language, e)
        conn.close()
        return {"language": language, "files": 0, "snippets": 0, "skipped": 0, "error": str(e)}

    stats = {"language": language, "files": 0, "snippets": 0, "skipped": 0}
    batch: list[dict] = []
    ngram_rows: list[tuple[str, int]] = []

    for item in ds:
        content = item.get("content", "")
        size = len(content.encode("utf-8", errors="replace"))

        if size < MIN_FILE_BYTES or size > MAX_FILE_BYTES:
            stats["skipped"] += 1
            continue

        if item.get("is_vendor") or item.get("is_generated"):
            stats["skipped"] += 1
            continue

        path = item.get("max_stars_repo_path", item.get("path", "unknown"))
        repo = item.get("max_stars_repo_name", "")
        source = f"{repo}:{path}" if repo else path

        file_snippets = extract_snippets_from_content(content, source, language)

        for snip in file_snippets:
            h = snip["hash"]
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            cursor = conn.execute(
                "INSERT INTO snippets (code, source, hash, language) VALUES (?, ?, ?, ?)",
                (snip["code"], snip["source"], h, snip["language"]),
            )
            sid = cursor.lastrowid
            grams = extract_ngrams(snip["code"])
            for gram in grams:
                ngram_rows.append((gram, sid))

            stats["snippets"] += 1

        stats["files"] += 1

        # Batch commit ngrams
        if len(ngram_rows) >= BATCH_SIZE * 20:
            conn.executemany("INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)", ngram_rows)
            conn.commit()
            ngram_rows = []

        if stats["files"] % PROGRESS_INTERVAL == 0:
            log.info(
                "[%s] %d files, %d snippets, %d skipped",
                language, stats["files"], stats["snippets"], stats["skipped"],
            )

        if max_files and stats["files"] >= max_files:
            log.info("[%s] Reached max_files=%d", language, max_files)
            break

    # Final flush
    if ngram_rows:
        conn.executemany("INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)", ngram_rows)

    conn.commit()
    conn.close()

    # Mark as complete
    Path(done_marker).write_text(json.dumps(stats))

    log.info(
        "[%s] Done: %d files, %d snippets, %d skipped",
        language, stats["files"], stats["snippets"], stats["skipped"],
    )
    return stats


# ── Phase 2: Merge snippets + shard ngrams ───────────────────────────────


def merge_and_shard(
    output_dir: str,
    languages: list[str],
    num_shards: int,
) -> dict:
    """Merge temp DBs into snippets.db + shard_{00..15}.db."""
    temp_dir = os.path.join(output_dir, "temp")
    t0 = time.perf_counter()

    # Create snippets.db
    snippets_db_path = os.path.join(output_dir, "snippets.db")
    if os.path.exists(snippets_db_path):
        os.remove(snippets_db_path)

    snippets_conn = sqlite3.connect(snippets_db_path)
    snippets_conn.execute("PRAGMA journal_mode=WAL")
    snippets_conn.execute("PRAGMA synchronous=NORMAL")
    snippets_conn.execute("PRAGMA cache_size=-256000")
    snippets_conn.executescript("""
        CREATE TABLE IF NOT EXISTS snippets (
            id INTEGER PRIMARY KEY,
            code TEXT NOT NULL,
            source TEXT DEFAULT '',
            hash TEXT NOT NULL,
            language TEXT DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_snippets_hash ON snippets(hash);
    """)

    # Create shard DBs
    shard_conns: list[sqlite3.Connection] = []
    for i in range(num_shards):
        shard_path = os.path.join(output_dir, f"shard_{i:02d}.db")
        if os.path.exists(shard_path):
            os.remove(shard_path)
        conn = sqlite3.connect(shard_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-128000")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ngrams (
                gram TEXT NOT NULL,
                snippet_id INTEGER NOT NULL
            )
        """)
        shard_conns.append(conn)

    # Process each language temp DB
    global_id = 0
    total_snippets = 0
    total_ngrams = 0
    lang_stats: dict[str, dict] = {}

    for language in languages:
        temp_db = os.path.join(temp_dir, f"{language}.db")
        if not os.path.exists(temp_db):
            log.warning("[merge] Temp DB for %s not found, skipping", language)
            continue

        src_conn = sqlite3.connect(temp_db)
        local_count = src_conn.execute("SELECT COUNT(*) FROM snippets").fetchone()[0]
        id_offset = global_id
        log.info(
            "[merge] %s: %d snippets, global offset=%d",
            language, local_count, id_offset,
        )

        # Copy snippets with remapped IDs
        batch_snippets = []
        id_map: dict[int, int] = {}  # local_id → global_id

        for local_id, code, source, h, lang in src_conn.execute(
            "SELECT id, code, source, hash, language FROM snippets ORDER BY id"
        ):
            new_id = global_id
            id_map[local_id] = new_id
            batch_snippets.append((new_id, code, source, h, lang))
            global_id += 1

            if len(batch_snippets) >= BATCH_SIZE:
                snippets_conn.executemany(
                    "INSERT INTO snippets (id, code, source, hash, language) VALUES (?, ?, ?, ?, ?)",
                    batch_snippets,
                )
                batch_snippets = []

        if batch_snippets:
            snippets_conn.executemany(
                "INSERT INTO snippets (id, code, source, hash, language) VALUES (?, ?, ?, ?, ?)",
                batch_snippets,
            )
        snippets_conn.commit()

        # Shard ngrams with remapped snippet IDs
        shard_buffers: list[list[tuple[str, int]]] = [[] for _ in range(num_shards)]
        ngram_count = 0

        for gram, local_sid in src_conn.execute("SELECT gram, snippet_id FROM ngrams"):
            new_sid = id_map.get(local_sid)
            if new_sid is None:
                continue
            shard_id = gram_to_shard(gram, num_shards)
            shard_buffers[shard_id].append((gram, new_sid))
            ngram_count += 1

            # Flush shard buffers periodically
            if len(shard_buffers[shard_id]) >= BATCH_SIZE * 10:
                shard_conns[shard_id].executemany(
                    "INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)",
                    shard_buffers[shard_id],
                )
                shard_buffers[shard_id] = []

        # Flush remaining
        for sid in range(num_shards):
            if shard_buffers[sid]:
                shard_conns[sid].executemany(
                    "INSERT INTO ngrams (gram, snippet_id) VALUES (?, ?)",
                    shard_buffers[sid],
                )

        src_conn.close()
        total_snippets += local_count
        total_ngrams += ngram_count
        lang_stats[language] = {"snippets": local_count, "ngrams": ngram_count}

        log.info("[merge] %s: %d ngrams sharded", language, ngram_count)

    # Commit all shards
    for conn in shard_conns:
        conn.commit()
        conn.close()
    snippets_conn.commit()
    snippets_conn.close()

    elapsed = time.perf_counter() - t0
    log.info(
        "[merge] Done: %d snippets, %d ngrams in %.1fs",
        total_snippets, total_ngrams, elapsed,
    )
    return {
        "snippets": total_snippets,
        "ngrams": total_ngrams,
        "elapsed_s": round(elapsed, 1),
        "lang_stats": lang_stats,
    }


# ── Phase 3: Parallel index creation ────────────────────────────────────


def create_shard_index(shard_path: str) -> dict:
    """Create index on a single shard DB. Runs in subprocess."""
    shard_name = os.path.basename(shard_path)
    t0 = time.perf_counter()
    log.info("[index] Creating index on %s ...", shard_name)

    conn = sqlite3.connect(shard_path)
    conn.execute("PRAGMA cache_size=-256000")
    conn.execute("PRAGMA mmap_size=1073741824")

    ngram_count = conn.execute("SELECT COUNT(*) FROM ngrams").fetchone()[0]
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ngrams_gram ON ngrams(gram)")
    conn.commit()

    # Checkpoint WAL to merge into main DB
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()

    elapsed = time.perf_counter() - t0
    log.info("[index] %s: %d ngrams indexed in %.1fs", shard_name, ngram_count, elapsed)
    return {"shard": shard_name, "ngrams": ngram_count, "elapsed_s": round(elapsed, 1)}


# ── Phase 4: Metadata + cleanup ─────────────────────────────────────────


def write_metadata(
    output_dir: str,
    languages: list[str],
    num_shards: int,
    merge_stats: dict,
    index_stats: list[dict],
    total_elapsed: float,
) -> None:
    """Write meta.json for the retriever to detect v3 format."""
    meta = {
        "version": 3,
        "format": "sharded_sqlite",
        "ngram_size": NGRAM_SIZE,
        "num_shards": num_shards,
        "num_snippets": merge_stats["snippets"],
        "num_ngrams": merge_stats["ngrams"],
        "languages": languages,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "build_time_s": round(total_elapsed, 1),
        "merge_stats": merge_stats,
        "index_stats": index_stats,
    }
    meta_path = os.path.join(output_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata written to %s", meta_path)


def cleanup_temp(output_dir: str) -> None:
    """Remove temp language DBs."""
    temp_dir = os.path.join(output_dir, "temp")
    if not os.path.exists(temp_dir):
        return
    for f in Path(temp_dir).iterdir():
        f.unlink()
        log.info("[cleanup] Removed %s", f.name)
    os.rmdir(temp_dir)
    log.info("[cleanup] Temp directory removed")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Parallel sharded corpus index builder (v3)"
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT,
        help="Output directory (default: %(default)s)",
    )
    parser.add_argument(
        "--languages", default="python,javascript,typescript,rust,go,c++",
        help="Comma-separated languages (default: %(default)s)",
    )
    parser.add_argument(
        "--shards", type=int, default=16,
        help="Number of ngram shards (default: %(default)s)",
    )
    parser.add_argument(
        "--max-files-per-lang", type=int, default=0,
        help="Max files per language (0=unlimited, for testing)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume: skip languages with .done markers, skip phases with existing outputs",
    )
    parser.add_argument(
        "--download-workers", type=int, default=6,
        help="Max parallel language downloads (default: %(default)s)",
    )
    parser.add_argument(
        "--index-workers", type=int, default=16,
        help="Max parallel index creation workers (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Phase 1 (use existing temp DBs)",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip Phase 2 (use existing shard DBs)",
    )
    parser.add_argument(
        "--cleanup-temp", action="store_true",
        help="Remove temp language DBs after successful build",
    )
    args = parser.parse_args()

    t_total = time.perf_counter()
    languages = [l.strip() for l in args.languages.split(",")]
    languages = [l for l in languages if l in LANGUAGE_MAP]
    output_dir = args.output

    os.makedirs(output_dir, exist_ok=True)
    log.info("=" * 60)
    log.info("Parallel Sharded Corpus Build v3")
    log.info("  Output:    %s", output_dir)
    log.info("  Languages: %s", ", ".join(languages))
    log.info("  Shards:    %d", args.shards)
    log.info("=" * 60)

    # ── Phase 1: Parallel downloads ──────────────────────────────────
    if not args.skip_download:
        log.info("=== Phase 1: Parallel downloads (%d languages) ===", len(languages))
        t1 = time.perf_counter()
        download_stats: dict[str, dict] = {}

        with ProcessPoolExecutor(max_workers=args.download_workers) as pool:
            futures = {
                pool.submit(download_language, lang, output_dir, args.max_files_per_lang): lang
                for lang in languages
            }
            for future in as_completed(futures):
                lang = futures[future]
                try:
                    stats = future.result()
                    download_stats[lang] = stats
                    log.info(
                        "[Phase 1] %s complete: %d snippets",
                        lang, stats.get("snippets", 0),
                    )
                except Exception as e:
                    log.error("[Phase 1] %s FAILED: %s", lang, e)
                    download_stats[lang] = {"error": str(e)}

        elapsed_p1 = time.perf_counter() - t1
        log.info("Phase 1 done in %.1fs", elapsed_p1)

        # Write incremental progress
        progress = {"phase1_downloads": download_stats, "phase1_elapsed_s": round(elapsed_p1, 1)}
        with open(os.path.join(output_dir, "build_progress.json"), "w") as f:
            json.dump(progress, f, indent=2)
    else:
        log.info("=== Phase 1: SKIPPED (--skip-download) ===")

    # ── Phase 2: Merge + shard ───────────────────────────────────────
    if not args.skip_merge:
        log.info("=== Phase 2: Merge snippets + shard ngrams ===")
        merge_stats = merge_and_shard(output_dir, languages, args.shards)
    else:
        log.info("=== Phase 2: SKIPPED (--skip-merge) ===")
        # Read existing stats
        meta_path = os.path.join(output_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                existing_meta = json.load(f)
            merge_stats = existing_meta.get("merge_stats", {
                "snippets": 0, "ngrams": 0, "elapsed_s": 0, "lang_stats": {},
            })
        else:
            merge_stats = {"snippets": 0, "ngrams": 0, "elapsed_s": 0, "lang_stats": {}}

    # ── Phase 3: Parallel index creation ─────────────────────────────
    log.info("=== Phase 3: Parallel index creation (%d shards) ===", args.shards)
    t3 = time.perf_counter()

    shard_paths = [
        os.path.join(output_dir, f"shard_{i:02d}.db")
        for i in range(args.shards)
    ]
    # Verify shards exist
    missing = [p for p in shard_paths if not os.path.exists(p)]
    if missing:
        log.error("Missing shard DBs: %s", missing)
        sys.exit(1)

    index_stats: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.index_workers) as pool:
        futures = {
            pool.submit(create_shard_index, path): path
            for path in shard_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                stats = future.result()
                index_stats.append(stats)
                log.info("[Phase 3] %s indexed", stats["shard"])
            except Exception as e:
                log.error("[Phase 3] %s FAILED: %s", os.path.basename(path), e)

    elapsed_p3 = time.perf_counter() - t3
    log.info("Phase 3 done in %.1fs", elapsed_p3)

    # Also checkpoint snippets.db WAL
    snippets_db = os.path.join(output_dir, "snippets.db")
    if os.path.exists(snippets_db):
        conn = sqlite3.connect(snippets_db)
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

    # ── Phase 4: Metadata + cleanup ──────────────────────────────────
    total_elapsed = time.perf_counter() - t_total
    write_metadata(output_dir, languages, args.shards, merge_stats, index_stats, total_elapsed)

    if args.cleanup_temp:
        cleanup_temp(output_dir)

    # Summary
    total_shard_size = sum(
        os.path.getsize(p) for p in shard_paths if os.path.exists(p)
    )
    snippets_size = os.path.getsize(snippets_db) if os.path.exists(snippets_db) else 0

    log.info("=" * 60)
    log.info("Build complete in %.1fs", total_elapsed)
    log.info("  Snippets:    %d", merge_stats.get("snippets", 0))
    log.info("  N-grams:     %d", merge_stats.get("ngrams", 0))
    log.info("  Shards:      %d x ~%.1f MB = %.1f MB total",
             args.shards,
             total_shard_size / args.shards / 1024 / 1024,
             total_shard_size / 1024 / 1024)
    log.info("  Snippets DB: %.1f MB", snippets_size / 1024 / 1024)
    log.info("  Output:      %s", output_dir)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
