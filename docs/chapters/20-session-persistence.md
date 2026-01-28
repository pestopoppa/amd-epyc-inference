# Chapter 20: Session Persistence & Checkpoint/Resume

## Introduction

The session persistence system enables long-running conversations that survive crashes, resume after idle periods, and maintain context across multiple sessions. This is critical for document analysis, iterative benchmarking, and multi-day research tasks.

**Key features:**
- Automatic checkpoints every 5 turns or 30 minutes idle
- Document change detection via SHA-256 hashing
- LLM-extracted findings with confidence scores
- ChromaDB-compatible storage protocol for future semantic search

The system was implemented in 7 phases (2026-01-21 to 2026-01-26) and uses SQLite + numpy for efficient storage.

## Architecture Overview

### Components

| Component | Purpose | Storage |
|-----------|---------|---------|
| `SessionPersister` | Checkpoint triggers & lifecycle | In-memory state |
| `SQLiteSessionStore` | Metadata persistence | `/workspace/orchestration/repl_memory/sessions/sessions.db` |
| `DocumentCache` | OCR result caching | Per-session SQLite: `state/{session_id}/ocr_cache.db` |
| Session models | Data classes | `src/session/models.py` |
| CLI | Session management | `src/cli_sessions.py` |

### Storage Layout

```
/workspace/orchestration/repl_memory/sessions/
├── sessions.db                        # Main session metadata (SQLite WAL mode)
├── session_embeddings.npy             # 896-dim embeddings (TaskEmbedder)
├── state/
│   ├── {session_id}/
│   │   └── ocr_cache.db              # Per-session document cache
│   └── scheduler.json                # Procedure scheduler state
```

## 7-Phase Implementation

The system was built incrementally over 7 phases:

| Phase | Date | Focus | Files Created |
|-------|------|-------|---------------|
| **Phase 1** | 2026-01-21 | Core models & protocol | `models.py`, `protocol.py` |
| **Phase 2** | 2026-01-22 | SQLite store implementation | `sqlite_store.py` |
| **Phase 3** | 2026-01-23 | Document caching layer | `document_cache.py` |
| **Phase 4** | 2026-01-24 | Checkpoint manager | `persister.py` |
| **Phase 5** | 2026-01-25 | CLI interface | `cli_sessions.py` |
| **Phase 6** | 2026-01-25 | API integration | `src/api.py` endpoints |
| **Phase 7** | 2026-01-26 | Testing & validation | `tests/integration/test_sessions.py` |

## Session Lifecycle

### Session Status

Sessions transition through 4 lifecycle states:

| Status | Trigger | Idle Time | Behavior on Resume |
|--------|---------|-----------|-------------------|
| `ACTIVE` | Recent activity | < 1 hour | Direct continuation |
| `IDLE` | No activity | 1 hour - 7 days | Brief context reminder |
| `STALE` | Long idle | 7 - 30 days | Full context injection with document change detection |
| `ARCHIVED` | Manual or 30+ days | > 30 days | "Welcome back" summary, LLM-generated context |

### Creating a Session

```python
from src.session import SQLiteSessionStore, Session

store = SQLiteSessionStore()

# Create new session
session = Session.create(
    name="Benchmark Analysis",
    project="Model Evaluation",
    working_directory="/mnt/raid0/llm/claude"
)

# Store in database
store.create_session(session)

print(f"Session ID: {session.id}")
print(f"Task ID (for MemRL): {session.task_id}")
```

### Session Activity Tracking

```python
# Update activity timestamp
session.update_activity()

# Increment message count
session.message_count += 1

# Update topic
session.last_topic = "Analyzing Qwen3-235B benchmark results"

# Save changes
store.update_session(session)
```

## Checkpoint System

### SessionPersister

The `SessionPersister` class handles automatic checkpoint triggers:

```python
from src.session import SessionPersister

persister = SessionPersister(
    session_store=store,
    session_id=session.id,
    llm_summarizer=None,  # Optional LLM function for summaries
    progress_logger=None   # Optional ProgressLogger for MemRL
)

# After each REPL turn
persister.on_turn(repl_env)

# Check if checkpoint needed
if persister.should_checkpoint():
    checkpoint = persister.save_checkpoint(repl_env)
    print(f"Checkpoint saved: {checkpoint.id}")
```

### Checkpoint Triggers

| Trigger | Condition | Frequency |
|---------|-----------|-----------|
| Turn count | Every 5 conversation turns | Common |
| Idle time | 30 minutes without activity | Moderate |
| Explicit save | User `/save` command | Rare |
| Auto-summary | 2 hours idle + no summary exists | Rare |

### Checkpoint Data Model

```python
from src.session.models import Checkpoint
import hashlib

# Create checkpoint
checkpoint = Checkpoint(
    id=str(uuid.uuid4()),
    session_id=session.id,
    created_at=datetime.utcnow(),
    context_hash=hashlib.sha256(context_str.encode()).hexdigest(),
    artifacts={
        "variables": {"x": 42, "results": [1, 2, 3]},
        "plots": ["plot_abc123.png"]
    },
    execution_count=45,
    exploration_calls=12,
    message_count=session.message_count,
    trigger="turns"  # or "idle", "explicit", "summary"
)

# Save to store
store.save_checkpoint(checkpoint)
```

## Document Tracking & Change Detection

### Adding Documents

```python
from src.session.models import SessionDocument
from pathlib import Path

# Process a PDF
file_path = Path("/mnt/raid0/llm/docs/whitepaper.pdf")

# Compute file hash
file_hash = SessionDocument.compute_file_hash(file_path)

# Create document record
doc = SessionDocument(
    id=str(uuid.uuid4()),
    session_id=session.id,
    file_path=str(file_path),
    file_hash=file_hash,
    processed_at=datetime.utcnow(),
    total_pages=42,
    cache_path="state/{session_id}/ocr_cache.db"
)

store.add_document(doc)
```

### OCR Caching

The `DocumentCache` stores OCR results to avoid reprocessing:

```python
from src.session.document_cache import DocumentCache

cache = DocumentCache(session_id=session.id, session_store=store)

# Check cache before OCR
cached_result = cache.get_cached("/path/to/document.pdf")

if cached_result:
    print(f"Cache hit! {cached_result.total_pages} pages")
else:
    # Run OCR
    result = run_ocr("/path/to/document.pdf")

    # Cache result
    cache.cache_result("/path/to/document.pdf", result, track_in_session=True)
```

### Change Detection on Resume

When resuming a session, the system detects document changes:

```python
# Build resume context (includes change detection)
resume_ctx = store.build_resume_context(session.id)

# Check for changes
if resume_ctx.document_changes:
    for change in resume_ctx.document_changes:
        if not change.exists:
            print(f"⚠ Document missing: {change.file_path}")
        elif change.new_hash != change.old_hash:
            print(f"⚠ Document changed: {change.file_path}")
```

## Findings System

### Finding Sources

Findings are key insights extracted during a session:

| Source | Confidence | Requires Confirmation |
|--------|------------|-----------------------|
| `USER_MARKED` | 1.0 | No (explicitly marked) |
| `LLM_EXTRACTED` | 0.0-1.0 | Yes (LLM can hallucinate) |
| `HEURISTIC` | 0.7-0.9 | Yes (rule-based) |

### Creating Findings

```python
from src.session.models import Finding, FindingSource

# User explicitly marks a finding
finding = Finding(
    id=str(uuid.uuid4()),
    session_id=session.id,
    content="Qwen3-235B achieves 6.75 t/s with MoE expert reduction to 4",
    source=FindingSource.USER_MARKED,
    created_at=datetime.utcnow(),
    confidence=1.0,
    confirmed=True,
    tags=["benchmark", "moe", "optimization"],
    source_file="/mnt/raid0/llm/claude/benchmarks/results/runs/2026-01-15/..."
)

store.add_finding(finding)
```

### LLM-Extracted Findings

```python
# LLM extracts finding during conversation
llm_finding = Finding(
    id=str(uuid.uuid4()),
    session_id=session.id,
    content="Document suggests using temperature=0.7 for VL models",
    source=FindingSource.LLM_EXTRACTED,
    created_at=datetime.utcnow(),
    confidence=0.75,  # LLM confidence
    confirmed=False,   # Needs user confirmation
    source_page=12
)

store.add_finding(llm_finding)

# User confirms later
llm_finding.confirmed = True
store.update_finding(llm_finding)
```

## Resume Context

### Building Resume Context

```python
# Build context for session resume
context = store.build_resume_context(session.id)

# Context includes:
print(f"Session: {context.session.name}")
print(f"Documents: {len(context.documents)}")
print(f"Findings: {len(context.findings)}")
print(f"Changes detected: {len(context.document_changes)}")

# Format for LLM injection
llm_context = context.format_for_injection()

# Inject at conversation start
messages = [
    {"role": "system", "content": llm_context},
    {"role": "user", "content": "Continue where we left off..."}
]
```

### Resume Context Format

The `format_for_injection()` method produces markdown:

```markdown
# Session Resumed: Benchmark Analysis
Last active: 2026-01-27 14:30 (47 messages)

## Documents
- /mnt/raid0/llm/docs/whitepaper.pdf (42 pages, processed)
- /mnt/raid0/llm/benchmarks/results/runs/.../results.json (1 pages, CHANGED)

## Key Findings from Previous Session
1. Qwen3-235B achieves 6.75 t/s with MoE expert reduction to 4
2. Prompt lookup provides 12.7x speedup on summarization tasks
3. SSM models (Qwen3-Next) cannot use speculative decoding
... and 7 more findings

## Last Conversation Topic
Analyzing optimal expert count for MoE models

## Warnings
- Source file changed: /mnt/raid0/llm/benchmarks/results/runs/.../results.json
```

## CLI Interface

The `cli_sessions.py` module provides command-line tools:

```bash
# List all active sessions
orch sessions list --status active

# Search sessions by topic
orch sessions search "benchmark"

# Show session details
orch sessions show abc123 --findings --checkpoints

# Resume a session (get context injection)
orch sessions resume abc123

# Archive old sessions
orch sessions archive abc123

# Delete a session (requires --force)
orch sessions delete abc123 --force
```

### Example CLI Output

```bash
$ orch sessions list --status active

● abc123  Benchmark Analysis      47 msgs   2h ago   [benchmark, moe]
● def456  Document Formalization  12 msgs   30m ago  [ocr, vision]
○ ghi789  Code Review             8 msgs    1d ago   [code, refactor]

3 sessions found
```

## Storage Performance

| Metric | Value | Notes |
|--------|-------|-------|
| Session metadata | ~2KB/session | SQLite row |
| Checkpoint | ~5-50KB | Depends on artifact count |
| Document record | ~1KB | Reference only, OCR cached separately |
| OCR cache | ~100KB-2MB/doc | Compressed JSON, no images |
| Embeddings | 896 × 4 bytes = 3.5KB | Per session (TaskEmbedder) |

**WAL mode benefits:**
- Crash-safe (writes to WAL first)
- Better concurrency (readers don't block writers)
- Automatic checkpoint merging on close

## References

- **Session models**: `src/session/models.py`
- **Protocol (abstract interface)**: `src/session/protocol.py`
- **SQLite store**: `src/session/sqlite_store.py`
- **Document cache**: `src/session/document_cache.py`
- **Checkpoint manager**: `src/session/persister.py`
- **CLI**: `src/cli_sessions.py`
- **API integration**: `src/api.py` (POST /sessions, GET /sessions/{id}/resume)

---

*Previous: [Chapter 19: Procedure Registry](19-procedure-registry.md)* | *Next: [Chapter 21: Benchmarking Framework](21-benchmarking-framework.md)*
