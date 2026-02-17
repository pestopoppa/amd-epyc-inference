#!/usr/bin/env python3
from __future__ import annotations

"""Build strategy-focused progress/handoff transition graphs.

Ingests strategy artifacts (excluding agent audit logs):
- structured progress logs (`logs/progress/*.jsonl`)
- markdown progress reports (`progress/**/*.md`, `orchestration/progress/*.md`)
- handoffs (`handoffs/{active,completed,blocked,archived}/*.md`)
- orchestration blocker board (`orchestration/BLOCKED_TASKS.md`)

Outputs:
- events.jsonl
- raw_graph.json
- coarse_graph_*.json (10 variants)
- motifs_failure_*.csv
- motifs_hypothesis_*.csv
- summary.md
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass
class Event:
    source_type: str
    source_file: str
    source_line: int
    session_key: str
    category: str
    level: str
    message: str
    details: str
    timestamp: str
    ordinal: int

    def payload(self) -> str:
        text = f"{self.message} || {self.details}".strip(" |")
        return text


def _safe_json_loads(line: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _normalize_text(text: str, max_len: int = 400) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    return text[:max_len]


def _timestamp_from_path(path: Path) -> str:
    m = re.search(r"(20\d{2}-\d{2}-\d{2})", path.name)
    if not m:
        return ""
    return f"{m.group(1)}T00:00:00"


def _doc_type_of_path(path: Path) -> str:
    s = str(path)
    if "/handoffs/" in s:
        return "handoff"
    if s.endswith("/orchestration/BLOCKED_TASKS.md"):
        return "blocked_tasks"
    return "progress"


def _source_type_for_markdown(path: Path) -> str:
    dt = _doc_type_of_path(path)
    if dt == "handoff":
        return "handoff_md"
    if dt == "blocked_tasks":
        return "blocked_tasks_md"
    return "progress_md"


def discover_sources(root: Path) -> dict[str, list[Path]]:
    progress_jsonl: list[Path] = []
    progress_md: list[Path] = []
    handoff_md: list[Path] = []
    blocked_tasks_md: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        s = str(p)

        if "/.git" in s or "__pycache__" in s or "/logs/strategy_graph" in s:
            dirnames[:] = []
            continue

        for fn in filenames:
            fp = p / fn
            if fn.endswith(".jsonl") and "/logs/progress" in s:
                progress_jsonl.append(fp)
                continue

            if not fn.endswith(".md"):
                continue

            if "/progress/" in s or s.endswith("/progress") or "/orchestration/progress" in s:
                progress_md.append(fp)
                continue

            if "/handoffs/active" in s or "/handoffs/completed" in s or "/handoffs/blocked" in s or "/handoffs/archived" in s:
                handoff_md.append(fp)
                continue

            if s.endswith("/orchestration") and fn == "BLOCKED_TASKS.md":
                blocked_tasks_md.append(fp)

    return {
        "progress_jsonl": sorted(progress_jsonl),
        "progress_md": sorted(progress_md),
        "handoff_md": sorted(handoff_md),
        "blocked_tasks_md": sorted(blocked_tasks_md),
    }


def _infer_level(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("critical", "outage", "severe", "incident")):
        return "ERROR"
    if any(k in low for k in ("fail", "error", "blocked", "regression", "timeout", "degraded", "risk")):
        return "WARN"
    return "INFO"


def _section_path(sections: list[str]) -> str:
    if not sections:
        return "ROOT"
    return " > ".join(sections)


def _parse_markdown_rich(path: Path, source_type: str, max_events: int | None = None) -> list[Event]:
    events: list[Event] = []
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$")
    task_re = re.compile(r"^-\s*\[([xX ])\]\s*(.+)$")
    bullet_re = re.compile(r"^[-*]\s+(.+)$")

    sections: list[str] = []
    prev_non_empty = ""

    with path.open("r", errors="ignore") as f:
        for i, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            category = ""
            msg = ""
            details = ""

            m = heading_re.match(stripped)
            if m:
                level = len(m.group(1))
                title = _normalize_text(m.group(2), 240)
                while len(sections) >= level:
                    sections.pop()
                sections.append(title)
                category = "md_heading"
                msg = title
            else:
                m = task_re.match(stripped)
                if m:
                    done = m.group(1).lower() == "x"
                    category = "progress_task_checked" if done else "progress_task_open"
                    msg = _normalize_text(m.group(2), 300)
                else:
                    m = bullet_re.match(stripped)
                    if m:
                        category = "progress_bullet"
                        msg = _normalize_text(m.group(1), 300)
                    elif stripped.startswith("|") and stripped.endswith("|"):
                        cells = [c.strip() for c in stripped.strip("|").split("|")]
                        if all(re.fullmatch(r"[-: ]*", c) for c in cells):
                            prev_non_empty = stripped
                            continue
                        category = "md_table_row"
                        msg = _normalize_text(cells[0] if cells else "table_row", 300)
                        details = _normalize_text("cells=" + " || ".join(cells), 900)
                    else:
                        category = "md_paragraph"
                        msg = _normalize_text(stripped, 400)

            if not category:
                prev_non_empty = stripped
                continue

            section = _section_path(sections)
            doc_type = _doc_type_of_path(path)
            if details:
                details = f"section={section} | doc_type={doc_type} | {details}"
            else:
                details = f"section={section} | doc_type={doc_type}"
            if prev_non_empty and category == "md_paragraph":
                details += f" | prev={_normalize_text(prev_non_empty, 120)}"

            events.append(
                Event(
                    source_type=source_type,
                    source_file=str(path),
                    source_line=i,
                    session_key=f"{doc_type}:{path}",
                    category=category,
                    level=_infer_level(msg),
                    message=msg,
                    details=_normalize_text(details, 1000),
                    timestamp=_timestamp_from_path(path),
                    ordinal=i,
                )
            )
            if max_events and len(events) >= max_events:
                break
            prev_non_empty = stripped

    return events


def parse_progress_jsonl(path: Path, max_events: int | None = None) -> list[Event]:
    events: list[Event] = []
    with path.open("r", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            obj = _safe_json_loads(line.strip())
            if not obj:
                continue
            data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
            msg = (
                data.get("objective")
                or data.get("reason")
                or data.get("strategy")
                or data.get("action_type")
                or ""
            )
            details_parts = [
                "doc_type=progress_jsonl",
                f"event_type={obj.get('event_type') or ''}",
                f"outcome={obj.get('outcome') or ''}",
                f"outcome_details={obj.get('outcome_details') or ''}",
                f"agent_role={obj.get('agent_role') or ''}",
                f"agent_tier={obj.get('agent_tier') or ''}",
            ]
            for k in ("task_type", "action_type", "route", "strategy", "reason"):
                if k in data:
                    details_parts.append(f"{k}={data.get(k)}")

            events.append(
                Event(
                    source_type="progress_jsonl",
                    source_file=str(path),
                    source_line=i,
                    session_key=f"task:{obj.get('task_id') or 'unknown'}",
                    category=_normalize_text(obj.get("event_type") or "UNKNOWN", 80),
                    level=_infer_level((obj.get("outcome") or "") + " " + (obj.get("outcome_details") or "") + " " + (obj.get("event_type") or "")),
                    message=_normalize_text(str(msg), 400),
                    details=_normalize_text(" | ".join(details_parts), 1000),
                    timestamp=_normalize_text(obj.get("timestamp") or _timestamp_from_path(path), 80),
                    ordinal=i,
                )
            )
            if max_events and len(events) >= max_events:
                break
    return events


def dedupe_events(events: list[Event]) -> list[Event]:
    seen = set()
    deduped: list[Event] = []
    for e in events:
        key = (
            e.source_type,
            e.session_key,
            e.timestamp,
            e.category,
            e.level,
            e.message,
            e.details,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)
    return deduped


def load_events_jsonl(path: Path) -> list[Event]:
    events: list[Event] = []
    if not path.exists():
        return events
    with path.open("r", errors="ignore") as f:
        for line in f:
            obj = _safe_json_loads(line.strip())
            if not obj:
                continue
            try:
                events.append(
                    Event(
                        source_type=str(obj.get("source_type", "unknown")),
                        source_file=str(obj.get("source_file", "")),
                        source_line=int(obj.get("source_line", 0)),
                        session_key=str(obj.get("session_key", "unknown")),
                        category=str(obj.get("category", "UNKNOWN")),
                        level=str(obj.get("level", "UNKNOWN")),
                        message=str(obj.get("message", "")),
                        details=str(obj.get("details", "")),
                        timestamp=str(obj.get("timestamp", "")),
                        ordinal=int(obj.get("ordinal", 0)),
                    )
                )
            except Exception:
                continue
    return events


def dataset_fingerprint(events: list[Event]) -> str:
    h = hashlib.sha256()
    for e in events:
        h.update(
            (
                f"{e.source_type}\x1f{e.session_key}\x1f{e.timestamp}\x1f"
                f"{e.category}\x1f{e.level}\x1f{e.message}\x1f{e.details}\n"
            ).encode("utf-8")
        )
    return h.hexdigest()[:12]


def _event_sort_key(e: Event) -> tuple[str, str, int, str, int]:
    return (e.session_key, e.timestamp or "", e.ordinal, e.source_file, e.source_line)


def _hashed_embedding(text: str, dim: int = 128) -> list[float]:
    vec = [0.0] * dim
    toks = re.findall(r"[a-z0-9_\-]{2,}", text.lower())
    if not toks:
        return vec
    for t in toks:
        h = int(hashlib.sha256(t.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 8) & 1) else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec


def _cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _bucket_signature(emb: list[float], topk: int = 6) -> str:
    idxs = sorted(range(len(emb)), key=lambda i: abs(emb[i]), reverse=True)[:topk]
    return "|".join(f"{i}:{'+' if emb[i] >= 0 else '-'}" for i in idxs)


def cluster_payloads(payloads: list[str], threshold: float = 0.82) -> dict[str, int]:
    centers: list[list[float]] = []
    counts: list[int] = []
    assign: dict[str, int] = {}
    bucket_to_clusters: dict[str, list[int]] = defaultdict(list)

    for text in payloads:
        emb = _hashed_embedding(text)
        sig = _bucket_signature(emb)
        if not centers:
            centers.append(emb)
            counts.append(1)
            assign[text] = 0
            bucket_to_clusters[sig].append(0)
            continue

        best_idx = -1
        best_sim = -1.0
        candidates = bucket_to_clusters.get(sig) or range(len(centers))
        for i in candidates:
            sim = _cosine(emb, centers[i])
            if sim > best_sim:
                best_sim = sim
                best_idx = i

        if best_sim >= threshold:
            n = counts[best_idx]
            centers[best_idx] = [
                (centers[best_idx][j] * n + emb[j]) / (n + 1)
                for j in range(len(emb))
            ]
            counts[best_idx] = n + 1
            assign[text] = best_idx
            if best_idx not in bucket_to_clusters[sig]:
                bucket_to_clusters[sig].append(best_idx)
        else:
            cid = len(centers)
            centers.append(emb)
            counts.append(1)
            assign[text] = cid
            bucket_to_clusters[sig].append(cid)

    return assign


def cluster_payloads_limited(
    payload_counts: dict[str, int],
    threshold: float = 0.82,
    semantic_max_payloads: int = 3000,
) -> dict[str, int]:
    ranked = sorted(payload_counts.items(), key=lambda kv: kv[1], reverse=True)
    train_payloads = [p for p, _ in ranked[:semantic_max_payloads]]
    tail_payloads = [p for p, _ in ranked[semantic_max_payloads:]]

    assign = cluster_payloads(train_payloads, threshold=threshold)
    if not tail_payloads:
        return assign

    sig_to_cluster_votes: dict[str, Counter[int]] = defaultdict(Counter)
    for p, cid in assign.items():
        sig = _bucket_signature(_hashed_embedding(p))
        sig_to_cluster_votes[sig][cid] += 1

    next_cluster = (max(assign.values()) + 1) if assign else 0
    unknown_sig_to_cluster: dict[str, int] = {}

    for p in tail_payloads:
        sig = _bucket_signature(_hashed_embedding(p))
        if sig in sig_to_cluster_votes:
            cid = sig_to_cluster_votes[sig].most_common(1)[0][0]
        else:
            cid = unknown_sig_to_cluster.get(sig)
            if cid is None:
                cid = next_cluster
                next_cluster += 1
                unknown_sig_to_cluster[sig] = cid
        assign[p] = cid
    return assign


def build_raw_graph(events: list[Event]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_session: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_session[e.session_key].append(e)
    for sk in by_session:
        by_session[sk].sort(key=_event_sort_key)

    nodes: dict[str, dict[str, Any]] = {}
    edges: Counter[tuple[str, str]] = Counter()
    transition_rows: list[dict[str, Any]] = []

    for sk, evs in by_session.items():
        for e in evs:
            payload = e.payload()
            node_id = f"{e.category}|{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"
            nodes.setdefault(
                node_id,
                {
                    "id": node_id,
                    "category": e.category,
                    "level": e.level,
                    "sample_message": e.message,
                    "source_type": e.source_type,
                    "count": 0,
                },
            )
            nodes[node_id]["count"] += 1
        for a, b in zip(evs, evs[1:]):
            pa = a.payload()
            pb = b.payload()
            na = f"{a.category}|{hashlib.sha256(pa.encode('utf-8')).hexdigest()[:12]}"
            nb = f"{b.category}|{hashlib.sha256(pb.encode('utf-8')).hexdigest()[:12]}"
            edges[(na, nb)] += 1
            transition_rows.append(
                {
                    "session_key": sk,
                    "from": na,
                    "to": nb,
                    "from_category": a.category,
                    "to_category": b.category,
                    "from_level": a.level,
                    "to_level": b.level,
                }
            )

    graph = {
        "nodes": list(nodes.values()),
        "edges": [{"source": s, "target": t, "count": c} for (s, t), c in edges.items()],
    }
    return graph, transition_rows


def build_mapped_graph(events: list[Event], mapper: Callable[[Event], str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    by_session: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_session[e.session_key].append(e)
    for sk in by_session:
        by_session[sk].sort(key=_event_sort_key)

    nodes: dict[str, dict[str, Any]] = {}
    edges: Counter[tuple[str, str]] = Counter()
    transitions: list[dict[str, Any]] = []

    for sk, evs in by_session.items():
        mapped = [mapper(e) for e in evs]
        for e, mid in zip(evs, mapped):
            nodes.setdefault(
                mid,
                {
                    "id": mid,
                    "category": e.category,
                    "source_type": e.source_type,
                    "sample_message": e.message,
                    "count": 0,
                },
            )
            nodes[mid]["count"] += 1
        for (a, ma), (b, mb) in zip(zip(evs, mapped), zip(evs[1:], mapped[1:])):
            edges[(ma, mb)] += 1
            transitions.append(
                {
                    "session_key": sk,
                    "from": ma,
                    "to": mb,
                    "from_category": a.category,
                    "to_category": b.category,
                    "from_level": a.level,
                    "to_level": b.level,
                }
            )

    graph = {
        "nodes": list(nodes.values()),
        "edges": [{"source": s, "target": t, "count": c} for (s, t), c in edges.items()],
    }
    return graph, transitions


def _normalize_template(text: str) -> str:
    t = text.lower()
    t = re.sub(r"/[a-z0-9._/\-]+", "<path>", t)
    t = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", t)
    t = re.sub(r"\b\d+\b", "<num>", t)
    t = re.sub(r"\b(chat|task|ses)-[a-z0-9]+\b", "<id>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:160] if t else "<empty>"


def _phase_of_event(e: Event) -> str:
    txt = f"{e.category} {e.message} {e.details}".lower()
    if any(k in txt for k in ("plan", "roadmap", "proposal", "scope", "design")):
        return "planning"
    if any(k in txt for k in ("implement", "fix", "refactor", "build", "wire", "add", "change")):
        return "implementation"
    if any(k in txt for k in ("test", "verify", "validation", "benchmark", "evaluate", "pass rate")):
        return "validation"
    if any(k in txt for k in ("deploy", "restart", "health", "timeout", "runtime", "ops", "incident")):
        return "operations"
    if any(k in txt for k in ("summary", "retrospective", "finding", "lesson", "analysis")):
        return "retrospective"
    return "other"


def _initiative_of_event(e: Event) -> str:
    txt = f"{Path(e.source_file).name} {e.message} {e.details}".lower()
    rules = [
        ("routing", ("routing", "delegate", "architect", "frontdoor")),
        ("memory", ("memrl", "memory", "hypothesis", "failure graph", "episodic")),
        ("benchmarks", ("benchmark", "suite", "pass rate", "score")),
        ("infra", ("timeout", "kv cache", "server", "worker", "health", "uvicorn", "numa")),
        ("prompting", ("prompt", "toon", "extract", "few-shot", "template")),
        ("debugger", ("debugger", "anomaly", "signal", "tap")),
        ("vision_docs", ("vision", "ocr", "document", "pipeline")),
    ]
    for name, kws in rules:
        if any(k in txt for k in kws):
            return name
    return "general"


def _outcome_polarity_of_event(e: Event) -> str:
    txt = f"{e.level} {e.category} {e.message} {e.details}".lower()
    if any(k in txt for k in ("resolved", "fixed", "complete", "passed", "success", "improved", "speedup")):
        return "success"
    if any(k in txt for k in ("fail", "error", "blocked", "regression", "timeout", "stop", "degraded")):
        return "failure"
    if any(k in txt for k in ("partial", "mixed", "tradeoff", "pending")):
        return "mixed"
    return "unknown"


def _strategy_action_of_event(e: Event) -> str:
    txt = f"{e.category} {e.message} {e.details}".lower()
    rules = [
        ("diagnosis", ("root cause", "analysis", "diagnos", "finding")),
        ("mitigation", ("mitigation", "workaround", "guard", "fallback")),
        ("optimization", ("optimiz", "speedup", "latency", "throughput", "compression")),
        ("refactor", ("refactor", "decompose", "extract", "cleanup")),
        ("benchmark", ("benchmark", "eval", "pass rate", "score")),
        ("rollback", ("rollback", "revert", "restore")),
        ("planning", ("plan", "roadmap", "phase", "next steps")),
    ]
    for name, kws in rules:
        if any(k in txt for k in kws):
            return name
    return "other"


def _status_of_event(e: Event) -> str:
    txt = f"{e.category} {e.message} {e.details}".lower()
    if any(k in txt for k in ("blocked", "blocker")):
        return "blocked"
    if any(k in txt for k in ("active", "in progress", "ongoing", "wip", "pending")):
        return "active"
    if any(k in txt for k in ("complete", "completed", "resolved", "done", "archived")):
        return "completed"
    if any(k in txt for k in ("risk", "warning", "critical", "incident", "degraded")):
        return "risk"
    return "unknown"


def _section_template_of_event(e: Event) -> str:
    m = re.search(r"section=([^|]+)", e.details)
    section = m.group(1).strip() if m else "ROOT"
    return _normalize_template(section)


def _doc_type_of_event(e: Event) -> str:
    if e.source_type == "blocked_tasks_md":
        return "blocked_tasks"
    if e.source_type == "handoff_md":
        return "handoff"
    if e.source_type == "progress_jsonl":
        return "progress_jsonl"
    return "progress_md"


def _is_success_transition(t: dict[str, Any]) -> bool:
    txt = f"{t.get('to','')} {t.get('to_category','')} {t.get('to_level','')}".lower()
    return any(k in txt for k in ("success", "resolved", "fixed", "complete", "task_completed", "passed"))


def _is_failure_transition(t: dict[str, Any]) -> bool:
    txt = f"{t.get('from','')} {t.get('to','')} {t.get('from_category','')} {t.get('to_category','')} {t.get('from_level','')} {t.get('to_level','')}".lower()
    return any(k in txt for k in ("failure", "fail", "error", "blocked", "regression", "timeout", "incident", "warn"))


def _is_hypothesis_start(t: dict[str, Any]) -> bool:
    txt = f"{t.get('from','')} {t.get('from_category','')}".lower()
    return any(k in txt for k in ("diagnosis", "decision", "plan", "finding", "hypothesis", "analysis", "strategy"))


def mine_motifs(coarse_graph: dict[str, Any], transitions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = (sum(1 for t in transitions if _is_success_transition(t)) / len(transitions)) if transitions else 0.0

    edge_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"count": 0.0, "succ": 0.0})
    sess_seen: dict[tuple[str, str], set[str]] = defaultdict(set)

    for t in transitions:
        key = (t["from"], t["to"])
        edge_stats[key]["count"] += 1.0
        if _is_success_transition(t):
            edge_stats[key]["succ"] += 1.0
        sess_seen[key].add(t["session_key"])

    failure_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []

    for (src, dst), st in edge_stats.items():
        sess = len(sess_seen[(src, dst)])
        succ_rate = st["succ"] / st["count"] if st["count"] else 0.0
        lift = succ_rate - baseline

        faux = {
            "from": src,
            "to": dst,
            "from_category": src,
            "to_category": dst,
            "from_level": "",
            "to_level": "",
        }
        if _is_failure_transition(faux):
            failure_rows.append(
                {
                    "from": src,
                    "to": dst,
                    "count": int(st["count"]),
                    "session_support": sess,
                    "success_rate": round(succ_rate, 4),
                    "risk_score": round((1.0 - succ_rate) * math.log1p(st["count"]) * max(sess, 1), 4),
                }
            )

        if _is_hypothesis_start(faux):
            hypothesis_rows.append(
                {
                    "from": src,
                    "to": dst,
                    "count": int(st["count"]),
                    "session_support": sess,
                    "success_rate": round(succ_rate, 4),
                    "lift_vs_baseline": round(lift, 4),
                }
            )

    failure_rows.sort(key=lambda r: (r["risk_score"], r["count"]), reverse=True)
    hypothesis_rows.sort(key=lambda r: (r["lift_vs_baseline"], r["count"]), reverse=True)
    return failure_rows, hypothesis_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _short_label(text: str, n: int = 54) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    if len(t) <= n:
        return t
    return t[: n - 1] + "…"


def export_canvas_graph(
    graph: dict[str, Any],
    path: Path,
    title: str,
    max_nodes: int = 280,
    max_edges: int = 800,
) -> None:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2))
        return

    node_score = {n["id"]: int(n.get("count", 1)) for n in nodes}
    top_nodes = sorted(nodes, key=lambda n: int(n.get("count", 0)), reverse=True)[:max_nodes]
    node_ids = {n["id"] for n in top_nodes}

    top_edges = [
        e
        for e in sorted(edges, key=lambda e: int(e.get("count", 0)), reverse=True)
        if e["source"] in node_ids and e["target"] in node_ids
    ][:max_edges]

    canvas_nodes = []
    canvas_edges = []

    cols = 6
    w, h = 260, 110
    xgap, ygap = 40, 40
    for i, n in enumerate(top_nodes):
        r, c = divmod(i, cols)
        x = c * (w + xgap)
        y = 120 + r * (h + ygap)
        nid = n["id"]
        label = _short_label(nid)
        low = nid.lower()
        color = "#6b7280"
        if any(k in low for k in ("error", "fail", "warn", "blocked", "risk")):
            color = "#ef4444"
        elif any(k in low for k in ("diagnosis", "decision", "strategy", "plan", "hypothesis")):
            color = "#3b82f6"
        elif any(k in low for k in ("success", "complete", "resolved", "fixed")):
            color = "#22c55e"
        canvas_nodes.append(
            {
                "id": nid,
                "type": "text",
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "text": f"{label}\\ncount={node_score.get(nid, 0)}",
                "color": color,
            }
        )

    canvas_nodes.append(
        {
            "id": "__title__",
            "type": "text",
            "x": 0,
            "y": 0,
            "width": 1000,
            "height": 80,
            "text": title,
            "color": "#111827",
        }
    )

    for i, e in enumerate(top_edges):
        canvas_edges.append(
            {
                "id": f"e{i}",
                "fromNode": e["source"],
                "toNode": e["target"],
                "label": str(e.get("count", 1)),
            }
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"nodes": canvas_nodes, "edges": canvas_edges}, indent=2))


def export_canvas_motifs(rows: list[dict[str, Any]], path: Path, title: str, score_key: str) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"nodes": [], "edges": []}, indent=2))
        return
    rows = rows[:220]
    nodes: dict[str, dict[str, Any]] = {}
    for r in rows:
        nodes.setdefault(r["from"], {"id": r["from"], "count": 0})
        nodes.setdefault(r["to"], {"id": r["to"], "count": 0})
        nodes[r["from"]]["count"] += int(r.get("count", 1))
        nodes[r["to"]]["count"] += int(r.get("count", 1))
    graph = {
        "nodes": [{"id": k, "count": v["count"]} for k, v in nodes.items()],
        "edges": [
            {"source": r["from"], "target": r["to"], "count": r.get("count", 1), "score": r.get(score_key, 0)}
            for r in rows
        ],
    }
    export_canvas_graph(graph, path, title, max_nodes=220, max_edges=500)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build strategy-focused progress/handoff graph")
    ap.add_argument("--root", type=Path, default=Path("/mnt/raid0/llm/claude"))
    ap.add_argument("--output-dir", type=Path, default=Path("logs/strategy_graph"))
    ap.add_argument("--max-events", type=int, default=0, help="Optional cap per parser (0 = no cap)")
    ap.add_argument("--cluster-threshold", type=float, default=0.82)
    ap.add_argument("--semantic-max-payloads", type=int, default=3000)
    ap.add_argument("--export-canvases", action="store_true")
    ap.add_argument("--reuse-events-cache", action="store_true")
    args = ap.parse_args()

    sources = discover_sources(args.root)
    max_events = args.max_events if args.max_events > 0 else None

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    events_cache_path = out / "events.jsonl"

    if args.reuse_events_cache and events_cache_path.exists():
        all_events = load_events_jsonl(events_cache_path)
        all_events.sort(key=_event_sort_key)
    else:
        all_events: list[Event] = []
        for p in sources["progress_jsonl"]:
            all_events.extend(parse_progress_jsonl(p, max_events=max_events))
        for p in sources["progress_md"]:
            all_events.extend(_parse_markdown_rich(p, "progress_md", max_events=max_events))
        for p in sources["handoff_md"]:
            all_events.extend(_parse_markdown_rich(p, "handoff_md", max_events=max_events))
        for p in sources["blocked_tasks_md"]:
            all_events.extend(_parse_markdown_rich(p, "blocked_tasks_md", max_events=max_events))

        all_events = dedupe_events(all_events)
        all_events.sort(key=_event_sort_key)

    with events_cache_path.open("w") as f:
        for e in all_events:
            f.write(json.dumps(e.__dict__) + "\n")

    # 1) Always build raw graph first
    raw_graph, _ = build_raw_graph(all_events)
    fp = dataset_fingerprint(all_events)
    raw_versioned = out / f"raw_graph_{fp}.json"
    if not raw_versioned.exists():
        raw_versioned.write_text(json.dumps(raw_graph, indent=2))
    (out / "raw_graph.json").write_text(json.dumps(raw_graph, indent=2))
    (out / "raw_graph_latest.json").write_text(json.dumps(raw_graph, indent=2))

    payload_counts = Counter(e.payload() for e in all_events)

    variant_builders: dict[str, Callable[[Event], str]] = {
        "category_only": lambda e: e.category,
        "phase_category": lambda e: f"{_phase_of_event(e)}|{e.category}",
        "initiative_phase": lambda e: f"{_initiative_of_event(e)}|{_phase_of_event(e)}",
        "source_doc_category": lambda e: f"{e.source_type}|{_doc_type_of_event(e)}|{e.category}",
        "outcome_category": lambda e: f"{_outcome_polarity_of_event(e)}|{e.category}",
        "strategy_action_category": lambda e: f"{_strategy_action_of_event(e)}|{e.category}",
        "status_category": lambda e: f"{_status_of_event(e)}|{e.category}",
        "section_template": lambda e: f"{e.category}|s:{_section_template_of_event(e)}",
        "payload_template": lambda e: f"{e.category}|p:{_normalize_template(e.payload())}",
    }

    coarse_variants: dict[str, dict[str, Any]] = {}
    for vname, mapper in variant_builders.items():
        g, trans = build_mapped_graph(all_events, mapper)
        frows, hrows = mine_motifs(g, trans)
        coarse_variants[vname] = {
            "graph": g,
            "failure_rows": frows,
            "hypothesis_rows": hrows,
        }

    cmap = cluster_payloads_limited(
        dict(payload_counts),
        threshold=args.cluster_threshold,
        semantic_max_payloads=max(1, args.semantic_max_payloads),
    )
    g10, trans10 = build_mapped_graph(all_events, lambda e: f"{e.category}|c{cmap.get(e.payload(), -1)}")
    f10, h10 = mine_motifs(g10, trans10)
    coarse_variants["embedding_t082"] = {
        "graph": g10,
        "failure_rows": f10,
        "hypothesis_rows": h10,
        "threshold": args.cluster_threshold,
    }

    # 2) Then write coarse variants (must be exactly 10)
    for variant, bundle in coarse_variants.items():
        (out / f"coarse_graph_{variant}.json").write_text(json.dumps(bundle["graph"], indent=2))
        write_csv(out / f"motifs_failure_{variant}.csv", bundle["failure_rows"])
        write_csv(out / f"motifs_hypothesis_{variant}.csv", bundle["hypothesis_rows"])
        if args.export_canvases:
            export_canvas_graph(
                bundle["graph"],
                out / f"canvas_graph_{variant}.canvas",
                f"{variant} strategy graph",
            )
            export_canvas_motifs(
                bundle["failure_rows"],
                out / f"canvas_failure_{variant}.canvas",
                f"{variant} failure motifs",
                "risk_score",
            )
            export_canvas_motifs(
                bundle["hypothesis_rows"],
                out / f"canvas_hypothesis_{variant}.canvas",
                f"{variant} hypothesis motifs",
                "lift_vs_baseline",
            )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_counts": {k: len(v) for k, v in sources.items()},
        "event_count": len(all_events),
        "dataset_fingerprint": fp,
        "raw_nodes": len(raw_graph["nodes"]),
        "raw_edges": len(raw_graph["edges"]),
        "coarse_variants": {},
    }
    for variant, bundle in coarse_variants.items():
        g = bundle["graph"]
        raw_nodes = max(1, len(raw_graph["nodes"]))
        raw_edges = max(1, len(raw_graph["edges"]))
        summary["coarse_variants"][variant] = {
            "nodes": len(g["nodes"]),
            "edges": len(g["edges"]),
            "failure_motifs": len(bundle["failure_rows"]),
            "hypothesis_motifs": len(bundle["hypothesis_rows"]),
            "node_compression": round(len(g["nodes"]) / raw_nodes, 4),
            "edge_compression": round(len(g["edges"]) / raw_edges, 4),
        }

    summary_md = [
        "# Strategy Graph Summary",
        "",
        f"- Generated: {summary['generated_at']}",
        f"- Source files: {summary['source_counts']}",
        f"- Events: {summary['event_count']}",
        f"- Raw graph: {summary['raw_nodes']} nodes / {summary['raw_edges']} edges",
        "- Coarse variants:",
    ]
    for variant, stats in summary["coarse_variants"].items():
        summary_md.append(
            f"  - {variant}: {stats['nodes']} nodes / {stats['edges']} edges "
            f"(failure motifs={stats['failure_motifs']}, hypothesis motifs={stats['hypothesis_motifs']}, "
            f"node_ratio={stats['node_compression']}, edge_ratio={stats['edge_compression']})"
        )

    (out / "summary.md").write_text("\n".join(summary_md) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
