#!/usr/bin/env python3
from __future__ import annotations

"""Build a unified audit/progress transition graph.

Ingests:
- agent audit logs (`**/agent_audit.log`)
- structured progress logs (`logs/progress/*.jsonl`)
- markdown progress reports (`progress/**/*.md`, `orchestration/progress/*.md`)

Outputs:
- events.jsonl
- raw_graph.json
- coarse_graph_*.json
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
from typing import Any


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


def discover_sources(root: Path) -> dict[str, list[Path]]:
    agent_logs: list[Path] = []
    progress_jsonl: list[Path] = []
    progress_md: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        p = Path(dirpath)
        s = str(p)

        if "/.git" in s or "__pycache__" in s:
            dirnames[:] = []
            continue

        for fn in filenames:
            fp = p / fn
            if fn == "agent_audit.log":
                agent_logs.append(fp)
            if fn.endswith(".jsonl") and "/logs/progress" in s:
                progress_jsonl.append(fp)
            if fn.endswith(".md") and ("/progress/" in s or s.endswith("/progress") or "/orchestration/progress" in s):
                progress_md.append(fp)

    return {
        "agent_logs": sorted(agent_logs),
        "progress_jsonl": sorted(progress_jsonl),
        "progress_md": sorted(progress_md),
    }


def parse_agent_audit(path: Path, max_events: int | None = None) -> list[Event]:
    events: list[Event] = []
    with path.open("r", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            obj = _safe_json_loads(line.strip())
            if not obj:
                continue
            events.append(
                Event(
                    source_type="agent_audit",
                    source_file=str(path),
                    source_line=i,
                    session_key=obj.get("session") or f"{path.name}:global",
                    category=_normalize_text(obj.get("cat") or "UNKNOWN", 80),
                    level=_normalize_text(obj.get("level") or "UNKNOWN", 40),
                    message=_normalize_text(obj.get("msg") or "", 400),
                    details=_normalize_text(obj.get("details") or "", 1000),
                    timestamp=_normalize_text(obj.get("ts") or "", 80),
                    ordinal=i,
                )
            )
            if max_events and len(events) >= max_events:
                break
    return events


def parse_progress_jsonl(path: Path, max_events: int | None = None) -> list[Event]:
    events: list[Event] = []
    with path.open("r", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            obj = _safe_json_loads(line.strip())
            if not obj:
                continue
            data = obj.get("data") if isinstance(obj.get("data"), dict) else {}
            msg = data.get("objective") or data.get("reason") or data.get("strategy") or ""
            details_parts = [
                f"outcome={obj.get('outcome') or ''}",
                f"outcome_details={obj.get('outcome_details') or ''}",
                f"agent_role={obj.get('agent_role') or ''}",
                f"agent_tier={obj.get('agent_tier') or ''}",
            ]
            events.append(
                Event(
                    source_type="progress_jsonl",
                    source_file=str(path),
                    source_line=i,
                    session_key=f"task:{obj.get('task_id') or 'unknown'}",
                    category=_normalize_text(obj.get("event_type") or "UNKNOWN", 80),
                    level="INFO",
                    message=_normalize_text(str(msg), 400),
                    details=_normalize_text(" | ".join(details_parts), 1000),
                    timestamp=_normalize_text(obj.get("timestamp") or "", 80),
                    ordinal=i,
                )
            )
            if max_events and len(events) >= max_events:
                break
    return events


def parse_progress_markdown(path: Path, max_events: int | None = None) -> list[Event]:
    events: list[Event] = []
    task_re = re.compile(r"^-\s*\[[xX ]\]\s*(.+)$")
    bullet_re = re.compile(r"^[-*]\s+(.+)$")

    with path.open("r", errors="ignore") as f:
        for i, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue

            cat = None
            msg = ""
            level = "INFO"

            m = task_re.match(line)
            if m:
                msg = m.group(1)
                cat = "progress_task_checked" if line.lower().startswith("- [x]") else "progress_task_open"
            else:
                m = bullet_re.match(line)
                if m:
                    msg = m.group(1)
                    cat = "progress_bullet"
                elif line.startswith("#"):
                    msg = line.lstrip("#").strip()
                    cat = "progress_heading"

            if not cat:
                continue

            low = msg.lower()
            if any(k in low for k in ("fail", "error", "blocked", "regression", "broken")):
                level = "WARN"
            if any(k in low for k in ("critical", "severe", "outage", "incident")):
                level = "ERROR"

            events.append(
                Event(
                    source_type="progress_md",
                    source_file=str(path),
                    source_line=i,
                    session_key=f"report:{path}",
                    category=cat,
                    level=level,
                    message=_normalize_text(msg, 400),
                    details="",
                    timestamp="",
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
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
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
    """LSH-like signature from largest absolute dimensions."""
    idxs = sorted(range(len(emb)), key=lambda i: abs(emb[i]), reverse=True)[:topk]
    parts = [f"{i}:{'+' if emb[i] >= 0 else '-'}" for i in idxs]
    return "|".join(parts)


def cluster_payloads(payloads: list[str], threshold: float = 0.82) -> dict[str, int]:
    """Approximate online cosine clustering (embeddings-only, LSH-pruned)."""
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

        # Compare only against candidate clusters in the same bucket.
        # Falls back to all clusters if bucket is empty.
        candidates = bucket_to_clusters.get(sig) or range(len(centers))
        for i in candidates:
            c = centers[i]
            sim = _cosine(emb, c)
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
    """Scale semantic clustering by training on frequent payloads, then bucketing tail."""
    ranked = sorted(payload_counts.items(), key=lambda kv: kv[1], reverse=True)
    train_payloads = [p for p, _ in ranked[:semantic_max_payloads]]
    tail_payloads = [p for p, _ in ranked[semantic_max_payloads:]]

    assign = cluster_payloads(train_payloads, threshold=threshold)
    if not tail_payloads:
        return assign

    # Map signatures from trained items to dominant cluster
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


def build_graphs(events: list[Event], cluster_map: dict[str, int]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    by_session: dict[str, list[Event]] = defaultdict(list)
    for e in events:
        by_session[e.session_key].append(e)
    for sk in by_session:
        by_session[sk].sort(key=_event_sort_key)

    fine_nodes: dict[str, dict[str, Any]] = {}
    coarse_nodes: dict[str, dict[str, Any]] = {}
    fine_edges: Counter[tuple[str, str]] = Counter()
    coarse_edges: Counter[tuple[str, str]] = Counter()

    transition_rows: list[dict[str, Any]] = []

    for sk, evs in by_session.items():
        for e in evs:
            payload = e.payload()
            cid = cluster_map.get(payload, -1)
            fine_id = f"{e.category}|{hashlib.md5(payload.encode('utf-8')).hexdigest()[:12]}"
            coarse_id = f"{e.category}|c{cid}"

            fine_nodes.setdefault(
                fine_id,
                {
                    "id": fine_id,
                    "category": e.category,
                    "level": e.level,
                    "cluster_id": cid,
                    "sample_message": e.message,
                    "source_type": e.source_type,
                    "count": 0,
                },
            )
            fine_nodes[fine_id]["count"] += 1

            coarse_nodes.setdefault(
                coarse_id,
                {
                    "id": coarse_id,
                    "category": e.category,
                    "cluster_id": cid,
                    "count": 0,
                },
            )
            coarse_nodes[coarse_id]["count"] += 1

        for a, b in zip(evs, evs[1:]):
            pa = a.payload()
            pb = b.payload()
            ca = cluster_map.get(pa, -1)
            cb = cluster_map.get(pb, -1)
            fa = f"{a.category}|{hashlib.md5(pa.encode('utf-8')).hexdigest()[:12]}"
            fb = f"{b.category}|{hashlib.md5(pb.encode('utf-8')).hexdigest()[:12]}"
            coa = f"{a.category}|c{ca}"
            cob = f"{b.category}|c{cb}"
            fine_edges[(fa, fb)] += 1
            coarse_edges[(coa, cob)] += 1
            transition_rows.append(
                {
                    "session_key": sk,
                    "from": coa,
                    "to": cob,
                    "from_category": a.category,
                    "to_category": b.category,
                    "from_level": a.level,
                    "to_level": b.level,
                }
            )

    fine = {
        "nodes": list(fine_nodes.values()),
        "edges": [{"source": s, "target": t, "count": c} for (s, t), c in fine_edges.items()],
    }
    coarse = {
        "nodes": list(coarse_nodes.values()),
        "edges": [{"source": s, "target": t, "count": c} for (s, t), c in coarse_edges.items()],
    }
    return fine, coarse, transition_rows


def build_raw_graph(events: list[Event]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build raw payload-level graph without semantic clustering."""
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
            node_id = f"{e.category}|{hashlib.md5(payload.encode('utf-8')).hexdigest()[:12]}"
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
            na = f"{a.category}|{hashlib.md5(pa.encode('utf-8')).hexdigest()[:12]}"
            nb = f"{b.category}|{hashlib.md5(pb.encode('utf-8')).hexdigest()[:12]}"
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


def build_category_graph(events: list[Event]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build category-only coarse graph."""
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
            nodes.setdefault(
                e.category,
                {
                    "id": e.category,
                    "category": e.category,
                    "count": 0,
                },
            )
            nodes[e.category]["count"] += 1
        for a, b in zip(evs, evs[1:]):
            edges[(a.category, b.category)] += 1
            transition_rows.append(
                {
                    "session_key": sk,
                    "from": a.category,
                    "to": b.category,
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


def _normalize_template(text: str) -> str:
    t = text.lower()
    t = re.sub(r"/[a-z0-9._/\-]+", "<path>", t)
    t = re.sub(r"\b[0-9a-f]{8,}\b", "<hex>", t)
    t = re.sub(r"\b\d+\b", "<num>", t)
    t = re.sub(r"\b(chat|task|ses)-[a-z0-9]+\b", "<id>", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:140] if t else "<empty>"


def _phase_of(category: str) -> str:
    c = category.lower()
    if "session" in c:
        return "session"
    if "task" in c:
        return "task"
    if "route" in c or "escalat" in c or "delegat" in c:
        return "routing"
    if "gate" in c or "schema" in c or "lint" in c or "test" in c:
        return "quality"
    if "observe" in c or "metric" in c or "diagnostic" in c:
        return "observe"
    if "file" in c or "edit" in c or "docs" in c:
        return "artifact"
    if "error" in c or "warning" in c or "fail" in c:
        return "failure"
    if "decision" in c or "finding" in c or "hypothesis" in c:
        return "reasoning"
    if "memory" in c or "q_" in c:
        return "learning"
    return "other"


def _topic_of(e: Event) -> str:
    text = f"{e.category} {e.message} {e.details}".lower()
    rules = [
        ("error_failure", ("error", "fail", "corrupt", "timeout", "sigsegv", "oom", "blocked", "mismatch")),
        ("decision_strategy", ("decision", "recommend", "hypothesis", "strategy", "plan")),
        ("task_execution", ("task_start", "task_end", "task_completed", "run", "build", "benchmark")),
        ("file_docs_ops", ("file_modify", "file_create", "edit", "docs", "patch", "write")),
        ("routing_escalation", ("routing", "escalation", "delegate", "architect", "coder", "worker")),
        ("observation_metrics", ("observe", "metric", "count", "detected", "stats", "inventory")),
        ("session_lifecycle", ("session_start", "session_end", "resume", "archive")),
        ("quality_validation", ("gate", "schema", "lint", "unit", "integration", "validate")),
    ]
    for name, kws in rules:
        if any(k in text for k in kws):
            return name
    return "other"


def build_mapped_graph(events: list[Event], mapper) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        e for e in sorted(edges, key=lambda e: int(e.get("count", 0)), reverse=True)
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
        if any(k in low for k in ("error", "fail", "warn", "blocked")):
            color = "#ef4444"
        elif any(k in low for k in ("decision", "hypothesis", "finding")):
            color = "#3b82f6"
        elif any(k in low for k in ("task", "complete", "success")):
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

    title_id = "__title__"
    canvas_nodes.append(
        {
            "id": title_id,
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
        "edges": [{"source": r["from"], "target": r["to"], "count": r.get("count", 1), "score": r.get(score_key, 0)} for r in rows],
    }
    export_canvas_graph(graph, path, title, max_nodes=220, max_edges=500)


def is_failure_like(category: str, level: str, node_id: str) -> bool:
    txt = f"{category} {level} {node_id}".lower()
    failure_kw = (
        "error",
        "warning",
        "failed",
        "task_failed",
        "gate_failed",
        "escalation_failed",
        "blocked",
        "regression",
    )
    return any(k in txt for k in failure_kw)


def is_success_like(category: str, node_id: str) -> bool:
    txt = f"{category} {node_id}".lower()
    success_kw = ("task_end", "task_completed", "success", "resolved", "fixed", "done", "complete")
    return any(k in txt for k in success_kw)


def mine_motifs(coarse_graph: dict[str, Any], transitions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    node_map = {n["id"]: n for n in coarse_graph["nodes"]}
    succ_targets = [t for t in transitions if is_success_like(t["to_category"], t["to"])]
    baseline = (len(succ_targets) / len(transitions)) if transitions else 0.0

    edge_stats: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"count": 0.0, "succ": 0.0, "sess": 0.0})
    sess_seen: dict[tuple[str, str], set[str]] = defaultdict(set)

    for t in transitions:
        key = (t["from"], t["to"])
        edge_stats[key]["count"] += 1.0
        if is_success_like(t["to_category"], t["to"]):
            edge_stats[key]["succ"] += 1.0
        sess_seen[key].add(t["session_key"])

    failure_rows: list[dict[str, Any]] = []
    hypothesis_rows: list[dict[str, Any]] = []

    for (src, dst), st in edge_stats.items():
        src_node = node_map.get(src, {})
        dst_node = node_map.get(dst, {})
        sess = len(sess_seen[(src, dst)])
        succ_rate = st["succ"] / st["count"] if st["count"] else 0.0
        lift = succ_rate - baseline

        src_cat = str(src_node.get("category", ""))
        dst_cat = str(dst_node.get("category", ""))

        if is_failure_like(src_cat, "", src) or is_failure_like(dst_cat, "", dst):
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

        if "decision" in src_cat.lower() or "finding" in src_cat.lower() or "observe" in src_cat.lower():
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
        with path.open("w", newline="") as f:
            f.write("\n")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build unified audit/progress graph")
    ap.add_argument("--root", type=Path, default=Path("/mnt/raid0/llm/claude"))
    ap.add_argument("--output-dir", type=Path, default=Path("logs/audit_graph"))
    ap.add_argument("--max-events", type=int, default=0, help="Optional cap per source parser (0 = no cap)")
    ap.add_argument("--cluster-threshold", type=float, default=0.82)
    ap.add_argument(
        "--coarse-thresholds",
        type=str,
        default="0.75,0.82,0.90",
        help="Comma-separated embedding clustering thresholds.",
    )
    ap.add_argument(
        "--semantic-coarse",
        action="store_true",
        help="Enable embeddings-only semantic coarse variants (can be expensive on large logs).",
    )
    ap.add_argument(
        "--semantic-max-payloads",
        type=int,
        default=3000,
        help="Max unique payloads for full clustering; long tail is signature-bucket assigned.",
    )
    ap.add_argument(
        "--export-canvases",
        action="store_true",
        help="Export JSON Canvas files for Obsidian inspection.",
    )
    ap.add_argument(
        "--reuse-events-cache",
        action="store_true",
        help="Reuse output-dir/events.jsonl if present instead of reparsing sources.",
    )
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
        all_events = []
        for p in sources["agent_logs"]:
            all_events.extend(parse_agent_audit(p, max_events=max_events))
        for p in sources["progress_jsonl"]:
            all_events.extend(parse_progress_jsonl(p, max_events=max_events))
        for p in sources["progress_md"]:
            all_events.extend(parse_progress_markdown(p, max_events=max_events))

        all_events = dedupe_events(all_events)
        all_events.sort(key=_event_sort_key)

    # Raw graph (payload-level nodes) - always on, fast
    raw_graph, _ = build_raw_graph(all_events)

    payload_counts = Counter(e.payload() for e in all_events)
    top_payloads = {p for p, _ in payload_counts.most_common(250)}

    # Build ~10 coarse-graining strategies
    variant_builders: dict[str, Any] = {
        "category_only": lambda e: e.category,
        "source_category": lambda e: f"{e.source_type}|{e.category}",
        "level_category": lambda e: f"{e.level}|{e.category}",
        "phase_category": lambda e: f"{_phase_of(e.category)}|{e.category}",
        "keyword_topic": lambda e: _topic_of(e),
        "category_msg_template": lambda e: f"{e.category}|m:{_normalize_template(e.message)}",
        "category_details_template": lambda e: f"{e.category}|d:{_normalize_template(e.details)}",
        "category_both_template": lambda e: f"{e.category}|b:{_normalize_template(e.payload())}",
        "category_top_payload": lambda e: (
            f"{e.category}|p:{hashlib.md5(e.payload().encode('utf-8')).hexdigest()[:8]}"
            if e.payload() in top_payloads
            else f"{e.category}|p:OTHER"
        ),
        "category_sig2": lambda e: f"{e.category}|s2:{_bucket_signature(_hashed_embedding(e.payload()), topk=2)}",
        "category_sig3": lambda e: f"{e.category}|s3:{_bucket_signature(_hashed_embedding(e.payload()), topk=3)}",
    }

    coarse_variants: dict[str, dict[str, Any]] = {}
    for vname, vmap in variant_builders.items():
        g, trans = build_mapped_graph(all_events, vmap)
        frows, hrows = mine_motifs(g, trans)
        coarse_variants[vname] = {
            "graph": g,
            "failure_rows": frows,
            "hypothesis_rows": hrows,
        }
    if args.semantic_coarse:
        # embeddings-only variants
        coarse_thresholds: list[float] = []
        for x in args.coarse_thresholds.split(","):
            x = x.strip()
            if not x:
                continue
            try:
                coarse_thresholds.append(float(x))
            except ValueError:
                pass
        if not coarse_thresholds:
            coarse_thresholds = [args.cluster_threshold]

        for t in coarse_thresholds:
            cmap = cluster_payloads_limited(
                dict(payload_counts),
                threshold=t,
                semantic_max_payloads=max(1, args.semantic_max_payloads),
            )
            _, cgraph, trans = build_graphs(all_events, cmap)
            frows, hrows = mine_motifs(cgraph, trans)
            coarse_variants[f"embedding_t{str(t).replace('.', '')}"] = {
                "graph": cgraph,
                "failure_rows": frows,
                "hypothesis_rows": hrows,
                "threshold": t,
            }

    with events_cache_path.open("w") as f:
        for e in all_events:
            f.write(json.dumps(e.__dict__) + "\n")

    fp = dataset_fingerprint(all_events)
    raw_versioned = out / f"raw_graph_{fp}.json"
    if not raw_versioned.exists():
        raw_versioned.write_text(json.dumps(raw_graph, indent=2))
    (out / "raw_graph.json").write_text(json.dumps(raw_graph, indent=2))
    (out / "raw_graph_latest.json").write_text(json.dumps(raw_graph, indent=2))

    for variant, bundle in coarse_variants.items():
        (out / f"coarse_graph_{variant}.json").write_text(json.dumps(bundle["graph"], indent=2))
        write_csv(out / f"motifs_failure_{variant}.csv", bundle["failure_rows"])
        write_csv(out / f"motifs_hypothesis_{variant}.csv", bundle["hypothesis_rows"])
        if args.export_canvases:
            export_canvas_graph(
                bundle["graph"],
                out / f"canvas_graph_{variant}.canvas",
                f"{variant} graph",
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
        "# Audit Graph Summary",
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
