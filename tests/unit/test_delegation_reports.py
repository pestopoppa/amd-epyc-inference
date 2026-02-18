#!/usr/bin/env python3
"""Unit tests for delegation report artifact helpers."""

from src.delegation_reports import extract_handle_id, load_report, store_report


def test_store_and_load_report_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_DELEGATION_REPORT_DIR", str(tmp_path))
    handle = store_report("hello delegation report", "worker_coder")
    assert handle is not None
    payload = load_report(handle["id"], offset=6, max_chars=10)
    assert payload["ok"] is True
    assert payload["content"] == "delegation report"


def test_extract_handle_id():
    text = "[REPORT_HANDLE id=worker_coder-123-abc chars=99 sha16=abc]\nSummary:\n..."
    assert extract_handle_id(text) == "worker_coder-123-abc"


def test_load_report_invalid_id():
    payload = load_report("../bad", offset=0, max_chars=100)
    assert payload["ok"] is False
    assert payload["error"] == "invalid_report_id"
