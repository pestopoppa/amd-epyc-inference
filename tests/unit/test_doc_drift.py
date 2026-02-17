"""Tests for scripts/validate/validate_doc_drift.py."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from scripts.validate.validate_doc_drift import (
    check_makefile_drift,
    check_path_drift,
    check_port_drift,
    extract_make_refs_from_docs,
    extract_markdown_links,
    extract_phony_targets,
    extract_port_map_from_code,
    extract_port_table_from_docs,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Port extraction helpers
# ---------------------------------------------------------------------------

SAMPLE_PORT_MAP_CODE = dedent("""\
    PORT_MAP = {
        "frontdoor": 8080,
        "coder_escalation": 8081,
        "worker_explore": 8082,
        "architect_general": 8083,
        "embedder": 8090,
        "orchestrator": 8000,
    }
""")

SAMPLE_PORT_TABLE_DOC = dedent("""\
    ### Local Model Routing (Orchestrator)

    | Task Type | Model | Port | Speed |
    |-----------|-------|------|-------|
    | Interactive chat | Qwen3-Coder-30B | 8080 | 47 t/s |
    | Code gen | Qwen2.5-Coder-32B | 8081 | 39 t/s |
    | Explore | Qwen2.5-7B | 8082 | 44 t/s |
    | Architecture | Qwen3-235B | 8083 | 6.1 t/s |

    ---
""")


def test_extract_port_map_from_code(tmp_path: Path):
    f = tmp_path / "stack.py"
    f.write_text(SAMPLE_PORT_MAP_CODE)
    ports = extract_port_map_from_code(f)
    assert ports[8080] == "frontdoor"
    assert ports[8081] == "coder_escalation"
    assert 8090 in ports  # embedder
    assert 8000 in ports  # orchestrator


def test_extract_port_map_missing_var(tmp_path: Path):
    f = tmp_path / "empty.py"
    f.write_text("x = 1\n")
    assert extract_port_map_from_code(f) == {}


def test_extract_port_table_from_docs():
    ports = extract_port_table_from_docs(SAMPLE_PORT_TABLE_DOC)
    assert ports[8080] == "Interactive chat"
    assert ports[8081] == "Code gen"
    assert len(ports) == 4


def test_extract_port_table_multi_port():
    doc = dedent("""\
        ### Local Model Routing (Orchestrator)

        | Task Type | Model | Port | Speed |
        |-----------|-------|------|-------|
        | Vision | VL-7B / VL-30B | 8086/8087 | ~15 t/s |

    """)
    ports = extract_port_table_from_docs(doc)
    assert 8086 in ports
    assert 8087 in ports


# ---------------------------------------------------------------------------
# Markdown link extraction
# ---------------------------------------------------------------------------

def test_extract_markdown_links_relative():
    text = "See [Guide](docs/GUIDE.md) and [API](https://example.com)"
    links = extract_markdown_links(text)
    assert len(links) == 1
    assert links[0] == ("Guide", "docs/GUIDE.md")


def test_extract_markdown_links_strips_anchor():
    text = "[Section](docs/FILE.md#heading)"
    links = extract_markdown_links(text)
    assert links[0] == ("Section", "docs/FILE.md")


def test_extract_markdown_links_skips_anchor_only():
    text = "[Jump](#section)"
    links = extract_markdown_links(text)
    assert len(links) == 0


# ---------------------------------------------------------------------------
# Makefile target extraction
# ---------------------------------------------------------------------------

SAMPLE_MAKEFILE = dedent("""\
    .PHONY: all gates schema shellcheck clean
    .PHONY: unit integration test-all

    all: gates
    gates: schema shellcheck
    """)


def test_extract_phony_targets(tmp_path: Path):
    f = tmp_path / "Makefile"
    f.write_text(SAMPLE_MAKEFILE)
    targets = extract_phony_targets(f)
    assert "gates" in targets
    assert "schema" in targets
    assert "test-all" in targets
    assert "clean" in targets


def test_extract_make_refs_from_docs():
    text = "Run `make gates` then `make test-all`. Also `make clean`."
    refs = extract_make_refs_from_docs(text)
    assert refs == {"gates", "test-all", "clean"}


# ---------------------------------------------------------------------------
# Integration: drift detection with synthetic files
# ---------------------------------------------------------------------------

def test_port_drift_detects_missing_doc_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port in code but not in docs -> drift error."""
    code = tmp_path / "stack.py"
    code.write_text(dedent("""\
        PORT_MAP = {
            "frontdoor": 8080,
            "worker_fast": 8102,
        }
    """))
    doc = tmp_path / "CLAUDE.md"
    doc.write_text(dedent("""\
        ### Local Model Routing (Orchestrator)

        | Task Type | Model | Port | Speed |
        |-----------|-------|------|-------|
        | Chat | M | 8080 | 1 t/s |

        ---
    """))
    monkeypatch.setattr("scripts.validate.validate_doc_drift.ORCHESTRATOR", code)
    monkeypatch.setattr("scripts.validate.validate_doc_drift.CLAUDE_MD", doc)
    errors = check_port_drift()
    assert any("8102" in e for e in errors)


def test_port_drift_detects_extra_doc_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Port in docs but not in code -> drift error."""
    code = tmp_path / "stack.py"
    code.write_text('PORT_MAP = {"frontdoor": 8080}\n')
    doc = tmp_path / "CLAUDE.md"
    doc.write_text(dedent("""\
        ### Local Model Routing (Orchestrator)

        | Task Type | Model | Port | Speed |
        |-----------|-------|------|-------|
        | Chat | M | 8080 | 1 t/s |
        | Extra | N | 9999 | 1 t/s |

        ---
    """))
    monkeypatch.setattr("scripts.validate.validate_doc_drift.ORCHESTRATOR", code)
    monkeypatch.setattr("scripts.validate.validate_doc_drift.CLAUDE_MD", doc)
    errors = check_port_drift()
    assert any("9999" in e for e in errors)


def test_path_drift_detects_broken_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("See [Missing](does/not/exist.md)\n")
    monkeypatch.setattr("scripts.validate.validate_doc_drift.CLAUDE_MD", doc)
    monkeypatch.setattr("scripts.validate.validate_doc_drift.ROOT", tmp_path)
    errors = check_path_drift()
    assert any("does/not/exist.md" in e for e in errors)


def test_path_drift_passes_valid_link(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "GUIDE.md").write_text("hi")
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("See [Guide](docs/GUIDE.md)\n")
    monkeypatch.setattr("scripts.validate.validate_doc_drift.CLAUDE_MD", doc)
    monkeypatch.setattr("scripts.validate.validate_doc_drift.ROOT", tmp_path)
    errors = check_path_drift()
    assert errors == []


def test_makefile_drift_detects_missing_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    makefile = tmp_path / "Makefile"
    makefile.write_text(".PHONY: gates clean\n")
    doc = tmp_path / "CLAUDE.md"
    doc.write_text("Run `make gates` and `make nonexistent`.\n")
    monkeypatch.setattr("scripts.validate.validate_doc_drift.MAKEFILE", makefile)
    monkeypatch.setattr("scripts.validate.validate_doc_drift.CLAUDE_MD", doc)
    errors = check_makefile_drift()
    assert any("nonexistent" in e for e in errors)
    assert not any("gates" in e for e in errors)


def test_run_all_checks_aggregates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """run_all_checks returns errors from all three vectors."""
    # Set up a synthetic environment with one error per vector
    code = tmp_path / "stack.py"
    code.write_text('PORT_MAP = {"new_role": 9999}\n')
    makefile = tmp_path / "Makefile"
    makefile.write_text(".PHONY: gates\n")
    doc = tmp_path / "CLAUDE.md"
    doc.write_text(dedent("""\
        See [Missing](no_file.md)

        ### Local Model Routing (Orchestrator)

        | Task Type | Model | Port | Speed |
        |-----------|-------|------|-------|

        ---

        Run `make bogus`.
    """))
    monkeypatch.setattr("scripts.validate.validate_doc_drift.ROOT", tmp_path)
    monkeypatch.setattr("scripts.validate.validate_doc_drift.CLAUDE_MD", doc)
    monkeypatch.setattr("scripts.validate.validate_doc_drift.ORCHESTRATOR", code)
    monkeypatch.setattr("scripts.validate.validate_doc_drift.MAKEFILE", makefile)
    errors = run_all_checks()
    categories = {e.split(":")[0] for e in errors}
    assert "path-drift" in categories
    assert "makefile-drift" in categories
