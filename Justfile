# Justfile for amd-epyc-inference
# ================================
#
# Requires: just (https://github.com/casey/just)
# Install: cargo install just
#
# Run all gates: just gates
# Run specific:  just shellcheck

set shell := ["bash", "-euo", "pipefail", "-c"]

py := "python3"

# Default recipe
default: gates

# ── Main Targets ──────────────────────────────────────────────────────────────

# Run all verification gates
gates: schema shellcheck lint
    @echo ""
    @echo "✅ All gates passed"

# Show available recipes
help:
    @just --list

# ── Schema Gates ──────────────────────────────────────────────────────────────

# Validate IR JSON files
schema:
    @echo "==> schema"
    @test ! -f orchestration/last_task_ir.json || {{py}} orchestration/validate_ir.py task orchestration/last_task_ir.json
    @test ! -f architecture/architecture_ir.json || {{py}} orchestration/validate_ir.py arch architecture/architecture_ir.json
    @echo "  ✓ schema validation complete"

# ── Shell Gates ───────────────────────────────────────────────────────────────

# Check shell scripts with shellcheck
shellcheck:
    @echo "==> shellcheck"
    @if command -v shellcheck >/dev/null 2>&1; then \
        find . -name '*.sh' -not -path './vendor/*' -exec shellcheck --severity=warning {} + && echo "  ✓ shellcheck passed"; \
    else \
        echo "  ⚠ shellcheck not installed"; \
    fi

# Format shell scripts with shfmt
shfmt:
    @echo "==> shfmt"
    @if command -v shfmt >/dev/null 2>&1; then \
        find . -name '*.sh' -not -path './vendor/*' -exec shfmt -w -i 2 -ci {} + && echo "  ✓ shfmt applied"; \
    else \
        echo "  ⚠ shfmt not installed"; \
    fi

# ── Markdown Gates ────────────────────────────────────────────────────────────

# Lint markdown files
mdlint:
    @echo "==> mdlint"
    @if command -v markdownlint >/dev/null 2>&1; then \
        find . -name '*.md' -not -path './vendor/*' -exec markdownlint {} + && echo "  ✓ markdownlint passed"; \
    else \
        echo "  ⚠ markdownlint not installed"; \
    fi

# ── Aggregate Targets ─────────────────────────────────────────────────────────

# Format all (shell + markdown)
format: shfmt

# Lint all (shellcheck + mdlint)
lint: shellcheck mdlint

# ── Python Gates (stubs) ──────────────────────────────────────────────────────

# Type check Python (stub)
typecheck:
    @echo "==> typecheck (stub)"
    @echo "  (no Python typechecker configured)"

# Run unit tests (stub)
unit:
    @echo "==> unit (stub)"
    @echo "  (no unit tests yet)"

# Run integration tests (stub)
integration:
    @echo "==> integration (stub)"
    @echo "  (no integration tests yet)"

# ── Cleanup ───────────────────────────────────────────────────────────────────

# Remove generated files
clean:
    @echo "==> clean"
    rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    @echo "  ✓ cleaned"

# ── Quick Checks ──────────────────────────────────────────────────────────────

# Quick check (format + lint only, no tests)
quick: format lint
    @echo "✅ Quick checks passed"

# Check for accidentally committed large files
check-large:
    @echo "==> check-large-files"
    @find . -type f -size +50M -not -path './.git/*' | grep -v -E '\.(gguf|bin|safetensors)$' || echo "  ✓ no unexpected large files"
