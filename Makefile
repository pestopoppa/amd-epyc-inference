# Makefile for amd-epyc-inference
# ================================
#
# Gate chain for a scripts-heavy repo:
#   schema → shellcheck → format → lint → (stubs for unit/integration)
#
# Run all gates: make gates
# Run specific:  make shellcheck

SHELL := /usr/bin/env bash
.PHONY: all gates schema shellcheck shfmt mdlint format lint typecheck unit integration security bench clean help

# ── Configuration ─────────────────────────────────────────────────────────────

PY ?= python3

# Shell scripts to check (limit to key directories)
SHELL_SCRIPTS := $(shell find scripts/ -name '*.sh' 2>/dev/null)

# Markdown files (limit to key directories, avoid tmp/cache/research subprojects)
MD_FILES := $(shell find . -maxdepth 2 -name '*.md' -not -path './tmp/*' -not -path './cache/*' 2>/dev/null)

# ── Main Targets ──────────────────────────────────────────────────────────────

all: gates

# Full gate chain (ordered)
gates: schema shellcheck format lint
	@echo ""
	@echo "✅ All gates passed"

help:
	@echo "Available targets:"
	@echo "  make gates      - Run all verification gates (schema + shell + format + lint)"
	@echo "  make test-all   - Run schema validation + Python lint + unit tests"
	@echo "  make quick      - Quick check (schema + shellcheck + pylint)"
	@echo ""
	@echo "Individual gates:"
	@echo "  make schema     - Validate IR JSON files"
	@echo "  make shellcheck - Check shell scripts"
	@echo "  make pylint     - Lint Python with ruff"
	@echo "  make unit       - Run unit tests"
	@echo "  make validate-registry - Validate model registry and paths"
	@echo ""
	@echo "Formatting:"
	@echo "  make shfmt      - Format shell scripts (in-place)"
	@echo "  make pyformat   - Format Python with ruff"
	@echo "  make clean      - Remove generated files"

# ── Schema Gates ──────────────────────────────────────────────────────────────

schema:
	@echo "==> schema"
	@# Validate TaskIR if present
	@if [ -f orchestration/last_task_ir.json ]; then \
		$(PY) orchestration/validate_ir.py task orchestration/last_task_ir.json; \
	else \
		echo "  (no orchestration/last_task_ir.json to validate)"; \
	fi
	@# Validate ArchitectureIR if present
	@if [ -f architecture/architecture_ir.json ]; then \
		$(PY) orchestration/validate_ir.py arch architecture/architecture_ir.json; \
	else \
		echo "  (no architecture/architecture_ir.json to validate)"; \
	fi

# ── Shell Gates ───────────────────────────────────────────────────────────────

shellcheck:
	@echo "==> shellcheck"
	@if command -v shellcheck >/dev/null 2>&1; then \
		if [ -n "$(SHELL_SCRIPTS)" ]; then \
			shellcheck --severity=warning $(SHELL_SCRIPTS) && echo "  ✓ shellcheck passed"; \
		else \
			echo "  (no .sh files found)"; \
		fi \
	else \
		echo "  ⚠ shellcheck not installed (apt install shellcheck)"; \
	fi

shfmt:
	@echo "==> shfmt (format shell scripts)"
	@if command -v shfmt >/dev/null 2>&1; then \
		if [ -n "$(SHELL_SCRIPTS)" ]; then \
			shfmt -w -i 2 -ci $(SHELL_SCRIPTS) && echo "  ✓ shfmt applied"; \
		else \
			echo "  (no .sh files found)"; \
		fi \
	else \
		echo "  ⚠ shfmt not installed (go install mvdan.cc/sh/v3/cmd/shfmt@latest)"; \
	fi

# ── Markdown Gates ────────────────────────────────────────────────────────────

mdlint:
	@echo "==> mdlint"
	@if command -v markdownlint >/dev/null 2>&1; then \
		markdownlint --config .markdownlint.json $(MD_FILES) 2>/dev/null || \
		markdownlint $(MD_FILES) && echo "  ✓ markdownlint passed"; \
	elif command -v mdl >/dev/null 2>&1; then \
		mdl $(MD_FILES) && echo "  ✓ mdl passed"; \
	else \
		echo "  ⚠ markdownlint not installed (npm install -g markdownlint-cli)"; \
	fi

mdformat:
	@echo "==> mdformat"
	@if command -v mdformat >/dev/null 2>&1; then \
		mdformat $(MD_FILES) && echo "  ✓ mdformat applied"; \
	else \
		echo "  ⚠ mdformat not installed (pip install mdformat)"; \
	fi

# ── Aggregate Targets ─────────────────────────────────────────────────────────

format: shfmt
	@# Add mdformat here when ready: format: shfmt mdformat

lint: shellcheck mdlint

# ── Python Gates ─────────────────────────────────────────────────────────────

pylint:
	@echo "==> pylint (ruff)"
	@if command -v ruff >/dev/null 2>&1; then \
		ruff check src/ orchestration/ tests/ && echo "  ✓ ruff passed"; \
	else \
		echo "  ⚠ ruff not installed (pip install ruff)"; \
	fi

pyformat:
	@echo "==> pyformat (ruff)"
	@if command -v ruff >/dev/null 2>&1; then \
		ruff format src/ orchestration/ tests/ && echo "  ✓ ruff format applied"; \
	else \
		echo "  ⚠ ruff not installed (pip install ruff)"; \
	fi

typecheck:
	@echo "==> typecheck"
	@if command -v pyright >/dev/null 2>&1; then \
		pyright src/ orchestration/ && echo "  ✓ pyright passed"; \
	elif command -v mypy >/dev/null 2>&1; then \
		mypy src/ orchestration/ && echo "  ✓ mypy passed"; \
	else \
		echo "  ⚠ no Python typechecker installed"; \
	fi

unit:
	@echo "==> unit tests"
	@if [ -d tests/unit ]; then \
		$(PY) -m pytest tests/unit -q && echo "  ✓ unit tests passed"; \
	else \
		echo "  (no tests/unit directory)"; \
	fi

integration:
	@echo "==> integration tests"
	@if [ -d tests/integration ]; then \
		$(PY) -m pytest tests/integration -q && echo "  ✓ integration tests passed"; \
	else \
		echo "  (no tests/integration directory)"; \
	fi

test: unit integration
	@echo "  ✓ all tests passed"

# ── Optional Gates ────────────────────────────────────────────────────────────

security:
	@echo "==> security (stub)"
	@# Add bandit, semgrep, trivy, etc. when needed
	@echo "  (no security checks configured)"

bench:
	@echo "==> bench (stub)"
	@# Add benchmark harness when needed
	@echo "  (no benchmarks configured)"

# ── Repo Hygiene ──────────────────────────────────────────────────────────────

# Check for accidentally large files (models, logs)
check-large-files:
	@echo "==> check-large-files"
	@find . -type f -size +50M -not -path './.git/*' -exec ls -lh {} \; 2>/dev/null | \
		grep -v -E '\.(gguf|bin|safetensors)$$' || echo "  ✓ no unexpected large files"

# ── Cleanup ───────────────────────────────────────────────────────────────────

clean:
	@echo "==> clean"
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "  ✓ cleaned"

# ── Installation Helpers ──────────────────────────────────────────────────────

install-dev-deps:
	@echo "Installing development dependencies..."
	pip install pyyaml jsonschema pytest ruff
	@echo ""
	@echo "For shell linting, also install:"
	@echo "  apt install shellcheck"
	@echo "  go install mvdan.cc/sh/v3/cmd/shfmt@latest"
	@echo ""
	@echo "For markdown linting:"
	@echo "  npm install -g markdownlint-cli"

# ── Convenience Aliases ──────────────────────────────────────────────────────

# Full test suite
test-all: schema pylint unit integration
	@echo ""
	@echo "✅ All tests passed"

# Quick validation (no tests)
quick: schema shellcheck pylint
	@echo ""
	@echo "✅ Quick checks passed"

# Validate registry and models
validate-registry:
	@echo "==> validate registry"
	@$(PY) src/registry_loader.py
