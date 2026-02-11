# Makefile for amd-epyc-inference
# ================================
#
# Gate chain for a scripts-heavy repo:
#   schema → shellcheck → format → lint → (stubs for unit/integration)
#
# Run all gates: make gates
# Run specific:  make shellcheck

SHELL := /usr/bin/env bash
.PHONY: all gates schema shellcheck shfmt mdlint format lint typecheck coverage unit integration security bench clean help setup bootstrap download-models validate-paths docker-build docker-build-dev docker-run docker-dev docker-test docker-lint docker-clean nix-develop nix-build nix-shell nextplaid-reindex

# ── Configuration ─────────────────────────────────────────────────────────────

PY ?= python3

# Shell scripts to check (limit to key directories)
SHELL_SCRIPTS := $(shell find scripts/ -name '*.sh' 2>/dev/null)

# Markdown files (limit to key directories, avoid tmp/cache/research subprojects)
MD_FILES := $(shell find . -maxdepth 2 -name '*.md' -not -path './tmp/*' -not -path './cache/*' 2>/dev/null)

# ── Main Targets ──────────────────────────────────────────────────────────────

all: gates

# Full gate chain (ordered)
gates: schema shellcheck format lint nextplaid-reindex
	@echo ""
	@echo "✅ All gates passed"

help:
	@echo "Available targets:"
	@echo "  make gates      - Run all verification gates (schema + shell + format + lint)"
	@echo "  make test-all   - Run schema validation + Python lint + unit tests"
	@echo "  make quick      - Quick check (schema + shellcheck + pylint)"
	@echo ""
	@echo "Setup (new installations):"
	@echo "  make setup      - Full setup (bootstrap + validate)"
	@echo "  make bootstrap  - Run bootstrap script (prereqs, dirs, deps)"
	@echo "  make download-models TIER=hot - Download models (hot/warm/all)"
	@echo "  make validate-paths - Check all configured paths exist"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     - Build production Docker image"
	@echo "  make docker-build-dev - Build development Docker image"
	@echo "  make docker-run       - Run orchestrator API in Docker"
	@echo "  make docker-dev       - Interactive development shell in Docker"
	@echo "  make docker-test      - Run tests in Docker"
	@echo "  make docker-clean     - Remove Docker images and containers"
	@echo ""
	@echo "Nix:"
	@echo "  make nix-develop      - Enter Nix development shell"
	@echo "  make nix-build        - Build llama-cpp via Nix"
	@echo ""
	@echo "Individual gates:"
	@echo "  make schema     - Validate IR JSON files"
	@echo "  make shellcheck - Check shell scripts"
	@echo "  make pylint     - Lint Python with ruff"
	@echo "  make unit       - Run unit tests"
	@echo "  make validate-registry - Validate model registry and paths"
	@echo "  make check-memory - Check available RAM before tests (prevents crashes)"
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

coverage:
	@echo "==> coverage"
	@$(PY) -m pytest tests/ -q --cov=src --cov-report=term-missing --cov-fail-under=30 \
		&& echo "  ✓ coverage passed (≥30%)"

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

# ── NextPLAID Index ──────────────────────────────────────────────────────────

# Re-index changed files into NextPLAID (skips gracefully if container not running)
nextplaid-reindex:
	@echo "==> nextplaid-reindex"
	@if curl -sf http://localhost:8088/health >/dev/null 2>&1; then \
		$(PY) scripts/nextplaid/reindex_changed.py && echo "  ✓ nextplaid reindex complete"; \
	else \
		echo "  ⚠ NextPLAID not running on :8088 (skipping)"; \
	fi

# ── Repo Hygiene ──────────────────────────────────────────────────────────────

# Check for accidentally large files (models, logs)
check-large-files:
	@echo "==> check-large-files"
	@find . -type f -size +50M -not -path './.git/*' -exec ls -lh {} \; 2>/dev/null | \
		grep -v -E '\.(gguf|bin|safetensors)$$' || echo "  ✓ no unexpected large files"

# Check available memory before running tests
# DANGER: pytest -n auto on 192-thread machine spawns ~192 workers that may load models
check-memory:
	@echo "==> check-memory"
	@FREE_GB=$$(python3 -c "import psutil; print(int(psutil.virtual_memory().available / (1024**3)))" 2>/dev/null || echo "unknown"); \
	if [ "$$FREE_GB" = "unknown" ]; then \
		echo "  ⚠ psutil not installed - cannot check memory (pip install psutil)"; \
	elif [ "$$FREE_GB" -lt 100 ]; then \
		echo "  ✗ DANGER: Only $${FREE_GB}GB free RAM!"; \
		echo "    Tests may crash the machine if they load models."; \
		echo "    Free up memory or wait for other processes."; \
		exit 1; \
	elif [ "$$FREE_GB" -lt 200 ]; then \
		echo "  ⚠ Low memory: $${FREE_GB}GB free (recommend 200GB+)"; \
	else \
		echo "  ✓ Memory OK: $${FREE_GB}GB free"; \
	fi

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

# Full test suite (runs memory check first to avoid crashes)
test-all: check-memory schema pylint unit integration
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

# ── Setup Targets ────────────────────────────────────────────────────────────

# Full setup for new installations
setup: bootstrap validate-paths gates
	@echo ""
	@echo "✅ Setup complete! See docs/SETUP.md for next steps."

# Run bootstrap script
bootstrap:
	@echo "==> bootstrap"
	@./scripts/setup/bootstrap.sh

# Download models (use TIER=hot, TIER=warm, or TIER=all)
TIER ?= hot
download-models:
	@echo "==> download-models (tier=$(TIER))"
	@$(PY) scripts/setup/download_models.py --tier $(TIER)

# Validate all configured paths exist
validate-paths:
	@echo "==> validate-paths"
	@$(PY) -c "\
from src.config import get_config; \
import sys; \
c = get_config(); \
errors = []; \
for name in ['project_root', 'llm_root', 'models_dir', 'cache_dir', 'tmp_dir', 'llama_cpp_bin']: \
    p = getattr(c.paths, name, None); \
    if p and not p.exists(): \
        errors.append(f'  ✗ {name}: {p}'); \
    elif p: \
        print(f'  ✓ {name}: {p}'); \
if errors: \
    print('Missing paths:'); \
    print('\n'.join(errors)); \
    print('\nRun: ./scripts/setup/bootstrap.sh --create-dirs'); \
    sys.exit(1); \
"

# ── Docker Targets ──────────────────────────────────────────────────────────

# Build production Docker image
docker-build:
	@echo "==> docker-build"
	docker build -t orchestrator:latest .

# Build development Docker image
docker-build-dev:
	@echo "==> docker-build-dev"
	docker build -f Dockerfile.dev -t orchestrator-dev:latest .

# Run production API in Docker
docker-run:
	@echo "==> docker-run (orchestrator API on port 8000)"
	docker-compose up orchestrator

# Run development shell in Docker
docker-dev:
	@echo "==> docker-dev (interactive shell)"
	docker-compose run --rm dev

# Run tests in Docker
docker-test:
	@echo "==> docker-test"
	docker-compose run --rm test

# Run linter in Docker
docker-lint:
	@echo "==> docker-lint"
	docker-compose run --rm lint

# Clean up Docker images and containers
docker-clean:
	@echo "==> docker-clean"
	docker-compose down --rmi local --volumes --remove-orphans 2>/dev/null || true
	docker rmi orchestrator:latest orchestrator-dev:latest 2>/dev/null || true
	@echo "  ✓ Docker cleanup complete"

# ── Nix Targets ─────────────────────────────────────────────────────────────

# Enter Nix development shell
nix-develop:
	@echo "==> nix-develop"
	@if command -v nix >/dev/null 2>&1; then \
		nix develop; \
	else \
		echo "  ✗ Nix not installed. See: https://nixos.org/download.html"; \
		exit 1; \
	fi

# Build llama-cpp via Nix
nix-build:
	@echo "==> nix-build"
	@if command -v nix >/dev/null 2>&1; then \
		nix build .#llama-cpp; \
	else \
		echo "  ✗ Nix not installed"; \
		exit 1; \
	fi

# Alias for nix-develop
nix-shell: nix-develop
