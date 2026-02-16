# Plan: Lint Fix (Auto-fixable only)

## Scope
Fix all **auto-fixable** ruff lint errors (268 fixable). Skip F821 (undefined names — mostly forward type refs) and F841 (unused variables — need manual review). Focus on safe, mechanical fixes:
- **F541** (110): f-strings without placeholders → remove `f` prefix
- **F401** (113): unused imports → remove them
- **F811** (1): redefined unused import → remove duplicate
- **W291/W293** (2): trailing/blank-line whitespace → strip

## Steps
1. Record current branch (`main`), create `nightshift/lint-fix` branch
2. Run `ruff check --fix --select F541,W291,W293,F811` (safe auto-fixes)
3. Run `ruff check --fix --select F401` (unused imports — safe with `--fix`)
4. Run `ruff format` to ensure consistent formatting post-fix
5. Run `pytest tests/ -x -q` to verify nothing breaks
6. Commit with trailers
7. Open PR, switch back to `main`
