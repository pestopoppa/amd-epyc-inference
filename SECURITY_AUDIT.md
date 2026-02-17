# Security Audit — 2026-02-17

## Findings & Remediations

### CRITICAL: eval() with full `__builtins__` (RCE)

- **File**: `orchestration/procedure_registry.py` (`_execute_python`, line ~835)
- **Issue**: `eval(command, {"__builtins__": __builtins__}, ...)` exposed all Python builtins including `__import__`, `exec`, `open`, `compile`, enabling arbitrary code execution.
- **Fix**: Replaced with `{"__builtins__": {}}` and an explicit allowlist of safe builtins (len, str, int, sorted, etc.). Removed `os` module from eval context.

### CRITICAL: Command injection in `git_status()`

- **File**: `orchestration/tools/code.py` (`git_status`, line ~128)
- **Issue**: `repo_path` interpolated unsanitized into shell command via f-string: `f"cd {repo_path} && git status ..."`. Allows injection via `"; rm -rf / #"`.
- **Fix**: Wrapped with `shlex.quote(repo_path)`.

### HIGH: eval() condition with `os` module

- **File**: `orchestration/procedure_registry.py` (`_eval_condition`, line ~787)
- **Issue**: `os` module available in eval context allows filesystem access via crafted conditions (e.g., `os.system("...")`, `os.remove("...")`).
- **Fix**: Removed `os` from eval context. Added safe type builtins (`len`, `str`, `int`, `bool`) instead.

### HIGH: shell=True with bypassable blocklist

- **File**: `orchestration/tools/code.py` (`run_shell`, lines ~65-98)
- **Issue**: Regex blocklist can be bypassed via newline injection (`\n`), null bytes, command substitution (`$()`), backticks, or encoding tricks.
- **Fix**: Added null-byte/newline/CR rejection before blocklist check. Extended blocklist with `curl|sh`, `wget|sh`, inline script execution, `eval`, backtick substitution, and `$()` patterns. Made patterns case-insensitive.

### MEDIUM: MD5 hashing (weak hash)

- **Files**: 8 files across `src/`, `scripts/`, `orchestration/`
- **Issue**: MD5 is cryptographically broken (collision attacks practical). Used for cache keys, content integrity, deduplication, and graph node IDs.
- **Fix**: Replaced all `hashlib.md5()` with `hashlib.sha256()`.

## Positive Findings (No Action Required)

| Area | Status |
|------|--------|
| YAML parsing | All uses are `yaml.safe_load()` |
| API key management | All from environment variables |
| exec() sandbox (REPL) | AST-based validation with strong restrictions |
| Temp file handling | Uses `tempfile` module properly |
| Path traversal | Tested and guarded in tool handlers |
| Shell execution in procedures | Uses `shlex.split()` (no `shell=True`) |
| SQL injection | Uses parameterized queries throughout |

## Files Modified

| File | Change |
|------|--------|
| `orchestration/procedure_registry.py` | Restricted eval builtins, removed os from context |
| `orchestration/tools/code.py` | shlex.quote in git_status, hardened blocklist |
| `src/prefix_cache.py` | MD5 -> SHA-256 |
| `src/tool_loader.py` | MD5 -> SHA-256 |
| `src/api/routes/chat_delegation.py` | MD5 -> SHA-256 |
| `scripts/benchmark/dataset_adapters.py` | MD5 -> SHA-256 |
| `scripts/corpus/build_index.py` | MD5 -> SHA-256 |
| `scripts/corpus/build_index_v2.py` | MD5 -> SHA-256 |
| `scripts/strategy_graph/build_strategy_graph.py` | MD5 -> SHA-256 |
| `scripts/audit_graph/build_audit_graph.py` | MD5 -> SHA-256 |
