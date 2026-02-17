# CLAUDE.md Coverage Matrix

This matrix defines governance boundaries for `CLAUDE.md` files discovered in this repository.

## Governed Files

| Path | Scope | Governance | Action |
|---|---|---|---|
| `CLAUDE.md` | Project root agent policy | governed | Maintain and keep aligned with hooks/skills/docs |

## Related Governance Files

| Path | Purpose |
|---|---|
| `CLAUDE_GUIDE.md` | Human-facing explanation of project-level agent configuration |

## Non-Governed Discovery Classes

| Class | Typical Path Prefix | Governance | Action |
|---|---|---|---|
| Vendor/plugin trees | `config/plugins/` and `config/.claude/plugins/` | external | Do not edit as part of this track |
| Cached plugin snapshots | `config/plugins/cache/` | external | Ignore for migration and checks |
| Backups | `backups/` | archival | Ignore except forensic reference |

## Policy

1. Only governed files are migration targets in this refactor track.
2. Non-governed entries are intentionally excluded to prevent accidental vendor edits.
3. If a new repo-owned `CLAUDE.md` is added, update this matrix and validators in the same change.

## Related Design Doc

- `docs/reference/agent-config/AGENT_FILE_LOGIC.md`
