# Safety Reviewer Workflow

## Review Gate

Approve risky operations only when all checks pass:

1. Filesystem policy compliance (`/mnt/raid0/` for LLM artifacts).
2. Logging enabled (`scripts/utils/agent_log.sh`).
3. Rollback command documented for system changes.
4. Retry count below threshold (max 3 same-command failures).
5. Explicit user confirmation for destructive actions.

## Stop Conditions

- Write path policy violation.
- Missing rollback for high-impact change.
- Repeated retries without new diagnosis.
- Destructive action without explicit approval.
