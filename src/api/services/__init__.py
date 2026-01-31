"""Services for the orchestrator API."""

# Import from unified prompt builders module
from src.prompt_builders import (
    PromptBuilder,
    PromptConfig,
    RootLMPrompt,
    EscalationPrompt,
    StepPrompt,
    build_root_lm_prompt,
    build_escalation_prompt,
    build_step_prompt,
    extract_code_from_response,
    classify_error,
)

from src.roles import Role

# Escalation role mapping (from lower to higher tier)
# Migrated from deleted src/api/services/orchestrator.py facade
ESCALATION_ROLES: dict[Role, Role] = {
    Role.WORKER_GENERAL: Role.CODER_PRIMARY,
    Role.CODER_PRIMARY: Role.ARCHITECT_GENERAL,
    Role.FRONTDOOR: Role.CODER_PRIMARY,
    Role.INGEST_LONG_CONTEXT: Role.ARCHITECT_GENERAL,
}

# Import MemRL services
from src.api.services.memrl import (
    load_optional_imports,
    ensure_memrl_initialized,
    score_completed_task,
    background_cleanup,
)

__all__ = [
    # Prompt builders (from unified module)
    "PromptBuilder",
    "PromptConfig",
    "RootLMPrompt",
    "EscalationPrompt",
    "StepPrompt",
    "build_root_lm_prompt",
    "build_escalation_prompt",
    "build_step_prompt",
    "extract_code_from_response",
    "classify_error",
    # Escalation mapping
    "ESCALATION_ROLES",
    # MemRL services
    "load_optional_imports",
    "ensure_memrl_initialized",
    "score_completed_task",
    "background_cleanup",
]
