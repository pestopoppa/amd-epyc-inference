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

# Import from local orchestrator service (escalation roles mapping only)
from src.api.services.orchestrator import ESCALATION_ROLES

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
    # Escalation mapping (from orchestrator service)
    "ESCALATION_ROLES",
    # MemRL services
    "load_optional_imports",
    "ensure_memrl_initialized",
    "score_completed_task",
    "background_cleanup",
]
