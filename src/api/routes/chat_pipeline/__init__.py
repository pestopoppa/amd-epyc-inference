"""Pipeline stage functions for _handle_chat() (Phase 1b extraction).

Each function corresponds to a named stage in the chat processing pipeline.
Extracted from the 1,091-line _handle_chat() to enable independent testing,
explicit failure signaling, and role-specific timeouts.

Stage functions:
    _route_request()     -- Routing decision, task_id, MemRL logging
    _preprocess()        -- Input formalization
    _init_primitives()   -- LLMPrimitives setup + backend validation
    _execute_mock()      -- Mock mode response
    _execute_vision()    -- Vision pipeline (OCR, VL, multi-file)
    _execute_delegated() -- Architect delegation mode
    _execute_react()     -- ReAct tool loop mode
    _execute_direct()    -- Direct LLM call mode
    _execute_repl()      -- REPL orchestration mode
    _annotate_error()    -- Detect error answers -> set error_code/error_detail

Decomposed from monolithic chat_pipeline.py into:
    routing.py       -- Stages 1-3, 5 (routing, preprocess, init, plan review)
    stages.py        -- Stages 4, 6-9, error annotation
    repl_executor.py -- Stage 10 (REPL orchestration loop)
"""

from src.api.routes.chat_pipeline.routing import (
    _route_request,
    _preprocess,
    _init_primitives,
    _plan_review_gate,
)
from src.api.routes.chat_pipeline.stages import (
    _execute_mock,
    _execute_vision,
    _execute_delegated,
    _parse_plan_steps,
    _execute_proactive,
    _execute_react,
    _execute_direct,
    _annotate_error,
)
from src.api.routes.chat_pipeline.repl_executor import _execute_repl

# Backward-compat re-exports (vision functions were imported here historically)
from src.api.routes.chat_vision import (  # noqa: F401
    _handle_multi_file_vision,
    _handle_vision_request,
    _vision_react_mode_answer,
)

__all__ = [
    "_route_request",
    "_preprocess",
    "_init_primitives",
    "_plan_review_gate",
    "_execute_mock",
    "_execute_vision",
    "_execute_delegated",
    "_parse_plan_steps",
    "_execute_proactive",
    "_execute_react",
    "_execute_direct",
    "_execute_repl",
    "_annotate_error",
]
