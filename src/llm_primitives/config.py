"""Configuration for LLM primitives."""


def _llm_cfg():
    from src.config import get_config
    return get_config().llm


class LLMPrimitivesConfig:
    """Configuration for LLM primitives.

    Defaults sourced from centralized config (src.config.LLMConfig).
    """

    def __init__(
        self,
        output_cap: int | None = None,
        batch_parallelism: int | None = None,
        call_timeout: int | None = None,
        mock_response_prefix: str | None = None,
        max_recursion_depth: int | None = None,
        default_prompt_rate: float | None = None,
        default_completion_rate: float | None = None,
    ):
        cfg = _llm_cfg()
        self.output_cap = output_cap if output_cap is not None else cfg.output_cap
        self.batch_parallelism = batch_parallelism if batch_parallelism is not None else cfg.batch_parallelism
        self.call_timeout = call_timeout if call_timeout is not None else cfg.call_timeout
        self.mock_response_prefix = mock_response_prefix if mock_response_prefix is not None else cfg.mock_response_prefix
        self.max_recursion_depth = max_recursion_depth if max_recursion_depth is not None else cfg.max_recursion_depth
        self.default_prompt_rate = default_prompt_rate if default_prompt_rate is not None else cfg.default_prompt_rate
        self.default_completion_rate = default_completion_rate if default_completion_rate is not None else cfg.default_completion_rate
