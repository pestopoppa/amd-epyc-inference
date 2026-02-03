"""Request models for the orchestrator API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    prompt: str = Field(..., description="The user prompt to process")
    context: str = Field(default="", description="Optional context to include")
    mock_mode: bool = Field(
        default=True, description="Use mock responses instead of real inference"
    )
    real_mode: bool = Field(
        default=False, description="Enable real inference with RadixAttention caching"
    )
    max_turns: int = Field(default=10, ge=1, le=50, description="Maximum orchestration turns")
    role: str = Field(
        default="", description="Initial role to use (empty = auto-route via _classify_and_route)"
    )
    force_role: str | None = Field(
        default=None,
        description="Force routing to a specific role, bypassing all routing logic. "
        "Used by comparative seeding to test specialist quality.",
    )
    force_mode: str | None = Field(
        default=None,
        description="Force execution mode ('direct', 'react', 'repl', or 'delegated'), "
        "bypassing _select_mode heuristics. 'delegated' enables architect delegation "
        "where the architect formulates investigation briefs for faster specialists.",
    )
    server_urls: dict[str, str] | None = Field(
        default=None,
        description="Server URLs for real mode (e.g., {'frontdoor': 'http://localhost:8080'})",
    )
    # Vision support — when set, routes to VL workers (8086/8087)
    image_path: str | None = Field(
        default=None, description="Path to image file for vision tasks (routes to VL worker)"
    )
    image_base64: str | None = Field(
        default=None, description="Base64-encoded image data for vision tasks"
    )
    files: list[str] | None = Field(
        default=None,
        description="List of file paths for multi-file vision/document tasks (archives auto-extracted)",
    )
    # Per-request cache control
    cache_prompt: bool | None = Field(
        default=None,
        description="Override cache_prompt for this request (None=backend default True). "
        "Set to False for benchmark seeding where prefix caching adds overhead.",
    )
    # Extended thinking support (Claude Code parity)
    thinking_budget: int = Field(
        default=0,
        ge=0,
        le=32000,
        description="Token budget for internal reasoning (0=disabled, max=32000)",
    )
    permission_mode: str = Field(
        default="normal", description="Permission mode: 'normal', 'auto-accept', or 'plan'"
    )


class RewardRequest(BaseModel):
    """Request model for injecting external rewards into MemRL."""

    task_description: str = Field(..., description="Description of the task that was scored")
    action: str = Field(..., description="Action taken, e.g. 'frontdoor:direct'")
    reward: float = Field(..., ge=-1.0, le=1.0, description="Reward value (-1.0 to 1.0)")
    context: dict | None = Field(
        default=None, description="Optional metadata (suite, tier, scoring_method)"
    )


class GateRequest(BaseModel):
    """Request model for running gates."""

    gate_names: list[str] | None = Field(
        default=None, description="Specific gates to run (None = all)"
    )
    stop_on_first_failure: bool = Field(
        default=True, description="Stop after first required gate fails"
    )
    required_only: bool = Field(default=False, description="Only run required gates")
