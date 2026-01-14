"""Request models for the orchestrator API."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""

    prompt: str = Field(..., description="The user prompt to process")
    context: str = Field(default="", description="Optional context to include")
    mock_mode: bool = Field(default=True, description="Use mock responses instead of real inference")
    real_mode: bool = Field(default=False, description="Enable real inference with RadixAttention caching")
    max_turns: int = Field(default=10, ge=1, le=50, description="Maximum orchestration turns")
    role: str = Field(default="frontdoor", description="Initial role to use")
    server_urls: dict[str, str] | None = Field(
        default=None,
        description="Server URLs for real mode (e.g., {'frontdoor': 'http://localhost:8080'})"
    )
    # Extended thinking support (Claude Code parity)
    thinking_budget: int = Field(
        default=0,
        ge=0,
        le=32000,
        description="Token budget for internal reasoning (0=disabled, max=32000)"
    )
    permission_mode: str = Field(
        default="normal",
        description="Permission mode: 'normal', 'auto-accept', or 'plan'"
    )


class GateRequest(BaseModel):
    """Request model for running gates."""

    gate_names: list[str] | None = Field(default=None, description="Specific gates to run (None = all)")
    stop_on_first_failure: bool = Field(default=True, description="Stop after first required gate fails")
    required_only: bool = Field(default=False, description="Only run required gates")
