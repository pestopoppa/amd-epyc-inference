"""Output quality detection classifier.

Extracted from src/api/routes/chat_review.py during Phase 1 classifier
refactoring. Pure text-based heuristics with config-driven thresholds.
No MemRL coupling — reads thresholds from ChatPipelineConfig.
"""

from __future__ import annotations

from src.config import get_config as _get_config


def detect_output_quality_issue(answer: str) -> str | None:
    """Detect quality issues in model output using text-based heuristics.

    Inspired by GenerationMonitor's entropy/repetition signals but operating
    on complete output text (no streaming logits needed). Returns a description
    of the issue if detected, None if output looks fine.

    This is SAFE routing: only triggers on detected failure patterns,
    never on input keywords (which caused Wave 2 regressions).

    Args:
        answer: The model's complete output.

    Returns:
        Issue description if quality problem detected, None otherwise.
    """
    if not answer or len(answer) < 20:
        return None  # Too short to analyze

    words = answer.split()
    n_words = len(words)

    _chat_thresholds = _get_config().chat

    # 1. High n-gram repetition (degeneration loops)
    if n_words >= 20:
        trigrams = [" ".join(words[i : i + 3]) for i in range(n_words - 2)]
        if trigrams:
            unique_ratio = len(set(trigrams)) / len(trigrams)
            if unique_ratio < _chat_thresholds.repetition_unique_ratio:
                return f"high_repetition (unique_ratio={unique_ratio:.2f})"

    # 2. Self-contradictory trace (model says X then says not-X)
    lines = answer.strip().split("\n")
    if n_words >= 50:
        # Check for confused analysis: very short non-empty lines mixed with long ones
        # indicates garbled/confused trace
        short_lines = sum(1 for line in lines if 0 < len(line.strip()) < 10)
        total_lines = sum(1 for line in lines if line.strip())
        if (
            total_lines > 5
            and short_lines / total_lines > _chat_thresholds.garbled_short_line_ratio
        ):
            return "garbled_output (mostly very short lines)"

    # 3. Empty or near-empty after stripping common prefixes
    stripped = answer.strip()
    for prefix in ["```", "```python", "```json", "Here is", "The answer is"]:
        stripped = stripped.removeprefix(prefix).strip()
    if len(stripped) < 10:
        return "near_empty_output"

    return None
