"""Persona management utilities."""


class PersonaMixin:
    """Mixin for persona-related methods."""

    def _apply_persona_prefix(self, prompt: str, persona: str | None) -> str:
        """Prepend persona system prompt if persona is set and feature enabled.

        Extracted from _llm_call_impl to share with batch methods.
        """
        if persona:
            from src.features import features as _get_features
            if _get_features().personas:
                from src.persona_loader import get_persona_registry
                persona_cfg = get_persona_registry().get(persona)
                if persona_cfg:
                    return f"{persona_cfg.system_prompt.strip()}\n\n{prompt}"
        return prompt
