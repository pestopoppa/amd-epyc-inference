"""Backend management for LLM primitives."""

from typing import Any


class BackendMixin:
    """Mixin for backend initialization and access methods."""

    def _init_caching_backends(self, server_urls: dict[str, str], num_slots: int) -> None:
        """Initialize CachingBackend instances for each role.

        Args:
            server_urls: Dict mapping role names to llama-server URLs.
            num_slots: Number of slots per server.
        """
        try:
            from src.backends.llama_server import LlamaServerBackend, ServerConfig
            from src.prefix_cache import CachingBackend, PrefixRouter

            for role, url in server_urls.items():
                config = ServerConfig(base_url=url, num_slots=num_slots)
                backend = LlamaServerBackend(config)
                router = PrefixRouter(num_slots=num_slots)
                self._backends[role] = CachingBackend(backend, router)

        except ImportError as e:
            # If RadixAttention modules not available, log and continue
            import logging
            logging.warning(f"CachingBackend not available: {e}. Using legacy mode.")

    def get_backend(self, role: str) -> Any | None:
        """Get the CachingBackend for a role.

        Args:
            role: Role name (e.g., "worker", "coder", "frontdoor").

        Returns:
            CachingBackend instance or None if not configured.
        """
        return self._backends.get(role)

    def get_cache_stats(self) -> dict[str, dict[str, Any]]:
        """Get cache statistics for all backends.

        Returns:
            Dict mapping role to cache stats dict.
        """
        stats = {}
        for role, backend in self._backends.items():
            if hasattr(backend, "get_stats"):
                stats[role] = backend.get_stats()
        return stats
