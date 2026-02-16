"""Domain-specific exceptions for the orchestrator system.

Replaces generic ``except Exception`` catches with typed exceptions that
allow callers to handle specific failure modes.  Infrastructure exceptions
(timeouts, connection failures) are distinguished from application-level
errors (bad input, delegation failure) so callers can retry vs. bail out.
"""


class OrchestratorError(Exception):
    """Base exception for all orchestrator errors."""


# -- Inference / LLM --------------------------------------------------------

class InferenceError(OrchestratorError):
    """LLM call failed (timeout, backend down, malformed response)."""


class InferenceTimeoutError(InferenceError):
    """LLM call timed out."""


class BackendUnavailableError(InferenceError):
    """Backend server is unreachable or returned 502/503."""


# -- Delegation / Routing ---------------------------------------------------

class DelegationError(OrchestratorError):
    """Architect delegation failed (bad decision, specialist failure)."""


class DelegationLoopError(DelegationError):
    """Delegation entered a zero-progress loop."""


# -- Vision ------------------------------------------------------------------

class VisionAnalysisError(OrchestratorError):
    """Vision pipeline failed (OCR, VL inference, image processing)."""


# -- Archive / Document ------------------------------------------------------

class ArchiveExtractionError(OrchestratorError):
    """Archive extraction failed (corrupt, too large, unsupported format)."""


class DocumentProcessingError(OrchestratorError):
    """Document preprocessing or parsing failed."""


# -- REPL / Execution -------------------------------------------------------

class REPLExecutionError(OrchestratorError):
    """REPL code execution failed."""


class REPLTimeoutError(REPLExecutionError):
    """REPL execution timed out."""


# -- Configuration -----------------------------------------------------------

class ConfigurationError(OrchestratorError):
    """Configuration is invalid or missing required values."""


class RegistryError(OrchestratorError):
    """Model registry loading or validation failed."""
