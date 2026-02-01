"""File I/O tool methods for the REPL environment.

This module is a re-export shim for backward compatibility.
The implementation has been split into focused mixins:
- file_exploration: peek, grep, list_dir, file_info
- archive_tools: archive_open, archive_extract, archive_file, archive_search
- external_access: web_fetch, run_shell
- file_mutation: log_append, file_write_safe, patch tools

All methods are combined in the _FileToolsMixin class.
"""

from __future__ import annotations

from src.repl_environment.file_exploration import _FileExplorationMixin
from src.repl_environment.archive_tools import _ArchiveToolsMixin
from src.repl_environment.external_access import _ExternalAccessMixin
from src.repl_environment.file_mutation import _FileMutationMixin


class _FileToolsMixin(
    _FileExplorationMixin,
    _ArchiveToolsMixin,
    _ExternalAccessMixin,
    _FileMutationMixin,
):
    """Combined mixin providing all file I/O tools for REPLEnvironment.

    This class inherits from all the focused mixins to provide the complete
    set of file tools. It exists for backward compatibility with code that
    imports _FileToolsMixin directly.

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        context: str — full input context
        artifacts: dict — collected artifacts
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        _grep_hits_buffer: list — grep results buffer
        _validate_file_path: Callable[[str], tuple[bool, str | None]] — path validation method
        _ocr_document: Callable[[str], str] — OCR document processor (from DocumentToolsMixin)

    Note: Individual mixin contracts are documented in:
        - file_exploration.py (_FileExplorationMixin)
        - archive_tools.py (_ArchiveToolsMixin)
        - external_access.py (_ExternalAccessMixin)
        - file_mutation.py (_FileMutationMixin)
    """
    pass
