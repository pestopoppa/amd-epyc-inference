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

    Expects the following attributes from the concrete class:
    - config: REPLConfig
    - context: str
    - artifacts: dict
    - _exploration_calls: int
    - _exploration_log: ExplorationLog
    - progress_logger: ProgressLogger | None
    - task_id: str
    - _grep_hits_buffer: list
    - ALLOWED_FILE_PATHS: list[str]
    - _validate_file_path(path) -> tuple[bool, str | None]
    """
    pass
