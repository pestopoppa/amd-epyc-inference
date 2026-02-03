"""File mutation and patch management tools for the REPL environment.

Log appending, safe file writes, and patch preparation/approval workflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _get_project_root() -> Path:
    """Get project root from config with fallback."""
    try:
        from src.config import get_config

        return get_config().paths.project_root
    except Exception:
        return Path("/mnt/raid0/llm/claude")


class _FileMutationMixin:
    """Mixin providing file write operations and patch management.

    Required attributes (provided by REPLEnvironment.__init__):
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        _validate_file_path: Callable[[str], tuple[bool, str | None]] — path validation method
    """

    def _log_append(self, log_name: str, message: str) -> str:
        """Append a message to a log file.

        Args:
            log_name: Name of the log file (without path).
            message: Message to append.

        Returns:
            Confirmation message.
        """
        self._exploration_calls += 1
        from datetime import datetime

        try:
            log_path = str(_get_project_root() / "logs" / log_name)

            # Validate path
            is_valid, error = self._validate_file_path(log_path)
            if not is_valid:
                return f"[ERROR: {error}]"

            timestamp = datetime.now().isoformat()
            log_entry = f"[{timestamp}] {message}\n"

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)

            return f"Appended to {log_name}"

        except Exception as e:
            logger.debug("log_append failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _file_write_safe(
        self,
        path: str,
        content: str,
        backup: bool = True,
    ) -> str:
        """Safely write content to a file with optional backup.

        Only allows writing to /mnt/raid0/ paths.

        Args:
            path: Absolute path to write to.
            content: Content to write.
            backup: Whether to create backup of existing file.

        Returns:
            Success/failure status.
        """
        self._exploration_calls += 1
        import os
        from datetime import datetime
        from pathlib import Path as P

        try:
            # Validate path
            is_valid, error = self._validate_file_path(path)
            if not is_valid:
                return f"[ERROR: {error}]"

            # Create backup if file exists and backup requested
            if backup and os.path.exists(path):
                backup_path = f"{path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                with open(path, "r", encoding="utf-8") as src:
                    with open(backup_path, "w", encoding="utf-8") as dst:
                        dst.write(src.read())

            # Ensure parent directory exists
            P(path).parent.mkdir(parents=True, exist_ok=True)

            # Write content
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

            self._exploration_log.add_event(
                "file_write_safe", {"path": path, "size": len(content)}, "success"
            )
            return f"Wrote {len(content)} bytes to {path}"

        except Exception as e:
            logger.debug("file_write_safe failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    # =========================================================================
    # Patch Tools
    # =========================================================================

    def _prepare_patch(self, files: list[str], description: str) -> str:
        """Generate unified diff for owner review.

        Args:
            files: List of file paths to include in the patch.
            description: Short description of the changes.

        Returns:
            Path to the generated patch file.
        """
        self._exploration_calls += 1
        import subprocess
        from datetime import datetime

        try:
            patches_dir = _get_project_root() / "orchestration" / "patches" / "pending"
            patches_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_desc = description.replace(" ", "_")[:30]
            patch_name = f"{timestamp}_{safe_desc}.patch"
            patch_path = patches_dir / patch_name

            # Generate unified diff
            result = subprocess.run(
                ["git", "diff", "--"] + files,
                capture_output=True,
                text=True,
                cwd=str(_get_project_root()),
            )

            if not result.stdout.strip():
                return "[INFO: No changes to create patch from]"

            # Write patch with metadata header
            with open(patch_path, "w", encoding="utf-8") as f:
                f.write(f"# Patch: {description}\n")
                f.write(f"# Created: {datetime.now().isoformat()}\n")
                f.write(f"# Files: {', '.join(files)}\n")
                f.write("# Status: PENDING APPROVAL\n")
                f.write("#\n")
                f.write(result.stdout)

            return f"Patch created: {patch_path}\nReview with: cat {patch_path}"

        except Exception as e:
            logger.debug("prepare_patch failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _list_patches(self, status: str = "pending") -> str:
        """List patches by status.

        Args:
            status: One of 'pending', 'approved', 'rejected', or 'all'.

        Returns:
            List of patches with metadata.
        """
        self._exploration_calls += 1
        import json

        try:
            patches_base = _get_project_root() / "orchestration" / "patches"
            results = []

            statuses = ["pending", "approved", "rejected"] if status == "all" else [status]

            for s in statuses:
                status_dir = patches_base / s
                if not status_dir.exists():
                    continue

                for patch_file in sorted(status_dir.glob("*.patch")):
                    # Read first few lines for metadata
                    with open(patch_file, encoding="utf-8") as f:
                        lines = f.readlines()[:5]

                    metadata = {"file": str(patch_file), "status": s}
                    for line in lines:
                        if line.startswith("# Patch:"):
                            metadata["description"] = line.split(":", 1)[1].strip()
                        elif line.startswith("# Created:"):
                            metadata["created"] = line.split(":", 1)[1].strip()
                        elif line.startswith("# Files:"):
                            metadata["files"] = line.split(":", 1)[1].strip()

                    results.append(metadata)

            if not results:
                return f"[INFO: No {status} patches found]"

            return json.dumps(results, indent=2)

        except Exception as e:
            logger.debug("list_patches failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _apply_approved_patch(self, patch_name: str) -> str:
        """Apply a patch after owner approval.

        Args:
            patch_name: Name of the patch file.

        Returns:
            Application status.
        """
        self._exploration_calls += 1
        import shutil
        import subprocess
        from datetime import datetime

        try:
            patches_base = _get_project_root() / "orchestration" / "patches"
            pending_path = patches_base / "pending" / patch_name
            approved_path = patches_base / "approved" / patch_name

            if not pending_path.exists():
                return f"[ERROR: Patch not found in pending: {patch_name}]"

            # Dry run first
            result = subprocess.run(
                ["git", "apply", "--check", str(pending_path)],
                capture_output=True,
                text=True,
                cwd=str(_get_project_root()),
            )

            if result.returncode != 0:
                return f"[ERROR: Patch cannot be applied cleanly: {result.stderr}]"

            # Apply the patch
            result = subprocess.run(
                ["git", "apply", str(pending_path)],
                capture_output=True,
                text=True,
                cwd=str(_get_project_root()),
            )

            if result.returncode != 0:
                return f"[ERROR: Failed to apply patch: {result.stderr}]"

            # Move to approved
            shutil.move(str(pending_path), str(approved_path))

            # Add approval timestamp
            with open(approved_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Applied: {datetime.now().isoformat()}\n")

            return f"Patch applied successfully and moved to approved: {approved_path}"

        except Exception as e:
            logger.debug("apply_approved_patch failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _reject_patch(self, patch_name: str, reason: str) -> str:
        """Reject a pending patch with reason.

        Args:
            patch_name: Name of the patch file.
            reason: Why the patch was rejected.

        Returns:
            Rejection status.
        """
        self._exploration_calls += 1
        import shutil
        from datetime import datetime

        try:
            patches_base = _get_project_root() / "orchestration" / "patches"
            pending_path = patches_base / "pending" / patch_name
            rejected_path = patches_base / "rejected" / patch_name

            if not pending_path.exists():
                return f"[ERROR: Patch not found in pending: {patch_name}]"

            # Move to rejected
            shutil.move(str(pending_path), str(rejected_path))

            # Add rejection metadata
            with open(rejected_path, "a", encoding="utf-8") as f:
                f.write(f"\n# Rejected: {datetime.now().isoformat()}\n")
                f.write(f"# Reason: {reason}\n")

            return f"Patch rejected and moved to: {rejected_path}"

        except Exception as e:
            logger.debug("reject_patch failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"
