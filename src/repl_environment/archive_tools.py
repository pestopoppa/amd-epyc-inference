"""Archive and document ingestion tools for the REPL environment.

Archive extraction, document processing, and search across compressed files.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class _ArchiveToolsMixin:
    """Mixin providing archive and document ingestion tools.

    Expects the following attributes from the concrete class:
    - config: REPLConfig
    - artifacts: dict
    - _exploration_calls: int
    - _exploration_log: ExplorationLog
    - _validate_file_path(path) -> tuple[bool, str | None]
    - _ocr_document(path) -> str  (from DocumentToolsMixin)
    """

    def _archive_open(self, path: str) -> str:
        """Open an archive and return its manifest.

        Supported formats: .zip, .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz, .7z

        Args:
            path: Absolute path to the archive file.

        Returns:
            JSON string with archive manifest.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        try:
            from src.services.archive_extractor import ArchiveExtractor

            extractor = ArchiveExtractor()
            archive_path = Path(path)

            # Validate archive first
            validation = extractor.validate(archive_path)
            if not validation.is_safe:
                return json.dumps({
                    "error": f"Archive validation failed: {validation.status.value}",
                    "issues": validation.issues,
                }, indent=2)

            # Get manifest
            manifest = extractor.list_contents(archive_path)

            # Store in artifacts for later use
            if "_archives" not in self.artifacts:
                self.artifacts["_archives"] = {}

            archive_name = archive_path.name
            self.artifacts["_archives"][archive_name] = {
                "manifest": manifest,
                "path": path,
                "extracted_to": None,
                "processed_files": {},
            }
            # Also track "current" archive for easier access
            self.artifacts["_archives"]["current"] = archive_name

            result = manifest.to_summary_dict()
            self._exploration_log.add_event("archive_open", {"path": path}, result)
            return json.dumps(result, indent=2)

        except ImportError:
            return "[ERROR: Archive extractor not available]"
        except Exception as e:
            logger.debug("archive_open failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _archive_extract(
        self,
        archive_name: str | None = None,
        pattern: str | None = None,
        files: list[str] | None = None,
        process_documents: bool = True,
    ) -> str:
        """Extract files from an opened archive.

        Must call archive_open() first.

        Args:
            archive_name: Name of the archive (from archive_open). If None,
                         uses the most recently opened archive.
            pattern: Glob pattern to match (e.g., "*.pdf", "docs/*.md").
            files: List of specific files to extract.
            process_documents: If True, route PDFs through OCR pipeline.

        Returns:
            JSON string with extraction results.
        """
        self._exploration_calls += 1
        import json
        from pathlib import Path

        try:
            from src.services.archive_extractor import ArchiveExtractor

            # Get archive info from artifacts
            if "_archives" not in self.artifacts or not self.artifacts["_archives"]:
                return "[ERROR: No archive opened. Call archive_open() first.]"

            if archive_name is None:
                # Use the "current" marker if available
                current = self.artifacts["_archives"].get("current")
                if current:
                    archive_name = current
                else:
                    # Fall back to last opened archive (excluding "current" key)
                    real_archives = [k for k in self.artifacts["_archives"].keys() if k != "current"]
                    if not real_archives:
                        return "[ERROR: No archive opened. Call archive_open() first.]"
                    archive_name = real_archives[-1]

            if archive_name not in self.artifacts["_archives"]:
                real_archives = [k for k in self.artifacts["_archives"].keys() if k != "current"]
                return f"[ERROR: Archive not found: {archive_name}. Opened: {real_archives}]"

            archive_info = self.artifacts["_archives"][archive_name]
            if not isinstance(archive_info, dict):
                return f"[ERROR: Invalid archive info for: {archive_name}]"
            archive_path = Path(archive_info["path"])
            manifest = archive_info["manifest"]

            extractor = ArchiveExtractor()

            # Determine what to extract
            if pattern:
                result = extractor.extract_pattern(archive_path, pattern)
            elif files:
                result = extractor.extract_files(archive_path, files)
            else:
                # Extract all
                result = extractor.extract_all(archive_path)

            if not result.success and not result.extracted_files:
                return json.dumps({
                    "error": "Extraction failed",
                    "errors": result.errors,
                }, indent=2)

            # Update artifacts with extracted files
            archive_info["extracted_to"] = str(result.extracted_files.get(
                list(result.extracted_files.keys())[0], ""
            ).parent if result.extracted_files else None)

            # Process documents if requested
            sections_total = 0
            figures_total = 0

            if process_documents:
                # Route document files through preprocessing
                doc_extensions = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}

                for filename, file_path in result.extracted_files.items():
                    ext = Path(filename).suffix.lower()

                    if ext in doc_extensions:
                        # Try to process via OCR
                        try:
                            ocr_result = self._ocr_document(str(file_path))
                            ocr_data = json.loads(ocr_result)

                            if "error" not in ocr_result.lower():
                                archive_info["processed_files"][filename] = {
                                    "type": "document",
                                    "sections": ocr_data.get("total_pages", 0),
                                    "figures": len(ocr_data.get("figures", [])),
                                    "text_preview": ocr_data.get("full_text", "")[:500],
                                }
                                sections_total += ocr_data.get("total_pages", 0)
                                figures_total += len(ocr_data.get("figures", []))
                        except Exception:
                            logger.debug("Archive document processing failed: %s", filename, exc_info=True)
                            # Store as unprocessed document
                            archive_info["processed_files"][filename] = {
                                "type": "document",
                                "error": "Processing failed",
                            }
                    else:
                        # Text/code files - read content directly
                        try:
                            content = file_path.read_text(encoding="utf-8", errors="replace")
                            archive_info["processed_files"][filename] = {
                                "type": "text",
                                "content": content[:50000],  # Cap at 50K
                                "lines": content.count("\n") + 1,
                            }
                        except Exception:
                            logger.debug("Archive binary processing failed: %s", filename, exc_info=True)
                            archive_info["processed_files"][filename] = {
                                "type": "binary",
                                "size": file_path.stat().st_size,
                            }

            summary = {
                "success": True,
                "extracted": len(result.extracted_files),
                "processed": len(archive_info["processed_files"]),
                "sections_total": sections_total,
                "figures_total": figures_total,
                "skipped": len(result.skipped_files),
                "stored_in": f"artifacts['_archives']['{archive_name}']",
            }

            self._exploration_log.add_event("archive_extract", {
                "archive_name": archive_name,
                "pattern": pattern,
                "files": files,
            }, summary)

            return json.dumps(summary, indent=2)

        except ImportError:
            return "[ERROR: Archive extractor not available]"
        except Exception as e:
            logger.debug("archive_extract failed", exc_info=True)
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _archive_file(self, filename: str, archive_name: str | None = None) -> str:
        """Get content of a specific file from an extracted archive.

        Args:
            filename: Path of the file within the archive.
            archive_name: Name of the archive. If None, searches all.

        Returns:
            File content (text) or document metadata (for processed PDFs).
        """
        self._exploration_calls += 1
        import json

        if "_archives" not in self.artifacts:
            return "[ERROR: No archives opened]"

        # Find the file
        archives_to_search = [archive_name] if archive_name else list(self.artifacts["_archives"].keys())

        for arch_name in archives_to_search:
            if arch_name not in self.artifacts["_archives"]:
                continue

            archive_info = self.artifacts["_archives"][arch_name]
            processed = archive_info.get("processed_files", {})

            if filename in processed:
                file_info = processed[filename]

                if file_info.get("type") == "text":
                    return file_info.get("content", "")
                elif file_info.get("type") == "document":
                    return json.dumps(file_info, indent=2)
                else:
                    return json.dumps(file_info, indent=2)

        return f"[ERROR: File not found: {filename}]"

    def _archive_search(self, query: str, archive_name: str | None = None) -> str:
        """Search across all extracted archive content.

        Args:
            query: Search query string.
            archive_name: Name of archive to search. If None, searches all.

        Returns:
            JSON string with search results.
        """
        self._exploration_calls += 1
        import json
        import re

        if "_archives" not in self.artifacts:
            return json.dumps([])

        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        if archive_name:
            archives_to_search = [archive_name]
        else:
            # Skip the "current" marker key
            archives_to_search = [k for k in self.artifacts["_archives"].keys() if k != "current"]

        for arch_name in archives_to_search:
            if arch_name not in self.artifacts["_archives"]:
                continue

            archive_info = self.artifacts["_archives"][arch_name]
            if not isinstance(archive_info, dict):
                continue
            processed = archive_info.get("processed_files", {})

            for filename, file_info in processed.items():
                file_type = file_info.get("type", "")

                if file_type == "text":
                    content = file_info.get("content", "")
                    for i, line in enumerate(content.splitlines(), 1):
                        if pattern.search(line):
                            results.append({
                                "archive": arch_name,
                                "file": filename,
                                "line": i,
                                "match": line[:200],
                            })
                            if len(results) >= 50:  # Cap results
                                break

                elif file_type == "document":
                    text = file_info.get("text_preview", "")
                    if pattern.search(text):
                        # Find match context
                        match = pattern.search(text)
                        if match:
                            start = max(0, match.start() - 50)
                            end = min(len(text), match.end() + 50)
                            results.append({
                                "archive": arch_name,
                                "file": filename,
                                "section": "document",
                                "match": text[start:end],
                            })

                if len(results) >= 50:
                    break

            if len(results) >= 50:
                break

        self._exploration_log.add_event("archive_search", {"query": query}, results)
        return json.dumps(results, indent=2)
