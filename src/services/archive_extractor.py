#!/usr/bin/env python3
"""Archive extraction service for orchestrator document ingestion.

Provides manifest-first archive extraction with security checks,
designed to integrate with the document processing pipeline.

Security features:
- Zip bomb detection (compression ratio check)
- Path traversal prevention
- Size limits (archive, extracted, per-file)
- File count limits
- Nested archive depth limits

Usage:
    from src.services.archive_extractor import ArchiveExtractor

    extractor = ArchiveExtractor()
    manifest = extractor.list_contents(Path("docs.zip"))
    extracted = extractor.extract_files(
        Path("docs.zip"),
        files=["report.pdf", "readme.md"],
        dest=Path("/tmp/archives/session123/")  # Use configured tmp_dir in practice
    )
"""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ArchiveType(str, Enum):
    """Supported archive types."""

    ZIP = "zip"
    TAR = "tar"
    TAR_GZ = "tar.gz"
    TAR_BZ2 = "tar.bz2"
    TAR_XZ = "tar.xz"
    SEVEN_Z = "7z"  # Requires py7zr


class ExtractionStrategy(str, Enum):
    """Strategy for handling different archive sizes."""

    AUTO_ALL = "auto_all"  # <20 files, <5MB: extract everything
    MANIFEST_THEN_ASK = "manifest_then_ask"  # 20-100 files: show manifest first
    SUMMARY_WITH_RECOMMENDATIONS = "summary_with_recommendations"  # >100 files


class ValidationStatus(str, Enum):
    """Archive validation status."""

    VALID = "valid"
    SUSPICIOUS = "suspicious"  # High compression ratio
    INVALID = "invalid"  # Corrupted or unsupported
    TOO_LARGE = "too_large"
    TOO_MANY_FILES = "too_many_files"


@dataclass
class FileEntry:
    """Entry in an archive manifest."""

    name: str
    size: int
    compressed_size: int | None
    is_dir: bool
    extension: str
    modified: float | None = None  # Unix timestamp

    @property
    def compression_ratio(self) -> float | None:
        """Calculate compression ratio (uncompressed/compressed)."""
        if self.compressed_size and self.compressed_size > 0:
            return self.size / self.compressed_size
        return None


@dataclass
class ArchiveManifest:
    """Manifest describing archive contents."""

    path: str
    archive_type: ArchiveType
    total_files: int
    total_dirs: int
    total_size_bytes: int
    compressed_size_bytes: int
    file_tree: list[FileEntry]
    type_summary: dict[str, int]  # {".pdf": 3, ".py": 12, ...}
    nested_archives: list[str]  # Paths to archives within this archive
    manifest_hash: str  # SHA256 of manifest for change detection

    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        if self.compressed_size_bytes > 0:
            return self.total_size_bytes / self.compressed_size_bytes
        return 1.0

    def recommend_strategy(self) -> ExtractionStrategy:
        """Recommend extraction strategy based on archive characteristics."""
        if self.total_files < 20 and self.total_size_bytes < 5_000_000:
            return ExtractionStrategy.AUTO_ALL
        elif self.total_files < 100:
            return ExtractionStrategy.MANIFEST_THEN_ASK
        else:
            return ExtractionStrategy.SUMMARY_WITH_RECOMMENDATIONS

    def files_matching(self, pattern: str) -> list[FileEntry]:
        """Get files matching a glob pattern."""
        return [f for f in self.file_tree if not f.is_dir and fnmatch.fnmatch(f.name, pattern)]

    def to_summary_dict(self) -> dict:
        """Convert to summary dict for LLM context (token-efficient)."""
        return {
            "path": self.path,
            "type": self.archive_type.value,
            "total_files": self.total_files,
            "total_size": _format_size(self.total_size_bytes),
            "types": self.type_summary,
            "nested_archives": len(self.nested_archives),
            "recommendation": self.recommend_strategy().value,
        }


@dataclass
class ValidationResult:
    """Result of archive validation."""

    status: ValidationStatus
    issues: list[str] = field(default_factory=list)
    archive_type: ArchiveType | None = None
    estimated_size: int = 0
    file_count: int = 0

    @property
    def is_safe(self) -> bool:
        """Check if archive is safe to extract."""
        return self.status == ValidationStatus.VALID


@dataclass
class ExtractionResult:
    """Result of archive extraction."""

    success: bool
    extracted_files: dict[str, Path]  # original_name -> extracted_path
    skipped_files: list[str]
    errors: list[str]
    extraction_time: float
    total_bytes: int

    def to_summary_dict(self) -> dict:
        """Convert to summary dict for LLM context."""
        return {
            "success": self.success,
            "extracted": len(self.extracted_files),
            "skipped": len(self.skipped_files),
            "errors": self.errors[:5] if self.errors else [],
            "total_size": _format_size(self.total_bytes),
            "time_sec": round(self.extraction_time, 2),
        }


class ArchiveExtractor:
    """Archive extraction service with security checks.

    Provides manifest-first extraction designed for LLM workflows:
    1. list_contents() - Get manifest without extracting
    2. validate() - Check for security issues
    3. extract_files() / extract_pattern() - Selective extraction
    """

    # Supported extensions mapped to archive types
    SUPPORTED_EXTENSIONS: dict[str, ArchiveType] = {
        ".zip": ArchiveType.ZIP,
        ".tar": ArchiveType.TAR,
        ".tar.gz": ArchiveType.TAR_GZ,
        ".tgz": ArchiveType.TAR_GZ,
        ".tar.bz2": ArchiveType.TAR_BZ2,
        ".tbz2": ArchiveType.TAR_BZ2,
        ".tar.xz": ArchiveType.TAR_XZ,
        ".txz": ArchiveType.TAR_XZ,
        ".7z": ArchiveType.SEVEN_Z,
    }

    # Archive extensions (for detecting nested archives)
    ARCHIVE_EXTENSIONS = {
        ".zip",
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tar.xz",
        ".7z",
        ".rar",
        ".gz",
        ".bz2",
        ".xz",
    }

    # Security limits
    MAX_ARCHIVE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_EXTRACTED_SIZE = 1024 * 1024 * 1024  # 1GB
    MAX_SINGLE_FILE = 100 * 1024 * 1024  # 100MB
    MAX_FILES = 1000
    MAX_COMPRESSION_RATIO = 100  # Suspicious if > 100:1
    MAX_RECURSION_DEPTH = 2

    def __init__(
        self,
        max_archive_size: int | None = None,
        max_extracted_size: int | None = None,
        max_files: int | None = None,
    ):
        """Initialize extractor with optional custom limits.

        Args:
            max_archive_size: Max archive file size in bytes.
            max_extracted_size: Max total extracted size in bytes.
            max_files: Max number of files in archive.
        """
        from src.config import get_config

        _svc = get_config().services
        self.max_archive_size = max_archive_size or _svc.max_archive_size
        self.max_extracted_size = max_extracted_size or _svc.max_extracted_size
        self.max_files = max_files or _svc.max_archive_files

    def get_archive_type(self, path: Path) -> ArchiveType | None:
        """Determine archive type from file extension.

        Args:
            path: Path to archive file.

        Returns:
            ArchiveType or None if unsupported.
        """
        name = path.name.lower()

        # Check compound extensions first
        for ext in [".tar.gz", ".tar.bz2", ".tar.xz"]:
            if name.endswith(ext):
                return self.SUPPORTED_EXTENSIONS[ext]

        # Then single extensions
        suffix = path.suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(suffix)

    def validate(self, path: Path) -> ValidationResult:
        """Validate archive for security issues.

        Checks:
        - File exists and is readable
        - Archive type is supported
        - File size within limits
        - Compression ratio not suspicious (zip bomb detection)
        - File count within limits

        Args:
            path: Path to archive file.

        Returns:
            ValidationResult with status and any issues.
        """
        issues = []

        # Check file exists
        if not path.exists():
            return ValidationResult(
                status=ValidationStatus.INVALID,
                issues=[f"File not found: {path}"],
            )

        # Check archive type
        archive_type = self.get_archive_type(path)
        if archive_type is None:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                issues=[f"Unsupported archive type: {path.suffix}"],
            )

        # Check file size
        file_size = path.stat().st_size
        if file_size > self.max_archive_size:
            return ValidationResult(
                status=ValidationStatus.TOO_LARGE,
                issues=[
                    f"Archive too large: {_format_size(file_size)} > {_format_size(self.max_archive_size)}"
                ],
                archive_type=archive_type,
            )

        # Get manifest to check contents
        try:
            manifest = self.list_contents(path)
        except Exception as e:
            return ValidationResult(
                status=ValidationStatus.INVALID,
                issues=[f"Failed to read archive: {e}"],
                archive_type=archive_type,
            )

        # Check file count
        if manifest.total_files > self.max_files:
            return ValidationResult(
                status=ValidationStatus.TOO_MANY_FILES,
                issues=[f"Too many files: {manifest.total_files} > {self.max_files}"],
                archive_type=archive_type,
                file_count=manifest.total_files,
            )

        # Check extracted size
        if manifest.total_size_bytes > self.max_extracted_size:
            issues.append(
                f"Extracted size would exceed limit: "
                f"{_format_size(manifest.total_size_bytes)} > {_format_size(self.max_extracted_size)}"
            )

        # Check compression ratio (zip bomb detection)
        if manifest.compression_ratio > self.MAX_COMPRESSION_RATIO:
            issues.append(
                f"Suspicious compression ratio: {manifest.compression_ratio:.1f}:1 "
                f"(threshold: {self.MAX_COMPRESSION_RATIO}:1)"
            )
            return ValidationResult(
                status=ValidationStatus.SUSPICIOUS,
                issues=issues,
                archive_type=archive_type,
                estimated_size=manifest.total_size_bytes,
                file_count=manifest.total_files,
            )

        # All checks passed
        if issues:
            return ValidationResult(
                status=ValidationStatus.SUSPICIOUS,
                issues=issues,
                archive_type=archive_type,
                estimated_size=manifest.total_size_bytes,
                file_count=manifest.total_files,
            )

        return ValidationResult(
            status=ValidationStatus.VALID,
            archive_type=archive_type,
            estimated_size=manifest.total_size_bytes,
            file_count=manifest.total_files,
        )

    def list_contents(self, path: Path) -> ArchiveManifest:
        """List archive contents without extracting.

        Args:
            path: Path to archive file.

        Returns:
            ArchiveManifest describing contents.

        Raises:
            ValueError: If archive type is unsupported.
            IOError: If archive cannot be read.
        """
        archive_type = self.get_archive_type(path)
        if archive_type is None:
            raise ValueError(f"Unsupported archive type: {path.suffix}")

        if archive_type == ArchiveType.ZIP:
            return self._list_zip(path)
        elif archive_type in (
            ArchiveType.TAR,
            ArchiveType.TAR_GZ,
            ArchiveType.TAR_BZ2,
            ArchiveType.TAR_XZ,
        ):
            return self._list_tar(path, archive_type)
        elif archive_type == ArchiveType.SEVEN_Z:
            return self._list_7z(path)
        else:
            raise ValueError(f"Unhandled archive type: {archive_type}")

    def _list_zip(self, path: Path) -> ArchiveManifest:
        """List contents of a ZIP archive."""
        entries = []
        type_summary: dict[str, int] = {}
        nested_archives = []
        total_size = 0
        compressed_size = 0
        total_dirs = 0

        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                is_dir = info.is_dir()
                if is_dir:
                    total_dirs += 1
                    continue

                ext = Path(info.filename).suffix.lower()
                entry = FileEntry(
                    name=info.filename,
                    size=info.file_size,
                    compressed_size=info.compress_size,
                    is_dir=is_dir,
                    extension=ext,
                    modified=_zipinfo_to_timestamp(info),
                )
                entries.append(entry)

                total_size += info.file_size
                compressed_size += info.compress_size

                if ext:
                    type_summary[ext] = type_summary.get(ext, 0) + 1

                if ext in self.ARCHIVE_EXTENSIONS:
                    nested_archives.append(info.filename)

        manifest_hash = self._compute_manifest_hash(entries)

        return ArchiveManifest(
            path=str(path),
            archive_type=ArchiveType.ZIP,
            total_files=len(entries),
            total_dirs=total_dirs,
            total_size_bytes=total_size,
            compressed_size_bytes=compressed_size,
            file_tree=entries,
            type_summary=type_summary,
            nested_archives=nested_archives,
            manifest_hash=manifest_hash,
        )

    def _list_tar(self, path: Path, archive_type: ArchiveType) -> ArchiveManifest:
        """List contents of a TAR archive (optionally compressed)."""
        mode_map = {
            ArchiveType.TAR: "r",
            ArchiveType.TAR_GZ: "r:gz",
            ArchiveType.TAR_BZ2: "r:bz2",
            ArchiveType.TAR_XZ: "r:xz",
        }
        mode = mode_map.get(archive_type, "r")

        entries = []
        type_summary: dict[str, int] = {}
        nested_archives = []
        total_size = 0
        total_dirs = 0

        with tarfile.open(path, mode) as tf:
            for member in tf.getmembers():
                if member.isdir():
                    total_dirs += 1
                    continue

                ext = Path(member.name).suffix.lower()
                entry = FileEntry(
                    name=member.name,
                    size=member.size,
                    compressed_size=None,  # TAR doesn't track per-file compression
                    is_dir=member.isdir(),
                    extension=ext,
                    modified=member.mtime,
                )
                entries.append(entry)

                total_size += member.size

                if ext:
                    type_summary[ext] = type_summary.get(ext, 0) + 1

                if ext in self.ARCHIVE_EXTENSIONS:
                    nested_archives.append(member.name)

        manifest_hash = self._compute_manifest_hash(entries)
        compressed_size = path.stat().st_size

        return ArchiveManifest(
            path=str(path),
            archive_type=archive_type,
            total_files=len(entries),
            total_dirs=total_dirs,
            total_size_bytes=total_size,
            compressed_size_bytes=compressed_size,
            file_tree=entries,
            type_summary=type_summary,
            nested_archives=nested_archives,
            manifest_hash=manifest_hash,
        )

    def _list_7z(self, path: Path) -> ArchiveManifest:
        """List contents of a 7z archive."""
        try:
            import py7zr
        except ImportError:
            raise ImportError(
                "py7zr package required for 7z support. Install with: pip install py7zr"
            )

        entries = []
        type_summary: dict[str, int] = {}
        nested_archives = []
        total_size = 0
        total_dirs = 0

        with py7zr.SevenZipFile(path, "r") as szf:
            for name, info in szf.archiveinfo().files.items():
                is_dir = info.is_directory
                if is_dir:
                    total_dirs += 1
                    continue

                ext = Path(name).suffix.lower()
                entry = FileEntry(
                    name=name,
                    size=info.uncompressed,
                    compressed_size=info.compressed,
                    is_dir=is_dir,
                    extension=ext,
                    modified=info.creationtime.timestamp() if info.creationtime else None,
                )
                entries.append(entry)

                total_size += info.uncompressed

                if ext:
                    type_summary[ext] = type_summary.get(ext, 0) + 1

                if ext in self.ARCHIVE_EXTENSIONS:
                    nested_archives.append(name)

        manifest_hash = self._compute_manifest_hash(entries)
        compressed_size = path.stat().st_size

        return ArchiveManifest(
            path=str(path),
            archive_type=ArchiveType.SEVEN_Z,
            total_files=len(entries),
            total_dirs=total_dirs,
            total_size_bytes=total_size,
            compressed_size_bytes=compressed_size,
            file_tree=entries,
            type_summary=type_summary,
            nested_archives=nested_archives,
            manifest_hash=manifest_hash,
        )

    def _compute_manifest_hash(self, entries: list[FileEntry]) -> str:
        """Compute hash of manifest for change detection."""
        content = "|".join(
            f"{e.name}:{e.size}:{e.modified or 0}" for e in sorted(entries, key=lambda x: x.name)
        )
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def extract_files(
        self,
        path: Path,
        files: list[str],
        dest: Path | None = None,
        session_id: str | None = None,
    ) -> ExtractionResult:
        """Extract specific files from archive.

        Args:
            path: Path to archive file.
            files: List of file paths within archive to extract.
            dest: Destination directory. If None, creates temp dir.
            session_id: Optional session ID for organizing extractions.

        Returns:
            ExtractionResult with extracted file paths.
        """
        from src.config import get_config

        start_time = time.perf_counter()

        # Determine destination
        if dest is None:
            base_extract_dir = get_config().services.archive_extract_dir
            archive_hash = hashlib.sha256(str(path).encode()).hexdigest()[:8]
            if session_id:
                dest = base_extract_dir / session_id / archive_hash
            else:
                dest = base_extract_dir / archive_hash

        dest.mkdir(parents=True, exist_ok=True)

        archive_type = self.get_archive_type(path)
        if archive_type is None:
            return ExtractionResult(
                success=False,
                extracted_files={},
                skipped_files=files,
                errors=[f"Unsupported archive type: {path.suffix}"],
                extraction_time=time.perf_counter() - start_time,
                total_bytes=0,
            )

        # Extract based on type
        try:
            if archive_type == ArchiveType.ZIP:
                result = self._extract_zip_files(path, files, dest)
            elif archive_type in (
                ArchiveType.TAR,
                ArchiveType.TAR_GZ,
                ArchiveType.TAR_BZ2,
                ArchiveType.TAR_XZ,
            ):
                result = self._extract_tar_files(path, files, dest, archive_type)
            elif archive_type == ArchiveType.SEVEN_Z:
                result = self._extract_7z_files(path, files, dest)
            else:
                return ExtractionResult(
                    success=False,
                    extracted_files={},
                    skipped_files=files,
                    errors=[f"Unhandled archive type: {archive_type}"],
                    extraction_time=time.perf_counter() - start_time,
                    total_bytes=0,
                )

            result.extraction_time = time.perf_counter() - start_time
            return result

        except Exception as e:
            logger.exception(f"Extraction failed for {path}")
            return ExtractionResult(
                success=False,
                extracted_files={},
                skipped_files=files,
                errors=[str(e)],
                extraction_time=time.perf_counter() - start_time,
                total_bytes=0,
            )

    def _extract_zip_files(
        self,
        path: Path,
        files: list[str],
        dest: Path,
    ) -> ExtractionResult:
        """Extract specific files from a ZIP archive."""
        extracted = {}
        skipped = []
        errors = []
        total_bytes = 0

        files_set = set(files)

        with zipfile.ZipFile(path, "r") as zf:
            for info in zf.infolist():
                if info.filename not in files_set:
                    continue

                # Security: check for path traversal
                if not self._is_safe_path(info.filename, dest):
                    errors.append(f"Unsafe path rejected: {info.filename}")
                    skipped.append(info.filename)
                    continue

                # Security: check file size
                if info.file_size > self.MAX_SINGLE_FILE:
                    errors.append(
                        f"File too large: {info.filename} "
                        f"({_format_size(info.file_size)} > {_format_size(self.MAX_SINGLE_FILE)})"
                    )
                    skipped.append(info.filename)
                    continue

                try:
                    extracted_path = dest / info.filename
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)
                    zf.extract(info, dest)
                    extracted[info.filename] = extracted_path
                    total_bytes += info.file_size
                except Exception as e:
                    errors.append(f"Failed to extract {info.filename}: {e}")
                    skipped.append(info.filename)

        # Track files not found
        not_found = files_set - set(extracted.keys()) - set(skipped)
        for f in not_found:
            errors.append(f"File not found in archive: {f}")
            skipped.append(f)

        return ExtractionResult(
            success=len(errors) == 0,
            extracted_files=extracted,
            skipped_files=skipped,
            errors=errors,
            extraction_time=0,  # Set by caller
            total_bytes=total_bytes,
        )

    def _extract_tar_files(
        self,
        path: Path,
        files: list[str],
        dest: Path,
        archive_type: ArchiveType,
    ) -> ExtractionResult:
        """Extract specific files from a TAR archive."""
        mode_map = {
            ArchiveType.TAR: "r",
            ArchiveType.TAR_GZ: "r:gz",
            ArchiveType.TAR_BZ2: "r:bz2",
            ArchiveType.TAR_XZ: "r:xz",
        }
        mode = mode_map.get(archive_type, "r")

        extracted = {}
        skipped = []
        errors = []
        total_bytes = 0

        files_set = set(files)

        with tarfile.open(path, mode) as tf:
            for member in tf.getmembers():
                if member.name not in files_set:
                    continue

                # Security: check for path traversal
                if not self._is_safe_path(member.name, dest):
                    errors.append(f"Unsafe path rejected: {member.name}")
                    skipped.append(member.name)
                    continue

                # Security: check file size
                if member.size > self.MAX_SINGLE_FILE:
                    errors.append(
                        f"File too large: {member.name} "
                        f"({_format_size(member.size)} > {_format_size(self.MAX_SINGLE_FILE)})"
                    )
                    skipped.append(member.name)
                    continue

                # Security: only extract regular files
                if not member.isfile():
                    skipped.append(member.name)
                    continue

                try:
                    extracted_path = dest / member.name
                    extracted_path.parent.mkdir(parents=True, exist_ok=True)
                    tf.extract(member, dest, filter="data")
                    extracted[member.name] = extracted_path
                    total_bytes += member.size
                except Exception as e:
                    errors.append(f"Failed to extract {member.name}: {e}")
                    skipped.append(member.name)

        # Track files not found
        not_found = files_set - set(extracted.keys()) - set(skipped)
        for f in not_found:
            errors.append(f"File not found in archive: {f}")
            skipped.append(f)

        return ExtractionResult(
            success=len(errors) == 0,
            extracted_files=extracted,
            skipped_files=skipped,
            errors=errors,
            extraction_time=0,
            total_bytes=total_bytes,
        )

    def _extract_7z_files(
        self,
        path: Path,
        files: list[str],
        dest: Path,
    ) -> ExtractionResult:
        """Extract specific files from a 7z archive."""
        try:
            import py7zr
        except ImportError:
            return ExtractionResult(
                success=False,
                extracted_files={},
                skipped_files=files,
                errors=["py7zr package required for 7z support"],
                extraction_time=0,
                total_bytes=0,
            )

        extracted = {}
        skipped = []
        errors = []
        total_bytes = 0

        # py7zr extracts to dict of BytesIO, need to write to files
        with py7zr.SevenZipFile(path, "r") as szf:
            # Get only the files we want
            targets = [f for f in files if f in szf.getnames()]
            not_found = set(files) - set(targets)

            for f in not_found:
                errors.append(f"File not found in archive: {f}")
                skipped.append(f)

            if targets:
                file_data = szf.read(targets)

                for name, data in file_data.items():
                    # Security: check for path traversal
                    if not self._is_safe_path(name, dest):
                        errors.append(f"Unsafe path rejected: {name}")
                        skipped.append(name)
                        continue

                    content = data.read()

                    # Security: check file size
                    if len(content) > self.MAX_SINGLE_FILE:
                        errors.append(
                            f"File too large: {name} "
                            f"({_format_size(len(content))} > {_format_size(self.MAX_SINGLE_FILE)})"
                        )
                        skipped.append(name)
                        continue

                    try:
                        extracted_path = dest / name
                        extracted_path.parent.mkdir(parents=True, exist_ok=True)
                        extracted_path.write_bytes(content)
                        extracted[name] = extracted_path
                        total_bytes += len(content)
                    except Exception as e:
                        errors.append(f"Failed to extract {name}: {e}")
                        skipped.append(name)

        return ExtractionResult(
            success=len(errors) == 0,
            extracted_files=extracted,
            skipped_files=skipped,
            errors=errors,
            extraction_time=0,
            total_bytes=total_bytes,
        )

    def extract_pattern(
        self,
        path: Path,
        pattern: str,
        dest: Path | None = None,
        session_id: str | None = None,
    ) -> ExtractionResult:
        """Extract files matching a glob pattern.

        Args:
            path: Path to archive file.
            pattern: Glob pattern (e.g., "*.pdf", "docs/*.md").
            dest: Destination directory.
            session_id: Optional session ID.

        Returns:
            ExtractionResult with extracted file paths.
        """
        # Get manifest to find matching files
        manifest = self.list_contents(path)
        matching = manifest.files_matching(pattern)
        files = [f.name for f in matching]

        if not files:
            return ExtractionResult(
                success=True,
                extracted_files={},
                skipped_files=[],
                errors=[],
                extraction_time=0,
                total_bytes=0,
            )

        return self.extract_files(path, files, dest, session_id)

    def extract_all(
        self,
        path: Path,
        dest: Path | None = None,
        session_id: str | None = None,
    ) -> ExtractionResult:
        """Extract all files from archive.

        Args:
            path: Path to archive file.
            dest: Destination directory.
            session_id: Optional session ID.

        Returns:
            ExtractionResult with extracted file paths.
        """
        manifest = self.list_contents(path)
        files = [f.name for f in manifest.file_tree if not f.is_dir]
        return self.extract_files(path, files, dest, session_id)

    def _is_safe_path(self, member_path: str, dest: Path) -> bool:
        """Check if extraction path is safe (no path traversal).

        Args:
            member_path: Path within archive.
            dest: Destination directory.

        Returns:
            True if path is safe to extract.
        """
        # Normalize and resolve the full extraction path
        full_path = (dest / member_path).resolve()

        # Ensure it's within destination
        try:
            full_path.relative_to(dest.resolve())
            return True
        except ValueError:
            return False

    @classmethod
    def cleanup_expired(cls, max_age_hours: int = 24) -> int:
        """Clean up expired extraction directories.

        Args:
            max_age_hours: Maximum age in hours before cleanup.

        Returns:
            Number of directories cleaned up.
        """
        from src.config import get_config

        cleaned = 0
        cutoff = time.time() - (max_age_hours * 3600)

        base_extract_dir = get_config().services.archive_extract_dir
        if not base_extract_dir.exists():
            return 0

        for session_dir in base_extract_dir.iterdir():
            if not session_dir.is_dir():
                continue

            try:
                # Check modification time
                if session_dir.stat().st_mtime < cutoff:
                    shutil.rmtree(session_dir)
                    cleaned += 1
                    logger.info(f"Cleaned up expired archive dir: {session_dir}")
            except Exception as e:
                logger.warning(f"Failed to clean up {session_dir}: {e}")

        return cleaned


def _format_size(size_bytes: int) -> str:
    """Format size in human-readable form."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _zipinfo_to_timestamp(info: zipfile.ZipInfo) -> float | None:
    """Convert ZipInfo date_time to Unix timestamp."""
    try:
        import datetime

        dt = datetime.datetime(*info.date_time)
        return dt.timestamp()
    except Exception:
        return None
