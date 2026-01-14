"""System tools - file operations, processes, environment."""

import os
import subprocess
import time
from pathlib import Path
from typing import Any

# Allowed paths for safety
ALLOWED_PATHS = ["/mnt/raid0/llm/", "/tmp/"]


def _is_path_allowed(path: str) -> bool:
    """Check if path is in allowed directories."""
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(allowed) for allowed in ALLOWED_PATHS)


def read_file(path: str, encoding: str = "utf-8", max_bytes: int = 1000000) -> dict:
    """Read file contents safely."""
    if not _is_path_allowed(path):
        return {"error": f"Path not allowed: {path}"}

    try:
        p = Path(path)
        if not p.exists():
            return {"error": f"File not found: {path}"}

        stat = p.stat()
        if stat.st_size > max_bytes:
            content = p.read_bytes()[:max_bytes].decode(encoding, errors="replace")
            content += f"\n... [truncated at {max_bytes} bytes]"
        else:
            content = p.read_text(encoding=encoding)

        return {
            "content": content,
            "size": stat.st_size,
            "modified": stat.st_mtime,
        }
    except Exception as e:
        return {"error": str(e)}


def write_file(path: str, content: str, mode: str = "w") -> dict:
    """Write content to file safely."""
    if not _is_path_allowed(path):
        return {"error": f"Path not allowed: {path}"}

    if mode not in ("w", "a"):
        return {"error": f"Invalid mode: {mode}"}

    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, mode, encoding="utf-8") as f:
            bytes_written = f.write(content)

        return {
            "bytes_written": bytes_written,
            "path": str(p.absolute()),
        }
    except Exception as e:
        return {"error": str(e)}


def list_directory(path: str, pattern: str = "*", recursive: bool = False) -> list[dict]:
    """List directory contents."""
    if not _is_path_allowed(path):
        return [{"error": f"Path not allowed: {path}"}]

    try:
        p = Path(path)
        if not p.exists():
            return [{"error": f"Directory not found: {path}"}]

        if recursive:
            items = list(p.rglob(pattern))
        else:
            items = list(p.glob(pattern))

        results = []
        for item in items[:1000]:  # Limit results
            try:
                stat = item.stat()
                results.append({
                    "name": item.name,
                    "path": str(item),
                    "type": "directory" if item.is_dir() else "file",
                    "size": stat.st_size if item.is_file() else 0,
                    "modified": stat.st_mtime,
                })
            except:
                continue

        return results
    except Exception as e:
        return [{"error": str(e)}]


def get_env(name: str, default: str | None = None) -> str | None:
    """Get environment variable."""
    return os.environ.get(name, default)


def disk_usage(path: str = "/mnt/raid0") -> dict:
    """Get disk usage statistics."""
    try:
        stat = os.statvfs(path)
        total = stat.f_blocks * stat.f_frsize
        free = stat.f_bfree * stat.f_frsize
        used = total - free
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "percent_used": round(used / total * 100, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def memory_usage() -> dict:
    """Get system memory usage."""
    try:
        with open("/proc/meminfo") as f:
            meminfo = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    value = int(parts[1])
                    meminfo[key] = value

        total = meminfo.get("MemTotal", 0) * 1024
        free = meminfo.get("MemFree", 0) * 1024
        available = meminfo.get("MemAvailable", 0) * 1024
        used = total - available

        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "available_gb": round(available / (1024**3), 2),
            "percent_used": round(used / total * 100, 1) if total > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}
