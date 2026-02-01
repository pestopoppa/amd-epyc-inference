#!/usr/bin/env python3
"""Unified orchestrator stack launcher.

Launches all models + orchestrator with granular reload support.

Usage:
    orchestrator_stack.py start [--hot-only] [--include-warm ROLE...] [--dev]
    orchestrator_stack.py stop [--all | COMPONENT...]
    orchestrator_stack.py reload COMPONENT...
    orchestrator_stack.py status

Examples:
    # Start HOT models only
    ./orchestrator_stack.py start --hot-only

    # Start with warm architect
    ./orchestrator_stack.py start --include-warm architect_general

    # Dev mode (single 0.5B model)
    ./orchestrator_stack.py start --dev

    # Reload orchestrator API after code changes
    ./orchestrator_stack.py reload orchestrator

    # Check status
    ./orchestrator_stack.py status
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.registry_loader import RegistryLoader

# =============================================================================
# Configuration
# =============================================================================

STATE_FILE = Path("/mnt/raid0/llm/claude/logs/orchestrator_state.json")
LLAMA_SERVER = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-server")
LOG_DIR = Path("/mnt/raid0/llm/claude/logs")

# Port assignments by role
PORT_MAP = {
    "frontdoor": 8080,
    "coder_primary": 8080,  # Shares with frontdoor (same model)
    "coder_escalation": 8081,
    # Worker pool ports (heterogeneous)
    "worker_general": 8082,  # Legacy alias -> worker_explore
    "worker_explore": 8082,  # Explore worker (7B)
    "worker_math": 8082,     # Shares with explore
    "worker_vision": 8086,   # Dedicated VL server
    "vision_escalation": 8087,  # VL escalation (Qwen3-VL-30B MoE)
    # worker_code REMOVED - route to coder_escalation (32B, faster + better quality)
    "worker_fast": 8102,     # Fast worker (1.5B, WARM, 4 slots)
    # Specialists
    "architect_general": 8083,
    "architect_coding": 8084,
    "ingest_long_context": 8085,
    "embedder": 8090,  # Embedding server for episodic memory
    "orchestrator": 8000,
    "document_formalizer": 9001,
}

# HOT roles (always started) - includes architects in HOT tier (510GB total, 45% of 1130GB RAM)
HOT_ROLES = {
    "frontdoor", "coder_escalation", "worker_explore", "embedder",
    "architect_general", "architect_coding", "ingest_long_context",
    "worker_vision", "vision_escalation",
}

# Servers to start (unique ports only)
# HOT tier uses ~510GB total (45% of 1130GB RAM), leaving 620GB for KV cache
HOT_SERVERS = [
    {"port": 8080, "roles": ["frontdoor", "coder_primary"]},
    {"port": 8081, "roles": ["coder_escalation", "worker_summarize"]},  # Added worker_summarize
    # Worker pool HOT tier
    {"port": 8082, "roles": ["worker_explore", "worker_general", "worker_math"],
     "worker_pool": True, "worker_type": "explore"},
    # Vision servers (VL models with multimodal projector, NO spec decode)
    {"port": 8086, "roles": ["worker_vision"], "vision": True, "vision_type": "worker"},
    {"port": 8087, "roles": ["vision_escalation"], "vision": True, "vision_type": "escalation"},
    # worker_code REMOVED - route to coder_escalation (32B is faster + better quality)
    {"port": 8090, "roles": ["embedder"], "embedding": True},  # Embedding server
    # Architects in HOT tier (always resident)
    {"port": 8083, "roles": ["architect_general"]},
    {"port": 8084, "roles": ["architect_coding"]},
    {"port": 8085, "roles": ["ingest_long_context"]},
]

# Embedding model (lightweight, always loaded)
EMBEDDING_MODEL_PATH = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

# Worker pool models (FIXED paths to existing files)
# NOTE: worker_code removed - route all code tasks to coder_escalation (32B, faster + better quality)
WORKER_POOL_MODELS = {
    "explore": "/mnt/raid0/llm/models/Qwen2.5-7B-Instruct-f16.gguf",
    "fast": "/mnt/raid0/llm/lmstudio/models/QuantFactory/Qwen2.5-Coder-1.5B-GGUF/Qwen2.5-Coder-1.5B.Q4_K_M.gguf",
}

# Draft model for speculative decoding on explore worker
EXPLORE_DRAFT_MODEL = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-Coder-0.5B-GGUF/Qwen2.5-Coder-0.5B-Q8_0.gguf"

# Vision models (VL) with multimodal projector
VISION_WORKER_MODEL = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
VISION_WORKER_MMPROJ = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf"
VISION_ESCALATION_MODEL = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf"
VISION_ESCALATION_MMPROJ = "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-VL-30B-A3B-Instruct-GGUF/mmproj-Qwen3-VL-30B-A3B-Instruct-F16.gguf"

WARM_SERVERS = [
    {"port": 8083, "roles": ["architect_general"]},
    {"port": 8084, "roles": ["architect_coding"]},
    {"port": 8085, "roles": ["ingest_long_context"]},
    # Worker pool WARM tier (single fast worker with 4 slots for burst capacity)
    {"port": 8102, "roles": ["worker_fast"],
     "worker_pool": True, "worker_type": "fast"},
]

DEV_MODEL = "Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"
DEV_MODEL_PATH = "/mnt/raid0/llm/models/Qwen2.5-Coder-0.5B-Instruct-Q8_0.gguf"


# =============================================================================
# Model Path Validation
# =============================================================================


def validate_model_paths() -> list[str]:
    """Validate all model paths exist. Returns list of errors.

    This prevents hallucinations about missing models by failing fast
    with clear error messages showing exactly what's missing.
    """
    errors = []

    # HOT tier models
    if not Path(EMBEDDING_MODEL_PATH).exists():
        errors.append(f"[HOT] Embedding: {EMBEDDING_MODEL_PATH}")

    for worker_type, path in WORKER_POOL_MODELS.items():
        if not Path(path).exists():
            errors.append(f"[HOT] Worker '{worker_type}': {path}")

    # Draft model for explore worker spec decode
    if not Path(EXPLORE_DRAFT_MODEL).exists():
        errors.append(f"[HOT] Explore draft: {EXPLORE_DRAFT_MODEL}")

    # Architect models (now in HOT tier)
    architect_models = [
        ("architect_general", "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-235B-A22B-GGUF/"),
        ("architect_coding", "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Coder-480B-A35B-Instruct-GGUF/"),
        ("ingest_long_context", "/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen3-Next-80B-A3B-Instruct-GGUF/"),
    ]
    for role, path in architect_models:
        if not Path(path).exists():
            errors.append(f"[HOT] {role}: {path}")

    # Vision models (VL with multimodal projector)
    for label, path in [
        ("worker_vision model", VISION_WORKER_MODEL),
        ("worker_vision mmproj", VISION_WORKER_MMPROJ),
        ("vision_escalation model", VISION_ESCALATION_MODEL),
        ("vision_escalation mmproj", VISION_ESCALATION_MMPROJ),
    ]:
        if not Path(path).exists():
            errors.append(f"[HOT] {label}: {path}")

    # Auxiliary services
    formalizer = "/mnt/raid0/llm/models/LightOnOCR-2-1B-bbox-Q4_K_M.gguf"
    if not Path(formalizer).exists():
        errors.append(f"[AUX] document_formalizer: {formalizer}")

    # Tool registry (required for deterministic tools)
    tool_registry = Path("/mnt/raid0/llm/claude/orchestration/tool_registry.yaml")
    if not tool_registry.exists():
        errors.append(f"[TOOL] tool_registry.yaml: {tool_registry}")

    # C++ math tools (optional - warn but don't fail)
    cpp_math_tools = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-math-tools")
    if not cpp_math_tools.exists():
        # This is a warning, not an error - append with different prefix
        pass  # Will be checked separately in init_memrl_and_tools

    return errors


# =============================================================================
# State Management
# =============================================================================


@dataclass
class ProcessInfo:
    """Information about a running process."""
    role: str
    pid: int
    port: int
    started_at: str
    model_path: str
    log_file: str


def load_state() -> dict[str, ProcessInfo]:
    """Load state from file."""
    if not STATE_FILE.exists():
        return {}
    try:
        with open(STATE_FILE) as f:
            data = json.load(f)
        return {k: ProcessInfo(**v) for k, v in data.items()}
    except (json.JSONDecodeError, TypeError):
        return {}


def save_state(state: dict[str, ProcessInfo]) -> None:
    """Save state to file."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({k: asdict(v) for k, v in state.items()}, f, indent=2)


# =============================================================================
# Process Management
# =============================================================================


def check_free_memory() -> int:
    """Return free memory in GB."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb // (1024 * 1024)
    return 0


def is_port_in_use(port: int) -> bool:
    """Check if a port is in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process(pid: int, timeout: int = 5) -> bool:
    """Kill a process gracefully, then forcefully."""
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(timeout):
            time.sleep(1)
            try:
                os.kill(pid, 0)  # Check if still alive
            except ProcessLookupError:
                return True
        # Force kill
        os.kill(pid, signal.SIGKILL)
        time.sleep(1)
        return True
    except ProcessLookupError:
        return True
    except PermissionError:
        print(f"  [!] Permission denied killing PID {pid}")
        return False


def wait_for_health(port: int, timeout: int = 120) -> bool:
    """Wait for server health endpoint."""
    import urllib.request
    import urllib.error

    url = f"http://localhost:{port}/health"
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    return False


# =============================================================================
# Server Launching
# =============================================================================


def build_server_command(
    role_config: Any,
    port: int,
    dev_mode: bool = False,
    embedding_mode: bool = False,
    worker_pool_mode: bool = False,
    worker_type: str = None,
    vision_mode: bool = False,
    vision_type: str = None,
) -> list[str]:
    """Build llama-server command from role config."""
    # Vision server mode - VL models with multimodal projector
    if vision_mode:
        if vision_type == "escalation":
            # Qwen3-VL-30B MoE - larger model, expert reduction
            return [
                str(LLAMA_SERVER),
                "-m", VISION_ESCALATION_MODEL,
                "--mmproj", VISION_ESCALATION_MMPROJ,
                "--override-kv", "qwen3vlmoe.expert_used_count=int:4",
                "--host", "127.0.0.1",
                "--port", str(port),
                "-np", "2",
                "-c", "16384",
                "-t", "96",
                "--flash-attn", "on",
            ]
        else:
            # Qwen2.5-VL-7B - smaller worker model
            return [
                str(LLAMA_SERVER),
                "-m", VISION_WORKER_MODEL,
                "--mmproj", VISION_WORKER_MMPROJ,
                "--host", "127.0.0.1",
                "--port", str(port),
                "-np", "2",
                "-c", "8192",
                "-t", "24",
                "--flash-attn", "on",
            ]

    # Embedding server mode - lightweight, dedicated to embeddings
    if embedding_mode:
        return [
            str(LLAMA_SERVER),
            "-m", EMBEDDING_MODEL_PATH,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-np", "4",  # 4 parallel slots for embedding requests
            "-c", "4096",  # Small context (embeddings don't need long context)
            "-t", "8",  # 8 threads sufficient for small model
            "--embeddings",  # Enable embedding endpoint
            "--flash-attn", "on",
        ]

    # Worker pool mode - heterogeneous workers with specific configs
    if worker_pool_mode and worker_type:
        model_path = WORKER_POOL_MODELS.get(worker_type)
        if not model_path:
            raise ValueError(f"Unknown worker type: {worker_type}")

        # Worker-type specific configuration
        if worker_type == "fast":
            # Fast worker: 1.5B model, 4 slots for parallel burst capacity
            return [
                str(LLAMA_SERVER),
                "-m", model_path,
                "--host", "127.0.0.1",
                "--port", str(port),
                "-np", "4",  # 4 parallel slots (consolidated from 2×2)
                "-c", "16384",  # 4K per slot
                "-t", "16",  # 16 threads for small model
                "--flash-attn", "on",
            ]
        else:
            # explore workers: 7B model with speculative decoding for 46 t/s
            # Summarization handled by worker_summarize (32B) on port 8081
            return [
                str(LLAMA_SERVER),
                "-m", model_path,
                "-md", EXPLORE_DRAFT_MODEL,  # Spec decode with 0.5B draft
                "--draft-max", "24",  # K=24 for optimal speedup
                "--lookup",  # Prompt n-gram lookup as fallback when spec insufficient
                "--host", "127.0.0.1",
                "--port", str(port),
                "-np", "2",  # 2 parallel slots
                "-c", "8192",  # 4K per slot
                "-t", "24",  # 24 threads for 7B model
                "--flash-attn", "on",
            ]

    if dev_mode:
        return [
            str(LLAMA_SERVER),
            "-m", DEV_MODEL_PATH,
            "--host", "127.0.0.1",
            "--port", str(port),
            "-np", "4",
            "-c", "4096",
            "-t", "16",
            "--flash-attn", "on",  # Flash attention
        ]

    model_path = role_config.model.full_path
    accel = role_config.acceleration

    cmd = [
        str(LLAMA_SERVER),
        "-m", model_path,
        "--host", "127.0.0.1",
        "--port", str(port),
        "-np", "2",  # Parallel slots (2 slots for larger context per slot)
        "-c", "32768",  # Context size (16K per slot with np=2)
        "-t", "96",  # Threads
        "--flash-attn", "on",  # Flash attention
    ]

    # Add acceleration based on type
    if accel.type == "moe_expert_reduction" and accel.experts:
        cmd.extend([
            "--override-kv",
            f"{accel.override_key}=int:{accel.experts}",
        ])
    elif accel.type == "speculative_decoding" and accel.draft_role:
        # Get draft model path from registry
        registry = RegistryLoader()
        draft_config = registry.get_role(accel.draft_role)
        if draft_config:
            cmd.extend([
                "-md", draft_config.model.full_path,
                "--draft-max", str(accel.k or 16),
            ])

    # Add prompt n-gram lookup (spec-first, lookup-fallback) for servers with spec decode
    # Combined mode: 5.4x vs 5.2x spec-only (production-consolidated commit 8e35dbc01)
    if accel.type == "speculative_decoding":
        cmd.append("--lookup")

    return cmd


def start_server(
    port: int,
    roles: list[str],
    registry: RegistryLoader,
    dev_mode: bool = False,
    embedding_mode: bool = False,
    worker_pool_mode: bool = False,
    worker_type: str = None,
    vision_mode: bool = False,
    vision_type: str = None,
) -> ProcessInfo | None:
    """Start a llama-server for the given roles."""
    # Vision mode - VL models with multimodal projector
    if vision_mode:
        log_file = LOG_DIR / f"vision-{vision_type or 'worker'}-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        if vision_type == "escalation":
            model_path = VISION_ESCALATION_MODEL
            model_name = "Qwen3-VL-30B-A3B (vision escalation)"
        else:
            model_path = VISION_WORKER_MODEL
            model_name = "Qwen2.5-VL-7B (vision worker)"

        cmd = build_server_command(
            None, port, vision_mode=True, vision_type=vision_type
        )

        print(f"  Starting vision server [{vision_type or 'worker'}] on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        with open(log_file, "w") as log:
            env = os.environ.copy()
            proc = subprocess.Popen(
                ["numactl", "--interleave=all"] + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )

        print(f"    PID: {proc.pid}")
        print(f"    Waiting for health...")

        # VL models take longer to load (mmproj + main model)
        timeout = 120 if vision_type == "escalation" else 90
        if wait_for_health(port, timeout=timeout):
            print(f"    [OK] Vision server {vision_type or 'worker'} ready")
            return ProcessInfo(
                role=roles[0],
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Vision server {vision_type or 'worker'} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Embedding mode uses dedicated config, no registry lookup needed
    if embedding_mode:
        log_file = LOG_DIR / f"llama-server-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        cmd = build_server_command(None, port, dev_mode=False, embedding_mode=True)
        model_name = "Qwen2.5-Coder-0.5B (embeddings)"

        print(f"  Starting port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:5])}...")

        with open(log_file, "w") as log:
            env = os.environ.copy()
            # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
            proc = subprocess.Popen(
                ["numactl", "--interleave=all"] + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )

        print(f"    PID: {proc.pid}")
        print(f"    Waiting for health...")

        if wait_for_health(port, timeout=60):  # Faster timeout for small model
            print(f"    [OK] Embedding server ready")
            return ProcessInfo(
                role="embedder",
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=EMBEDDING_MODEL_PATH,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Embedding server did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Worker pool mode - heterogeneous workers
    if worker_pool_mode and worker_type:
        log_file = LOG_DIR / f"worker-{worker_type}-{port}.log"
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        model_path = WORKER_POOL_MODELS.get(worker_type)
        if not model_path:
            print(f"  [!] Unknown worker type: {worker_type}")
            return None

        cmd = build_server_command(
            None, port, worker_pool_mode=True, worker_type=worker_type
        )
        model_name = Path(model_path).stem

        print(f"  Starting worker pool [{worker_type}] on port {port}: {model_name}")
        print(f"    Roles: {', '.join(roles)}")
        print(f"    Command: {' '.join(cmd[:6])}...")

        with open(log_file, "w") as log:
            env = os.environ.copy()
            # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
            proc = subprocess.Popen(
                ["numactl", "--interleave=all"] + cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
            )

        print(f"    PID: {proc.pid}")
        print(f"    Waiting for health...")

        # Faster timeout for smaller models
        timeout = 60 if worker_type == "fast" else 90
        if wait_for_health(port, timeout=timeout):
            print(f"    [OK] Worker {worker_type} ready")
            return ProcessInfo(
                role=f"worker_{worker_type}",
                pid=proc.pid,
                port=port,
                started_at=datetime.now().isoformat(),
                model_path=model_path,
                log_file=str(log_file),
            )
        else:
            print(f"    [FAIL] Worker {worker_type} did not become healthy")
            print(f"    Check log: {log_file}")
            kill_process(proc.pid)
            return None

    # Use first role's config for the server
    primary_role = roles[0]
    role_config = registry.get_role(primary_role)

    if not role_config and not dev_mode:
        print(f"  [!] Role {primary_role} not found in registry")
        return None

    log_file = LOG_DIR / f"llama-server-{port}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Build command
    cmd = build_server_command(role_config, port, dev_mode)

    model_name = DEV_MODEL if dev_mode else role_config.model.name

    print(f"  Starting port {port}: {model_name}")
    print(f"    Roles: {', '.join(roles)}")
    print(f"    Command: {' '.join(cmd[:5])}...")

    # Start process
    with open(log_file, "w") as log:
        env = os.environ.copy()
        # NOTE: Do NOT set OMP_NUM_THREADS=1 - it disables parallel tensor repack (2.2x slower loading)
        proc = subprocess.Popen(
            ["numactl", "--interleave=all"] + cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f"    PID: {proc.pid}")

    # Wait for health
    print(f"    Waiting for health...")
    if wait_for_health(port, timeout=180):
        print(f"    [OK] Server ready")
        return ProcessInfo(
            role=primary_role,
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path=DEV_MODEL_PATH if dev_mode else role_config.model.full_path,
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] Server did not become healthy")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def start_orchestrator() -> ProcessInfo | None:
    """Start the orchestrator API."""
    log_file = LOG_DIR / "orchestrator.log"

    print("  Starting orchestrator API on port 8000")

    # Set environment — enable production feature flags
    env = os.environ.copy()
    env["HF_HOME"] = "/mnt/raid0/llm/cache/huggingface"
    env["TMPDIR"] = "/mnt/raid0/llm/tmp"
    # Feature flags: enable production capabilities
    env["ORCHESTRATOR_MEMRL"] = "1"
    env["ORCHESTRATOR_TOOLS"] = "1"
    env["ORCHESTRATOR_SCRIPTS"] = "1"
    # NOTE: Do NOT set ORCHESTRATOR_REPL here — it collides with
    # OrchestratorSettings.repl (REPLSettings model) in config.py.
    # The repl feature flag defaults to True in features.py already.
    env["ORCHESTRATOR_CACHING"] = "1"
    env["ORCHESTRATOR_STREAMING"] = "1"
    env["ORCHESTRATOR_MOCK_MODE"] = "0"
    env["ORCHESTRATOR_GENERATION_MONITOR"] = "1"
    env["ORCHESTRATOR_REACT_MODE"] = "1"

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "src.api:app",
                "--host", "127.0.0.1",
                "--port", "8000",
            ],
            cwd="/mnt/raid0/llm/claude",
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f"    PID: {proc.pid}")
    print(f"    Waiting for health...")

    if wait_for_health(8000, timeout=60):
        print(f"    [OK] Orchestrator ready")
        return ProcessInfo(
            role="orchestrator",
            pid=proc.pid,
            port=8000,
            started_at=datetime.now().isoformat(),
            model_path="uvicorn",
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] Orchestrator did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


def start_document_formalizer() -> ProcessInfo | None:
    """Start the document formalizer (LightOnOCR-2) server."""
    log_file = LOG_DIR / "document_formalizer.log"
    port = 9001

    print(f"  Starting document_formalizer (LightOnOCR-2) on port {port}")

    # Set environment
    env = os.environ.copy()
    env["LIGHTONOCR_WORKERS"] = "8"
    env["LIGHTONOCR_THREADS"] = "12"
    env["LIGHTONOCR_MAX_TOKENS"] = "2048"
    env["LIGHTONOCR_TIMEOUT"] = "120"

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            [
                sys.executable,
                "/mnt/raid0/llm/claude/src/services/lightonocr_llama_server.py",
                "--port", str(port),
            ],
            cwd="/mnt/raid0/llm/claude",
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )

    print(f"    PID: {proc.pid}")
    print(f"    Waiting for health...")

    if wait_for_health(port, timeout=60):
        print(f"    [OK] Document formalizer ready")
        return ProcessInfo(
            role="document_formalizer",
            pid=proc.pid,
            port=port,
            started_at=datetime.now().isoformat(),
            model_path="LightOnOCR-2-1B-bbox",
            log_file=str(log_file),
        )
    else:
        print(f"    [FAIL] Document formalizer did not start")
        print(f"    Check log: {log_file}")
        kill_process(proc.pid)
        return None


# =============================================================================
# Commands
# =============================================================================


def cmd_start(args: argparse.Namespace) -> int:
    """Start the orchestrator stack."""
    print("=" * 60)
    print("ORCHESTRATOR STACK STARTUP")
    print("=" * 60)
    print()

    # Check memory
    free_gb = check_free_memory()
    print(f"[i] Free memory: {free_gb} GB")
    if free_gb < 100 and not args.dev:
        print("[!] WARNING: Less than 100GB free. Consider --dev mode.")
        if input("Continue? (y/N) ").lower() != "y":
            return 1
    print()

    # Load registry
    registry = RegistryLoader()
    state: dict[str, ProcessInfo] = {}

    # Validate model paths (prevents hallucinations about missing models)
    if not args.dev:
        print("[0.5] Validating model paths...")
        errors = validate_model_paths()
        if errors:
            print("[!] MODEL VALIDATION FAILED:")
            for err in errors:
                print(f"    - {err}")
            print("\nFix missing models or update paths in orchestrator_stack.py")
            print("Check /mnt/raid0/llm/models/ and /mnt/raid0/llm/lmstudio/models/")
            return 1
        print("  [OK] All model paths validated")
        print()

    # Kill existing processes on target ports
    print("[1] Cleaning up existing processes...")
    for server in HOT_SERVERS + WARM_SERVERS:
        port = server["port"]
        if is_port_in_use(port):
            print(f"  Port {port} in use, attempting cleanup...")
            # Find PID from lsof
            try:
                result = subprocess.run(
                    ["lsof", "-t", f"-i:{port}"],
                    capture_output=True,
                    text=True,
                )
                if result.stdout.strip():
                    for pid_str in result.stdout.strip().split("\n"):
                        pid = int(pid_str)
                        kill_process(pid)
            except Exception as e:
                print(f"  [!] Error cleaning port {port}: {e}")
    print()

    # Determine which servers to start
    servers_to_start = []

    if args.dev:
        print("[2] Starting in DEV mode (single 0.5B model)...")
        servers_to_start = [{"port": 8080, "roles": ["dev"]}]
    else:
        print("[2] Starting HOT servers...")
        servers_to_start = HOT_SERVERS.copy()

        # Add warm servers if requested
        if args.include_warm:
            for warm_server in WARM_SERVERS:
                for role in warm_server["roles"]:
                    if role in args.include_warm:
                        servers_to_start.append(warm_server)
                        print(f"  Including WARM server: port {warm_server['port']} ({role})")
                        break

    print()

    # Start servers sequentially
    print("[3] Starting llama-servers...")
    for i, server in enumerate(servers_to_start):
        port = server["port"]
        roles = server["roles"]
        embedding_mode = server.get("embedding", False)
        worker_pool_mode = server.get("worker_pool", False)
        worker_type = server.get("worker_type")
        vision_mode = server.get("vision", False)
        vision_type = server.get("vision_type")

        info = start_server(
            port, roles, registry, args.dev,
            embedding_mode=embedding_mode,
            worker_pool_mode=worker_pool_mode,
            worker_type=worker_type,
            vision_mode=vision_mode,
            vision_type=vision_type,
        )
        if info:
            state[f"server_{port}"] = info
            # Also map all roles to this server
            for role in roles:
                if role not in state:
                    state[role] = info
        else:
            print(f"  [!] Failed to start server on port {port}")
            # Embedding/worker_pool/vision server failure is non-fatal (fallback available)
            is_optional = embedding_mode or worker_pool_mode or vision_mode
            if not args.dev and not is_optional:
                return 1

        # Brief cooldown between large models to allow mmap settling
        # With parallel tensor repack enabled, 5s is sufficient
        is_small_model = embedding_mode or (worker_pool_mode and worker_type == "fast") or (vision_mode and vision_type != "escalation")
        if i < len(servers_to_start) - 1 and not args.dev and not is_small_model:
            print("  Cooldown (5s)...")
            time.sleep(5)

    print()

    # Start orchestrator
    print("[4] Starting orchestrator API...")
    info = start_orchestrator()
    if info:
        state["orchestrator"] = info
    else:
        print("  [!] Failed to start orchestrator")
        return 1

    print()

    # Start document formalizer (optional, non-fatal)
    if not args.dev:
        print("[5] Starting document formalizer (LightOnOCR-2)...")
        info = start_document_formalizer()
        if info:
            state["document_formalizer"] = info
        else:
            print("  [!] Document formalizer failed (non-fatal, continuing)")

        print()

        # Initialize MemRL databases and tool registry
        init_memrl_and_tools()

        print()

    # Save state
    save_state(state)
    print(f"[i] State saved to {STATE_FILE}")
    print()

    # Final status
    print("=" * 60)
    print("STACK READY")
    print("=" * 60)
    cmd_status(args)

    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    """Stop components."""
    state = load_state()

    if not state:
        print("No running components found")
        return 0

    targets = []
    if args.all:
        targets = list(state.keys())
    elif args.components:
        targets = args.components
    else:
        print("Specify --all or component names")
        return 1

    for name in targets:
        if name in state:
            info = state[name]
            print(f"Stopping {name} (PID {info.pid})...")
            if kill_process(info.pid):
                del state[name]
                print(f"  [OK] Stopped")
            else:
                print(f"  [!] Failed to stop")
        else:
            print(f"  [?] {name} not found in state")

    save_state(state)
    return 0


def cmd_reload(args: argparse.Namespace) -> int:
    """Reload components."""
    state = load_state()
    registry = RegistryLoader()

    for component in args.components:
        print(f"Reloading {component}...")

        if component == "orchestrator":
            # Stop existing
            if "orchestrator" in state:
                kill_process(state["orchestrator"].pid)
                time.sleep(2)

            # Start new
            info = start_orchestrator()
            if info:
                state["orchestrator"] = info
            else:
                print(f"  [!] Failed to restart orchestrator")
                return 1

        elif component in PORT_MAP:
            port = PORT_MAP[component]
            key = f"server_{port}"

            # Find roles and config for this port
            roles = [component]
            worker_pool_mode = False
            worker_type = None
            embedding_mode = False
            vision_mode = False
            vision_type = None

            for server in HOT_SERVERS + WARM_SERVERS:
                if server["port"] == port:
                    roles = server["roles"]
                    worker_pool_mode = server.get("worker_pool", False)
                    worker_type = server.get("worker_type")
                    embedding_mode = server.get("embedding", False)
                    vision_mode = server.get("vision", False)
                    vision_type = server.get("vision_type")
                    break

            # Stop existing
            if key in state:
                kill_process(state[key].pid)
                time.sleep(2)

            # Start new
            info = start_server(
                port, roles, registry, dev_mode=False,
                embedding_mode=embedding_mode,
                worker_pool_mode=worker_pool_mode,
                worker_type=worker_type,
                vision_mode=vision_mode,
                vision_type=vision_type,
            )
            if info:
                state[key] = info
                for role in roles:
                    state[role] = info
            else:
                print(f"  [!] Failed to restart {component}")
                return 1

        else:
            print(f"  [?] Unknown component: {component}")

    save_state(state)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show status of all components."""
    state = load_state()

    if not state:
        print("No components running")
        return 0

    print()
    print(f"{'COMPONENT':<25} {'PORT':<8} {'PID':<10} {'STATUS':<10} {'MODEL'}")
    print("-" * 80)

    seen_pids = set()
    for name, info in sorted(state.items()):
        if info.pid in seen_pids:
            continue  # Skip duplicates (roles sharing servers)
        seen_pids.add(info.pid)

        # Check if process is alive
        try:
            os.kill(info.pid, 0)
            alive = True
        except ProcessLookupError:
            alive = False

        # Check health endpoint
        healthy = False
        if alive:
            healthy = wait_for_health(info.port, timeout=3)

        status = "healthy" if healthy else ("running" if alive else "dead")
        model = Path(info.model_path).stem if info.model_path != "uvicorn" else "uvicorn"

        print(f"{name:<25} {info.port:<8} {info.pid:<10} {status:<10} {model[:30]}")

    print()
    print(f"State file: {STATE_FILE}")
    return 0


# =============================================================================
# MemRL and Tool Registry Initialization
# =============================================================================


def init_memrl_and_tools() -> bool:
    """Initialize MemRL databases and tool registry for the session.

    This ensures all deterministic tools (41 total) are ready and
    the REPL memory system is initialized with seed examples.
    """
    success = True

    # [6] REPL Memory Initialization
    print("[6] Initializing MemRL databases...")

    # Initialize REPL seed examples
    seed_loader_path = Path("/mnt/raid0/llm/claude/orchestration/repl_memory/seed_loader.py")
    if seed_loader_path.exists():
        result = subprocess.run(
            [sys.executable, str(seed_loader_path), "--init"],
            capture_output=True,
            text=True,
            cwd="/mnt/raid0/llm/claude",
        )
        if result.returncode == 0:
            print("  [OK] REPL seed examples loaded")
        else:
            print(f"  [WARN] Seed loader failed: {result.stderr[:100] if result.stderr else 'no output'}")

    # Warm up embedding model with test query
    try:
        import urllib.request
        import urllib.error

        test_payload = json.dumps({"content": "test embedding warmup"}).encode()
        req = urllib.request.Request(
            "http://localhost:8090/embedding",
            data=test_payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status == 200:
                print("  [OK] Embedding model warmed up")
    except Exception as e:
        print(f"  [WARN] Embedding warmup failed: {e}")

    # [7] Tool Registry Initialization
    print("[7] Initializing deterministic tool registry...")

    # Validate tool registry exists
    tool_registry_path = Path("/mnt/raid0/llm/claude/orchestration/tool_registry.yaml")
    if not tool_registry_path.exists():
        print(f"  [!] Tool registry not found: {tool_registry_path}")
        success = False
    else:
        # Load and validate tool executor
        try:
            # Add src to path for imports
            import sys as _sys
            _sys.path.insert(0, "/mnt/raid0/llm/claude")
            from orchestration.tools.executor import get_executor
            executor = get_executor()
            tools = executor.list_tools()
            print(f"  [OK] Tool registry loaded: {len(tools)} tools")

            # Categorize tools
            categories: dict[str, int] = {}
            for t in tools:
                cat = t.get("category", "other")
                categories[cat] = categories.get(cat, 0) + 1
            for cat, count in sorted(categories.items()):
                print(f"      {cat}: {count}")
        except Exception as e:
            print(f"  [WARN] Tool executor init failed: {e}")

    # Verify C++ math tools binary
    cpp_binary = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-math-tools")
    if cpp_binary.exists():
        print("  [OK] C++ math tools binary found")
    else:
        print(f"  [WARN] C++ math tools not built: {cpp_binary}")
        print("        Run: cd /mnt/raid0/llm/llama.cpp && make llama-math-tools")

    return success


# =============================================================================
# Checkpoint Hooks for Self-Management Procedures
# =============================================================================

CHECKPOINT_DIR = Path("/mnt/raid0/llm/claude/orchestration/checkpoints")


def checkpoint_create(name: str, include_state: bool = True) -> dict[str, Any]:
    """Create a checkpoint of the orchestrator stack state.

    Called by self-management procedures before making changes.

    Args:
        name: Descriptive checkpoint name.
        include_state: Whether to include server state.

    Returns:
        Dict with checkpoint_id and path.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_id = f"{name}_{timestamp}"
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_id}.json"

    checkpoint_data = {
        "id": checkpoint_id,
        "name": name,
        "created_at": datetime.now().isoformat(),
        "state": {},
        "registry_snapshot": None,
    }

    # Capture current state
    if include_state:
        state = load_state()
        checkpoint_data["state"] = {k: asdict(v) for k, v in state.items()}

    # Snapshot of registry (just metadata, not full file)
    registry_path = Path("/mnt/raid0/llm/claude/orchestration/model_registry.yaml")
    if registry_path.exists():
        checkpoint_data["registry_snapshot"] = {
            "path": str(registry_path),
            "mtime": registry_path.stat().st_mtime,
            "size": registry_path.stat().st_size,
        }

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    return {
        "checkpoint_id": checkpoint_id,
        "path": str(checkpoint_path),
        "created_at": checkpoint_data["created_at"],
    }


def checkpoint_restore(checkpoint_id: str) -> dict[str, Any]:
    """Restore orchestrator stack from a checkpoint.

    Args:
        checkpoint_id: ID from checkpoint_create.

    Returns:
        Dict with restoration status.
    """
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_id}.json"

    if not checkpoint_path.exists():
        return {"success": False, "error": f"Checkpoint not found: {checkpoint_id}"}

    try:
        with open(checkpoint_path) as f:
            checkpoint_data = json.load(f)

        # Restore state (process info)
        if checkpoint_data.get("state"):
            saved_state = {
                k: ProcessInfo(**v)
                for k, v in checkpoint_data["state"].items()
            }
            save_state(saved_state)

        return {
            "success": True,
            "checkpoint_id": checkpoint_id,
            "restored_at": datetime.now().isoformat(),
            "original_created_at": checkpoint_data.get("created_at"),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def checkpoint_list(limit: int = 10) -> list[dict[str, Any]]:
    """List available checkpoints.

    Args:
        limit: Maximum number to return (newest first).

    Returns:
        List of checkpoint summaries.
    """
    if not CHECKPOINT_DIR.exists():
        return []

    checkpoints = []
    for cp_path in sorted(CHECKPOINT_DIR.glob("*.json"), reverse=True)[:limit]:
        try:
            with open(cp_path) as f:
                data = json.load(f)
            checkpoints.append({
                "id": data.get("id", cp_path.stem),
                "name": data.get("name"),
                "created_at": data.get("created_at"),
                "path": str(cp_path),
            })
        except Exception:
            pass

    return checkpoints


def checkpoint_delete(checkpoint_id: str) -> bool:
    """Delete a checkpoint.

    Args:
        checkpoint_id: Checkpoint to delete.

    Returns:
        True if deleted, False if not found.
    """
    checkpoint_path = CHECKPOINT_DIR / f"{checkpoint_id}.json"
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        return True
    return False


# Export hooks for use by procedure_registry
__checkpoint_hooks__ = {
    "create": checkpoint_create,
    "restore": checkpoint_restore,
    "list": checkpoint_list,
    "delete": checkpoint_delete,
}


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Orchestrator stack manager")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Start command
    start_parser = subparsers.add_parser("start", help="Start the stack")
    start_parser.add_argument("--hot-only", action="store_true", help="Start HOT models only")
    start_parser.add_argument("--include-warm", nargs="+", metavar="ROLE", help="Include WARM models")
    start_parser.add_argument("--dev", action="store_true", help="Dev mode (single 0.5B model)")

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop components")
    stop_parser.add_argument("--all", action="store_true", help="Stop all components")
    stop_parser.add_argument("components", nargs="*", help="Components to stop")

    # Reload command
    reload_parser = subparsers.add_parser("reload", help="Reload components")
    reload_parser.add_argument("components", nargs="+", help="Components to reload")

    # Status command
    subparsers.add_parser("status", help="Show status")

    args = parser.parse_args()

    if args.command == "start":
        return cmd_start(args)
    elif args.command == "stop":
        return cmd_stop(args)
    elif args.command == "reload":
        return cmd_reload(args)
    elif args.command == "status":
        return cmd_status(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
