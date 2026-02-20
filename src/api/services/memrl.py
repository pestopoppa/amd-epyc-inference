"""MemRL service for lazy-loaded memory-based RL components.

This module handles:
- Lazy loading of optional MemRL imports based on feature flags
- Initialization of Q-scorer and HybridRouter on first use
- Background task scoring for completed tasks
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING

from src.features import features

if TYPE_CHECKING:
    from src.api.state import AppState

logger = logging.getLogger(__name__)

# Optional imports - populated by load_optional_imports()
ProgressLogger: type | None = None
ProgressReader: type | None = None
EpisodicStore: type | None = None
TaskEmbedder: type | None = None
QScorer: type | None = None
ScoringConfig: type | None = None
TwoPhaseRetriever: type | None = None
HybridRouter: type | None = None
RuleBasedRouter: type | None = None
RetrievalConfig: type | None = None
ToolRegistry: type | None = None
ScriptRegistry: type | None = None
SkillBank: type | None = None
SkillRetriever: type | None = None
SkillAugmentedRouter: type | None = None


def _background_scoring_enabled() -> bool:
    """Return whether idle-time MemRL background scoring is enabled.

    Env override:
    - ORCHESTRATOR_MEMRL_BACKGROUND=0 disables background cleanup scoring.
    """
    raw = os.environ.get("ORCHESTRATOR_MEMRL_BACKGROUND", "1").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def load_optional_imports() -> None:
    """Load optional module imports based on feature flags.

    This is called during app startup to load only the modules needed
    for the enabled features, improving startup time and reducing
    memory usage when features are disabled.
    """
    global ProgressLogger, ProgressReader, EpisodicStore, TaskEmbedder
    global QScorer, ScoringConfig, TwoPhaseRetriever, HybridRouter
    global RuleBasedRouter, RetrievalConfig, ToolRegistry, ScriptRegistry
    global SkillBank, SkillRetriever, SkillAugmentedRouter

    f = features()

    # MemRL components
    if f.memrl:
        from orchestration.repl_memory.progress_logger import (
            ProgressLogger as _PL,
            ProgressReader as _PR,
        )
        from orchestration.repl_memory.episodic_store import EpisodicStore as _ES
        from orchestration.repl_memory.embedder import TaskEmbedder as _TE
        from orchestration.repl_memory.q_scorer import QScorer as _QS, ScoringConfig as _SC
        from orchestration.repl_memory.retriever import (
            TwoPhaseRetriever as _TPR,
            HybridRouter as _HR,
            RuleBasedRouter as _RBR,
            RetrievalConfig as _RC,
        )

        ProgressLogger = _PL
        ProgressReader = _PR
        EpisodicStore = _ES
        TaskEmbedder = _TE
        QScorer = _QS
        ScoringConfig = _SC
        TwoPhaseRetriever = _TPR
        HybridRouter = _HR
        RuleBasedRouter = _RBR
        RetrievalConfig = _RC

        # SkillBank components (requires memrl + skillbank flags)
        if f.skillbank:
            from orchestration.repl_memory.skill_bank import SkillBank as _SB
            from orchestration.repl_memory.skill_retriever import SkillRetriever as _SR2
            from orchestration.repl_memory.retriever import SkillAugmentedRouter as _SAR

            SkillBank = _SB
            SkillRetriever = _SR2
            SkillAugmentedRouter = _SAR
    else:
        # Minimal progress logger that doesn't require MemRL
        from orchestration.repl_memory.progress_logger import (
            ProgressLogger as _PL,
            ProgressReader as _PR,
        )

        ProgressLogger = _PL
        ProgressReader = _PR

    # Tool registry
    if f.tools:
        from src.tool_registry import ToolRegistry as _TR

        ToolRegistry = _TR

    # Script registry (requires tools)
    if f.scripts:
        from src.script_registry import ScriptRegistry as _SR

        ScriptRegistry = _SR


def get_progress_logger_class() -> type | None:
    """Get the ProgressLogger class (after load_optional_imports called)."""
    return ProgressLogger


def get_tool_registry_class() -> type | None:
    """Get the ToolRegistry class (after load_optional_imports called)."""
    return ToolRegistry


def get_script_registry_class() -> type | None:
    """Get the ScriptRegistry class (after load_optional_imports called)."""
    return ScriptRegistry


def _do_score(state: "AppState", task_id: str, mode: str | None = None) -> None:
    """Synchronous scoring helper — runs in a worker thread."""
    try:
        if mode is not None:
            state.q_scorer._score_task(task_id, mode_context=mode)
        else:
            state.q_scorer._score_task(task_id)
        if state.progress_logger:
            state.progress_logger.flush()
    except Exception as e:
        logger.warning(f"Q-scoring failed for task {task_id}: {e}", exc_info=True)


import threading
from concurrent.futures import ThreadPoolExecutor

# Bounded thread pool for Q-scoring. Prevents unbounded thread accumulation
# when embedder (port 8090) is slow or unresponsive. Threads that can't
# acquire a slot are silently dropped (scoring is non-critical).
#
# The pool is lazily created and can be recreated after shutdown, solving
# the test-leaking bug where lifespan teardown permanently killed the pool.
_score_pool: ThreadPoolExecutor | None = None
_score_pool_lock = threading.Lock()


def _get_score_pool() -> ThreadPoolExecutor:
    """Get or create the Q-scorer thread pool (thread-safe, lazy)."""
    global _score_pool
    if _score_pool is None:
        with _score_pool_lock:
            if _score_pool is None:
                _score_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="q-scorer")
    return _score_pool


def score_completed_task(
    state: "AppState",
    task_id: str,
    *,
    force_role: str | None = None,
    real_mode: bool = False,
) -> None:
    """Fire-and-forget task scoring in the Q-scorer thread pool.

    Scoring involves embedder calls (port 8090) which are synchronous HTTP.
    Running in the event loop would block all concurrent requests.

    Args:
        state: Application state containing Q-scorer.
        task_id: The task ID to score.
        force_role: Optional forced role from request context.
        real_mode: Whether request was real_mode.
    """
    if should_skip_background_scoring(force_role=force_role, real_mode=real_mode):
        return
    scorer = getattr(state, "q_scorer", None)
    enabled = getattr(state, "q_scorer_enabled", False)
    if scorer is None or enabled is not True:
        return
    _get_score_pool().submit(_do_score, state, task_id)


def should_skip_background_scoring(
    *,
    force_role: str | None = None,
    real_mode: bool = False,
) -> bool:
    """Return True when background task scoring should be suppressed.

    Forced-role real-mode calls are benchmark/eval probes where rewards are
    injected externally (e.g., /chat/reward). Skipping background scoring
    reduces extra embedder/lock load that can skew latency measurements.
    """
    return bool(force_role) and bool(real_mode)


def score_completed_task_for_request(
    state: "AppState",
    task_id: str,
    *,
    force_role: str | None = None,
    real_mode: bool = False,
) -> None:
    """Request-aware wrapper around score_completed_task()."""
    score_completed_task(
        state,
        task_id,
        force_role=force_role,
        real_mode=real_mode,
    )


def score_completed_task_with_mode(
    state: "AppState",
    task_id: str,
    mode: str = "direct",
) -> None:
    """Fire-and-forget task scoring with execution mode context.

    Args:
        state: Application state containing Q-scorer.
        task_id: The task ID to score.
        mode: Execution mode used ("direct", "react", or "repl").
    """
    scorer = getattr(state, "q_scorer", None)
    enabled = getattr(state, "q_scorer_enabled", False)
    if scorer is None or enabled is not True:
        return
    _get_score_pool().submit(_do_score, state, task_id, mode)


def shutdown_scoring() -> None:
    """Shutdown the Q-scorer thread pool. Called during app lifespan shutdown.

    The pool will be lazily recreated on the next submit call, making this
    safe across multiple test lifespan cycles.
    """
    global _score_pool
    with _score_pool_lock:
        if _score_pool is not None:
            _score_pool.shutdown(wait=False)
            _score_pool = None


def store_external_reward(
    state: "AppState",
    task_description: str,
    action: str,
    reward: float,
    context: dict | None = None,
    embedding: list[float] | None = None,
) -> bool:
    """Store an externally-computed reward in MemRL.

    Public API for external reward injection (e.g., from the MemRL learning loop).
    Bypasses progress log reader and directly updates episodic memory.

    Args:
        state: Application state with Q-scorer and episodic store.
        task_description: Description of the task that was scored.
        action: The action taken (e.g., "frontdoor:direct").
        reward: Pre-computed reward value (-1.0 to 1.0).
        context: Optional additional context to store with the memory.
        embedding: Precomputed embedding for task_description (avoids re-embedding).

    Returns:
        True if reward was stored, False otherwise.
    """
    if not features().memrl:
        return False

    if not state.q_scorer or not state.episodic_store:
        return False

    try:
        result = state.q_scorer.score_external_result(
            task_description=task_description,
            action=action,
            reward=reward,
            context=context or {},
            embedding=embedding,
        )
        if state.progress_logger:
            state.progress_logger.flush()
        return result.get("memories_created", 0) > 0 or result.get("memories_updated", 0) > 0
    except Exception as e:
        logger.warning(f"External reward storage failed: {e}", exc_info=True)
        return False


async def background_cleanup(state: "AppState") -> None:
    """Background task for opportunistic cleanup when idle.

    Processes backlog of unscored tasks when server has no active requests.
    This catches any tasks that weren't scored immediately (e.g., from restarts).

    Args:
        state: Application state to check for idle and Q-scorer availability.
    """
    while True:
        try:
            # Check every 10 seconds
            await asyncio.sleep(10)

            if not _background_scoring_enabled():
                continue

            # Only run when idle and Q-scorer is available
            # Note: Don't call ensure_memrl_initialized() here - only init on real use
            if state.active_requests == 0 and state.q_scorer and state.q_scorer_enabled:
                # Score a small batch of pending tasks (run in thread to avoid
                # blocking the event loop — this does DB reads + embeddings)
                results = await asyncio.to_thread(state.q_scorer.score_pending_tasks)

                if results and not results.get("skipped"):
                    tasks_processed = results.get("tasks_processed", 0)
                    if tasks_processed > 0 and state.progress_logger:
                        state.progress_logger.flush()

        except asyncio.CancelledError:
            break
        except Exception as e:
            # Log background task errors but continue running
            logger.warning(f"Background cleanup error: {e}", exc_info=True)


def ensure_memrl_initialized(state: "AppState") -> bool:
    """Lazy-load MemRL components on first real use.

    This function respects the memrl feature flag - if disabled, returns False
    immediately without attempting to load any MemRL components.

    MemRL components are only needed for:
    - Real inference with Q-scoring/HybridRouter
    - Background Q-value updates

    Args:
        state: Application state to initialize.

    Returns:
        True if MemRL is initialized and available, False otherwise.
    """
    # Check feature flag first
    if not features().memrl:
        return False

    if state._memrl_initialized:
        return state.q_scorer is not None

    # Mark as initialized (even if it fails) to prevent repeated attempts
    state._memrl_initialized = True

    # Ensure optional imports are loaded
    if EpisodicStore is None or TaskEmbedder is None or QScorer is None:
        logger.warning("MemRL feature enabled but imports not available")
        return False

    try:
        state.episodic_store = EpisodicStore()
        embedder = TaskEmbedder()
        reader = ProgressReader()
        config = ScoringConfig(
            use_claude_judge=False,  # Basic mode only - no LLM required
            min_score_interval_seconds=30,  # Background cleanup interval
            batch_size=10,  # Process 10 tasks per cleanup round
        )
        state.q_scorer = QScorer(
            store=state.episodic_store,
            embedder=embedder,
            logger=state.progress_logger,
            reader=reader,
            config=config,
        )

        # Initialize HybridRouter for learned routing using centralized retrieval defaults.
        retrieval_config = RetrievalConfig()

        # Phase 3: Use GraphEnhancedRetriever when specialist routing is enabled
        if features().specialist_routing:
            try:
                from orchestration.repl_memory.failure_graph import FailureGraph
                from orchestration.repl_memory.hypothesis_graph import HypothesisGraph
                from orchestration.repl_memory.retriever import GraphEnhancedRetriever

                failure_graph = FailureGraph()
                hypothesis_graph = HypothesisGraph()
                retriever = GraphEnhancedRetriever(
                    store=state.episodic_store,
                    embedder=embedder,
                    failure_graph=failure_graph,
                    hypothesis_graph=hypothesis_graph,
                    config=retrieval_config,
                )
                state.failure_graph = failure_graph
                state.hypothesis_graph = hypothesis_graph
                logger.info("MemRL initialized with GraphEnhancedRetriever (specialist routing)")
            except Exception as e:
                logger.warning(
                    f"GraphEnhancedRetriever init failed, falling back to TwoPhaseRetriever: {e}",
                    exc_info=True,
                )
                retriever = TwoPhaseRetriever(
                    store=state.episodic_store,
                    embedder=embedder,
                    config=retrieval_config,
                )
        else:
            retriever = TwoPhaseRetriever(
                store=state.episodic_store,
                embedder=embedder,
                config=retrieval_config,
            )

        # Load routing hints from registry for rule-based fallback
        # Reuse existing registry if loaded at startup, otherwise load it
        if state.registry is None:
            from src.registry_loader import RegistryLoader

            state.registry = RegistryLoader(validate_paths=False)
        rule_router = RuleBasedRouter(routing_hints=state.registry.routing_hints)

        # Phase 3+: Initialize GraphRouter (GNN-based parallel routing signal)
        graph_router_predictor = None
        if features().graph_router:
            try:
                from pathlib import Path as _Path

                from orchestration.repl_memory.routing_graph import BipartiteRoutingGraph
                from orchestration.repl_memory.lightweight_gat import LightweightGAT
                from orchestration.repl_memory.graph_router_predictor import GraphRouterPredictor

                routing_graph = BipartiteRoutingGraph()
                gat = LightweightGAT()
                weights_path = _Path(
                    "/mnt/raid0/llm/claude/orchestration/repl_memory/graph_router_weights.npz"
                )
                if weights_path.exists():
                    gat.load(weights_path)
                graph_router_predictor = GraphRouterPredictor(
                    routing_graph, gat, embedder,
                )
                state.routing_graph = routing_graph
                logger.info("GraphRouter initialized (GNN cold-start routing signal)")
            except Exception as e:
                logger.warning("GraphRouter init failed: %s", e)
                graph_router_predictor = None

        hybrid_router = HybridRouter(
            retriever=retriever,
            rule_based_router=rule_router,
            graph_router=graph_router_predictor,
        )

        # SkillBank: wrap HybridRouter with skill retrieval if enabled
        if features().skillbank and SkillBank is not None and SkillRetriever is not None:
            try:
                from pathlib import Path

                skill_db_path = Path(
                    state.episodic_store.db_path
                ).parent / "skills.db" if hasattr(state.episodic_store, "db_path") else Path(
                    "/mnt/raid0/llm/claude/orchestration/repl_memory/sessions/skills.db"
                )
                faiss_dir = skill_db_path.parent

                state.skill_bank = SkillBank(
                    db_path=skill_db_path,
                    faiss_path=faiss_dir,
                    embedding_dim=1024,  # BGE-large dimension
                )
                state.skill_retriever = SkillRetriever(skill_bank=state.skill_bank)

                if SkillAugmentedRouter is not None:
                    state.hybrid_router = SkillAugmentedRouter(
                        hybrid_router=hybrid_router,
                        skill_retriever=state.skill_retriever,
                        embedder=embedder,
                    )
                    logger.info(
                        "SkillBank initialized: %d skills, wrapped with SkillAugmentedRouter",
                        state.skill_bank.count(),
                    )
                else:
                    state.hybrid_router = hybrid_router
            except Exception as e:
                logger.warning(
                    "SkillBank init failed, continuing without skills: %s", e,
                    exc_info=True,
                )
                state.skill_bank = None
                state.skill_retriever = None
                state.hybrid_router = hybrid_router
        else:
            state.hybrid_router = hybrid_router

        return True
    except Exception as e:
        # MemRL initialization failed - log and continue without it
        logger.warning(f"MemRL initialization failed, continuing without it: {e}", exc_info=True)
        state.q_scorer = None
        state.episodic_store = None
        state.hybrid_router = None
        return False
