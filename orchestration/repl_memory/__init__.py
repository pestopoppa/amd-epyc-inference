"""
REPL Memory: MemRL-inspired episodic memory for orchestration learning.

This module implements non-parametric reinforcement learning on episodic memory,
enabling the orchestration system to improve task routing, escalation decisions,
and exploration strategies through runtime learning without modifying model weights.

Architecture:
    - EpisodicStore: SQLite + FAISS embeddings for persistent memory
    - GraphEnhancedStore: Wrapper with failure/hypothesis graph integration
    - FailureGraph: Kuzu-backed graph for failure patterns and mitigations
    - HypothesisGraph: Kuzu-backed graph for action-task confidence tracking
    - TaskEmbedder: Embeds TaskIR for semantic retrieval
    - TwoPhaseRetriever: Semantic filtering + Q-value ranking
    - GraphEnhancedRetriever: Extended retriever with graph-based scoring
    - ProgressLogger: Lightweight structured logging for all tiers
    - QScorer: Async agent that updates Q-values from progress logs

Based on: MemRL (arXiv:2601.03192) - Zhang et al. 2025
Enhanced with: Graphiti-inspired failure anti-memory and hypothesis tracking
"""

from __future__ import annotations

from .episodic_store import EpisodicStore, MemoryEntry, GraphEnhancedStore, extract_symptoms
from .embedder import TaskEmbedder
from .retriever import TwoPhaseRetriever, GraphEnhancedRetriever, RetrievalConfig, RetrievalResult
from .progress_logger import ProgressLogger, ProgressEntry
from .q_scorer import QScorer
from .progress_logger import ProgressReader

# Replay evaluation harness (optional — all deps are local)
try:
    from .replay import (
        Trajectory,
        TrajectoryExtractor,
        ReplayEngine,
        ReplayMetrics,
        DesignCandidate,
        DesignArchive,
        WarmStartProtocol,
        MetaAgentWorkflow,
    )
    _REPLAY_AVAILABLE = True
except ImportError:
    _REPLAY_AVAILABLE = False

# Graph modules are optional (require kuzu)
try:
    from .failure_graph import FailureGraph, FailureMode, Symptom, Mitigation
    from .hypothesis_graph import HypothesisGraph, Hypothesis, Evidence
    _GRAPHS_AVAILABLE = True
except ImportError:
    _GRAPHS_AVAILABLE = False
    FailureGraph = None
    HypothesisGraph = None

__all__ = [
    # Core memory
    "EpisodicStore",
    "MemoryEntry",
    "GraphEnhancedStore",
    "extract_symptoms",
    # Embedding
    "TaskEmbedder",
    # Retrieval
    "TwoPhaseRetriever",
    "GraphEnhancedRetriever",
    "RetrievalConfig",
    "RetrievalResult",
    # Logging
    "ProgressLogger",
    "ProgressEntry",
    # Q-learning
    "QScorer",
    "ProgressReader",
    # Replay harness
    "Trajectory",
    "TrajectoryExtractor",
    "ReplayEngine",
    "ReplayMetrics",
    "DesignCandidate",
    "DesignArchive",
    "WarmStartProtocol",
    "MetaAgentWorkflow",
    # Graphs (optional)
    "FailureGraph",
    "FailureMode",
    "Symptom",
    "Mitigation",
    "HypothesisGraph",
    "Hypothesis",
    "Evidence",
]
