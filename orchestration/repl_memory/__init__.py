"""
REPL Memory: MemRL-inspired episodic memory for orchestration learning.

This module implements non-parametric reinforcement learning on episodic memory,
enabling the orchestration system to improve task routing, escalation decisions,
and exploration strategies through runtime learning without modifying model weights.

Architecture:
    - EpisodicStore: SQLite + numpy embeddings for persistent memory
    - TaskEmbedder: Embeds TaskIR for semantic retrieval
    - TwoPhaseRetriever: Semantic filtering + Q-value ranking
    - ProgressLogger: Lightweight structured logging for all tiers
    - QScorer: Async agent that updates Q-values from progress logs

Based on: MemRL (arXiv:2601.03192) - Zhang et al. 2025
"""

from .episodic_store import EpisodicStore, MemoryEntry
from .embedder import TaskEmbedder
from .retriever import TwoPhaseRetriever
from .progress_logger import ProgressLogger, ProgressEntry
from .q_scorer import QScorer

__all__ = [
    "EpisodicStore",
    "MemoryEntry",
    "TaskEmbedder",
    "TwoPhaseRetriever",
    "ProgressLogger",
    "ProgressEntry",
    "QScorer",
]
