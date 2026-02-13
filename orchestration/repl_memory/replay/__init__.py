"""Replay evaluation harness for offline memory design evolution.

Provides trajectory extraction, replay engine, metrics, design archive,
warm-start protocol, and Claude meta-agent integration.

Based on: ALMA (Xiong et al., Feb 2026) — meta-learned memory designs
outperform hand-crafted ones.
"""

from .trajectory import Trajectory, TrajectoryExtractor
from .engine import ReplayEngine, ReplayStepResult, NullEmbedder
from .metrics import ReplayMetrics
from .candidates import DesignCandidate, DesignArchive
from .warm_start import WarmStartProtocol, RoleConfig, WarmStartStats
from .meta_agent import MetaAgentWorkflow

__all__ = [
    "Trajectory",
    "TrajectoryExtractor",
    "ReplayEngine",
    "ReplayStepResult",
    "NullEmbedder",
    "ReplayMetrics",
    "DesignCandidate",
    "DesignArchive",
    "WarmStartProtocol",
    "RoleConfig",
    "WarmStartStats",
    "MetaAgentWorkflow",
]
