"""Pipeline monitor: anomaly detection, diagnostics, and Claude-in-the-loop debugging."""

from src.pipeline_monitor.anomaly import compute_anomaly_signals, anomaly_score
from src.pipeline_monitor.diagnostic import build_diagnostic, append_diagnostic

__all__ = [
    "compute_anomaly_signals",
    "anomaly_score",
    "build_diagnostic",
    "append_diagnostic",
]
