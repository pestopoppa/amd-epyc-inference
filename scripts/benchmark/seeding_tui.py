"""Rich split-screen TUI for the seeding script.

Activated via ``--tui`` on ``seed_specialist_routing.py``.  Provides a
single-command split-screen experience:

- **Left panel**: Seeding progress log (captured from Python ``logging``)
- **Right panel**: Live inference stream (tailed from the inference tap file)
- **Bottom bar**: Current question index, suite, action, elapsed time

Requires ``rich>=13.7.0`` (already a project dependency).
"""

from __future__ import annotations

import atexit
import collections
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console, ConsoleOptions, RenderResult
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENTINEL_PATH = "/mnt/raid0/llm/tmp/.inference_tap_active"
_DEFAULT_TAP_PATH = "/mnt/raid0/llm/tmp/inference_tap.log"

# ---------------------------------------------------------------------------
# DequeHandler — capture log records for the left panel
# ---------------------------------------------------------------------------


class DequeHandler(logging.Handler):
    """Logging handler that stores formatted records in a bounded deque.

    Attach to the root logger while the TUI is active; original handlers
    are saved and restored on exit.
    """

    def __init__(self, maxlen: int = 500) -> None:
        super().__init__()
        self.records: collections.deque[str] = collections.deque(maxlen=maxlen)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.records.append(msg)
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# TapTailer — daemon thread that tails the inference tap file
# ---------------------------------------------------------------------------


class TapTailer:
    """Daemon thread that tails the inference tap file with polling.

    Tracks the *current section* (text between ``========`` markers) so
    that the right panel always shows the most recent inference call.
    """

    def __init__(self, tap_path: str, poll_interval: float = 0.05) -> None:
        self._path = tap_path
        self._poll = poll_interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._current_section: list[str] = []
        self._thread: threading.Thread | None = None

    # -- public API --

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="tap-tailer"
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_current_section(self) -> list[str]:
        with self._lock:
            return list(self._current_section)

    # -- internal --

    def _run(self) -> None:
        # Wait for the file to appear (the API may not have written yet)
        while not self._stop.is_set():
            if os.path.exists(self._path):
                break
            self._stop.wait(self._poll)

        if self._stop.is_set():
            return

        with open(self._path) as fh:
            # Seek to end — we only care about new output
            fh.seek(0, 2)
            while not self._stop.is_set():
                line = fh.readline()
                if line:
                    self._process_line(line.rstrip("\n"))
                else:
                    self._stop.wait(self._poll)

    def _process_line(self, line: str) -> None:
        with self._lock:
            if line.startswith("=" * 20):
                # New section boundary — reset
                self._current_section = [line]
            else:
                self._current_section.append(line)


# ---------------------------------------------------------------------------
# TUIProgress — mutable status for the bottom bar
# ---------------------------------------------------------------------------


@dataclass
class TUIProgress:
    total_questions: int = 0
    current_index: int = 0
    current_suite: str = ""
    current_qid: str = ""
    current_action: str = ""
    session_id: str = ""
    start_time: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# SeedingTUI — context manager orchestrating the Rich Live display
# ---------------------------------------------------------------------------


class SeedingTUI:
    """Context manager that runs the Rich TUI.

    Usage::

        with SeedingTUI(session_id="3way_20260208_1400") as tui:
            tui.update_progress(0, 30, "thinking", "q1")
            # ... seeding loop ...
    """

    def __init__(
        self,
        session_id: str = "",
        tap_path: str = _DEFAULT_TAP_PATH,
        refresh_per_second: int = 4,
    ) -> None:
        self._session_id = session_id
        self._tap_path = tap_path
        self._refresh = refresh_per_second

        self._console = Console()
        self._deque_handler = DequeHandler(maxlen=500)
        self._tailer = TapTailer(tap_path)
        self._progress = TUIProgress(session_id=session_id)
        self._live: Live | None = None

        # Saved state for handler restoration
        self._saved_handlers: list[logging.Handler] = []
        self._saved_level: int = logging.INFO

    # -- public API --

    def update_progress(
        self,
        idx: int,
        total: int,
        suite: str,
        qid: str,
        action: str = "",
    ) -> None:
        self._progress.current_index = idx
        self._progress.total_questions = total
        self._progress.current_suite = suite
        self._progress.current_qid = qid
        self._progress.current_action = action

    # -- context manager --

    def __enter__(self) -> "SeedingTUI":
        # 1. Write sentinel so the API discovers the tap path
        self._write_sentinel()
        atexit.register(self._cleanup)

        # 2. Swap log handlers
        root = logging.getLogger()
        self._saved_handlers = list(root.handlers)
        self._saved_level = root.level
        root.handlers.clear()
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        self._deque_handler.setFormatter(fmt)
        root.addHandler(self._deque_handler)
        root.setLevel(logging.DEBUG)

        # 3. Start tap tailer
        self._tailer.start()

        # 4. Start Rich Live (passes self as renderable — __rich_console__ is
        #    called on every refresh tick to rebuild the layout dynamically)
        self._live = Live(
            self,
            console=self._console,
            screen=True,
            refresh_per_second=self._refresh,
        )
        self._live.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore[no-untyped-def]
        # 1. Stop Live
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass

        # 2. Stop tailer
        self._tailer.stop()

        # 3. Restore log handlers
        root = logging.getLogger()
        root.handlers.clear()
        for h in self._saved_handlers:
            root.addHandler(h)
        root.setLevel(self._saved_level)

        # 4. Cleanup sentinel + tap file
        self._cleanup()

        return False

    # -- Rich renderable protocol --

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        yield self._make_layout()

    # -- layout building --

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="main", ratio=9),
            Layout(name="status", size=3),
        )
        layout["main"].split_row(
            Layout(name="log", ratio=1),
            Layout(name="stream", ratio=1),
        )

        # Left panel: seeding log
        log_lines = list(self._deque_handler.records)[-40:]
        log_text = Text("\n".join(log_lines) if log_lines else "(waiting for log output...)")
        layout["log"].update(Panel(log_text, title="Seeding Progress", border_style="green"))

        # Right panel: inference stream
        section = self._tailer.get_current_section()[-40:]
        stream_text = Text("\n".join(section) if section else "(waiting for inference tap...)")
        layout["stream"].update(Panel(stream_text, title="Inference Stream", border_style="cyan"))

        # Bottom bar: status
        p = self._progress
        elapsed = time.monotonic() - p.start_time
        mins, secs = divmod(int(elapsed), 60)
        status_str = (
            f"[{p.current_index}/{p.total_questions}] "
            f"{p.current_suite}/{p.current_qid}"
        )
        if p.current_action:
            status_str += f" | {p.current_action}"
        status_str += f" | {mins}m{secs:02d}s"
        if p.session_id:
            status_str += f" | {p.session_id}"

        layout["status"].update(
            Panel(Text(status_str, justify="center"), border_style="yellow")
        )

        return layout

    # -- sentinel management --

    def _write_sentinel(self) -> None:
        try:
            Path(_SENTINEL_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(_SENTINEL_PATH, "w") as f:
                f.write(self._tap_path)
        except OSError:
            pass

    def _cleanup(self) -> None:
        for path in (_SENTINEL_PATH, self._tap_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        # Also make sure the console isn't stuck in alt screen
        try:
            self._console.set_alt_screen(False)
        except Exception:
            pass
