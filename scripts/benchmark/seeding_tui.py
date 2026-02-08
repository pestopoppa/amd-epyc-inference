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
from pathlib import Path

from rich.console import Console
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

    Buffers partial lines so that character-at-a-time SSE streaming
    doesn't produce one-char-per-line output.
    """

    def __init__(self, tap_path: str, poll_interval: float = 0.10,
                 max_lines: int = 200) -> None:
        self._path = tap_path
        self._poll = poll_interval
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._current_section: collections.deque[str] = collections.deque(maxlen=max_lines)
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

    def get_current_section(self, tail: int = 0) -> list[str]:
        with self._lock:
            items = list(self._current_section)
            if tail > 0:
                return items[-tail:]
            return items

    # -- internal --

    def _run(self) -> None:
        # Wait for the file to appear (the API may not have written yet)
        while not self._stop.is_set():
            if os.path.exists(self._path):
                break
            self._stop.wait(0.5)

        if self._stop.is_set():
            return

        with open(self._path) as fh:
            # Seek to end — we only care about new output
            fh.seek(0, 2)
            while not self._stop.is_set():
                # Read all available data at once (not line-by-line)
                chunk = fh.read(8192)
                if chunk:
                    self._process_chunk(chunk)
                else:
                    self._stop.wait(self._poll)

    def _process_chunk(self, chunk: str) -> None:
        """Process a chunk of text, appending to rolling buffer."""
        with self._lock:
            lines = chunk.split("\n")
            for i, fragment in enumerate(lines):
                if i == 0 and self._current_section:
                    # First fragment continues the last incomplete line
                    self._current_section[-1] += fragment
                elif fragment:  # skip empty strings from split
                    self._current_section.append(fragment)


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
        self._deque_handler.setLevel(logging.INFO)
        root.addHandler(self._deque_handler)
        root.setLevel(logging.INFO)

        # Silence noisy third-party loggers
        for name in ("filelock", "datasets", "huggingface_hub", "urllib3", "fsspec"):
            logging.getLogger(name).setLevel(logging.WARNING)

        # 3. Start tap tailer
        self._tailer.start()

        # 4. Start Rich Live with get_renderable callback.
        #    Rich Live's auto_refresh thread calls get_renderable() on each
        #    tick → _make_layout() → fresh layout with current deque/tap data.
        self._live = Live(
            console=self._console,
            screen=True,
            refresh_per_second=self._refresh,
            get_renderable=self._make_layout,
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
        # Calculate visible lines based on terminal size.
        # Panel border (2) + status bar (3) = 5 overhead, main ratio=9/10.
        try:
            panel_height = max(10, (self._console.height - 5) * 9 // 10 - 2)
            panel_width = max(20, self._console.width // 2 - 4)
        except Exception:
            panel_height = 25
            panel_width = 70
        # Truncate each line to panel width so 1 record = 1 display line (no wrap)
        raw = list(self._deque_handler.records)[-(panel_height):]
        log_lines = [line[:panel_width] for line in raw]
        log_text = Text("\n".join(log_lines) if log_lines else "(waiting for log output...)")
        layout["log"].update(Panel(
            log_text,
            title=f"Seeding Progress ({len(self._deque_handler.records)} records)",
            border_style="green",
        ))

        # Right panel: inference stream — filter out verbose PROMPT sections,
        # keep headers, RESPONSE tokens, and TIMINGS.
        raw_section = self._tailer.get_current_section()
        filtered: list[str] = []
        in_prompt = False
        for line in raw_section:
            if line.startswith("PROMPT:") or line == "PROMPT:":
                in_prompt = True
                filtered.append("PROMPT: [...]")
                continue
            if in_prompt:
                # End of prompt section: a line of dashes or a new section marker
                if line.startswith("-" * 20) or line.startswith("=" * 20):
                    in_prompt = False
                    # Don't add the dash separator — RESPONSE: follows
                else:
                    continue
            if line.startswith("RESPONSE:"):
                filtered.append("")  # visual separator
            filtered.append(line)

        display_lines = filtered[-(panel_height):]
        stream_text = Text("\n".join(display_lines) if display_lines else "(waiting for inference tap...)")
        layout["stream"].update(Panel(stream_text, title="Inference Stream", border_style="cyan"))

        # Bottom bar: status — extract current action from last log line
        p = self._progress
        elapsed = time.monotonic() - p.start_time
        mins, secs = divmod(int(elapsed), 60)
        status_str = (
            f"[{p.current_index}/{p.total_questions}] "
            f"{p.current_suite}/{p.current_qid}"
        )
        # Show current action from last meaningful log line (e.g. "→ SELF:direct (frontdoor:direct)...")
        last_action = ""
        for rec in reversed(list(self._deque_handler.records)):
            if "→ " in rec:
                # Extract e.g. "SELF:direct (frontdoor:direct)"
                try:
                    last_action = rec.split("→ ", 1)[1].split("...")[0].strip()
                except (IndexError, ValueError):
                    pass
                break
        if last_action:
            status_str += f" | {last_action}"
        elif p.current_action:
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
