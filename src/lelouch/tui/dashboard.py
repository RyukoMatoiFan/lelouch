"""Compact Rich Live dashboard for harvest monitoring."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, List

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from ..models.harvest import ClipResult

if TYPE_CHECKING:
    from ..agents.harvester import VideoHarvester
    from ..models.harvest import HarvestRequest, HarvestResult

console = Console()

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class _StatusBar:
    """Progress bar + metrics + current action spinner."""

    def __init__(self) -> None:
        self.iteration = 0
        self.max_iterations = 20
        self.collected = 0.0
        self.target = 0.0
        self.status = "Initializing"
        self.candidates_found = 0
        self.clips_accepted = 0
        self.clips_rejected = 0
        self.errors = 0
        self._start = time.monotonic()
        self._tick = 0

    def __rich__(self) -> Panel:
        self._tick += 1
        pct = min(100, (self.collected / self.target * 100) if self.target else 0)
        elapsed = time.monotonic() - self._start
        spinner = _SPINNER[self._tick % len(_SPINNER)]

        grid = Table.grid(expand=True)
        grid.add_column(ratio=1)
        grid.add_column(justify="right")

        # Row 1: progress bar + collected/target
        right = Text()
        right.append(f"{self.collected:.1f}s", style="bold green")
        right.append(f"/{self.target:.0f}s ", style="dim")
        right.append(f"{pct:.0f}%  ", style="bold")
        right.append(self._fmt_elapsed(elapsed), style="dim")
        grid.add_row(ProgressBar(total=100, completed=pct, width=None), right)

        # Row 2: metrics
        m = Text()
        m.append(f"  iter {self.iteration}/{self.max_iterations}", style="bold white")
        m.append(" \u2502 ", style="dim")
        m.append(f"{self.candidates_found}", style="bold cyan")
        m.append(" found ", style="dim")
        m.append(f"{self.clips_accepted}", style="bold green")
        m.append(" kept ", style="dim")
        m.append(f"{self.clips_rejected}", style="bold red")
        m.append(" skip", style="dim")
        if self.errors:
            m.append(f" {self.errors}", style="bold red")
            m.append(" err", style="dim")
        grid.add_row(m, "")

        # Row 3: spinner + current action
        s = Text()
        s.append(f"  {spinner} ", style="bold magenta")
        s.append(self.status, style="white")
        grid.add_row(s, "")

        return Panel(
            grid,
            title="[bold cyan]lelouch[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )

    @staticmethod
    def _fmt_elapsed(s: float) -> str:
        m, sec = divmod(int(s), 60)
        return f"{m}m{sec:02d}s" if m else f"{sec}s"


class _ActivityLog:
    """Scrolling timestamped event log."""

    def __init__(self) -> None:
        self.entries: List[Text] = []

    def add(self, message: str, style: str = "") -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        entry = Text()
        entry.append(f"  {ts}  ", style="dim")
        entry.append(message, style=style)
        self.entries.append(entry)
        if len(self.entries) > 200:
            self.entries = self.entries[-200:]

    def __rich__(self) -> Panel:
        text = Text()
        for entry in self.entries[-20:]:
            text.append_text(entry)
            text.append("\n")
        return Panel(text, title="[bold]activity[/bold]", border_style="blue", padding=(0, 0))


class _ClipsPanel:
    """Compact accepted clips list."""

    def __init__(self) -> None:
        self.clips: List[ClipResult] = []

    def add(self, clip: ClipResult) -> None:
        self.clips.append(clip)

    def __rich__(self) -> Panel:
        total = sum(c.duration for c in self.clips)

        if not self.clips:
            return Panel(
                Text("  (waiting for clips)", style="dim"),
                title="[dim]clips[/dim]",
                border_style="dim",
                padding=(0, 0),
            )

        table = Table(
            expand=True, padding=(0, 1), show_header=False,
            show_edge=False, box=None,
        )
        table.add_column(width=3, style="dim")
        table.add_column(ratio=1, no_wrap=True, overflow="ellipsis")
        table.add_column(justify="right", width=7)
        table.add_column(justify="right", width=5, style="green")
        table.add_column(width=14, style="cyan", no_wrap=True)

        show = self.clips[-6:]
        start_idx = max(0, len(self.clips) - 6) + 1
        for i, clip in enumerate(show, start=start_idx):
            name = clip.clip_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
            table.add_row(
                f"{i}",
                clip.source_title[:40],
                f"{clip.duration:.1f}s",
                f"{clip.confidence:.0%}",
                name,
            )

        title = f"[bold]clips[/bold] ({len(self.clips)}) [bold green]{total:.1f}s[/bold green]"
        return Panel(table, title=title, border_style="yellow", padding=(0, 0))


class _Dashboard:
    """Three-panel layout: status, activity log, clips."""

    def __init__(self) -> None:
        self.status = _StatusBar()
        self.activity = _ActivityLog()
        self.clips = _ClipsPanel()

    def __rich__(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self.status, name="status", size=5),
            Layout(self.activity, name="activity", ratio=1),
            Layout(self.clips, name="clips", size=9),
        )
        return layout


# ── TUI runner ────────────────────────────────────────────────────────


class HarvestTUI:
    """Runs the harvester with a compact live-updating dashboard."""

    def __init__(self) -> None:
        self.dashboard = _Dashboard()
        self._search_counts: list[int] = []
        self._last_analysis = None

    def handle_event(self, event_type: str, *args) -> None:
        """Route harvest events to dashboard panels."""
        d = self.dashboard

        match event_type:
            case "start":
                req = args[0]
                d.status.target = req.target_duration
                d.status.max_iterations = req.max_iterations
                d.status.status = "Starting harvest"
                d.activity.add(
                    f"Target: {req.target_duration:.0f}s \u2014 {req.description}",
                    style="bold",
                )

            case "iteration_start":
                n = args[0]
                d.status.iteration = n
                d.status.status = "Generating queries"
                self._search_counts = []
                d.activity.add(f"\u2500\u2500 iteration {n} \u2500\u2500", style="bold cyan")

            case "generating_queries":
                d.status.status = "Generating queries"

            case "queries_generated":
                queries = args[0]
                compact = " \u2022 ".join(q[:35] for q in queries[:4])
                if len(queries) > 4:
                    compact += f" +{len(queries) - 4}"
                d.activity.add(compact, style="dim")
                d.status.status = "Searching"

            case "searching":
                d.status.status = f"Searching: {args[0][:50]}"

            case "search_done":
                self._search_counts.append(args[1])

            case "candidates_found":
                n = args[0]
                counts = "+".join(str(c) for c in self._search_counts)
                style = "green" if n > 0 else "dim"
                d.activity.add(f"search {counts} \u2192 {n} new", style=style)
                d.status.candidates_found += n

            case "evaluating":
                c = args[0]
                d.status.status = f"Evaluating: {c.title[:45]}"
                d.activity.add(f"\u25b8 {c.title[:55]}")

            case "checking_info":
                d.status.status = f"Checking: {args[0].title[:45]}"

            case "info_ready":
                c, dur = args[0], args[1]
                m, s = divmod(int(dur), 60)
                d.status.status = f"{m}m{s:02d}s \u2014 {c.title[:40]}"

            case "downloading":
                d.status.status = f"Downloading: {args[0].title[:40]}"

            case "download_progress":
                mb, pct, speed, _eta = args
                parts = []
                if pct >= 0:
                    parts.append(f"{pct:.0f}%")
                parts.append(f"{mb:.1f}MB")
                if speed > 0:
                    parts.append(f"{speed / 1048576:.1f}MB/s")
                dl_info = " \u2502 ".join(parts)
                d.status.status = f"Downloading {dl_info}"

            case "downloaded":
                d.status.status = "Extracting keyframes"

            case "extracting_frames":
                d.status.status = "Building keyframe grid"

            case "keyframes_extracted":
                pass

            case "analyzing":
                c = args[0]
                d.status.status = f"Vision LLM: {c.title[:40]}"

            case "analysis_complete":
                self._last_analysis = args[0]

            case "clipping":
                start, end, _dur = args
                d.status.status = f"Clipping {start:.1f}s \u2013 {end:.1f}s"

            case "clip_accepted":
                clip = args[0]
                d.clips.add(clip)
                d.status.collected += clip.duration
                d.status.clips_accepted += 1
                reason = ""
                if self._last_analysis:
                    reason = f" \u2014 {self._last_analysis.reason[:45]}"
                    self._last_analysis = None
                d.activity.add(
                    f"  \u2713 +{clip.duration:.1f}s {clip.confidence:.0%}{reason}",
                    style="bold green",
                )

            case "clip_rejected":
                reason = args[1] if len(args) > 1 else "unknown"
                d.status.clips_rejected += 1
                self._last_analysis = None
                d.activity.add(f"  \u2717 {reason[:65]}", style="dim red")

            case "error":
                msg = args[1] if len(args) > 1 else "unknown"
                d.status.errors += 1
                self._last_analysis = None
                d.activity.add(f"  ! {msg[:65]}", style="bold red")

            case "iteration_end":
                mem = args[0]
                d.status.status = "Iteration complete"
                d.activity.add(
                    f"{mem.total_duration:.1f}s collected \u2502 "
                    f"{len(mem.accepted_clips)} clips \u2502 "
                    f"{len(mem.rejected)} rejected",
                    style="cyan",
                )

            case "log":
                d.activity.add(str(args[0]) if args else "", style="dim")

            case "complete":
                d.status.status = "Complete"
                d.activity.add("Harvest complete!", style="bold green")

    def run(
        self, harvester: VideoHarvester, request: HarvestRequest
    ) -> HarvestResult:
        """Run harvester with the live dashboard."""
        harvester.emit = self.handle_event
        with Live(
            self.dashboard,
            console=console,
            refresh_per_second=4,
            screen=False,
        ):
            result = harvester.harvest(request)

        console.print()
        console.print(self._summary(result))
        return result

    @staticmethod
    def _summary(result: HarvestResult) -> Panel:
        """Final summary panel after harvest completes."""
        table = Table.grid(padding=(0, 2))
        table.add_column(style="bold")
        table.add_column()

        table.add_row("Duration", f"{result.total_duration:.1f}s")
        table.add_row("Clips", str(len(result.clips)))
        table.add_row("Iterations", str(result.iterations_used))
        table.add_row("Evaluated", str(result.candidates_evaluated))
        table.add_row("Rejected", str(result.candidates_rejected))
        if result.request:
            table.add_row("Output", result.request.output_dir)

        return Panel(
            table,
            title="[bold green]harvest complete[/bold green]",
            border_style="green",
        )
