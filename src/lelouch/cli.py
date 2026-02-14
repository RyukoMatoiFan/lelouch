"""CLI entry point for Lelouch video harvester."""

from __future__ import annotations

import sys
import warnings
from datetime import datetime

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import Config, OUTPUT_DIR, MAX_ITERATIONS
from .llm_client import LLMClient

# Suppress noisy ResourceWarnings from yt-dlp / urllib3
warnings.filterwarnings("ignore", category=ResourceWarning)

console = Console()

SETTINGS_FILE = "settings.yaml"

# Check for --simple flag before anything else
_SIMPLE_MODE = "--simple" in sys.argv


def _mask_key(key: str | None) -> str:
    if not key:
        return "not set"
    if len(key) > 12:
        return f"{key[:4]}...{key[-4:]}"
    return "***"


def _print_settings(cfg: Config) -> None:
    """Print current settings summary."""
    table = Table(show_header=False, show_edge=False, box=None, padding=(0, 2), expand=True)
    table.add_column("Setting", min_width=20)
    table.add_column("Value", style="green")

    table.add_row("  Text model", cfg.llm.model)
    table.add_row("  Text API key", _mask_key(cfg.llm.api_key))
    table.add_row("  Vision model", cfg.llm.vision_model)
    table.add_row("  Vision API key", _mask_key(cfg.llm.vision_api_key))
    table.add_row("  Agentic mode", "ON" if cfg.llm.agentic_mode else "OFF")

    console.print(
        Panel(
            table,
            title="[bold cyan]Current Settings[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()


def _simple_event_handler(event_type: str, *args) -> None:
    """Print harvest events as plain console lines (no Live dashboard)."""
    ts = datetime.now().strftime("%H:%M:%S")

    match event_type:
        case "start":
            req = args[0]
            console.print(
                f"\n[bold cyan][{ts}][/] Starting harvest: "
                f"{req.target_duration:.0f}s of '{req.description}'"
            )
        case "iteration_start":
            console.print(f"\n[bold cyan][{ts}][/] --- Iteration {args[0]} ---")
        case "queries_generated":
            for q in args[0]:
                console.print(f"  [dim]>[/dim] {q}")
        case "search_done":
            count = args[1]
            style = "green" if count > 0 else "red"
            console.print(f"[{style}][{ts}]   -> {count} videos[/{style}]")
        case "candidates_found":
            console.print(f"[{ts}] {args[0]} new candidates after filtering")
        case "evaluating":
            console.print(f"\n[{ts}] Evaluating: {args[0].title[:60]}")
        case "info_ready":
            dur = args[1]
            m, s = divmod(int(dur), 60)
            console.print(f"[dim][{ts}]   Duration: {m}m{s:02d}s[/dim]")
        case "downloading":
            console.print(f"[dim][{ts}]   Downloading...[/dim]")
        case "downloaded":
            console.print(f"[dim][{ts}]   Downloaded[/dim]")
        case "analyzing":
            console.print(f"[dim][{ts}]   Analyzing with vision LLM...[/dim]")
        case "clip_accepted":
            clip = args[0]
            console.print(
                f"[bold green][{ts}]   ACCEPTED +{clip.duration:.1f}s "
                f"({clip.confidence:.0%})[/bold green]"
            )
        case "clip_rejected":
            reason = args[1] if len(args) > 1 else "unknown"
            console.print(f"[dim red][{ts}]   SKIP: {reason[:70]}[/dim red]")
        case "error":
            msg = args[1] if len(args) > 1 else "unknown error"
            console.print(f"[bold red][{ts}]   ERROR: {msg[:70]}[/bold red]")
        case "iteration_end":
            mem = args[0]
            console.print(
                f"[cyan][{ts}] Collected: {mem.total_duration:.1f}s | "
                f"Accepted: {len(mem.accepted_clips)} | "
                f"Rejected: {len(mem.rejected)}[/cyan]"
            )
        case "log":
            console.print(f"[dim][{ts}] {args[0] if args else ''}[/dim]")
        case "complete":
            console.print(f"\n[bold green][{ts}] Harvest complete![/bold green]")


def _run_harvest(cfg: Config) -> None:
    """Run harvest with dashboard (default) or simple console output (--simple)."""
    try:
        llm = LLMClient(cfg)

        request_text = click.prompt("What footage do you need?")

        from .agents.request_parser import RequestParser

        parser = RequestParser(llm)
        console.print("[dim]Parsing your request...[/dim]")
        harvest_request = parser.parse(request_text)
        harvest_request.output_dir = OUTPUT_DIR
        harvest_request.max_iterations = MAX_ITERATIONS

        console.print(
            Panel(
                f"[bold]Duration:[/] {harvest_request.target_duration:.0f}s\n"
                f"[bold]Criteria:[/] {', '.join(harvest_request.content_criteria)}\n"
                f"[bold]Output:[/]   {harvest_request.output_dir}",
                title="Parsed Request",
                border_style="cyan",
            )
        )

        from .agents.harvester import VideoHarvester

        if _SIMPLE_MODE:
            harvester = VideoHarvester(cfg, llm, event_callback=_simple_event_handler)
            result = harvester.harvest(harvest_request)
        else:
            from .tui.dashboard import HarvestTUI

            harvester = VideoHarvester(cfg, llm)
            tui = HarvestTUI()
            result = tui.run(harvester, harvest_request)

        if result.clips:
            console.print(f"\n[bold green]Done![/] {len(result.clips)} clips, "
                          f"{result.total_duration:.1f}s total in {result.request.output_dir}")
        else:
            console.print("[yellow]No matching clips found.[/yellow]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")


def main():
    """Entry point — interactive menu."""
    try:
        import questionary
        from questionary import Style

        MENU_STYLE = Style([
            ("qmark", "fg:ansibrightcyan bold"),
            ("question", "fg:ansiwhite bold"),
            ("answer", "fg:ansicyan bold"),
            ("pointer", "fg:ansibrightmagenta bold"),
            ("highlighted", "fg:ansibrightcyan bold"),
            ("selected", "fg:ansicyan"),
        ])
    except ImportError:
        console.print("[red]Requires 'questionary'. Install with: pip install questionary[/red]")
        sys.exit(1)

    cfg = Config.load_default()

    while True:
        console.clear()
        _print_settings(cfg)

        choice = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("  Harvest Videos", value="harvest"),
                questionary.Choice("  Settings", value="settings"),
                questionary.Choice("  Exit", value="exit"),
            ],
            style=MENU_STYLE,
        ).ask()

        if choice is None or choice == "exit":
            break
        elif choice == "harvest":
            _run_harvest(cfg)
            click.prompt("\nPress Enter to continue", default="", show_default=False)
        elif choice == "settings":
            from .tui.settings import SettingsScreen

            screen = SettingsScreen(cfg)
            cfg = screen.run()
            # Auto-save after editing
            cfg.save_to_file(SETTINGS_FILE)
            console.print(f"[green]Settings saved to {SETTINGS_FILE}[/green]")


if __name__ == "__main__":
    main()
