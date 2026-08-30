"""Rich-based progress display for concurrent experiment runners."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.live import Live
from rich.table import Table
from rich.panel import Panel


console = Console()


def create_progress() -> Progress:
    """Create a progress bar with standard columns."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    )


@contextmanager
def experiment_progress() -> Generator[Progress, None, None]:
    """Context manager for experiment progress tracking."""
    progress = create_progress()
    with progress:
        yield progress


class MultiExperimentProgress:
    """Track progress across multiple concurrent experiments."""

    def __init__(self):
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[cyan]{task.fields[status]}"),
            TimeElapsedColumn(),
            console=console,
            expand=False,
            refresh_per_second=2,
        )
        self.live = Live(self.progress, console=console, refresh_per_second=2)
        self.tasks: dict[str, int] = {}

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, *args):
        self.live.stop()

    def add_experiment(self, name: str, total: int) -> int:
        """Add an experiment to track. Returns task ID."""
        task_id = self.progress.add_task(name, total=total, status="starting...")
        self.tasks[name] = task_id
        return task_id

    def update(self, name: str, advance: int = 1, status: str | None = None):
        """Update progress for an experiment."""
        if name in self.tasks:
            kwargs = {"advance": advance}
            if status:
                kwargs["status"] = status
            self.progress.update(self.tasks[name], **kwargs)

    def set_status(self, name: str, status: str):
        """Set status message for an experiment."""
        if name in self.tasks:
            self.progress.update(self.tasks[name], status=status)

    def complete(self, name: str, status: str = "done"):
        """Mark an experiment as complete."""
        if name in self.tasks:
            task = self.progress.tasks[self.tasks[name]]
            self.progress.update(self.tasks[name], completed=task.total, status=status)


class RequestProgress:
    """Track API request progress within an experiment."""

    def __init__(self, progress: Progress, task_id: int, parent_name: str):
        self.progress = progress
        self.task_id = task_id
        self.parent_name = parent_name
        self.completed = 0
        self.total = 0

    def set_total(self, total: int):
        """Set total number of requests."""
        self.total = total
        self.progress.update(self.task_id, total=total)

    def update(self, n: int = 1):
        """Called when requests complete."""
        self.completed += n
        self.progress.update(self.task_id, advance=n)

    def __call__(self):
        """Callback for on_complete."""
        self.update(1)


def print_summary(results: dict[str, dict | Exception], debug: bool = False):
    """Print a summary table of experiment results."""
    table = Table(title="Experiment Results")
    table.add_column("Experiment", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Configs", justify="right")
    table.add_column("Successes", justify="right", style="green")
    table.add_column("Failures", justify="right", style="red")
    table.add_column("Cache Hits", justify="right", style="cyan")

    for label, result in results.items():
        if isinstance(result, Exception):
            table.add_row(label, "[red]FAILED", "-", "-", str(result)[:50], "-")
        else:
            status = "[green]OK" if result.get("failures", 0) == 0 else "[yellow]PARTIAL"
            table.add_row(
                label,
                status,
                str(result.get("total_runs", 0)),
                str(result.get("successes", 0)),
                str(result.get("failures", 0)),
                str(result.get("cache_hits", 0)),
            )

    console.print()
    console.print(table)

    # Collect failure stats across all results
    all_failure_cats: dict[str, int] = {}
    all_failure_examples: dict[str, list[str]] = {}
    for result in results.values():
        if isinstance(result, dict):
            if result.get("failure_categories"):
                for cat, count in result["failure_categories"].items():
                    all_failure_cats[cat] = all_failure_cats.get(cat, 0) + count
            if result.get("failure_examples"):
                for cat, examples in result["failure_examples"].items():
                    if cat not in all_failure_examples:
                        all_failure_examples[cat] = []
                    for ex in examples:
                        if len(all_failure_examples[cat]) < 5:
                            all_failure_examples[cat].append(ex)

    if all_failure_cats:
        console.print()
        console.print("[bold]Failure Breakdown:")
        sorted_cats = sorted(all_failure_cats.items(), key=lambda x: -x[1])
        for cat, count in sorted_cats:
            console.print(f"  {cat}: [red]{count}[/red]")

    if debug and all_failure_examples:
        console.print()
        console.print("[bold]Example Errors (up to 5 per category):")
        for cat in sorted(all_failure_examples.keys()):
            console.print(f"\n  [yellow]{cat}[/yellow]:")
            for i, example in enumerate(all_failure_examples[cat], 1):
                # Truncate long messages
                truncated = example[:200] + "..." if len(example) > 200 else example
                console.print(f"    {i}. {truncated}")
