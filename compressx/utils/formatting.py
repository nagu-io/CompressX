from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from compressx.context import CompressionContext

try:
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover - optional dependency
    box = None
    Console = None
    Panel = None
    Table = None


@dataclass(frozen=True, slots=True)
class CompletionSummary:
    """Structured data used to render the final compression summary."""

    title: str
    rows: tuple[tuple[str, str], ...]


def _format_size_gb(value: float | None) -> str:
    """Format a size in gigabytes using stable precision."""

    size_gb = float(value or 0.0)
    if size_gb >= 10:
        return f"{size_gb:.1f} GB"
    if size_gb >= 1:
        return f"{size_gb:.2f} GB"
    return f"{size_gb:.3f} GB"


def _format_ratio(value: float | None) -> str:
    """Format a compression ratio using the conventional x suffix."""

    ratio = float(value or 0.0)
    return f"{ratio:.1f}x"


def _format_accuracy_percent(value: float | None) -> str:
    """Format accuracy retention as a percentage."""

    accuracy = float(value or 0.0)
    return f"{accuracy:.1f}%"


def _format_duration(seconds: float | None) -> str:
    """Format elapsed time into a compact human-readable string."""

    total_seconds = int(round(float(seconds or 0.0)))
    if total_seconds < 60:
        return f"{total_seconds}s"

    minutes, remaining_seconds = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {remaining_seconds}s"

    hours, remaining_minutes = divmod(minutes, 60)
    return f"{hours}h {remaining_minutes}m {remaining_seconds}s"


def build_completion_summary(context: CompressionContext) -> CompletionSummary:
    """Build a normalized completion summary from pipeline context."""

    ratio = context.compression_ratio
    if not ratio and context.original_size_gb and context.current_size_gb:
        if isfinite(context.current_size_gb) and context.current_size_gb > 0:
            ratio = context.original_size_gb / context.current_size_gb

    return CompletionSummary(
        title="CompressX Compression Complete",
        rows=(
            ("Model:", context.config.model_id),
            ("Original:", _format_size_gb(context.original_size_gb)),
            ("Final:", _format_size_gb(context.current_size_gb)),
            ("Ratio:", _format_ratio(ratio)),
            ("Accuracy:", _format_accuracy_percent(context.accuracy_retention_percent)),
            ("Stop Reason:", context.target_stop_reason or "n/a"),
            ("Passes:", str(context.optimization_passes)),
            ("Time:", _format_duration(context.total_time_seconds)),
        ),
    )


def _render_rich_summary(summary: CompletionSummary) -> str | None:
    """Render the completion summary using Rich when available."""

    if Console is None or Panel is None or Table is None or box is None:
        return None

    table = Table.grid(padding=(0, 1))
    table.add_column(justify="left", no_wrap=True)
    table.add_column(justify="left")
    for label, value in summary.rows:
        table.add_row(label, value)

    console = Console(record=True, force_terminal=False, color_system=None, width=80)
    console.print(Panel.fit(table, title=summary.title, border_style="white", box=box.ROUNDED))
    return console.export_text(styles=False).rstrip()


def _render_ascii_summary(summary: CompletionSummary) -> str:
    """Render the completion summary using plain ASCII borders."""

    label_width = max(len(label) for label, _ in summary.rows)
    content_lines = [
        f"  {label.ljust(label_width)} {value}"
        for label, value in summary.rows
    ]
    inner_width = max(len(summary.title) + 4, *(len(line) + 2 for line in content_lines))

    lines = [
        "+" + "-" * inner_width + "+",
        "| " + summary.title.ljust(inner_width - 2) + " |",
        "+" + "-" * inner_width + "+",
    ]
    lines.extend(
        "| " + line.ljust(inner_width - 2) + " |"
        for line in content_lines
    )
    lines.append("+" + "-" * inner_width + "+")
    return "\n".join(lines)


def render_completion_summary(context: CompressionContext) -> str:
    """Render the final pipeline summary using Rich when available."""

    summary = build_completion_summary(context)
    rich_output = _render_rich_summary(summary)
    if rich_output is not None:
        return rich_output
    return _render_ascii_summary(summary)


def render_summary_table(context: CompressionContext) -> str:
    """Backward-compatible wrapper for callers using the old name."""

    return render_completion_summary(context)
