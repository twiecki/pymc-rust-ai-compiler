"""Autoresearcher-style analysis and plotting for optimization runs.

Reads results.tsv (or a CompilationResult.optimization_log) and generates
progress plots inspired by Karpathy's autoresearch and Luca Fiaschi's
Bayesian autoresearcher.

Usage:
    # From a CompilationResult:
    from transpailer.analysis import plot_optimization_progress
    fig = plot_optimization_progress(result)

    # From a results.tsv file:
    fig = plot_optimization_progress("path/to/results.tsv")
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transpailer.compiler import CompilationResult


@dataclass
class _BenchmarkRecord:
    """Parsed benchmark event for plotting."""

    turn: int
    timestamp_s: float
    us_per_eval: float
    status: str  # KEEP | DISCARD | OK
    description: str
    code_hash: str


def _load_from_tsv(path: str | Path) -> list[_BenchmarkRecord]:
    """Load benchmark events from a results.tsv file."""
    records = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row["event_type"] != "benchmark" or not row["us_per_eval"]:
                continue
            records.append(
                _BenchmarkRecord(
                    turn=int(row["turn"]),
                    timestamp_s=float(row["timestamp_s"]),
                    us_per_eval=float(row["us_per_eval"]),
                    status=row["status"],
                    description=row["description"],
                    code_hash=row["code_hash"],
                )
            )
    return records


def _load_from_result(result: CompilationResult) -> list[_BenchmarkRecord]:
    """Extract benchmark events from a CompilationResult."""
    records = []
    for ev in result.optimization_log:
        if ev.event_type != "benchmark" or ev.us_per_eval is None:
            continue
        records.append(
            _BenchmarkRecord(
                turn=ev.turn,
                timestamp_s=ev.timestamp,
                us_per_eval=ev.us_per_eval,
                status=ev.status,
                description=ev.description,
                code_hash=ev.code_hash,
            )
        )
    return records


def load_benchmark_records(
    source: str | Path | CompilationResult,
) -> list[_BenchmarkRecord]:
    """Load benchmark records from a TSV path or CompilationResult."""
    if isinstance(source, (str, Path)):
        return _load_from_tsv(source)
    return _load_from_result(source)


def plot_optimization_progress(
    source: str | Path | CompilationResult,
    title: str = "Optimization Progress",
    figsize: tuple[float, float] = (12, 5),
):
    """Generate an autoresearcher-style optimization progress plot.

    Plots all benchmark measurements as scatter points, color-coded by
    status (KEEP=green, DISCARD=gray). A step line tracks the running
    best (frontier). Kept improvements are annotated.

    Args:
        source: Path to results.tsv or a CompilationResult object.
        title: Plot title.
        figsize: Figure size (width, height).

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    records = load_benchmark_records(source)
    if not records:
        raise ValueError("No benchmark events found in source")

    fig, ax = plt.subplots(figsize=figsize)

    # Separate by status
    keep_idx, keep_us, keep_turns = [], [], []
    discard_idx, discard_us, discard_turns = [], [], []

    for i, rec in enumerate(records):
        if rec.status == "KEEP":
            keep_idx.append(i)
            keep_us.append(rec.us_per_eval)
            keep_turns.append(rec.turn)
        else:
            discard_idx.append(i)
            discard_us.append(rec.us_per_eval)
            discard_turns.append(rec.turn)

    # Plot discarded as faint gray
    if discard_idx:
        ax.scatter(
            discard_idx,
            discard_us,
            c="#cccccc",
            s=40,
            zorder=2,
            label="Discarded",
            edgecolors="#aaaaaa",
            linewidths=0.5,
        )

    # Plot kept as green
    if keep_idx:
        ax.scatter(
            keep_idx,
            keep_us,
            c="#2ecc71",
            s=80,
            zorder=3,
            label="Kept",
            edgecolors="#27ae60",
            linewidths=1,
        )

    # Running minimum (frontier) as step line
    running_min = []
    current_min = float("inf")
    for rec in records:
        current_min = min(current_min, rec.us_per_eval)
        running_min.append(current_min)

    ax.step(
        range(len(records)),
        running_min,
        where="post",
        color="#2c3e50",
        linewidth=2,
        zorder=4,
        label="Best so far",
    )

    # Annotate kept improvements
    kept_records = [records[i] for i in keep_idx]
    for i, us, rec in zip(keep_idx, keep_us, kept_records):
        ax.annotate(
            rec.code_hash if len(records) <= 30 else "",
            (i, us),
            textcoords="offset points",
            xytext=(5, 8),
            fontsize=7,
            color="#27ae60",
        )

    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("us/eval (lower is better)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)

    # Summary stats in text box
    if records:
        baseline = records[0].us_per_eval
        best = min(r.us_per_eval for r in records)
        improvement_pct = (1 - best / baseline) * 100 if baseline > 0 else 0
        n_keep = len(keep_idx)
        n_total = len(records)

        stats_text = (
            f"Baseline: {baseline:.2f} us/eval\n"
            f"Best: {best:.2f} us/eval\n"
            f"Improvement: {improvement_pct:.1f}%\n"
            f"Kept: {n_keep}/{n_total} ({100 * n_keep / n_total:.0f}%)"
        )
        ax.text(
            0.02,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
        )

    fig.tight_layout()
    return fig


def plot_waterfall(
    source: str | Path | CompilationResult,
    title: str = "Optimization Waterfall",
    figsize: tuple[float, float] = (10, 5),
):
    """Waterfall chart showing the contribution of each kept optimization.

    Each bar shows the delta improvement from the previous best.

    Args:
        source: Path to results.tsv or a CompilationResult object.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    records = load_benchmark_records(source)
    kept = [r for r in records if r.status == "KEEP"]

    if len(kept) < 2:
        raise ValueError("Need at least 2 kept experiments for a waterfall")

    fig, ax = plt.subplots(figsize=figsize)

    labels = []
    deltas = []
    for i in range(1, len(kept)):
        delta = kept[i - 1].us_per_eval - kept[i].us_per_eval
        deltas.append(delta)
        labels.append(kept[i].code_hash or f"Step {i}")

    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]

    bars = ax.bar(
        range(len(deltas)), deltas, color=colors, edgecolor="#2c3e50", linewidth=0.5
    )

    # Value labels on bars
    for bar, delta in zip(bars, deltas):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            f"{delta:+.2f}",
            ha="center",
            va="bottom" if y >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Delta us/eval (improvement)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="#2c3e50", linewidth=0.8)
    ax.grid(True, alpha=0.3, axis="y")

    total = sum(deltas)
    ax.text(
        0.98,
        0.98,
        f"Total improvement: {total:.2f} us/eval",
        transform=ax.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout()
    return fig


def plot_timeline(
    source: str | Path | CompilationResult,
    title: str = "Optimization Timeline",
    figsize: tuple[float, float] = (12, 4),
):
    """Timeline plot showing all events over wall-clock time.

    Useful for understanding how time is spent across builds,
    benchmarks, and validations.

    Args:
        source: Path to results.tsv or a CompilationResult object.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    import matplotlib.pyplot as plt

    # Load all events (not just benchmarks)
    if isinstance(source, (str, Path)):
        events = []
        with open(source) as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                events.append(row)
    else:
        events = [
            {
                "timestamp_s": str(ev.timestamp),
                "event_type": ev.event_type,
                "status": ev.status,
                "us_per_eval": str(ev.us_per_eval)
                if ev.us_per_eval is not None
                else "",
            }
            for ev in source.optimization_log
        ]

    if not events:
        raise ValueError("No events found")

    fig, ax = plt.subplots(figsize=figsize)

    type_colors = {
        "benchmark": "#3498db",
        "build": "#e67e22",
        "validation": "#9b59b6",
        "write_code": "#1abc9c",
    }
    status_markers = {
        "KEEP": "^",
        "DISCARD": "v",
        "PASS": "o",
        "FAIL": "x",
        "CRASH": "X",
        "OK": "s",
    }

    for ev in events:
        t = float(ev["timestamp_s"])
        etype = ev["event_type"]
        status = ev["status"]
        color = type_colors.get(etype, "#95a5a6")
        marker = status_markers.get(status, "o")

        # Y-axis: us_per_eval if benchmark, otherwise fixed row
        if etype == "benchmark" and ev["us_per_eval"]:
            y = float(ev["us_per_eval"])
        else:
            y_map = {"write_code": -1, "build": -2, "validation": -3}
            y = y_map.get(etype, -4)

        ax.scatter(t, y, c=color, marker=marker, s=50, zorder=3, alpha=0.8)

    # Legend entries
    for etype, color in type_colors.items():
        ax.scatter([], [], c=color, label=etype, s=40)

    ax.set_xlabel("Wall-clock time (seconds)", fontsize=11)
    ax.set_ylabel("us/eval (benchmarks) / event type", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def print_summary(source: str | Path | CompilationResult) -> str:
    """Print a text summary of the optimization run.

    Returns:
        Summary string.
    """
    records = load_benchmark_records(source)
    if not records:
        return "No benchmark events found."

    kept = [r for r in records if r.status == "KEEP"]
    discarded = [r for r in records if r.status == "DISCARD"]

    baseline = records[0].us_per_eval
    best = min(r.us_per_eval for r in records)
    improvement_pct = (1 - best / baseline) * 100 if baseline > 0 else 0

    lines = [
        "Optimization Summary",
        f"{'=' * 40}",
        f"Total experiments: {len(records)}",
        f"  Kept:     {len(kept)}",
        f"  Discarded: {len(discarded)}",
        f"  Keep rate: {100 * len(kept) / len(records):.0f}%",
        "",
        f"Baseline:    {baseline:.3f} us/eval",
        f"Best:        {best:.3f} us/eval",
        f"Improvement: {improvement_pct:.1f}%",
        f"Speedup:     {baseline / best:.2f}x" if best > 0 else "",
        "",
        "Kept experiments (chronological):",
    ]

    prev_us = baseline
    for rec in kept:
        delta = prev_us - rec.us_per_eval
        lines.append(
            f"  [{rec.code_hash}] {rec.us_per_eval:.3f} us/eval "
            f"(delta: {delta:+.3f}, {rec.description})"
        )
        prev_us = rec.us_per_eval

    summary = "\n".join(lines)
    print(summary)
    return summary
