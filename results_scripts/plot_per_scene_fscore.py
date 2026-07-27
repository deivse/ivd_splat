#!/usr/bin/env python3
"""Produce per-scene line charts of reconstruction accuracy metrics from a JSON.

For each metric (F-score, precision, recall) generates a pair of charts:
  1. Per-scene metric averaged over all init methods (columns), one line per strategy.
  2. Per-scene metric averaged over all strategies (excluding "At Init"), one line per init method.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from results_scripts.constants import STRATEGY_NAMES

AT_INIT = "At Init"
METRICS = ["fscore", "precision", "recall"]

# Pretty display names for the metrics (values are shown as percentages).
METRIC_DISPLAY_NAMES = {
    "fscore": "F-Score",
    "precision": "Precision",
    "recall": "Recall",
}

# Pretty display names for the init-method columns as they appear in the JSON.
INIT_METHOD_DISPLAY_NAMES = {
    "sfm=default": "SfM",
    "edgs=default": "EDGS",
    "monodepth=default": "Monodepth",
    "da3=floater_removal=True": "DA3",
    "da3=output_gaussians=True_max_num_images=150": "DA3 (G.S.)",
    "laser_scan=default": "Laser Scan",
    "laser_scan=no_sparse": "Laser Scan (No Sparse)",
}


def pretty_strategy(strategy: str) -> str:
    """Map a JSON strategy value (e.g. ``strategy=MCMCStrategy``) to a display name."""
    key = strategy.split("=", 1)[1] if strategy.startswith("strategy=") else strategy
    return STRATEGY_NAMES.get(key, key)


def pretty_init_method(column: str) -> str:
    return INIT_METHOD_DISPLAY_NAMES.get(column, column)


def load_data(path: Path):
    with open(path) as f:
        return json.load(f)


def short_scene(scene: str) -> str:
    return scene.split("/")[-1]


def scene_labels(scenes):
    """First 4 characters of each scene name; errors out if they are not unique."""
    labels = [short_scene(s)[:4] for s in scenes]
    if len(set(labels)) != len(labels):
        seen, dupes = set(), set()
        for label in labels:
            (dupes if label in seen else seen).add(label)
        raise ValueError(
            f"First-4-letter scene labels are not unique: {sorted(dupes)}. "
            "Increase the truncation length."
        )
    return labels


def collect(data, metric):
    """Return nested dict: scene -> strategy -> column -> metric value."""
    table = defaultdict(lambda: defaultdict(dict))
    for scene, runs in data["resolved_runs"].items():
        for run in runs:
            if not run.get("exists") or run.get("metrics") is None:
                continue
            value = run["metrics"].get(metric)
            if value is None:
                continue
            table[scene][run["strategy"]][run["column"]] = value
    return table


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _style_axes(ax, scenes, metric, avg_over, title):
    x = list(range(len(scenes)))
    ax.set_xticks(x)
    ax.set_xticklabels(scene_labels(scenes), rotation=0, ha="center")
    ax.set_xlabel("Scene", labelpad=10)
    ax.set_ylabel(f"{METRIC_DISPLAY_NAMES[metric]} (%)  —  mean over {avg_over}")
    ax.set_title(title, fontweight="bold", pad=14)
    ax.grid(True, axis="y", alpha=0.4)
    ax.grid(True, axis="x", alpha=0.15)
    ax.margins(x=0.02)


def _draw_lines(ax, x, series):
    """series: list of (label, y_values). Returns (label, color) pairs for the legend."""
    handles = []
    for label, ys in series:
        ys_pct = [None if v is None else v * 100 for v in ys]
        (line,) = ax.plot(
            x,
            ys_pct,
            marker="o",
            markersize=4,
            linewidth=2,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=label,
        )
        handles.append((label, line.get_color()))
    return handles


def _finalize(fig, ax, handles, ncol, output_path):
    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in handles
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=ncol,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderaxespad=0.0,
        handlelength=1.2,
        handleheight=1.2,
    )
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path}")


def plot_avg_over_columns(table, scenes, metric, output_path):
    """One line per strategy; each point = mean metric over all columns for that scene."""
    strategies = sorted({s for scene in scenes for s in table[scene]})
    x = list(range(len(scenes)))

    series = []
    for strategy in strategies:
        ys = [mean(list(table[scene].get(strategy, {}).values())) for scene in scenes]
        series.append((pretty_strategy(strategy), ys))

    fig, ax = plt.subplots(figsize=(max(11, len(scenes) * 0.75), 6.5))
    handles = _draw_lines(ax, x, series)
    _style_axes(
        ax,
        scenes,
        metric,
        "init methods",
        f"Per-Scene {METRIC_DISPLAY_NAMES[metric]} Averaged Over All Init Methods",
    )
    _finalize(fig, ax, handles, ncol=min(len(series), 4), output_path=output_path)


def plot_avg_over_strategies(table, scenes, metric, output_path):
    """One line per init method; each point = mean metric over all strategies (excl. At Init)."""
    columns = sorted(
        {
            c
            for scene in scenes
            for strat, cols in table[scene].items()
            if strat != AT_INIT
            for c in cols
        }
    )
    x = list(range(len(scenes)))

    series = []
    for column in columns:
        ys = []
        for scene in scenes:
            vals = [
                cols[column]
                for strat, cols in table[scene].items()
                if strat != AT_INIT and column in cols
            ]
            ys.append(mean(vals))
        series.append((pretty_init_method(column), ys))

    fig, ax = plt.subplots(figsize=(max(11, len(scenes) * 0.75), 6.5))
    handles = _draw_lines(ax, x, series)
    _style_axes(
        ax,
        scenes,
        metric,
        "strategies (excl. At Init)",
        f"Per-Scene {METRIC_DISPLAY_NAMES[metric]} Averaged Over All Strategies",
    )
    _finalize(fig, ax, handles, ncol=min(len(series), 4), output_path=output_path)


def plot_at_init(table, scenes, metric, output_path):
    """One line per init method; each point = the 'At Init' metric for that scene."""
    columns = sorted({c for scene in scenes for c in table[scene].get(AT_INIT, {})})
    x = list(range(len(scenes)))

    series = []
    for column in columns:
        ys = [table[scene].get(AT_INIT, {}).get(column) for scene in scenes]
        series.append((pretty_init_method(column), ys))

    fig, ax = plt.subplots(figsize=(max(11, len(scenes) * 0.75), 6.5))
    handles = _draw_lines(ax, x, series)
    _style_axes(
        ax,
        scenes,
        metric,
        "",
        f"Per-Scene {METRIC_DISPLAY_NAMES[metric]} At Init Per Init Method",
    )
    ax.set_ylabel(f"{METRIC_DISPLAY_NAMES[metric]} (%)  —  at init")
    _finalize(fig, ax, handles, ncol=min(len(series), 4), output_path=output_path)


def configure_style():
    for style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid"):
        if style in plt.style.available:
            plt.style.use(style)
            break
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "axes.titlesize": 17,
            "axes.labelsize": 14,
            "axes.prop_cycle": plt.cycler(color=plt.get_cmap("tab10").colors),
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "font.size": 13,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "json_file",
        nargs="?",
        default="final_recon_accuracy_merged.json",
        help="Path to the recon accuracy JSON file.",
    )
    parser.add_argument(
        "--outdir",
        default=".",
        help="Directory to write the output PNG charts into.",
    )
    args = parser.parse_args()

    configure_style()

    data = load_data(Path(args.json_file))
    scenes = data["scenes"]

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.json_file).stem

    for metric in METRICS:
        table = collect(data, metric)
        plot_avg_over_columns(
            table, scenes, metric, outdir / f"{stem}_{metric}_by_strategy.png"
        )
        plot_avg_over_strategies(
            table, scenes, metric, outdir / f"{stem}_{metric}_by_init.png"
        )
        plot_at_init(table, scenes, metric, outdir / f"{stem}_{metric}_at_init.png")


if __name__ == "__main__":
    main()
