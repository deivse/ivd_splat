from dataclasses import dataclass
import json
from pathlib import Path
import re

from matplotlib import pyplot as plt

from results_scripts.constants import STRATEGY_NAMES


def name_to_path(value: str, allow_subdirs: bool) -> str:
    value = value.replace("scannet++", "scannetpp")
    for strategy, short_name in STRATEGY_NAMES.items():
        value = value.replace(strategy, short_name.lower().replace(" ", "_"))
    pattern = r"[^/a-zA-Z0-9._-]+" if allow_subdirs else r"[^a-zA-Z0-9._-]+"
    slug = re.sub(pattern, "_", value.strip())
    return slug.strip("_") or "figure"


def load_json(path: Path) -> dict[str, int]:
    with path.open("r") as handle:
        return json.load(handle)


def write_file(path: Path | str, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(content)


@dataclass
class OutputDirHelper:
    output_dir: Path

    def get_graph_path(self, section_subdir: str, figure_name: str) -> Path:
        return (
            self.output_dir
            / "graphs"
            / name_to_path(section_subdir, allow_subdirs=True)
            / f"{name_to_path(figure_name, allow_subdirs=False)}.svg"
        )

    def get_table_path(self, table_name: str) -> Path:
        return (
            self.output_dir
            / "tables"
            / f"{name_to_path(table_name, allow_subdirs=False)}.tex"
        )

    def get_stats_path(self, name: str, suffix: str = "csv") -> Path:
        return (
            self.output_dir
            / "statistics"
            / f"{name_to_path(name, allow_subdirs=False)}.{suffix}"
        )


def fraction_name(fraction: str | float) -> str:
    return f"{float(fraction) * 100:.0f}% $G_\\mathit{{max}}$"


def print_friedman_summary(
    records: list[tuple[str, str, str, float | None]],
    *,
    alpha: float = 0.05,
    title: str = "Friedman test p-values (* = row passed, omnibus null rejected):",
) -> None:
    """Print per-dataset tables of Friedman outcomes to stdout.

    Each record is ``(group, row, metric, p_value)``: ``group`` is the dataset
    label (use ``""`` when unused), ``row`` the tested row (strategy), ``metric``
    the metric-column label and ``p_value`` the Friedman p-value (or ``None`` when
    the test could not be run). One table is printed per dataset with strategies
    as rows and metrics as columns; a cell shows the p-value with a trailing ``*``
    when the null was rejected (``n/a`` when the test could not run).
    """
    if not records:
        return

    # Group records by dataset, preserving first-seen order.
    groups: list[str] = []
    per_group: dict[str, list[tuple[str, str, float | None]]] = {}
    for group, row, metric, p_value in records:
        if group not in per_group:
            per_group[group] = []
            groups.append(group)
        per_group[group].append((row, metric, p_value))

    def render_table(group_records: list[tuple[str, str, float | None]]) -> str:
        metric_cols: list[str] = []
        row_keys: list[str] = []
        cells: dict[str, dict[str, str]] = {}
        for row, metric, p_value in group_records:
            if metric not in metric_cols:
                metric_cols.append(metric)
            if row not in cells:
                cells[row] = {}
                row_keys.append(row)
            if p_value is None:
                cells[row][metric] = "n/a"
            else:
                cells[row][metric] = f"{p_value:.2g}" + ("*" if p_value < alpha else "")

        headers = ["Strategy"] + metric_cols
        table_rows = [
            [row] + [cells[row].get(metric, "") for metric in metric_cols]
            for row in row_keys
        ]
        widths = [
            max(len(headers[i]), *(len(r[i]) for r in table_rows))
            for i in range(len(headers))
        ]

        def fmt_row(cols: list[str]) -> str:
            return "  ".join(col.ljust(widths[i]) for i, col in enumerate(cols))

        lines = [fmt_row(headers), "  ".join("-" * width for width in widths)]
        lines += [fmt_row(row) for row in table_rows]
        return "\n".join(lines)

    print()
    print(title)
    for group in groups:
        print()
        if group:
            print(f"[{group}]")
        print(render_table(per_group[group]))
    print()



def gmax_fraction_label(fraction: str | float) -> str:
    """LaTeX label for a multiple of the max Gaussian count, e.g.
    ``$0.75\\mathcal{G}_\\mathit{max}$``.

    Used as the canonical notation for "fraction of $\\mathcal{G}_\\mathit{max}$"
    across all tables so the style stays consistent.
    """
    value = float(fraction)
    text = f"{value:g}"
    if "." not in text:
        text += ".0"
    return rf"${text}\text  {{G}}_\mathit{{m}}$"


def save_figure_svg(
    fig: plt.Figure,
    output: Path,
    **kwargs,
):
    output.parent.mkdir(parents=True, exist_ok=True)

    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(output, format="svg", **kwargs)
    print(f"Saved: {output}")
