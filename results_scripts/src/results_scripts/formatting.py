from dataclasses import dataclass
from enum import StrEnum
import enum
from typing import Callable, Literal
import typing

from results_scripts.plots import format_number_compactly
import numpy as np
import pandas as pd

from results_scripts.constants import TABLE_ROUNDING_PER_METRIC


class TableCellType(StrEnum):
    mean = enum.auto()
    std = enum.auto()
    minmax = enum.auto()
    range_only = enum.auto()
    scene_std = enum.auto()
    median_and_mean_std = enum.auto()


class MetricsLayout(StrEnum):
    # One tabular section per metric, stacked vertically.
    vertical = enum.auto()
    # Metrics laid out horizontally as \multicolumn groups in a single tabular.
    horizontal = enum.auto()


@dataclass
class FormatOptions:
    cell_type: TableCellType = TableCellType.std
    table_size: Literal["default", "small"] = "small"
    resizebox: bool = True
    tabcolsep_fraction: float = 2.5
    # How to arrange metrics in multi-metric tables: "stacked" places one
    # tabular section per metric vertically; "side_by_side" lays the metrics out
    # horizontally as \multicolumn groups within a single tabular.
    metrics_layout: MetricsLayout = MetricsLayout.vertical
    table_env_override: Literal["table", "table*"] | None = None
    # When True, per-dataset tables are emitted as ``subtable`` blocks and
    # combined into a single floating ``table``/``table*`` environment instead of
    # one separate float per dataset. Requires the ``subcaption`` LaTeX package.
    combine_datasets_as_subtables: bool = True
    # Upper bound on the color intensity (in [0, 1]). 1.0 uses the full color map
    # range; lower values compress the gradient toward its neutral center so the
    # strongest cells stay paler. 0.0 disables coloring entirely (cells stay
    # white, only the text is rendered).
    color_intensity: float = 1.0
    # When True, all cell text is rendered black regardless of cell background
    # (instead of switching to light text on dark cells).
    force_black_text: bool = False  # Fraction (in [0, 1]) of the top of the value range to color in value-cmap
    # tables; cells below `vmax - color_best_fract * (vmax - vmin)` stay white.
    # 1.0 colors everything. The gradient is restretched over the colored slice
    # so those cells use the full colormap. Has no effect on diverging tables.
    color_best_fract: float = 1.0

    def get_latex_size(self) -> str:
        if self.table_size == "default":
            return ""
        elif self.table_size == "small":
            return r"\small "
        else:
            raise ValueError(f"Invalid table size: {self.table_size}")

    def get_tabcolsep_cmd_begin(self) -> str:
        if self.tabcolsep_fraction == 1.0:
            return ""
        return f"\\begingroup \\setlength{{\\tabcolsep}}{{{self.tabcolsep_fraction}\\tabcolsep}}"

    def get_tabcolsep_cmd_end(self) -> str:
        if self.tabcolsep_fraction == 1.0:
            return ""
        return "\\endgroup"

    def get_table_env(self) -> str:
        if self.table_env_override is not None:
            return self.table_env_override
        return "table"


class CellData(typing.NamedTuple):
    metric_id: str
    mean: float
    seed_stddev: float
    min: float
    max: float

    scene_stddev: float

    median_seed_stddev: float = -1

    @staticmethod
    def for_metric(df: pd.DataFrame, metric_id: str) -> "CellData":
        return CellData(
            metric_id=metric_id,
            mean=df[metric_id].map(lambda x: np.array(x).mean()).mean(skipna=False),
            seed_stddev=df[metric_id]
            .map(lambda x: np.array(x).var(ddof=1) if len(x) > 1 else 0.0)
            .map(np.sqrt)
            .mean(skipna=False),
            median_seed_stddev=df[metric_id]
            .map(lambda x: np.array(x).std(ddof=1) if len(x) > 1 else 0.0)
            .map(np.median)
            .mean(skipna=False),
            min=df[metric_id].map(lambda x: np.array(x).min()).mean(skipna=False),
            max=df[metric_id].map(lambda x: np.array(x).max()).mean(skipna=False),
            scene_stddev=df[metric_id]
            .map(lambda x: np.array(x).mean())
            .std(skipna=False),
        )


TableCellFormatter = Callable[[CellData], str]


def make_cell_formatter(
    cell_type: TableCellType,
    rounding_per_metric: dict[str, int] | None = None,
    always_show_sign: bool = False,
) -> TableCellFormatter:
    def get_rounding(cell_data: CellData) -> int:
        return (rounding_per_metric or TABLE_ROUNDING_PER_METRIC).get(
            cell_data.metric_id, 2
        )

    def mean_with_sign(x: CellData) -> str:
        if always_show_sign:
            return f"{x.mean:+.{get_rounding(x)}f}"
        else:
            return f"{x.mean:.{get_rounding(x)}f}"

    if cell_type == TableCellType.mean:
        return lambda x: f"${mean_with_sign(x)}$"
    elif cell_type == TableCellType.std:
        return lambda x: (
            f"${mean_with_sign(x)} \\pm {x.seed_stddev:.{get_rounding(x)}f}$"
        )
    elif cell_type == TableCellType.minmax:
        return lambda x: (
            f"${mean_with_sign(x)} \\in [{format_number_compactly(x.min)},{format_number_compactly(x.max)}]$"
        )
    elif cell_type == TableCellType.range_only:
        return lambda x: f"[{x.min:.{get_rounding(x)}f}, {x.max:.{get_rounding(x)}f}]"
    elif cell_type == TableCellType.scene_std:
        return lambda x: (
            f"${mean_with_sign(x)} \\pm {x.scene_stddev:.{get_rounding(x)}f}$"
        )
    else:
        raise ValueError(f"Invalid format table cell type: {cell_type}")
