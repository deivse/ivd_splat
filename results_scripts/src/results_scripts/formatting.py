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
    all_runwise = enum.auto()
    scene_std = enum.auto()


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
    tabcolsep_fraction: float = 0.8
    # How to arrange metrics in multi-metric tables: "stacked" places one
    # tabular section per metric vertically; "side_by_side" lays the metrics out
    # horizontally as \multicolumn groups within a single tabular.
    metrics_layout: MetricsLayout = MetricsLayout.vertical
    table_env_override: Literal["table", "table*"] | None = None
    # When True, per-dataset tables are emitted as ``subtable`` blocks and
    # combined into a single floating ``table``/``table*`` environment instead of
    # one separate float per dataset. Requires the ``subcaption`` LaTeX package.
    combine_datasets_as_subtables: bool = True

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
    stddev: float
    min: float
    max: float

    scene_stddev: float

    mean_measurement_count: float

    @staticmethod
    def for_metric(df: pd.DataFrame, metric_id: str) -> "CellData":
        return CellData(
            metric_id=metric_id,
            mean=df[metric_id].map(lambda x: np.array(x).mean()).mean(),
            stddev=df[metric_id].map(lambda x: np.array(x).std()).mean(),
            min=df[metric_id].map(lambda x: np.array(x).min()).mean(),
            max=df[metric_id].map(lambda x: np.array(x).max()).mean(),
            scene_stddev=df[metric_id].map(lambda x: np.array(x).mean()).std(),
            mean_measurement_count=df[metric_id].map(lambda x: len(x)).mean(),
        )


TableCellFormatter = Callable[[CellData], str]


def make_cell_formatter(
    cell_type: TableCellType,
    rounding_per_metric: dict[str, int] | None = None,
) -> TableCellFormatter:
    def get_rounding(cell_data: CellData) -> int:
        return (rounding_per_metric or TABLE_ROUNDING_PER_METRIC).get(
            cell_data.metric_id, 2
        )

    if cell_type == TableCellType.mean:
        return lambda x: f"${x.mean:.{get_rounding(x)}f}$"
    elif cell_type == TableCellType.std:
        return lambda x: (
            f"${x.mean:.{get_rounding(x)}f} \\pm {x.stddev:.{get_rounding(x)}f}$"
        )
    elif cell_type == TableCellType.minmax:
        return (
            lambda x: f"${x.mean:.{get_rounding(x)}f} \\in [{format_number_compactly(x.min)},{format_number_compactly(x.max)}]$"
        )
    elif cell_type == TableCellType.range_only:
        return lambda x: f"[{x.min:.{get_rounding(x)}f}, {x.max:.{get_rounding(x)}f}]"
    elif cell_type == TableCellType.all_runwise:
        return (
            lambda x: f"${x.mean:.{get_rounding(x)}f} \\pm {x.stddev:.{get_rounding(x)}f} ({x.min:.{get_rounding(x)}f}, {x.max:.{get_rounding(x)}f})$"
        )
    elif cell_type == TableCellType.scene_std:
        return (
            lambda x: f"${x.mean:.{get_rounding(x)}f} \\pm {x.scene_stddev:.{get_rounding(x)}f}$"
        )
    else:
        raise ValueError(f"Invalid format table cell type: {cell_type}")
