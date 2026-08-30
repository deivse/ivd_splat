"""Declarative run selection for results tables.

Section functions describe *what* columns they want (init method, strategy config,
init size, ...) as :class:`ColumnSpec` objects and hand them to
:func:`collect_columns`, which performs the actual ``RunsInfo`` queries and
assembles the nested ``row -> column -> per-scene DataFrame`` structure consumed
by the table/plot builders. This keeps the "which runs" concern out of the
processing (see ``statistics``) and formatting (see ``tables``) layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Literal, Mapping, Sequence

import pandas as pd

from results_scripts.base import RunsInfo, drop_scenes_not_present_in_all
from results_scripts.constants import (
    DEFAULT_TABLE_METRICS,
    LASER_DATASETS,
    STRATEGY_NAMES,
)

# row label -> column label -> per-scene metrics DataFrame.
ColumnData = dict[str, dict[str, pd.DataFrame]]

# Maps a strategy id to extra query params (e.g. its default-config overrides).
StrategyOverrides = Callable[[str], Mapping[str, Any]]


@dataclass(frozen=True)
class ColumnSpec:
    """Description of a single table column's run selection.

    ``params`` are merged with the collector's ``common_args`` and the injected
    ``strategy`` to form the query. ``gt_only`` columns (e.g. laser scan) are only
    available for laser datasets and are skipped elsewhere.
    """

    label: str
    params: Mapping[str, Any] = field(default_factory=dict)
    metrics: Sequence[str] = tuple(DEFAULT_TABLE_METRICS)
    gt_only: bool = False


def collect_columns(
    runs: RunsInfo,
    strategies: Sequence[str],
    columns: Sequence[ColumnSpec],
    *,
    common_args: Mapping[str, Any] | None = None,
    strategy_overrides: StrategyOverrides | None = None,
    dataset: str | None = None,
    row_label: Callable[[str], str] = STRATEGY_NAMES.__getitem__,
    on_error: Literal["raise", "warn"] = "raise",
    skip_empty: bool = False,
    into: ColumnData | None = None,
) -> ColumnData:
    """Query per-scene metrics for every ``(strategy, column)`` pair.

    The query for a cell is ``{**common_args, "strategy": strategy,
    **column.params, **strategy_overrides(strategy)}``. Results are stored under
    ``row_label(strategy) -> column.label``.

    - ``gt_only`` columns are skipped when ``dataset`` is not a laser dataset.
    - ``on_error="warn"`` logs and skips a column whose query raises instead of
      propagating the exception.
    - ``skip_empty`` drops columns whose query returns an empty frame.
    - ``into`` lets several calls accumulate into the same structure (columns with
      differing ``common_args``/strategy subsets are merged by row/column label).
    """
    common = dict(common_args or {})
    data: ColumnData = into if into is not None else {}
    for strategy in strategies:
        overrides = dict(strategy_overrides(strategy)) if strategy_overrides else {}
        label = row_label(strategy)
        for column in columns:
            if column.gt_only and dataset is not None and dataset not in LASER_DATASETS:
                continue
            params = {
                **common,
                "strategy": strategy,
                **dict(column.params),
                **overrides,
            }
            try:
                result = runs.get_per_scene_metrics_for_params(
                    params, metrics=list(column.metrics)
                )
            except Exception as exc:
                if on_error == "warn":
                    logging.warning(
                        "Skipping column %r for %s: %s", column.label, label, exc
                    )
                    continue
                raise
            if skip_empty and result.empty:
                continue
            data.setdefault(label, {})[column.label] = result
    return data


def iter_frames(data: ColumnData) -> list[pd.DataFrame]:
    """Flatten a :data:`ColumnData` structure into a list of its DataFrames."""
    return [df for columns in data.values() for df in columns.values()]


def drop_uncommon_scenes(data: ColumnData, debug_out: bool = False) -> None:
    """Drop scenes absent from any column, in place, across the whole structure."""
    drop_scenes_not_present_in_all(*iter_frames(data), debug_out=debug_out)


def concat_columns_into(dest: ColumnData, source: ColumnData) -> None:
    """Row/column-wise append ``source`` frames onto ``dest`` (e.g. across datasets)."""
    for row_label, columns in source.items():
        for column_label, df in columns.items():
            if column_label in dest.setdefault(row_label, {}):
                dest[row_label][column_label] = pd.concat(
                    [dest[row_label][column_label], df], axis=0, ignore_index=True
                )
            else:
                dest[row_label][column_label] = df
