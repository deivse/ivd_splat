"""Numeric processing for results tables.

This module is the single home for the statistical/aggregation grunt work used by
the section functions: reducing per-scene run data into scalar summaries,
aggregating across strategies, and combining across datasets. It deliberately
knows nothing about LaTeX/plot formatting (see ``tables``/``plots``) nor about
how run data is queried (see ``data_access``).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from results_scripts.formatting import CellData


def series_mean_frame_mean(df: pd.DataFrame | pd.Series) -> pd.Series:
    """Mean over eval-iter lists in each cell, then mean over scenes."""
    return df.map(lambda x: np.array(x).mean()).mean()


def cell_data_across_strategies(
    metric: str, strategy_dfs: Sequence[pd.DataFrame]
) -> CellData:
    """Aggregate a metric across strategies into a single ``CellData``.

    Each strategy contributes its scene-averaged mean; ``mean`` is the mean of
    those per-strategy values and ``stddev``/``scene_stddev``/``min``/``max``
    describe the spread across strategies.
    """
    strategy_means = np.array(
        [float(series_mean_frame_mean(df[metric])) for df in strategy_dfs],
        dtype=float,
    )
    strategy_means = strategy_means[~np.isnan(strategy_means)]
    if strategy_means.size == 0:
        nan = float("nan")
        return CellData(metric, nan, nan, nan, nan, nan, 0.0)
    spread = float(strategy_means.std())
    return CellData(
        metric_id=metric,
        mean=float(strategy_means.mean()),
        stddev=spread,
        min=float(strategy_means.min()),
        max=float(strategy_means.max()),
        scene_stddev=spread,
        mean_measurement_count=float(strategy_means.size),
    )
