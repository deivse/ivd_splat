"""Numeric processing for results tables.

This module is the single home for the statistical/aggregation grunt work used by
the section functions: reducing per-scene run data into scalar summaries,
aggregating across strategies, and combining across datasets. It deliberately
knows nothing about LaTeX/plot formatting (see ``tables``/``plots``) nor about
how run data is queried (see ``data_access``).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from results_scripts.formatting import CellData


def series_mean_frame_mean(df: pd.DataFrame | pd.Series) -> pd.Series:
    """Mean over eval-iter lists in each cell, then mean over scenes."""
    return df.map(lambda x: np.array(x).mean()).mean()


def per_scene_metric_difference(
    method: pd.Series,
    baseline: pd.Series,
    label: str = "",
) -> pd.Series:
    """Per-scene, seed-paired difference ``method - baseline`` for one metric.

    Both inputs are per-scene Series whose cells are arrays of per-seed values
    (ordered consistently by ``eval_iter``; see
    ``RunsInfo.get_per_scene_metrics_for_params``). For each shared scene the seed
    arrays are subtracted elementwise, so the result is a per-scene Series of
    per-seed delta arrays that preserves the seed-level spread. Scenes present in
    ``method`` but missing from ``baseline`` (or with mismatched seed counts) are
    warned about via ``label`` and aligned by truncation to the common seeds.
    """
    common = method.index.intersection(baseline.index)
    if len(common) < len(method.index):
        missing = list(set(method.index) - set(common))
        logging.warning(
            "per_scene_metric_difference[%s]: %d scene(s) missing in baseline: %s",
            label,
            len(missing),
            missing,
        )

    deltas: dict[str, np.ndarray] = {}
    for scene in method.index:
        if scene not in common:
            deltas[scene] = np.array([np.nan])
            continue
        a = np.atleast_1d(np.asarray(method.loc[scene], dtype=float))
        b = np.atleast_1d(np.asarray(baseline.loc[scene], dtype=float))
        n = min(a.size, b.size)
        if a.size != b.size:
            logging.warning(
                "per_scene_metric_difference[%s]: scene %s seed-count mismatch "
                "(%d vs %d); truncating to %d.",
                label,
                scene,
                a.size,
                b.size,
                n,
            )
        deltas[scene] = a[:n] - b[:n]

    return pd.Series(deltas, name=method.name).reindex(method.index)


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
        seed_stddev=spread,
        scene_stddev=spread,
        min=float(strategy_means.min()),
        max=float(strategy_means.max()),
    )


# ---------------------------------------------------------------------------
# Comparison of multiple methods over multiple data sets, following
# Demšar (2006), "Statistical Comparisons of Classifiers over Multiple Data
# Sets", JMLR 7:1-30. Here each densification strategy is compared over the
# scenes of a dataset: the "classifiers" are the initialization methods and the
# "data sets" are the scenes. A Friedman test with the Iman-Davenport correction
# tests the omnibus null (all initializations equivalent); if it is rejected,
# Holm's step-down procedure with a fixed control (SfM) identifies which
# initializations significantly improve over the control.
# ---------------------------------------------------------------------------


@dataclass
class FriedmanResult:
    """Outcome of a Friedman test with the Iman-Davenport F correction."""

    n_data_sets: int  # N, the number of blocks (scenes)
    k_methods: int  # k, the number of methods (initializations)
    friedman_chi2: float
    iman_davenport_f: float
    df1: int
    df2: int
    p_value: float
    # Average rank per method (rank 1 == best on a data set).
    avg_ranks: dict[str, float]


def friedman_iman_davenport(
    matrix: np.ndarray,
    methods: Sequence[str],
    *,
    lower_is_better: bool,
) -> FriedmanResult:
    """Friedman test with the Iman-Davenport correction over an ``N x k`` matrix.

    ``matrix`` holds one metric value per (data set, method); rows are data sets
    (scenes), columns are methods (in the order of ``methods``). The tie-corrected
    Friedman chi-square is taken from ``scipy.stats.friedmanchisquare`` and
    converted to the less conservative Iman-Davenport F statistic, distributed as
    F with ``k - 1`` and ``(k - 1)(N - 1)`` degrees of freedom. Average ranks
    (rank 1 == best on a data set) are computed with the metric's direction so the
    control comparison can tell improvements from regressions. Requires ``k >= 3``
    (the minimum for the Friedman test).
    """
    n, k = matrix.shape
    if k < 3:
        raise ValueError(f"Friedman test requires at least 3 methods, got {k}.")

    # rankdata assigns rank 1 to the smallest value; negate for higher-is-better
    # metrics so that rank 1 always denotes the best method on a scene.
    to_rank = matrix if lower_is_better else -matrix
    ranks = np.vstack([stats.rankdata(row) for row in to_rank])
    avg_ranks = ranks.mean(axis=0)

    # Tie-corrected chi-square (direction-independent) from SciPy.
    chi2 = float(stats.friedmanchisquare(*[matrix[:, j] for j in range(k)]).statistic)

    df1, df2 = k - 1, (k - 1) * (n - 1)
    denom = n * (k - 1) - chi2
    if denom <= 0:
        # All data sets rank the methods identically: maximal separation.
        f_stat = float("inf")
        p_value = 0.0
    else:
        f_stat = (n - 1) * chi2 / denom
        p_value = float(stats.f.sf(f_stat, df1, df2))

    return FriedmanResult(
        n_data_sets=n,
        k_methods=k,
        friedman_chi2=chi2,
        iman_davenport_f=float(f_stat),
        df1=df1,
        df2=df2,
        p_value=p_value,
        avg_ranks={m: float(r) for m, r in zip(methods, avg_ranks)},
    )


def holm_improvements_over_control(
    avg_ranks: Mapping[str, float],
    *,
    control: str,
    n_data_sets: int,
    k_methods: int,
    alpha: float = 0.05,
) -> set[str]:
    """Holm step-down comparison of each method against a control (Demšar 2006).

    The test statistic per method is ``z = (R_i - R_control) / SE`` with
    ``SE = sqrt(k(k + 1) / (6N))``. A method improves over the control when its
    average rank is lower, so the one-sided improvement p-value is ``Phi(z)``.
    These p-values are Holm-corrected with ``statsmodels`` and the methods whose
    corrected hypothesis is rejected at ``alpha`` are returned.
    """
    se = float(np.sqrt(k_methods * (k_methods + 1) / (6.0 * n_data_sets)))
    others = [m for m in avg_ranks if m != control]
    if se == 0.0 or not others:
        return set()

    control_rank = avg_ranks[control]
    # One-sided (improvement) p-values: a lower rank than the control yields a
    # negative z and hence a small p-value.
    pvalues = [
        float(stats.norm.cdf((avg_ranks[m] - control_rank) / se)) for m in others
    ]
    reject, _, _, _ = multipletests(pvalues, alpha=alpha, method="holm")
    return {method for method, rejected in zip(others, reject) if rejected}


def friedman_holm_improvements_over_control(
    per_method: Mapping[str, pd.Series],
    *,
    control: str,
    lower_is_better: bool,
    alpha: float = 0.05,
) -> tuple[FriedmanResult | None, set[str]]:
    """Demšar procedure for one strategy/metric: Friedman gate then Holm vs control.

    ``per_method`` maps each method name (including ``control``) to a per-scene
    scalar Series (index == scene). Scenes present for every method form the
    complete blocks required by the Friedman test. If the Friedman/Iman-Davenport
    omnibus null is rejected at ``alpha``, Holm's step-down procedure returns the
    non-control methods that significantly improve over the control; otherwise the
    set is empty.

    Returns ``(friedman, significant)`` where ``friedman`` is the Friedman result
    (or ``None`` when the test could not be run for lack of complete blocks), so
    callers can report whether the omnibus null was rejected.
    """
    methods = list(per_method)
    if control not in methods or len(methods) < 2:
        return None, set()

    common: pd.Index | None = None
    for series in per_method.values():
        idx = series.dropna().index
        common = idx if common is None else common.intersection(idx)
    if common is None or len(common) < 2:
        return None, set()

    matrix = np.column_stack(
        [per_method[m].reindex(common).to_numpy(dtype=float) for m in methods]
    )
    matrix = matrix[~np.isnan(matrix).any(axis=1)]
    # The Friedman test needs at least 2 blocks (scenes) and 3 methods.
    if matrix.shape[0] < 2 or matrix.shape[1] < 3:
        return None, set()

    friedman = friedman_iman_davenport(matrix, methods, lower_is_better=lower_is_better)
    if not (friedman.p_value < alpha):
        return friedman, set()

    significant = holm_improvements_over_control(
        friedman.avg_ranks,
        control=control,
        n_data_sets=matrix.shape[0],
        k_methods=matrix.shape[1],
        alpha=alpha,
    )
    return friedman, significant
