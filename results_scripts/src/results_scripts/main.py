from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, make_dataclass, replace
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

from eval_scripts.common.ansi_escapes import ANSIEscapes, ansiesc_print
import matplotlib.pyplot as plt
from mlflow.pyfunc import DATA
import numpy as np
import pandas as pd

from results_scripts.base import (
    DEFAULT_TABLE_METRICS,
    RunsInfo,
    drop_scenes_not_present_in_all,
    load_and_prepare_dataset_runs,
    load_init_method_runs,
)
from results_scripts.data_access import (
    ColumnSpec,
    collect_columns,
    concat_columns_into,
    drop_uncommon_scenes,
)
from results_scripts.param_conversions import PARAM_CONVERSIONS
from results_scripts.plots import format_number_compactly, per_scene_metric_dotplots
from results_scripts.statistics import (
    cell_data_across_strategies,
    hedges_g,
    per_scene_metric_difference,
    series_mean_frame_mean,
)
from results_scripts.statistics import (
    friedman_holm_improvements_over_control,
)
from results_scripts.tables import (
    tabular_colored_from_numeric_with_custom_text,
)
import tyro

from results_scripts.constants import (
    ALL_DATASETS,
    ALL_DATASETS_WITHOUT_ETH3D,
    ALL_STRATEGIES,
    ALL_STRATEGIES_EXCEPT_NO_D,
    BASE_DATASETS_WITHOUT_ETH3D,
    DATASET_NAMES,
    DENSE_INIT_METRICS,
    LASER_DATASETS,
    LASER_DATASETS_WITHOUT_ETH3D,
    LOWER_IS_BETTER_METRICS,
    METRIC_NAME_MAP,
    METRIC_PRETTY_NAMES,
    PHOTOMETRIC_METRICS,
    SCANNETPP_DA3_TEST_SCENE_SELECTION,
    SCANNETPP_SCENE_SELECTION,
    STRATEGY_NAMES,
    TABLE_ROUNDING_PER_METRIC,
    TRACKING_URI,
)
from results_scripts.formatting import (
    CellData,
    FormatOptions,
    MetricsLayout,
    TableCellType,
    make_cell_formatter,
)
from results_scripts.tables import (
    DIVERGING_CMAP,
    VALUE_CMAP,
    finalize_per_dataset_tables,
    make_aggregated_metric_table,
    make_latex_table_for_metrics,
    wrap_tabulars_as_float,
)
from results_scripts.base import get_cache_dir, load_or_download_runs
from results_scripts.utils import (
    OutputDirHelper,
    gmax_fraction_label,
    load_json,
    name_to_path,
    print_friedman_summary,
    save_figure_svg,
    write_file,
)

# Registry mapping a section function's name to the dataclass holding its extra,
# section-specific CLI args. Populated by the ``section_config`` decorator.
SECTION_CONFIG_TYPES: dict[str, type] = {}

_ConfigT = TypeVar("_ConfigT")


def section_config(
    config_cls: type[_ConfigT],
) -> Callable[[Callable[..., None]], Callable[..., None]]:
    """Attach a config dataclass with extra CLI args to a section function.

    A section function decorated with ``@section_config(MyArgs)`` must accept the
    config instance as its third positional argument::

        @dataclass
        class MyArgs:
            my_option: int = 3

        @section_config(MyArgs)
        def my_section(ctx, format_options, section_args: MyArgs) -> None:
            ...

    Each field of ``config_cls`` becomes a CLI option named
    ``<section_name>.<field>=<value>`` (e.g. ``my_section.my-option=5``).
    """

    def decorator(fn: Callable[..., None]) -> Callable[..., None]:
        SECTION_CONFIG_TYPES[fn.__name__] = config_cls
        return fn

    return decorator


@dataclass
class BaseArgs:
    tracking_uri: str = TRACKING_URI
    main_experiment_name: str = "main"

    # Whether to download runs from the tracking server if not present in cache.
    download: bool = False
    # If set, only load evaluation metrics up to this training iteration (inclusive).
    max_eval_iter: int | None = None
    workdir: Path = Path("./processed_results")


@dataclass
class ResultsContext:
    workdir: Path
    tracking_uri: str
    output_helper: OutputDirHelper
    num_pts_per_scene: dict[str, int]
    sfm_init_num_pts_per_scene: dict[str, int]
    real_init_num_pts_per_scene: dict[str, int]
    runs_per_dataset: dict[str, RunsInfo]
    download: bool = False
    init_method_runs: dict[str, RunsInfo] = field(default_factory=dict)

    @property
    def cache_dir(self) -> Path:
        return get_cache_dir(self.workdir, self.tracking_uri)

    def get_init_method_runs(self, experiment_name: str) -> RunsInfo:
        if experiment_name not in self.init_method_runs:
            cache_path = self.cache_dir / "init_method_runs" / f"{experiment_name}.pkl"
            self.init_method_runs[experiment_name] = load_or_download_runs(
                cache_path=cache_path,
                loader=lambda: load_init_method_runs(
                    experiment_name=experiment_name,
                    tracking_uri=self.tracking_uri,
                ),
                download=self.download,
                label=f"init-method runs '{experiment_name}'",
            )
        return self.init_method_runs[experiment_name]

    @staticmethod
    def create(args: BaseArgs) -> ResultsContext:
        workdir = args.workdir
        input_dir = workdir / "input"
        output_dir = workdir / "output"
        num_pts_per_scene = load_json(input_dir / "gmax_per_scene.json")
        sfm_init_num_pts_per_scene = load_json(
            input_dir / "init_sfm_pts_per_scene.json"
        )
        real_init_num_pts_per_scene = load_json(
            input_dir / "real_init_num_points_per_scene.json"
        )

        cache_dir = get_cache_dir(workdir, args.tracking_uri)
        runs_per_dataset: dict[str, RunsInfo] = {}
        for dataset in ALL_DATASETS:
            cache_path = (
                cache_dir
                / "datasets"
                / f"{name_to_path(dataset, allow_subdirs=False)}_{args.max_eval_iter if args.max_eval_iter is not None else 'all'}.pkl"
            )
            runs_per_dataset[dataset] = load_or_download_runs(
                cache_path=cache_path,
                loader=lambda dataset_name=dataset: load_and_prepare_dataset_runs(
                    dataset=dataset_name,
                    tracking_uri=args.tracking_uri,
                    main_experiment_name=args.main_experiment_name,
                    num_pts_per_scene=num_pts_per_scene,
                    sfm_init_num_pts_per_scene=sfm_init_num_pts_per_scene,
                    real_init_num_pts_per_scene=real_init_num_pts_per_scene,
                    max_eval_iter=args.max_eval_iter,
                ),
                download=args.download,
                label=f"dataset runs '{dataset}'",
            )

        return ResultsContext(
            workdir=workdir,
            tracking_uri=args.tracking_uri,
            output_helper=OutputDirHelper(output_dir),
            num_pts_per_scene=num_pts_per_scene,
            sfm_init_num_pts_per_scene=sfm_init_num_pts_per_scene,
            real_init_num_pts_per_scene=real_init_num_pts_per_scene,
            runs_per_dataset=runs_per_dataset,
            download=args.download,
        )


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s[%(levelname)s]: %(message)s",
        datefmt="%H:%M:%S",
    )


def compute_plot_limits(
    plot_starts_per_dataset: dict[str, dict[str, float]],
    plot_ranges_per_metric: dict[str, float],
    dataset: str,
) -> dict[str, tuple[float, float]]:
    starts = plot_starts_per_dataset.get(dataset, {})
    return {
        metric: (starts.get(metric, 0), starts.get(metric, 0) + range_value)
        for metric, range_value in plot_ranges_per_metric.items()
    }


# def get_sfm_baseline_metrics(
#     runs: RunsInfo,
#     strategies: list[str],
#     common_args: dict | None = None,
# ) -> dict[str, pd.DataFrame]:
#     """Per-scene metrics of the SfM baseline run for each strategy, keyed by short name."""
#     common_args = common_args or {}
#     return {
#         STRATEGY_NAMES[strategy]: runs.get_per_scene_metrics_for_params(
#             {**common_args, "init_group": "sfm_baseline", "strategy": strategy}
#         )
#         for strategy in strategies
#     }


def build_means_with_sfm_baseline(
    sfm_baselines: dict[str, pd.DataFrame],
    data: dict[str, dict[str, pd.DataFrame]],
    sfm_label: str = "SfM (Size differs)",
) -> dict[str, dict[str, pd.Series]]:
    data_means: dict[str, dict[str, pd.Series]] = {
        name: {sfm_label: df.mean()} for name, df in sfm_baselines.items()
    }
    for strategy, method_dict in data.items():
        for method, df in method_dict.items():
            data_means.setdefault(strategy, {})[method] = df.mean()
    return data_means


def significant_improvement_cells(
    data_per_dataset: dict[str, dict[str, dict[str, pd.DataFrame]]],
    *,
    sfm_column: str,
    metrics: list[str] = PHOTOMETRIC_METRICS,
    alpha: float = 0.05,
) -> dict[str, set[tuple[str, str, str]]]:
    """Mark statistically significant improvements over SfM (Demšar, 2006).

    For each (dataset, strategy, metric) the initializations are compared over the
    scenes with a Friedman test using the Iman--Davenport correction. When its
    omnibus null hypothesis is rejected, Holm's step-down procedure with SfM as the
    control flags the initializations that significantly *improve* over SfM. Each
    scene contributes a single value per initialization (the mean over eval
    iterations / seeds). Returns per-dataset ``(metric, strategy, init-column)``
    cells to mark.
    """

    def per_scene_scalars(series: pd.Series) -> pd.Series:
        return series.map(
            lambda values: float(np.mean(values)) if np.size(values) else np.nan
        )

    marked: dict[str, set[tuple[str, str, str]]] = {}
    friedman_records: list[tuple[str, str, str, float | None]] = []
    for dataset, data in data_per_dataset.items():
        total_num_cells_except_sfm = 0
        num_significant_cells = 0
        for strategy, columns in data.items():
            if sfm_column not in columns:
                continue

            total_num_cells_except_sfm += (len(columns) - 1) * len(metrics)

            for metric in metrics:
                per_method = {
                    column: per_scene_scalars(df[metric])
                    for column, df in columns.items()
                    if metric in df
                }
                if sfm_column not in per_method or len(per_method) < 2:
                    assert False

                friedman, significant = friedman_holm_improvements_over_control(
                    per_method,
                    control=sfm_column,
                    lower_is_better=metric in LOWER_IS_BETTER_METRICS,
                    alpha=alpha,
                )

                friedman_records.append(
                    (
                        DATASET_NAMES.get(dataset, dataset),
                        strategy,
                        METRIC_NAME_MAP.get(metric, metric),
                        friedman.p_value if friedman is not None else None,
                    )
                )
                for column in significant:
                    marked.setdefault(dataset, set()).add((metric, strategy, column))

                num_significant_cells += len(significant)
        print(
            f"Dataset: {dataset}, Total cells: {total_num_cells_except_sfm}, Significant cells: {num_significant_cells} ({num_significant_cells / total_num_cells_except_sfm * 100:.1f}%)"
        )

    print_friedman_summary(friedman_records, alpha=alpha)

    return marked


def save_bar_chart_legend(
    ctx: ResultsContext,
    section_subdir: str,
    name: str,
    handles,
    labels,
) -> None:
    legend_fig = plt.figure(figsize=(6, 1))
    legend = legend_fig.legend(
        handles,
        labels,
        loc="center",
        ncol=len(handles),
        columnspacing=0.25,
        handletextpad=0.15,
        frameon=False,
    )
    legend_fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(legend_fig.dpi_scale_trans.inverted())
    save_figure_svg(
        legend_fig,
        ctx.output_helper.get_graph_path(section_subdir, name),
        bbox_inches=bbox,
    )
    plt.close(legend_fig)


def save_line_chart_legend(
    ctx: ResultsContext,
    section_subdir: str,
    axes,
    fig_downscale: float = 1.15,
    name: str = "main_graph_legend",
) -> None:
    legend_fig = plt.figure(figsize=(18 / fig_downscale, 0.6 / fig_downscale))
    legend_handles, legend_labels = axes[0].get_legend_handles_labels()
    legend_fig.legend(
        legend_handles,
        legend_labels,
        loc="center",
        ncol=len(legend_labels),
        frameon=False,
    )
    save_figure_svg(legend_fig, ctx.output_helper.get_graph_path(section_subdir, name))
    plt.close(legend_fig)


##############################################################################
##############################################################################
@dataclass
class IncludeSparseArgs:
    include_sparse: bool = True


@section_config(IncludeSparseArgs)
def laser_scan_tables(
    ctx: ResultsContext, format_options: FormatOptions, cfg: IncludeSparseArgs
) -> None:
    common_args = {
        "is_default_strategy_config": True,
        "is_default_init_config": True,
        "init.position_noise_std": "0.0",
        "gaussian_cap_fraction": "1.0",
    }

    COL_SFM = "SfM"
    COL_AS_SFM = "$|\\mathcal{G}_\\mathit{init}^\\text{SfM}|$"
    COL_0_5 = gmax_fraction_label("0.5")
    COL_0_75 = gmax_fraction_label("0.75")
    COL_1_0 = gmax_fraction_label("1.0")
    COL_PER_FRACTION = {
        "0.5": COL_0_5,
        "0.75": COL_0_75,
        "1.0": COL_1_0,
    }

    COL_ORDER = [
        COL_SFM,
        COL_AS_SFM,
        COL_0_5,
        COL_0_75,
        COL_1_0,
    ]

    if cfg.include_sparse:
        print("Including sparse points in Laser Scan tables (except eth3d)!")

    data_per_dataset: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for dataset in LASER_DATASETS:
        runs = ctx.runs_per_dataset[dataset].copy()

        sfm_column = ColumnSpec(COL_SFM, {"init_group": "sfm_baseline"})
        as_sfm_column = ColumnSpec(
            COL_AS_SFM,
            {"init_method": "laser_scan", "init_size_matches_sfm": True},
            metrics=DENSE_INIT_METRICS,
        )
        fraction_columns = [
            ColumnSpec(
                COL_PER_FRACTION[size_fraction],
                {
                    "init_method": "laser_scan",
                    "init_size_matches_gmax": True,
                    "dense_init.target_points_fraction": size_fraction,
                    "dense_init.include_sparse": (
                        cfg.include_sparse and dataset != "eth3d"
                    ),
                },
                metrics=DENSE_INIT_METRICS,
            )
            for size_fraction in ["0.5", "0.75", "1.0"]
        ]

        # The SfM baseline column carries no shared config args; the laser-scan
        # columns do. The "No D." strategy only has the fraction columns.
        data = collect_columns(runs, ALL_STRATEGIES_EXCEPT_NO_D, [sfm_column])
        collect_columns(
            runs,
            ALL_STRATEGIES_EXCEPT_NO_D,
            [as_sfm_column],
            common_args=common_args,
            into=data,
        )
        collect_columns(
            runs, ALL_STRATEGIES, fraction_columns, common_args=common_args, into=data
        )

        drop_uncommon_scenes(data, debug_out=True)

        data_per_dataset[dataset] = data

    significant_cells = significant_improvement_cells(
        data_per_dataset, sfm_column=COL_SFM
    )

    tables: dict[str, str] = {}
    for dataset, data in data_per_dataset.items():
        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"laser_scan_main_{dataset}",
            column_order=COL_ORDER,
            row_order=[STRATEGY_NAMES[strategy] for strategy in ALL_STRATEGIES],
            format_args=format_options,
            significant_cells=significant_cells.get(dataset),
        )

    path = ctx.output_helper.get_table_path("laser_scan")
    write_file(
        path,
        finalize_per_dataset_tables(
            tables,
            format_options,
            combined_caption=(
                "Laser scan initialization performance strategies and initialization sizes. "
                r"$^{*}$ indicates a statistically significant improvement over SfM "
                "(Friedman test with Holm's step-down procedure)."
            ),
            combined_label="laser_scan_main",
        ),
    )
    print(f"Saved main Laser Scan table to {path}")


@dataclass
class LaserScanAnalysisPlotsArgs:
    datasets: list[str] = field(default_factory=lambda: list(LASER_DATASETS))
    strategies: list[str] = field(default_factory=lambda: list(ALL_STRATEGIES))
    metrics: list[str] = field(default_factory=lambda: list(PHOTOMETRIC_METRICS))
    size_fractions: list[str] = field(default_factory=lambda: ["0.5", "0.75", "1.0"])
    include_sparse: bool = True

    show_scene_labels: bool = True
    show_strategy_title: bool = True


@section_config(LaserScanAnalysisPlotsArgs)
def laser_scan_analysis_plots(
    ctx: ResultsContext,
    format_options: FormatOptions,
    cfg: LaserScanAnalysisPlotsArgs,
) -> None:
    """Plot laser-scan initialization sizes for each densification strategy."""
    del format_options

    common_args = {
        "is_default_strategy_config": True,
        "is_default_init_config": True,
        "init.position_noise_std": "0.0",
        "gaussian_cap_fraction": "1.0",
    }
    labels = {
        "sfm": "SfM",
        "as_sfm": "Laser (SfM size)",
        **{
            fraction: f"Laser ({float(fraction):g} Gm)"
            for fraction in cfg.size_fractions
        },
    }
    data: dict[str, dict[str, list[pd.DataFrame]]] = {
        strategy: {config: [] for config in labels} for strategy in cfg.strategies
    }

    for dataset in cfg.datasets:
        runs = ctx.runs_per_dataset[dataset].copy()
        for strategy in cfg.strategies:
            dataset_frames: list[pd.DataFrame] = []
            if strategy != "DefaultWithoutADCStrategy":
                sfm_frame = runs.get_per_scene_metrics_for_params(
                    {"init_group": "sfm_baseline", "strategy": strategy},
                    metrics=cfg.metrics,
                )
                as_sfm_frame = runs.get_per_scene_metrics_for_params(
                    {
                        **common_args,
                        "strategy": strategy,
                        "init_method": "laser_scan",
                        "init_size_matches_sfm": True,
                    },
                    metrics=cfg.metrics,
                )
                if not sfm_frame.empty:
                    data[strategy]["sfm"].append(sfm_frame)
                    dataset_frames.append(sfm_frame)
                if not as_sfm_frame.empty:
                    data[strategy]["as_sfm"].append(as_sfm_frame)
                    dataset_frames.append(as_sfm_frame)

            for fraction in cfg.size_fractions:
                frame = runs.get_per_scene_metrics_for_params(
                    {
                        **common_args,
                        "strategy": strategy,
                        "init_method": "laser_scan",
                        "init_size_matches_gmax": True,
                        "dense_init.target_points_fraction": fraction,
                        "dense_init.include_sparse": (
                            cfg.include_sparse and dataset != "eth3d"
                        ),
                    },
                    metrics=cfg.metrics,
                )
                if frame.empty:
                    continue
                data[strategy][fraction].append(frame)
                dataset_frames.append(frame)

            if dataset_frames:
                drop_scenes_not_present_in_all(*dataset_frames, debug_out=False)

    for strategy in cfg.strategies:
        frames_per_config = {
            config: pd.concat(frames)
            for config, frames in data[strategy].items()
            if frames
        }
        if not frames_per_config:
            logging.warning("No laser-scan plotting data found for %s", strategy)
            continue
        fig, _ = per_scene_metric_dotplots(
            data=frames_per_config,
            labels={config: labels[config] for config in frames_per_config},
            metrics=cfg.metrics,
            title=(
                STRATEGY_NAMES.get(strategy, strategy)
                if cfg.show_strategy_title
                else None
            ),
            show_scene_labels=cfg.show_scene_labels,
        )
        output_path = ctx.output_helper.get_graph_path("laser_scan_analysis", strategy)
        save_figure_svg(fig, output_path)
        plt.close(fig)


InitMethodId = Literal[
    "sfm", "laser_scan", "monodepth", "edgs", "edgs_sh", "da3", "da3_no_fr", "da3_gs"
]


@dataclass
class ImprovementTablesArgs:
    init_methods: list[InitMethodId] = field(
        default_factory=lambda: ["laser_scan", "monodepth"]
    )
    datasets: list[str] = field(
        default_factory=lambda: list(BASE_DATASETS_WITHOUT_ETH3D)
    )
    include_sparse_laser: bool = True
    include_sparse_other: bool = False


@section_config(ImprovementTablesArgs)
def improvement_tables(
    ctx: ResultsContext, format_options: FormatOptions, cfg: ImprovementTablesArgs
) -> None:
    common_args = {
        "is_default_init_config": True,
        "is_default_strategy_config": True,
        "init.position_noise_std": "0.0",
        "gaussian_cap_fraction": "1.0",
    }
    common_non_laser_args = {
        "dense_init.include_sparse": cfg.include_sparse_other,
    }

    metrics = [
        "eval-all-test/psnr",
        "eval-all-test/ssim",
        "eval-all-test/lpips",
    ]
    strat_names = [STRATEGY_NAMES[strategy] for strategy in ALL_STRATEGIES_EXCEPT_NO_D]

    laser_label = (
        "$\\text{{Laser}}^+$" if cfg.include_sparse_laser else "$\\text{{Laser}}$"
    )
    # Per init-method: column label, query params (merged with common_args and the
    # strategy), which metrics to load, and whether it only has data for GT datasets.
    # ``gt_only`` methods (e.g. laser scan) are silently skipped for non-GT datasets.
    init_method_specs: dict[str, dict[str, Any]] = {
        "laser_scan": {
            "label": rf"{gmax_fraction_label('0.75')} {laser_label}",
            "params": {
                "dense_init.target_points_fraction": "0.75",
                "init_method": "laser_scan",
                "init_size_matches_gmax": True,
                "dense_init.include_sparse": cfg.include_sparse_laser,
            },
            "metrics": DENSE_INIT_METRICS,
            "gt_only": True,
        },
        "monodepth": {
            "label": "Monodepth",
            "params": {
                "init_method": "monodepth",
                "dense_init.target_points_fraction": "1.0",
                **common_non_laser_args,
            },
            "metrics": DEFAULT_TABLE_METRICS,
            "gt_only": False,
        },
        "edgs": {
            "label": r"$\text{EDGS}^*$",
            "params": {
                "init_method": "edgs",
                "init_method_config": "default",
                "splat_init.increase_scale_with_fewer_splats": True,
                **common_non_laser_args,
            },
            "metrics": DEFAULT_TABLE_METRICS,
            "gt_only": False,
        },
        "edgs_sh": {
            "label": r"$\text{EDGS}$",
            "params": {
                "init_method": "edgs",
                "init_method_config": "full_sh_init=True",
                "splat_init.increase_scale_with_fewer_splats": True,
                **common_non_laser_args,
            },
            "metrics": DEFAULT_TABLE_METRICS,
            "gt_only": False,
        },
        "da3": {
            "label": "DA3",
            "params": {
                "init_method": "da3",
                "init_method_config": "floater_removal=True",
                **common_non_laser_args,
            },
            "metrics": DEFAULT_TABLE_METRICS,
            "gt_only": False,
        },
        "da3_no_fr": {
            "label": r"$\text{DA3}^\text{No F.R.}$",
            "params": {
                "init_method": "da3",
                "init_method_config": "default",
                **common_non_laser_args,
            },
            "metrics": DEFAULT_TABLE_METRICS,
            "gt_only": False,
        },
        "da3_gs": {
            "label": r"$\text{DA3}^\text{G.S.}$",
            "params": {
                "init_method": "da3",
                "init_method_config": "output_gaussians=True_max_num_images=150",
                **common_non_laser_args,
            },
            "metrics": DEFAULT_TABLE_METRICS,
            "gt_only": False,
        },
    }

    init_labels: dict[str, str] = {
        init: init_method_specs[init]["label"] for init in cfg.init_methods
    }

    def col_id(init: str, metric: str) -> str:
        return f"{init}_{metric}"

    metric_headers = " & ".join(
        rf"\textbf{{{label}}}"
        for label in (
            r"$\Delta$PSNR $\uparrow$",
            r"$\Delta$SSIM $\uparrow$",
            r"$\Delta$LPIPS $\downarrow$",
        )
    )

    def side_by_side_header(active: list[str]) -> str:
        # One \multicolumn metric group per init method; only the last group omits
        # the trailing column separator.
        multicols = " & ".join(
            rf"\multicolumn{{3}}{{{'c' if i == len(active) - 1 else 'c|'}}}{{{init_labels[init]}}}"
            for i, init in enumerate(active)
        )
        return (
            rf"& {multicols} \\"
            "\n"
            rf"\textbf{{Strategy}} & "
            + " & ".join([metric_headers] * len(active))
            + r" \\"
        )

    format_cell = make_cell_formatter(
        format_options.cell_type,
        rounding_per_metric=TABLE_ROUNDING_PER_METRIC,
    )

    print(
        f"Include Sparse: laser={cfg.include_sparse_laser}, other={cfg.include_sparse_other}"
    )

    tables_per_dataset: dict[str, str] = {}
    for dataset in cfg.datasets:
        # ``gt_only`` init methods (laser scan) only have data for GT datasets.
        active_init_methods: list[str] = [
            init
            for init in cfg.init_methods
            if not (
                init_method_specs[init]["gt_only"] and dataset not in LASER_DATASETS
            )
        ]
        if not active_init_methods:
            print(f"Skipping dataset {dataset}: no applicable init methods.")
            continue

        # Layout includes an extra synthetic summary group when requested; the
        # underlying data is still only fetched for the real init methods.

        columns = [
            col_id(init, metric) for init in active_init_methods for metric in metrics
        ]

        runs = ctx.runs_per_dataset[dataset].copy()

        sfm_data: dict[str, pd.DataFrame] = {}
        improvement_data: dict[str, dict[str, pd.DataFrame]] = {
            init: {} for init in active_init_methods
        }
        for strategy in ALL_STRATEGIES_EXCEPT_NO_D:
            strat_name = STRATEGY_NAMES[strategy]
            sfm_data[strat_name] = runs.get_per_scene_metrics_for_params(
                {"init_group": "sfm_baseline", "strategy": strategy}
            )
            for init in active_init_methods:
                spec = init_method_specs[init]
                improvement_data[init][strat_name] = (
                    runs.get_per_scene_metrics_for_params(
                        {**common_args, "strategy": strategy, **spec["params"]},
                        metrics=spec["metrics"],
                    )
                )

        drop_scenes_not_present_in_all(
            *sfm_data.values(),
            *(
                df
                for per_strat in improvement_data.values()
                for df in per_strat.values()
            ),
            debug_out=True,
        )

        # color_table drives the background gradient, text_table the cell text.
        color_table = pd.DataFrame(index=strat_names, columns=columns, dtype=float)
        text_table = pd.DataFrame(index=strat_names, columns=columns, dtype=object)

        for metric in metrics:
            # Lower-is-better metrics (e.g. LPIPS): flip the sign so positive ==
            # improvement (warm color) consistently with the other metrics.
            multiplier = -1.0 if metric in LOWER_IS_BETTER_METRICS else 1.0
            rounding = TABLE_ROUNDING_PER_METRIC[metric]
            cell_means: dict[tuple[str, str], float] = {}
            metric_max_abs = 0.0

            for init in active_init_methods:
                for strat_name in strat_names:
                    improvement = (
                        improvement_data[init][strat_name][metric]
                        - sfm_data[strat_name][metric]
                    )
                    mean_improvement = series_mean_frame_mean(improvement)
                    pooled_seed_variances_sfm = sfm_data[strat_name][metric].map(
                        lambda values: (
                            np.var(values, ddof=1) if np.size(values) else np.nan
                        )
                    )
                    pooled_seed_variances_other = improvement_data[init][strat_name][
                        metric
                    ].map(
                        lambda values: (
                            np.var(values, ddof=1) if np.size(values) else np.nan
                        )
                    )
                    pooled_seed_variances = pd.concat(
                        [pooled_seed_variances_sfm, pooled_seed_variances_other]
                    )
                    pooled_std_dev_sfm = np.sqrt(np.nanmean(pooled_seed_variances_sfm))
                    pooled_std_dev_other = np.sqrt(
                        np.nanmean(pooled_seed_variances_other)
                    )
                    pooled_std_dev = np.sqrt(np.nanmean(pooled_seed_variances))
                    effect_sizes = hedges_g(
                        sfm_data[strat_name][metric],
                        improvement_data[init][strat_name][metric],
                    )
                    mean_effect_size = np.nanmean(effect_sizes)

                    rounded_mean = round(mean_improvement, rounding)
                    cell_means[(init, strat_name)] = rounded_mean
                    metric_max_abs = max(metric_max_abs, abs(rounded_mean))

                    def fn(num):
                        return format_number_compactly(num, strip_leading_zero=True)

                    text_table.loc[strat_name, col_id(init, metric)] = (
                        rf"${rounded_mean:.{rounding}f} ({fn(pooled_std_dev_sfm)}, {fn(pooled_std_dev_other)}, {fn(mean_effect_size)})$"
                    )

            # Normalize colors per metric across all init methods. This keeps the
            # coloring scale consistent across metrics (which differ in magnitude)
            # and identical between init-method column groups, so colors remain
            # comparable across the per-init-method sub-tables.
            for (init, strat_name), mean in cell_means.items():
                normalized = (
                    multiplier * mean / metric_max_abs if metric_max_abs else 0.0
                )
                color_table.loc[strat_name, col_id(init, metric)] = normalized

        # Explicit shared color range (matches the helper's "centered" range with
        # padding), so the stacked sub-tables use an identical scale instead of
        # each recomputing its own from only one init method's values.
        max_abs = float(color_table.abs().max().max())
        color_range = (-1.2 * max_abs, 1.2 * max_abs)

        stack_init_methods = format_options.metrics_layout == MetricsLayout.vertical
        combine_as_subtables = format_options.combine_datasets_as_subtables
        table_env = format_options.get_table_env()
        if combine_as_subtables:
            resize_width = r"\linewidth"
        else:
            resize_width = r"\textwidth" if table_env == "table*" else r"\columnwidth"

        def wrap_resize(tabular: str) -> str:
            if not format_options.resizebox:
                return tabular
            return tabular.replace(
                r"\begin{tabular}",
                rf"\resizebox{{{resize_width}}}{{!}}{{\begin{{tabular}}",
            ).replace(r"\end{tabular}", r"\end{tabular}}")

        if stack_init_methods:
            # Stack the init methods as sections of a SINGLE tabular, so the column
            # widths stay aligned (separate tabulars + \resizebox would each scale
            # independently and misalign).
            section_tabulars: list[str] = []
            for init in active_init_methods:
                init_cols = [col_id(init, metric) for metric in metrics]
                sub_color = color_table[init_cols].set_axis(metrics, axis=1)
                sub_text = text_table[init_cols].set_axis(metrics, axis=1)
                section_tabulars.append(
                    tabular_colored_from_numeric_with_custom_text(
                        top_left_label="",
                        table=sub_color,
                        text_table=sub_text,
                        hide_nulls=False,
                        column_format="l|ccc",
                        header_block=(
                            rf"\textbf{{{init_labels[init]}}} & {metric_headers} \\"
                        ),
                        color_range=color_range,
                    )
                )

            first_lines = section_tabulars[0].splitlines()
            # First section: keep everything up to (but excluding) its \bottomrule.
            bottom_idx = next(
                i for i, ln in enumerate(first_lines) if r"\bottomrule" in ln
            )
            combined_lines = first_lines[:bottom_idx]
            # Subsequent sections: splice in everything after their \toprule
            # (header row + \midrule + body + \bottomrule + \end{tabular}),
            # separated from the previous section by a \midrule.
            for section in section_tabulars[1:]:
                lines = section.splitlines()
                top_idx = next(i for i, ln in enumerate(lines) if r"\toprule" in ln)
                combined_lines.append(r"\midrule")
                combined_lines.extend(lines[top_idx + 1 :])
            tabular = wrap_resize("\n".join(combined_lines))
        else:
            tabular = wrap_resize(
                tabular_colored_from_numeric_with_custom_text(
                    top_left_label="",
                    table=color_table,
                    text_table=text_table,
                    hide_nulls=False,
                    column_format="l|" + "|".join(["ccc"] * len(active_init_methods)),
                    header_block=side_by_side_header(active_init_methods),
                    color_range=color_range,
                )
            )

        if combine_as_subtables:
            env_begin = r"\begin{subtable}[t]{\linewidth}"
            env_end = r"\end{subtable}"
        else:
            env_begin = rf"\begin{{{table_env}}}[t]"
            env_end = rf"\end{{{table_env}}}"

        output_lines = [
            env_begin,
            r"\centering",
            rf"\caption{{{DATASET_NAMES.get(dataset, dataset)}}}",
            rf"\label{{tab:{dataset}_dense_improvement}}",
            "{" + format_options.get_latex_size(),
            format_options.get_tabcolsep_cmd_begin(),
            tabular,
            format_options.get_tabcolsep_cmd_end(),
            r"}",
            env_end,
        ]
        tables_per_dataset[dataset] = "\n".join(output_lines)

    path = ctx.output_helper.get_table_path("improvement_from_dense_init")
    write_file(
        path,
        finalize_per_dataset_tables(
            tables_per_dataset,
            format_options,
            combined_caption=("Improvement from dense initialization over SfM init."),
            combined_label="dense_improvement",
        ),
    )
    print(f"Saved improvement from dense init table to {path}")


def noise_resiliency(ctx: ResultsContext, format_options: FormatOptions) -> None:
    common_args = {
        "is_default_strategy_config": True,
        "init_size_matches_gmax": True,
        "dense_init.target_points_fraction": "0.5",
        "gaussian_cap_fraction": "1.0",
        "init_method": "laser_scan",
        "dense_init.include_sparse": False,
    }

    noise_levels = ["0.0", "0.01", "0.1"]

    def noise_name(noise: str | float) -> str:
        return str(noise)

    tables: dict[str, str] = {}
    for dataset in LASER_DATASETS:
        print("Dataset:", dataset)
        runs = ctx.runs_per_dataset[dataset].copy()

        columns = [
            ColumnSpec(noise_name(noise), {"init.position_noise_std": noise})
            for noise in noise_levels
        ]
        data = collect_columns(
            runs, ALL_STRATEGIES_EXCEPT_NO_D, columns, common_args=common_args
        )
        drop_uncommon_scenes(data, debug_out=True)

        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"noise_resiliency_{dataset}",
            column_order=[noise_name(noise) for noise in noise_levels],
            row_order=[
                STRATEGY_NAMES[strategy] for strategy in ALL_STRATEGIES_EXCEPT_NO_D
            ],
            format_args=format_options,
            horizontal_cols_label="Noise std.",
        )

    path = ctx.output_helper.get_table_path("noise_resiliency")
    write_file(
        path,
        finalize_per_dataset_tables(
            tables,
            format_options,
            combined_caption=(
                "Noise resiliency of laser scan initialization across strategies "
                "and position noise levels using Laser Scan init with "
                f"{gmax_fraction_label('0.5')} initial points."
            ),
            combined_label="noise_resiliency",
        ),
    )
    print(f"Saved noise resiliency table to {path}")


def init_times(ctx: ResultsContext, format_options: FormatOptions) -> None:
    monodepth_init_runs = ctx.get_init_method_runs("monodepth_init")
    edgs_init_runs_all = ctx.get_init_method_runs("edgs_init")
    edgs_init_runs = edgs_init_runs_all.get_runs_with_params({"full_sh_init": False})
    edgs_init_runs_full_sh = edgs_init_runs_all.get_runs_with_params(
        {"full_sh_init": True}
    )
    da3_init_runs_all = ctx.get_init_method_runs("da3_init")
    da3_init_runs_fr = da3_init_runs_all.get_runs_with_params({"floater_removal": True})
    da3_init_runs_no_fr = da3_init_runs_all.get_runs_with_params(
        {"output_gaussians": False, "floater_removal": False}
    )
    da3_init_runs_gs = da3_init_runs_all.get_runs_with_params(
        {"output_gaussians": True}
    )

    labeled_runs = {
        "EDGS": edgs_init_runs,
        "EDGS (full SH init)": edgs_init_runs_full_sh,
        "Monodepth": monodepth_init_runs,
        "DA3 (floater removal)": da3_init_runs_fr,
        "DA3 (no floater removal)": da3_init_runs_no_fr,
        "DA3 (gaussian splats)": da3_init_runs_gs,
    }

    for label, runs in labeled_runs.items():
        print(
            f"{label} mean init time (all datasets): "
            f"{runs.df['init_only_runtime'].mean():.2f}s"
        )


@dataclass
class PracticalTablesArgs:
    init_methods: list[InitMethodId] = field(
        default_factory=lambda: [
            "sfm",
            "edgs",
            # "edgs_sh",
            "monodepth",
            "da3",
            "da3_gs",
            "laser_scan",
        ]
    )
    strategies: list[str] = field(default_factory=lambda: ALL_STRATEGIES)
    strategy_args: dict[str, dict[str, str]] = field(
        default_factory=lambda: {name: dict() for name in STRATEGY_NAMES.keys()}
    )
    datasets: list[str] = field(default_factory=lambda: BASE_DATASETS_WITHOUT_ETH3D)
    include_sparse_for_all: Literal["yes", "no", "both"] = "no"
    include_half_init_size_for_all: Literal["yes", "no", "both"] = "no"
    include_sparse_for_laser_scannet: bool = True
    lpips_vgg: bool = False


@section_config(PracticalTablesArgs)
def practical_tables(
    ctx: ResultsContext, format_options: FormatOptions, cfg: PracticalTablesArgs
) -> None:
    COL_SFM = "SfM"
    COL_EDGS = "$\\text{EDGS}^*$"
    COL_EDGS_FULL_SH_INIT = "$\\text{EDGS}$"
    COL_MONODEPTH = "M. D."
    COL_DA3_NO_FLOATER_REMOVAL = "$\\text{DA3}^\\text{No F.R.}$"
    COL_DA3 = "DA3"
    COL_DA3_GS_INIT = "$\\text{DA3}^\\text{G.S.}$"
    COL_LASER = "Laser"

    metrics_to_collect = DEFAULT_TABLE_METRICS
    metrics_to_show = PHOTOMETRIC_METRICS
    if cfg.lpips_vgg:
        metrics_to_collect += ["eval-all-test/lpips_vgg"]
        metrics_to_show += ["eval-all-test/lpips_vgg"]


    method_id_to_col = {
        "sfm": COL_SFM,
        "edgs": COL_EDGS,
        "edgs_sh": COL_EDGS_FULL_SH_INIT,
        "monodepth": COL_MONODEPTH,
        "da3": COL_DA3,
        "da3_no_fr": COL_DA3_NO_FLOATER_REMOVAL,
        "da3_gs": COL_DA3_GS_INIT,
        "laser_scan": COL_LASER,
    }

    PRACTICAL_COLS = [
        method_id_to_col[method_id]
        for method_id in cfg.init_methods
        if method_id not in ["sfm", "laser_scan"]
    ]
    ALL_COLS = [COL_SFM] if "sfm" in cfg.init_methods else []
    ALL_COLS += PRACTICAL_COLS
    ALL_COLS += [COL_LASER] if "laser_scan" in cfg.init_methods else []

    def _strat_arg_overrides(strategy: str) -> dict[str, Any]:
        args = cfg.strategy_args.get(strategy, {})
        if len(args) == 0:
            return {"is_default_strategy_config": True}
        return {k: PARAM_CONVERSIONS.get(k, lambda x: x)(v) for k, v in args.items()}

    # Dict:  strategy -> column -> per-scene dataframe across all datasets for all metrics with lists of values per eval iter in cells.
    all_datasets_data: dict[str, dict[str, pd.DataFrame]] = {}

    INCLUDE_SPARSE_COLS = [
        COL_MONODEPTH,
        COL_DA3,
        COL_DA3_NO_FLOATER_REMOVAL,
        COL_LASER,
    ]
    HALF_INIT_SIZE_COLS = [
        COL_EDGS,
        COL_EDGS_FULL_SH_INIT,
        COL_MONODEPTH,
        COL_DA3,
        COL_DA3_NO_FLOATER_REMOVAL,
        COL_DA3_GS_INIT,
        COL_LASER,
    ]
    MARK_SPARSE = "+"
    MARK_HALF = "0.5"

    # Base query params per practical init-method column (strategy is injected by
    # the collector). The per-config dense-init size/sparse flags are added below.
    params_per_col_base: dict[str, dict[str, Any]] = {
        COL_EDGS: {
            "init_method": "edgs",
            "init_method_config": "default",
            "splat_init.increase_scale_with_fewer_splats": True,
        },
        COL_EDGS_FULL_SH_INIT: {
            "init_method": "edgs",
            "init_method_config": "full_sh_init=True",
            "splat_init.increase_scale_with_fewer_splats": True,
        },
        COL_MONODEPTH: {
            "init_method": "monodepth",
        },
        COL_DA3_NO_FLOATER_REMOVAL: {
            "init_method": "da3",
            "init_method_config": "default",
        },
        COL_DA3: {
            "init_method": "da3",
            "init_method_config": "floater_removal=True",
        },
        COL_DA3_GS_INIT: {
            "init_method": "da3",
            "init_method_config": "output_gaussians=True_max_num_images=150",
        },
    }

    default_target_fraction = (
        "0.5" if cfg.include_half_init_size_for_all == "yes" else "1.0"
    )

    def practical_column_specs() -> list[ColumnSpec]:
        specs: list[ColumnSpec] = []
        for col in PRACTICAL_COLS:
            base = params_per_col_base[col]
            specs.append(
                ColumnSpec(
                    col,
                    {
                        **base,
                        "dense_init.target_points_fraction": default_target_fraction,
                        "dense_init.include_sparse": (
                            cfg.include_sparse_for_all == "yes"
                        ),
                    },
                    metrics=metrics_to_collect,
                )
            )
            if cfg.include_sparse_for_all == "both" and col in INCLUDE_SPARSE_COLS:
                # Sparse-only variant (relies on the default init size).
                specs.append(
                    ColumnSpec(
                        f"$\\text{{{col}}}^{{{MARK_SPARSE}}}$",
                        {**base, "dense_init.include_sparse": True},
                    ),
                    metrics=metrics_to_collect,
                )
            if (
                cfg.include_half_init_size_for_all == "both"
                and col in HALF_INIT_SIZE_COLS
            ):
                specs.append(
                    ColumnSpec(
                        f"$\\text{{{col}}}^{{{MARK_HALF}}}$",
                        {
                            **base,
                            "dense_init.target_points_fraction": "0.5",
                            "dense_init.include_sparse": (
                                cfg.include_sparse_for_all == "yes"
                                and col in INCLUDE_SPARSE_COLS
                            ),
                        },
                        metrics=metrics_to_collect
                    )
                )
        return specs

    def laser_column_specs() -> list[ColumnSpec]:
        base = {"init_method": "laser_scan", "init_size_matches_real_init": True}
        specs = [
            ColumnSpec(
                COL_LASER,
                {
                    **base,
                    "dense_init.target_points_fraction": default_target_fraction,
                    "dense_init.include_sparse": (cfg.include_sparse_for_all == "yes")
                    or (
                        cfg.include_sparse_for_laser_scannet
                        and "scannet++" in dataset
                        and cfg.include_sparse_for_all != "both"
                    ),
                },
                gt_only=True,
                metrics=metrics_to_collect
            )
        ]
        if cfg.include_sparse_for_all == "both":
            specs.append(
                ColumnSpec(
                    f"$\\text{{{COL_LASER}}}^{{{MARK_SPARSE}}}$",
                    {
                        **base,
                        "dense_init.target_points_fraction": "1.0",
                        "dense_init.include_sparse": True,
                    },
                    gt_only=True,
                    metrics=metrics_to_collect
                )
            )
        if cfg.include_half_init_size_for_all == "both":
            specs.append(
                ColumnSpec(
                    f"$\\text{{{COL_LASER}}}^{{{MARK_HALF}}}$",
                    {
                        **base,
                        "dense_init.target_points_fraction": "0.5",
                        "dense_init.include_sparse": (
                            cfg.include_sparse_for_all == "yes"
                        )
                        or (
                            cfg.include_sparse_for_laser_scannet
                            and "scannet++" in dataset
                        ),
                    },
                    gt_only=True,
                    metrics=metrics_to_collect
                )
            )
        return specs

    data_per_dataset: dict[str, dict[str, dict[str, pd.DataFrame]]] = {}
    for dataset in cfg.datasets:
        print("Dataset:", dataset)
        runs = ctx.runs_per_dataset[dataset].copy()

        common_args = {
            "is_default_init_config": True,
            "gaussian_cap_fraction": "1.0",
            "init.position_noise_std": "0.0",
        }

        data: dict[str, dict[str, pd.DataFrame]] = {}

        if "sfm" in cfg.init_methods:
            # SfM baseline carries no shared init config; drop empty results.
            collect_columns(
                runs,
                [s for s in cfg.strategies if s != "DefaultWithoutADCStrategy"],
                [ColumnSpec(COL_SFM, {"init_group": "sfm_baseline"}, metrics=metrics_to_collect)],
                
                strategy_overrides=_strat_arg_overrides,
                skip_empty=True,
                into=data,
            )

        collect_columns(
            runs,
            cfg.strategies,
            practical_column_specs(),
            common_args=common_args,
            strategy_overrides=_strat_arg_overrides,
            on_error="warn",
            into=data,
        )

        if "laser_scan" in cfg.init_methods:
            collect_columns(
                runs,
                cfg.strategies,
                laser_column_specs(),
                common_args=common_args,
                strategy_overrides=_strat_arg_overrides,
                dataset=dataset,
                into=data,
            )

        try:
            drop_uncommon_scenes(data, debug_out=True)
        except Exception as e:
            print(f"Scene mismatch error for dataset {dataset}: {e}")
            continue

        concat_columns_into(all_datasets_data, data)

        data_per_dataset[dataset] = data

    significant_cells = (
        significant_improvement_cells(data_per_dataset, sfm_column=COL_SFM, metrics=metrics_to_show)
        if "sfm" in cfg.init_methods
        else {}
    )

    col_order = ALL_COLS.copy()
    if cfg.include_sparse_for_all == "both":
        # Add the sparse-only versions of the applicable methods after their main
        # columns.
        for col in [COL_MONODEPTH, COL_DA3, COL_DA3_NO_FLOATER_REMOVAL, COL_LASER]:
            if col in col_order:
                sparse_col = f"$\\text{{{col}}}^{{{MARK_SPARSE}}}$"
                col_order.insert(col_order.index(col) + 1, sparse_col)
    if cfg.include_half_init_size_for_all == "both":
        # Add the half-size versions of the applicable methods after their main
        # columns.
        for col in [
            COL_EDGS,
            COL_EDGS_FULL_SH_INIT,
            COL_MONODEPTH,
            COL_DA3,
            COL_DA3_NO_FLOATER_REMOVAL,
            COL_DA3_GS_INIT,
            COL_LASER,
        ]:
            if col in col_order:
                half_col = f"$\\text{{{col}}}^{{{MARK_HALF}}}$"
                col_order.insert(col_order.index(col), half_col)

    tables = {}
    for dataset, data in data_per_dataset.items():
        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"practical_main_{dataset}",
            metrics=metrics_to_show,
            column_order=col_order,
            row_order=[STRATEGY_NAMES[strategy] for strategy in cfg.strategies],
            format_args=format_options,
            significant_cells=significant_cells.get(dataset),
        )
    label_suffix = ""

    caption = "Practical initialization performance across densification strategies."
    if cfg.include_sparse_for_all == "yes":
        caption += " (With SfM points included.)"
        label_suffix += "_sparse=yes"
    if cfg.include_sparse_for_all == "both":
        caption += f" (``{MARK_SPARSE}'' indicates SfM points included.)"
        label_suffix += "_sparse=both"
    if cfg.include_half_init_size_for_all == "yes":
        caption += " (With half the number of initial points.)"
        label_suffix += "_half_init=yes"
    if cfg.include_half_init_size_for_all == "both":
        caption += f" (``{MARK_HALF}'' indicates half the number of initial points.)"
        label_suffix += "_half_init=both"
    if "sfm" in cfg.init_methods:
        caption += (
            r" ($^{*}$ indicates a statistically significant improvement over SfM "
            "using the Friedman test with Holm's step-down procedure.)"
        )
    path = ctx.output_helper.get_table_path("practical_main" + label_suffix)
    write_file(
        path,
        finalize_per_dataset_tables(
            tables,
            format_options,
            combined_caption=caption,
            combined_label="practical_main" + label_suffix,
        ),
    )
    print(f"Saved Practical Initialization table to {path}")

    if cfg.include_sparse_for_all != "no" or cfg.include_half_init_size_for_all != "no":
        print("Skipping training times table.")
        return

    format_args_times = deepcopy(format_options)
    format_args_times.cell_type = TableCellType.mean
    format_args_times.metrics_layout = MetricsLayout.vertical
    # Not applicable
    format_args_times.combine_datasets_as_subtables = False
    init_times_table = make_latex_table_for_metrics(
        data=all_datasets_data,
        latex_caption="Mean training times across all datasets.",
        latex_label="practical_train_times",
        column_order=ALL_COLS,
        row_order=[STRATEGY_NAMES[strategy] for strategy in cfg.strategies],
        format_args=format_args_times,
        metrics=["train/total-train-time"],
    )
    path = ctx.output_helper.get_table_path("practical_train_times")
    write_file(path, init_times_table)
    print(f"Saved Training Times table to {path}")


@dataclass
class PracticalAnalysisPlotsArgs:
    init_methods: list[InitMethodId] = field(
        default_factory=lambda: [
            "sfm",
            "edgs",
            "monodepth",
            "da3",
            "da3_gs",
            "laser_scan",
        ]
    )
    strategies: list[str] = field(default_factory=lambda: ALL_STRATEGIES)
    strategy_args: dict[str, dict[str, str]] = field(
        default_factory=lambda: {name: dict() for name in STRATEGY_NAMES.keys()}
    )
    datasets: list[str] = field(default_factory=lambda: BASE_DATASETS_WITHOUT_ETH3D)
    metrics: list[str] = field(default_factory=lambda: list(PHOTOMETRIC_METRICS))
    include_sparse_for_laser: bool = True
    show_scene_labels: bool = True
    show_strategy_title: bool = True


@section_config(PracticalAnalysisPlotsArgs)
def practical_analysis_plots(
    ctx: ResultsContext,
    format_options: FormatOptions,
    cfg: PracticalAnalysisPlotsArgs,
) -> None:
    """Plot every eval iteration for practical initializations, grouped by scene."""
    del format_options

    method_specs: dict[InitMethodId, tuple[str, dict[str, Any], bool]] = {
        "sfm": ("SfM", {"init_group": "sfm_baseline"}, False),
        "edgs": (
            "EDGS*",
            {
                "init_method": "edgs",
                "init_method_config": "default",
                "splat_init.increase_scale_with_fewer_splats": True,
            },
            False,
        ),
        "edgs_sh": (
            "EDGS",
            {
                "init_method": "edgs",
                "init_method_config": "full_sh_init=True",
                "splat_init.increase_scale_with_fewer_splats": True,
            },
            False,
        ),
        "monodepth": ("Monodepth", {"init_method": "monodepth"}, False),
        "da3_no_fr": (
            "DA3 (No F.R.)",
            {"init_method": "da3", "init_method_config": "default"},
            False,
        ),
        "da3": (
            "DA3",
            {"init_method": "da3", "init_method_config": "floater_removal=True"},
            False,
        ),
        "da3_gs": (
            "DA3 (G.S.)",
            {
                "init_method": "da3",
                "init_method_config": "output_gaussians=True_max_num_images=150",
            },
            False,
        ),
        "laser_scan": (
            "Laser",
            {
                "init_method": "laser_scan",
                "init_size_matches_real_init": True,
                "dense_init.include_sparse": cfg.include_sparse_for_laser,
            },
            True,
        ),
    }
    common_args = {
        "is_default_init_config": True,
        "gaussian_cap_fraction": "1.0",
        "init.position_noise_std": "0.0",
        "dense_init.target_points_fraction": "1.0",
        "dense_init.include_sparse": False,
    }

    def strategy_overrides(strategy: str) -> dict[str, Any]:
        args = cfg.strategy_args.get(strategy, {})
        if not args:
            return {"is_default_strategy_config": True}
        return {
            key: PARAM_CONVERSIONS.get(key, lambda value: value)(value)
            for key, value in args.items()
        }

    data: dict[str, dict[str, list[pd.DataFrame]]] = {
        strategy: {method: [] for method in cfg.init_methods}
        for strategy in cfg.strategies
    }
    for dataset in cfg.datasets:
        runs = ctx.runs_per_dataset[dataset].copy()
        for strategy in cfg.strategies:
            dataset_frames: list[pd.DataFrame] = []
            for method in cfg.init_methods:
                label, method_args, gt_only = method_specs[method]
                if gt_only and dataset not in LASER_DATASETS:
                    continue
                if method == "sfm" and strategy == "DefaultWithoutADCStrategy":
                    continue
                query = {
                    **({} if method == "sfm" else common_args),
                    "strategy": strategy,
                    **method_args,
                    **strategy_overrides(strategy),
                }
                try:
                    frame = runs.get_per_scene_metrics_for_params(
                        query, metrics=cfg.metrics
                    )
                except ValueError as exc:
                    logging.warning(
                        "Skipping %s / %s / %s: %s", dataset, strategy, label, exc
                    )
                    continue
                if frame.empty:
                    continue
                dataset_frames.append(frame)
                data[strategy][method].append(frame)
            if dataset_frames:
                drop_scenes_not_present_in_all(*dataset_frames, debug_out=False)

    for strategy in cfg.strategies:
        frames_per_method = {
            method: pd.concat(frames)
            for method, frames in data[strategy].items()
            if frames
        }
        if not frames_per_method:
            logging.warning("No plotting data found for strategy %s", strategy)
            continue

        fig, _ = per_scene_metric_dotplots(
            data=frames_per_method,
            labels={method: method_specs[method][0] for method in frames_per_method},
            metrics=cfg.metrics,
            title=(
                STRATEGY_NAMES.get(strategy, strategy)
                if cfg.show_strategy_title
                else None
            ),
            show_scene_labels=cfg.show_scene_labels,
        )
        output_path = ctx.output_helper.get_graph_path("practical_analysis", strategy)
        save_figure_svg(fig, output_path)
        plt.close(fig)


def gaussian_cap_ablation(ctx: ResultsContext, format_options: FormatOptions) -> None:
    # Two fully separate tables, one per init method: SfM and laser scan at
    # 0.5 G_max. Each table has a subtable per dataset, with strategies in rows
    # and side-by-side metric triplets (PSNR/SSIM/LPIPS) for the cap fractions.
    init_method_args: dict[str, dict[str, object]] = {
        "sfm": {
            "init_method": "sfm",
        },
        "laser_scan": {
            "init_method": "laser_scan",
            "init_size_matches_gmax": True,
            "dense_init.target_points_fraction": "0.5",
            "dense_init.include_sparse": False,
        },
    }
    init_method_captions = {
        "sfm": "SfM initialization",
        "laser_scan": f"Laser scan initialization at {gmax_fraction_label('0.5')}",
    }

    cap_fractions = ["0.75", "1.0", "1.25"]
    # LaTeX-safe column labels (fraction of the Gaussian cap relative to G_max).
    cap_fraction_labels = {cap: gmax_fraction_label(cap) for cap in cap_fractions}

    for init_method, extra_args in init_method_args.items():
        print("========== Init method:", init_method, "==========")

        tables: dict[str, str] = {}
        for dataset in LASER_DATASETS:
            print("Dataset:", dataset)
            runs = ctx.runs_per_dataset[dataset].copy()

            columns = [
                ColumnSpec(
                    cap_fraction_labels[cap_fraction],
                    {"gaussian_cap_fraction": cap_fraction},
                )
                for cap_fraction in cap_fractions
            ]
            data = collect_columns(
                runs,
                ALL_STRATEGIES_EXCEPT_NO_D,
                columns,
                common_args={
                    "is_default_strategy_config": True,
                    "init.position_noise_std": "0.0",
                    **extra_args,
                },
            )
            drop_uncommon_scenes(data, debug_out=True)

            tables[dataset] = make_latex_table_for_metrics(
                data=data,
                latex_caption=DATASET_NAMES[dataset],
                latex_label=f"gaussian_cap_ablation_{init_method}_{dataset}",
                column_order=[cap_fraction_labels[cap] for cap in cap_fractions],
                row_order=[
                    STRATEGY_NAMES[strategy] for strategy in ALL_STRATEGIES_EXCEPT_NO_D
                ],
                format_args=format_options,
                horizontal_cols_label="Cap frac.",
            )

        path = ctx.output_helper.get_table_path(f"gaussian_cap_ablation_{init_method}")
        write_file(
            path,
            finalize_per_dataset_tables(
                tables,
                format_options,
                combined_caption=(
                    "Gaussian cap fraction ablation across strategies and cap "
                    f"fractions using {init_method_captions[init_method]}."
                ),
                combined_label=f"gaussian_cap_ablation_{init_method}",
            ),
        )
        print(f"Saved Gaussian cap ablation ({init_method}) table to {path}")


def _ablation(
    ctx: ResultsContext,
    format_options: FormatOptions,
    section_name: str,
    common_args: dict[str, object],
    args: list[dict[str, object]],
    labels: list[str],
    strategies=ALL_STRATEGIES,
    metrics=DEFAULT_TABLE_METRICS,
    datasets=ALL_DATASETS_WITHOUT_ETH3D,
    caption: str | None = None,
    summary_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_variants = len(args)
    if (num_variants != len(labels)) or (num_variants < 2):
        raise ValueError(
            f"Number of args ({num_variants}) must match number of labels ({len(labels)}) and be at least 2."
        )

    row_order = [STRATEGY_NAMES.get(strategy, strategy) for strategy in strategies]

    # Accumulate per-variant dataframes across all strategies and datasets so we
    # can print overall means at the end.
    all_datasets: list[list[pd.DataFrame]] = [[] for _ in labels]

    tables: dict[str, str] = {}
    # label -> metric -> mean acro
    means_per_dataset: dict[str, pd.DataFrame] = {
        dataset: pd.DataFrame(index=labels, columns=metrics) for dataset in datasets
    }
    for dataset in datasets:
        runs = ctx.runs_per_dataset[dataset].copy()

        # row (strategy) -> column (variant label) -> per-scene metrics dataframe
        columns = [
            ColumnSpec(label, args_i, metrics=metrics)
            for label, args_i in zip(labels, args)
        ]
        data = collect_columns(
            runs,
            strategies,
            columns,
            common_args=common_args,
            row_label=lambda s: STRATEGY_NAMES.get(s, s),
        )
        drop_uncommon_scenes(data, debug_out=False)

        for i, label in enumerate(labels):
            for strategy_label in data:
                all_datasets[i].append(data[strategy_label][label])

        for label in labels:
            for metric in metrics:
                means_per_dataset[dataset].loc[label, metric] = float(
                    series_mean_frame_mean(
                        pd.concat(
                            data[strategy_label][label][metric]
                            for strategy_label in data
                        )
                    )
                )

        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"{section_name}_{dataset}",
            metrics=metrics,
            column_order=labels,
            row_order=row_order,
            format_args=format_options,
            cmap=DIVERGING_CMAP,
        )

    all_datasets_dfs = [
        pd.concat(dfs, axis=0, ignore_index=True) for dfs in all_datasets
    ]
    comb_row: dict[str, float | str] = {"-": "All datasets"}
    for metric in metrics:
        pretty_name = METRIC_NAME_MAP.get(metric, metric)
        for i, df in enumerate(all_datasets_dfs):
            comb_row[f"{pretty_name} ({labels[i]})"] = float(
                series_mean_frame_mean(df[metric])
            )
    print("Overall means:")
    print(pd.DataFrame([comb_row]).set_index("-"))

    if summary_only:
        print("Skipping table output due to summary_only=True.")
        return pd.DataFrame([comb_row]).set_index("-"), means_per_dataset

    path = ctx.output_helper.get_table_path(section_name)
    write_file(
        path,
        finalize_per_dataset_tables(
            tables,
            format_options,
            combined_caption=caption,
            combined_label=section_name,
        ),
    )
    print(f"Saved {section_name} ablation table to {path}")
    return pd.DataFrame([comb_row]).set_index("-"), means_per_dataset


def _ablation_strategies_side_by_side(
    ctx: ResultsContext,
    format_options: FormatOptions,
    section_name: str,
    common_args: dict[str, object],
    args: list[dict[str, object]],
    labels: list[str],
    strategies=ALL_STRATEGIES,
    metrics=DEFAULT_TABLE_METRICS,
    datasets=ALL_DATASETS_WITHOUT_ETH3D,
    caption: str | None = None,
    delta: bool = False,
) -> None:
    """Ablation table that keeps every strategy as its own column in one tabular.

    Like ``_ablation_aggregate_strategies`` this uses one row per arg/label pair,
    but instead of collapsing the strategies into a single aggregated cell, every
    strategy is rendered side by side in the same ``tabular``. With the horizontal
    metrics layout the columns are grouped per metric (e.g.
    ``PSNR: strat1 | strat2 | ... || SSIM: strat1 | strat2 | ...``).

    When ``delta`` is set, the first arg/label pair is the reference row and the
    remaining rows show signed deltas relative to it.
    """
    num_variants = len(args)
    if (num_variants != len(labels)) or (num_variants < 2):
        raise ValueError(
            f"Number of args ({num_variants}) must match number of labels ({len(labels)}) and be at least 2."
        )

    strategy_labels = [
        STRATEGY_NAMES.get(strategy, strategy) for strategy in strategies
    ]

    tables: dict[str, str] = {}
    for dataset in datasets:
        runs = ctx.runs_per_dataset[dataset].copy()

        # variant label (row) -> strategy label (column) -> per-scene metrics df.
        # Transposed vs ``collect_columns`` (strategies are columns here), so the
        # queries are issued directly.
        data: dict[str, dict[str, pd.DataFrame]] = {}
        for label, args_i in zip(labels, args):
            for strategy, strategy_label in zip(strategies, strategy_labels):
                data.setdefault(label, {})[strategy_label] = (
                    runs.get_per_scene_metrics_for_params(
                        {"strategy": strategy, **common_args, **args_i},
                        metrics=metrics,
                    )
                )

        drop_uncommon_scenes(data, debug_out=False)

        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"{section_name}_{dataset}",
            metrics=metrics,
            column_order=strategy_labels,
            row_order=labels,
            format_args=format_options,
            cmap=DIVERGING_CMAP,
            delta=delta,
        )

    path = ctx.output_helper.get_table_path(section_name)
    write_file(
        path,
        finalize_per_dataset_tables(
            tables,
            format_options,
            combined_caption=caption,
            combined_label=section_name,
        ),
    )
    print(f"Saved {section_name} ablation table to {path}")


def edgs_scale_increase_ablation(
    ctx: ResultsContext, format_options: FormatOptions
) -> None:
    _, per_dataset_means = _ablation(
        ctx,
        format_options,
        section_name="edgs_scale_increase_ablation",
        common_args={
            "init_method": "edgs",
            "gaussian_cap_fraction": "1.0",
            "init_method_config": "default",
            "is_default_strategy_config": True,
        },
        args=[
            {
                "splat_init.increase_scale_with_fewer_splats": False,
            },
            {
                "splat_init.increase_scale_with_fewer_splats": True,
            },
        ],
        labels=["No Scale Increase", "Scale Increase"],
        strategies=["DefaultWithGaussianCapStrategy", "MCMCStrategy", "IDHFRStrategy"],
    )
    print("Per-dataset means for EDGS scale increase ablation:")
    for dataset, means_df in per_dataset_means.items():
        print(f"Dataset: {dataset}")
        print(means_df)
        print()


@dataclass
class InitScaleAblationArgs:
    datasets: list[str] = field(
        default_factory=lambda: ["mipnerf360", "tanksandtemples"]
    )
    init_method: str = "da3"
    da3_config: str = "floater_removal=True"
    init_scales: list[str] = field(default_factory=lambda: ["0.1", "0.25"])
    strategies: list[str] = field(default_factory=lambda: ["MCMCStrategy"])


@section_config(InitScaleAblationArgs)
def init_scale_ablation(
    ctx: ResultsContext, format_options: FormatOptions, cfg: InitScaleAblationArgs
) -> None:
    init_method_configs = {
        "da3": cfg.da3_config,
    }
    args = []
    labels = []
    for scale in cfg.init_scales:
        if scale == "0.1":
            args.append({"is_default_strategy_config": True})
        else:
            args.append({"init.scale_mult": scale})
        labels.append(scale)

    _ablation(
        ctx,
        format_options,
        section_name="init_scale_ablation",
        common_args={
            "init_method": cfg.init_method,
            "gaussian_cap_fraction": "1.0",
            "init_method_config": init_method_configs.get(cfg.init_method, "default"),
        },
        args=args,
        labels=labels,
        strategies=cfg.strategies,
        datasets=cfg.datasets,
    )


@dataclass
class ColorSimilarityScaleIncreaseAblationArgs:
    datasets: list[str] = field(
        default_factory=lambda: [
            "scannet++",
            "eval_on_train_set_scannet++",
            "mipnerf360",
            "tanksandtemples",
        ]
    )
    init_method: str = "laser_scan"
    da3_config: str = "floater_removal=True"
    strategies: list[str] = field(
        default_factory=lambda: ["IDHFRStrategy", "MCMCStrategy"]
    )
    factors: list[str] = field(default_factory=lambda: ["0.5"])


@section_config(ColorSimilarityScaleIncreaseAblationArgs)
def color_similarity_scale_increase_ablation(
    ctx: ResultsContext,
    format_options: FormatOptions,
    cfg: ColorSimilarityScaleIncreaseAblationArgs,
) -> None:
    init_method_configs = {
        "da3": cfg.da3_config,
    }
    factors = [None, *cfg.factors]
    args = [{"init.scale_color_dist_factor": factor} for factor in factors]
    labels = ["-" if factor is None else factor for factor in factors]

    common_args: dict[str, Any] = {
        "init_method": cfg.init_method,
        "init_method_config": init_method_configs.get(cfg.init_method, "default"),
        "dense_init.sampling": "uniform",
        "gaussian_cap_fraction": "1.0",
        "init.target_median_scale": None,
        "is_default_strategy_config": True,
    }
    if cfg.init_method == "laser_scan":
        common_args["init_size_matches_real_init"] = True
        common_args["dense_init.target_points_fraction"] = "1.0"

    _ablation(
        ctx,
        format_options,
        section_name="color_similarity_scale_increase_ablation",
        common_args=common_args,
        args=args,
        labels=labels,
        strategies=cfg.strategies,
        datasets=cfg.datasets,
    )


@dataclass
class DA3GSElementsAblationArgs:
    datasets: list[str] = field(default_factory=lambda: ALL_DATASETS_WITHOUT_ETH3D)
    strategies: list[str] = field(
        default_factory=lambda: [
            "DefaultWithGaussianCapStrategy",
            "MCMCStrategy",
            "IDHFRStrategy",
        ]
    )
    metrics: list[str] = field(
        default_factory=lambda: [
            "eval-all-test/psnr",
            "eval-all-test/ssim",
            "eval-all-test/lpips",
        ]
    )


@section_config(DA3GSElementsAblationArgs)
def da3_gs_components_ablation(
    ctx: ResultsContext,
    format_options: FormatOptions,
    cfg: DA3GSElementsAblationArgs,
) -> None:
    args = [
        {"is_default_init_config": True},
        {"splat_init.simulate_point_init": "True"},
        {"splat_init.init_scale_with_knn": "True"},
        {"splat_init.init_scale_isotropic_mean": "True"},
        {"splat_init.opacity_uniform_override": "0.1"},
        {"splat_init.rotation_noise_angle_std_deg": "45.0"},
        {"splat_init.color_noise_std": "0.5"},
    ]
    labels = [
        "Base",
        "Simulate point init",
        "kNN scale",
        "Isotropic scale",
        "uniform opacity",
        "Rotation noise 45°",
        "Color noise 0.5",
    ]

    common_args: dict[str, Any] = {
        "init_method": "da3",
        "init_method_config": "output_gaussians=True_max_num_images=150",
        "dense_init.sampling": "uniform",
        "gaussian_cap_fraction": "1.0",
        "init.target_median_scale": None,
        "init.scale_color_dist_factor": None,
        "dense_init.target_points_fraction": "1.0",
        "is_default_strategy_config": True,
    }

    _ablation_strategies_side_by_side(
        ctx,
        format_options,
        section_name="da3_gs_components_ablation",
        caption=r"Ablation on $\text{DA3}^\text{GS}$ initialization components.",
        common_args=common_args,
        args=args,
        labels=labels,
        strategies=cfg.strategies,
        datasets=cfg.datasets,
        metrics=cfg.metrics,
        delta=True,
    )


def idhfr_means_lr_ablation(ctx: ResultsContext, format_options: FormatOptions) -> None:
    _ablation(
        ctx,
        format_options,
        section_name="idhfr_means_lr_ablation",
        common_args={
            "init_method": "sfm",
            "gaussian_cap_fraction": "1.0",
        },
        args=[
            {
                "is_default_strategy_config": True,
            },
            {
                "means_lr_init": "4e-05",
            },
        ],
        labels=["Default LR", "LR 4e-05"],
        strategies=["IDHFRStrategy"],
        datasets=ALL_DATASETS,
    )


def da3_floater_removal_ablation(
    ctx: ResultsContext, format_options: FormatOptions
) -> None:
    _, per_dataset_means = _ablation(
        ctx,
        format_options,
        section_name="da3_floater_removal_ablation",
        common_args={
            "init_method": "da3",
            "gaussian_cap_fraction": "1.0",
            "is_default_strategy_config": True,
            "is_default_init_config": True,
            "dense_init.target_points_fraction": "1.0",
            "dense_init.include_sparse": False,
        },
        args=[
            {
                "init_method_config": "default",
            },
            {
                "init_method_config": "floater_removal=True",
            },
        ],
        labels=["No F.R.", "F.R."],
    )
    print("Per-dataset means for DA3 floater removal ablation:")
    for dataset, means_df in per_dataset_means.items():
        print(f"Dataset: {dataset}")
        print(means_df)
        print()


def da3_scene_selection_ablation(
    ctx: ResultsContext, format_options: FormatOptions
) -> None:
    """Improvement over SfM for DA3 / DA3 (GS) on all ScanNet++ scenes vs the
    subset that lies in the DA3 test set.

    One small table per ScanNet++ dataset (on- and off-trajectory) with a row per
    init method and, per scene set, the mean delta over the SfM baseline across
    strategies. A second table with raw metric values (not deltas over SfM) is
    also produced, along with printed aggregate statistics comparing the DA3 test
    scenes to the rest.
    """
    datasets = ["scannet++", "eval_on_train_set_scannet++"]
    strategies = ALL_STRATEGIES_EXCEPT_NO_D
    metrics = ["eval-all-test/psnr", "eval-all-test/ssim", "eval-all-test/lpips"]

    common_args = {
        "is_default_init_config": True,
        "is_default_strategy_config": True,
        "init.position_noise_std": "0.0",
        "gaussian_cap_fraction": "1.0",
        "dense_init.target_points_fraction": "1.0",
        "dense_init.include_sparse": False,
    }
    init_method_specs: dict[str, dict[str, Any]] = {
        "DA3": {"init_method": "da3", "init_method_config": "floater_removal=True"},
        r"$\text{DA3}^\text{G.S.}$": {
            "init_method": "da3",
            "init_method_config": "output_gaussians=True_max_num_images=150",
        },
    }
    scene_set_labels = ["All Scenes", "DA3 Test Scenes", "Excluded Scenes"]
    metric_delta_headers = {
        "eval-all-test/psnr": r"$\Delta$PSNR $\uparrow$",
        "eval-all-test/ssim": r"$\Delta$SSIM $\uparrow$",
        "eval-all-test/lpips": r"$\Delta$LPIPS $\downarrow$",
    }
    metric_raw_headers = {
        "eval-all-test/psnr": r"PSNR $\uparrow$",
        "eval-all-test/ssim": r"SSIM $\uparrow$",
        "eval-all-test/lpips": r"LPIPS $\downarrow$",
    }

    format_cell = make_cell_formatter(
        format_options.cell_type, rounding_per_metric=TABLE_ROUNDING_PER_METRIC
    )

    def col_id(scene_set: str, metric: str) -> str:
        return f"{scene_set}::{metric}"

    def make_header_block(metric_headers_map: dict[str, str]) -> str:
        metric_headers = " & ".join(
            rf"\textbf{{{metric_headers_map[metric]}}}" for metric in metrics
        )
        return (
            "& "
            + " & ".join(
                rf"\multicolumn{{{len(metrics)}}}"
                rf"{{{'c' if i == len(scene_set_labels) - 1 else 'c|'}}}"
                rf"{{\textbf{{{scene_set}}}}}"
                for i, scene_set in enumerate(scene_set_labels)
            )
            + r" \\"
            "\n"
            r"\textbf{Init} & "
            + " & ".join([metric_headers] * len(scene_set_labels))
            + r" \\"
        )

    row_labels = list(init_method_specs.keys())
    columns = [
        col_id(scene_set, metric)
        for scene_set in scene_set_labels
        for metric in metrics
    ]

    # Per-dataset raw + delta-over-SfM per-strategy frames and scene sets, used
    # for the printed per-dataset aggregate statistics below (DA3 test scenes vs
    # the rest). Raw frames give the reported mean metric; delta frames drive the
    # percentual comparison so it accounts for inherent per-scene difficulty.
    raw_dfs_per_init_per_dataset: dict[str, dict[str, list[pd.DataFrame]]] = {}
    delta_dfs_per_init_per_dataset: dict[str, dict[str, list[pd.DataFrame]]] = {}
    scene_sets_per_dataset: dict[str, tuple[set[str], set[str]]] = {}

    tables_per_dataset: dict[str, str] = {}
    tables_per_dataset_raw: dict[str, str] = {}
    for dataset in datasets:
        runs = ctx.runs_per_dataset[dataset].copy()
        da3_test_scenes = {
            f"{dataset}/{scene}" for scene in SCANNETPP_DA3_TEST_SCENE_SELECTION
        }
        excluded_scenes = {
            f"{dataset}/{scene}"
            for scene in SCANNETPP_SCENE_SELECTION
            if scene not in SCANNETPP_DA3_TEST_SCENE_SELECTION
        }
        scene_sets_per_dataset[dataset] = (da3_test_scenes, excluded_scenes)
        scene_sets: dict[str, set[str] | None] = {
            "All Scenes": None,
            "DA3 Test Scenes": da3_test_scenes,
            "Excluded Scenes": excluded_scenes,
        }

        # init label -> list of per-strategy per-scene raw / delta-over-SfM dfs.
        raw_dfs_per_init: dict[str, list[pd.DataFrame]] = {}
        delta_dfs_per_init: dict[str, list[pd.DataFrame]] = {}
        for init_label, spec in init_method_specs.items():
            per_strategy_raw: list[pd.DataFrame] = []
            per_strategy_delta: list[pd.DataFrame] = []
            for strategy in strategies:
                sfm_df = runs.get_per_scene_metrics_for_params(
                    {"init_group": "sfm_baseline", "strategy": strategy},
                    metrics=metrics,
                )
                init_df = runs.get_per_scene_metrics_for_params(
                    {**common_args, "strategy": strategy, **spec}, metrics=metrics
                )
                drop_scenes_not_present_in_all(sfm_df, init_df, debug_out=False)
                delta = init_df.copy()
                for metric in metrics:
                    delta[metric] = per_scene_metric_difference(
                        init_df[metric],
                        sfm_df[metric],
                        label=f"{dataset}/{init_label}/{strategy}",
                    )
                per_strategy_raw.append(init_df)
                per_strategy_delta.append(delta)
            raw_dfs_per_init[init_label] = per_strategy_raw
            delta_dfs_per_init[init_label] = per_strategy_delta
        raw_dfs_per_init_per_dataset[dataset] = raw_dfs_per_init
        delta_dfs_per_init_per_dataset[dataset] = delta_dfs_per_init

        def build_tabular(
            dfs_per_init: dict[str, list[pd.DataFrame]],
            metric_headers_map: dict[str, str],
            *,
            centered: bool,
            scene_sets=scene_sets,
        ) -> str:
            # ``centered`` picks the color scale: a diverging, zero-centered
            # scale for delta-over-SfM tables, or a per-metric min/max scale
            # (inverted for lower-is-better metrics) for raw-value tables.
            color_table = pd.DataFrame(index=row_labels, columns=columns, dtype=float)
            text_table = pd.DataFrame(index=row_labels, columns=columns, dtype=object)

            for metric in metrics:
                invert = metric in LOWER_IS_BETTER_METRICS
                rounding = TABLE_ROUNDING_PER_METRIC[metric]
                cell_means: dict[tuple[str, str], float] = {}
                for init_label in row_labels:
                    for scene_set in scene_set_labels:
                        scene_filter = scene_sets[scene_set]
                        strategy_dfs = [
                            df
                            if scene_filter is None
                            else df.loc[df.index.intersection(scene_filter)]
                            for df in dfs_per_init[init_label]
                        ]
                        cell = cell_data_across_strategies(metric, strategy_dfs)
                        rounded_mean = round(cell.mean, rounding)
                        cell_means[(init_label, scene_set)] = rounded_mean
                        text_table.loc[init_label, col_id(scene_set, metric)] = (
                            format_cell(cell)
                        )

                values = np.array(list(cell_means.values()), dtype=float)
                if centered:
                    multiplier = -1.0 if invert else 1.0
                    metric_max_abs = (
                        float(np.nanmax(np.abs(values))) if values.size else 0.0
                    )
                    for (init_label, scene_set), mean in cell_means.items():
                        normalized = (
                            multiplier * mean / metric_max_abs
                            if metric_max_abs
                            else 0.0
                        )
                        color_table.loc[init_label, col_id(scene_set, metric)] = (
                            normalized
                        )
                else:
                    vmin = float(np.nanmin(values)) if values.size else 0.0
                    vmax = float(np.nanmax(values)) if values.size else 0.0
                    pad = (vmax - vmin) * 0.1 if vmax > vmin else 0.0
                    lo, span = vmin - pad, (vmax + pad) - (vmin - pad)
                    for (init_label, scene_set), mean in cell_means.items():
                        normalized = (mean - lo) / span if span else 0.5
                        color_table.loc[init_label, col_id(scene_set, metric)] = (
                            (1.0 - normalized) if invert else normalized
                        )

            if centered:
                max_abs = float(color_table.abs().max().max())
                color_range = (-1.2 * max_abs, 1.2 * max_abs)
                cmap = DIVERGING_CMAP
            else:
                color_range = (0.0, 1.0)
                cmap = VALUE_CMAP

            return tabular_colored_from_numeric_with_custom_text(
                top_left_label="",
                table=color_table,
                text_table=text_table,
                hide_nulls=False,
                column_format="l|" + "|".join(["ccc"] * len(scene_set_labels)),
                header_block=make_header_block(metric_headers_map),
                color_range=color_range,
                color_intensity=format_options.color_intensity,
                force_black_text=format_options.force_black_text,
                cmap=cmap,
            )

        tabular = build_tabular(delta_dfs_per_init, metric_delta_headers, centered=True)
        tables_per_dataset[dataset] = wrap_tabulars_as_float(
            [tabular],
            DATASET_NAMES.get(dataset, dataset),
            f"da3_scene_selection_{name_to_path(dataset, allow_subdirs=False)}",
            format_options,
        )

        tabular_raw = build_tabular(
            raw_dfs_per_init, metric_raw_headers, centered=False
        )
        tables_per_dataset_raw[dataset] = wrap_tabulars_as_float(
            [tabular_raw],
            DATASET_NAMES.get(dataset, dataset),
            f"da3_scene_selection_raw_{name_to_path(dataset, allow_subdirs=False)}",
            format_options,
        )

    path = ctx.output_helper.get_table_path("da3_scene_selection_ablation")
    write_file(
        path,
        finalize_per_dataset_tables(
            tables_per_dataset,
            format_options,
            combined_caption=(
                "Improvement over SfM initialization for DA3 and "
                r"$\text{DA3}^\text{G.S.}$ when evaluated over all ScanNet++ scenes "
                "versus only the scenes in the DA3 test set. Values are the mean "
                "delta over the SfM baseline across strategies."
            ),
            combined_label="da3_scene_selection_ablation",
        ),
    )
    print(f"Saved DA3 scene selection ablation table to {path}")

    path_raw = ctx.output_helper.get_table_path("da3_scene_selection_ablation_raw")
    write_file(
        path_raw,
        finalize_per_dataset_tables(
            tables_per_dataset_raw,
            format_options,
            combined_caption=(
                "Raw metric values (not deltas over SfM) for DA3 and "
                r"$\text{DA3}^\text{G.S.}$ when evaluated over all ScanNet++ scenes "
                "versus only the scenes in the DA3 test set. Values are the mean "
                "over strategies."
            ),
            combined_label="da3_scene_selection_ablation_raw",
        ),
    )
    print(f"Saved DA3 scene selection raw values table to {path_raw}")

    print(
        "\n===== DA3 scene selection: DA3 test scenes vs the rest (aggregate stats) ====="
    )

    def print_aggregate_stats(
        title: str,
        raw_dfs_per_init: dict[str, list[pd.DataFrame]],
        delta_dfs_per_init: dict[str, list[pd.DataFrame]],
        da3_test_scenes: set[str],
        excluded_scenes: set[str],
    ) -> None:
        print(f"===== {title} =====")
        for init_label in init_method_specs:
            print(f"--- {init_label} ---")
            combined = pd.concat(raw_dfs_per_init[init_label])
            for metric in metrics:
                overall_mean = float(series_mean_frame_mean(combined[metric]))

                # Difference of the improvement over SfM on the DA3 test scenes
                # vs the rest, computed separately per strategy run (on the
                # delta-over-SfM frames to account for inherent per-scene
                # difficulty) so we can report the spread across those runs.
                test_vs_rest_deltas: list[float] = []
                for df in delta_dfs_per_init[init_label]:
                    test_scenes = df.index.intersection(da3_test_scenes)
                    rest_scenes = df.index.intersection(excluded_scenes)
                    if test_scenes.empty or rest_scenes.empty:
                        continue
                    test_mean = float(
                        series_mean_frame_mean(df.loc[test_scenes, metric])
                    )
                    rest_mean = float(
                        series_mean_frame_mean(df.loc[rest_scenes, metric])
                    )
                    if np.isnan(test_mean) or np.isnan(rest_mean):
                        continue
                    test_vs_rest_deltas.append(test_mean - rest_mean)

                pretty = METRIC_NAME_MAP.get(metric, metric)
                rounding = TABLE_ROUNDING_PER_METRIC.get(metric, 3)
                if test_vs_rest_deltas:
                    print(
                        f"  {pretty}: mean={overall_mean:.3f} | Δ(test-rest) of SfM-deltas -> "
                        f"min={min(test_vs_rest_deltas):.{rounding}f}, "
                        f"max={max(test_vs_rest_deltas):.{rounding}f}, "
                        f"median={float(np.median(test_vs_rest_deltas)):.{rounding}f}, "
                        f"mean={float(np.mean(test_vs_rest_deltas)):.{rounding}f}"
                    )
                else:
                    print(
                        f"  {pretty}: mean={overall_mean:.3f} | Δ(test-rest) of SfM-deltas -> N/A"
                    )

    for dataset in datasets:
        da3_test_scenes, excluded_scenes = scene_sets_per_dataset[dataset]
        print_aggregate_stats(
            DATASET_NAMES.get(dataset, dataset),
            raw_dfs_per_init_per_dataset[dataset],
            delta_dfs_per_init_per_dataset[dataset],
            da3_test_scenes,
            excluded_scenes,
        )

    # Combined over both datasets: pool the per-(dataset, strategy) frames and
    # union the scene sets so the spread reflects every dataset/strategy run.
    combined_raw = {
        init_label: [
            df
            for dataset in datasets
            for df in raw_dfs_per_init_per_dataset[dataset][init_label]
        ]
        for init_label in init_method_specs
    }
    combined_delta = {
        init_label: [
            df
            for dataset in datasets
            for df in delta_dfs_per_init_per_dataset[dataset][init_label]
        ]
        for init_label in init_method_specs
    }
    combined_test_scenes = set().union(
        *(scene_sets_per_dataset[dataset][0] for dataset in datasets)
    )
    combined_excluded_scenes = set().union(
        *(scene_sets_per_dataset[dataset][1] for dataset in datasets)
    )
    print_aggregate_stats(
        "All Datasets",
        combined_raw,
        combined_delta,
        combined_test_scenes,
        combined_excluded_scenes,
    )


def laser_scan_hybrid_init_ablation(
    ctx: ResultsContext, format_options: FormatOptions
) -> None:
    """Improvement from hybrid laser-scan init (adding sparse SfM points) across
    strategies and init sizes.

    Layout mirrors ``laser_scan_tables`` (strategies in rows, init sizes in
    columns); each cell is the mean and (across-scene) std of the per-scene
    metric delta between ``dense_init.include_sparse`` True and False.
    """
    common_args = {
        "is_default_strategy_config": True,
        "is_default_init_config": True,
        "init.position_noise_std": "0.0",
        "gaussian_cap_fraction": "1.0",
        "init_method": "laser_scan",
        "init_size_matches_gmax": True,
    }
    metrics = ["eval-all-test/psnr", "eval-all-test/ssim", "eval-all-test/lpips"]
    strategies = ALL_STRATEGIES_EXCEPT_NO_D
    size_fractions = ["0.5", "0.75", "1.0"]
    size_labels = {
        fraction: gmax_fraction_label(fraction) for fraction in size_fractions
    }

    tables: dict[str, str] = {}
    for dataset in LASER_DATASETS:
        print("Dataset:", dataset)
        runs = ctx.runs_per_dataset[dataset].copy()

        # strategy label -> size label -> per-scene hybrid-minus-plain delta df.
        data: dict[str, dict[str, pd.DataFrame]] = {}
        try:
            for strategy in strategies:
                strat_name = STRATEGY_NAMES[strategy]
                for fraction in size_fractions:
                    base = {
                        **common_args,
                        "strategy": strategy,
                        "dense_init.target_points_fraction": fraction,
                    }
                    hybrid = runs.get_per_scene_metrics_for_params(
                        {**base, "dense_init.include_sparse": True}, metrics=metrics
                    )
                    plain = runs.get_per_scene_metrics_for_params(
                        {**base, "dense_init.include_sparse": False}, metrics=metrics
                    )
                    drop_scenes_not_present_in_all(hybrid, plain, debug_out=False)
                    delta = hybrid.copy()
                    for metric in metrics:
                        delta[metric] = per_scene_metric_difference(
                            hybrid[metric],
                            plain[metric],
                            label=f"{dataset}/{strat_name}/{size_labels[fraction]}",
                        )
                    data.setdefault(strat_name, {})[size_labels[fraction]] = delta
        except ValueError as error:
            ansiesc_print(
                f"!!!!! Skipping dataset '{dataset}' for hybrid-init ablation: {error}",
                ANSIEscapes.RED,
            )
            continue

        drop_uncommon_scenes(data, debug_out=False)

        # Per-metric mean/median delta from hybrid init for this dataset, pooled
        # over all strategies and init sizes.
        print(f"Hybrid-init delta over all strategies and init sizes ({dataset}):")
        for metric in metrics:
            values = np.concatenate(
                [
                    np.asarray(delta[metric].loc[scene], dtype=float).ravel()
                    for size_dict in data.values()
                    for delta in size_dict.values()
                    for scene in delta.index
                ]
            )
            pretty = METRIC_NAME_MAP.get(metric, metric)
            rounding = TABLE_ROUNDING_PER_METRIC.get(metric, 3)
            print(
                f"  {pretty}: mean={float(np.nanmean(values)):.{rounding}f}, "
                f"median={float(np.nanmedian(values)):.{rounding}f}"
            )

        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"laser_scan_hybrid_init_ablation_{dataset}",
            metrics=metrics,
            column_order=[size_labels[fraction] for fraction in size_fractions],
            row_order=[STRATEGY_NAMES[strategy] for strategy in strategies],
            format_args=format_options,
            horizontal_cols_label="Init size",
            cmap=DIVERGING_CMAP,
            center_zero=True,
        )

    path = ctx.output_helper.get_table_path("laser_scan_hybrid_init_ablation")
    write_file(
        path,
        finalize_per_dataset_tables(
            tables,
            format_options,
            combined_caption=(
                "Improvement from hybrid laser-scan initialization (adding sparse "
                "SfM points) across strategies and initialization sizes. Each cell "
                "is the mean and across-scene std of the per-scene metric delta."
            ),
            combined_label="laser_scan_hybrid_init_ablation",
        ),
    )
    print(f"Saved laser scan hybrid init ablation table to {path}")


def dense_init_half_size_ablation(
    ctx: ResultsContext, format_options: FormatOptions
) -> None:
    COL_MONODEPTH = "M. D."
    COL_DA3 = "DA3"
    COL_DA3_GS_INIT = "$\\text{DA3}^\\text{G.S.}$"
    COL_LASER = "Laser"
    INIT_METHOD_LABELS = {
        "monodepth": COL_MONODEPTH,
        "da3": COL_DA3,
        "da3_gs": COL_DA3_GS_INIT,
        "laser_scan": COL_LASER,
    }

    init_method_cfgs = {
        "da3": "floater_removal=True",
        "da3_gs": "output_gaussians=True_max_num_images=150",
    }
    rows: list[pd.DataFrame] = []
    for init_method in ["monodepth", "da3", "da3_gs"]:  # "laser_scan"]:
        print(f"========== Init method: {init_method} ==========")
        comb_row, _ = _ablation(
            ctx,
            format_options,
            section_name=f"dense_init_half_size_ablation_{init_method}",
            common_args={
                "init_method": init_method.rstrip("_gs"),
                "init_method_config": init_method_cfgs.get(init_method, "default"),
                "gaussian_cap_fraction": "1.0",
                "is_default_strategy_config": True,
                "is_default_init_config": True,
                "dense_init.include_sparse": (init_method == "laser_scan"),
            },
            args=[
                {
                    "dense_init.target_points_fraction": "0.5",
                },
                {
                    "dense_init.target_points_fraction": "1.0",
                },
            ],
            labels=["Half Size", "Full Size"],
            strategies=[
                "DefaultWithGaussianCapStrategy",
                "MCMCStrategy",
                "IDHFRStrategy",
            ],
            summary_only=True,
            datasets=(
                ALL_DATASETS_WITHOUT_ETH3D
                if init_method != "laser_scan"
                else LASER_DATASETS_WITHOUT_ETH3D
            ),
            metrics=[
                "eval-all-test/psnr",
                "eval-all-test/ssim",
                "eval-all-test/lpips",
            ],
        )
        # add the init method value as a column
        rows += [
            comb_row.assign(Init=INIT_METHOD_LABELS[init_method]).set_index("Init")
        ]

    table = pd.concat(rows, axis=0)
    table = table.reset_index().set_index("Init")
    print("Overall means:")
    print(table)

    # save to colored latex table
    metrics = [
        "eval-all-test/psnr",
        "eval-all-test/ssim",
        "eval-all-test/lpips",
    ]
    labels = ["Half Size", "Full Size"]
    row_labels = list(table.index)

    def col_id(metric: str, label: str) -> str:
        return f"{METRIC_NAME_MAP[metric]} ({label})"

    ordered_ids = [col_id(metric, label) for metric in metrics for label in labels]
    color_table = pd.DataFrame(index=row_labels, columns=ordered_ids, dtype=float)
    text_table = pd.DataFrame(index=row_labels, columns=ordered_ids, dtype=object)

    for metric in metrics:
        rounding = TABLE_ROUNDING_PER_METRIC.get(metric, 2)
        invert = metric in LOWER_IS_BETTER_METRICS
        cols = [col_id(metric, label) for label in labels]
        values = table[cols].apply(pd.to_numeric, errors="coerce")
        vmin = float(values.min().min())
        vmax = float(values.max().max())
        pad = (vmax - vmin) * 0.1
        lo, span = vmin - pad, (vmax + pad) - (vmin - pad)
        for col in cols:
            for row in row_labels:
                val = float(table.loc[row, col])
                normalized = (val - lo) / span if span else 0.5
                color_table.loc[row, col] = (1.0 - normalized) if invert else normalized
                text_table.loc[row, col] = f"{val:.{rounding}f}"

    metric_headers = [
        rf"\multicolumn{{{len(labels)}}}{{{'c' if i == len(metrics) - 1 else 'c|'}}}"
        rf"{{\textbf{{{METRIC_PRETTY_NAMES[metric]}}}}}"
        for i, metric in enumerate(metrics)
    ]
    sub_headers = [label for _ in metrics for label in labels]
    header_block = (
        "& " + " & ".join(metric_headers) + r" \\"
        "\n"
        "Init & " + " & ".join(sub_headers) + r" \\"
    )
    column_format = "l|" + "|".join(["c" * len(labels)] * len(metrics))

    tabular = tabular_colored_from_numeric_with_custom_text(
        top_left_label="",
        table=color_table,
        text_table=text_table,
        column_format=column_format,
        header_block=header_block,
        color_range=(0.0, 1.0),
        color_intensity=format_options.color_intensity,
        force_black_text=format_options.force_black_text,
        cmap=DIVERGING_CMAP,
    )

    path = ctx.output_helper.get_table_path("dense_init_half_size_ablation")
    write_file(
        path,
        wrap_tabulars_as_float(
            [tabular],
            "Ablation on the effect of using half the number of initial points for "
            "dense initialization. Averaged across all scenes except ETH3D.",
            "dense_init_half_size_ablation_compact",
            replace(format_options, combine_datasets_as_subtables=False),
        ),
    )
    print(f"Saved dense_init_half_size_ablation table to {path}")


SectionFn = Callable[..., None]

SECTION_FUNCTIONS: list[SectionFn] = [
    laser_scan_tables,
    laser_scan_analysis_plots,
    improvement_tables,
    practical_tables,
    practical_analysis_plots,
    init_times,
    # Ablations:
    noise_resiliency,
    da3_gs_components_ablation,
    da3_floater_removal_ablation,
    da3_scene_selection_ablation,
    laser_scan_hybrid_init_ablation,
    edgs_scale_increase_ablation,
    idhfr_means_lr_ablation,
    gaussian_cap_ablation,
    init_scale_ablation,
    color_similarity_scale_increase_ablation,
    dense_init_half_size_ablation,
]

SECTION_FUNCTION_NAMES: list[str] = [fn.__name__ for fn in SECTION_FUNCTIONS]

DEFAULT_SECTIONS = [
    laser_scan_tables.__name__,
]

DEFAULT_SECTION_FORMAT_OVERRIDES = {
    laser_scan_tables.__name__: FormatOptions(
        cell_type=TableCellType.mean,
        metrics_layout=MetricsLayout.horizontal,
        resizebox=True,
    ),
    improvement_tables.__name__: FormatOptions(
        cell_type=TableCellType.std,
        resizebox=True,
        metrics_layout=MetricsLayout.horizontal,
        tabcolsep_fraction=2.0,
    ),
    practical_tables.__name__: FormatOptions(
        cell_type=TableCellType.mean,
        metrics_layout=MetricsLayout.horizontal,
        resizebox=True,
    ),
    noise_resiliency.__name__: FormatOptions(
        cell_type=TableCellType.mean,
        metrics_layout=MetricsLayout.horizontal,
        table_env_override="table",
        resizebox=True,
    ),
    da3_gs_components_ablation.__name__: FormatOptions(
        cell_type=TableCellType.mean,
        metrics_layout=MetricsLayout.horizontal,
        table_env_override="table",
        resizebox=True,
    ),
    da3_scene_selection_ablation.__name__: FormatOptions(
        cell_type=TableCellType.scene_std,
        metrics_layout=MetricsLayout.horizontal,
        table_env_override="table",
        resizebox=True,
    ),
    laser_scan_hybrid_init_ablation.__name__: FormatOptions(
        cell_type=TableCellType.mean,
        metrics_layout=MetricsLayout.horizontal,
        table_env_override="table",
        resizebox=True,
    ),
    gaussian_cap_ablation.__name__: FormatOptions(
        cell_type=TableCellType.mean,
        metrics_layout=MetricsLayout.horizontal,
        table_env_override="table",
        resizebox=True,
    ),
    dense_init_half_size_ablation.__name__: FormatOptions(
        cell_type=TableCellType.mean,
        metrics_layout=MetricsLayout.horizontal,
        table_env_override="table",
        resizebox=False,
        color_intensity=0.0,
    ),
}

# Sanity check
for override_name, _ in DEFAULT_SECTION_FORMAT_OVERRIDES.items():
    if override_name not in SECTION_FUNCTION_NAMES:
        raise ValueError(
            f"Default format override specified for section '{override_name}' which is not in the list of section functions."
        )

if not TYPE_CHECKING:
    SectionsChoiceList = list[Literal[tuple(SECTION_FUNCTION_NAMES) + ("all",)]]  # type: ignore
else:
    SectionsChoiceList = list[str]


def _default_format_map() -> dict[str, FormatOptions]:
    return {
        fn.__name__: DEFAULT_SECTION_FORMAT_OVERRIDES.get(fn.__name__, FormatOptions())
        for fn in SECTION_FUNCTIONS
    }


@dataclass
class CommonArgs(BaseArgs):
    # Subset of notebook sections to execute, or "all" to run all sections.
    sections: SectionsChoiceList = field(default_factory=lambda: DEFAULT_SECTIONS)
    # Default format options to use for all sections.
    format_default: FormatOptions = field(default_factory=FormatOptions)
    # Format options to use for specific sections, overriding the default options.
    format: dict[str, FormatOptions] = field(default_factory=_default_format_map)


# ``Args`` is built dynamically so that every section registered via
# ``@section_config`` contributes a top-level field named exactly after the
# section function. tyro then exposes its options as
# ``<section_name>.<field>=<value>`` (e.g. ``nanogs_simplify_test.per-scene-graphs=False``).
Args = make_dataclass(
    "Args",
    [
        (section_name, config_cls, field(default_factory=config_cls))
        for section_name, config_cls in SECTION_CONFIG_TYPES.items()
    ],
    bases=(CommonArgs,),
)


def main() -> None:
    configure_logging()
    args: CommonArgs = tyro.cli(Args)
    ctx = ResultsContext.create(args)

    selected_sections = (
        list(SECTION_FUNCTION_NAMES) if "all" in args.sections else args.sections
    )
    for section_name in selected_sections:
        ansiesc_print(
            "______________________________________________", ANSIEscapes.BLUE
        )
        ansiesc_print(f"===== Running section: {section_name} =====", ANSIEscapes.BLUE)
        section_fn = next(fn for fn in SECTION_FUNCTIONS if fn.__name__ == section_name)
        format_options = args.format.get(section_name, args.format_default)
        if section_name in SECTION_CONFIG_TYPES:
            section_fn(ctx, format_options, getattr(args, section_name))
        else:
            section_fn(ctx, format_options)


if __name__ == "__main__":
    main()
