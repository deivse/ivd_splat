from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, make_dataclass, replace
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar, cast

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
from results_scripts.param_conversions import PARAM_CONVERSIONS
from results_scripts.plots import (
    grouped_per_metric_barplots_for_each_config,
    grouped_per_metric_line_charts_for_each_config,
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
    DATASET_NAMES,
    DENSE_INIT_METRICS,
    LASER_DATASETS,
    LASER_DATASETS_WITHOUT_ETH3D,
    LINE_CHART_PLOT_STARTS,
    LOWER_IS_BETTER_METRICS,
    METRIC_NAME_MAP,
    METRIC_PRETTY_NAMES,
    PLOT_RANGES_PER_METRIC,
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
    finalize_per_dataset_tables,
    make_aggregated_metric_table,
    make_latex_table_for_metrics,
    wrap_tabulars_as_float,
)
from results_scripts.base import get_cache_dir, load_or_download_runs
from results_scripts.utils import (
    OutputDirHelper,
    fraction_name,
    load_json,
    name_to_path,
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
                / f"{name_to_path(dataset, allow_subdirs=False)}_{args.max_eval_iter or 'all'}.pkl"
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


def series_mean_frame_mean(df: pd.DataFrame | pd.Series) -> pd.Series:
    return df.map(lambda x: np.array(x).mean()).mean()


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
    COL_0_5 = "$0.5\\mathcal{G}_\\mathit{max}$"
    COL_0_75 = "$.75\\mathcal{G}_\\mathit{max}$"
    COL_1_0 = "$1.0\\mathcal{G}_\\mathit{max}$"
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

    tables: dict[str, str] = {}
    for dataset in LASER_DATASETS:
        runs = ctx.runs_per_dataset[dataset].copy()

        data: dict[str, dict[str, pd.DataFrame]] = {}

        for strategy in ALL_STRATEGIES_EXCEPT_NO_D:
            data.setdefault(STRATEGY_NAMES[strategy], {})[COL_SFM] = (
                runs.get_per_scene_metrics_for_params(
                    {
                        "init_group": "sfm_baseline",
                        "strategy": strategy,
                    }
                )
            )
            data.setdefault(STRATEGY_NAMES[strategy], {})[COL_AS_SFM] = (
                runs.get_per_scene_metrics_for_params(
                    {
                        **common_args,
                        "strategy": strategy,
                        "init_method": "laser_scan",
                        "init_size_matches_sfm": True,
                    },
                    metrics=DENSE_INIT_METRICS,
                )
            )

        for strategy in ALL_STRATEGIES:
            for size_fraction in ["0.5", "0.75", "1.0"]:
                args = {
                    **common_args,
                    "strategy": strategy,
                    "dense_init.target_points_fraction": size_fraction,
                    "init_method": "laser_scan",
                    "init_size_matches_gmax": True,
                    "dense_init.include_sparse": (
                        cfg.include_sparse and dataset != "eth3d"
                    ),
                }
                result = runs.get_per_scene_metrics_for_params(
                    args,
                    metrics=DENSE_INIT_METRICS,
                )
                data.setdefault(STRATEGY_NAMES[strategy], {})[
                    COL_PER_FRACTION[size_fraction]
                ] = result

        drop_scenes_not_present_in_all(
            *[df for values in data.values() for df in values.values()], debug_out=False
        )

        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"laser_scan_main_{dataset}",
            column_order=COL_ORDER,
            row_order=[STRATEGY_NAMES[strategy] for strategy in ALL_STRATEGIES],
            format_args=format_options,
        )

    path = ctx.output_helper.get_table_path("laser_scan")
    write_file(
        path,
        finalize_per_dataset_tables(
            tables,
            format_options,
            combined_caption=(
                "Laser scan initialization performance strategies and initialization sizes."
            ),
            combined_label="laser_scan_main",
        ),
    )
    print(f"Saved main Laser Scan table to {path}")


InitMethodId = Literal[
    "sfm", "laser_scan", "monodepth", "edgs", "edgs_sh", "da3", "da3_no_fr", "da3_gs"
]


@dataclass
class ImprovementTablesArgs:
    init_methods: list[InitMethodId] = field(
        default_factory=lambda: ["laser_scan", "monodepth"]
    )
    datasets: list[str] = field(
        default_factory=lambda: list(ALL_DATASETS_WITHOUT_ETH3D)
    )
    # If set, append a summary column group aggregating the improvement across all
    # included init methods (per dataset, over the methods applicable to it). The
    # central value is the mean/median of the per-method improvements and the
    # reported spread is the std of those per-method values across init methods.
    include_summary: bool = False
    # If set, show ONLY the summary column group (still computed over all included
    # init methods), hiding the per-method columns.
    summary_only: bool = False
    # How to aggregate the per-method improvements into the summary central value.
    summary_type: Literal["mean", "median"] = "mean"
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

    # Per init-method: column label, query params (merged with common_args and the
    # strategy), which metrics to load, and whether it only has data for GT datasets.
    # ``gt_only`` methods (e.g. laser scan) are silently skipped for non-GT datasets.
    init_method_specs: dict[str, dict[str, Any]] = {
        "laser_scan": {
            "label": r"$0.75G_\mathit{max}$ Laser",
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
            "params": {"init_method": "monodepth", **common_non_laser_args},
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

    SUMMARY_COL = "summary"
    init_labels: dict[str, str] = {
        init: init_method_specs[init]["label"] for init in cfg.init_methods
    }
    if cfg.include_summary or cfg.summary_only:
        init_labels[SUMMARY_COL] = cfg.summary_type.capitalize()

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
        if cfg.summary_only:
            display_init_methods = [SUMMARY_COL]
        else:
            display_init_methods = active_init_methods + (
                [SUMMARY_COL] if cfg.include_summary else []
            )

        columns = [
            col_id(init, metric) for init in display_init_methods for metric in metrics
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

            for init in display_init_methods:
                for strat_name in strat_names:
                    if init == SUMMARY_COL:
                        # Aggregate across all init methods applicable to this
                        # dataset: take each method's mean improvement, then
                        # summarize those per-method values. The central value is
                        # the mean/median across methods and the reported spread
                        # is the std of the per-method values across methods.
                        per_method_means = [
                            CellData.for_metric(
                                (
                                    improvement_data[m][strat_name][metric]
                                    - sfm_data[strat_name][metric]
                                ).to_frame(),
                                metric,
                            ).mean
                            for m in active_init_methods
                        ]
                        values = np.array(per_method_means, dtype=float)
                        central = (
                            float(np.median(values))
                            if cfg.summary_type == "median"
                            else float(values.mean())
                        )
                        spread = float(values.std())
                        cell = CellData(
                            metric_id=metric,
                            mean=central,
                            stddev=spread,
                            min=float(values.min()),
                            max=float(values.max()),
                            scene_stddev=spread,  # dirty hack, but ok.
                            mean_measurement_count=len(values),
                        )
                    else:
                        improvement = (
                            improvement_data[init][strat_name][metric]
                            - sfm_data[strat_name][metric]
                        )
                        cell = CellData.for_metric(improvement.to_frame(), metric)
                    rounded_mean = round(cell.mean, rounding)
                    cell_means[(init, strat_name)] = rounded_mean
                    metric_max_abs = max(metric_max_abs, abs(rounded_mean))
                    text_table.loc[strat_name, col_id(init, metric)] = format_cell(cell)

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
            for init in display_init_methods:
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
                    column_format="l|" + "|".join(["ccc"] * len(display_init_methods)),
                    header_block=side_by_side_header(display_init_methods),
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
        data: dict[str, dict[str, pd.DataFrame]] = {}

        for noise in noise_levels:
            for strategy in ALL_STRATEGIES_EXCEPT_NO_D:
                args = {
                    **common_args,
                    "strategy": strategy,
                    "init.position_noise_std": noise,
                }
                data.setdefault(STRATEGY_NAMES[strategy], {})[noise_name(noise)] = (
                    runs.get_per_scene_metrics_for_params(args)
                )

        all_dfs = [df for values in data.values() for df in values.values()]
        drop_scenes_not_present_in_all(*all_dfs)

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
                "and position noise levels using Laser Scan init with $0.5\\mathcal{G}_\\mathit{max}$ initial points."
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
    datasets: list[str] = field(default_factory=lambda: ALL_DATASETS_WITHOUT_ETH3D)
    include_sparse_for_all: Literal["yes", "no", "both"] = "no"
    include_half_init_size_for_all: Literal["yes", "no", "both"] = "no"
    include_sparse_for_laser: bool = True


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

    tables = {}
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
            for strategy in cfg.strategies:
                if strategy == "DefaultWithoutADCStrategy":
                    continue
                r = runs.get_per_scene_metrics_for_params(
                    {
                        "init_group": "sfm_baseline",
                        "strategy": strategy,
                        **_strat_arg_overrides(strategy),
                    }
                )
                if r.empty:
                    print(
                        f"Warning: No SfM baseline runs found for strategy {strategy} on dataset {dataset}."
                    )
                else:
                    data.setdefault(STRATEGY_NAMES[strategy], {})[COL_SFM] = r

        for strategy in cfg.strategies:
            params_per_col: dict[str, Any] = {
                COL_EDGS: {
                    "strategy": strategy,
                    "init_method": "edgs",
                    "init_method_config": "default",
                    "splat_init.increase_scale_with_fewer_splats": True,
                },
                COL_EDGS_FULL_SH_INIT: {
                    "strategy": strategy,
                    "init_method": "edgs",
                    "init_method_config": "full_sh_init=True",
                    "splat_init.increase_scale_with_fewer_splats": True,
                },
                COL_MONODEPTH: {
                    "strategy": strategy,
                    "init_method": "monodepth",
                },
                COL_DA3_NO_FLOATER_REMOVAL: {
                    "strategy": strategy,
                    "init_method": "da3",
                    "init_method_config": "default",
                },
                COL_DA3: {
                    "strategy": strategy,
                    "init_method": "da3",
                    "init_method_config": "floater_removal=True",
                },
                COL_DA3_GS_INIT: {
                    "strategy": strategy,
                    "init_method": "da3",
                    "init_method_config": "output_gaussians=True_max_num_images=150",
                },
            }

            strat_name = STRATEGY_NAMES[strategy]
            for col in PRACTICAL_COLS:
                try:
                    data.setdefault(strat_name, {})[col] = (
                        runs.get_per_scene_metrics_for_params(
                            {
                                **common_args,
                                **params_per_col[col],
                                "dense_init.target_points_fraction": (
                                    "0.5"
                                    if cfg.include_half_init_size_for_all == "yes"
                                    else "1.0"
                                ),
                                "dense_init.include_sparse": (
                                    cfg.include_sparse_for_all == "yes"
                                ),
                                **_strat_arg_overrides(strategy),
                            }
                        )
                    )
                    if (
                        cfg.include_sparse_for_all == "both"
                        and col in INCLUDE_SPARSE_COLS
                    ):
                        # Also fetch the sparse-only version for these methods, which support it.
                        data.setdefault(strat_name, {})[
                            f"$\\text{{{col}}}^{{{MARK_SPARSE}}}$"
                        ] = runs.get_per_scene_metrics_for_params(
                            {
                                **common_args,
                                **params_per_col[col],
                                "dense_init.include_sparse": True,
                                **_strat_arg_overrides(strategy),
                            }
                        )
                    if (
                        cfg.include_half_init_size_for_all == "both"
                        and col in HALF_INIT_SIZE_COLS
                    ):
                        # Also fetch the half-size version for these methods, which support it.
                        data.setdefault(strat_name, {})[
                            f"$\\text{{{col}}}^{{{MARK_HALF}}}$"
                        ] = runs.get_per_scene_metrics_for_params(
                            {
                                **common_args,
                                **params_per_col[col],
                                "dense_init.target_points_fraction": "0.5",
                                "dense_init.include_sparse": (
                                    cfg.include_sparse_for_all == "yes"
                                    and col in INCLUDE_SPARSE_COLS
                                ),
                                **_strat_arg_overrides(strategy),
                            }
                        )

                except Exception as e:
                    print(
                        f"Error processing {col} for strategy {strat_name} on dataset {dataset}: {e}"
                    )

            if dataset in LASER_DATASETS and "laser_scan" in cfg.init_methods:
                data.setdefault(strat_name, {})[COL_LASER] = (
                    runs.get_per_scene_metrics_for_params(
                        {
                            **common_args,
                            "strategy": strategy,
                            "init_method": "laser_scan",
                            "init_size_matches_real_init": True,
                            "dense_init.target_points_fraction": (
                                "0.5"
                                if cfg.include_half_init_size_for_all == "yes"
                                else "1.0"
                            ),
                            "dense_init.include_sparse": (
                                cfg.include_sparse_for_all == "yes"
                            )
                            or (
                                cfg.include_sparse_for_laser
                                and cfg.include_sparse_for_all != "both"
                            ),
                            **_strat_arg_overrides(strategy),
                        }
                    )
                )
                if cfg.include_sparse_for_all == "both":
                    data.setdefault(strat_name, {})[
                        f"$\\text{{{COL_LASER}}}^{{{MARK_SPARSE}}}$"
                    ] = runs.get_per_scene_metrics_for_params(
                        {
                            **common_args,
                            "strategy": strategy,
                            "init_method": "laser_scan",
                            "init_size_matches_real_init": True,
                            "dense_init.target_points_fraction": "1.0",
                            "dense_init.include_sparse": True,
                            **_strat_arg_overrides(strategy),
                        }
                    )
                if cfg.include_half_init_size_for_all == "both":
                    data.setdefault(strat_name, {})[
                        f"$\\text{{{COL_LASER}}}^{{{MARK_HALF}}}$"
                    ] = runs.get_per_scene_metrics_for_params(
                        {
                            **common_args,
                            "strategy": strategy,
                            "init_method": "laser_scan",
                            "init_size_matches_real_init": True,
                            "dense_init.target_points_fraction": "0.5",
                            "dense_init.include_sparse": (
                                cfg.include_sparse_for_all == "yes"
                            )
                            or cfg.include_sparse_for_laser,
                            **_strat_arg_overrides(strategy),
                        }
                    )

        all_dfs = [
            df for strategy_dict in data.values() for df in strategy_dict.values()
        ]
        try:
            drop_scenes_not_present_in_all(*all_dfs)
        except Exception as e:
            print(f"Scene mismatch error for dataset {dataset}: {e}")
            continue

        for strategy, col_dict in data.items():
            for col, df in col_dict.items():
                if col in all_datasets_data.setdefault(strategy, {}):
                    all_datasets_data[strategy][col] = pd.concat(
                        [all_datasets_data[strategy][col], df],
                        axis=0,
                        ignore_index=True,
                    )
                else:
                    all_datasets_data[strategy][col] = df
        col_order = ALL_COLS.copy()
        if cfg.include_sparse_for_all == "both":
            # Add the sparse-only versions of the applicable methods after their
            # main columns.
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

        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"practical_main_{dataset}",
            column_order=col_order,
            row_order=[STRATEGY_NAMES[strategy] for strategy in cfg.strategies],
            format_args=format_options,
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

    if (
        cfg.include_sparse_for_all != "yes"
        or cfg.include_half_init_size_for_all != "yes"
    ):
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


def generate_gaussian_cap_fraction_gt(ctx: ResultsContext) -> None:
    init_method_args: dict[str, dict[str, object]] = {
        "laser_scan": {
            "init_method": "laser_scan",
            "init_size_matches_gmax": True,
            "dense_init.target_points_fraction": "0.5",
        },
        "sfm": {
            "init_method": "sfm",
        },
    }

    for init_method in ["laser_scan", "sfm"]:
        print("========== Init method:", init_method, "==========")
        section_subdir = f"gaussian_cap_fractions/gt/{init_method}"
        for dataset in LASER_DATASETS:
            print("Dataset:", dataset)
            runs = ctx.runs_per_dataset[dataset].copy()
            data: dict[str, dict[str, pd.DataFrame]] = {}

            for strategy in [
                "DefaultWithGaussianCapStrategy",
                "INRIAStrategy",
                "MCMCStrategy",
                "IDHFRStrategy",
                "RevDGSStrategy",
            ]:
                strategy_common = {
                    "is_default_strategy_config": True,
                    "strategy": strategy,
                    "init.position_noise_std": "0.0",
                    **init_method_args[init_method],
                }
                for cap_fraction in ["0.75", "1.0", "1.25"]:
                    metrics_for_cap = runs.get_per_scene_metrics_for_params(
                        {**strategy_common, "gaussian_cap_fraction": cap_fraction}
                    )
                    data.setdefault(STRATEGY_NAMES[strategy], {})[
                        fraction_name(cap_fraction)
                    ] = metrics_for_cap

            all_dataframes = [
                df for strategy_dict in data.values() for df in strategy_dict.values()
            ]
            drop_scenes_not_present_in_all(*all_dataframes)

            data_means = {
                strategy: {
                    cap_fraction: df.mean() for cap_fraction, df in cap_dict.items()
                }
                for strategy, cap_dict in data.items()
            }
            fig, _, _ = grouped_per_metric_barplots_for_each_config(
                cast(dict[str, dict[str, pd.DataFrame]], data_means),
                metrics_to_plot=[
                    "eval-all-test/psnr",
                    "eval-all-test/ssim",
                    "eval-all-test/lpips",
                ],
                plot_limits_per_metric=PLOT_RANGES_PER_METRIC[dataset],
                label_all_bars=False,
                columns=3,
                figsize=(12, 1.5),
                legend_y_offset=0.15,
                padding_factor=0.5,
                show_legend=False,
                font_scale=1.5,
                y_ticks_pad_scale=0.5,
            )
            save_figure_svg(
                fig,
                ctx.output_helper.get_graph_path(
                    section_subdir, f"{dataset}_cap_fraction_gt"
                ),
            )
            plt.close(fig)


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
) -> pd.DataFrame:
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
    for dataset in datasets:
        runs = ctx.runs_per_dataset[dataset].copy()

        # row (strategy) -> column (variant label) -> per-scene metrics dataframe
        data: dict[str, dict[str, pd.DataFrame]] = {}
        for strategy in strategies:
            strategy_label = STRATEGY_NAMES.get(strategy, strategy)
            for label, args_i in zip(labels, args):
                data.setdefault(strategy_label, {})[label] = (
                    runs.get_per_scene_metrics_for_params(
                        {"strategy": strategy, **common_args, **args_i},
                        metrics=metrics,
                    )
                )

        drop_scenes_not_present_in_all(
            *[df for columns in data.values() for df in columns.values()],
            debug_out=False,
        )

        for i, label in enumerate(labels):
            for strategy_label in data:
                all_datasets[i].append(data[strategy_label][label])

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
        return pd.DataFrame([comb_row]).set_index("-")

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
    return pd.DataFrame([comb_row]).set_index("-")


def _cell_data_across_strategies(
    metric: str, strategy_dfs: list[pd.DataFrame]
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


def _ablation_aggregate_strategies(
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
    top_left_label: str = "",
    delta: bool = False,
) -> None:
    """Ablation table with metrics in columns and one row per arg/label pair.

    Unlike ``_ablation`` (which keeps one row per strategy), every cell here
    aggregates across all ``strategies``: the mean is the mean of the per-strategy
    scene means, and the reported spread (std/min/max) is computed across
    strategies. Per-dataset results are emitted as subtables or separate tables
    according to ``format_options``.

    The first arg/label pair is the reference row (color map center); when
    ``delta`` is set the remaining rows show signed deltas relative to it.
    """
    num_variants = len(args)
    if (num_variants != len(labels)) or (num_variants < 2):
        raise ValueError(
            f"Number of args ({num_variants}) must match number of labels ({len(labels)}) and be at least 2."
        )

    tables: dict[str, str] = {}
    for dataset in datasets:
        runs = ctx.runs_per_dataset[dataset].copy()

        # variant label -> per-strategy per-scene metrics dataframes
        per_variant_strategy_dfs: dict[str, list[pd.DataFrame]] = {
            label: [
                runs.get_per_scene_metrics_for_params(
                    {"strategy": strategy, **common_args, **args_i},
                    metrics=metrics,
                )
                for strategy in strategies
            ]
            for label, args_i in zip(labels, args)
        }

        drop_scenes_not_present_in_all(
            *[df for dfs in per_variant_strategy_dfs.values() for df in dfs],
            debug_out=False,
        )

        # variant label -> metric -> CellData aggregated across strategies
        cell_data: dict[str, dict[str, CellData]] = {
            label: {
                metric: _cell_data_across_strategies(metric, strategy_dfs)
                for metric in metrics
            }
            for label, strategy_dfs in per_variant_strategy_dfs.items()
        }

        tables[dataset] = make_aggregated_metric_table(
            cell_data=cell_data,
            metrics=metrics,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"{section_name}_{dataset}",
            format_args=format_options,
            row_order=labels,
            top_left_label=top_left_label,
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

        # variant label (row) -> strategy label (column) -> per-scene metrics df
        data: dict[str, dict[str, pd.DataFrame]] = {}
        for label, args_i in zip(labels, args):
            for strategy, strategy_label in zip(strategies, strategy_labels):
                data.setdefault(label, {})[strategy_label] = (
                    runs.get_per_scene_metrics_for_params(
                        {"strategy": strategy, **common_args, **args_i},
                        metrics=metrics,
                    )
                )

        drop_scenes_not_present_in_all(
            *[df for columns in data.values() for df in columns.values()],
            debug_out=False,
        )

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
    _ablation(
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
    )


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
def da3_gs_elements_ablation(
    ctx: ResultsContext,
    format_options: FormatOptions,
    cfg: DA3GSElementsAblationArgs,
) -> None:
    args = [
        {"is_default_init_config": True},
        {"splat_init.init_scale_with_knn": "True"},
        {"splat_init.init_scale_isotropic_mean": "True"},
        {"splat_init.opacity_uniform_override": "0.1"},
        {"splat_init.rotation_noise_angle_std_deg": "45.0"},
        {"splat_init.color_noise_std": "0.5"},
    ]
    labels = [
        "Base",
        "kNN scale",
        "Isotropic scale",
        "uniform opacity",
        "Rotation noise 45°",
        "Color noise 0.5"
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
        section_name="da3_gs_elements_ablation",
        caption=r"Ablation on $\text{DA3}^\text{GS}$ initializationcomponents.",
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
    )


def da3_floater_removal_ablation(
    ctx: ResultsContext, format_options: FormatOptions
) -> None:
    _ablation(
        ctx,
        format_options,
        section_name="da3_floater_removal_ablation",
        common_args={
            "init_method": "da3",
            "gaussian_cap_fraction": "1.0",
            "is_default_strategy_config": True,
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
        comb_row = _ablation(
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
    improvement_tables,
    practical_tables,
    init_times,
    # Ablations:
    noise_resiliency,
    da3_gs_elements_ablation,
    da3_floater_removal_ablation,
    edgs_scale_increase_ablation,
    idhfr_means_lr_ablation,
    generate_gaussian_cap_fraction_gt,
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
        cell_type=TableCellType.scene_std,
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
    da3_gs_elements_ablation.__name__: FormatOptions(
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
