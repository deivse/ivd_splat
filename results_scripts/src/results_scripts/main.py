from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, cast

from eval_scripts.common.ansi_escapes import ANSIEscapes, ansiesc_print
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from results_scripts.base import (
    DEFAULT_TABLE_METRICS,
    RunsInfo,
    drop_scenes_not_present_in_all,
    load_and_prepare_dataset_runs,
    load_init_method_runs,
)
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
    GT_DATASETS,
    GT_DATASETS_WITHOUT_ETH3D,
    LINE_CHART_PLOT_STARTS,
    METRIC_NAME_MAP,
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
    finalize_per_dataset_tables,
    make_latex_table_for_metrics,
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


def laser_scan_graphs(ctx: ResultsContext, latex_options: FormatOptions) -> None:
    common_args = {
        "is_default_strategy_config": True,
        "init.position_noise_std": "0.0",
        "gaussian_cap_fraction": "1.0",
    }
    section_subdir = "gt_init_diff_strategies_and_sizes/main"

    for dataset in GT_DATASETS:
        print("Dataset:", dataset)
        runs = ctx.runs_per_dataset[dataset].copy()

        data: dict[str, dict[str | float, pd.DataFrame]] = {}
        sfm_data: dict[str, pd.DataFrame] = {}

        for strategy in ALL_STRATEGIES_EXCEPT_NO_D:
            sfm_data[strategy] = runs.get_per_scene_metrics_for_params(
                {
                    "init_group": "sfm_baseline",
                    "strategy": strategy,
                }
            )

        for strategy in ALL_STRATEGIES_EXCEPT_NO_D:
            args = {
                **common_args,
                "strategy": strategy,
                "init_method": "laser_scan",
                "init_size_matches_sfm": True,
            }
            result = runs.get_per_scene_metrics_for_params(
                args, metrics=DENSE_INIT_METRICS
            )
            data.setdefault(STRATEGY_NAMES[strategy], {})["as_sfm"] = result

        for strategy in ALL_STRATEGIES:
            for size_fraction in ["0.5", "0.75", "1.0"]:
                args = {
                    **common_args,
                    "strategy": strategy,
                    "dense_init.target_points_fraction": size_fraction,
                    "init_method": "laser_scan",
                    "init_size_matches_gmax": True,
                }
                result = runs.get_per_scene_metrics_for_params(
                    args, metrics=DENSE_INIT_METRICS
                )
                data.setdefault(STRATEGY_NAMES[strategy], {})[size_fraction] = result

        all_dfs = list(sfm_data.values()) + [
            df for values in data.values() for df in values.values()
        ]
        drop_scenes_not_present_in_all(*all_dfs)

        data_with_num_points: dict[str, dict[float, pd.DataFrame]] = {}
        for strategy, df_per_fraction in data.items():
            data_with_num_points[strategy] = {}
            for fraction, df in df_per_fraction.items():
                df["dense_init.target_num_points"] = (
                    df["dense_init.target_num_points"]
                    .map(lambda x: x if isinstance(x, int) else np.unique(x).item())
                    .astype(float)
                )
                df["dense_init.target_points_fraction"] = (
                    df["dense_init.target_points_fraction"]
                    .map(lambda x: x if isinstance(x, int) else np.unique(x).item())
                    .astype(float)
                )
                assert (
                    df["dense_init.target_points_fraction"]
                    == df["dense_init.target_points_fraction"].iloc[0]
                ).all()
                num_points = (
                    df["dense_init.target_num_points"]
                    * df["dense_init.target_points_fraction"].iloc[0]
                )
                data_with_num_points[strategy][num_points.mean()] = df

        summarized_data = {
            strategy: {
                num_points: series_mean_frame_mean(df) for num_points, df in dfs.items()
            }
            for strategy, dfs in data_with_num_points.items()
        }

        plot_ranges_per_metric_per_dataset = {
            "default": {
                "eval-all-test/psnr": 2.5,
                "eval-all-test/ssim": 0.03,
                "eval-all-test/lpips": 0.06,
                "train/total-train-time": 10,
            },
            "eth3d": {
                "eval-all-test/psnr": 5,
                "eval-all-test/ssim": 0.12,
                "eval-all-test/lpips": 0.2,
                "train/total-train-time": 9,
            },
        }
        plot_limits_per_metric = compute_plot_limits(
            LINE_CHART_PLOT_STARTS,
            plot_ranges_per_metric_per_dataset.get(
                dataset, plot_ranges_per_metric_per_dataset["default"]
            ),
            dataset,
        )

        fig_downscale = 1.15
        fig, axes = grouped_per_metric_line_charts_for_each_config(
            cast(dict[str, dict[float, pd.DataFrame]], summarized_data),
            extra_data_lines=cast(
                dict[str, pd.DataFrame],
                {
                    STRATEGY_NAMES[strategy]: series_mean_frame_mean(df)
                    for strategy, df in sfm_data.items()
                },
            ),
            metrics_to_plot=[
                "eval-all-test/psnr",
                "eval-all-test/ssim",
                "eval-all-test/lpips",
                "train/total-train-time",
            ],
            columns=4,
            plot_limits_per_metric=plot_limits_per_metric,
            xlabel="Mean initialization size",
            figsize=(18 / fig_downscale, 4 / fig_downscale),
            remove_y_axis_labels=True,
            show_legend=True,
            legend_y_offset=0.1,
            fontsize_scale=1.65,
        )
        save_figure_svg(
            fig,
            ctx.output_helper.get_graph_path(section_subdir, f"{dataset}_line_metrics"),
        )

        save_line_chart_legend(ctx, section_subdir, axes, fig_downscale)
        plt.close(fig)


def laser_scan_tables(ctx: ResultsContext, format_options: FormatOptions) -> None:
    common_args = {
        "is_default_strategy_config": True,
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

    tables: dict[str, str] = {}
    for dataset in GT_DATASETS:
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


def improvement_tables(
    ctx: ResultsContext,
    format_options: FormatOptions,
) -> None:
    common_args = {
        "is_default_strategy_config": True,
        "init.position_noise_std": "0.0",
        "gaussian_cap_fraction": "1.0",
    }

    metrics = [
        "eval-all-test/psnr",
        "eval-all-test/ssim",
        "eval-all-test/lpips",
    ]
    strat_names = [STRATEGY_NAMES[strategy] for strategy in ALL_STRATEGIES_EXCEPT_NO_D]
    init_methods = ["laser", "monodepth"]

    def col_id(init: str, metric: str) -> str:
        return f"{init}_{metric}"

    columns = [col_id(init, metric) for init in init_methods for metric in metrics]

    metric_headers = " & ".join(
        rf"\textbf{{{label}}}"
        for label in (
            r"$\Delta$PSNR $\uparrow$",
            r"$\Delta$SSIM $\uparrow$",
            r"$\Delta$LPIPS $\downarrow$",
        )
    )
    init_labels = {
        "laser": r"$0.75G_\mathit{max}$ Laser",
        "monodepth": "Monodepth",
    }
    # Side-by-side header: two metric groups under a \multicolumn per init method.
    side_by_side_header = (
        r"& \multicolumn{3}{c|}{" + init_labels["laser"] + r"} "
        r"& \multicolumn{3}{c}{" + init_labels["monodepth"] + r"} \\"
        "\n"
        rf"\textbf{{Strategy}} & {metric_headers} & {metric_headers} \\"
    )

    format_cell = make_cell_formatter(
        format_options.cell_type,
        rounding_per_metric=TABLE_ROUNDING_PER_METRIC,
    )

    tables_per_dataset: dict[str, str] = {}
    for dataset in GT_DATASETS_WITHOUT_ETH3D:
        runs = ctx.runs_per_dataset[dataset].copy()

        sfm_data: dict[str, pd.DataFrame] = {}
        improvement_data: dict[str, dict[str, pd.DataFrame]] = {
            init: {} for init in init_methods
        }
        for strategy in ALL_STRATEGIES_EXCEPT_NO_D:
            strat_name = STRATEGY_NAMES[strategy]
            sfm_data[strat_name] = runs.get_per_scene_metrics_for_params(
                {"init_group": "sfm_baseline", "strategy": strategy}
            )
            improvement_data["laser"][strat_name] = (
                runs.get_per_scene_metrics_for_params(
                    {
                        **common_args,
                        "strategy": strategy,
                        "dense_init.target_points_fraction": "0.75",
                        "init_method": "laser_scan",
                        "init_size_matches_gmax": True,
                    },
                    metrics=DENSE_INIT_METRICS,
                )
            )
            improvement_data["monodepth"][strat_name] = (
                runs.get_per_scene_metrics_for_params(
                    {
                        **common_args,
                        "strategy": strategy,
                        "init_method": "monodepth",
                    },
                    metrics=DEFAULT_TABLE_METRICS,
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
            # LPIPS is lower-is-better: flip its sign so positive == improvement
            # (warm color) consistently with the other metrics.
            multiplier = -1.0 if metric.lower().endswith("lpips") else 1.0
            rounding = TABLE_ROUNDING_PER_METRIC[metric]
            cell_means: dict[tuple[str, str], float] = {}
            metric_max_abs = 0.0

            for init in init_methods:
                for strat_name in strat_names:
                    improvement = (
                        improvement_data[init][strat_name][metric]
                        - sfm_data[strat_name][metric]
                    )
                    cell = CellData.for_metric(improvement.to_frame(), metric)
                    rounded_mean = round(cell.mean, rounding)
                    cell_means[(init, strat_name)] = rounded_mean
                    metric_max_abs = max(metric_max_abs, abs(rounded_mean))
                    text_table.loc[strat_name, col_id(init, metric)] = format_cell(cell)

            # Normalize colors per metric across both init methods. This keeps the
            # coloring scale consistent across metrics (which differ in magnitude)
            # and identical between the two init-method column groups, so colors
            # remain comparable across the laser/monodepth tables.
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
            # Stack the two init methods as sections of a SINGLE tabular, so the
            # column widths stay aligned (separate tabulars + \resizebox would
            # each scale independently and misalign).
            section_tabulars: list[str] = []
            for init in init_methods:
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
                    column_format="l|ccc|ccc",
                    header_block=side_by_side_header,
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
    }

    noise_levels = ["0.0", "0.01", "0.1"]

    def noise_name(noise: str | float) -> str:
        return str(noise)

    tables: dict[str, str] = {}
    for dataset in GT_DATASETS:
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


def practical_tables(ctx: ResultsContext, format_options: FormatOptions) -> None:
    COL_SFM = "SfM"
    COL_EDGS = "$\\text{EDGS}^*$"
    COL_EDGS_FULL_SH_INIT = "$\\text{EDGS}$"
    COL_MONODEPTH = "M. Depth"
    COL_DA3_NO_FLOATER_REMOVAL = "$\\text{DA3}^\\text{No F.R.}$"
    COL_DA3 = "DA3"
    COL_DA3_GS_INIT = "$\\text{DA3}^\\text{G.S.}$"
    COL_LASER = "Laser"

    PRACTICAL_COLS = [
        COL_EDGS,
        # COL_EDGS_FULL_SH_INIT,
        COL_MONODEPTH,
        # COL_DA3_NO_FLOATER_REMOVAL,
        COL_DA3,
        COL_DA3_GS_INIT,
    ]
    COLUMN_ORDER = [
        COL_SFM,
        *PRACTICAL_COLS,
        COL_LASER,
    ]

    PRACTICAL_STRATS = ALL_STRATEGIES
    # Dict:  strategy -> column -> per-scene dataframe across all datasets for all metrics with lists of values per eval iter in cells.
    all_datasets_data: dict[str, dict[str, pd.DataFrame]] = {}

    tables = {}
    for dataset in ALL_DATASETS_WITHOUT_ETH3D:
        print("Dataset:", dataset)
        runs = ctx.runs_per_dataset[dataset].copy()

        common_args = {
            "is_default_strategy_config": True,
            "gaussian_cap_fraction": "1.0",
            "init.position_noise_std": "0.0",
        }

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

        for strategy in PRACTICAL_STRATS:
            params_per_col = {
                COL_EDGS: {
                    **common_args,
                    "strategy": strategy,
                    "init_method": "edgs",
                    "init_method_config": "default",
                    "splat_init.increase_scale_with_fewer_splats": True,
                },
                COL_EDGS_FULL_SH_INIT: {
                    **common_args,
                    "strategy": strategy,
                    "init_method": "edgs",
                    "init_method_config": "full_sh_init=True",
                    "splat_init.increase_scale_with_fewer_splats": True,
                },
                COL_MONODEPTH: {
                    **common_args,
                    "strategy": strategy,
                    "init_method": "monodepth",
                },
                COL_DA3_NO_FLOATER_REMOVAL: {
                    **common_args,
                    "strategy": strategy,
                    "init_method": "da3",
                    "init_method_config": "default",
                },
                COL_DA3: {
                    **common_args,
                    "strategy": strategy,
                    "init_method": "da3",
                    "init_method_config": "floater_removal=True",
                },
                COL_DA3_GS_INIT: {
                    **common_args,
                    "strategy": strategy,
                    "init_method": "da3",
                    "init_method_config": "output_gaussians=True_max_num_images=150",
                },
            }

            strat_name = STRATEGY_NAMES[strategy]
            for col in PRACTICAL_COLS:
                try:
                    data.setdefault(strat_name, {})[col] = (
                        runs.get_per_scene_metrics_for_params(params_per_col[col])
                    )
                except Exception as e:
                    print(
                        f"Error processing {col} for strategy {strat_name} on dataset {dataset}: {e}"
                    )

            if dataset in GT_DATASETS:
                data.setdefault(strat_name, {})[COL_LASER] = (
                    runs.get_per_scene_metrics_for_params(
                        {
                            **common_args,
                            "strategy": strategy,
                            "init_method": "laser_scan",
                            "init_size_matches_real_init": True,
                            "dense_init.target_points_fraction": "1.0",
                        }
                    )
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

        tables[dataset] = make_latex_table_for_metrics(
            data=data,
            latex_caption=DATASET_NAMES[dataset],
            latex_label=f"practical_main_{dataset}",
            column_order=COLUMN_ORDER,
            row_order=[STRATEGY_NAMES[strategy] for strategy in ALL_STRATEGIES],
            format_args=format_options,
        )

    path = ctx.output_helper.get_table_path("practical_init")
    write_file(
        path,
        finalize_per_dataset_tables(
            tables,
            format_options,
            combined_caption=(
                "Practical initialization performance across densificatiojn strategies."
            ),
            combined_label="practical_main",
        ),
    )
    print(f"Saved Practical Initialization table to {path}")

    format_args_times = deepcopy(format_options)
    format_args_times.cell_type = TableCellType.mean
    format_args_times.metrics_layout = MetricsLayout.vertical
    # Not applicable
    format_args_times.combine_datasets_as_subtables = False
    init_times_table = make_latex_table_for_metrics(
        data=all_datasets_data,
        latex_caption="Mean training times across all datasets.",
        latex_label="practical_train_times",
        column_order=COLUMN_ORDER,
        row_order=[STRATEGY_NAMES[strategy] for strategy in ALL_STRATEGIES],
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
        for dataset in GT_DATASETS:
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
    common_args: dict[str, object],
    args: list[dict[str, object]],
    labels: list[str],
    strategies=ALL_STRATEGIES,
    metrics=DEFAULT_TABLE_METRICS,
) -> None:
    rows: list[dict[str, float | str]] = []

    num_variants = len(args)
    if (num_variants != len(labels)) or (num_variants < 2):
        raise ValueError(
            f"Number of args ({num_variants}) must match number of labels ({len(labels)}) and be at least 2."
        )

    all_datasets = [[] for _ in range(num_variants)]
    for dataset in ALL_DATASETS_WITHOUT_ETH3D:
        runs = ctx.runs_per_dataset[dataset].copy()
        all_for_dataset: list[list[pd.DataFrame]] = [[] for _ in range(num_variants)]

        for strategy in strategies:
            strategy_args = {
                "strategy": strategy,
            }
            for i, args_i in enumerate(args):
                all_for_dataset[i].append(
                    runs.get_per_scene_metrics_for_params(
                        {**strategy_args, **common_args, **args_i}
                    )
                )

        drop_scenes_not_present_in_all(
            *[x for sublist in all_for_dataset for x in sublist]
        )

        all_for_dataset_dfs = [
            pd.concat(dfs, axis=0, ignore_index=True) for dfs in all_for_dataset
        ]
        for all_datasets_list, dfs in zip(all_datasets, all_for_dataset_dfs):
            all_datasets_list.append(dfs)

        row: dict[str, float | str] = {"Dataset": dataset}
        for metric in metrics:
            pretty_name = METRIC_NAME_MAP.get(metric, metric)
            for i, df in enumerate(all_for_dataset_dfs):
                row[f"{pretty_name} ({labels[i]})"] = float(
                    series_mean_frame_mean(df[metric])
                )
        rows.append(row)

    summary_df = pd.DataFrame(rows).set_index("Dataset")
    print("Per dataset means:")
    print(summary_df)

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


def edgs_scale_increase_ablation(
    ctx: ResultsContext, format_options: FormatOptions
) -> None:
    _ablation(
        ctx,
        format_options,
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


def idhfr_means_lr_ablation(ctx: ResultsContext, format_options: FormatOptions) -> None:
    # rows: list[dict[str, float | str]] = []

    # for dataset in ALL_DATASETS:
    #     runs = ctx.runs_per_dataset[dataset].copy()
    #     strategy_args = {
    #         "init_method": "sfm",
    #         "gaussian_cap_fraction": "1.0",
    #         **get_default_strategy_args("IDHFRStrategy", dataset),
    #     }

    #     base = runs.get_per_scene_metrics_for_params(
    #         {
    #             **strategy_args,
    #             "is_default_strategy_config": True,
    #         }
    #     )
    #     lr_change = runs.get_per_scene_metrics_for_params(
    #         {**strategy_args, "means_lr_init": "4e-05"}
    #     )
    #     drop_scenes_not_present_in_all(base, lr_change)

    #     row: dict[str, float | str] = {"Dataset": dataset}
    #     for metric in [
    #         "eval-all-test/psnr",
    #         "eval-all-test/ssim",
    #         "eval-all-test/lpips",
    #     ]:
    #         pretty_name = METRIC_NAME_MAP.get(metric, metric)
    #         row[pretty_name] = pd.to_numeric(base[metric], errors="raise").mean()
    #         row[f"{pretty_name} (LR)"] = pd.to_numeric(
    #             lr_change[metric],
    #             errors="raise",
    #         ).mean()
    #     rows.append(row)

    # summary_df = pd.DataFrame(rows).set_index("Dataset")
    # print(summary_df)
    # print(summary_df.to_latex(float_format="%.3f"))
    _ablation(
        ctx,
        format_options,
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
    # rows: list[dict[str, float | str]] = []

    # all_datasets_da3 = []
    # all_datasets_da3_fr = []
    # for dataset in ALL_DATASETS_WITHOUT_ETH3D:
    #     runs = ctx.runs_per_dataset[dataset].copy()
    #     all_for_dataset: list[pd.DataFrame] = []
    #     all_for_dataset_floater_removal: list[pd.DataFrame] = []

    #     for strategy in ALL_STRATEGIES:
    #         strategy_args = {
    #             "init_method": "da3",
    #             "gaussian_cap_fraction": "1.0",
    #             "strategy": strategy,
    #             "is_default_strategy_config": True,
    #         }
    #         all_for_dataset.append(
    #             runs.get_per_scene_metrics_for_params(
    #                 {
    #                     **strategy_args,
    #                     "init_method_config": "default",
    #                 }
    #             )
    #         )
    #         all_for_dataset_floater_removal.append(
    #             runs.get_per_scene_metrics_for_params(
    #                 {
    #                     **strategy_args,
    #                     "init_method_config": "floater_removal=True",
    #                 }
    #             )
    #         )

    #     drop_scenes_not_present_in_all(
    #         *all_for_dataset,
    #         *all_for_dataset_floater_removal,
    #     )

    #     da3_all = pd.concat(all_for_dataset, axis=0, ignore_index=True)
    #     da3_fr_all = pd.concat(
    #         all_for_dataset_floater_removal,
    #         axis=0,
    #         ignore_index=True,
    #     )
    #     all_datasets_da3.append(da3_all)
    #     all_datasets_da3_fr.append(da3_fr_all)

    #     row: dict[str, float | str] = {"Dataset": dataset}
    #     for metric in [
    #         "eval-all-test/psnr",
    #         "eval-all-test/ssim",
    #         "eval-all-test/lpips",
    #     ]:
    #         pretty_name = METRIC_NAME_MAP.get(metric, metric)
    #         row[pretty_name] = float(series_mean_frame_mean(da3_all[metric]))
    #         row[f"{pretty_name} (Floater Removal)"] = float(
    #             series_mean_frame_mean(da3_fr_all[metric])
    #         )
    #     rows.append(row)

    _ablation(
        ctx,
        format_options,
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


def _nanogs_per_scene_graph(
    ctx: ResultsContext,
    dataset: str = "scannet++",
    strategies: list[str] = [
        "DefaultWithGaussianCapStrategy",
        "MCMCStrategy",
    ],
    metrics: list[str] = [
        "eval-all-test/psnr",
        "eval-all-test/ssim",
        "eval-all-test/lpips",
        "train/total-train-time",
    ],
) -> None:
    """Per-scene bar chart comparing results with and without NanoGS simplification.

    Bars for the with/without NanoGS configs are drawn side by side for each scene,
    averaging across the given strategies and runs.
    """
    section_subdir = "nanogs_simplify"
    common_args = {"init_method": "laser_scan"}
    config_args: dict[str, dict[str, object]] = {
        "Without NanoGS": {
            "is_default_strategy_config": True,
            "gaussian_cap_fraction": "1.0",
            "dense_init.target_points_fraction": "01.",
        },
        "With NanoGS": {"nanogs_simplify_iter": "500"},
    }

    runs = ctx.runs_per_dataset[dataset].copy()
    # config label -> per-scene scalar metrics (averaged across strategies and runs)
    per_config: dict[str, pd.DataFrame] = {}
    for label, cfg in config_args.items():
        per_strategy_dfs = [
            runs.get_per_scene_metrics_for_params(
                {**common_args, "strategy": strategy, **cfg},
                metrics=metrics,
            )
            for strategy in strategies
        ]
        drop_scenes_not_present_in_all(*per_strategy_dfs, debug_out=False)
        scalar_dfs = [df.map(lambda x: np.array(x).mean()) for df in per_strategy_dfs]
        per_config[label] = cast(
            pd.DataFrame, pd.concat(scalar_dfs).groupby(level=0).mean()
        )

    drop_scenes_not_present_in_all(*per_config.values(), debug_out=False)
    scenes = sorted(next(iter(per_config.values())).index)
    scene_labels = [scene.split("/")[-1] for scene in scenes]

    config_labels = list(per_config.keys())
    colors = plt.cm.tab10.colors  # type: ignore
    x = np.arange(len(scenes))
    bar_width = 0.8 / len(config_labels)

    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(max(8.0, len(scenes) * 0.9), 3.0 * len(metrics)),
    )
    axes = np.atleast_1d(axes)
    for ax, metric in zip(axes, metrics):
        for i, label in enumerate(config_labels):
            values = per_config[label].loc[scenes, metric].to_numpy(dtype=float)
            ax.bar(
                x + i * bar_width,
                values,
                width=bar_width,
                label=label,
                color=colors[i],
            )
        ax.set_ylabel(METRIC_NAME_MAP.get(metric, metric))
        ax.set_xticks(x + bar_width * (len(config_labels) - 1) / 2)
        ax.set_xticklabels(scene_labels, rotation=45, ha="right")
    axes[0].legend()
    fig.suptitle(
        f"Per-scene NanoGS simplification comparison on {DATASET_NAMES[dataset]}"
    )
    fig.tight_layout()
    save_figure_svg(
        fig,
        ctx.output_helper.get_graph_path(section_subdir, f"{dataset}_per_scene"),
    )
    plt.close(fig)


def nanogs_simplify_test(ctx: ResultsContext, format_options: FormatOptions) -> None:
    _ablation(
        ctx,
        format_options,
        common_args={
            "init_method": "monodepth",
        },
        args=[
            {
                "is_default_strategy_config": True,
            },
            {"nanogs_simplify_iter": "500"},
        ],
        labels=["Default", "NanoGS Simplify"],
        strategies=["DefaultWithGaussianCapStrategy", "MCMCStrategy"],
        metrics=[
            "eval-all-test/psnr",
            "eval-all-test/ssim",
            "eval-all-test/lpips",
            "train/total-train-time",
        ],
    )

    for dataset in ALL_DATASETS_WITHOUT_ETH3D:
        try:
            _nanogs_per_scene_graph(ctx, dataset)
        except Exception as e:
            print(f"Error generating NanoGS per-scene graph for {dataset}: {e}")


SectionFn = Callable[[ResultsContext, FormatOptions], None]

SECTION_FUNCTIONS: list[SectionFn] = [
    laser_scan_graphs,
    laser_scan_tables,
    improvement_tables,
    practical_tables,
    init_times,
    # Ablations:
    noise_resiliency,
    da3_floater_removal_ablation,
    edgs_scale_increase_ablation,
    idhfr_means_lr_ablation,
    nanogs_simplify_test,
    generate_gaussian_cap_fraction_gt,
]

SECTION_FUNCTION_NAMES: list[str] = [fn.__name__ for fn in SECTION_FUNCTIONS]

DEFAULT_SECTIONS = [
    laser_scan_tables.__name__,
]

DEFAULT_SECTION_FORMAT_OVERRRIDES = {
    laser_scan_tables.__name__: FormatOptions(
        cell_type=TableCellType.mean,
        metrics_layout=MetricsLayout.horizontal,
        resizebox=True,
    ),
    improvement_tables.__name__: FormatOptions(
        cell_type=TableCellType.scene_std,
        resizebox=True,
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
}

# Sanity check
for override_name, _ in DEFAULT_SECTION_FORMAT_OVERRRIDES.items():
    if override_name not in SECTION_FUNCTION_NAMES:
        raise ValueError(
            f"Default format override specified for section '{override_name}' which is not in the list of section functions."
        )

if not TYPE_CHECKING:
    SectionsChoiceList = list[Literal[tuple(SECTION_FUNCTION_NAMES) + ("all",)]]  # type: ignore
else:
    SectionsChoiceList = list[str]


@dataclass
class Args(BaseArgs):
    # Subset of notebook sections to execute, or "all" to run all sections.
    sections: SectionsChoiceList = field(default_factory=lambda: DEFAULT_SECTIONS)
    # Default format options to use for all sections.
    format_default: FormatOptions = field(default_factory=FormatOptions)
    # Format options to use for specific sections, overriding the default options.
    format: dict[str, FormatOptions] = field(
        default_factory=lambda: {
            fn.__name__: DEFAULT_SECTION_FORMAT_OVERRRIDES.get(
                fn.__name__, FormatOptions()
            )
            for fn in SECTION_FUNCTIONS
        }
    )


def main() -> None:
    configure_logging()
    args = tyro.cli(Args)
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
        section_fn(
            ctx,
            args.format.get(section_name, args.format_default),
        )


if __name__ == "__main__":
    main()
