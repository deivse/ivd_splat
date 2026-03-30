from typing import Iterable, Literal
import typing

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd

from .base import METRIC_PRETTY_NAMES, TABLE_METRICS


def format_number_compactly(val: float):
    if abs(val) < 1:
        return f"{val:.3f}"[1:]
    if abs(val) < 10:
        return f"{val:.2f}"
    if abs(val) < 1000:
        return f"{val:.1f}"
    suffixes = ["", "K", "M", "B", "T"]
    suffix_index = 0
    while val >= 1000 and suffix_index < len(suffixes) - 1:
        val /= 1000
        suffix_index += 1
    num_digits = 1 if val < 100 else 0
    return f"{val:.{num_digits}f}{suffixes[suffix_index]}"


def per_metric_barplots_for_each_config(
    means: Iterable[pd.DataFrame],
    labels: Iterable[str],
    columns: int = 2,
    figsize=(12, 8),
    cmap: plt.Colormap = plt.cm.Set2,  # type: ignore
    metrics_to_plot: list[str] = TABLE_METRICS,
    metric_pretty_names: dict[str, str] = METRIC_PRETTY_NAMES,
    plot_limits_per_metric: dict[str, tuple[float, float]] = {},
    label_bars: bool = True,
):
    """
    Create bar plots for each metric, comparing different configurations.
    Each dataframe in `means` should have metrics as index and a single column with the mean values for that configuration.
    """
    means = list(means)
    labels = list(labels)
    bars_per_metric = len(means)

    rows = (len(metrics_to_plot) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=figsize, gridspec_kw={"hspace": 0.3}
    )
    axes = axes.flatten()

    means_cat = pd.concat(means, axis=1, keys=labels)

    for ax, metric in zip(axes, metrics_to_plot):
        metric_values = means_cat.loc[metric]
        colors = cmap.colors  # type: ignore
        metric_values.plot.bar(
            ax=ax,
            title=metric_pretty_names.get(metric, metric),
            color=colors,
        )
        if label_bars:
            for p in ax.patches:
                val = p.get_height()
                ax.annotate(
                    format_number_compactly(val),
                    (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center",
                    va="bottom",
                )

        if metric in plot_limits_per_metric:
            y_min, y_max = plot_limits_per_metric[metric]
            ax.set_ylim(y_min, y_max)
        ax.set_xticklabels([])

    for i in range(len(metrics_to_plot), len(axes)):
        fig.delaxes(axes[i])

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in cmap.colors[:bars_per_metric]  # type: ignore
    ]

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 1.05),
        frameon=False,
    )
    return fig, axes


def draw_out_of_range_marker(
    ax,
    x,
    y,
    ylimits,
    color,
    custom_text: str | None = None,
    font_size=7,
    label_offset_fract=0.035,
    out_of_range_marker_offset_fract_delta=0.025,
    text_color="black",
    marker_alpha=1.0,
):

    yrange = ylimits[1] - ylimits[0]
    if y > ylimits[1]:
        ax.text(
            x,
            ylimits[1]
            - (0.02 + out_of_range_marker_offset_fract_delta + label_offset_fract)
            * yrange,
            format_number_compactly(y) if custom_text is None else custom_text,
            ha="center",
            va="top",
            fontsize=font_size,
            color=text_color,
        )
        y = (
            ylimits[1] - out_of_range_marker_offset_fract_delta * yrange
        )  # cap values that exceed y-axis limit for better visualization
        marker = "^"
    elif y < ylimits[0]:
        ax.text(
            x,
            ylimits[0]
            + (0.02 + out_of_range_marker_offset_fract_delta + label_offset_fract)
            * yrange,
            format_number_compactly(y) if custom_text is None else custom_text,
            ha="center",
            va="bottom",
            fontsize=font_size,
            color=text_color,
        )
        y = (
            ylimits[0] + out_of_range_marker_offset_fract_delta * yrange
        )  # cap values that are below y-axis limit for better visualization
        marker = "v"
    ax.scatter(
        x,
        y,
        color=color,
        marker=marker,
        alpha=marker_alpha,
    )


def grouped_per_metric_barplots_for_each_config(
    data: dict[str, dict[str, pd.DataFrame]],
    additional_data: dict[str, pd.DataFrame] = {},
    figsize=(8, 8),
    colors=plt.cm.tab10.colors,  # type: ignore
    metrics_to_plot: list[str] = TABLE_METRICS,
    metric_pretty_names: dict[str, str] = METRIC_PRETTY_NAMES,
    columns: int = 2,
    label_all_bars: bool = False,
    plot_limits_per_metric: dict[str, tuple[float, float]] = {},
    dpi=None,
    line_at_y: float | None = None,
    show_legend: bool = True,
    legend_y_offset=0.075,
    rotate_labels_angle: float = 0,
    rotate_bar_labels_angle: float = 0,
    bar_labels_font_scale: float = 1.0,
    rotate_all_labels: bool = False,
    additional_data_legend_name: str | None = None,
    padding_factor: float = 1.0,
    label_metric_in: Literal["title", "axis", "none"] = "title",
    font_scale: float = 1.0,
    y_ticks_pad_scale: float = 1.0,
):
    """
    Create bar plots for each metric, comparing different configurations with 2 varying dimensions (e.g. noise level and strategy).
    `data` should be a nested dictionary where the first key is the group (e.g. strategy), the second key is the subgroup (e.g. noise level), and the value is a dataframe with metrics as columns.
    The groups are differentiated by position on X axis, and the subgroups are differentiated by color.

    Additional_data is a dict of dataframes that will be plotted as additional bars (e.g. SfM results), where the key is the label for the legend and the dataframe has metrics as index and a single column with the mean values.
    """

    groups = list(data.keys())
    subgroups_per_item = [list(subgroups.keys()) for subgroups in data.values()]
    longest_subgroup = max((subgroups for subgroups in subgroups_per_item), key=len)
    color_per_subgroup = {
        subgroup: colors[i % len(colors)] for i, subgroup in enumerate(longest_subgroup)
    }

    rows = (len(metrics_to_plot) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=figsize, gridspec_kw={"hspace": 0.5}, dpi=dpi
    )
    if (rows == 1 and columns == 1) or len(metrics_to_plot) == 1:
        axes = np.array([axes])  # type: ignore
    axes = axes.flatten()

    # padding, sfm, padding, group1, padding, group2, padding, ...
    # Everything as fractions of axis width
    nonextra_group_base_width = 4
    longest_subgroups_len = len(longest_subgroup)
    bar_width = nonextra_group_base_width / longest_subgroups_len
    additional_group_width = bar_width * 0.75

    bar_font_size = 7
    font_size = 7 * font_scale
    label_offset_fract = 0.035

    def draw_bar(
        x, y, color, ylimits: tuple[float, float] | None = None, width=bar_width * 0.7
    ):
        marker = "."
        alpha = 1.0
        out_of_range = False
        if ylimits is not None:
            if y > ylimits[1] or y < ylimits[0]:
                draw_out_of_range_marker(
                    ax,
                    x,
                    y,
                    ylimits,
                    color,
                    font_size=bar_font_size * bar_labels_font_scale,
                    label_offset_fract=label_offset_fract,
                )
                return

        if out_of_range:
            return
        ax.scatter(
            x,
            y,
            color=color,
            marker=marker,
            alpha=alpha,
        )

        # Rect width to match jitter
        ax.add_patch(
            plt.Rectangle(
                (x - width / 2, 0),
                width,
                y,
                color=color,
                alpha=0.25,
            )
        )

    ax: Axes
    for ax, metric in zip(axes, metrics_to_plot):
        ylimits: tuple[float, float] | None = None
        if metric in plot_limits_per_metric:
            ylimits = plot_limits_per_metric[metric]
            ax.set_ylim(*ylimits)
        else:
            # adapt limits to include outliers while keeping most bars well visualized
            all_values = []
            for i, (group, subgroups) in enumerate(zip(groups, subgroups_per_item)):
                for j, subgroup in enumerate(subgroups):
                    df = data[group][subgroup]
                    all_values.append(df.loc[metric])
            if all_values:
                min_value = min(all_values)
                max_value = max(all_values)
                r = max_value - min_value
                ylimits = (min_value - 0.1 * r, max_value + 0.1 * r)
                ax.set_ylim(*ylimits)

        additional_group_ticks: list[float] = []
        additional_bar_end = padding_factor / 2
        for additional_label, additional_df in additional_data.items():
            additional_group_ticks.append(
                additional_bar_end + additional_group_width / 2
            )
            draw_bar(
                x=additional_bar_end + additional_group_width / 2,
                y=additional_df.loc[metric],
                color="gray",
                ylimits=ylimits,
            )
            additional_bar_end += additional_group_width + padding_factor

        nonextra_group_widths = [
            len(subgroups) / longest_subgroups_len * nonextra_group_base_width
            for subgroups in subgroups_per_item
        ]
        group_starts = [
            additional_bar_end
            + (np.cumsum(nonextra_group_widths[:i])[-1] if i > 0 else 0)
            + i * padding_factor
            for i in range(len(groups))
        ]
        group_ticks = [
            start + nonextra_group_widths[i] / 2 for i, start in enumerate(group_starts)
        ]

        for i, (group, subgroups) in enumerate(zip(groups, subgroups_per_item)):
            for j, subgroup in enumerate(subgroups):
                df = data[group][subgroup]
                mark_x_pos = group_starts[i] + bar_width * j + bar_width / 2
                y_value = df[metric]

                draw_bar(mark_x_pos, y_value, color_per_subgroup[subgroup], ylimits)

        if label_all_bars:
            final_ylim = ax.get_ylim()
            final_yrange = final_ylim[1] - final_ylim[0]

            for i, additional_df in enumerate(additional_data.values()):
                y_value = additional_df.loc[metric]
                if y_value < final_ylim[0] or y_value > final_ylim[1]:
                    continue  # already labeled since out of range
                ax.text(
                    additional_group_ticks[i],
                    additional_df.loc[metric] + label_offset_fract * final_yrange,
                    format_number_compactly(additional_df.loc[metric]),
                    ha="center",
                    va="bottom",
                    fontsize=bar_font_size * bar_labels_font_scale,
                    rotation=rotate_bar_labels_angle,
                )

            for i, (group, subgroups) in enumerate(zip(groups, subgroups_per_item)):
                for j, subgroup in enumerate(subgroups):
                    df = data[group][subgroup]
                    mark_x_pos = group_starts[i] + bar_width * j + bar_width / 2
                    y_value = df.loc[metric]
                    if y_value < final_ylim[0] or y_value > final_ylim[1]:
                        continue  # already labeled since out of range

                    ax.text(
                        mark_x_pos + 0.1,
                        y_value + label_offset_fract * final_yrange,
                        format_number_compactly(y_value),
                        ha="center",
                        va="bottom",
                        fontsize=bar_font_size * bar_labels_font_scale,
                        rotation=rotate_bar_labels_angle,
                    )

        if label_metric_in == "title":
            ax.set_title(metric_pretty_names.get(metric, metric))
        elif label_metric_in == "axis":
            ax.set_ylabel(metric_pretty_names.get(metric, metric))
        # Set x-tick fontsize
        ax.tick_params(axis="x", which="major", labelsize=7 * font_scale, pad=3)
        ax.tick_params(
            axis="y",
            which="major",
            labelsize=6.2 * font_scale,
            pad=2.5 * y_ticks_pad_scale,
        )

        ax.set_xticks(additional_group_ticks + group_ticks)
        ax.set_xticklabels(list(additional_data.keys()) + groups)

        # Force ticks to be created
        plt.draw()

        # Get tick label objects
        labels = ax.get_xticklabels()

        # Rotate only selected ones (example: even indices)
        for i, label in enumerate(labels):
            if rotate_all_labels or i < len(additional_group_ticks):
                label.set_rotation(rotate_labels_angle)

        if line_at_y is not None:
            ax.axhline(line_at_y, color="gray", linestyle="--", linewidth=1)

    for i in range(len(metrics_to_plot), len(axes)):
        axes[i].set_visible(False)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=color, markersize=10
        )
        for color in colors[:longest_subgroups_len]
    ]
    labels = max((subgroups for subgroups in subgroups_per_item), key=len)
    if additional_data_legend_name is not None:
        handles.insert(
            0,
            plt.Line2D(
                [0], [0], marker="o", color="w", markerfacecolor="gray", markersize=10
            ),
        )
        labels.insert(0, additional_data_legend_name)

    if show_legend:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(subgroups) + int(additional_data_legend_name is not None),
            bbox_to_anchor=(0.5, -legend_y_offset * 3 / figsize[1]),
            # reduce space between legend items
            columnspacing=0.6,
            # reduce space between marker and label
            handletextpad=0.15,
            frameon=False,
            fontsize=7 * font_scale,
        )
    return fig, axes, (handles, labels)


def grouped_per_metric_line_charts_for_each_config(
    data: dict[str, dict[float, pd.DataFrame]],
    extra_data_lines: dict[str, pd.DataFrame] = {},
    metrics_to_plot: list[str] = TABLE_METRICS,
    metric_pretty_names: dict[str, str] = METRIC_PRETTY_NAMES,
    columns: int = 2,
    plot_limits_per_metric: dict[str, tuple[float, float]] = {},
    figsize=(12, 8),
    xlabel: str | None = None,
    cmap: plt.Colormap = plt.cm.tab10,  # type: ignore
    lines_cmap: plt.Colormap = plt.cm.tab10,  # type: ignore
    x_scale: str = "linear",
    no_custom_ticks: bool = False,
    show_extra_data_text_labels: bool = False,
    legend_y_offset=0.01,
    remove_y_axis_labels: bool = False,
    show_legend: bool = True,
    fontsize_scale: float = 1.0,
):
    """
    Create line charts for each metric, comparing different configurations.
    Input format: configuration (determines color) -> x value (e.g. num points) -> dataframe with metrics as columns

    Optional arg extra_data_lines can be used to plot additional lines (e.g. SfM results), where the key is the label for the legend and the dataframe has metrics as index and a single column with the mean values.
    """
    groups = list(data.keys())
    x_values_per_group = [list(subgroups.keys()) for subgroups in data.values()]
    colors = cmap.colors  # type: ignore

    rows = (len(metrics_to_plot) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=figsize, gridspec_kw={"hspace": 0.3}
    )
    axes = axes.flatten()
    label_offset_fract = 0.01

    for ax, metric in zip(axes, metrics_to_plot):
        if metric in plot_limits_per_metric:
            ax.set_ylim(*plot_limits_per_metric[metric])

        ax.set_xscale(x_scale)
        # Set fontsize for axis labels and title
        ax.tick_params(
            axis="both", which="major", labelsize=7 * fontsize_scale, pad=2.5
        )
        # Increase spacing between subplots
        if ax is axes[0]:
            fig.subplots_adjust(hspace=0.5, wspace=0.25)

        for i, group in enumerate(groups):
            x_values = x_values_per_group[i]
            y_values = [data[group][x][metric] for x in x_values]
            ax.plot(x_values, y_values, label=group, color=colors[i], alpha=0.6)
            ax.scatter(x_values, y_values, color=colors[i], s=20)

        ax.set_title(
            metric_pretty_names.get(metric, metric), fontsize=9 * fontsize_scale
        )
        if xlabel is not None:
            ax.set_xlabel(
                xlabel, fontsize=9 * fontsize_scale, labelpad=8 * fontsize_scale
            )
        if not remove_y_axis_labels:
            ax.set_ylabel(
                metric_pretty_names.get(metric, metric), fontsize=8 * fontsize_scale
            )
        if not no_custom_ticks:
            longest_x_values = max(x_values_per_group, key=len)
            ax.set_xticks(longest_x_values)
            ax.set_xticklabels([format_number_compactly(x) for x in longest_x_values])
        handles, labels = ax.get_legend_handles_labels()

        for i, (extra_label, extra_df) in enumerate(extra_data_lines.items()):
            extra_y_value = extra_df.loc[metric]
            # Add the extra line to the legend
            handles.append(
                plt.Line2D([0], [0], color=lines_cmap.colors[i], linestyle="--")
            )
            labels.append(extra_label)
            if metric in plot_limits_per_metric:
                y_min, y_max = plot_limits_per_metric[metric]
                x_pos = ax.get_xlim()[0] + (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.5
                if extra_y_value < y_min or extra_y_value > y_max:
                    draw_out_of_range_marker(
                        ax,
                        x_pos,
                        extra_y_value,
                        (y_min, y_max),
                        lines_cmap.colors[i],
                        custom_text=f"{extra_label}: {extra_y_value:.3f}",
                        font_size=9 * fontsize_scale,
                        label_offset_fract=0.005,
                        marker_alpha=1.0,
                    )
                    continue
            ax.axhline(
                extra_y_value,
                color=lines_cmap.colors[i],
                linestyle="--",
                linewidth=1,
                label=extra_label,
            )
            if show_extra_data_text_labels:
                ax.text(
                    x=0.95,
                    y=extra_y_value
                    + label_offset_fract * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    s=extra_label,
                    color=lines_cmap.colors[i],
                    fontsize=8 * fontsize_scale,
                    ha="right",
                    va="bottom",
                    transform=ax.get_yaxis_transform(),
                )

        if show_legend:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=len(labels),
                bbox_to_anchor=(0.5, -legend_y_offset * 8 / figsize[1]),
                frameon=False,
            )

    for i in range(len(metrics_to_plot), len(axes)):
        fig.delaxes(axes[i])

    return fig, axes
