import logging
import re
from typing import Literal

import pandas as pd
from results_scripts.constants import METRIC_PRETTY_NAMES, TABLE_ROUNDING_PER_METRIC
from results_scripts.formatting import (
    CellData,
    FormatOptions,
    MetricsLayout,
    make_cell_formatter,
)

_LOGGER = logging.getLogger(__name__)


def tabular_colored_from_numeric_with_custom_text(
    top_left_label: str,
    table: pd.DataFrame,
    text_table: pd.DataFrame,
    range_mode: Literal["default", "centered"] = "default",
    hide_nulls: bool = True,
    column_format: str | None = None,
    header_block: str | None = None,
    color_range: tuple[float, float] | None = None,
):
    """Render ``table`` as a colored LaTeX ``tabular`` using ``text_table`` cell text.

    column_format: overrides the default ``l`` + ``c`` * ncols column spec (e.g.
        ``"l|ccc|ccc"`` for grouped columns).
    header_block: if given, fully replaces the auto-generated single header row
        (the row beginning with ``&``). Use this to provide a multi-row header
        with ``\\multicolumn`` groups. The trailing ``\\\\`` must be included.
        When provided, ``top_left_label`` is ignored.
    color_range: if given, an explicit ``(vmin, vmax)`` for the gradient, used
        verbatim (no padding) and overriding ``range_mode``. Use this to share an
        identical color scale across multiple tables rendered separately.
    """
    numeric_table = table.apply(pd.to_numeric, errors="coerce")
    # TODO: make this global across datasets??? or no?
    if color_range is not None:
        vmin, vmax = color_range
    elif range_mode == "default":
        vmin = numeric_table.min().min()
        vmax = numeric_table.max().max()
        pad = (vmax - vmin) * 0.1 if pd.notna(vmin) and pd.notna(vmax) else 0.0
        vmin -= pad
        vmax += pad
    elif range_mode == "centered":
        max_abs = numeric_table.abs().max().max()
        vmin = -max_abs
        vmax = max_abs
        pad = (vmax - vmin) * 0.1 if pd.notna(vmin) and pd.notna(vmax) else 0.0
        vmin -= pad
        vmax += pad
    else:
        raise ValueError(f"Invalid range_mode: {range_mode}")

    styled_table = numeric_table.style.background_gradient(
        cmap="coolwarm", axis=None, vmin=vmin, vmax=vmax
    ).highlight_null(
        color=None,
        props=f"background-color: white; color: {'white' if hide_nulls else 'black'}",
    )  # type: ignore

    for r in text_table.index:
        for c in text_table.columns:
            styled_table = styled_table.format(
                {c: (lambda s=text_table.loc[r, c]: (lambda _v: s))()},
                subset=pd.IndexSlice[[r], [c]],
                na_rep="--" if hide_nulls else None,
            )

    # Get only the tabular environment (no per-metric table wrapper)
    latex_str = styled_table.to_latex(
        convert_css=True,
        hrules=True,
        column_format=column_format or ("l" + "c" * len(table.columns)),
    )
    lines = latex_str.splitlines()
    start = next(i for i, ln in enumerate(lines) if r"\begin{tabular}" in ln)
    end = next(i for i, ln in enumerate(lines) if r"\end{tabular}" in ln)
    tabular_only = "\n".join(lines[start : end + 1])

    # The Styler emits a single header row beginning with ``&``. Body rows start
    # with their index label, so the regex only matches that header row.
    header_regex = r"^(\s*)&(.*)$"
    if header_block is not None:
        # Lambda replacement avoids re.sub backslash-escaping of the LaTeX text.
        tabular_only = re.sub(
            header_regex,
            lambda _m: header_block,
            tabular_only,
            count=1,
            flags=re.MULTILINE,
        )
    else:
        tabular_only = re.sub(
            header_regex,
            lambda m: f"{m.group(1)}{top_left_label} &{m.group(2)}",
            tabular_only,
            count=1,
            flags=re.MULTILINE,
        )

    return tabular_only


def make_latex_table_for_metrics(
    data: dict[str, dict[str, pd.DataFrame]],
    latex_caption: str,
    latex_label: str,
    metrics: list[str] | None = None,
    column_order: list[str] | None = None,
    column_format: str | None = None,
    row_order: list[str] | None = None,
    format_args: FormatOptions = FormatOptions(),
    horizontal_cols_label: str = "",
) -> str:
    """
    data: strategy -> column -> per-scene dataframe for all metrics with lists of values per eval iter in cells.

    The metrics layout is controlled by ``format_args.metrics_layout``:
        "vertical" (default): one ``tabular`` section per metric, stacked
            vertically (each metric labelled in the top-left corner).
        "horizontal": a single ``tabular`` where each metric is a
            ``\\multicolumn`` group of columns laid out horizontally, e.g.
            ``PSNR: SfM | EDGS | ... || SSIM: SfM | EDGS | ...``. Each metric
            group is colored independently (metrics have different scales).
    """
    metrics = metrics or [
        "eval-all-test/psnr",
        "eval-all-test/ssim",
        "eval-all-test/lpips",
    ]

    def col_key_set():
        return set(key for strat_dict in data.values() for key in strat_dict.keys())

    if column_order is not None:
        col_keys = col_key_set()
        column_labels = [col for col in column_order if col in col_keys]
    else:
        column_labels = list(col_key_set())

    if row_order is not None:
        if set(row_order) != set(data.keys()):
            raise ValueError(
                "row_order must contain the same elements as the data keys."
            )
        row_labels = row_order
    else:
        row_labels = list(data.keys())

    tabular_blocks: list[str] = []

    format_cell = make_cell_formatter(format_args.cell_type)

    # Build the numeric (means) and formatted-text tables for every metric.
    per_metric_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for metric in metrics:
        table = pd.DataFrame(
            columns=column_labels,
            index=row_labels,
        )
        text_table = pd.DataFrame(
            columns=column_labels,
            index=row_labels,
        )

        for row_label, df_per_column in data.items():
            for col_label, df in df_per_column.items():
                cell_data = CellData.for_metric(df, metric)
                table.loc[row_label, col_label] = cell_data.mean
                text_table.loc[row_label, col_label] = format_cell(cell_data)

        table = table.round(TABLE_ROUNDING_PER_METRIC.get(metric, 2))
        per_metric_tables[metric] = (table, text_table)

    if format_args.metrics_layout == MetricsLayout.vertical:
        # Render one section per metric, then splice them into a SINGLE tabular
        # (\midrule-separated) so the column widths stay aligned. Separate
        # tabulars (each wrapped in its own \resizebox) would scale
        # independently and misalign.
        section_tabulars = [
            tabular_colored_from_numeric_with_custom_text(
                f"\\textbf{{{METRIC_PRETTY_NAMES[metric]}}}",
                table,
                text_table,
                column_format=column_format,
            )
            for metric, (table, text_table) in per_metric_tables.items()
        ]

        first_lines = section_tabulars[0].splitlines()
        # First section: keep everything up to (but excluding) its \bottomrule
        # (i.e. \begin{tabular} + \toprule + header + \midrule + body). Remember
        # the closing lines (\bottomrule + \end{tabular}) to re-append once.
        bottom_idx = next(i for i, ln in enumerate(first_lines) if r"\bottomrule" in ln)
        combined_lines = first_lines[:bottom_idx]
        closing_lines = first_lines[bottom_idx:]
        # Subsequent sections: splice in only the inner content (header +
        # \midrule + body, between \toprule and \bottomrule), separated from the
        # previous section by a \midrule.
        for section in section_tabulars[1:]:
            lines = section.splitlines()
            top_idx = next(i for i, ln in enumerate(lines) if r"\toprule" in ln)
            sec_bottom_idx = next(
                i for i, ln in enumerate(lines) if r"\bottomrule" in ln
            )
            combined_lines.append(r"\midrule")
            combined_lines.extend(lines[top_idx + 1 : sec_bottom_idx])
        combined_lines.extend(closing_lines)
        tabular_blocks = ["\n".join(combined_lines)]
    elif format_args.metrics_layout == MetricsLayout.horizontal:
        # Per-metric column group: strip the leading row-label column from a
        # caller-supplied format (e.g. "l|c|ccc" -> "|c|ccc") and repeat it once
        # per metric, or fall back to a plain "|c...c" group.
        group_format = (
            column_format[1:]
            if column_format is not None
            else "|" + "c" * len(column_labels)
        )
        side_format = "l" + group_format * len(metrics)

        # Two-row header: metric groups, then the column labels repeated.
        group_cells = [
            rf"\multicolumn{{{len(column_labels)}}}"
            rf"{{{'c' if i == len(metrics) - 1 else 'c|'}}}"
            rf"{{\textbf{{{METRIC_PRETTY_NAMES[metric]}}}}}"
            for i, metric in enumerate(metrics)
        ]
        repeated_cols = [str(col) for _ in metrics for col in column_labels]
        header_block = (
            "& " + " & ".join(group_cells) + r" \\"
            "\n"
            f"{horizontal_cols_label}& " + " & ".join(repeated_cols) + r" \\"
        )

        # Color each metric group independently (different scales) by normalizing
        # its means into [0, 1] using the same min/max + 10% padding the helper
        # would apply per-table, then color the combined table over (0, 1).
        combined_color = pd.DataFrame(index=row_labels)
        combined_text = pd.DataFrame(index=row_labels)
        ordered_ids: list[str] = []
        for metric in metrics:
            table, text_table = per_metric_tables[metric]
            numeric = table.apply(pd.to_numeric, errors="coerce")
            vmin = numeric.min().min()
            vmax = numeric.max().max()
            pad = (vmax - vmin) * 0.1 if pd.notna(vmin) and pd.notna(vmax) else 0.0
            lo, span = vmin - pad, (vmax + pad) - (vmin - pad)
            for col_label in column_labels:
                cid = f"{metric}::{col_label}"
                ordered_ids.append(cid)
                combined_color[cid] = (numeric[col_label] - lo) / span if span else 0.5
                combined_text[cid] = text_table[col_label]

        tabular_blocks = [
            tabular_colored_from_numeric_with_custom_text(
                top_left_label="",
                table=combined_color[ordered_ids],
                text_table=combined_text[ordered_ids],
                column_format=side_format,
                header_block=header_block,
                color_range=(0.0, 1.0),
            )
        ]
    else:
        raise ValueError(f"Invalid metrics_layout: {format_args.metrics_layout}")

    table_size = (
        f"\\{format_args.table_size}" if format_args.table_size != "default" else ""
    )
    tabcolsep_command = format_args.get_tabcolsep_cmd_begin()

    # The horizontal layout is wide, so span the full text width using the
    # starred table* environment and resize to \textwidth instead of
    # \columnwidth.
    table_env = format_args.get_table_env()
    if format_args.combine_datasets_as_subtables:
        # Emit a subtable block (no float of its own); the caller wraps several
        # of these in a single outer float. Resize to the subtable's \linewidth.
        env_begin = r"\begin{subtable}[t]{\linewidth}"
        env_end = r"\end{subtable}"
        resize_width = r"\linewidth"
    else:
        env_begin = rf"\begin{{{table_env}}}[t]"
        env_end = rf"\end{{{table_env}}}"
        resize_width = r"\textwidth" if table_env == "table*" else r"\columnwidth"

    output_lines = [
        env_begin,
        r"\centering",
        rf"\caption{{{latex_caption}}}",
        rf"\label{{tab:{latex_label}}}",
        "{" + table_size,
        tabcolsep_command,
    ]
    for index, tabular_only in enumerate(tabular_blocks):
        lines = tabular_only.splitlines()
        if index < len(tabular_blocks) - 1:
            lines = [line.replace(r"\bottomrule", "") for line in lines]
        if format_args.resizebox:
            lines = [
                line.replace(
                    r"\begin{tabular}",
                    rf"\resizebox{{{resize_width}}}{{!}}{{\begin{{tabular}}",
                )
                for line in lines
            ]
            lines = [
                line.replace(r"\end{tabular}", r"\end{tabular}}") for line in lines
            ]
        output_lines.append("\n".join(lines))

    output_lines.append(format_args.get_tabcolsep_cmd_end())

    output_lines.extend([r"}", env_end])
    print(f"Created LaTeX table with caption {latex_caption}.")
    return "\n".join(output_lines)


def join_per_dataset_tables_with_latex_comments(tables: dict[str, str]) -> str:
    output = []
    for dataset, table in tables.items():
        output.append(
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Dataset: "
            f"{dataset} "
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        )
        output.append(table)
        output.append("\n")

    return "\n".join(output)


def combine_per_dataset_subtables(
    subtables: dict[str, str],
    table_env: str = "table*",
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Wrap several per-dataset ``subtable`` blocks into a single float.

    ``subtables`` must hold strings produced by ``make_latex_table_for_metrics``
    with ``combine_datasets_as_subtables=True`` (i.e. each is a
    ``\\begin{subtable}...\\end{subtable}`` block). Requires the ``subcaption``
    LaTeX package.
    """
    output: list[str] = [rf"\begin{{{table_env}}}[t]", r"\centering"]
    if caption is not None:
        output.append(rf"\caption{{{caption}}}")
    if label is not None:
        output.append(rf"\label{{tab:{label}}}")

    items = list(subtables.items())
    for index, (dataset, subtable) in enumerate(items):
        output.append(
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Dataset: "
            f"{dataset} "
            "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
        )
        output.append(subtable)

    output.append(rf"\end{{{table_env}}}")
    return "\n".join(output)


def finalize_per_dataset_tables(
    tables: dict[str, str],
    format_args: FormatOptions,
    combined_caption: str | None = None,
    combined_label: str | None = None,
) -> str:
    """Combine per-dataset tables according to ``format_args``.

    If ``format_args.combine_datasets_as_subtables`` is set, wrap the per-dataset
    ``subtable`` blocks in a single float; otherwise join the independent floats
    with section comments.
    """
    if format_args.combine_datasets_as_subtables:
        return combine_per_dataset_subtables(
            tables,
            table_env=format_args.get_table_env(),
            caption=combined_caption,
            label=combined_label,
        )
    return join_per_dataset_tables_with_latex_comments(tables)
