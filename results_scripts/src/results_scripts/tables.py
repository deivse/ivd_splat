import re
from typing import Literal

import pandas as pd
from results_scripts.constants import METRIC_PRETTY_NAMES, TABLE_ROUNDING_PER_METRIC
from results_scripts.formatting import CellData, FormatOptions, make_cell_formatter


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
    column_labels: dict[str, str],
    metrics: list[str] | None = None,
    format_args: FormatOptions = FormatOptions(),
) -> str:
    """
    data: strategy -> column -> per-scene dataframe for all metrics with lists of values per eval iter in cells.
    """
    metrics = metrics or [
        "eval-all-test/psnr",
        "eval-all-test/ssim",
        "eval-all-test/lpips",
    ]

    tabular_blocks: list[str] = []

    format_cell = make_cell_formatter(format_args.table_cell_type)

    for metric in metrics:
        table = pd.DataFrame(
            columns=[
                column_labels[key] for key in ["sfm", "as_sfm", "0.5", "0.75", "1.0"]
            ],
            index=["AbsGS", "INRIA", "MCMC", "IDHFR", "No D."],
        )
        text_table = pd.DataFrame(
            columns=[
                column_labels[key] for key in ["sfm", "as_sfm", "0.5", "0.75", "1.0"]
            ],
            index=["AbsGS", "INRIA", "MCMC", "IDHFR", "No D."],
        )

        for strat_name, df_per_column in data.items():
            for col_id, df in df_per_column.items():
                cell_data = CellData.for_metric(df, metric)
                table.loc[strat_name, column_labels[col_id]] = cell_data.mean
                text_table.loc[strat_name, column_labels[col_id]] = format_cell(
                    cell_data
                )

        table = table.round(TABLE_ROUNDING_PER_METRIC.get(metric, 2))
        tabular_blocks.append(
            tabular_colored_from_numeric_with_custom_text(
                f"\\textbf{{{METRIC_PRETTY_NAMES[metric]}}}",
                table,
                text_table,
            )
        )

    table_size = (
        f"\\{format_args.table_size}" if format_args.table_size != "default" else ""
    )
    tabcolsep_command = format_args.get_tabcolsep_cmd_begin()

    output_lines = [
        r"\begin{table}[t]",
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
        if format_args.resize_to_column:
            lines = [
                line.replace(
                    r"\begin{tabular}",
                    r"\resizebox{\columnwidth}{!}{\begin{tabular}",
                )
                for line in lines
            ]
        lines = [line.replace(r"\end{tabular}", r"\end{tabular}}") for line in lines]
        output_lines.append("\n".join(lines))

    output_lines.append(format_args.get_tabcolsep_cmd_end())

    output_lines.extend([r"}", r"\end{table}"])
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
