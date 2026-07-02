import logging
import re
from typing import Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from results_scripts.constants import (
    LOWER_IS_BETTER_METRICS,
    METRIC_PRETTY_NAMES,
    TABLE_ROUNDING_PER_METRIC,
)
from results_scripts.formatting import (
    CellData,
    FormatOptions,
    MetricsLayout,
    make_cell_formatter,
)

_LOGGER = logging.getLogger(__name__)


# A diverging colormap with a pure-white center (so the neutral middle value is
# indistinguishable from the page background). Built from ColorBrewer's ``RdBu``
# anchors, which keep clean blue/red hues when blended toward white (unlike
# ``coolwarm``, whose blue/red turn muddy/purple when forced through white). The
# 0.15/0.85 anchors avoid ``RdBu``'s very dark endpoints so cells stay legible.
# Used for delta/ablation tables, where cells encode signed differences from a
# reference and a neutral center carries meaning.
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "rdbu_white",
    [
        mpl.colormaps["RdBu"](0.85),  # blue
        mpl.colormaps["RdBu"](0.65),  # light blue
        (1.0, 1.0, 1.0, 1.0),  # white center
        mpl.colormaps["RdBu"](0.35),  # light red
        mpl.colormaps["RdBu"](0.15),  # red
    ],
)


def make_desaturated_cmap(
    cmap: "mpl.colors.Colormap", saturation: float
) -> LinearSegmentedColormap:

    # Get the colormap colors, multiply them with the factor "a", and create new colormap
    retval = cmap(np.arange(cmap.N))
    retval[:, 0:3] = retval[:, 0:3] * saturation + (1 - saturation) * np.array(
        [1, 1, 1]
    )
    retval = ListedColormap(retval)
    retval.name = f"{cmap.name}_desat{saturation:.2f}"
    return retval


# Colormap for tables whose cells hold absolute values (not deltas vs a baseline
# or ablation reference)
VALUE_CMAP = DIVERGING_CMAP

# Which colormaps are diverging (have a meaningful neutral center) vs sequential,
# keyed by colormap name (Colormap objects are unhashable). This drives how
# ``color_intensity`` reduces a table's colors: for a diverging map both extremes
# are pulled toward the center, while for a sequential map only the high (strong)
# end is dimmed and the low/neutral end stays put. Add an entry here when
# introducing a new colormap; unlisted maps are treated as diverging.
CMAP_IS_DIVERGING: dict[str, bool] = {
    DIVERGING_CMAP.name: True,
    VALUE_CMAP.name: True,
}


def _truncated_colormap(
    cmap: "mpl.colors.Colormap", intensity: float
) -> LinearSegmentedColormap:
    """Return a lower-``intensity`` variant of ``cmap``.

    ``intensity`` of ``1.0`` reproduces the full colormap. For a diverging map
    (per ``CMAP_IS_DIVERGING``) smaller values clamp *both* extremes toward the
    neutral center, so the strongest cells on either side appear paler. For a
    sequential map only the high end is dimmed (the colormap is truncated to
    ``[0, intensity]``), keeping the low/neutral end fixed.
    """
    intensity = min(1.0, max(0.0, intensity))
    if CMAP_IS_DIVERGING.get(cmap.name, True):
        lo = 0.5 - intensity / 2.0
        hi = 0.5 + intensity / 2.0
    else:
        lo = 0.0
        hi = intensity
    return LinearSegmentedColormap.from_list(
        f"{cmap.name}_trunc", cmap(np.linspace(lo, hi, 256))
    )


def tabular_colored_from_numeric_with_custom_text(
    top_left_label: str,
    table: pd.DataFrame,
    text_table: pd.DataFrame,
    range_mode: Literal["default", "centered"] = "default",
    hide_nulls: bool = True,
    column_format: str | None = None,
    header_block: str | None = None,
    color_range: tuple[float, float] | None = None,
    hrule_after_first_row: bool = False,
    color_intensity: float = 1.0,
    force_black_text: bool = False,
    cmap: "mpl.colors.Colormap" = DIVERGING_CMAP,
    color_best_fract: float = 1.0,
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
    hrule_after_first_row: if set, insert a ``\\midrule`` after the first body
        row (e.g. to separate a reference row from the rest).
    color_intensity: upper bound on the color intensity in ``[0, 1]``. ``1.0``
        uses the full color map; lower values compress the gradient toward its
        neutral center so the strongest cells stay paler. ``0.0`` disables
        coloring entirely (cells stay white, only the text is rendered).
    force_black_text: if set, render all cell text black regardless of the cell
        background (instead of switching to light text on dark cells).
    cmap: base colormap for the gradient. Defaults to the white-centered diverging
        map; pass ``VALUE_CMAP`` (or any colormap) for absolute-value tables. How
        ``color_intensity`` dims it depends on ``CMAP_IS_DIVERGING``.
    color_best_fract: fraction in ``[0, 1]`` of the top of the *value range* to
        color; cells whose value falls below ``vmax - color_best_fract * (vmax -
        vmin)`` are left white. The gradient is restretched over just the colored
        slice so those cells use the full colormap. Only applies to non-diverging
        (value) colormaps.
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

    if color_intensity <= 0.0:
        # Skip the gradient entirely; leave every cell white and only render text.
        styled_table = numeric_table.style.applymap(
            lambda _v: "background-color: white; color: black"
        )
    else:
        cmap_name = cmap.name
        cmap = _truncated_colormap(cmap, color_intensity)

        # For value (non-diverging) maps, optionally color only the top
        # ``color_best_fract`` slice of the value range and stretch the gradient
        # across just that slice, so the colored cells use the full colormap.
        gradient_vmin = vmin
        whiten_threshold = None
        if (
            not CMAP_IS_DIVERGING.get(cmap_name, True)
            and color_best_fract < 1.0
            and pd.notna(vmin)
            and pd.notna(vmax)
        ):
            whiten_threshold = vmax - color_best_fract * (vmax - vmin)
            gradient_vmin = whiten_threshold

        styled_table = numeric_table.style.background_gradient(
            cmap=cmap,
            axis=None,
            vmin=gradient_vmin,
            vmax=vmax,
            # ``text_color_threshold=0`` keeps all text dark; the default switches
            # to light text on dark cells.
            text_color_threshold=0 if force_black_text else 0.408,
        ).highlight_null(
            color=None,
            props=f"background-color: white; color: {'white' if hide_nulls else 'black'}",
        )  # type: ignore

        if whiten_threshold is not None:
            # ``< threshold`` is False for NaN, so genuinely-null cells stay
            # handled by ``highlight_null`` above.
            whiten_mask = numeric_table < whiten_threshold
            styled_table = styled_table.apply(
                lambda _df, m=whiten_mask: m.replace(
                    {True: "background-color: white; color: black", False: ""}
                ),
                axis=None,
            )

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

    if hrule_after_first_row:
        lines = tabular_only.splitlines()
        # The header separator is the first ``\midrule``; the first body row is
        # the next line, after which we insert another ``\midrule``.
        midrule_index = next(i for i, ln in enumerate(lines) if r"\midrule" in ln)
        lines.insert(midrule_index + 2, r"\midrule")
        tabular_only = "\n".join(lines)

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
    cmap: "mpl.colors.Colormap" = VALUE_CMAP,
    delta: bool = False,
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

    When ``delta`` is set, the first row (per ``row_order``) is treated as a
    reference: it shows absolute values and lands on the color map's neutral
    center, while every other row shows the signed difference from the reference
    within the same column, colored on a symmetric scale per metric group.
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
                f"row_order must contain the same elements as the data keys: {set(data.keys())}, but got {set(row_order)}"
            )
        row_labels = row_order
    else:
        row_labels = list(data.keys())

    tabular_blocks: list[str] = []

    reference_row = row_labels[0]
    format_cell = make_cell_formatter(format_args.cell_type)
    format_cell_signed = make_cell_formatter(
        format_args.cell_type, always_show_sign=True
    )

    # Build the numeric (means or, in delta mode, per-column deltas) and
    # formatted-text tables for every metric.
    per_metric_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for metric in metrics:
        means = pd.DataFrame(columns=column_labels, index=row_labels)
        cells: dict[tuple[str, str], CellData] = {}
        for row_label, df_per_column in data.items():
            for col_label, df in df_per_column.items():
                cell_data = CellData.for_metric(df, metric)
                cells[(row_label, col_label)] = cell_data
                means.loc[row_label, col_label] = cell_data.mean

        means = means.round(TABLE_ROUNDING_PER_METRIC.get(metric, 2))
        text_table = pd.DataFrame(columns=column_labels, index=row_labels)

        if delta:
            # Per-column signed difference from the reference (first) row; the
            # reference row keeps its absolute value and the rest show deltas.
            rounding = TABLE_ROUNDING_PER_METRIC.get(metric, 2)
            ref_means = {
                col: round(cells[(reference_row, col)].mean, rounding)
                for col in column_labels
            }
            color_table = pd.DataFrame(columns=column_labels, index=row_labels)
            for row_label in row_labels:
                for col_label in column_labels:
                    cell = cells[(row_label, col_label)]
                    cell_delta = round(cell.mean, rounding) - ref_means[col_label]
                    color_table.loc[row_label, col_label] = cell_delta
                    if row_label == reference_row:
                        text_table.loc[row_label, col_label] = format_cell(cell)
                    else:
                        text_table.loc[row_label, col_label] = format_cell_signed(
                            cell._replace(mean=cell_delta, min=-1.0, max=-1.0)
                        )
            per_metric_tables[metric] = (color_table, text_table)
        else:
            for (row_label, col_label), cell in cells.items():
                text_table.loc[row_label, col_label] = format_cell(cell)
            per_metric_tables[metric] = (means, text_table)

    if format_args.metrics_layout == MetricsLayout.vertical:
        # Render one section per metric, then splice them into a SINGLE tabular
        # (\midrule-separated) so the column widths stay aligned. Separate
        # tabulars (each wrapped in its own \resizebox) would scale
        # independently and misalign.
        section_tabulars = [
            tabular_colored_from_numeric_with_custom_text(
                f"\\textbf{{{METRIC_PRETTY_NAMES[metric]}}}",
                # Flip the gradient for lower-is-better metrics so "better" stays
                # on the warm end of the color map.
                -table if metric in LOWER_IS_BETTER_METRICS else table,
                text_table,
                column_format=column_format,
                # In delta mode color symmetrically so the reference row (delta 0)
                # is neutral, and rule it off from the delta rows.
                range_mode="centered" if delta else "default",
                hrule_after_first_row=delta,
                color_intensity=format_args.color_intensity,
                force_black_text=format_args.force_black_text,
                cmap=cmap,
                color_best_fract=format_args.color_best_fract,
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
            invert = metric in LOWER_IS_BETTER_METRICS
            if delta:
                # ``table`` holds per-column deltas vs the reference row; color
                # them on a symmetric scale so the reference (delta 0) is neutral.
                max_abs = numeric.abs().max().max()
                direction = -1.0 if invert else 1.0
            else:
                vmin = numeric.min().min()
                vmax = numeric.max().max()
                pad = (vmax - vmin) * 0.1 if pd.notna(vmin) and pd.notna(vmax) else 0.0
                lo, span = vmin - pad, (vmax + pad) - (vmin - pad)
            for col_label in column_labels:
                cid = f"{metric}::{col_label}"
                ordered_ids.append(cid)
                if delta:
                    if pd.notna(max_abs) and max_abs:
                        normalized = (
                            0.5 + 0.5 * (direction * numeric[col_label] / max_abs)
                        ).clip(0.0, 1.0)
                    else:
                        normalized = 0.5
                    combined_color[cid] = normalized
                else:
                    normalized = (numeric[col_label] - lo) / span if span else 0.5
                    # Flip lower-is-better metrics so "better" stays on the warm end.
                    combined_color[cid] = (1.0 - normalized) if invert else normalized
                combined_text[cid] = text_table[col_label]

        tabular_blocks = [
            tabular_colored_from_numeric_with_custom_text(
                top_left_label="",
                table=combined_color[ordered_ids],
                text_table=combined_text[ordered_ids],
                column_format=side_format,
                header_block=header_block,
                color_range=(0.0, 1.0),
                hrule_after_first_row=delta,
                color_intensity=format_args.color_intensity,
                force_black_text=format_args.force_black_text,
                cmap=cmap,
                color_best_fract=format_args.color_best_fract,
            )
        ]
    else:
        raise ValueError(f"Invalid metrics_layout: {format_args.metrics_layout}")

    return wrap_tabulars_as_float(
        tabular_blocks, latex_caption, latex_label, format_args
    )


def wrap_tabulars_as_float(
    tabular_blocks: list[str],
    latex_caption: str,
    latex_label: str,
    format_args: FormatOptions,
) -> str:
    """Wrap rendered ``tabular`` blocks in a float (or ``subtable``) environment.

    Applies the table size, ``\\tabcolsep`` scaling and optional ``\\resizebox``
    from ``format_args``. When ``format_args.combine_datasets_as_subtables`` is
    set, a ``subtable`` block (no float of its own) is emitted so the caller can
    wrap several of them in a single outer float; otherwise a standalone
    ``table``/``table*`` float is produced.
    """
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


def make_aggregated_metric_table(
    cell_data: dict[str, dict[str, "CellData"]],
    metrics: list[str],
    latex_caption: str,
    latex_label: str,
    format_args: FormatOptions = FormatOptions(),
    row_order: list[str] | None = None,
    top_left_label: str = "",
    delta: bool = False,
) -> str:
    """Render a table with metrics in columns and one row per ``cell_data`` key.

    ``cell_data`` maps a row label to a mapping of ``metric id -> CellData``, with
    the aggregate values (mean, std, min, max) already populated.

    The first row acts as a reference: it is always placed at the center of the
    color map, and every other cell is colored (per metric column, on a symmetric
    scale) according to how far its value lies from the reference value.

    When ``delta`` is set, the first row shows the actual value while the other
    rows show signed deltas relative to it. Otherwise the cell text comes from the
    formatter selected by ``format_args.cell_type``.
    """
    row_labels = row_order if row_order is not None else list(cell_data.keys())
    metric_labels = [METRIC_PRETTY_NAMES.get(metric, metric) for metric in metrics]

    format_cell_row0 = make_cell_formatter(
        format_args.cell_type, always_show_sign=False
    )
    format_cell_rowN = make_cell_formatter(
        format_args.cell_type, always_show_sign=delta
    )

    color_table = pd.DataFrame(index=row_labels, columns=metric_labels, dtype=float)
    text_table = pd.DataFrame(index=row_labels, columns=metric_labels, dtype=object)

    for metric, metric_label in zip(metrics, metric_labels):
        # Round means to the same precision as the displayed text so the colors
        # reflect exactly the (rounded) numbers the reader sees.
        rounding = TABLE_ROUNDING_PER_METRIC.get(metric, 2)
        rounded_means = {
            row: (
                round(cell_data[row][metric].mean, rounding)
                if pd.notna(cell_data[row][metric].mean)
                else cell_data[row][metric].mean
            )
            for row in row_labels
        }
        reference_mean = rounded_means[row_labels[0]]
        deltas = {row: rounded_means[row] - reference_mean for row in row_labels}
        # Symmetric range around the reference (delta 0), so the reference row
        # lands on the color map center and the rest spread out from there.
        max_abs = max((abs(d) for d in deltas.values() if pd.notna(d)), default=0.0)
        # Flip lower-is-better metrics so an improvement (lower value) is warm.
        direction = -1.0 if metric in LOWER_IS_BETTER_METRICS else 1.0

        for index, row in enumerate(row_labels):
            cell = cell_data[row][metric]
            cell_delta = deltas[row]
            if not pd.notna(cell_delta) or max_abs == 0:
                color = 0.5
            else:
                color = 0.5 + 0.5 * (direction * cell_delta / max_abs)
            color_table.loc[row, metric_label] = min(1.0, max(0.0, color))

            if delta and index != 0:
                # Shift the value-like fields into delta space (the spread fields
                # are invariant under the shift) so the chosen formatter renders
                # the delta with its own std/min/max.
                cell = cell._replace(
                    mean=cell_delta,
                    min=-1,  # no clear meaning for these, so set to a sentinel value
                    max=-1,
                )
            format_cell = format_cell_row0 if index == 0 else format_cell_rowN
            text_table.loc[row, metric_label] = format_cell(cell)

    tabular_only = tabular_colored_from_numeric_with_custom_text(
        top_left_label=top_left_label,
        table=color_table,
        text_table=text_table,
        column_format="l" + "c" * len(metric_labels),
        color_range=(0.0, 1.0),
        hrule_after_first_row=True,
        color_intensity=format_args.color_intensity,
        force_black_text=format_args.force_black_text,
    )

    return wrap_tabulars_as_float(
        [tabular_only], latex_caption, latex_label, format_args
    )


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
