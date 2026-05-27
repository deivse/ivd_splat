import re
from typing import Literal

from eval_scripts.results_processing.base import METRIC_PRETTY_NAMES
import pandas as pd


def tabular_colored_from_numeric_with_custom_text(
    top_left_label: str,
    table: pd.DataFrame,
    text_table: pd.DataFrame,
    range_mode: Literal["default", "centered"] = "default",
):
    numeric_table = table.apply(pd.to_numeric, errors="coerce")
    # TODO: make this global across datasets??? or no?
    if range_mode == "default":
        vmin = numeric_table.min().min()
        vmax = numeric_table.max().max()
    elif range_mode == "centered":
        max_abs = numeric_table.abs().max().max()
        vmin = -max_abs
        vmax = max_abs
    else:
        raise ValueError(f"Invalid range_mode: {range_mode}")
    pad = (vmax - vmin) * 0.1 if pd.notna(vmin) and pd.notna(vmax) else 0.0
    vmin -= pad
    vmax += pad

    styled_table = numeric_table.style.background_gradient(
        cmap="coolwarm", axis=None, vmin=vmin, vmax=vmax
    ).highlight_null(color="white")
    for r in text_table.index:
        for c in text_table.columns:
            styled_table = styled_table.format(
                {c: (lambda s=text_table.loc[r, c]: (lambda _v: s))()},
                subset=pd.IndexSlice[[r], [c]],
                na_rep="--",
            )

    # Get only the tabular environment (no per-metric table wrapper)
    latex_str = styled_table.to_latex(
        convert_css=True,
        hrules=True,
        column_format="l" + "c" * len(table.columns),
    )
    lines = latex_str.splitlines()
    start = next(i for i, ln in enumerate(lines) if r"\begin{tabular}" in ln)
    end = next(i for i, ln in enumerate(lines) if r"\end{tabular}" in ln)
    tabular_only = "\n".join(lines[start : end + 1])

    # start of row with space and &
    regex = r"^(\s*)&(.*)$"
    tabular_only = re.sub(
        regex,
        lambda m: f"{m.group(1)}{top_left_label} &{m.group(2)}",
        tabular_only,
        count=1,
        flags=re.MULTILINE,
    )

    return tabular_only
