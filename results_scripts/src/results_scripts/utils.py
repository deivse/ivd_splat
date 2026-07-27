from dataclasses import dataclass
import json
from pathlib import Path
import re

from matplotlib import pyplot as plt

from results_scripts.constants import STRATEGY_NAMES


def name_to_path(value: str, allow_subdirs: bool) -> str:
    value = value.replace("scannet++", "scannetpp")
    for strategy, short_name in STRATEGY_NAMES.items():
        value = value.replace(strategy, short_name.lower().replace(" ", "_"))
    pattern = r"[^/a-zA-Z0-9._-]+" if allow_subdirs else r"[^a-zA-Z0-9._-]+"
    slug = re.sub(pattern, "_", value.strip())
    return slug.strip("_") or "figure"


def load_json(path: Path) -> dict[str, int]:
    with path.open("r") as handle:
        return json.load(handle)


def write_file(path: Path | str, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write(content)


@dataclass
class OutputDirHelper:
    output_dir: Path

    def get_graph_path(self, section_subdir: str, figure_name: str) -> Path:
        return (
            self.output_dir
            / "graphs"
            / name_to_path(section_subdir, allow_subdirs=True)
            / f"{name_to_path(figure_name, allow_subdirs=False)}.svg"
        )

    def get_table_path(self, table_name: str) -> Path:
        return (
            self.output_dir
            / "tables"
            / f"{name_to_path(table_name, allow_subdirs=False)}.tex"
        )


def fraction_name(fraction: str | float) -> str:
    return f"{float(fraction) * 100:.0f}% $G_\\mathit{{max}}$"


def gmax_fraction_label(fraction: str | float) -> str:
    """LaTeX label for a multiple of the max Gaussian count, e.g.
    ``$0.75\\mathcal{G}_\\mathit{max}$``.

    Used as the canonical notation for "fraction of $\\mathcal{G}_\\mathit{max}$"
    across all tables so the style stays consistent.
    """
    value = float(fraction)
    text = f"{value:g}"
    if "." not in text:
        text += ".0"
    return rf"${text}\text  {{G}}_\mathit{{m}}$"


def save_figure_svg(
    fig: plt.Figure,
    output: Path,
    **kwargs,
):
    output.parent.mkdir(parents=True, exist_ok=True)

    kwargs.setdefault("bbox_inches", "tight")
    fig.savefig(output, format="svg", **kwargs)
    print(f"Saved: {output}")
