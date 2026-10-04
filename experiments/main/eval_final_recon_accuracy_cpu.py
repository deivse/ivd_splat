"""
CPU stage of the two-stage final-reconstruction accuracy evaluation.

Loads the reconstructions produced by ``eval_final_recon_accuracy_gpu.py`` from
``--intermediate-dir`` (described by its ``manifest.json``) and computes the
geometry metrics for every cell: the in-house KD-tree F-score against the
processed laser-scan reference, or the official ``ETH3DMultiViewEvaluation``
tool for ETH3D scenes. The per-scene / aggregated metrics are written to
``--output`` (JSON) and rendered as a colored LaTeX table.

This stage performs no GPU work, so it can run on a CPU-only node with many
cores (the KD-tree nearest-neighbour queries parallelise across them).

See ``eval_final_recon_accuracy_common.py`` for the shared implementation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import tyro

from eval_final_recon_accuracy_common import (
    PointSet,
    apply_cpu_thread_limit,
    compute_reconstruction_metrics,
    load_manifest,
    read_point_set_ply,
    render_latex_table,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    # Directory produced by the GPU stage (eval_final_recon_accuracy_gpu.py):
    # holds the rendered reconstructions and the manifest.json. Not required when
    # --load-existing is set.
    intermediate_dir: Path | None = None

    # Directory to use for temporary files (ETH3D reconstruction PLYs).
    temp_dir_override: Path | None = None

    # Maximum number of CPU threads / parallel workers for the KD-tree
    # nearest-neighbour F-score queries. When None, all available cores are used.
    max_cpu_threads: int | None = None

    # F-score inlier distance threshold, in meters (on the GT / laser-scan
    # scale). A reconstruction/GT point counts as matched when its nearest
    # neighbour in the other cloud is within this distance.
    fscore_threshold_meters: float = 0.05

    # Enable debug-level logging.
    debug: bool = False

    # Output JSON file for per-scene and aggregated metrics.
    output: Path = Path("final_recon_accuracy.json")

    # Skip all recomputation and instead load previously computed metrics from
    # the --output JSON file, then (re)write the LaTeX table from them. Useful to
    # re-render the table with different --latex-metrics / --latex-output without
    # rerunning the metric computation.
    load_existing: bool = False

    # Colored LaTeX F-score table output (rows = densification strategies,
    # columns = init methods), rendered with the results_scripts table helpers.
    # Defaults to the JSON output path with a ``.tex`` suffix.
    latex_output: Path | None = None
    # Which computed metric(s) to tabulate. All requested metrics are shown in a
    # single cell separated by '/', in this order (e.g. "F-Score / Precision /
    # Recall"), and cells are colored by the first metric. Defaults to F-score,
    # precision and recall; pass e.g. ``--latex-metrics fscore`` for F-score only.
    latex_metrics: list[str] = field(
        default_factory=lambda: ["fscore", "precision", "recall"]
    )

    # Optional list of scenes to include in the output table. If not set, all scenes in the
    # GPU-stage manifest are included. Scenes should be specified without dataset, e.g. 'pipes', not 'eth3d/pipes'.
    table_scenes: list[str] | None = None


def main() -> None:
    # ``force=True`` so we reconfigure even if an imported dependency already
    # installed a root handler (otherwise basicConfig is a no-op and the root
    # logger stays at WARNING, dropping our INFO logs).
    logging.basicConfig(
        level=logging.INFO,
        force=True,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = tyro.cli(Args)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    apply_cpu_thread_limit(args.max_cpu_threads)

    if args.load_existing:
        if not args.output.exists():
            raise FileNotFoundError(
                f"--load-existing was set but the metrics JSON {args.output} does "
                "not exist. Run without --load-existing first, or point --output at "
                "an existing results file."
            )
        existing = json.loads(args.output.read_text())
        threshold = existing.get(
            "fscore_threshold_meters", args.fscore_threshold_meters
        )
        _LOGGER.info("Loaded existing metrics from %s", args.output)
        render_latex_table(
            args.latex_output,
            args.output,
            args.latex_metrics,
            existing["resolved_runs"],
            threshold,
            scene_list=args.table_scenes,
        )
        return

    if args.intermediate_dir is None:
        raise ValueError(
            "--intermediate-dir is required (unless --load-existing). Point it at "
            "the directory written by the GPU stage."
        )
    intermediate_dir = args.intermediate_dir
    manifest = load_manifest(intermediate_dir)

    scenes = manifest["scenes"]
    columns = manifest["columns"]
    scenes_data = manifest["scenes_data"]

    resolved_runs: dict[str, list[dict]] = {}
    for scene in scenes:
        scene = str(scene)
        scene_data = scenes_data.get(scene)
        if scene_data is None:
            _LOGGER.warning(
                "[%s] No data in manifest (GPU stage may not have reached it); "
                "skipping.",
                scene,
            )
            continue

        eth3d_project = scene_data.get("eth3d_meshlab_project_path")
        eth3d_project_path = Path(eth3d_project) if eth3d_project is not None else None

        reference: PointSet | None = None
        reference_rel = scene_data.get("reference_path")
        if reference_rel is not None:
            reference = read_point_set_ply(intermediate_dir / reference_rel)

        scene_entries: list[dict] = []
        for entry_in in scene_data["entries"]:
            entry: dict = {
                "column": entry_in["column"],
                "strategy": entry_in["strategy"],
                "output_dir": entry_in.get("output_dir"),
                "exists": entry_in.get("exists", True),
                "metrics": None,
            }
            recon_rel = entry_in.get("recon_path")
            if recon_rel is None:
                if not entry["exists"]:
                    _LOGGER.warning(
                        "[%s] column=%s strategy=%s: missing run, no reconstruction.",
                        scene,
                        entry["column"],
                        entry["strategy"],
                    )
                else:
                    _LOGGER.warning(
                        "[%s] column=%s strategy=%s: no reconstruction saved "
                        "(empty geometry); leaving cell empty.",
                        scene,
                        entry["column"],
                        entry["strategy"],
                    )
                scene_entries.append(entry)
                continue

            reconstruction = read_point_set_ply(intermediate_dir / recon_rel)
            metrics = compute_reconstruction_metrics(
                reconstruction,
                reference,
                eth3d_project_path,
                args.fscore_threshold_meters,
                args.temp_dir_override,
            )
            entry["metrics"] = metrics
            _LOGGER.info(
                "[%s] column=%s strategy=%s -> F=%.4f P=%.4f R=%.4f (thr=%.3f m)",
                scene,
                entry["column"],
                entry["strategy"],
                metrics["fscore"],
                metrics["precision"],
                metrics["recall"],
                args.fscore_threshold_meters,
            )
            scene_entries.append(entry)

        resolved_runs[scene] = scene_entries

        # Periodically write the JSON so partial results survive interruption.
        output = {
            "dataset": manifest.get("dataset"),
            "scenes": [str(s) for s in scenes],
            "columns": list(columns),
            "fscore_threshold_meters": args.fscore_threshold_meters,
            "resolved_runs": resolved_runs,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(output, f, indent=2)

    _LOGGER.info("Wrote reconstruction accuracy metrics to %s", args.output)

    render_latex_table(
        args.latex_output,
        args.output,
        args.latex_metrics,
        resolved_runs,
        args.fscore_threshold_meters,
        scene_list=args.table_scenes,
    )


if __name__ == "__main__":
    main()
