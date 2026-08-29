"""
GPU stage of the two-stage final-reconstruction accuracy evaluation.

For every scene x init-method x densification-strategy cell this script renders
the trained (and "at init") reconstructions and TSDF-fuses them into comparable
point sets, writing each reconstruction (plus the per-scene processed laser-scan
reference) as a PLY under ``--intermediate-dir`` together with a
``manifest.json`` describing every cell.

The CPU stage (``eval_final_recon_accuracy_cpu.py``) then loads these
reconstructions and computes the geometry metrics / LaTeX table. Splitting this
way keeps the GPU-only work (gsplat rendering + TSDF fusion) on a GPU node while
the metric computation (KD-tree F-score / external ETH3D tool) runs on a
CPU-only node.

See ``eval_final_recon_accuracy_common.py`` for the shared implementation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro

from eval_scripts.common.dataset_scenes import get_scenes_from_args

from eval_final_recon_accuracy_common import (
    AT_INIT_ROW_LABEL,
    DEFAULT_TSDF_BLOCK_COUNT,
    DEFAULT_VOXEL_SIZE,
    REFERENCE_FILENAME,
    apply_cpu_thread_limit,
    build_columns,
    build_init_reconstruction,
    load_scene_geometry_inputs,
    process_laser_scan_point_cloud,
    process_splats_via_tsdf,
    resolve_runs_for_scene,
    resolve_world_frame_splats,
    save_manifest,
    save_reconstruction_ply,
    load_trained_splats,
    _is_empty_tsdf_error,
    _sanitize_for_path,
)

_LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    # Directory the rendered / TSDF-fused reconstructions and the manifest are
    # written to (input to the CPU stage). Must live on a filesystem shared with
    # the CPU-stage node.
    intermediate_dir: Path

    # One column per entry, each "init_method=init_method_config" (the config may
    # itself contain '='). These are the columns of the output table, e.g.
    #   --init-methods da3=max_num_images=30 laser_scan=default sfm=default
    init_methods: list[str] = field(default_factory=list)

    # ivd-splat training config strings (rows of the table, i.e. densification
    # strategies), exactly as accepted by `ivd_splat_runner --configs`.
    # Shared across all columns, optionally with a per-init-method suffix (see
    # --ivd-splat-config-suffix).
    ivd_splat_configs: list[str] = field(default_factory=lambda: [""])

    # Optional per-init-method suffix appended to every --ivd-splat-configs entry
    # for that init method, keyed by the exact --init-methods spec. Example:
    #   --ivd-splat-configs-suffix da3=default "increase_scale_with_fewer_splats=False"
    # makes the da3=default column use, for each base config string,
    #   f"{base_config_string} increase_scale_with_fewer_splats=False".
    ivd_splat_configs_suffix: dict[str, str] = field(default_factory=dict)

    # Scenes to evaluate, in "dataset/scene" form or as local paths.
    # Takes precedence over --dataset when non-empty.
    scenes: list[str] = field(default_factory=list)
    # Dataset to expand into scenes when --scenes is not given.
    dataset: str | None = "scannet++"

    # Base results directory containing the trained method outputs.
    results_dir: Path = Path("results")

    # Directory to use for temporary files (trained-output archive extraction).
    # When set, temporary directories are created here instead of the system
    # default (``$TMPDIR`` / ``/tmp``). Useful on clusters where the default temp
    # location is a slow/network filesystem; point this at fast local scratch.
    temp_dir_override: Path | None = None

    # Maximum number of CPU threads / parallel workers to use for CPU-bound work
    # (Open3D CPU TSDF integration/extraction, the init-alignment KD-tree query,
    # and torch CPU ops). When None, all available cores are used.
    max_cpu_threads: int | None = None

    # External data needed to reproduce trained output directory names 1:1 with
    # ivd_splat_runner (must match what was passed at training time).
    gaussian_cap_per_scene_file: str | None = None
    gaussian_cap_fraction: float = 1.0
    init_size_per_scene_file: str | None = None
    extra_tags: list[str] = field(default_factory=list)
    eval_iter: int = 0

    # Voxel size for point merging, in meters (world / laser-scan frame). Also
    # used as the TSDF voxel length and the laser-scan downsample voxel size
    # (both grids are origin-aligned).
    voxel_size: float = DEFAULT_VOXEL_SIZE

    # gsplat depth-render near/far planes, in meters (world / laser-scan frame).
    # far_plane also acts as the TSDF depth truncation.
    near_plane: float = 0.01
    far_plane: float = 100.0
    # TSDF signed-distance truncation as a multiple of the voxel size.
    tsdf_sdf_trunc_voxel_multiplier: float = 3.0
    # Minimum accumulated splat alpha for a rendered depth pixel to be fused.
    min_render_alpha: float = 0.5
    # TSDF fusion backend.
    #
    # "gpu" uses Open3D's CUDA tensor VoxelBlockGrid (needs a CUDA device and a
    # few GB of spare VRAM; auto-falls back to "cpu" when none is available).
    # Its metrics differ slightly from "cpu".
    #
    # "cpu" (default) uses Open3D's multithreaded ScalableTSDFVolume.
    tsdf_backend: Literal["gpu", "cpu"] = "cpu"

    # Number of 16^3 voxel blocks the GPU VoxelBlockGrid pre-allocates (~80 KB
    # each). Must exceed the number of occupied blocks in a scene or Open3D spams
    # "stdgpu::vector::size ... out of bounds" warnings and drops geometry. Raise
    # on large-memory GPUs (300k ~= 24 GB), lower on smaller ones. Only used by
    # the "gpu" backend.
    tsdf_block_count: int = DEFAULT_TSDF_BLOCK_COUNT

    # Integer factor by which to downscale the rendered (and TSDF-integrated)
    # images. 1 (default) renders at full dataset resolution; e.g. 4 renders at
    # 1/4 the width and height (~16x fewer pixels), which speeds up rendering and
    # TSDF integration at the cost of geometric detail. Intrinsics are scaled
    # accordingly.
    render_downscale: int = 1

    # Enable debug-level logging (e.g. per-scene camera render resolutions).
    debug: bool = False

    # For monodepth / da3 (point-cloud) trained runs, both the SfM-derived and
    # an init-point-derived normalization transform are tried and the better
    # aligned one is used. This is the acceptable median init-alignment distance
    # (meters): if neither transform aligns within it and the two are too close
    # to distinguish, resolve_world_frame_splats raises.
    init_transform_check_threshold_meters: float = 0.05
    init_transform_fatal_threshold_meters: float = 0.1

    debug_export_dir: Path | None = None


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

    columns = build_columns(
        args.init_methods, args.ivd_splat_configs, args.ivd_splat_configs_suffix
    )
    if not columns:
        raise ValueError("No init methods to evaluate. Provide --init-methods.")

    if len(list(args.scenes)) == 0 and args.dataset is None:
        raise ValueError("No scenes to evaluate. Provide --scenes or --dataset.")
    scenes = get_scenes_from_args(
        list(args.scenes), [args.dataset] if args.dataset is not None else []
    )
    if not scenes:
        raise ValueError("No scenes to evaluate. Provide --scenes or --dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    intermediate_dir = args.intermediate_dir
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    scenes_data: dict[str, dict] = {}
    for scene in scenes:
        scene = str(scene)
        runs = resolve_runs_for_scene(scene, columns, args)
        for run in runs:
            level = logging.INFO if run.exists else logging.WARNING
            _LOGGER.log(
                level,
                "[%s] column=%s strategy=%s -> %s%s",
                scene,
                run.column_label,
                run.strategy_id,
                run.output_dir,
                "" if run.exists else "  (MISSING)",
            )

        geometry = load_scene_geometry_inputs(scene)
        cameras = geometry.cameras

        scene_dir = intermediate_dir / _sanitize_for_path(scene)
        scene_dir.mkdir(parents=True, exist_ok=True)

        scene_debug_dir = (
            args.debug_export_dir / _sanitize_for_path(scene)
            if args.debug_export_dir is not None
            else None
        )

        # Reference (laser-scan) point set, processed once per scene and saved.
        # Not needed for ETH3D scenes, which are scored by the external ETH3D
        # tool instead.
        reference_rel: str | None = None
        if not geometry.is_eth3d:
            reference = process_laser_scan_point_cloud(
                geometry.laser_points_world,
                geometry.laser_colors,
                cameras,
                args.voxel_size,
                debug_export_dir=scene_debug_dir,
                debug_prefix="laser_scan",
            )
            reference_path = scene_dir / REFERENCE_FILENAME
            if save_reconstruction_ply(reference_path, reference):
                reference_rel = str(reference_path.relative_to(intermediate_dir))

        entries: list[dict] = []

        # "At Init" row: initial point cloud / splats (before training),
        # rendered/processed once per column and placed as the first rows.
        for column in columns:
            entry: dict = {
                "column": column.label,
                "strategy": AT_INIT_ROW_LABEL,
                "output_dir": None,
                "exists": True,
                "recon_path": None,
            }
            try:
                init_reconstruction = build_init_reconstruction(
                    column,
                    scene,
                    args,
                    geometry,
                    device,
                    debug_export_dir=scene_debug_dir,
                )
            except FileNotFoundError as exc:
                entry["exists"] = False
                _LOGGER.warning(
                    "[%s] No init geometry for column=%s: %s",
                    scene,
                    column.label,
                    exc,
                )
            else:
                recon_path = scene_dir / (
                    _sanitize_for_path(f"init__{column.label}") + ".ply"
                )
                if save_reconstruction_ply(recon_path, init_reconstruction):
                    entry["recon_path"] = str(recon_path.relative_to(intermediate_dir))
                _LOGGER.info(
                    "[%s] column=%s strategy=%s: saved init reconstruction (%d points)",
                    scene,
                    column.label,
                    AT_INIT_ROW_LABEL,
                    init_reconstruction.points.shape[0],
                )
            entries.append(entry)

        column_by_label = {column.label: column for column in columns}
        for run in runs:
            entry = {
                "column": run.column_label,
                "strategy": run.row_id,
                "output_dir": str(run.output_dir),
                "exists": run.exists,
                "recon_path": None,
            }
            if not run.exists:
                _LOGGER.warning(
                    "[%s] Skipping missing run column=%s strategy=%s (path %s)",
                    scene,
                    run.column_label,
                    run.strategy_id,
                    run.output_dir,
                )
                entries.append(entry)
                continue

            splats = load_trained_splats(run.output_dir, args.temp_dir_override)
            # Bring the trained splats back from the normalized frame into the
            # world / laser-scan frame so metrics are computed at metric scale.
            # For older monodepth / da3 point-cloud runs the normalization
            # transform may have been derived from the init points instead of
            # the SfM points; resolve_world_frame_splats verifies and corrects
            # this.
            splats_world = resolve_world_frame_splats(
                splats, column_by_label[run.column_label], scene, args, geometry
            )
            prefix = _sanitize_for_path(f"{run.column_label}__{run.strategy_id}")
            try:
                reconstruction = process_splats_via_tsdf(
                    splats_world,
                    cameras,
                    args.voxel_size,
                    args.near_plane,
                    args.far_plane,
                    args.tsdf_sdf_trunc_voxel_multiplier,
                    args.min_render_alpha,
                    device,
                    debug_export_dir=scene_debug_dir,
                    debug_prefix=prefix,
                    tsdf_backend=args.tsdf_backend,
                    render_downscale=args.render_downscale,
                    tsdf_block_count=args.tsdf_block_count,
                )
            except RuntimeError as exc:
                if not _is_empty_tsdf_error(exc):
                    raise
                _LOGGER.warning(
                    "[%s] column=%s strategy=%s: TSDF fusion produced no geometry "
                    "(%s); leaving cell empty.",
                    scene,
                    run.column_label,
                    run.strategy_id,
                    exc,
                )
                entries.append(entry)
                continue

            recon_path = scene_dir / (prefix + ".ply")
            if save_reconstruction_ply(recon_path, reconstruction):
                entry["recon_path"] = str(recon_path.relative_to(intermediate_dir))
            _LOGGER.info(
                "[%s] column=%s strategy=%s: saved reconstruction (%d points)",
                scene,
                run.column_label,
                run.strategy_id,
                reconstruction.points.shape[0],
            )
            entries.append(entry)

        scenes_data[scene] = {
            "is_eth3d": geometry.is_eth3d,
            "eth3d_meshlab_project_path": (
                str(geometry.eth3d_meshlab_project_path)
                if geometry.eth3d_meshlab_project_path is not None
                else None
            ),
            "reference_path": reference_rel,
            "entries": entries,
        }

        # Write the manifest after each scene so partial results survive an
        # interrupted run.
        manifest = {
            "dataset": args.dataset,
            "scenes": [str(s) for s in scenes],
            "columns": [c.label for c in columns],
            "scenes_data": scenes_data,
        }
        save_manifest(intermediate_dir, manifest)

    _LOGGER.info(
        "GPU stage complete. Wrote reconstructions + manifest to %s",
        intermediate_dir,
    )


if __name__ == "__main__":
    main()
