from __future__ import annotations

import copy
import functools
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import open3d as o3d
import open3d.core as o3c
import torch
import tyro
from torch import Tensor
from torch.utils.dlpack import to_dlpack
from scipy.spatial import cKDTree
from gsplat.rendering import rasterization
from nerfbaselines.datasets import load_dataset
from nerfbaselines.utils import pad_poses

from eval_scripts.common.dataset_scenes import (
    get_scenes_from_args,
    scene_id_to_nerfbaselines_data_value,
)
from eval_scripts.common.results_dir import ResultsDirectory
from eval_scripts.common.config_strings import load_configs
from eval_scripts.ivd_splat_runner import (
    CONFIG_STR_PARAM_RENAMES,
    IVDRunnerArguments,
    resolve_trained_output_dir,
)

from ivd_splat.datasets.normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_points,
)
from ivd_splat.initialization import (
    decompose_rotation_translation_and_uniform_scale,
    rotate_quaternions,
    transform_shs,
)
from shared.point_cloud_io import load_pointcloud_ply
from shared.splat_ply_io import load_splat_ply, SplatData
from shared.save_init_info import INIT_INFO_JSON_FILENAME

_LOGGER = logging.getLogger(__name__)

DEFAULT_VOXEL_SIZE = 0.02


# GPU TSDF (Open3D tensor ``VoxelBlockGrid``) sparse-hash parameters. Blocks are
# 16^3 voxels; each voxel stores tsdf + weight + rgb (~20 bytes), so a block is
# ~80 KB. 80k blocks is ~6.4 GB, comfortably within a 16 GB budget (and far more
# on the 40 GB cluster GPUs) while giving many more blocks than a room-scale
# scene needs (the grid errors if exceeded, so it is kept generous).
_TSDF_BLOCK_RESOLUTION = 16
_TSDF_BLOCK_COUNT = 80000
# Minimum accumulated voxel weight for a voxel to contribute to the extracted
# GPU point cloud. Open3D's tensor default is 3.0; 1.0 keeps voxels observed by
# at least one frame, which produces a point set closer to the CPU backend (a
# higher threshold prunes many more voxels and diverges further). The tensor
# VoxelBlockGrid extracts a slightly thicker band than the CPU iso-surface, so
# the two backends do not produce identical point sets regardless of this value.
_TSDF_GPU_WEIGHT_THRESHOLD = 1.0


@functools.lru_cache(maxsize=1)
def _default_device() -> torch.device:
    """The device used for GPU-accelerated geometry ops (CUDA when available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Max CPU workers/threads for parallel CPU-bound ops (KD-tree queries, Open3D,
# torch CPU ops). -1 follows the SciPy KD-tree convention of "use all cores";
# overridden by --max-cpu-threads via ``_apply_cpu_thread_limit``.
_CPU_WORKERS = -1


def _apply_cpu_thread_limit(max_threads: int | None) -> None:
    """
    Limit the number of CPU threads / parallel workers used for CPU-bound work:
    torch CPU intra-op threads, Open3D's parallel ops (TSDF integrate/extract),
    and the KD-tree nearest-neighbour queries (via the module-level
    ``_CPU_WORKERS`` used in ``_nearest_neighbor_distances``).

    When ``max_threads`` is None, defaults are left untouched (all cores used).
    """
    global _CPU_WORKERS
    if max_threads is None:
        return
    if max_threads < 1:
        raise ValueError(f"max_cpu_threads must be >= 1, got {max_threads}.")
    _CPU_WORKERS = max_threads
    torch.set_num_threads(max_threads)
    # Open3D and the BLAS backends used by numpy/scipy have no runtime
    # thread-count API in this build; they honour these OpenMP/BLAS environment
    # variables instead. Setting them here caps thread pools created after this
    # point; for full effect start the process with them already exported.
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(max_threads)
    _LOGGER.info("Limited CPU threads/workers to %d.", max_threads)


# Spherical harmonics DC -> RGB constant (see ivd_splat.utils.runner_utils).
_SH_C0 = 0.28209479177387814

# Row label for the pre-training ("at initialization") F-score row.
AT_INIT_ROW_LABEL = "At Init"

# Name of the archived training output inside a trained run directory, and the
# path (relative to the archive root) of the final trained splat PLY within it.
TRAINED_OUTPUT_ARCHIVE_NAME = "output.zip"
TRAINED_SPLATS_ARCHIVE_MEMBER = "checkpoint/splats_30000.ply"


@dataclass
class InitMethodColumn:
    """
    One column of the output table: an initialization method + its init-method
    config (the ``init_method=init_method_config`` pair given on the CLI), together
    with the ivd-splat training config strings (the table rows / densification
    strategies) to evaluate for this column.
    """

    init_method: str
    init_method_config: str
    # ivd-splat training configs (as accepted by ivd_splat_runner) whose trained
    # outputs form the rows of the table for this column. Each entry is a
    # ``(base_config, full_config)`` pair, where ``full_config`` is
    # ``base_config`` with this column's per-init-method suffix appended. The
    # table row is keyed by ``base_config`` so suffixed variants of the same base
    # strategy collapse into a single row.
    training_configs: list[tuple[str, str]]

    @property
    def label(self) -> str:
        """Column identifier used in the output table and file names."""
        return f"{self.init_method}={self.init_method_config}"


def parse_init_method(spec: str) -> tuple[str, str]:
    """
    Parse an ``init_method=init_method_config`` spec. The config may itself contain
    ``=``, so only the first ``=`` separates the two, e.g. ``da3=max_num_images=30``
    -> init method ``da3``, init-method config ``max_num_images=30``.
    """
    if "=" not in spec:
        raise ValueError(
            f"Invalid --init-methods entry '{spec}'. "
            "Expected 'init_method=init_method_config'."
        )
    init_method, init_method_config = spec.split("=", 1)
    init_method, init_method_config = init_method.strip(), init_method_config.strip()
    if not init_method or not init_method_config:
        raise ValueError(
            f"Invalid --init-methods entry '{spec}'. "
            "Expected 'init_method=init_method_config'."
        )
    return init_method, init_method_config


@dataclass
class Args:
    # One column per entry, each "init_method=init_method_config" (the config may
    # itself contain '='). These are the columns of the output table, e.g.
    #   --init-methods da3=max_num_images=30 laser_scan=default sfm=default
    # Required unless --load-existing is set.
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
    # (KD-tree nearest-neighbour queries, Open3D TSDF integration/extraction, and
    # torch CPU ops). When None, all available cores are used. Set this to avoid
    # oversubscribing shared cluster nodes.
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
    # Its point-cloud extraction returns a slightly thicker near-surface band
    # than the CPU iso-surface, so its metrics differ slightly from "cpu".
    #
    # "cpu" (default) uses Open3D's multithreaded ScalableTSDFVolume and is the
    # verified, reference path.
    tsdf_backend: Literal["gpu", "cpu"] = "cpu"

    # Integer factor by which to downscale the rendered (and TSDF-integrated)
    # images. 1 (default) renders at full dataset resolution; e.g. 4 renders at
    # 1/4 the width and height (~16x fewer pixels), which speeds up rendering and
    # TSDF integration at the cost of geometric detail. Intrinsics are scaled
    # accordingly.
    render_downscale: int = 1

    # Enable debug-level logging (e.g. per-scene camera render resolutions).
    debug: bool = False

    # F-score inlier distance threshold, in meters (on the GT / laser-scan
    # scale). A reconstruction/GT point counts as matched when its nearest
    # neighbour in the other cloud is within this distance.
    fscore_threshold_meters: float = 0.05

    # For monodepth / da3 (point-cloud) trained runs, both the SfM-derived and
    # an init-point-derived normalization transform are tried and the better
    # aligned one is used. This is the acceptable median init-alignment distance
    # (meters): if neither transform aligns within it and the two are too close
    # to distinguish, resolve_world_frame_splats raises.
    init_transform_check_threshold_meters: float = 0.05
    init_transform_fatal_threshold_meters: float = 0.1

    # Output JSON file for per-scene and aggregated metrics.
    output: Path = Path("final_recon_accuracy.json")

    # Skip all recomputation and instead load previously computed metrics from
    # the --output JSON file, then (re)write the LaTeX table from them. Useful to
    # re-render the table with different --latex-metrics / --latex-output without
    # rerunning the (expensive) geometry evaluation.
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

    debug_export_dir: Path | None = None


def build_columns(args: Args) -> list[InitMethodColumn]:
    """
    Build the table columns from --init-methods, appending the optional
    per-init-method --ivd-splat-config-suffix to each shared --ivd-splat-configs
    entry.
    """
    unknown = set(args.ivd_splat_configs_suffix) - set(args.init_methods)
    if unknown:
        raise ValueError(
            "--ivd-splat-config-suffix keys must match --init-methods entries "
            f"exactly. Unknown keys: {sorted(unknown)}."
        )

    columns: list[InitMethodColumn] = []
    for spec in args.init_methods:
        init_method, init_method_config = parse_init_method(spec)
        suffix = args.ivd_splat_configs_suffix.get(spec, "")
        training_configs = [
            (base, f"{base} {suffix}".strip() if suffix else base)
            for base in args.ivd_splat_configs
        ]
        columns.append(
            InitMethodColumn(
                init_method=init_method,
                init_method_config=init_method_config,
                training_configs=training_configs,
            )
        )
    return columns


def _runner_args_for_column(column: InitMethodColumn, args: Args) -> IVDRunnerArguments:
    """
    Construct the ivd_splat_runner arguments that reproduce the trained output
    directory names for a given column, mirroring how the runner was invoked.
    """
    return IVDRunnerArguments(
        init_method=column.init_method,
        init_method_config=column.init_method_config,
        method="ivd-splat",
        output_dir=args.results_dir,
        gaussian_cap_per_scene_file=args.gaussian_cap_per_scene_file,
        gaussian_cap_fraction=args.gaussian_cap_fraction,
        init_size_per_scene_file=args.init_size_per_scene_file,
        extra_tags=list(args.extra_tags),
        eval_iter=args.eval_iter,
    )


@dataclass
class ResolvedRun:
    """A single resolved trained ivd-splat run: one table cell for one scene."""

    scene: str
    column_label: str
    # Table row id: the config name of the *base* config string (before this
    # column's per-init-method suffix), so suffixed variants of the same base
    # strategy share a single row.
    row_id: str
    # Densification strategy id of the full (suffixed) config, unique per cell.
    # Used for file/debug prefixes and logging.
    strategy_id: str
    # The exact trained nerfbaselines output directory for this cell.
    output_dir: Path
    exists: bool


def resolve_runs_for_scene(
    scene: str, columns: list[InitMethodColumn], args: Args
) -> list[ResolvedRun]:
    """
    Resolve the trained output directories for every (column, strategy) cell of a
    scene, reproducing ivd_splat_runner's directory names 1:1 from the config
    strings and the provided external data.
    """
    results_dir = ResultsDirectory(args.results_dir)
    runs: list[ResolvedRun] = []
    for column in columns:
        runner_args = _runner_args_for_column(column, args)
        for base_config, config_string in column.training_configs:
            # Each config string must map to exactly one trained run here: the
            # multi-run {a,b} expansion is only meaningful for training, whereas
            # this table has exactly one cell per densification strategy x init
            # method x init-method config.
            param_lists = load_configs([config_string], None)
            if len(param_lists) != 1:
                raise ValueError(
                    f"Config string '{config_string}' expands to {len(param_lists)} "
                    "runs, but exactly one is required here. Remove the '{a,b}' "
                    "multi-value expansion and specify a single value per parameter."
                )
            param_list = param_lists[0]
            strategy_id = param_list.make_config_name(CONFIG_STR_PARAM_RENAMES)
            # Row id comes from the base config only, so per-init-method suffixes
            # collapse into a single row.
            row_id = load_configs([base_config], None)[0].make_config_name(
                CONFIG_STR_PARAM_RENAMES
            )

            runner_args_copy = copy.deepcopy(runner_args)
            param_list_list = list(param_list)

            # Special case - base SfM with this strat has no cap.
            if (
                column.init_method == "sfm"
                and len(param_list_list) == 1
                and param_list_list[0]
                == (
                    "strategy",
                    "DefaultWithGaussianCapStrategy",
                )
            ):
                runner_args_copy.gaussian_cap_per_scene_file = None
                runner_args_copy.extra_tags = []

            resolved = resolve_trained_output_dir(
                results_dir, column.init_method, param_list, scene, runner_args_copy
            )
            runs.append(
                ResolvedRun(
                    scene=scene,
                    column_label=column.label,
                    row_id=row_id,
                    strategy_id=strategy_id,
                    output_dir=resolved.output_dir,
                    exists=resolved.output_dir.exists(),
                )
            )
    return sorted(runs, key=lambda r: (r.row_id, r.column_label))


# --------------------------------------------------------------------------- #
# Point set loading
# --------------------------------------------------------------------------- #


@dataclass
class PointSet:
    points: np.ndarray  # (N, 3) float
    colors: np.ndarray | None  # (N, 3) float in [0, 1], or None


@dataclass
class Cameras:
    """
    Training cameras for a scene, expressed in the *world* (laser-scan / SfM)
    frame. This is the frame in which the laser scan lives and into which the
    trained splats are transformed back, so that all geometry metrics are
    computed at true metric (meter) scale.
    """

    camtoworlds: np.ndarray  # (C, 4, 4) camera-to-world in the world frame
    Ks: np.ndarray  # (C, 3, 3) pinhole intrinsics
    widths: np.ndarray  # (C,) int image widths
    heights: np.ndarray  # (C,) int image heights

    def __len__(self) -> int:
        return self.camtoworlds.shape[0]


def _extract_archive_member(archive_path: Path, member: str, dest_dir: Path) -> Path:
    """
    Extract a single ``member`` from the zip ``archive_path`` into ``dest_dir``,
    returning the path of the extracted file.

    Prefers the system ``unzip`` binary, which is substantially faster than
    Python's ``zipfile`` for large archives; falls back to ``zipfile`` when
    ``unzip`` is unavailable or does not produce the expected file.
    """
    out_path = dest_dir / member
    if shutil.which("unzip") is not None:
        try:
            subprocess.run(
                ["unzip", "-o", "-q", str(archive_path), member, "-d", str(dest_dir)],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            _LOGGER.warning(
                "unzip failed for %s!%s (%s); falling back to zipfile.",
                archive_path,
                member,
                exc,
            )
        else:
            if out_path.exists():
                return out_path
    with zipfile.ZipFile(archive_path) as archive:
        return Path(archive.extract(member, dest_dir))


def load_trained_splats(
    output_dir: Path, temp_dir_override: Path | None = None
) -> SplatData:
    """
    Load the final trained splats for a run.

    The trained output is stored as ``output.zip`` inside the run directory; the
    final splat PLY lives at ``checkpoint/splats_30000.ply`` relative to the
    archive root. The archive member is extracted into a temporary directory
    (removed on return) and the PLY is loaded with the shared splat IO helpers.

    When ``temp_dir_override`` is set, the temporary directory is created there
    instead of the system default (useful to avoid a slow default temp
    filesystem on clusters).
    """
    archive_path = output_dir / TRAINED_OUTPUT_ARCHIVE_NAME
    if not archive_path.exists():
        raise FileNotFoundError(f"Trained output archive not found: {archive_path}")

    if temp_dir_override is not None:
        temp_dir_override.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=temp_dir_override) as tmp_dir:
        extracted = _extract_archive_member(
            archive_path, TRAINED_SPLATS_ARCHIVE_MEMBER, Path(tmp_dir)
        )
        _LOGGER.info(
            "Loading trained splats from %s!%s",
            archive_path,
            TRAINED_SPLATS_ARCHIVE_MEMBER,
        )
        return load_splat_ply(extracted)


def transform_splat_data(splats: SplatData, transform: np.ndarray) -> SplatData:
    """
    Apply a 4x4 similarity ``transform`` to trained Gaussian splats, mirroring
    ivd_splat's ``splat_init``: means go through the full transform, scales are
    multiplied by the uniform scale factor (in log space), and rotations /
    higher-order SH are rotated by the rotation part. Opacities and the DC SH
    term (``sh0``) are rotation-invariant and left unchanged.

    Used here to bring normalized-frame trained splats back into the world /
    laser-scan frame by passing the inverse normalization transform, so the
    rendered geometry is at true metric scale.
    """
    rotation, _translation, scale = decompose_rotation_translation_and_uniform_scale(
        transform
    )
    rot_torch = torch.from_numpy(rotation).float()

    means = np.asarray(splats.means, dtype=np.float64)
    means_world = transform_points(transform, means)

    scales = np.asarray(splats.scales, dtype=np.float64)
    scales_world = np.log(np.exp(scales) * scale)

    quats = torch.as_tensor(np.asarray(splats.quats), dtype=torch.float32)
    quats_world = rotate_quaternions(quats, rot_torch)

    shN = torch.as_tensor(np.asarray(splats.shN), dtype=torch.float32)
    shN_world = transform_shs(shN, rot_torch)

    return SplatData(
        means=torch.from_numpy(means_world).float(),
        scales=torch.from_numpy(scales_world).float(),
        quats=quats_world,
        opacities=torch.as_tensor(np.asarray(splats.opacities), dtype=torch.float32),
        sh0=torch.as_tensor(np.asarray(splats.sh0), dtype=torch.float32),
        shN=shN_world,
    )


def _build_cameras(cams) -> Cameras:
    """
    Build world-frame ``Cameras`` from a nerfbaselines cameras object.

    The dataset poses are already valid camera-to-world matrices in the world
    (laser-scan / SfM) frame, so they are used as-is (no normalization), keeping
    rendered depths and projected points at true metric scale.
    """
    camtoworlds = np.asarray(pad_poses(cams.poses), dtype=np.float64)

    intrinsics = np.asarray(cams.intrinsics, dtype=np.float64)  # (C, 4): fx, fy, cx, cy
    num_cameras = intrinsics.shape[0]
    Ks = np.zeros((num_cameras, 3, 3), dtype=np.float64)
    Ks[:, 0, 0] = intrinsics[:, 0]
    Ks[:, 1, 1] = intrinsics[:, 1]
    Ks[:, 0, 2] = intrinsics[:, 2]
    Ks[:, 1, 2] = intrinsics[:, 3]
    Ks[:, 2, 2] = 1.0

    image_sizes = np.asarray(cams.image_sizes, dtype=np.int64)  # (C, 2): width, height
    return Cameras(
        camtoworlds=np.asarray(camtoworlds, dtype=np.float64),
        Ks=Ks,
        widths=image_sizes[:, 0],
        heights=image_sizes[:, 1],
    )


@dataclass
class SceneGeometryInputs:
    """Everything needed to process one scene's geometry (loaded in one pass)."""

    # Dense laser-scan point cloud in the *world* frame (colors in [0, 1] or None).
    laser_points_world: np.ndarray
    laser_colors: np.ndarray | None
    # Sparse SfM points in the *world* frame (the ``sfm`` init) and their colors.
    sfm_points_world: np.ndarray
    sfm_colors: np.ndarray | None
    # Training cameras in the *world* frame.
    cameras: Cameras
    # 4x4 world -> normalized-frame transform (the frame the trained splats live
    # in); its inverse maps splats back into the world frame.
    transform: np.ndarray


def compute_normalization_transform(
    points_world: np.ndarray, camtoworlds: np.ndarray
) -> np.ndarray:
    """
    Compute the 4x4 world -> normalized-frame transform exactly as ivd_splat's
    NerfbaselinesParser does: a similarity transform derived from the camera
    poses (normalizing scene scale), followed by a principal-axes alignment of
    ``points_world``.

    ``points_world`` is whatever point cloud was present in the dataset at
    training time (the SfM points normally, but the init method's own points for
    older runs that replaced the SfM points in the dataset).
    """
    t1 = similarity_from_cameras(camtoworlds)
    t2 = align_principle_axes(transform_points(t1, points_world))
    return t2 @ t1


def load_scene_geometry_inputs(scene: str) -> SceneGeometryInputs:
    """
    Load everything needed to process a scene's geometry, loading the
    nerfbaselines dataset only once.

    The transform is computed exactly as in ivd_splat's NerfbaselinesParser:
    a similarity transform derived from the camera poses (which normalizes the
    scene scale), followed by a principal-axes alignment of the SfM points.
    """
    nb_data_value = scene_id_to_nerfbaselines_data_value(scene)
    dataset = load_dataset(
        nb_data_value, "train", features=["points3D_xyz", "points3D_rgb"]
    )

    sfm_points = np.asarray(dataset["points3D_xyz"], dtype=np.float64)
    sfm_rgbs = dataset.get("points3D_rgb")
    sfm_colors: np.ndarray | None = None
    if sfm_rgbs is not None:
        sfm_colors = np.asarray(sfm_rgbs, dtype=np.float64)
        if sfm_colors.size and sfm_colors.max() > 1.0:
            sfm_colors = sfm_colors / 255.0

    cameras = _build_cameras(dataset["cameras"])
    transform = compute_normalization_transform(sfm_points, cameras.camtoworlds)

    if "dense_points3D_path" not in dataset["metadata"]:
        raise ValueError(
            f"Dataset {nb_data_value} does not contain a dense point cloud path in its metadata."
        )
    dense_points_path = Path(dataset["metadata"]["dense_points3D_path"])

    _LOGGER.info(
        "Loading laser scan points from path specified in Nerfbaselines dataset metadata: %s",
        dense_points_path,
    )
    points, rgbs = load_pointcloud_ply(dense_points_path)
    # load_pointcloud_ply already returns colors in the [0, 1] range (or None).
    colors = np.asarray(rgbs, dtype=np.float64) if rgbs is not None else None

    return SceneGeometryInputs(
        laser_points_world=np.asarray(points, dtype=np.float64),
        laser_colors=colors,
        sfm_points_world=sfm_points,
        sfm_colors=sfm_colors,
        cameras=cameras,
        transform=transform,
    )


def _sanitize_for_path(name: str) -> str:
    """Make a string safe to use as a file/directory name."""
    return "".join(c if c.isalnum() or c in "-._=" else "_" for c in name)


def _write_point_set_ply(out_path: Path, point_set: PointSet) -> None:
    """Write a point set to a PLY file (for debugging)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set.points)
    if point_set.colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(point_set.colors)
    o3d.io.write_point_cloud(str(out_path), pcd)
    _LOGGER.info(
        "Exported debug point set (%d points) to %s",
        point_set.points.shape[0],
        out_path,
    )


# --------------------------------------------------------------------------- #
# Geometry processing paths (comparable point sets for accuracy/completeness)
# --------------------------------------------------------------------------- #


def voxel_downsample_aligned(
    points: np.ndarray, colors: np.ndarray | None, voxel_size: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Voxel-downsample points on a grid whose voxels are aligned to the origin
    ``(0, 0, 0)`` (voxel index = ``floor(point / voxel_size)``), averaging the
    points (and colors) that fall into each occupied voxel.

    Aligning to the origin makes this grid coincide with Open3D's TSDF voxel grid
    (used for the splat path), so the two comparable point sets share voxels.

    Runs on the GPU (when available): points are bucketed into voxels via a
    linear voxel index and averaged with ``scatter_add``. This is far faster than
    a CPU ``np.unique`` over tens of millions of points.
    """
    if points.shape[0] == 0:
        return points, colors

    device = _default_device()
    pts = torch.as_tensor(points, dtype=torch.float64, device=device)

    keys = torch.floor(pts / voxel_size).to(torch.int64)  # (N, 3)
    keys -= keys.amin(dim=0)  # shift to non-negative voxel indices
    dims = keys.amax(dim=0) + 1  # (3,) grid extent per axis

    # Prefer a compact 1D linear voxel index (fast unique); fall back to a row
    # unique when the grid extent would overflow int64 (e.g. far-away floaters).
    if torch.prod(dims.double()).item() < float(2**62):
        lin = (keys[:, 0] * dims[1] + keys[:, 1]) * dims[2] + keys[:, 2]  # (N,)
        _, inverse = torch.unique(lin, return_inverse=True)
    else:
        _, inverse = torch.unique(keys, dim=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    num_voxels = int(inverse.amax().item()) + 1

    counts = torch.zeros(num_voxels, dtype=torch.float64, device=device)
    counts.scatter_add_(0, inverse, torch.ones_like(inverse, dtype=torch.float64))

    idx3 = inverse.unsqueeze(1).expand(-1, 3)
    sums = torch.zeros((num_voxels, 3), dtype=torch.float64, device=device)
    sums.scatter_add_(0, idx3, pts)
    down_points = (sums / counts.unsqueeze(1)).cpu().numpy()

    down_colors = None
    if colors is not None:
        col = torch.as_tensor(colors, dtype=torch.float64, device=device)
        csums = torch.zeros((num_voxels, 3), dtype=torch.float64, device=device)
        csums.scatter_add_(0, idx3, col)
        down_colors = (csums / counts.unsqueeze(1)).cpu().numpy()

    return down_points, down_colors


def filter_points_visible_in_cameras(
    points: np.ndarray, colors: np.ndarray | None, cameras: Cameras
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Keep only points that project into the image frustum of at least one training
    camera (in front of the camera and within image bounds). This is a pure
    visibility/frustum test (no occlusion test).

    Projection is done on the GPU (when available), looping over cameras and
    OR-ing the per-camera visibility masks.
    """
    num_points = points.shape[0]
    if num_points == 0:
        return points, colors

    device = _default_device()
    pts = torch.as_tensor(points, dtype=torch.float32, device=device)
    homogeneous = torch.cat(
        [pts, torch.ones((num_points, 1), dtype=torch.float32, device=device)], dim=1
    )  # (N, 4)

    camtoworlds = torch.as_tensor(
        cameras.camtoworlds, dtype=torch.float32, device=device
    )  # (C, 4, 4)
    world_to_cams = torch.linalg.inv(camtoworlds)  # (C, 4, 4)
    Ks = torch.as_tensor(cameras.Ks, dtype=torch.float32, device=device)  # (C, 3, 3)
    widths = torch.as_tensor(cameras.widths, dtype=torch.float32, device=device)
    heights = torch.as_tensor(cameras.heights, dtype=torch.float32, device=device)

    visible = torch.zeros(num_points, dtype=torch.bool, device=device)
    for cam_idx in range(len(cameras)):
        cam_points = homogeneous @ world_to_cams[cam_idx].T  # (N, 4)
        z = cam_points[:, 2]
        in_front = z > 1e-6
        safe_z = torch.where(in_front, z, torch.ones_like(z))

        K = Ks[cam_idx]
        u = K[0, 0] * cam_points[:, 0] / safe_z + K[0, 2]
        v = K[1, 1] * cam_points[:, 1] / safe_z + K[1, 2]

        in_bounds = (
            in_front
            & (u >= 0)
            & (u < widths[cam_idx])
            & (v >= 0)
            & (v < heights[cam_idx])
        )
        visible |= in_bounds
        if bool(visible.all()):
            break

    vis_mask = visible.cpu().numpy()
    return points[vis_mask], (colors[vis_mask] if colors is not None else None)


def process_laser_scan_point_cloud(
    points_world: np.ndarray,
    colors: np.ndarray | None,
    cameras: Cameras,
    voxel_size: float,
    debug_export_dir: Path | None = None,
    debug_prefix: str = "laser_scan",
) -> PointSet:
    """
    Laser-scan processing path (all in the world / laser-scan frame): voxel-
    downsample the dense laser scan on the origin-aligned grid, and drop every
    point that is not visible in any training image.

    The result is directly comparable to the splat TSDF point set (same frame,
    same voxel grid, same visibility restriction), at true metric scale.

    When ``debug_export_dir`` is given, the voxel-downsampled cloud (before the
    visibility filter) and the final visible cloud are written there as PLYs.
    """
    points = np.asarray(points_world, dtype=np.float64)

    down_points, down_colors = voxel_downsample_aligned(points, colors, voxel_size)
    _LOGGER.info(
        "Laser scan voxel downsample: %d -> %d points (voxel size %.5f)",
        points.shape[0],
        down_points.shape[0],
        voxel_size,
    )
    if debug_export_dir is not None:
        _write_point_set_ply(
            debug_export_dir / f"{debug_prefix}_voxel_downsampled.ply",
            PointSet(points=down_points, colors=down_colors),
        )

    vis_points, vis_colors = filter_points_visible_in_cameras(
        down_points, down_colors, cameras
    )
    _LOGGER.info(
        "Laser scan visibility filter: %d -> %d points visible in training cameras",
        down_points.shape[0],
        vis_points.shape[0],
    )
    result = PointSet(points=vis_points, colors=vis_colors)
    if debug_export_dir is not None:
        _write_point_set_ply(debug_export_dir / f"{debug_prefix}_visible.ply", result)
    return result


def _splat_render_inputs(
    splats: SplatData, device: torch.device
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, int]:
    """
    Build the activated gsplat rasterization inputs from trained ``SplatData``.

    Returns ``(means, quats, scales, opacities, sh_coeffs, sh_degree)`` where the
    activations mirror ``IVDRunner.rasterize_splats`` (scales = exp, opacities =
    sigmoid, quats normalized internally by gsplat) and ``sh_coeffs`` are the
    concatenated ``[sh0, shN]`` spherical-harmonics coefficients.
    """
    means = torch.as_tensor(splats.means, dtype=torch.float32, device=device)
    quats = torch.as_tensor(splats.quats, dtype=torch.float32, device=device)
    scales = torch.exp(
        torch.as_tensor(splats.scales, dtype=torch.float32, device=device)
    )
    opacities = torch.sigmoid(
        torch.as_tensor(splats.opacities, dtype=torch.float32, device=device)
    ).reshape(-1)

    sh0 = torch.as_tensor(splats.sh0, dtype=torch.float32, device=device)  # (N, 1, 3)
    shN = torch.as_tensor(splats.shN, dtype=torch.float32, device=device)  # (N, K-1, 3)
    sh_coeffs = torch.cat([sh0, shN], dim=1)  # (N, K, 3)

    num_sh = sh_coeffs.shape[1]
    sh_degree = int(round(num_sh**0.5)) - 1

    return means, quats, scales, opacities, sh_coeffs, sh_degree


def process_splats_via_tsdf(
    splats: SplatData,
    cameras: Cameras,
    voxel_size: float,
    near_plane: float,
    far_plane: float,
    sdf_trunc_voxel_multiplier: float,
    min_render_alpha: float,
    device: torch.device,
    debug_export_dir: Path | None = None,
    debug_prefix: str = "splats",
    tsdf_backend: str = "gpu",
    render_downscale: int = 1,
) -> PointSet:
    """
    Splat processing path: render a depth + color image from every training
    camera via gsplat, then TSDF-fuse the depths into a voxel grid whose voxel
    size and origin match the laser-scan downsampling grid, and extract the fused
    point cloud.

    Rendering happens in the world (laser-scan / SfM) frame: the cameras are in
    the world frame and the splats have been transformed back into it, so the
    fused points are directly comparable to the processed laser scan at true
    metric scale.

    Rendering is always done on the GPU (gsplat). The depth fusion backend is
    selected by ``tsdf_backend``:

    - ``"gpu"``: Open3D's CUDA tensor ``VoxelBlockGrid``. The rendered depth /
      color tensors are handed to Open3D as CUDA tensors (zero-copy via DLPack),
      keeping everything on the GPU. Requires a CUDA device and a few GB of spare
      VRAM; falls back to ``"cpu"`` when no CUDA device is available.
    - ``"cpu"``: Open3D's multithreaded ``ScalableTSDFVolume``.

    Both backends extract the same zero-crossing surface points (the GPU grid's
    extraction weight threshold is set to keep every observed voxel, matching the
    CPU path), so their outputs agree up to numerical differences.

    When ``debug_export_dir`` is given, the fused point cloud is written there as
    a PLY.
    """
    means, quats, scales, opacities, sh_coeffs, sh_degree = _splat_render_inputs(
        splats, device
    )

    use_gpu = tsdf_backend == "gpu" and device.type == "cuda"
    if tsdf_backend == "gpu" and device.type != "cuda":
        _LOGGER.warning(
            "tsdf_backend='gpu' requested but no CUDA device is available; "
            "falling back to the CPU ScalableTSDFVolume."
        )

    if use_gpu:
        o3d_device = o3c.Device(f"CUDA:{device.index or 0}")
        vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1,), (1,), (3,)),
            voxel_size=voxel_size,
            block_resolution=_TSDF_BLOCK_RESOLUTION,
            block_count=_TSDF_BLOCK_COUNT,
            device=o3d_device,
        )
    else:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=voxel_size * sdf_trunc_voxel_multiplier,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        )

    if _LOGGER.isEnabledFor(logging.DEBUG):
        full_sizes = sorted(
            {(int(w), int(h)) for w, h in zip(cameras.widths, cameras.heights)}
        )
        if render_downscale > 1:
            rendered_sizes = sorted(
                {
                    (
                        max(1, int(w) // render_downscale),
                        max(1, int(h) // render_downscale),
                    )
                    for w, h in zip(cameras.widths, cameras.heights)
                }
            )
            _LOGGER.debug(
                "Rendering %d cameras at 1/%d resolution: full (w, h) %s -> "
                "rendered (w, h) %s",
                len(cameras),
                render_downscale,
                full_sizes,
                rendered_sizes,
            )
        else:
            _LOGGER.debug(
                "Rendering %d cameras at full resolution (w, h): %s",
                len(cameras),
                full_sizes,
            )

    render_seconds = 0.0
    integrate_seconds = 0.0
    for cam_idx in range(len(cameras)):
        full_width = int(cameras.widths[cam_idx])
        full_height = int(cameras.heights[cam_idx])
        K = cameras.Ks[cam_idx]
        if render_downscale > 1:
            # Render (and TSDF-integrate) at a lower resolution to save time.
            # Scale the intrinsics by the actual width/height ratio (which may
            # differ slightly from 1/render_downscale when the size is not evenly
            # divisible) so the pinhole model stays consistent with the pixel grid.
            width = max(1, full_width // render_downscale)
            height = max(1, full_height // render_downscale)
            scale_x = width / full_width
            scale_y = height / full_height
            K = K.copy()
            K[0, 0] *= scale_x
            K[0, 2] *= scale_x
            K[1, 1] *= scale_y
            K[1, 2] *= scale_y
        else:
            width = full_width
            height = full_height
        camtoworld = cameras.camtoworlds[cam_idx]
        viewmat = np.linalg.inv(camtoworld)

        render_start = time.perf_counter()
        viewmats = torch.as_tensor(
            viewmat[None], dtype=torch.float32, device=device
        )  # (1, 4, 4)
        Ks = torch.as_tensor(K[None], dtype=torch.float32, device=device)  # (1, 3, 3)

        render_colors, render_alphas, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=sh_coeffs,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=sh_degree,
            near_plane=near_plane,
            far_plane=far_plane,
            render_mode="RGB+ED",
            camera_model="pinhole",
        )

        rgb = render_colors[0, ..., :3]  # (H, W, 3)
        depth = render_colors[0, ..., 3]  # (H, W) expected depth
        alpha = render_alphas[0, ..., 0]  # (H, W)

        # Drop depths where too little was rendered (expected depth is unreliable).
        depth = torch.where(alpha >= min_render_alpha, depth, torch.zeros_like(depth))

        if use_gpu:
            # Hand the rendered tensors to Open3D as device-resident tensors via
            # DLPack (zero-copy): depth (H, W, 1) float32, color (H, W, 3) float32.
            depth_t = depth.unsqueeze(-1).contiguous()
            color_t = rgb.clamp(0.0, 1.0).contiguous()
            depth_img = o3d.t.geometry.Image(o3c.Tensor.from_dlpack(to_dlpack(depth_t)))
            color_img = o3d.t.geometry.Image(o3c.Tensor.from_dlpack(to_dlpack(color_t)))
        else:
            color_np = np.ascontiguousarray(
                (rgb.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
            )
            depth_np = np.ascontiguousarray(depth.cpu().numpy().astype(np.float32))
        render_seconds += time.perf_counter() - render_start

        integrate_start = time.perf_counter()
        if use_gpu:
            intrinsic = o3c.Tensor(K, dtype=o3c.float64, device=o3c.Device("CPU:0"))
            extrinsic = o3c.Tensor(
                viewmat, dtype=o3c.float64, device=o3c.Device("CPU:0")
            )
            frustum_blocks = vbg.compute_unique_block_coordinates(
                depth_img,
                intrinsic,
                extrinsic,
                depth_scale=1.0,
                depth_max=far_plane,
                trunc_voxel_multiplier=sdf_trunc_voxel_multiplier,
            )
            vbg.integrate(
                frustum_blocks,
                depth_img,
                color_img,
                intrinsic,
                extrinsic,
                depth_scale=1.0,
                depth_max=far_plane,
                trunc_voxel_multiplier=sdf_trunc_voxel_multiplier,
            )
        else:
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_np),
                o3d.geometry.Image(depth_np),
                depth_scale=1.0,
                depth_trunc=far_plane,
                convert_rgb_to_intensity=False,
            )
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width,
                height,
                K[0, 0],
                K[1, 1],
                K[0, 2],
                K[1, 2],
            )
            volume.integrate(rgbd, intrinsic, viewmat)
        integrate_seconds += time.perf_counter() - integrate_start

    extract_start = time.perf_counter()
    if use_gpu:
        # Extract the fused surface points on the GPU.
        pcd = vbg.extract_point_cloud(
            weight_threshold=_TSDF_GPU_WEIGHT_THRESHOLD
        ).to_legacy()
    else:
        pcd = volume.extract_point_cloud()
    points = np.asarray(pcd.points, dtype=np.float64)
    colors = np.asarray(pcd.colors, dtype=np.float64) if pcd.has_colors() else None
    if colors is not None and colors.size:
        # Both backends already return colors in [0, 1] (verified: the CPU
        # ScalableTSDFVolume divides its stored uint8 color by 255, and the GPU
        # VoxelBlockGrid stores the [0, 1] color we feed it). Just clip: a few
        # surface points can get a slightly out-of-range interpolated color
        # (near-zero TSDF-weight denominators), and a global "max > 1 => /255"
        # rescale would then blacken the entire cloud from a single outlier.
        colors = np.clip(colors, 0.0, 1.0)
    extract_seconds = time.perf_counter() - extract_start
    _LOGGER.info(
        "TSDF fusion (%s) produced %d points (%d cams: render %.1fs, "
        "integrate %.1fs, extract %.1fs)",
        "gpu" if use_gpu else "cpu",
        points.shape[0],
        len(cameras),
        render_seconds,
        integrate_seconds,
        extract_seconds,
    )
    result = PointSet(points=points, colors=colors)
    if debug_export_dir is not None:
        _write_point_set_ply(
            debug_export_dir / f"{debug_prefix}_tsdf_fusion.ply", result
        )
    return result


# --------------------------------------------------------------------------- #
# Initialization ("at init") geometry
# --------------------------------------------------------------------------- #


def _find_init_ply(init_dir: Path) -> tuple[Path, str]:
    """
    Locate the geometry PLY produced by an init method and its init type.

    The init type (one of ``"splat"``, ``"dense"``, ``"sparse"``) and required
    files are read from ``init_info.json`` when present; otherwise the known
    per-method output filenames are used as a fallback.

    Raises ``FileNotFoundError`` when no geometry PLY can be located.
    """
    init_info_path = init_dir / INIT_INFO_JSON_FILENAME
    if init_info_path.exists():
        init_info = json.loads(init_info_path.read_text())
        init_type = init_info.get("init_type", "")
        for name in init_info.get("required_files", []):
            if name.endswith(".ply") and "sfm" not in name.lower():
                return init_dir / name, init_type

    for name, init_type in (
        ("da3_gaussians.ply", "splat"),
        ("da3_points.ply", "dense"),
        ("edgs.ply", "splat"),
        ("points3D.ply", "dense"),
    ):
        candidate = init_dir / name
        if candidate.exists():
            return candidate, init_type

    raise FileNotFoundError(f"Could not find an init geometry PLY in {init_dir}.")


def _splat_centers_point_set(splats: SplatData) -> PointSet:
    """Build a point set from splat centers (means), coloring points from sh0."""
    points = splats.means.detach().cpu().numpy().astype(np.float64)
    rgbs = (splats.sh0.squeeze(1) * _SH_C0 + 0.5).detach().cpu().numpy()
    colors = np.clip(rgbs, 0.0, 1.0).astype(np.float64)
    return PointSet(points=points, colors=colors)


def _process_point_cloud_reconstruction(
    point_set: PointSet,
    cameras: Cameras,
    voxel_size: float,
    debug_export_dir: Path | None = None,
    debug_prefix: str = "init",
) -> PointSet:
    """
    Make a point-cloud reconstruction comparable to the processed laser scan:
    voxel-downsample on the origin-aligned grid and keep only points visible in
    at least one training camera (same processing as the laser-scan reference).
    """
    down_points, down_colors = voxel_downsample_aligned(
        np.asarray(point_set.points, dtype=np.float64), point_set.colors, voxel_size
    )
    vis_points, vis_colors = filter_points_visible_in_cameras(
        down_points, down_colors, cameras
    )
    _LOGGER.info(
        "Init point cloud '%s': %d -> %d (downsample) -> %d (visible) points",
        debug_prefix,
        point_set.points.shape[0],
        down_points.shape[0],
        vis_points.shape[0],
    )
    result = PointSet(points=vis_points, colors=vis_colors)
    if debug_export_dir is not None:
        _write_point_set_ply(debug_export_dir / f"{debug_prefix}_visible.ply", result)
    return result


def build_init_reconstruction(
    column: InitMethodColumn,
    scene: str,
    args: Args,
    geometry: SceneGeometryInputs,
    device: torch.device,
    debug_export_dir: Path | None = None,
) -> PointSet:
    """
    Build the "at initialization" reconstruction point set for one column, in the
    world / laser-scan frame (init outputs already live in that frame, so no
    transform is applied). The result is directly comparable to the processed
    laser-scan reference.

    Per-method handling:
    - ``sfm``: the sparse SfM points from the dataset;
    - ``laser_scan``: the dense laser scan itself (the init is the GT scan);
    - point-cloud inits (e.g. ``monodepth``, ``da3`` without gaussians): the init
      point cloud, downsample + visibility filtered like the reference;
    - ``da3`` gaussians: TSDF depth fusion (same path as trained splats);
    - other splat inits (e.g. ``edgs``): the splat *centers* (EDGS splats are tiny
      and dense with opacity <= 0.5, so depth rendering is unreliable), downsample
      + visibility filtered.

    Raises ``FileNotFoundError`` when the init output cannot be located.
    """
    prefix = _sanitize_for_path(f"init__{column.label}")

    if column.init_method == "sfm":
        return _process_point_cloud_reconstruction(
            PointSet(geometry.sfm_points_world, geometry.sfm_colors),
            geometry.cameras,
            args.voxel_size,
            debug_export_dir,
            prefix,
        )
    if column.init_method == "laser_scan":
        return _process_point_cloud_reconstruction(
            PointSet(geometry.laser_points_world, geometry.laser_colors),
            geometry.cameras,
            args.voxel_size,
            debug_export_dir,
            prefix,
        )

    init_dir = ResultsDirectory(args.results_dir).get_init_method_output_dir(
        scene, column.init_method_config, column.init_method
    )
    ply_path, init_type = _find_init_ply(init_dir)

    if init_type == "splat":
        splats = load_splat_ply(ply_path)
        if column.init_method == "da3":
            return process_splats_via_tsdf(
                splats,
                geometry.cameras,
                args.voxel_size,
                args.near_plane,
                args.far_plane,
                args.tsdf_sdf_trunc_voxel_multiplier,
                args.min_render_alpha,
                device,
                debug_export_dir,
                prefix,
                tsdf_backend=args.tsdf_backend,
                render_downscale=args.render_downscale,
            )
        if column.init_method == "edgs":
            # EDGS (and any other splat init): use splat centers.
            return _process_point_cloud_reconstruction(
                _splat_centers_point_set(splats),
                geometry.cameras,
                args.voxel_size,
                debug_export_dir,
                prefix,
            )
        raise NotImplementedError(
            f"Init method '{column.init_method}' with splat init type is not supported."
        )

    points, rgbs = load_pointcloud_ply(ply_path)
    colors = np.asarray(rgbs, dtype=np.float64) if rgbs is not None else None
    return _process_point_cloud_reconstruction(
        PointSet(np.asarray(points, dtype=np.float64), colors),
        geometry.cameras,
        args.voxel_size,
        debug_export_dir,
        prefix,
    )


# --------------------------------------------------------------------------- #
# Trained-splat normalization transform resolution
# --------------------------------------------------------------------------- #

# Init methods whose older runs may have replaced the SfM points in the
# nerfbaselines dataset, so their normalization transform was computed from the
# init points instead of the SfM points. Laser-scan and the gaussian-based
# methods (edgs, da3 with output gaussians) always appended a path instead, so
# the SfM-derived transform is correct for them.
_INIT_TRANSFORM_CHECK_METHODS = ("monodepth", "da3")

# When neither the SfM-derived nor the init-derived transform aligns well (best
# alignment error above the acceptable threshold) and the two errors are within
# this relative factor of each other, we cannot tell which transform is correct.
_INIT_TRANSFORM_AMBIGUOUS_RATIO = 1.5


def _load_init_world_points(
    column: InitMethodColumn, scene: str, args: Args
) -> np.ndarray | None:
    """
    Load the raw world-frame point cloud output by a point-cloud init method
    (the exact points that would have been placed in the dataset at training
    time). Returns ``None`` for splat inits (their transform was never affected).
    """
    init_dir = ResultsDirectory(args.results_dir).get_init_method_output_dir(
        scene, column.init_method_config, column.init_method
    )
    ply_path, init_type = _find_init_ply(init_dir)
    if init_type == "splat":
        return None
    points, _ = load_pointcloud_ply(ply_path)
    return np.asarray(points, dtype=np.float64)


def _init_alignment_error(
    splats_world: SplatData, init_points_world: np.ndarray, sample_size: int = 10000
) -> float:
    """
    Median nearest-neighbour distance (meters) from a random sample of the init
    points to the world-frame trained splat centers.

    Querying init-point -> nearest-splat is robust to trained-splat floaters
    (extra far splats only add targets) and to reconstruction incompleteness
    (the median ignores the uncovered tail). A correct transform lands the splats
    on the init surface (small distances); a wrong transform offsets the whole
    reconstruction by scene-scale distances.
    """
    means = np.asarray(splats_world.means, dtype=np.float64)
    if means.shape[0] == 0 or init_points_world.shape[0] == 0:
        return float("inf")

    query = init_points_world
    if query.shape[0] > sample_size:
        rng = np.random.default_rng(0)
        query = query[rng.choice(query.shape[0], size=sample_size, replace=False)]

    distances = _nearest_neighbor_distances(query, means)
    return float(np.median(distances))


def resolve_world_frame_splats(
    splats: SplatData,
    column: InitMethodColumn,
    scene: str,
    args: Args,
    geometry: SceneGeometryInputs,
) -> SplatData:
    """
    Bring trained splats from the normalized frame back into the world frame.

    Normally the inverse of the SfM-derived normalization transform is correct.
    But older ``monodepth`` / ``da3`` (point-cloud) runs replaced the SfM points
    in the dataset, so their transform was computed from the init points instead.
    For those methods we always compute *both* candidate placements (from the
    SfM-derived transform and from an init-point-derived transform), measure how
    well each lands on the init surface, and keep the better-aligned one. If
    neither aligns well and the two are too close to distinguish, we raise.
    """
    inverse_sfm = np.linalg.inv(geometry.transform)
    splats_world = transform_splat_data(splats, inverse_sfm)

    if column.init_method not in _INIT_TRANSFORM_CHECK_METHODS:
        return splats_world

    try:
        init_points = _load_init_world_points(column, scene, args)
    except FileNotFoundError:
        return splats_world
    if init_points is None:  # splat init (e.g. da3 gaussians): SfM transform is fine.
        return splats_world

    # Recompute the transform from the init points exactly as an (older) training
    # run did when it replaced the SfM points, and compare both placements.
    transform_init = compute_normalization_transform(
        init_points, geometry.cameras.camtoworlds
    )
    splats_world_init = transform_splat_data(splats, np.linalg.inv(transform_init))

    error_sfm = _init_alignment_error(splats_world, init_points)
    error_init = _init_alignment_error(splats_world_init, init_points)

    best_error = min(error_sfm, error_init)
    worst_error = max(error_sfm, error_init)
    threshold_check = args.init_transform_check_threshold_meters
    threshold_fatal = args.init_transform_fatal_threshold_meters

    # If neither transform aligns the splats to the init surface, and the two
    # errors are within a small relative factor, we cannot tell them apart.
    if (
        best_error > threshold_check
        and worst_error <= best_error * _INIT_TRANSFORM_AMBIGUOUS_RATIO
    ):
        raise RuntimeError(
            f"[{scene}] column={column.label}: cannot determine the correct "
            f"normalization transform for the trained splats. SfM-derived "
            f"alignment {error_sfm:.4f} m and init-derived alignment "
            f"{error_init:.4f} m are both poor (> {threshold_check:.4f} m) and too "
            f"close to distinguish."
        )

    if best_error > threshold_fatal:
        raise RuntimeError(
            f"[{scene}] column={column.label}: trained splats are poorly aligned "
            f"to the init surface despite acceptable ratio (best alignment {best_error:.4f} m > "
            f"{threshold_fatal:.4f} m). This is likely "
            f"due to a misaligned normalization transform."
        )

    if error_init < error_sfm:
        _LOGGER.warning(
            "[%s] column=%s: init-point-derived normalization transform aligns "
            "better (median %.4f m) than the SfM-derived one (median %.4f m); "
            "using the init-derived transform.",
            scene,
            column.label,
            error_init,
            error_sfm,
        )
        return splats_world_init

    return splats_world


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

# Metrics that are fractions in [0, 1] and are reported as percentages (x100) in
# the LaTeX table.
_PERCENT_METRICS = {"fscore", "precision", "recall"}


def _nearest_neighbor_distances(query: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbor distance from every ``query`` point to the ``target`` set.

    Uses a CPU KD-tree with parallel queries (``workers=_CPU_WORKERS``; -1 uses
    all cores, or the --max-cpu-threads limit when set), which for these point
    counts is far faster than a brute-force GPU search.
    """
    if query.shape[0] == 0 or target.shape[0] == 0:
        return np.empty((0,), dtype=np.float64)

    tree = cKDTree(target)
    distances, _ = tree.query(query, k=1, workers=_CPU_WORKERS)
    return np.asarray(distances, dtype=np.float64)


def compute_fscore_metrics(
    reconstruction: PointSet, reference: PointSet, threshold_meters: float
) -> dict:
    """
    Compute F-score geometry metrics between a reconstruction and the reference
    (laser-scan) point set, both expressed in the world / laser-scan frame so
    that ``threshold_meters`` is a true metric distance.

    - precision (accuracy): fraction of reconstruction points whose nearest
      reference point is within ``threshold_meters``;
    - recall (completeness): fraction of reference points whose nearest
      reconstruction point is within ``threshold_meters``;
    - fscore: harmonic mean of precision and recall.

    Also reports the mean/median nearest-neighbor distances in both directions
    (in meters).
    """
    nn_start = time.perf_counter()
    dist_recon_to_ref = _nearest_neighbor_distances(
        reconstruction.points, reference.points
    )  # accuracy / precision direction
    dist_ref_to_recon = _nearest_neighbor_distances(
        reference.points, reconstruction.points
    )  # completeness / recall direction
    _LOGGER.info(
        "F-score NN distances (%d recon vs %d ref points): %.1fs",
        reconstruction.points.shape[0],
        reference.points.shape[0],
        time.perf_counter() - nn_start,
    )

    def _fraction_within(distances: np.ndarray) -> float:
        if distances.size == 0:
            return float("nan")
        return float(np.mean(distances <= threshold_meters))

    precision = _fraction_within(dist_recon_to_ref)
    recall = _fraction_within(dist_ref_to_recon)
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0.0:
        fscore = 0.0
    else:
        fscore = 2.0 * precision * recall / (precision + recall)

    def _stat(distances: np.ndarray, fn) -> float:
        return float(fn(distances)) if distances.size else float("nan")

    return {
        "threshold_meters": threshold_meters,
        "precision": precision,
        "recall": recall,
        "fscore": fscore,
        "mean_accuracy_meters": _stat(dist_recon_to_ref, np.mean),
        "median_accuracy_meters": _stat(dist_recon_to_ref, np.median),
        "mean_completeness_meters": _stat(dist_ref_to_recon, np.mean),
        "median_completeness_meters": _stat(dist_ref_to_recon, np.median),
        "num_reconstruction_points": int(reconstruction.points.shape[0]),
        "num_reference_points": int(reference.points.shape[0]),
    }


def _latex_escape_label(text: str) -> str:
    """Escape LaTeX special characters so config-string labels compile as text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(c, c) for c in text)


def _combined_header_label(metric_keys: list[str]) -> str:
    """
    Build the top-left header label for the combined-metric table, e.g.
    ``F-Score / Precision / Recall (\\%) ↑``. A shared ``(\\%)`` suffix is added
    when every metric is a percentage metric, and a single up/down arrow when all
    metrics agree on direction.
    """
    from results_scripts.constants import LOWER_IS_BETTER_METRICS

    short_names = {"fscore": "F-Score", "precision": "Precision", "recall": "Recall"}
    names = " / ".join(short_names.get(m, m) for m in metric_keys)

    suffix = ""
    if all(m in _PERCENT_METRICS for m in metric_keys):
        suffix += r" (\%)"
    if all(m not in LOWER_IS_BETTER_METRICS for m in metric_keys):
        suffix += " ↑"
    elif all(m in LOWER_IS_BETTER_METRICS for m in metric_keys):
        suffix += " ↓"
    return f"{names}{suffix}"


def write_metrics_latex_table(
    resolved_runs: dict[str, list[dict]],
    metric_keys: list[str],
    out_path: Path,
    caption: str,
    label: str,
) -> None:
    """
    Render the computed metrics as a single colored LaTeX table: one row per
    densification strategy, one column per init method. Each cell shows every
    requested metric on one line, separated by ``/`` (e.g. ``F-Score / Precision
    / Recall``), and is colored by the *first* requested metric.

    Each metric is aggregated over scenes (mean across scenes); cells with no
    available run become ``NaN`` and render as ``--``. Percentage metrics are
    scaled by 100.
    """
    import pandas as pd
    from results_scripts.constants import (
        LOWER_IS_BETTER_METRICS,
        TABLE_ROUNDING_PER_METRIC,
    )
    from results_scripts.formatting import FormatOptions
    from results_scripts.tables import (
        VALUE_CMAP,
        tabular_colored_from_numeric_with_custom_text,
        wrap_tabulars_as_float,
    )

    if not metric_keys:
        raise ValueError("At least one metric is required for the LaTeX table.")

    scenes = sorted(resolved_runs.keys())

    # Discover the (strategy row, init-method column) grid in first-seen order.
    strategies_raw: list[str] = []
    columns_raw: list[str] = []
    for scene in scenes:
        for entry in resolved_runs[scene]:
            if entry["strategy"] not in strategies_raw:
                strategies_raw.append(entry["strategy"])
            if entry["column"] not in columns_raw:
                columns_raw.append(entry["column"])

    # Always render the "at init" baseline as the first table row.
    if AT_INIT_ROW_LABEL in strategies_raw:
        strategies_raw.remove(AT_INIT_ROW_LABEL)
        strategies_raw.insert(0, AT_INIT_ROW_LABEL)

    def mean_metric(strategy: str, column: str, metric: str) -> float:
        """Mean of ``metric`` over scenes (percentage-scaled), or NaN if absent."""
        values: list[float] = []
        for scene in scenes:
            for entry in resolved_runs[scene]:
                if entry["strategy"] == strategy and entry["column"] == column:
                    metrics = entry.get("metrics")
                    value = metrics.get(metric) if metrics else None
                    if value is not None:
                        values.append(
                            value * 100.0 if metric in _PERCENT_METRICS else value
                        )
                    break
        return float(np.mean(values)) if values else float("nan")

    row_labels = [_latex_escape_label(s) for s in strategies_raw]
    col_labels = [_latex_escape_label(c) for c in columns_raw]

    color_metric = metric_keys[0]
    color_table = pd.DataFrame(index=row_labels, columns=col_labels, dtype=float)
    text_table = pd.DataFrame(index=row_labels, columns=col_labels, dtype=object)

    for strategy, row_label in zip(strategies_raw, row_labels):
        for column, col_label in zip(columns_raw, col_labels):
            means = {m: mean_metric(strategy, column, m) for m in metric_keys}
            color_table.loc[row_label, col_label] = means[color_metric]
            if np.isnan(means[color_metric]):
                text_table.loc[row_label, col_label] = np.nan
                continue
            parts = []
            for metric in metric_keys:
                rounding = TABLE_ROUNDING_PER_METRIC.get(metric, 2)
                value = means[metric]
                parts.append("--" if np.isnan(value) else f"{value:.{rounding}f}")
            text_table.loc[row_label, col_label] = "$" + " / ".join(parts) + "$"

    # Flip the gradient for lower-is-better color metrics so "better" stays warm.
    numeric_for_color = (
        -color_table if color_metric in LOWER_IS_BETTER_METRICS else color_table
    )

    tabular = tabular_colored_from_numeric_with_custom_text(
        f"\\textbf{{{_combined_header_label(metric_keys)}}}",
        numeric_for_color,
        text_table,
        cmap=VALUE_CMAP,
    )
    latex = wrap_tabulars_as_float(
        [tabular],
        latex_caption=caption,
        latex_label=label,
        format_args=FormatOptions(combine_datasets_as_subtables=False),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex)
    _LOGGER.info("Wrote metrics LaTeX table to %s", out_path)


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
    _apply_cpu_thread_limit(args.max_cpu_threads)

    if args.load_existing:
        if not args.output.exists():
            raise FileNotFoundError(
                f"--load-existing was set but the metrics JSON {args.output} does "
                "not exist. Run without --load-existing first, or point --output at "
                "an existing results file."
            )
        existing = json.loads(args.output.read_text())
        resolved_runs = existing["resolved_runs"]
        threshold = existing.get(
            "fscore_threshold_meters", args.fscore_threshold_meters
        )
        _LOGGER.info("Loaded existing metrics from %s", args.output)

        latex_path = args.latex_output or args.output.with_suffix(".tex")
        write_metrics_latex_table(
            resolved_runs,
            args.latex_metrics,
            latex_path,
            caption=(
                "Reconstruction accuracy " f"(inlier threshold {threshold:g}\\,m)"
            ),
            label="final_recon_fscore",
        )
        return

    columns = build_columns(args)
    if not columns:
        raise ValueError("No init methods to evaluate. Provide --init-methods.")
    column_by_label = {column.label: column for column in columns}

    if len(list(args.scenes)) == 0 and args.dataset is None:
        raise ValueError("No scenes to evaluate. Provide --scenes or --dataset.")
    scenes = get_scenes_from_args(
        list(args.scenes), [args.dataset] if args.dataset is not None else []
    )
    if not scenes:
        raise ValueError("No scenes to evaluate. Provide --scenes or --dataset.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved_runs: dict[str, list[dict]] = {}
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
        points_world = geometry.laser_points_world
        colors = geometry.laser_colors
        cameras = geometry.cameras

        scene_debug_dir = (
            args.debug_export_dir / _sanitize_for_path(scene)
            if args.debug_export_dir is not None
            else None
        )

        # Reference (laser-scan) point set, processed once per scene.
        reference = process_laser_scan_point_cloud(
            points_world,
            colors,
            cameras,
            args.voxel_size,
            debug_export_dir=scene_debug_dir,
            debug_prefix="laser_scan",
        )

        scene_entries: list[dict] = []

        # "At Init" row: F-score of each column's initial point cloud / splats
        # (before training), computed once per column and placed as the first row.
        for column in columns:
            init_entry: dict = {
                "column": column.label,
                "strategy": AT_INIT_ROW_LABEL,
                "output_dir": None,
                "exists": True,
                "metrics": None,
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
                init_entry["exists"] = False
                _LOGGER.warning(
                    "[%s] No init geometry for column=%s: %s",
                    scene,
                    column.label,
                    exc,
                )
            else:
                init_metrics = compute_fscore_metrics(
                    init_reconstruction, reference, args.fscore_threshold_meters
                )
                init_entry["metrics"] = init_metrics
                _LOGGER.info(
                    "[%s] column=%s strategy=%s -> F=%.4f P=%.4f R=%.4f "
                    "(thr=%.3f m)",
                    scene,
                    column.label,
                    AT_INIT_ROW_LABEL,
                    init_metrics["fscore"],
                    init_metrics["precision"],
                    init_metrics["recall"],
                    args.fscore_threshold_meters,
                )
            scene_entries.append(init_entry)

        for run in runs:
            entry: dict = {
                "column": run.column_label,
                "strategy": run.row_id,
                "output_dir": str(run.output_dir),
                "exists": run.exists,
                "metrics": None,
            }
            if not run.exists:
                _LOGGER.warning(
                    "[%s] Skipping missing run column=%s strategy=%s (path %s)",
                    scene,
                    run.column_label,
                    run.strategy_id,
                    run.output_dir,
                )
                scene_entries.append(entry)
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
            )

            metrics = compute_fscore_metrics(
                reconstruction, reference, args.fscore_threshold_meters
            )
            entry["metrics"] = metrics
            _LOGGER.info(
                "[%s] column=%s strategy=%s -> F=%.4f P=%.4f R=%.4f " "(thr=%.3f m)",
                scene,
                run.column_label,
                run.strategy_id,
                metrics["fscore"],
                metrics["precision"],
                metrics["recall"],
                args.fscore_threshold_meters,
            )
            scene_entries.append(entry)

        resolved_runs[scene] = scene_entries

        # Periodically write the JSON so we can inspect partial results if the script is interrupted.
        output = {
            "dataset": args.dataset,
            "scenes": [str(s) for s in scenes],
            "columns": [c.label for c in columns],
            "fscore_threshold_meters": args.fscore_threshold_meters,
            "resolved_runs": resolved_runs,
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as f:
            json.dump(output, f, indent=2)

    _LOGGER.info("Wrote reconstruction accuracy metrics to %s", args.output)

    latex_path = args.latex_output or args.output.with_suffix(".tex")
    write_metrics_latex_table(
        resolved_runs,
        args.latex_metrics,
        latex_path,
        caption=(
            "Reconstruction accuracy "
            f"(inlier threshold {args.fscore_threshold_meters:g}\\,m)"
        ),
        label="final_recon_fscore",
    )


if __name__ == "__main__":
    main()
