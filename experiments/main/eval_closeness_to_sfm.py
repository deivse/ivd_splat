"""
Geometry accuracy evaluation for two or more DA3 initialization configs.

Given a list of ``init_method=config`` reconstructions (each config being a
subfolder name under ``<results_dir>/<scene>/<init_method>/<config>/``), this
script:

1. For every scene (specified directly via ``--scenes`` or expanded from
   ``--datasets``) loads the point set produced by each reconstruction. Each
   output may be either a point cloud (``da3_points.ply``) or gaussians
   (``da3_gaussians.ply``); the type is detected from ``init_info.json``.
2. Loads the sparse SfM points for the scene from the nerfbaselines dataset.
3. Merges points that are closer to each other than a per-dataset configurable
   voxel size (via Open3D voxel downsampling) for the SfM points and every
   reconstruction.
4. Computes geometry accuracy metrics from the merged point sets.
5. Aggregates the metrics across all evaluated scenes.

Example::

    python experiments/main/eval_closeness_to_sfm.py \
        --init-methods da3=max_num_images=30 \
                       da3=max_num_images=30_output_gaussians=True \
                       monodepth=default \
        --datasets mipnerf360 \
        --results-dir results
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import open3d as o3d
import tyro
from scipy.spatial import cKDTree
from nerfbaselines.datasets import load_dataset
from nerfbaselines.utils import pad_poses

from eval_scripts.common.dataset_scenes import (
    get_scenes_from_args,
    scene_id_to_nerfbaselines_data_value,
)
from eval_scripts.common.results_dir import ResultsDirectory

from ivd_splat.datasets.normalize import (
    align_principle_axes,
    similarity_from_cameras,
    transform_points,
)
from shared.point_cloud_io import load_pointcloud_ply
from shared.splat_ply_io import load_splat_ply
from shared.save_init_info import INIT_INFO_JSON_FILENAME

_LOGGER = logging.getLogger(__name__)

# Spherical harmonics DC -> RGB constant (see ivd_splat.utils.runner_utils).
_SH_C0 = 0.28209479177387814

DEFAULT_VOXEL_SIZE = 0.02

# Nearest-neighbour distances above this percentile (computed over both A and B
# distance sets together) are filtered out of the SfM distance metric.
DISTANCE_PERCENTILE_CUTOFF = 90

# Names of the point sets merged per scene. Used as keys internally and for logging.
POINT_SET_SFM = "sfm"


@dataclass
class Reconstruction:
    """A single reconstruction to evaluate: an init method + one of its configs."""

    method: str
    config: str

    @property
    def label(self) -> str:
        """Unique, filename/JSON-safe identifier used in metric and file names."""
        return f"{self.method}_{self.config}"


def parse_reconstruction(spec: str) -> Reconstruction:
    """
    Parse an ``init_method=config`` spec. The config may itself contain ``=``, so
    only the first ``=`` separates the method from the config, e.g.
    ``da3=max_num_images=30`` -> method ``da3``, config ``max_num_images=30``.
    """
    if "=" not in spec:
        raise ValueError(
            f"Invalid --init-methods entry '{spec}'. Expected 'init_method=config'."
        )
    method, config = spec.split("=", 1)
    method, config = method.strip(), config.strip()
    if not method or not config:
        raise ValueError(
            f"Invalid --init-methods entry '{spec}'. Expected 'init_method=config'."
        )
    return Reconstruction(method=method, config=config)


@dataclass
class Args:
    # Reconstructions to compare, each given as "init_method=config" (the config
    # may itself contain '='). Example:
    #   --init-methods da3=max_num_images=30 monodepth=default
    init_methods: list[str]

    # Scenes to evaluate, in "dataset/scene" form or as local paths.
    # Takes precedence over --datasets when non-empty.
    scenes: list[str] = field(default_factory=list)
    # Datasets to expand into scenes when --scenes is not given.
    datasets: list[str] = field(default_factory=list)

    # Base results directory containing the init method outputs.
    results_dir: Path = Path("results")

    # Per-dataset voxel size, mapping dataset id -> voxel size. Applied in the
    # normalized scene coordinate frame (see scene normalization below).
    # Datasets not listed here fall back to --default-voxel-size.
    voxel_size_per_dataset: dict[str, float] = field(default_factory=dict)
    # Voxel size used for datasets not present in --voxel-size-per-dataset.
    default_voxel_size: float = DEFAULT_VOXEL_SIZE

    # Output JSON file for per-scene and aggregated metrics.
    output: Path = Path("closeness_to_sfm.json")

    # If set, export the merged point sets (SfM + every reconstruction) per scene
    # as PLY files into this directory for debugging/inspection.
    debug_export_dir: Path | None = None


# --------------------------------------------------------------------------- #
# Point set loading
# --------------------------------------------------------------------------- #


@dataclass
class PointSet:
    points: np.ndarray  # (N, 3) float
    colors: np.ndarray | None  # (N, 3) float in [0, 1], or None


def _find_ply_from_init_info(init_dir: Path) -> tuple[Path, str]:
    """
    Determine which PLY file holds the geometry output and its init type.

    Returns:
        (ply_path, init_type) where init_type is one of "splat", "dense",
        "sparse". Falls back to filename-based detection if init_info.json is
        missing or incomplete.
    """
    init_info_path = init_dir / INIT_INFO_JSON_FILENAME
    if init_info_path.exists():
        init_info = json.loads(init_info_path.read_text())
        init_type = init_info.get("init_type", "")
        for name in init_info.get("required_files", []):
            if name.endswith(".ply") and "sfm" not in name.lower():
                return init_dir / name, init_type

    # Fallback: look for known DA3 output filenames.
    for name, init_type in (
        ("da3_gaussians.ply", "splat"),
        ("da3_points.ply", "dense"),
    ):
        candidate = init_dir / name
        if candidate.exists():
            return candidate, init_type

    raise FileNotFoundError(
        f"Could not find a geometry PLY output in {init_dir}. "
        "Expected da3_points.ply or da3_gaussians.ply."
    )


def load_point_set(init_dir: Path) -> PointSet:
    """Load a point set from an init method output directory (points or gaussians)."""
    ply_path, init_type = _find_ply_from_init_info(init_dir)

    if init_type == "splat":
        splat = load_splat_ply(ply_path)
        points = splat.means.detach().cpu().numpy().astype(np.float64)
        rgbs = (splat.sh0.squeeze(1) * _SH_C0 + 0.5).detach().cpu().numpy()
        colors = np.clip(rgbs, 0.0, 1.0).astype(np.float64)
    else:
        points, colors = load_pointcloud_ply(ply_path)
        points = np.asarray(points, dtype=np.float64)
        if colors is not None:
            colors = np.asarray(colors, dtype=np.float64)

    _LOGGER.info(
        "Loaded %d %s from %s",
        points.shape[0],
        "gaussians" if init_type == "splat" else "points",
        ply_path,
    )
    return PointSet(points=points, colors=colors)


def load_sfm_and_normalization(scene: str) -> tuple[PointSet, np.ndarray]:
    """
    Load the sparse SfM points for a scene together with a scene normalization
    transform.

    The transform is computed exactly as in ivd_splat's NerfbaselinesParser:
    a similarity transform derived from the camera poses (which normalizes the
    scene scale), followed by a principal-axes alignment of the SfM points.
    """
    nb_data_value = scene_id_to_nerfbaselines_data_value(scene)
    dataset = load_dataset(
        nb_data_value, "train", features=["points3D_xyz", "points3D_rgb"]
    )

    points = np.asarray(dataset["points3D_xyz"], dtype=np.float64)
    colors = dataset.get("points3D_rgb")
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.max() > 1.0:
            colors = colors / 255.0
    _LOGGER.info("Loaded %d SfM points for %s", points.shape[0], scene)

    camtoworlds = pad_poses(dataset["cameras"].poses)
    t1 = similarity_from_cameras(camtoworlds)
    t2 = align_principle_axes(transform_points(t1, points))
    transform = t2 @ t1

    return PointSet(points=points, colors=colors), transform


def transform_point_set(point_set: PointSet, transform: np.ndarray) -> PointSet:
    """Apply a 4x4 SE(3)/similarity transform to a point set's positions."""
    return PointSet(
        points=transform_points(transform, point_set.points),
        colors=point_set.colors,
    )


# --------------------------------------------------------------------------- #
# Point merging
# --------------------------------------------------------------------------- #


def merge_close_points(point_set: PointSet, voxel_size: float) -> PointSet:
    """
    Merge points that are closer to each other than `voxel_size`.

    Uses Open3D voxel downsampling: the space is partitioned into a regular grid
    of `voxel_size`-sided cubes and all points falling into the same voxel are
    merged into a single (averaged) point.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_set.points)
    if point_set.colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(point_set.colors)

    merged = pcd.voxel_down_sample(voxel_size=voxel_size)

    points = np.asarray(merged.points, dtype=np.float64)
    colors = (
        np.asarray(merged.colors, dtype=np.float64) if merged.has_colors() else None
    )

    _LOGGER.info(
        "Merged %d -> %d points (voxel size %.5f)",
        point_set.points.shape[0],
        points.shape[0],
        voxel_size,
    )
    return PointSet(points=points, colors=colors)


# --------------------------------------------------------------------------- #
# Geometry accuracy metrics
# --------------------------------------------------------------------------- #


def sfm_nearest_neighbors(
    sfm: PointSet, reconstruction: PointSet
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every SfM point, find the nearest point in the reconstruction.

    Returns:
        (distances, indices) where distances[i] is the distance from SfM point i
        to its nearest reconstruction point, and indices[i] is that point's
        index in `reconstruction.points`. Both arrays are empty if either point
        set is empty.
    """
    if reconstruction.points.shape[0] == 0 or sfm.points.shape[0] == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.int64)
    tree = cKDTree(reconstruction.points)
    distances, indices = tree.query(sfm.points, k=1)
    return distances, indices


def sfm_distance_threshold(distances_per_recon: dict[str, np.ndarray]) -> float:
    """
    Shared distance cutoff = the `DISTANCE_PERCENTILE_CUTOFF`th percentile over
    all reconstructions' nearest-neighbour distances combined.
    """
    combined = np.concatenate(list(distances_per_recon.values()))
    if combined.size == 0:
        return float("nan")
    return float(np.percentile(combined, DISTANCE_PERCENTILE_CUTOFF))


def sfm_points_filtered_out(
    distances_per_recon: dict[str, np.ndarray], threshold: float
) -> np.ndarray | None:
    """
    Boolean mask over SfM points that are dropped from every metric: an SfM point
    is filtered out only when *every* reconstruction's nearest-neighbour distance
    exceeds `threshold`. Returns None if no non-empty distances are available or
    the threshold is undefined (in which case nothing is filtered out).
    """
    non_empty = [d for d in distances_per_recon.values() if d.size > 0]
    if not non_empty or np.isnan(threshold):
        return None
    return np.logical_and.reduce([d > threshold for d in non_empty])


def compute_geometry_accuracy_metrics(
    distances_per_recon: dict[str, np.ndarray],
    filtered_out: np.ndarray | None,
) -> dict[str, float]:
    """
    Compute geometry accuracy metrics from the SfM nearest-neighbour distances.

    `distances_per_recon` maps each reconstruction's label to its SfM
    nearest-neighbour distances.

    The "distance from SfM" metric measures how faithful a reconstruction is to
    the (sparse) SfM point cloud as the sum of nearest-neighbour distances from
    each SfM point to the reconstruction (lower is better). SfM points where
    *every* reconstruction's distance exceeds the shared percentile cutoff (i.e.
    where `filtered_out` is True) are dropped from all sums.
    """

    def filtered_sum(distances: np.ndarray) -> float:
        if distances.size == 0:
            return float("nan")
        if filtered_out is None:
            return float(np.sum(distances))
        return float(np.sum(distances[~filtered_out]))

    return {
        f"dist_from_sfm_{label}": filtered_sum(distances)
        for label, distances in distances_per_recon.items()
    }


# --------------------------------------------------------------------------- #
# Per-scene evaluation + aggregation
# --------------------------------------------------------------------------- #


def voxel_size_for_scene(scene: str, args: Args) -> float:
    dataset_id = str(scene).split("/")[0]
    return args.voxel_size_per_dataset.get(dataset_id, args.default_voxel_size)


def export_debug_point_sets(
    merged: dict[str, PointSet],
    scene: str,
    debug_export_dir: Path,
) -> None:
    """Export merged point sets for a scene as PLY files for debugging."""
    scene_dir = debug_export_dir / str(scene).replace("/", "_")
    scene_dir.mkdir(parents=True, exist_ok=True)
    for name, point_set in merged.items():
        out_path = scene_dir / f"merged_{name}.ply"
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_set.points)
        if point_set.colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(point_set.colors)
        o3d.io.write_point_cloud(str(out_path), pcd)
        _LOGGER.info("Exported debug merged point set for '%s' to %s", name, out_path)


def _write_colored_line_set_ply(
    out_path: Path,
    points: np.ndarray,
    lines: np.ndarray,
    line_colors: np.ndarray,
) -> None:
    """
    Write a line set to an ASCII PLY with per-vertex colors.

    Open3D's LineSet PLY export stores colors on the edge element, which MeshLab
    does not display (lines render grey). Instead we color the two endpoints of
    every line with that line's color so viewers that render vertex colors show
    the intended colors. Each vertex is assumed to belong to exactly one line.
    """
    vertex_colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    rgb = np.clip(line_colors * 255.0, 0, 255).astype(np.uint8)
    for (v0, v1), color in zip(lines, rgb):
        vertex_colors[v0] = color
        vertex_colors[v1] = color

    with out_path.open("w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element edge {lines.shape[0]}\n")
        f.write("property int vertex1\n")
        f.write("property int vertex2\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, vertex_colors):
            f.write(f"{x} {y} {z} {int(r)} {int(g)} {int(b)}\n")
        for v0, v1 in lines:
            f.write(f"{int(v0)} {int(v1)}\n")


def export_debug_distance_lines(
    merged: dict[str, PointSet],
    nn: dict[str, tuple[np.ndarray, np.ndarray]],
    filtered_out: np.ndarray | None,
    scene: str,
    debug_export_dir: Path,
) -> None:
    """
    Export, for each reconstruction, the SfM-accuracy nearest-neighbour distances
    as line segments connecting each SfM point to its closest reconstruction
    point.

    Lines for SfM points kept by the metric are drawn red; lines for points that
    are filtered out (every reconstruction's distance above the cutoff) are drawn
    blue.
    """
    scene_dir = debug_export_dir / str(scene).replace("/", "_")
    scene_dir.mkdir(parents=True, exist_ok=True)

    sfm = merged[POINT_SET_SFM]
    for label, (distances, indices) in nn.items():
        if indices.size == 0:
            continue

        nn_points = merged[label].points[indices]
        num = sfm.points.shape[0]

        colors = np.tile([1.0, 0.0, 0.0], (num, 1))
        if filtered_out is not None:
            colors[filtered_out] = [0.0, 0.0, 1.0]

        points = np.vstack([sfm.points, nn_points])
        lines = np.column_stack([np.arange(num), np.arange(num) + num])

        out_path = scene_dir / f"distances_{label}.ply"
        _write_colored_line_set_ply(out_path, points, lines, colors)
        _LOGGER.info("Exported debug distance lines for '%s' to %s", label, out_path)


def evaluate_scene(
    scene: str, args: Args, reconstructions: list[Reconstruction]
) -> dict[str, float]:
    results_dir = ResultsDirectory(args.results_dir)

    sfm_point_set, transform = load_sfm_and_normalization(scene)
    point_sets = {POINT_SET_SFM: sfm_point_set}
    for recon in reconstructions:
        init_dir = results_dir.get_init_method_output_dir(
            scene, recon.config, recon.method
        )
        point_sets[recon.label] = load_point_set(init_dir)

    # Normalize the scene scale (using the SfM point cloud / cameras) by applying
    # the same transform to all point sets before merging.
    point_sets = {
        name: transform_point_set(ps, transform) for name, ps in point_sets.items()
    }

    voxel_size = voxel_size_for_scene(scene, args)
    _LOGGER.info("Using voxel size %.5f for scene %s", voxel_size, scene)

    merged = {
        name: merge_close_points(ps, voxel_size) for name, ps in point_sets.items()
    }

    sfm = merged[POINT_SET_SFM]
    nn = {
        recon.label: sfm_nearest_neighbors(sfm, merged[recon.label])
        for recon in reconstructions
    }
    distances_per_recon = {label: dist for label, (dist, _) in nn.items()}
    threshold = sfm_distance_threshold(distances_per_recon)
    filtered_out = sfm_points_filtered_out(distances_per_recon, threshold)

    if args.debug_export_dir is not None:
        export_debug_point_sets(merged, scene, args.debug_export_dir)
        export_debug_distance_lines(
            merged, nn, filtered_out, scene, args.debug_export_dir
        )

    return compute_geometry_accuracy_metrics(distances_per_recon, filtered_out)


def aggregate_metrics(
    per_scene: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Aggregate per-scene metrics by averaging each named metric across scenes."""
    aggregated: dict[str, float] = {}
    metric_names = {name for metrics in per_scene.values() for name in metrics}
    for name in sorted(metric_names):
        values = [metrics[name] for metrics in per_scene.values() if name in metrics]
        if values:
            aggregated[name] = float(np.mean(values))
    return aggregated


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)

    reconstructions = [parse_reconstruction(spec) for spec in args.init_methods]
    if not reconstructions:
        raise ValueError("No reconstructions to evaluate. Provide --init-methods.")

    scenes = get_scenes_from_args(list(args.scenes), list(args.datasets))
    if not scenes:
        raise ValueError("No scenes to evaluate. Provide --scenes or --datasets.")

    per_scene: dict[str, dict[str, float]] = {}
    for scene in scenes:
        scene = str(scene)
        try:
            per_scene[scene] = evaluate_scene(scene, args, reconstructions)
            _LOGGER.info("Metrics for %s: %s", scene, per_scene[scene])
        except Exception as e:
            _LOGGER.error("Error evaluating scene %s: %s", scene, e, exc_info=True)

    output = {
        "reconstructions": [
            {"method": r.method, "config": r.config, "label": r.label}
            for r in reconstructions
        ],
        "per_scene": per_scene,
        "aggregated": aggregate_metrics(per_scene),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(output, f, indent=2)
    _LOGGER.info("Wrote output to %s", args.output)
    _LOGGER.info("Aggregated metrics: %s", output["aggregated"])


if __name__ == "__main__":
    main()
