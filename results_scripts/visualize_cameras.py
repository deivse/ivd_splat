"""Interactively visualize the train/test cameras and dense laser-scan point
cloud of an NVS dataset.

Cameras are drawn as small frustums: train cameras in blue, test cameras in red.
The dense laser-scan point cloud (path taken from the nerfbaselines dataset
metadata, same as ``src/ivd_splat/initialization.py``) is shown subsampled for
performance.

The Open3D window supports the usual mouse camera controls (drag to rotate,
scroll to zoom, ctrl/shift+drag to pan). Press ``S`` to export the current view
to an image, ``Q``/``Esc`` to quit.

Example::

    python visualize_cameras.py --scene mipnerf360/garden
    python visualize_cameras.py --scene scannet++/b0a08200c9 --point-subsample 50
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import tyro
from eval_scripts.common.dataset_scenes import scene_id_to_nerfbaselines_data_value
from nerfbaselines.datasets import load_dataset
from nerfbaselines.utils import pad_poses

from shared.point_cloud_io import load_pointcloud_ply

_LOGGER = logging.getLogger(__name__)

TRAIN_COLOR = np.array([0.1, 0.3, 1.0])  # blue
TEST_COLOR = np.array([1.0, 0.1, 0.1])  # red
BACKGROUND_COLOR = np.array([1.0, 1.0, 1.0])  # white


@dataclass
class Args:
    # Scene in our <dataset_id>/<scene_id> format, e.g. "mipnerf360/garden" or
    # "scannet++/<unreadable_hex_string>" (same convention as src/da3/main.py).
    scene: str
    # Keep only 1/point_subsample of the dense laser-scan points (for
    # performance). Set to 1 to keep all points.
    point_subsample: int = 100
    # Fake point-cloud transparency in [0, 1]: point colors are blended toward
    # the background (1.0 = fully opaque, lower = more see-through). The legacy
    # Open3D viewer has no true per-point alpha, so this approximates it.
    point_opacity: float = 1.0
    # Frustum size as a fraction of the scene scale (the camera-spread radius).
    frustum_scale: float = 0.05
    # Frustum edge (line) thickness as a fraction of the frustum size. Rendered
    # as cylinders so the thickness is respected on all OpenGL backends.
    frustum_line_width: float = 0.03
    # Path the current view is written to when you press 'S' in the viewer.
    screenshot_path: Path = Path("camera_visualization")
    # Skip loading/showing the dense laser-scan point cloud.
    no_points: bool = False


def _load_cameras(nb_data_value: str, split: str):
    """Load a split's cameras (poses + intrinsics) without image features.

    Returns ``(poses, intrinsics, image_sizes, dataset)`` or ``None`` if the
    split is unavailable for this dataset.
    """
    try:
        dataset = load_dataset(
            nb_data_value,
            split=split,
            features=frozenset(),
            supported_camera_models=frozenset(["pinhole"]),
            load_features=False,
        )
    except Exception as exc:  # noqa: BLE001 - split may simply not exist
        _LOGGER.warning("Could not load '%s' split: %s", split, exc)
        return None

    cameras = dataset["cameras"]
    poses = np.asarray(pad_poses(cameras.poses), dtype=np.float64)  # (C, 4, 4)
    intrinsics = np.asarray(cameras.intrinsics, dtype=np.float64)  # (C, 4) fx,fy,cx,cy
    image_sizes = np.asarray(cameras.image_sizes, dtype=np.float64)  # (C, 2) w,h
    return poses, intrinsics, image_sizes, dataset


def _rotations_z_to(directions: np.ndarray) -> np.ndarray:
    """Per-row rotation matrices mapping +z onto each (unit) direction.

    ``directions`` is (E, 3); returns (E, 3, 3). Uses Rodrigues' formula, with a
    special case for directions anti-parallel to +z.
    """
    num = directions.shape[0]
    cos = directions[:, 2]
    # cross(z, dir) = (-dy, dx, 0)
    axis = np.stack([-directions[:, 1], directions[:, 0], np.zeros(num)], axis=1)

    skew = np.zeros((num, 3, 3))
    skew[:, 0, 1] = -axis[:, 2]
    skew[:, 0, 2] = axis[:, 1]
    skew[:, 1, 0] = axis[:, 2]
    skew[:, 1, 2] = -axis[:, 0]
    skew[:, 2, 0] = -axis[:, 1]
    skew[:, 2, 1] = axis[:, 0]

    identity = np.eye(3)[None]
    factor = np.where(np.abs(1.0 + cos) < 1e-8, 0.0, 1.0 / (1.0 + cos))
    rot = (
        identity + skew + np.einsum("eij,ejk->eik", skew, skew) * factor[:, None, None]
    )

    degenerate = np.abs(1.0 + cos) < 1e-8
    if np.any(degenerate):
        rot[degenerate] = np.diag([1.0, -1.0, -1.0])
    return rot


def _cylinders_mesh(
    starts: np.ndarray, ends: np.ndarray, radius: float, resolution: int = 8
) -> o3d.geometry.TriangleMesh:
    """Build one TriangleMesh containing a cylinder for every start/end pair."""
    angles = np.linspace(0.0, 2.0 * np.pi, resolution, endpoint=False)
    ring = np.stack([np.cos(angles), np.sin(angles)], axis=1)  # (R, 2)

    # Unit-cylinder template: bottom ring at z=0, top ring at z=1.
    template = np.empty((2 * resolution, 3))
    template[0::2, :2] = ring
    template[0::2, 2] = 0.0
    template[1::2, :2] = ring
    template[1::2, 2] = 1.0

    tri_list = []
    for i in range(resolution):
        j = (i + 1) % resolution
        tri_list.append([2 * i, 2 * j, 2 * i + 1])
        tri_list.append([2 * j, 2 * j + 1, 2 * i + 1])
    template_tris = np.asarray(tri_list, dtype=np.int64)  # (T, 3)

    axis = ends - starts
    lengths = np.linalg.norm(axis, axis=1)
    valid = lengths > 1e-9
    starts, axis, lengths = starts[valid], axis[valid], lengths[valid]
    directions = axis / lengths[:, None]
    num_edges = starts.shape[0]

    scaled = np.broadcast_to(
        template * np.array([radius, radius, 1.0]), (num_edges, *template.shape)
    ).copy()
    scaled[:, :, 2] *= lengths[:, None]

    rot = _rotations_z_to(directions)
    world = np.einsum("eij,evj->evi", rot, scaled) + starts[:, None, :]

    verts = world.reshape(-1, 3)
    vert_offsets = (np.arange(num_edges) * template.shape[0])[:, None, None]
    tris = (template_tris[None] + vert_offsets).reshape(-1, 3)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(tris)
    return mesh


def _make_frustums(
    poses: np.ndarray,
    intrinsics: np.ndarray,
    image_sizes: np.ndarray,
    scale: float,
    color: np.ndarray,
    line_radius: float,
) -> o3d.geometry.TriangleMesh:
    """Build a TriangleMesh with a thick-edged frustum for every camera.

    Poses are camera-to-world in the OpenCV convention (camera looks along +z),
    matching the nerfbaselines datasets used throughout this repo. Edges are
    rendered as cylinders so their thickness is honoured on every OpenGL backend
    (the legacy LineSet line width is ignored by most Mesa/WSL drivers).
    """
    # Edges of a frustum, as index pairs into the 5 per-camera points
    # (0 = apex, 1..4 = image-plane corners).
    edge_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]

    starts: list[np.ndarray] = []
    ends: list[np.ndarray] = []
    for pose, (fx, fy, cx, cy), (width, height) in zip(poses, intrinsics, image_sizes):
        # Image-plane corners at depth `scale`, back-projected into camera space.
        corners_pix = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        cam_pts_list = [np.zeros(3)]  # apex (camera origin)
        for u, v in corners_pix:
            x = (u - cx) / fx * scale
            y = (v - cy) / fy * scale
            cam_pts_list.append(np.array([x, y, scale]))
        cam_pts = np.stack(cam_pts_list)  # (5, 3)

        world_pts = cam_pts @ pose[:3, :3].T + pose[:3, 3]
        for a, b in edge_pairs:
            starts.append(world_pts[a])
            ends.append(world_pts[b])

    mesh = _cylinders_mesh(np.asarray(starts), np.asarray(ends), line_radius)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def _scene_scale(poses: np.ndarray) -> float:
    """Scene scale measured as the max camera distance from the camera centroid."""
    centers = poses[:, :3, 3]
    centroid = centers.mean(axis=0)
    dists = np.linalg.norm(centers - centroid, axis=1)
    scale = float(dists.max())
    return scale if scale > 0 else 1.0


def _load_point_cloud(
    dataset, subsample: int, opacity: float = 1.0
) -> o3d.geometry.PointCloud | None:
    metadata = dataset["metadata"]
    if "dense_points3D_path" not in metadata:
        _LOGGER.warning(
            "Dataset metadata has no 'dense_points3D_path'; skipping point cloud."
        )
        return None

    dense_path = metadata["dense_points3D_path"]
    _LOGGER.info("Loading dense laser-scan point cloud from %s", dense_path)
    points, colors = load_pointcloud_ply(dense_path)

    if subsample > 1:
        points = points[::subsample]
        colors = colors[::subsample] if colors is not None else None
    _LOGGER.info("Showing %d points (subsample=%d)", points.shape[0], subsample)

    if colors is None:
        colors = np.full((points.shape[0], 3), 0.5)  # neutral gray
    if opacity < 1.0:
        # Fake transparency: blend toward the background color.
        colors = opacity * colors + (1.0 - opacity) * BACKGROUND_COLOR

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    args = tyro.cli(Args)

    nb_data_value = scene_id_to_nerfbaselines_data_value(args.scene)
    _LOGGER.info("Loading scene '%s' (%s)", args.scene, nb_data_value)

    train = _load_cameras(nb_data_value, "train")
    if train is None:
        raise RuntimeError(f"Failed to load train cameras for scene {args.scene}")
    train_poses, train_intr, train_sizes, train_dataset = train
    _LOGGER.info("Train cameras: %d", train_poses.shape[0])

    test = _load_cameras(nb_data_value, "test")
    if test is not None:
        test_poses, test_intr, test_sizes, _ = test
        _LOGGER.info("Test cameras: %d", test_poses.shape[0])
    else:
        _LOGGER.warning("No test split available; showing train cameras only.")

    scene_scale = _scene_scale(train_poses)
    frustum_size = args.frustum_scale * scene_scale
    line_radius = args.frustum_line_width * frustum_size

    geometries: list[o3d.geometry.Geometry] = []

    geometries.append(
        _make_frustums(
            train_poses, train_intr, train_sizes, frustum_size, TRAIN_COLOR, line_radius
        )
    )
    if test is not None:
        geometries.append(
            _make_frustums(
                test_poses, test_intr, test_sizes, frustum_size, TEST_COLOR, line_radius
            )
        )

    if not args.no_points:
        pcd = _load_point_cloud(train_dataset, args.point_subsample, args.point_opacity)
        if pcd is not None:
            geometries.append(pcd)

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frustum_size * 2.0, origin=train_poses[:, :3, 3].mean(axis=0)
    )
    geometries.append(coord_frame)

    _run_viewer(geometries, args.screenshot_path, args.scene)


def _prefer_x11_backend() -> None:
    """Steer Open3D's bundled GLFW to X11 instead of Wayland on WSL.

    Under WSLg, GLFW defaults to the Wayland backend, which fails to create a
    working OpenGL context ("Failed to initialize GLEW"), leaving the visualizer
    without a render option. WSLg always exposes an Xwayland server via $DISPLAY,
    over which Open3D's legacy visualizer works reliably, so we drop
    $WAYLAND_DISPLAY to make GLFW fall back to X11.
    """
    if os.environ.get("WAYLAND_DISPLAY") and os.environ.get("DISPLAY"):
        _LOGGER.info(
            "Wayland detected; unsetting WAYLAND_DISPLAY so Open3D uses X11 "
            "(DISPLAY=%s).",
            os.environ["DISPLAY"],
        )
        os.environ.pop("WAYLAND_DISPLAY", None)


def _run_viewer(
    geometries: list[o3d.geometry.Geometry], screenshot_path: Path, scene: str
) -> None:
    _prefer_x11_backend()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Camera visualization", width=1600, height=1000)
    for geom in geometries:
        vis.add_geometry(geom)

    render_option = vis.get_render_option()
    if render_option is None:
        vis.destroy_window()
        raise RuntimeError(
            "Open3D could not create an OpenGL context (get_render_option() "
            "returned None). On WSL make sure an X server is available "
            "(WSLg provides one via $DISPLAY, usually ':0'). If it is still "
            "failing, try forcing software rendering with "
            "'LIBGL_ALWAYS_SOFTWARE=1 python visualize_cameras.py ...'."
        )
    render_option.background_color = BACKGROUND_COLOR
    render_option.point_size = 2.0

    def _save_screenshot(vis: o3d.visualization.Visualizer) -> bool:
        path = Path(screenshot_path) / scene
        path = path.with_suffix(".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(path), do_render=True)
        _LOGGER.info("Saved screenshot to %s", path.resolve())
        return False

    vis.register_key_callback(ord("S"), _save_screenshot)

    _LOGGER.info(
        "Controls: drag=rotate, scroll=zoom, ctrl/shift+drag=pan, "
        "'S'=save screenshot, 'Q'/Esc=quit."
    )
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
