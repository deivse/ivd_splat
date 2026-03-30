import logging
from pathlib import Path
from typing import Optional

import numpy as np
import open3d
import torch

_LOGGER = logging.getLogger(__name__)


def export_pointcloud_ply(
    pts: np.ndarray | torch.Tensor,
    rgbs: Optional[np.ndarray | torch.Tensor],
    path: Path | str,
):
    """Exports a point cloud to a PLY file."""
    pcd = open3d.geometry.PointCloud()

    if isinstance(pts, torch.Tensor):
        pts = pts.cpu().numpy()
    pcd.points = open3d.utility.Vector3dVector(pts)

    if rgbs is not None:
        if isinstance(rgbs, torch.Tensor):
            rgbs = rgbs.cpu().numpy()
        if rgbs.max() > 1.0:
            rgbs = rgbs.astype(np.float64) / 255.0
        pcd.colors = open3d.utility.Vector3dVector(rgbs)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    open3d.io.write_point_cloud(str(path), pcd, write_ascii=False)
    _LOGGER.info(f"Saved point cloud to {path}")


def load_pointcloud_ply(
    path: Path | str,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Loads a point cloud from a PLY file.

    Returns:
        pts: Nx3 array of point coordinates
        rgbs: Nx3 array of point colors (float, <0, 1> range), or None if no colors are present
    """
    pcd = open3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)
    rgbs = None
    if pcd.has_colors():
        rgbs = np.asarray(pcd.colors)
    if pts is None:
        raise RuntimeError(
            f"Failed to load point cloud from {path}, see open3d log messages."
        )
    return pts, rgbs


def save_mesh(
    triangles: np.ndarray,
    pts: np.ndarray,
    normals: np.ndarray,
    rgbs: np.ndarray | None,
    path: Path | str,
):
    """Exports a mesh to a PLY file."""
    mesh = open3d.geometry.TriangleMesh()
    mesh.triangles = open3d.utility.Vector3iVector(triangles)
    mesh.vertices = open3d.utility.Vector3dVector(pts)
    # mesh.vertex_normals = open3d.utility.Vector3dVector(normals)

    if rgbs is not None:
        if rgbs.max() > 1.0:
            rgbs = rgbs.astype(np.float64) / 255.0
        mesh.vertex_colors = open3d.utility.Vector3dVector(rgbs)
    open3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False)
    _LOGGER.info(f"Saved mesh to {path}")


def load_normals(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as normals_path:
        normals = np.fromfile(normals_path, dtype=np.float32).reshape(-1, 3)
    return normals


def save_normals(normals: np.ndarray, path: str | Path):
    with Path(path).open("wb") as normals_path:
        normals.astype(np.float32).tofile(normals_path)
