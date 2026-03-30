import logging
from pathlib import Path
from typing import Optional

import numpy as np
import open3d


def export_point_cloud_to_ply(
    pts: np.ndarray,
    rgbs: Optional[np.ndarray],
    output_dir: Path,
    depth_pts_filename: str,
    outlier_std_dev: Optional[float] = None,
    sphere_sizes: Optional[np.ndarray] = None,
):
    """Saves point cloud to a .ply file. Optionally adds spheres around each point.

    Args:
        pts: (N, 3) array of point coordinates.
        rgbs: (N, 3) array of RGB colors in [0, 1], or None.
        output_dir: Output directory as Path.
        depth_pts_filename: Filename (without extension).
        outlier_std_dev: If set, removes outliers beyond this std deviation.
        sphere_sizes: (N,) array of sphere radii for each point, or None.
    """
    if outlier_std_dev is not None:
        mean = np.mean(pts, axis=0)
        std_dev = np.std(pts, axis=0)
        mask = np.all(np.abs((pts - mean) / std_dev) < outlier_std_dev, axis=1)
        pts = pts[mask]
        if rgbs is not None:
            rgbs = rgbs[mask]
        if sphere_sizes is not None:
            sphere_sizes = sphere_sizes[mask]

    # Add point cloud
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pts)
    if rgbs is not None:
        pcd.colors = open3d.utility.Vector3dVector(rgbs)

    if sphere_sizes is not None:
        # Normalize sphere_sizes to [0, 1] for coloring
        min_size = np.percentile(sphere_sizes, 1)  # Avoid outlier influence
        max_size = np.percentile(sphere_sizes, 80)  # Avoid outlier influence

        norm_sizes = (sphere_sizes - min_size) / (max_size - min_size + 1e-8)
        # Map normalized sizes to a color gradient (e.g., blue to red)
        colors_by_size = np.stack(
            [
                norm_sizes,  # Red channel
                np.zeros_like(norm_sizes),  # Green channel
                1.0 - norm_sizes,  # Blue channel
            ],
            axis=1,
        )
        pcd_size_color = open3d.geometry.PointCloud()
        pcd_size_color.points = open3d.utility.Vector3dVector(pts)
        pcd_size_color.colors = open3d.utility.Vector3dVector(colors_by_size)
        open3d.io.write_point_cloud(
            str(output_dir / f"{depth_pts_filename}_size_color.ply"),
            pcd_size_color,
            write_ascii=False,
        )
        logging.info(
            f"Saved point cloud with sphere_size-based color to {output_dir / f'{depth_pts_filename}_size_color.ply'}"
        )

    # Add spheres if requested
    if sphere_sizes is not None:
        mesh = open3d.geometry.TriangleMesh()
        subsample_factor = 1
        for i, (pt, radius) in enumerate(
            zip(pts[::subsample_factor], sphere_sizes[::subsample_factor])
        ):
            sphere = open3d.geometry.TriangleMesh.create_sphere(
                radius=radius, resolution=2
            )
            sphere.translate(pt)
            if rgbs is not None:
                color = rgbs[i]
                sphere.paint_uniform_color(color)
            mesh += sphere

        # Save mesh (spheres) and point cloud separately
        open3d.io.write_triangle_mesh(
            str(output_dir / f"{depth_pts_filename}_spheres.ply"),
            mesh,
            write_ascii=False,
        )
        logging.info(
            f"Saved spheres mesh to {output_dir / f'{depth_pts_filename}_spheres.ply'}"
        )

    # Save point cloud
    open3d.io.write_point_cloud(
        str(output_dir / f"{depth_pts_filename}.ply"), pcd, write_ascii=False
    )
    logging.info(f"Saved point cloud to {output_dir / f'{depth_pts_filename}.ply'}")
