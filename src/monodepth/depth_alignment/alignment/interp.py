from pathlib import Path
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay
import torch
import logging

from monodepth.config import Config
from monodepth.depth_alignment.alignment.interp_common import (
    debug_export_outlier_classification,
    interp_common,
)

from monodepth.depth_alignment.config import (
    InterpConfig,
)
from monodepth.depth_prediction.interface import (
    CameraIntrinsics,
    PredictedDepth,
)
from ..interface import DepthAlignmentResult, DepthAlignmentStrategy

LOGGER = logging.getLogger(__name__)


def image_space_linear_interpolation(
    coords: torch.Tensor,
    values: torch.Tensor,
    W: int,
    H: int,
) -> torch.Tensor:
    coords_np = coords.T.cpu().numpy()
    values_np = values.cpu().numpy()

    # add values at the corners to stabilize interpolation
    corner_coords = np.array([[0, 0], [0, H - 1], [W - 1, 0], [W - 1, H - 1]])
    corner_indices = np.arange(coords_np.shape[0], coords_np.shape[0] + 4)
    coords_np = np.vstack((coords_np, corner_coords))
    values_np = np.hstack((values_np, np.empty(4, dtype=values_np.dtype)))

    dt = Delaunay(coords_np)
    for corner_ix in corner_indices:
        indptr, indices = dt.vertex_neighbor_vertices
        neighbors = indices[indptr[corner_ix] : indptr[corner_ix + 1]]
        # exclude other corners
        neighbors = np.setdiff1d(neighbors, corner_indices)
        distances = np.linalg.norm(coords_np[neighbors] - coords_np[corner_ix], axis=1)
        weights = 1.0 / (distances + 1e-8)
        weights /= np.sum(weights)
        corner_value = np.sum(values_np[neighbors] * weights)
        if np.isnan(corner_value):
            corner_value = np.median(values_np[neighbors])
        values_np[corner_ix] = corner_value

    X = np.linspace(0, W - 1, W)
    Y = np.linspace(0, H - 1, H)
    X, Y = np.meshgrid(X, Y)
    interp = LinearNDInterpolator(dt, values_np, fill_value=np.median(values_np))
    return torch.from_numpy(interp(X, Y)).to(values)


def align_depth_interpolate(
    predicted_depth: PredictedDepth,
    sfm_points_camera_coords: torch.Tensor,
    sfm_points_depth: torch.Tensor,
    sfm_points_error: torch.Tensor,
    intrinsics: CameraIntrinsics,
    cam2world: torch.Tensor,
    config: Config,
    debug_export_dir: Path | None,
):
    """
    Args:
        depth: torch.Tensor of shape (Width, Height)
        sfm_points_camera_coords: torch.Tensor of shape (2, N)
                where N is the number of points, the first row is y and the second row is x.
        gt_depth: torch.Tensor of shape (N,)
    """
    H, W = predicted_depth.depth.shape

    (
        prealigned,
        sfm_points_camera_coords,
        sfm_points_depth,
        scale_factors,
        initial_sfm_point_coords,
        outlier_type,
    ) = interp_common(
        predicted_depth,
        sfm_points_camera_coords,
        sfm_points_depth,
        sfm_points_error,
        intrinsics,
        cam2world,
        config,
        debug_export_dir,
    )

    try:
        scale_map = image_space_linear_interpolation(
            sfm_points_camera_coords,
            scale_factors,
            W,
            H,
        )
    except Exception as e:
        LOGGER.error(
            "Scale factor interpolation failed; using median scale instead of interpolation.",
            e,
        )
        scale_map = scale_factors.median()

    if config.alignment.interp.scale_outlier_removal and debug_export_dir is not None:
        debug_export_outlier_classification(
            initial_sfm_point_coords,
            outlier_type,
            scale_map * prealigned.aligned_depth * prealigned.mask,
            debug_export_dir,
        )

    return DepthAlignmentResult(scale_map * prealigned.aligned_depth, prealigned.mask)


class DepthAlignmentInterpolate(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: PredictedDepth,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        sfm_points_error: torch.Tensor,
        intrinsics: CameraIntrinsics,
        cam2world: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None,
    ) -> DepthAlignmentResult:
        return align_depth_interpolate(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            sfm_points_error,
            intrinsics,
            cam2world,
            config,
            debug_export_dir,
        )
