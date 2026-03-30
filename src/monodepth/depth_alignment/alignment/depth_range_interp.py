from pathlib import Path
import torch
import logging

from monodepth.config import Config
from monodepth.depth_alignment.alignment.interp_common import (
    debug_export_outlier_classification,
    interp_common,
)

from monodepth.depth_prediction.interface import (
    CameraIntrinsics,
    PredictedDepth,
)
from ..interface import DepthAlignmentResult, DepthAlignmentStrategy

LOGGER = logging.getLogger(__name__)


def piecewise_linear_depth_space_interp(
    depths: torch.Tensor, z_pred: torch.Tensor, z_colmap: torch.Tensor
) -> torch.Tensor:
    # Based on https://github.com/OpsiClear/DepthDensifier/blob/44148aa16c6b20c3e3060ca0673c071bedc415e0/src/depthdensifier/depth_refiner.py#L141

    median_scale = torch.median(z_colmap / (z_pred + 1e-6))
    depths = depths * median_scale
    z_pred = z_pred * median_scale

    if len(depths) < 4:
        return depths

    sorted_indices = torch.argsort(z_pred)
    z_pred_sorted = z_pred[sorted_indices]
    z_colmap_sorted = z_colmap[sorted_indices]

    # Ensure we have at least 2 points for interpolation
    if len(z_pred_sorted) < 2:
        return depths

    # Find bin for each
    indices = torch.searchsorted(z_pred_sorted, depths, right=False)
    indices = torch.clamp(indices, 1, len(z_pred_sorted) - 1)

    # Bin intervals
    x0 = z_pred_sorted[indices - 1]
    x1 = z_pred_sorted[indices]
    y0 = z_colmap_sorted[indices - 1]
    y1 = z_colmap_sorted[indices]

    # Lengths of intervals
    dx = x1 - x0
    dx[dx == 0] = 1e-6

    # lerp
    t = (depths - x0) / dx
    t = torch.clamp(t, 0, 1)
    result = y0 + t * (y1 - y0)

    return result


def median_filter_3x3(depth: torch.Tensor) -> torch.Tensor:
    depth_padded = torch.nn.functional.pad(
        depth[None, None], (1, 1, 1, 1), mode="replicate"
    )
    patches = torch.nn.functional.unfold(depth_padded, kernel_size=3, stride=1)
    patches = patches.view(1, 9, -1).permute(0, 2, 1)
    median_values, _ = torch.median(patches, dim=2)
    return median_values.view(depth.shape)


def align_depth_range_interpolate(
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
    imsize = predicted_depth.depth.shape

    try:
        depth = piecewise_linear_depth_space_interp(
            prealigned.aligned_depth.reshape(-1),
            prealigned.aligned_depth[
                sfm_points_camera_coords[1], sfm_points_camera_coords[0]
            ],
            sfm_points_depth,
        )
        depth = depth.reshape(imsize)
        depth = median_filter_3x3(depth)
    except Exception as e:
        LOGGER.error(
            "Scale factor interpolation failed; using median scale instead of interpolation.",
            e,
        )
        depth = prealigned.aligned_depth * scale_factors.median()

    if config.alignment.interp.scale_outlier_removal and debug_export_dir is not None:
        debug_export_outlier_classification(
            initial_sfm_point_coords,
            outlier_type,
            depth * prealigned.mask,
            debug_export_dir,
        )

    return DepthAlignmentResult(depth, prealigned.mask)


class DepthAlignmentDepthRangeInterpolate(DepthAlignmentStrategy):
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
        return align_depth_range_interpolate(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            sfm_points_error,
            intrinsics,
            cam2world,
            config,
            debug_export_dir,
        )
