from pathlib import Path
from typing import Callable

from monodepth.config import Config
from monodepth.depth_alignment.alignment.lstsqrs import align_depth_least_squares
from monodepth.depth_alignment.config import RansacConfig
from monodepth.depth_prediction.interface import (
    CameraIntrinsics,
    PredictedDepth,
)
import math
import torch

from ..interface import DepthAlignmentResult, DepthAlignmentStrategy

from align_depth_ransac import align_depth_ransac


class DepthAlignmentRansac(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: PredictedDepth,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        sfm_points_error: torch.Tensor | None,
        intrinsics: CameraIntrinsics,
        cam2world: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None = None,
        *args,
        **kwargs,
    ) -> DepthAlignmentResult:
        return _align_depth_ransac_generic(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            sfm_points_error,
            False,
            config.alignment.ransac,
            debug_export_dir,
        )


class DepthAlignmentMsac(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: PredictedDepth,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        sfm_points_error: torch.Tensor | None,
        intrinsics: CameraIntrinsics,
        cam2world: torch.Tensor,
        config: Config,
        debug_export_dir: Path | None = None,
        *args,
        **kwargs,
    ) -> DepthAlignmentResult:
        return _align_depth_ransac_generic(
            predicted_depth,
            sfm_points_camera_coords,
            sfm_points_depth,
            sfm_points_error,
            True,
            config.alignment.ransac,
            debug_export_dir,
        )


def _ransac_loss(dists: torch.Tensor, inlier_threshold: float):
    return torch.sum(dists >= inlier_threshold)


def _msac_loss(dists: torch.Tensor, inlier_threshold: float):
    return torch.sum(torch.minimum(dists, torch.full_like(dists, inlier_threshold)))


RansacLossFunc = Callable[[torch.Tensor, float], float]
"""
A function that computes the loss of the alignment between the predicted and ground truth depth maps.

Args:
    squared_distances: The squared distances between the predicted and ground truth depth maps.
    inlier_threshold: The threshold for considering a point an inlier.
Returns: loss
"""


def _required_samples(
    inlier_count: int,
    total_correspondence_count: int,
    min_sample_size: int,
    confidence: float,
):
    # k = log(η)/ log(1 − P_I).
    # P_I ≈ ε^m
    prob_inlier = (inlier_count / total_correspondence_count) ** min_sample_size
    if prob_inlier == 0:
        return float("inf")
    if prob_inlier == 1:
        return 0
    return math.log(1 - confidence) / math.log(1 - prob_inlier)


def _l2_dists_squared(
    h: tuple[float, float], depth: torch.Tensor, gt_depth: torch.Tensor
) -> torch.Tensor:
    return (h[0] * depth + h[1] - gt_depth) ** 2


def _align_depth_ransac_generic(
    predicted_depth: PredictedDepth,
    gt_points_camera_coords: torch.Tensor,
    gt_depth: torch.Tensor,
    sfm_points_error: torch.Tensor | None,
    use_msac_loss: bool,
    config: RansacConfig,
    debug_export_dir: Path | None = None,
) -> DepthAlignmentResult:
    full_predicted_depth = predicted_depth.depth
    depth = predicted_depth.depth[
        gt_points_camera_coords[1], gt_points_camera_coords[0]
    ].flatten()

    (scale, shift, num_inliers_best_lo, num_inliers_best_pre_lo, iteration) = (
        align_depth_ransac(
            depth.cpu().numpy(),
            gt_depth.cpu().numpy(),
            use_msac_loss,
            config.sample_size,
            config.max_iters,
            config.inlier_threshold,
            config.confidence,
            config.min_iters,
        )
    )
    inlier_ratio = num_inliers_best_lo / depth.shape[0]

    ransac_info_str = f"[RANSAC] Iterations: {iteration}, Inliers: {num_inliers_best_lo}/{depth.shape[0]} ({inlier_ratio*100:.0f}%), Inliers before LO: {num_inliers_best_pre_lo}, Scale: {scale}, Shift: {shift}"

    if debug_export_dir is not None:
        debug_export_dir.mkdir(parents=True, exist_ok=True)
        with open(debug_export_dir / "ransac_log.txt", "a") as f:
            f.write(ransac_info_str + "\n")

    print(ransac_info_str)

    return DepthAlignmentResult(
        aligned_depth=full_predicted_depth * scale + shift,
        mask=predicted_depth.mask,
    )
