from enum import IntEnum
from pathlib import Path
from typing import NamedTuple
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
import torch
import logging

from monodepth.config import Config
from monodepth.depth_alignment.alignment.lstsqrs import DepthAlignmentLstSqrs
from monodepth.depth_alignment.alignment.ransacs import (
    DepthAlignmentMsac,
    DepthAlignmentRansac,
)
from monodepth.depth_prediction.interface import (
    CameraIntrinsics,
    PredictedDepth,
)
from ..interface import DepthAlignmentResult
import matplotlib.pyplot as plt

_LOGGER = logging.getLogger(__name__)


def pick_rbf_point_subset(
    num_points,
    max_points,
    sfm_pts_camera_coords,
    sfm_depth,
    device,
):
    indices = torch.randperm(
        num_points,
        device=device,
    )[:max_points]
    return (
        sfm_pts_camera_coords[:, indices],
        sfm_depth[indices],
        indices,
    )


class OutlierType(IntEnum):
    REGULAR = 0
    SCALE_ONLY = 1
    POSITION_ONLY = 2
    BOTH = 3


class OutlierClassification(NamedTuple):
    scale_only_outliers: torch.Tensor
    both_outliers: torch.Tensor
    position_only_outliers: torch.Tensor
    regular: torch.Tensor

    @staticmethod
    def all_inliers(num_points, device):
        return OutlierClassification(
            scale_only_outliers=torch.zeros(
                num_points, dtype=torch.bool, device=device
            ),
            both_outliers=torch.zeros(num_points, dtype=torch.bool, device=device),
            position_only_outliers=torch.zeros(
                num_points, dtype=torch.bool, device=device
            ),
            regular=torch.ones(num_points, dtype=torch.bool, device=device),
        )


def debug_export_outlier_classification(
    sfm_points_camera_coords: torch.Tensor,
    outlier_classification: OutlierClassification,
    aligned_depth: torch.Tensor,
    debug_export_dir: Path,
):
    # Convert tensors to numpy for plotting

    H, W = aligned_depth.shape[:2]

    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Display the aligned depth
    im = ax.imshow(aligned_depth.cpu().numpy(), cmap="viridis")

    # Add colorbar for depth values
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Aligned Depth", rotation=270, labelpad=15)

    # Extract coordinates for each outlier type
    regular_coords = sfm_points_camera_coords[:, outlier_classification.regular]
    scale_only_coords = sfm_points_camera_coords[
        :, outlier_classification.scale_only_outliers
    ]
    position_only_coords = sfm_points_camera_coords[
        :, outlier_classification.position_only_outliers
    ]
    both_coords = sfm_points_camera_coords[:, outlier_classification.both_outliers]

    # Plot different markers for each outlier type
    if len(regular_coords[0]) > 0:
        ax.scatter(
            regular_coords[0].cpu(),
            regular_coords[1].cpu(),
            c="green",
            marker=".",
            s=20,
            alpha=0.8,
            label="Regular",
        )

    if len(scale_only_coords[0]) > 0:
        ax.scatter(
            scale_only_coords[0].cpu(),
            scale_only_coords[1].cpu(),
            c="red",
            marker="x",
            s=20,
            alpha=0.8,
            label="Scale Outliers",
        )

    if len(position_only_coords[0]) > 0:
        ax.scatter(
            position_only_coords[0].cpu(),
            position_only_coords[1].cpu(),
            c="purple",
            marker="+",
            s=20,
            alpha=0.8,
            label="Position Outliers",
        )

    if len(both_coords[0]) > 0:
        ax.scatter(
            both_coords[0].cpu(),
            both_coords[1].cpu(),
            c="yellow",
            marker="^",
            s=20,
            alpha=0.8,
            label="Both Outliers",
        )

    ax.legend(loc="upper right")
    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)

    # Save the visualization
    plt.tight_layout()
    plt.savefig(
        debug_export_dir / "interp_alignment_scale_outliers.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()


def scale_factor_outlier_removal(
    coords: torch.Tensor,
    scales: torch.Tensor,
):
    K_lof = 10
    K_scale_knn = 5

    num_pts = coords.shape[0]
    if num_pts < min(K_lof + 1, K_scale_knn + 1):
        return OutlierClassification(
            scale_only_outliers=torch.zeros(num_pts, dtype=torch.bool),
            both_outliers=torch.zeros(num_pts, dtype=torch.bool),
            position_only_outliers=torch.zeros(num_pts, dtype=torch.bool),
            regular=torch.ones(num_pts, dtype=torch.bool),
        )

    clf = LocalOutlierFactor(n_neighbors=K_lof, n_jobs=-1)
    coords_np = coords.cpu().numpy()
    pred_pts_only = clf.fit_predict(coords_np)
    position_outliers_np = pred_pts_only == -1

    model = NearestNeighbors(n_neighbors=K_scale_knn + 1, metric="euclidean").fit(
        coords_np
    )
    knn_distances, knn_indices = model.kneighbors(coords_np)

    # remove self-distance/index (first column)
    knn_distances = knn_distances[:, 1:]
    knn_indices = knn_indices[:, 1:]
    knn_median_scale = torch.median(scales[knn_indices], dim=1).values
    scale_diff = torch.abs(scales - knn_median_scale)

    iqr_scale = None
    if iqr_scale is None:
        # use 99th percentile of scale_diff as threshold
        threshold = torch.quantile(scale_diff, 0.99)
        scale_outliers = scale_diff > threshold
    else:
        scale_diff_iqr = torch.quantile(scale_diff, 0.75) - torch.quantile(
            scale_diff, 0.25
        )
        lower_bound = torch.quantile(scale_diff, 0.25) - iqr_scale * scale_diff_iqr
        upper_bound = torch.quantile(scale_diff, 0.75) + iqr_scale * scale_diff_iqr
        scale_outliers = (scale_diff < lower_bound) | (scale_diff > upper_bound)

    scale_outliers = scale_outliers & (scale_diff > 0.001)

    position_outliers = torch.from_numpy(position_outliers_np).to(scale_outliers)

    return OutlierClassification(
        scale_only_outliers=scale_outliers & ~position_outliers,
        both_outliers=scale_outliers & position_outliers,
        position_only_outliers=position_outliers & ~scale_outliers,
        regular=~(scale_outliers | position_outliers),
    )


def iqr_scale_factor_outlier_removal(
    scale_factors: torch.Tensor,
) -> OutlierClassification:
    # TODO: Move to config
    IQR_SCALE_FACTOR_OUTLIER_REMOVAL_THRESH = 2.5

    median_ratio = torch.median(scale_factors)

    q75, q25 = torch.quantile(
        scale_factors, torch.tensor([0.75, 0.25]).to(scale_factors)
    )
    iqr = q75 - q25
    threshold = IQR_SCALE_FACTOR_OUTLIER_REMOVAL_THRESH * iqr

    inliers = torch.abs(scale_factors - median_ratio) < threshold

    return OutlierClassification(
        scale_only_outliers=~inliers,
        both_outliers=torch.zeros_like(inliers, dtype=torch.bool),
        position_only_outliers=torch.zeros_like(inliers, dtype=torch.bool),
        regular=inliers,
    )


def initial_alignment(
    predicted_depth: PredictedDepth,
    sfm_points_camera_coords: torch.Tensor,
    gt_depth: torch.Tensor,
    sfm_points_error: torch.Tensor,
    intrinsics: CameraIntrinsics,
    cam2world: torch.Tensor,
    config: Config,
    debug_export_dir: Path | None = None,
) -> DepthAlignmentResult:
    if config.alignment.interp.init is None:
        return DepthAlignmentResult(predicted_depth.depth, predicted_depth.mask)
    if config.alignment.interp.init == "lstsqrs":
        return DepthAlignmentLstSqrs.align(
            predicted_depth,
            sfm_points_camera_coords,
            gt_depth,
            sfm_points_error,
            intrinsics,
            cam2world,
            config,
            debug_export_dir,
        )
    elif config.alignment.interp.init == "ransac":
        return DepthAlignmentRansac.align(
            predicted_depth,
            sfm_points_camera_coords,
            gt_depth,
            sfm_points_error,
            intrinsics,
            cam2world,
            config,
            debug_export_dir,
        )
    elif config.alignment.interp.init == "msac":
        return DepthAlignmentMsac.align(
            predicted_depth,
            sfm_points_camera_coords,
            gt_depth,
            sfm_points_error,
            intrinsics,
            cam2world,
            config,
            debug_export_dir,
        )
    else:
        raise ValueError(
            f"Unknown interp alignment init method: {config.alignment.interp.init}"
        )


def interp_common(
    predicted_depth: PredictedDepth,
    sfm_points_camera_coords: torch.Tensor,
    sfm_points_depth: torch.Tensor,
    sfm_points_error: torch.Tensor,
    intrinsics: CameraIntrinsics,
    cam2world: torch.Tensor,
    config: Config,
    debug_export_dir: Path | None,
):
    prealigned = initial_alignment(
        predicted_depth,
        sfm_points_camera_coords,
        sfm_points_depth,
        sfm_points_error,
        intrinsics,
        cam2world,
        config,
        debug_export_dir,
    )

    # apply prealigned.mask
    unmasked_sfm_points = prealigned.mask[
        sfm_points_camera_coords[1], sfm_points_camera_coords[0]
    ]
    sfm_points_camera_coords = sfm_points_camera_coords[:, unmasked_sfm_points]
    sfm_points_depth = sfm_points_depth[unmasked_sfm_points]
    num_sfm_pts = sfm_points_camera_coords.shape[1]

    scale_factors = (
        sfm_points_depth
        / prealigned.aligned_depth[
            sfm_points_camera_coords[1], sfm_points_camera_coords[0]
        ]
    )
    interp_config = config.alignment.interp
    initial_sfm_point_coords = sfm_points_camera_coords.clone()
    if interp_config.scale_outlier_removal == "complex":
        outlier_type = scale_factor_outlier_removal(
            sfm_points_camera_coords.T, scale_factors
        )
        outlier_mask = outlier_type.scale_only_outliers
        if outlier_mask.sum() > 0:
            _LOGGER.info(
                "Removed %d/%d scale outlier points.",
                outlier_mask.sum().item(),
                num_sfm_pts,
            )
        scale_factors = scale_factors[~outlier_mask]
        sfm_points_camera_coords = sfm_points_camera_coords[:, ~outlier_mask]
        sfm_points_depth = sfm_points_depth[~outlier_mask]
    elif interp_config.scale_outlier_removal == "iqr":
        outlier_type = iqr_scale_factor_outlier_removal(scale_factors)
        outlier_mask = outlier_type.scale_only_outliers
        if outlier_mask.sum() > 0:
            _LOGGER.info(
                "Removed %d/%d scale outlier points using IQR method.",
                outlier_mask.sum().item(),
                num_sfm_pts,
            )
        scale_factors = scale_factors[~outlier_mask]
        sfm_points_camera_coords = sfm_points_camera_coords[:, ~outlier_mask]
        sfm_points_depth = sfm_points_depth[~outlier_mask]
    else:
        outlier_type = OutlierClassification.all_inliers(
            num_sfm_pts, predicted_depth.depth.device
        )
    return (
        prealigned,
        sfm_points_camera_coords,
        sfm_points_depth,
        scale_factors,
        initial_sfm_point_coords,
        outlier_type,
    )
