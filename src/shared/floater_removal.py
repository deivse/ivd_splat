from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm

from shared.geom_utils import project_points


def floater_removal(
    intrinsics: Iterable[torch.Tensor],
    extrinsics: Iterable[torch.Tensor],
    depth_maps: Iterable[torch.Tensor],
    all_points_tensor: torch.Tensor,
    all_normals_tensor: torch.Tensor | None = None,
    device="cpu",
) -> np.ndarray:
    """
    This is a modified version of the floater removal from https://github.com/OpsiClear/DepthDensifier
    The main difference is that we count both floater votes and consistent votes, and decide based on the fraction,
    not a threshold on the number of floater votes. This accounts for the fact that different points are visible in different numbers of views.

    Args:
        intrinsics: Iterable of tensors of shape (3, 3) containing camera intrinsics for each view.
        extrinsics: Iterable of tensors of shape (4, 4) containing camera extrinsics for each view.
        depth_maps: Iterable of tensors of shape (H, W) containing depth maps for each view.
        all_points_tensor: Tensor of shape (M, 3) containing the 3D points to be filtered.
        all_normals_tensor: Optional tensor of shape (M, 3) containing normals for the 3D points.
                            If provided, a grazing angle check will be performed to avoid considering points that are at a grazing angle to the camera
                            to avoid false positives in floater detection.
        device: The device to perform computations on.
    Returns:
        A boolean numpy array of shape (M,) where True indicates points to keep and False indicates points classified as floaters.
    """
    consistent_votes = torch.zeros(
        all_points_tensor.shape[0], dtype=torch.int, device=device
    )
    floater_votes = torch.zeros(
        all_points_tensor.shape[0], dtype=torch.int, device=device
    )

    all_points_tensor = all_points_tensor.to(device)
    if all_normals_tensor is not None:
        all_normals_tensor = all_normals_tensor.to(device)

    for data in tqdm(
        list(zip(intrinsics, extrinsics, depth_maps)),
        desc="Floater removal",
    ):
        intrinsics = data[0].to(device)
        extrinsics = data[1].to(device)
        depth = data[2].to(device)
        h, w = depth.shape

        P = intrinsics @ extrinsics
        R = extrinsics[:3, :3]
        C = -R.T @ extrinsics[:3, 3]

        curr_view_pts_2d, curr_view_pts2d_depths = project_points(all_points_tensor, P)

        if all_normals_tensor is not None:
            # --- Grazing Angle Check ---
            # Calculate the viewing direction from the camera to each point.
            viewing_dirs = all_points_tensor - C
            viewing_dirs /= torch.linalg.norm(viewing_dirs, dim=1)[:, None]

            # Calculate the dot product between the point's normal and the viewing direction.
            # A dot product close to zero means a grazing angle.
            # We negate the viewing direction because the normal points "out" of the surface.
            dot_products = torch.sum(all_normals_tensor * -viewing_dirs, dim=1)

            # Create a mask to only consider points that are not at a grazing angle.
            # We use a threshold (e.g., cos(85 degrees) approx 0.087) to filter.
            not_grazing_mask = dot_products > 0.087
        else:
            not_grazing_mask = torch.ones(all_points_tensor.shape[0], dtype=torch.bool)

        u, v = curr_view_pts_2d[:, 0], curr_view_pts_2d[:, 1]

        # Create a mask for points that project inside the image bounds AND are not at a grazing angle
        mask_in_bounds = (
            (u >= 0)
            & (u < w)
            & (v >= 0)
            & (v < h)
            & (curr_view_pts2d_depths > 0)
            & not_grazing_mask
        )
        if not torch.any(mask_in_bounds):
            continue

        # Get integer coordinates for depth lookup
        u_valid = u[mask_in_bounds].to(torch.int)
        v_valid = v[mask_in_bounds].to(torch.int)

        projected_depths_valid = curr_view_pts2d_depths[mask_in_bounds]
        refined_depths_at_projections = depth[v_valid, u_valid]

        # Create a mask for where the lookup is valid (non-zero depth)
        valid_lookup_mask = refined_depths_at_projections > 0

        DEPTH_THRESHOLD = 0.7

        # A point is a "floater" if its projected depth is significantly
        # LESS than the depth map's value (i.e., it's between the camera and the surface).
        inconsistent_mask = (
            projected_depths_valid[valid_lookup_mask]
            < DEPTH_THRESHOLD * refined_depths_at_projections[valid_lookup_mask]
        )
        consistent_mask = torch.logical_and(
            ~inconsistent_mask,
            projected_depths_valid[valid_lookup_mask]
            < 0.3 * refined_depths_at_projections[valid_lookup_mask],
        )

        # Get the original indices of inconsistent points and increment their vote count
        original_indices_in_bounds = torch.where(mask_in_bounds)[0]
        indices_with_valid_lookup = original_indices_in_bounds[valid_lookup_mask]
        inconsistent_indices = indices_with_valid_lookup[inconsistent_mask]
        consistent_indices = indices_with_valid_lookup[consistent_mask]

        floater_votes[inconsistent_indices] += 1
        consistent_votes[consistent_indices] += 1

    FLOATER_VOTE_FRACTION_THRESH = 0.4
    MIN_IMAGES = 3
    few_images_mask = (floater_votes + consistent_votes) < MIN_IMAGES
    voted_non_floater_mask = (
        floater_votes / (floater_votes + consistent_votes + 1e-6)
        < FLOATER_VOTE_FRACTION_THRESH
    )
    points_to_keep_mask = few_images_mask | voted_non_floater_mask
    return points_to_keep_mask.cpu().numpy()
