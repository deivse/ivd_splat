import torch
import numpy as np

from monodepth.depth_subsampling.config import NumSfMPointsMaskConfig


def calculate_patch_sizes(
    image_shape: tuple[int, int], num_patches_small_axis
) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Calculate patch size and grid based on image shape and number of patches on the smaller axis.

    Args:
        image_shape: Shape of the image (height, width).
        num_patches_small_axis: Number of patches on the smaller image axis.
    Returns:
        A tuple containing:
        - patch_size: A tuple (patch_height, patch_width).
        - patch_grid: A tuple (num_patches_height, num_patches_width).
    """
    small_axis = np.argmin([image_shape[0], image_shape[1]])
    large_axis = 1 - small_axis
    patch_size_small_axis = int(image_shape[small_axis] // num_patches_small_axis)
    num_patches_large_axis = int(
        np.ceil(image_shape[large_axis] / patch_size_small_axis)
    )
    patch_size_large_axis = int(image_shape[large_axis] // num_patches_large_axis)

    if small_axis == 0:
        patch_grid = (num_patches_small_axis, int(num_patches_large_axis))
        patch_size = (patch_size_small_axis, patch_size_large_axis)
    else:
        patch_grid = (int(num_patches_large_axis), num_patches_small_axis)
        patch_size = (patch_size_large_axis, patch_size_small_axis)
    return patch_size, patch_grid


def num_sfm_points_mask(
    sfm_points_camera: torch.Tensor,
    imsize: tuple[int, int],
    sfm_pts_mask_config: NumSfMPointsMaskConfig,
) -> torch.Tensor:
    mask = torch.ones(imsize, dtype=bool)
    patch_size, patch_grid = calculate_patch_sizes(
        imsize, sfm_pts_mask_config.num_patches_small_axis
    )

    for i in range(patch_grid[0]):
        for j in range(patch_grid[1]):
            y_start = i * patch_size[0]
            y_end = min((i + 1) * patch_size[0], imsize[0])
            x_start = j * patch_size[1]
            x_end = min((j + 1) * patch_size[1], imsize[1])

            points_in_patch = (
                (sfm_points_camera[0, :] >= x_start)
                & (sfm_points_camera[0, :] < x_end)
                & (sfm_points_camera[1, :] >= y_start)
                & (sfm_points_camera[1, :] < y_end)
            )

            if points_in_patch.sum().item() > sfm_pts_mask_config.threshold:
                mask[y_start:y_end, x_start:x_end] = False
    return mask
