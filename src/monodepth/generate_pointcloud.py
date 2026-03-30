import gc
import cv2
from nerfbaselines._types import Dataset
from nerfbaselines.io import save_depth as nb_save_depth
from tqdm import tqdm
from torch.types import Device as TorchDevice
from monodepth.config import Config


import logging
from pathlib import Path
from typing import Optional
import numpy as np
import torch

from matplotlib import pyplot as plt

from shared.point_cloud_io import export_pointcloud_ply
from monodepth.depth_alignment.debug_export import debug_export_alignment
from monodepth.depth_alignment.exceptions import LowDepthAlignmentConfidenceError
from monodepth.depth_alignment.interface import DepthAlignmentResult
from monodepth.depth_prediction.interface import (
    CameraIntrinsics,
    DepthPredictor,
    PredictedDepth,
)
from monodepth.depth_subsampling.num_sfm_points_mask import num_sfm_points_mask
from shared.scene_subdir import get_scene_subdir
from shared.select_cameras_kmeans import select_cameras_kmeans

_LOGGER = logging.getLogger(__name__)


def downsample_image(
    image: np.ndarray, intrinsics: CameraIntrinsics, target_max_dim: int
) -> tuple[np.ndarray, CameraIntrinsics]:
    max_dim = max(image.shape[0], image.shape[1])
    if max_dim > target_max_dim:
        scale_factor = target_max_dim / max_dim
        new_width = round(image.shape[1] * scale_factor)
        new_height = round(image.shape[0] * scale_factor)
        _LOGGER.debug(
            f"Downsampling image from {image.shape[1]}x{image.shape[0]} to {new_width}x{new_height}"
        )
        new_size = (new_width, new_height)
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)
        intrinsics = CameraIntrinsics(data=intrinsics.data * scale_factor)
    return image, intrinsics


def debug_export_depth_map(depth: torch.Tensor, mask: torch.Tensor, dir: Path):
    dir.mkdir(exist_ok=True, parents=True)
    depth_np: np.ndarray = depth.cpu().numpy()
    mask_np: np.ndarray = mask.cpu().numpy()
    depth_vis: np.ndarray = depth_np.copy()

    vmax = np.percentile(depth_np[mask_np.reshape(depth.shape)], 99)
    depth_vis = depth_vis / vmax
    depth_vis = np.clip(depth_vis, 0.0, 1.0)

    colormap = plt.get_cmap("viridis")
    depth_vis = colormap(depth_vis)[:, :, :3]
    depth_vis[~mask_np.reshape(depth.shape)] = 0.0
    plt.imsave(dir / "masked_depth.png", depth_vis)


def get_valid_sfm_pts(
    sfm_pts_camera,
    sfm_pts_camera_depth,
    sfm_points_error,
    mask,
    imsize,
    debug_export_dir,
    image,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    valid_sfm_pt_indices = torch.logical_and(
        torch.logical_and(sfm_pts_camera[0] >= 0, sfm_pts_camera[0] < imsize[0]),
        torch.logical_and(sfm_pts_camera[1] >= 0, sfm_pts_camera[1] < imsize[1]),
    )
    valid_sfm_pt_indices = torch.logical_and(
        valid_sfm_pt_indices, sfm_pts_camera_depth >= 0
    )
    print(
        f"Num invalid reprojected SfM points: {sfm_pts_camera.shape[1] - torch.sum(valid_sfm_pt_indices)} out of {sfm_pts_camera.shape[1]}"
    )

    if debug_export_dir is not None:
        debug_export_dir.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(10, 10))
        plt.scatter(
            sfm_pts_camera[0, ~valid_sfm_pt_indices].cpu(),
            sfm_pts_camera[1, ~valid_sfm_pt_indices].cpu(),
            s=1,
            c="red",
        )
        plt.scatter(
            sfm_pts_camera[0, valid_sfm_pt_indices].cpu(),
            sfm_pts_camera[1, valid_sfm_pt_indices].cpu(),
            s=1,
            c="blue",
        )
        plt.imshow(image.clone().cpu())
        plt.axis("off")
        plt.savefig(
            debug_export_dir / "sfm_points_on_image.png", bbox_inches="tight", dpi=300
        )
        plt.close()

    if torch.sum(valid_sfm_pt_indices) < sfm_pts_camera.shape[1] / 4:
        raise LowDepthAlignmentConfidenceError(
            "Less than 1/4 of SFM points",
            f" ({torch.sum(valid_sfm_pt_indices).item()} / {sfm_pts_camera.shape[1]})"
            f" reprojected into image bounds. Depth < 0: {(sfm_pts_camera_depth < 0).sum().item()} points, rest are out of image bounds.",
        )

    # Set invalid points to 0 so we can index the mask with them
    # Will be filtered out later anyways
    sfm_pts_camera[:, ~valid_sfm_pt_indices] = torch.zeros_like(
        sfm_pts_camera[:, ~valid_sfm_pt_indices]
    )
    valid_sfm_pt_indices = torch.logical_and(
        valid_sfm_pt_indices, mask[sfm_pts_camera[1], sfm_pts_camera[0]]
    )
    return (
        sfm_pts_camera[:, valid_sfm_pt_indices],
        sfm_pts_camera_depth[valid_sfm_pt_indices],
        (
            sfm_points_error[valid_sfm_pt_indices]
            if sfm_points_error is not None
            else None
        ),
    )


def project_points(
    points3D: torch.Tensor, P: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    points3D_h = torch.hstack(
        [points3D, torch.ones((points3D.shape[0], 1), device=points3D.device)]
    )
    points2D_h = points3D_h @ P.T
    points2D = points2D_h[:, :2] / points2D_h[:, 2:3]
    depth = points2D_h[:, 2]
    return points2D, depth


def project_and_filter_sfm_pts(
    image: torch.Tensor,
    sfm_points: torch.Tensor,
    sfm_points_error: torch.Tensor | None,
    P: torch.Tensor,
    imsize: torch.Tensor,
    predicted_depth: PredictedDepth,
    debug_export_dir: Path | None,
):
    sfm_points_camera, sfm_points_depth = project_points(sfm_points, P)

    sfm_points_camera = torch.round(sfm_points_camera).to(torch.int).T
    sfm_points_camera, sfm_points_depth, sfm_points_error = get_valid_sfm_pts(
        sfm_points_camera,
        sfm_points_depth,
        sfm_points_error,
        predicted_depth.mask,
        imsize,
        debug_export_dir,
        image,
    )
    return sfm_points_camera, sfm_points_depth, sfm_points_error


def depth_gradient_mask(
    depth: torch.Tensor,
    gradient_threshold: float,
) -> torch.Tensor:
    """Computes a mask of pixels where the depth gradient is below a threshold.

    Args:
        depth: (H, W) tensor of depth values.
        gradient_threshold: Threshold for the depth gradient.
    Returns:
        A boolean tensor of shape (H, W) where True indicates that the depth
        gradient is below the threshold.
    """
    depth_dx = torch.abs(depth[:, 1:] - depth[:, :-1])
    depth_dy = torch.abs(depth[1:, :] - depth[:-1, :])
    depth_grad_both = torch.zeros_like(depth, dtype=depth.dtype)
    depth_grad_both[:, 1:] += depth_dx
    depth_grad_both[1:, :] += depth_dy
    depth_grad_both = depth_grad_both - depth_grad_both.min()
    depth_grad_both = depth_grad_both / (depth_grad_both.max() + 1e-8)
    return depth_grad_both <= gradient_threshold


def get_depth_map(
    predictor: DepthPredictor,
    image: torch.Tensor,
    cam2world: torch.Tensor,
    intrinsics: CameraIntrinsics,
    point_indices: torch.Tensor,
    image_path: Path,
    dataset: Dataset,
    scene_subdir: Path,
    device: TorchDevice,
    config: Config,
    debug_export_dir: Path | None,
):
    predicted_depth = predictor.predict_depth_or_get_cached_depth(
        image, intrinsics, Path(image_path).name, scene_subdir
    )
    if torch.any(torch.isinf(predicted_depth.depth[predicted_depth.mask])):
        _LOGGER.warning("Encountered infinite depths in predicted depth map.")

    imsize = predicted_depth.depth.T.shape

    cam2world = cam2world.float()
    R = cam2world[:3, :3].T
    C = cam2world[:3, 3]
    K = intrinsics.get_K_matrix()
    P = K @ R @ torch.hstack([torch.eye(3), -C[:, None]])

    sfm_points = (
        torch.from_numpy(dataset["points3D_xyz"][point_indices]).to(device).float()
    )
    sfm_points_error = (
        None
        if dataset["points3D_error"] is None
        else (
            torch.from_numpy(dataset["points3D_error"][point_indices])
            .to(device)
            .float()
        )
    )

    P = P.to(device)
    K = K.to(device)

    sfm_points_camera, sfm_points_depth, sfm_points_error = project_and_filter_sfm_pts(
        image.data,
        sfm_points,
        sfm_points_error,
        P,
        imsize,
        predicted_depth,
        debug_export_dir,
    )

    aligned_depth, mask = config.alignment_method.get_implementation().align(
        predicted_depth,
        sfm_points_camera,
        sfm_points_depth,
        sfm_points_error,
        intrinsics,
        cam2world,
        config,
        debug_export_dir,
    )
    if debug_export_dir is not None:
        debug_export_alignment(
            predicted_depth, DepthAlignmentResult(aligned_depth, mask), debug_export_dir
        )

    if torch.any(torch.isinf(aligned_depth[mask])):
        _LOGGER.warning("Encountered negative depths in aligned depth map.")

    subsampling_mask = config.instantiate_subsampler().get_mask(
        image.data, aligned_depth, mask
    )

    mask = (mask & (aligned_depth >= 0)).flatten()

    if config.depth_grad_mask_thresh is not None:
        depth_grad_mask = depth_gradient_mask(
            aligned_depth, config.depth_grad_mask_thresh
        ).flatten()
        mask &= depth_grad_mask
    if config.use_num_sfm_points_mask:
        mask &= (
            num_sfm_points_mask(
                sfm_points_camera,
                (imsize[1], imsize[0]),
                config.num_sfm_points_mask,
            )
            .flatten()
            .to(mask.device)
        )

    if debug_export_dir is not None:
        debug_export_depth_map(aligned_depth, mask, debug_export_dir)

    mask = mask & subsampling_mask
    return aligned_depth, mask, predicted_depth.normal


def monocular_depth_init(
    dataset: Dataset, config: Config, sfm_pts_mask: np.ndarray, device: TorchDevice
):
    predictor = config.instantiate_predictor(device)
    images, cameras, image_paths, pts_indices = (
        dataset["images"],
        dataset["cameras"],
        dataset["image_paths"],
        dataset["images_points3D_indices"],
    )
    scene_subdir = get_scene_subdir(config.scene)
    base_debug_export_dir: Optional[Path] = None
    if config.debug_output:
        _LOGGER.info(
            "Debug output enabled, outputting to %s", config.output_dir / "debug_out"
        )
        base_debug_export_dir = config.output_dir / "debug_out"
        base_debug_export_dir.mkdir(parents=True, exist_ok=True)

    all_points = []
    all_colors = []
    all_normals = []

    sfm_masked_out_indices = np.where(~sfm_pts_mask)[0]

    if config.max_num_images is not None and len(images) > config.max_num_images:
        final_rows = np.zeros((len(images), 1, 4), dtype=cameras.poses.dtype)
        final_rows[:, :, 3] = 1.0
        poses = np.concatenate([cameras.poses, final_rows], axis=1)

        camera_poses_flattened = torch.from_numpy(poses.reshape(-1, 16)).float()
        camera_indices = select_cameras_kmeans(
            camera_poses_flattened, config.max_num_images
        )
        _LOGGER.info(
            f"Dataset has {len(images)} images, but max_num_images is set to {config.max_num_images}. Selecting a subset of images to load using kmeans camera selection."
        )
    else:
        camera_indices = np.arange(len(images))

    depth_iterables = list(zip(images, cameras, image_paths, pts_indices))
    cached_data: list[dict] = []

    for i in tqdm(camera_indices, desc="Predicting & aligning depth maps"):
        image, camera, image_path, point_indices = depth_iterables[i]
        point_indices = point_indices[~np.isin(point_indices, sfm_masked_out_indices)]

        camera = camera.item()
        intrinsics = CameraIntrinsics(camera.intrinsics)
        if config.target_image_size > 0:
            image, intrinsics = downsample_image(
                image, intrinsics, config.target_image_size
            )

        cam2world = torch.from_numpy(camera.poses)
        image = torch.from_numpy(image).to(device).float() / 255.0

        debug_export_dir: Optional[Path] = None
        if base_debug_export_dir is not None:
            debug_export_dir = base_debug_export_dir / Path(image_path).stem
            debug_export_dir.mkdir(parents=True, exist_ok=True)

        try:
            depth, mask, predicted_depth_normal = get_depth_map(
                predictor,
                image,
                cam2world,
                intrinsics,
                point_indices,
                Path(image_path),
                dataset,
                scene_subdir,
                device,
                config,
                debug_export_dir,
            )
        except LowDepthAlignmentConfidenceError as e:
            _LOGGER.warning(
                f"Low depth alignment confidence for image {image_path}: {e}. Skipping this image."
            )
            continue

        imsize = depth.T.shape
        pts_camera: torch.Tensor = torch.dstack(
            [
                torch.from_numpy(np.mgrid[0 : imsize[0], 0 : imsize[1]].T).to(device),
                depth,
            ],
        ).reshape(-1, 3)[mask]

        pts_camera[:, 0] = (pts_camera[:, 0] + 0.5) * pts_camera[:, 2]
        pts_camera[:, 1] = (pts_camera[:, 1] + 0.5) * pts_camera[:, 2]

        pts_world = (
            torch.linalg.inv(intrinsics.get_K_matrix().to(device))
            @ pts_camera.reshape((-1, 3)).T
        )
        pts_world = (
            cam2world.to(device)
            @ torch.vstack([pts_world, torch.ones(pts_world.shape[1], device=device)])
        )[:3].T

        all_points.append(pts_world.cpu())
        all_colors.append(image.reshape(-1, 3)[mask].cpu())
        if predicted_depth_normal is not None:
            all_normals.append(predicted_depth_normal.reshape(-1, 3)[mask].cpu())
        if config.save_depth_maps:
            depth_save_path = (
                config.output_dir
                / "depth_maps"
                / f"aligned_depth_{Path(image_path).stem}.bin"
            )
            depth_save_path.parent.mkdir(parents=True, exist_ok=True)
            nb_save_depth(depth_save_path, depth.cpu().numpy())

        cached_data.append(
            {
                "depth": depth.cpu(),
                "mask": mask.cpu(),
                "intrinsics": intrinsics,
                "cam2world": cam2world.cpu(),
            }
        )

    all_points_tensor = torch.cat(all_points, dim=0)
    all_colors_tensor = torch.cat(all_colors, dim=0)
    all_normals_tensor = torch.cat(all_normals, dim=0) if all_normals else None

    if config.floater_removal:
        # This section is a modified version of the floater removal from https://github.com/OpsiClear/DepthDensifier
        # The main difference is that we count both floater votes and consistent votes, and decide based on the fraction,
        # not a threshold on the number of floater votes. This accounts for the fact that different points are visible in different numbers of views.
        consistent_votes = torch.zeros(
            all_points_tensor.shape[0], dtype=torch.int, device="cpu"
        )
        floater_votes = torch.zeros(
            all_points_tensor.shape[0], dtype=torch.int, device="cpu"
        )
        for data in tqdm(cached_data, desc="Floater removal"):
            refined_depth: torch.Tensor = data["depth"]
            mask = data["mask"]
            intrinsics = data["intrinsics"]
            cam2world = data["cam2world"]
            h, w = refined_depth.shape

            cam2world = cam2world.float()
            R = cam2world[:3, :3].T
            C = cam2world[:3, 3]
            K = intrinsics.get_K_matrix()
            P = K @ R @ torch.hstack([torch.eye(3), -C[:, None]])

            curr_view_pts_2d, curr_view_pts2d_depths = project_points(
                all_points_tensor, P
            )

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
                not_grazing_mask = torch.ones(
                    all_points_tensor.shape[0], dtype=torch.bool
                )

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
            refined_depths_at_projections = refined_depth[v_valid, u_valid]

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

        all_points_tensor = all_points_tensor[points_to_keep_mask]
        all_colors_tensor = all_colors_tensor[points_to_keep_mask]

    return (
        all_points_tensor.cpu().numpy(),
        all_colors_tensor.cpu().numpy(),
    )
