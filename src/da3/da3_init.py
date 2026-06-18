from __future__ import annotations

from dataclasses import dataclass
import logging
import os

from torch.types import Device as TorchDevice

from depth_anything_3.specs import Prediction
import numpy as np
import torch
import typer

from nerfbaselines import Dataset, camera_model_from_int
from depth_anything_3.services.inference_service import InferenceService
from depth_anything_3.utils.export.glb import (
    _as_homogeneous44,
)
from da3.config import DA3Config
from shared.floater_removal import floater_removal
from shared.select_cameras_kmeans import select_cameras_kmeans

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

_LOGGER = logging.getLogger(__name__)


@dataclass
class ProcessResult:
    """Data class to hold processing results"""

    image_paths: list[str]
    extrinsics: np.ndarray
    intrinsics: np.ndarray


def process_and_filter_nerfbaselines_dataset(
    dataset: Dataset, config: DA3Config
) -> ProcessResult:
    cameras = dataset["cameras"]

    if config.max_num_images is not None and len(cameras) > config.max_num_images:
        _LOGGER.info(
            f"Dataset has {len(cameras)} images, but max_num_images is set to {config.max_num_images}. Selecting a subset of images to load using kmeans camera selection."
        )
        final_rows = np.zeros((len(cameras), 1, 4), dtype=cameras.poses.dtype)
        final_rows[:, :, 3] = 1.0
        poses = np.concatenate([cameras.poses, final_rows], axis=1)

        camera_poses_flattened = torch.from_numpy(poses.reshape(-1, 16)).float()
        camera_indices = select_cameras_kmeans(
            camera_poses_flattened, config.max_num_images
        )
    else:
        camera_indices = list(range(len(cameras)))

    image_paths = [dataset["image_paths"][i] for i in camera_indices]
    extrinsics = []
    intrinsics = []

    for i in camera_indices:
        camera = cameras[i].item()

        # Create extrinsic matrix (world to camera)
        c2w = camera.poses

        extrinsic = np.eye(4)
        extrinsic[:3, :3] = c2w[:3, :3].T  # Transpose rotation for world-to-camera
        extrinsic[:3, 3] = -c2w[:3, :3].T @ c2w[:3, 3]  # Invert translation
        extrinsics.append(extrinsic)

        # Create intrinsics matrix
        assert camera_model_from_int(camera.camera_models) == "pinhole"
        fx, fy, cx, cy = camera.intrinsics
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    return ProcessResult(
        image_paths=image_paths,
        extrinsics=np.array(extrinsics),
        intrinsics=np.array(intrinsics),
    )


class ModifiedInferenceService(InferenceService):
    def run_local_inference_batch_all_no_export(
        self,
        data: ProcessResult,
        process_res: int = 504,
        process_res_method: str = "upper_bound_resize",
        align_to_input_ext_scale: bool = True,
        use_ray_pose: bool = False,
        ref_view_strategy: str = "saddle_balanced",
        infer_gs: bool = False,
    ) -> Prediction:
        model = self.load_model()

        typer.echo(f"Running inference on {len(data.image_paths)} images...")

        inference_kwargs = {
            "image": data.image_paths,
            "process_res": process_res,
            "process_res_method": process_res_method,
            "align_to_input_ext_scale": align_to_input_ext_scale,
            "use_ray_pose": use_ray_pose,
            "ref_view_strategy": ref_view_strategy,
            "extrinsics": data.extrinsics,
            "intrinsics": data.intrinsics,
            "infer_gs": infer_gs,
        }

        prediction = model.inference(**inference_kwargs)

        # assert prediction.intrinsics is not None and np.allclose(
        #     prediction.intrinsics, data.intrinsics
        # )
        # assert prediction.extrinsics is not None and np.allclose(
        #     prediction.extrinsics, data.extrinsics
        # )

        return prediction


@dataclass
class ConfLevel:
    threshold: float
    probability: float


def normals_from_world_pts(points_world: torch.Tensor, h: int, w: int) -> torch.Tensor:
    P = points_world.view(h, w, 3)
    # Central differences of 3D positions
    dPdu = torch.zeros_like(P)
    dPdv = torch.zeros_like(P)

    # Interior pixels
    dPdu[:, 1:-1] = 0.5 * (P[:, 2:] - P[:, :-2])
    dPdv[1:-1, :] = 0.5 * (P[2:, :] - P[:-2, :])

    # Left/right boundaries
    dPdu[:, 0] = P[:, 1] - P[:, 0]
    dPdu[:, -1] = P[:, -1] - P[:, -2]

    # Top/bottom boundaries
    dPdv[0] = P[1] - P[0]
    dPdv[-1] = P[-1] - P[-2]

    # Cross product of tangents
    normals = torch.cross(dPdu, dPdv, dim=-1)

    # Normalize
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1e-8)
    return normals


def filter_by_conf_project_and_estimate_normals(
    depth: np.ndarray,
    K: np.ndarray,
    ext_w2c: np.ndarray,
    images_u8: np.ndarray,
    conf: np.ndarray,
    conf_levels: list[ConfLevel],
    device: TorchDevice = "cuda",
) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    For each frame, transform (u,v,1) through K^{-1} to get rays,
    multiply by depth to camera frame, then use (w2c)^{-1} to transform to world frame.
    Simultaneously extract colors.
    """
    N, H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    ones = np.ones_like(us)
    pix = np.stack([us, vs, ones], axis=-1).reshape(-1, 3)  # (H*W,3)

    pts_all, col_all, normals_all = [], [], []

    for i in range(N):
        d = depth[i]  # (H,W)
        rand_floats = np.random.rand(*d.shape)
        valid = np.zeros_like(d, dtype=bool)
        prev_threshold = float("inf")
        for level in conf_levels:
            valid |= (
                (conf[i] < prev_threshold)
                & (conf[i] >= level.threshold)
                & (rand_floats <= level.probability)
            )
            prev_threshold = level.threshold
        valid &= np.isfinite(d) & (d > 0)
        if not np.any(valid):
            continue

        d_flat = d.reshape(-1)
        vidx = np.flatnonzero(valid.reshape(-1))

        K_inv = np.linalg.inv(K[i])  # (3,3)
        c2w = np.linalg.inv(_as_homogeneous44(ext_w2c[i]))  # (4,4)

        rays = K_inv @ pix.T  # (3,M)
        Xc = rays * d_flat[None, :]  # (3,M)
        Xc_h = np.vstack([Xc, np.ones((1, Xc.shape[1]))])
        Xw = (c2w @ Xc_h)[:3].T.astype(np.float32)  # (M,3)

        Xw_tensor = torch.from_numpy(Xw).float().to(device)
        n_flat = normals_from_world_pts(Xw_tensor, H, W).reshape(-1, 3)

        pts_all.append(Xw[vidx])
        col_all.append(images_u8[i].reshape(-1, 3)[vidx].astype(np.uint8))  # (M,3))
        normals_all.append(n_flat[vidx])

    if len(pts_all) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            torch.zeros((0, 3), dtype=torch.float32),
        )

    return (
        np.concatenate(pts_all, 0),
        np.concatenate(col_all, 0),
        torch.cat(normals_all, 0),
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


def da3_init(
    dataset: Dataset,
    config: DA3Config,
    device: TorchDevice = "cuda",
) -> tuple[np.ndarray, np.ndarray, Prediction]:
    """Run pose conditioned depth estimation on dataset.

    Args:
        dataset: Input dataset
        config: DA3 init configuration
        device: Device to use
    """

    try:
        processed_dataset = process_and_filter_nerfbaselines_dataset(dataset, config)
    except Exception as e:
        raise RuntimeError(
            f"Failed to convert NerfBaselines dataset to DA3 conventions: {e}"
        ) from e

    inference_service = ModifiedInferenceService(
        model_dir=config.model_dir, device=device
    )
    prediction = inference_service.run_local_inference_batch_all_no_export(
        processed_dataset,
        align_to_input_ext_scale=True,
        process_res=config.process_res,
        process_res_method=config.process_res_method,
        use_ray_pose=config.use_ray_pose,
        ref_view_strategy=config.ref_view_strategy,
        infer_gs=config.output_gaussians,
    )

    conf_levels = [
        ConfLevel(threshold=1.5, probability=1.0),
        ConfLevel(threshold=1.25, probability=0.75),
        ConfLevel(threshold=1.1, probability=0.5),
        ConfLevel(threshold=1.005, probability=0.25),
        ConfLevel(threshold=1.001, probability=0.1),
        ConfLevel(threshold=1.000, probability=0.05),
    ]
    all_points_np, all_colors_np, all_normals_tensor = (
        filter_by_conf_project_and_estimate_normals(
            prediction.depth,
            prediction.intrinsics,
            prediction.extrinsics,  # w2c
            prediction.processed_images,
            prediction.conf,
            conf_levels,
            device=device,
        )
    )

    if config.floater_removal:
        all_points_tensor = torch.from_numpy(all_points_np).float().to(device)
        non_floaters = floater_removal(
            intrinsics=torch.from_numpy(prediction.intrinsics).to(device),
            extrinsics=torch.from_numpy(prediction.extrinsics).to(device),
            depth_maps=torch.from_numpy(prediction.depth).to(device),
            all_points_tensor=all_points_tensor,
            all_normals_tensor=all_normals_tensor,
            device=device,
        )
        all_points_np = all_points_np[non_floaters]
        all_colors_np = all_colors_np[non_floaters]

    return (all_points_np, all_colors_np, prediction)
