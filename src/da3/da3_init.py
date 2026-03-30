from __future__ import annotations

from dataclasses import dataclass
import logging
import os

from torch.types import Device as TorchDevice

from depth_anything_3 import Prediction
import numpy as np
import torch
import typer

from nerfbaselines import Dataset, camera_model_from_int
from depth_anything_3.services.inference_service import InferenceService
from depth_anything_3.utils.export.glb import _depths_to_world_points_with_colors
from da3.config import DA3Config
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
        }

        prediction = model.inference(**inference_kwargs)

        # assert prediction.intrinsics is not None and np.allclose(
        #     prediction.intrinsics, data.intrinsics
        # )
        # assert prediction.extrinsics is not None and np.allclose(
        #     prediction.extrinsics, data.extrinsics
        # )

        return prediction


def da3_init(
    dataset: Dataset,
    config: DA3Config,
    device: TorchDevice = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
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
    )

    conf_thresh = np.percentile(prediction.conf, config.conf_thresh_percentile)
    return _depths_to_world_points_with_colors(
        prediction.depth,
        prediction.intrinsics,
        prediction.extrinsics,  # w2c
        prediction.processed_images,
        prediction.conf,
        conf_thresh,
    )
