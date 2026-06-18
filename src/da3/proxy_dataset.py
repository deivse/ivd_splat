import os
from pathlib import Path
from typing import Optional
import numpy as np
import json

from nerfbaselines._types import Dataset, DatasetFeature
from nerfbaselines.datasets import load_dataset
from depth_anything_3.specs import Prediction
from depth_anything_3.utils.gsply_helpers import (
    save_gaussian_ply as da3_save_gaussian_ply,
)
import torch

from shared.point_cloud_io import (
    export_pointcloud_ply,
    load_pointcloud_ply,
)
from shared.splat_ply_io import SplatData, export_splat_ply

PROXY_DATASET_ID = "da3"
POINTS_FILE_NAME = "da3_points.ply"
SPLATS_FILE_NAME = "da3_gaussians.ply"
NB_META_FILE_NAME = "nb-info.json"


def da3_export_to_gs_ply(
    prediction: Prediction,
    path: Path,
    gs_views_interval: Optional[
        int
    ] = 1,  # export GS every N views, useful for extremely dense inputs
):
    gs_world = prediction.gaussians
    pred_depth = (
        torch.from_numpy(prediction.depth).unsqueeze(-1).to(gs_world.means)
    )  # v h w 1
    if gs_views_interval is None:  # select around 12 views in total
        gs_views_interval = max(pred_depth.shape[0] // 12, 1)
    da3_save_gaussian_ply(
        gaussians=gs_world,
        save_path=str(path),
        ctx_depth=pred_depth,
        shift_and_scale=False,
        save_sh_dc_only=True,
        gs_views_interval=gs_views_interval,
        inv_opacity=True,
        prune_by_depth_percent=0.9,
        prune_border_gs=True,
        match_3dgs_mcmc_dev=False,
    )


def write_proxy_dataset_to_disk(
    original_dataset_str: str,
    dataset: Dataset,
    points: np.ndarray,
    rgbs: np.ndarray,
    prediction: Prediction,
    path: Path,
) -> None:
    """
    Create a directory with our own special proxy dataset which contains the modified point and a reference to the original dataset.

    Args:
        points: (N, 3) array of 3D points.
        rgbs: (N, 3) array of RGB colors.
        gaussians: Prediction object containing DA3 gaussians.
        path: Directory path where the proxy dataset will be saved.
    """

    if "id" in dataset["metadata"]:
        id = dataset["metadata"]["id"]
    else:
        id = Path(original_dataset_str).parent.name
    if "scene" in dataset["metadata"]:
        scene = dataset["metadata"]["scene"]
    else:
        scene = Path(original_dataset_str).stem

    nb_info = {
        "loader": PROXY_DATASET_ID,
        "id": id,
        "scene": scene,
        "original_dataset": original_dataset_str,
        "ivd_splat_dense_init": True,
    }

    path.mkdir(parents=True, exist_ok=True)
    with (path / NB_META_FILE_NAME).open("w") as f:
        json.dump(nb_info, f)

    if prediction is not None and prediction.gaussians is not None:
        da3_export_to_gs_ply(prediction, path / SPLATS_FILE_NAME)
    else:
        export_pointcloud_ply(points, rgbs, path / POINTS_FILE_NAME)


def da3_proxy_dataset_loader(
    path: str | Path, split: str, features: frozenset[DatasetFeature], **kwargs
) -> Dataset:
    """
    Loader for our proxy dataset with a modified point cloud from DA3.
    """
    if "points3D_xyz" not in features:
        raise RuntimeError(
            "Using DA3 proxy dataset without loading points3D_xyz is redundant."
        )

    path = Path(path)
    with (path / NB_META_FILE_NAME).open("r") as f:
        nb_info = json.load(f)
    original_dataset_str = nb_info["original_dataset"]
    dataset = load_dataset(
        original_dataset_str,
        split=split,
        features=features,
        **kwargs,
    )

    at_least_one = False
    if (path / POINTS_FILE_NAME).exists():
        # Or replace SfM points for compatibility with existing methods
        pts, rgbs = load_pointcloud_ply(path / POINTS_FILE_NAME)
        if rgbs is None:
            raise RuntimeError("Proxy dataset pointcloud does not contain colors.")
        dataset["points3D_xyz"] = pts
        dataset["points3D_rgb"] = rgbs * 255.0
        at_least_one = True
    if (path / SPLATS_FILE_NAME).exists():
        dataset["metadata"]["ivd_splat_splat_init_path"] = str(path / SPLATS_FILE_NAME)
        at_least_one = True

    if not at_least_one:
        raise RuntimeError(
            f"Proxy dataset at {path} does not contain any of the expected files: {POINTS_FILE_NAME}, {SPLATS_FILE_NAME}"
        )

    return dataset
