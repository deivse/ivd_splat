import os
from pathlib import Path
from typing import Optional
import numpy as np
import json

from nerfbaselines._types import Dataset, DatasetFeature
from nerfbaselines.datasets import load_dataset
from depth_anything_3.specs import Prediction
from depth_anything_3.specs import Gaussians as DA3Gaussians
from depth_anything_3.utils.gsply_helpers import export_ply as da3_export_ply
from einops import rearrange
import torch

from da3.config import DA3Config
from shared.point_cloud_io import (
    export_pointcloud_ply,
    load_pointcloud_ply,
)

PROXY_DATASET_ID = "da3"
POINTS_FILE_NAME = "da3_points.ply"
SPLATS_FILE_NAME = "da3_gaussians.ply"
NB_META_FILE_NAME = "nb-info.json"


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x / (1 - x))


def da3_save_gaussian_ply(
    gaussians: DA3Gaussians,
    save_path: str,
    ctx_depth: torch.Tensor,  # depth of input views; for getting shape and filtering, "v h w 1"
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = True,
    gs_views_interval: int = 1,
    inv_opacity: Optional[bool] = True,
    prune_by_depth_percent: Optional[float] = 1.0,
    prune_border_gs: Optional[bool] = True,
    match_3dgs_mcmc_dev: Optional[bool] = False,
    min_opacity: Optional[float] = None,
):
    """Copied from DA3 code, slightly adjusted to allow for filtering by min_opacity."""

    b = gaussians.means.shape[0]
    assert b == 1, "must set batch_size=1 when exporting 3D gaussians"
    src_v, out_h, out_w, _ = ctx_depth.shape

    # extract gs params
    world_means = gaussians.means
    world_shs = gaussians.harmonics
    world_rotations = gaussians.rotations
    gs_scales = gaussians.scales
    gs_opacities = (
        inverse_sigmoid(gaussians.opacities) if inv_opacity else gaussians.opacities
    )

    # Create a mask to filter the Gaussians.

    # throw away Gaussians at the borders, since they're generally of lower quality.
    if prune_border_gs:
        mask = torch.zeros_like(ctx_depth, dtype=torch.bool)
        gstrim_h = int(8 / 256 * out_h)
        gstrim_w = int(8 / 256 * out_w)
        mask[:, gstrim_h:-gstrim_h, gstrim_w:-gstrim_w, :] = 1
    else:
        mask = torch.ones_like(ctx_depth, dtype=torch.bool)

    # trim the far away point based on depth;
    if prune_by_depth_percent is not None and prune_by_depth_percent < 1:
        in_depths = ctx_depth
        d_percentile = torch.quantile(
            in_depths.view(in_depths.shape[0], -1), q=prune_by_depth_percent, dim=1
        ).view(-1, 1, 1)
        d_mask = (in_depths[..., 0] <= d_percentile).unsqueeze(-1)
        mask = mask & d_mask

    mask = mask.squeeze(-1)  # v h w
    if min_opacity is not None:
        tmp_opacities = rearrange(
            gaussians.opacities[0],
            "(v h w) ... -> v h w ...",
            v=src_v,
            h=out_h,
            w=out_w,
        )
        opacity_mask = tmp_opacities > min_opacity
        mask = mask & opacity_mask

    # helper fn, must place after mask
    def trim_select_reshape(element):
        selected_element = rearrange(
            element[0], "(v h w) ... -> v h w ...", v=src_v, h=out_h, w=out_w
        )
        selected_element = selected_element[::gs_views_interval][
            mask[::gs_views_interval]
        ]
        return selected_element

    da3_export_ply(
        means=trim_select_reshape(world_means),
        scales=trim_select_reshape(gs_scales),
        rotations=trim_select_reshape(world_rotations),
        harmonics=trim_select_reshape(world_shs),
        opacities=trim_select_reshape(gs_opacities),
        path=Path(save_path),
        shift_and_scale=shift_and_scale,
        save_sh_dc_only=save_sh_dc_only,
        match_3dgs_mcmc_dev=match_3dgs_mcmc_dev,
    )


def da3_export_to_gs_ply(
    prediction: Prediction,
    path: Path,
    gs_views_interval: Optional[
        int
    ] = 1,  # export GS every N views, useful for extremely dense inputs
    min_opacity: Optional[
        float
    ] = None,  # filter out gaussians with opacity below this threshold
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
        min_opacity=min_opacity,  # TODO: config
    )


def write_proxy_dataset_to_disk(
    original_dataset_str: str,
    dataset: Dataset,
    points: np.ndarray,
    rgbs: np.ndarray,
    prediction: Prediction,
    config: DA3Config,
) -> None:
    """
    Create a directory with our own special proxy dataset which contains the modified point and a reference to the original dataset.

    Args:
        points: (N, 3) array of 3D points.
        rgbs: (N, 3) array of RGB colors.
        gaussians: Prediction object containing DA3 gaussians.
        path: Directory path where the proxy dataset will be saved.
        config: DA3Config object containing configuration parameters.
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
    }

    path = config.output_dir
    path.mkdir(parents=True, exist_ok=True)
    with (path / NB_META_FILE_NAME).open("w") as f:
        json.dump(nb_info, f)

    if prediction is not None and prediction.gaussians is not None:
        da3_export_to_gs_ply(
            prediction, path / SPLATS_FILE_NAME, min_opacity=config.min_gaussian_opacity
        )
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
        dataset["metadata"]["dense_points3D_path"] = str(path / POINTS_FILE_NAME)
        at_least_one = True
    if (path / SPLATS_FILE_NAME).exists():
        dataset["metadata"]["ivd_splat_splat_init_path"] = str(path / SPLATS_FILE_NAME)
        at_least_one = True

    if not at_least_one:
        raise RuntimeError(
            f"Proxy dataset at {path} does not contain any of the expected files: {POINTS_FILE_NAME}, {SPLATS_FILE_NAME}"
        )

    return dataset
