from pathlib import Path
import numpy as np
import json

from nerfbaselines._types import Dataset, DatasetFeature
from nerfbaselines.datasets import load_dataset

from shared.point_cloud_io import (
    export_pointcloud_ply,
    load_pointcloud_ply,
)

PROXY_DATASET_ID = "monodepth"
POINTS_FILE_NAME = "points.ply"
NB_META_FILE_NAME = "nb-info.json"


def write_proxy_dataset_to_disk(
    original_dataset_str: str,
    dataset: Dataset,
    points: np.ndarray,
    rgbs: np.ndarray,
    path: Path,
) -> None:
    """
    Create a directory with our own special proxy dataset which contains the modified point and a reference to the original dataset.

    Args:
        points: (N, 3) array of 3D points.
        rgbs: (N, 3) array of RGB colors.
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

    export_pointcloud_ply(points, rgbs, path / POINTS_FILE_NAME)


def monodepth_proxy_dataset_loader(
    path: str | Path, split: str, features: frozenset[DatasetFeature], **kwargs
) -> Dataset:
    """
    Loader for our proxy dataset with a modified point cloud from monocular depth fusion.
    """
    if "points3D_xyz" not in features:
        raise RuntimeError(
            "Using monodepth proxy dataset without loading points3D_xyz is redundant."
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

    pts, rgbs = load_pointcloud_ply(path / POINTS_FILE_NAME)
    if rgbs is None:
        raise RuntimeError("Proxy dataset pointcloud does not contain colors.")
    dataset["points3D_xyz"] = pts
    dataset["points3D_rgb"] = rgbs * 255.0  # convert back to [0,255] range for consistency with other datasets

    return dataset
