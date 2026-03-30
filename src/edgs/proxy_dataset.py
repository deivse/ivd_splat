from pathlib import Path
import json

from nerfbaselines._types import Dataset, DatasetFeature
from nerfbaselines.datasets import load_dataset

from shared.splat_ply_io import SplatData, export_splat_ply

PROXY_DATASET_ID = "edgs"
GAUSSIANS_FILE_NAME = "edgs_splats.ply"
NB_META_FILE_NAME = "nb-info.json"

def write_proxy_dataset_to_disk(
    original_dataset_str: str,
    dataset: Dataset,
    splat_data: SplatData,
    path: Path,
) -> None:
    """
    Create a directory with our own special proxy dataset which contains the modified point and a reference to the original dataset.

    Args:
        original_dataset_str: The original nerfbaselines dataset id string, e.g. "external://mipnerf360/garden".
        splat_data: SplatData object containing the modified point cloud data.
        path: Directory path where the proxy dataset will be saved.
    """

    if "id" in dataset["metadata"]:
        id = dataset["metadata"]["id"]
    else:
        # If id is not in dataset metadata, assume it's a local path and use the parent directory name as id
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

    path.mkdir(parents=True, exist_ok=True)
    with (path / NB_META_FILE_NAME).open("w") as f:
        json.dump(nb_info, f)

    export_splat_ply(path / GAUSSIANS_FILE_NAME, splat_data)


def edgs_proxy_dataset_loader(
    path: str | Path, split: str, features: frozenset[DatasetFeature], **kwargs
) -> Dataset:
    """
    Loader for our proxy dataset with a modified point cloud from monocular depth fusion.
    """

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

    dataset["initialization_splat_path"] = str(path / GAUSSIANS_FILE_NAME)
    return dataset
