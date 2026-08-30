from pathlib import Path
from typing import FrozenSet, Optional, Union

from nerfbaselines import Dataset, UnloadedDataset, DatasetFeature
from nerfbaselines.datasets.colmap import load_colmap_dataset, dataset_index_select


def load_and_sparsify_colmap_dataset(
    path: Union[Path, str],
    split: str,
    features: Optional[FrozenSet[DatasetFeature]] = None,
    **kwargs,
):
    kwargs.pop("split", None)
    kwargs.pop("features", None)
    kwargs.pop("path", None)
    kwargs.pop("test_indices", None)

    source_path = kwargs.pop("source_path", None)
    if source_path is None:
        raise ValueError("source_path must be provided in kwargs")
    
    full_dataset: UnloadedDataset | Dataset = load_colmap_dataset(
        path=source_path,
        split=None,
        features=features,
        **kwargs,
    )
    if split == "train":
        indices = range(0, len(full_dataset["cameras"]), 8)
    elif split == "test":
        indices = range(3, len(full_dataset["cameras"]), 8)
    else:
        raise ValueError(f"Unknown split: {split}")
    return dataset_index_select(full_dataset, list(indices))

def download_not_implemented(*args, **kwargs):
    raise NotImplementedError("This dataset does not have a download function implemented.")
