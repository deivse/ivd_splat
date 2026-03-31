from datetime import datetime
import logging

import mlflow
import numpy as np
import torch
import tyro
from eval_scripts.common.dataset_scenes import (
    scene_id_to_nerfbaselines_data_value,
)
from nerfbaselines._types import Dataset
from nerfbaselines.datasets import dataset_index_select, load_dataset

from da3.da3_init import da3_init
from da3.config import DA3Config
from da3.proxy_dataset import (
    NB_META_FILE_NAME,
    POINTS_FILE_NAME,
    write_proxy_dataset_to_disk,
)
from shared.point_cloud_io import export_pointcloud_ply
from shared.save_init_info import save_init_info_json
from shared.serializable_config import mlflow_log_config_params


@torch.no_grad()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = tyro.cli(tyro.conf.FlagConversionOff[DA3Config])
    mlflow_run = mlflow_log_config_params(config)

    # Load train dataset
    logging.info("Loading dataset")
    nerfbaselines_dataset_string = scene_id_to_nerfbaselines_data_value(config.scene)
    dataset: Dataset = load_dataset(
        nerfbaselines_dataset_string,
        split="train",
        features=[
            "color",
            "points3D_xyz",
            "points3D_rgb",
        ],
        supported_camera_models=frozenset(["pinhole"]),
        load_features=True,
    )
    logging.info("Dataset loaded")
    logging.info(f"Number of training images: {len(dataset['images'])}")
    logging.info(f"Number of 3D points: {dataset['points3D_xyz'].shape[0]}")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    save_init_info_json(
        config.output_dir,
        ivd_splat_init_type="dense",
        required_files=[POINTS_FILE_NAME, NB_META_FILE_NAME],
    )

    sfm_pts = dataset["points3D_xyz"]
    sfm_rgbs = dataset["points3D_rgb"]
    export_pointcloud_ply(
        sfm_pts,
        sfm_rgbs,
        path=config.output_dir / "sfm_points3D.ply",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = datetime.now()
    points, rgbs = da3_init(dataset, config, device)
    end_time = datetime.now()
    if mlflow_run is not None:
        mlflow.log_metric("init_only_runtime", (end_time - start_time).total_seconds())

    write_proxy_dataset_to_disk(
        scene_id_to_nerfbaselines_data_value(config.scene),
        dataset,
        points,
        rgbs,
        path=config.output_dir,
    )


if __name__ == "__main__":
    main()
