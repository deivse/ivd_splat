import logging

import numpy as np
import torch
import tyro
from eval_scripts.common.dataset_scenes import (
    scene_id_to_nerfbaselines_data_value,
)
from nerfbaselines._types import Dataset
from nerfbaselines.datasets import dataset_index_select, load_dataset

from edgs.config import EDGSConfig
from edgs.init import edgs_init
from edgs.proxy_dataset import (
    GAUSSIANS_FILE_NAME,
    NB_META_FILE_NAME,
    write_proxy_dataset_to_disk,
)
from edgs.sh_utils import SH2RGB
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

    config = tyro.cli(tyro.conf.FlagConversionOff[EDGSConfig])
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
        ivd_splat_init_type="splat",
        required_files=[GAUSSIANS_FILE_NAME, NB_META_FILE_NAME],
    )

    sfm_pts = dataset["points3D_xyz"]
    sfm_rgbs = dataset["points3D_rgb"]
    export_pointcloud_ply(
        sfm_pts,
        sfm_rgbs,
        path=config.output_dir / "sfm_points3D.ply",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Extracting point cloud")

    if (
        config.max_images_to_load is not None
        and len(dataset["images"]) > config.max_images_to_load
    ):
        logging.info(
            f"Dataset has {len(dataset['images'])} images, but max_images_to_load is set to {config.max_images_to_load}. Selecting a subset of images to load."
        )
        indices = np.random.permutation(len(dataset["images"]))[
            : config.max_images_to_load
        ]
        dataset = dataset_index_select(dataset, indices)

    splats = edgs_init(config, dataset, device, mlflow_run)

    if config.output_pointcloud:
        points = splats.means
        rgbs = SH2RGB(splats.sh0).squeeze()
        export_pointcloud_ply(
            points,
            rgbs,
            path=config.output_dir / "edgs_points3D.ply",
        )

    write_proxy_dataset_to_disk(
        original_dataset_str=nerfbaselines_dataset_string,
        dataset=dataset,
        splat_data=splats,
        path=config.output_dir,
    )


if __name__ == "__main__":
    main()
