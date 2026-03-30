from datetime import datetime
import logging
from eval_scripts.common.dataset_scenes import (
    scene_id_to_nerfbaselines_data_value,
)
import mlflow
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
import torch
import tyro
from monodepth.config import Config
from nerfbaselines._types import Dataset
from nerfbaselines.datasets import load_dataset

from monodepth.generate_pointcloud import monocular_depth_init
from monodepth.proxy_dataset import write_proxy_dataset_to_disk
from shared.point_cloud_io import export_pointcloud_ply
from shared.serializable_config import mlflow_log_config_params


def sfm_error_percentile_mask(
    sfm_pts: np.ndarray,
    sfm_points_error: np.ndarray,
    curr_mask: np.ndarray,
    config: Config,
) -> np.ndarray:
    assert (
        config.sfm_points_max_err_percentile is not None
    ), "sfm_points_max_err_percentile must be set"
    err_thresh = np.percentile(
        sfm_points_error[curr_mask], config.sfm_points_max_err_percentile
    )
    new_mask_on_curr = sfm_points_error[curr_mask] <= err_thresh
    logging.info(
        f"Filtering out {(~new_mask_on_curr).sum()} SfM points with reprojection error above {config.sfm_points_max_err_percentile:.2f}th percentile (error > {err_thresh:.4f})"
    )

    out_mask = curr_mask.copy()
    out_mask[curr_mask] = new_mask_on_curr

    colors = np.full(
        (sfm_pts.shape[0], 3), fill_value=[0.0, 0.0, 1.0]
    )  # Blue for inliers
    colors[~out_mask] = np.array([1.0, 0.0, 0.0])  # Red for outliers
    export_pointcloud_ply(
        sfm_pts,
        colors,
        path=config.output_dir / "sfm_points_outliers_reproj_error.ply",
    )
    return out_mask


def sfm_lof_outlier_mask(
    sfm_points: np.ndarray, curr_mask: np.ndarray, config: Config
) -> np.ndarray:
    lof = LocalOutlierFactor(n_neighbors=config.sfm_lof_n_neighbors, n_jobs=-1)
    outlier_labels = lof.fit_predict(sfm_points[curr_mask])
    new_mask_on_curr = outlier_labels != -1

    logging.info(
        f"LOF detected {(~new_mask_on_curr).sum()} outliers among {new_mask_on_curr.sum()} SfM points."
    )

    out_mask = curr_mask.copy()
    out_mask[curr_mask] = new_mask_on_curr

    colors = np.full(
        (sfm_points.shape[0], 3), fill_value=[0.0, 0.0, 1.0]
    )  # Blue for inliers
    colors[~out_mask] = np.array([1.0, 0.0, 0.0])  # Red for outliers
    export_pointcloud_ply(
        sfm_points,
        colors,
        path=config.output_dir / "sfm_points_outliers_lof.ply",
    )

    return out_mask


def final_outlier_removal(
    points: np.ndarray, rgbs: np.ndarray, config: Config
) -> tuple[np.ndarray, np.ndarray]:
    if config.final_outlier_removal is None:
        return points, rgbs

    if config.final_outlier_removal == "lof":
        mask = sfm_lof_outlier_mask(
            points, np.ones(points.shape[0], dtype=bool), config
        )
        return points[mask], rgbs[mask]

    if config.final_outlier_removal == "nn":
        # Find nearest neighbor distances between points
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=4, n_jobs=-1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        # distances[:, 0] is the distance to itself (0), so take the rest
        median_nn_distance = np.median(distances[:, 1:], axis=1)
        # Mark points with nearest neighbor distance above a threshold as outliers
        dist_thresh = np.percentile(median_nn_distance, 95)
        mask = median_nn_distance < dist_thresh
        logging.info(
            f"Removing {(~mask).sum()} outliers based on nearest neighbor distance (threshold = {dist_thresh:.4f})"
        )
        return points[mask], rgbs[mask]

    if config.final_outlier_removal == "pcd_stat":
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.5)
        mask = np.zeros(points.shape[0], dtype=bool)
        mask[ind] = True
        logging.info(
            f"Removing {(~mask).sum()} outliers based on Open3D statistical outlier removal"
        )
        return points[mask], rgbs[mask]

    raise ValueError(
        f"Unknown final_outlier_removal method: {config.final_outlier_removal}"
    )


@torch.no_grad()
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = tyro.cli(tyro.conf.FlagConversionOff[Config])

    requested_features = frozenset(
        [
            "color",
            "points3D_xyz",
            "points3D_rgb",
            "images_points3D_indices",
            "images_points2D_xy",
            "points3D_error",
        ]
    )

    # Load train dataset
    logging.info("Loading dataset")
    dataset: Dataset = load_dataset(
        scene_id_to_nerfbaselines_data_value(config.scene),
        split="train",
        features=requested_features,
        supported_camera_models=frozenset(["pinhole"]),
        load_features=True,
    )
    logging.info("Dataset loaded")
    logging.info(f"Number of training images: {len(dataset['images'])}")
    logging.info(f"Number of 3D points: {dataset['points3D_xyz'].shape[0]}")

    config.process()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    mlflow_run = mlflow_log_config_params(config)

    sfm_pts = dataset["points3D_xyz"]
    sfm_rgbs = dataset["points3D_rgb"]
    export_pointcloud_ply(
        sfm_pts,
        sfm_rgbs,
        path=config.output_dir / "sfm_points3D.ply",
    )

    start_time = datetime.now()
    sfm_pts_mask = np.ones(sfm_pts.shape[0], dtype=bool)
    if (
        config.sfm_points_max_err_percentile is not None
        and dataset["points3D_error"] is not None
    ):
        sfm_pts_mask = sfm_error_percentile_mask(
            sfm_pts, dataset["points3D_error"], sfm_pts_mask, config
        )
    if config.sfm_outlier_removal:
        sfm_pts_mask = sfm_lof_outlier_mask(sfm_pts, sfm_pts_mask, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    points, rgbs = monocular_depth_init(dataset, config, sfm_pts_mask, device=device)

    points, rgbs = final_outlier_removal(points, rgbs, config)

    if config.include_sfm_points:
        points = np.concatenate([points, sfm_pts[sfm_pts_mask]], axis=0)
        rgbs = np.concatenate([rgbs, sfm_rgbs[sfm_pts_mask]], axis=0)

    end_time = datetime.now()
    if mlflow_run is not None:
        mlflow.log_metric("init_only_runtime", (end_time - start_time).total_seconds())

    logging.info(f"Number of points in point cloud: {points.shape[0]}")

    write_proxy_dataset_to_disk(
        scene_id_to_nerfbaselines_data_value(config.scene),
        dataset,
        points,
        rgbs,
        path=config.output_dir,
    )


if __name__ == "__main__":
    main()
