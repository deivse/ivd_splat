from dataclasses import dataclass
import logging

import typing
import numpy as np
import torch
from ivd_splat.config import Config
from ivd_splat.datasets.colmap import Parser
from ivd_splat.datasets.normalize import transform_points
from ivd_splat.nerfbaselines_integration.parser import NerfbaselinesParser
from ivd_splat.utils.large_tensor_quantile import large_tensor_quantile
from ivd_splat.utils.runner_utils import knn, rgb_to_sh

from shared.point_cloud_io import load_pointcloud_ply
from shared.splat_ply_io import SplatData, load_splat_ply

from e3nn import o3

_LOGGER = logging.getLogger(__name__)


def decompose_rotation_translation_and_uniform_scale(
    similarity_transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    assert similarity_transform.shape == (4, 4)
    rotation_matrix = similarity_transform[:3, :3]
    translation = similarity_transform[:3, 3]

    # Extract uniform scale from the rotation matrix
    # For an NxN matrix A and a scalar k, the determinant of the scalar-multiplied matrix kA
    # is det(kA) = k^N * det(A).
    scale = np.cbrt(np.linalg.det(rotation_matrix))
    rotation_matrix = rotation_matrix / scale

    return rotation_matrix, translation, scale


def default_init_shN(
    num_splats: int, sh_degree: int, device: torch.device
) -> torch.Tensor:
    # Initialize SH coefficients to zero (except for the constant term)
    shN = torch.zeros((num_splats, (sh_degree + 1) ** 2 - 1, 3), device=device)
    return shN


def default_init_opacities(
    num_splats: int, device: torch.device, config: Config
) -> torch.Tensor:
    return torch.logit(torch.full((num_splats,), config.init.opacity, device=device))


def default_init_scales(
    means: torch.Tensor, scene_scale: float, config: Config
) -> torch.Tensor:
    dist_avg = (knn(means, 3)[0]).mean(dim=-1)  # [N,]
    if config.init.target_median_scale is not None:
        logging.info(
            f"Scaling scales such that median scale is {config.init.target_median_scale}."
        )
        median_dist = torch.median(dist_avg)
        logging.info(
            f"Stats before: median: {median_dist.item()}, mean: {dist_avg.mean().item()}, min: {dist_avg.min().item()}, max: {dist_avg.max().item()}"
        )
        dist_avg = dist_avg * (config.init.target_median_scale / median_dist)
        logging.info("Multiplier: %f", config.init.target_median_scale / median_dist)
        logging.info(
            f"Stats after: median: {dist_avg.median().item()}, mean: {dist_avg.mean().item()}, min: {dist_avg.min().item()}, max: {dist_avg.max().item()}"
        )

    scales = (dist_avg * config.init.scale_mult).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    if config.init.clamp_scales:
        scales = torch.clamp(scales, max=scene_scale / 100)
    scales = torch.log(scales)
    return scales


class InitResult(typing.NamedTuple):
    points: torch.Tensor
    rgbs: torch.Tensor
    scales: torch.Tensor
    quats: torch.Tensor

    def to_splat_data(self, config: Config) -> SplatData:
        sh_degree = config.sh_degree
        sh0 = rgb_to_sh(self.rgbs).unsqueeze(1)  # [N, 1, 3]
        shN = default_init_shN(
            self.points.shape[0], sh_degree, self.points.device
        )  # [N, K, 3]

        return SplatData(
            means=self.points,
            scales=self.scales,
            quats=self.quats,
            opacities=default_init_opacities(
                self.points.shape[0], self.points.device, config
            ),
            sh0=sh0,
            shN=shN,
        )


@dataclass
class RawInitData:
    points: torch.Tensor
    rgbs: torch.Tensor
    sparse_points: torch.Tensor | None = None
    sparse_rgbs: torch.Tensor | None = None


def get_point_data_from_parser(
    config: Config,
    parser: Parser | NerfbaselinesParser,
) -> RawInitData:
    if config.init_type not in ("sparse", "dense"):
        raise ValueError(
            f"Unsupported init_type {config.init_type} for get_point_data_from_parser."
        )

    _LOGGER.info("using %s points from parser", config.init_type)
    if parser.points_rgb is None:
        raise RuntimeError("Parser does not provide point colors for initialization.")

    if config.init_type == "dense":
        if not isinstance(parser, NerfbaselinesParser):
            _LOGGER.warning(
                "Dense initialization expects a NerfbaselinesParser. Initialization will proceed, but double check that everything is correct. Number of points: %d.",
                parser.points.shape[0],
            )
        elif "dense_points3D_path" in parser.nerfbaselines_dataset["metadata"]:
            dense_points_path = parser.nerfbaselines_dataset["metadata"][
                "dense_points3D_path"
            ]
            _LOGGER.info(
                "Loading dense initialization points from path specified in Nerfbaselines dataset metadata: %s",
                dense_points_path,
            )
            points, rgbs = load_pointcloud_ply(dense_points_path)
            points = transform_points(parser.transform, points)

            raw_init_data = RawInitData(
                points=torch.from_numpy(points).float(),
                rgbs=torch.from_numpy(rgbs).float(),
            )
            if config.dense_init.include_sparse:
                _LOGGER.info(
                    "Including sparse points from parser in addition to dense points for initialization."
                )
                raw_init_data.sparse_points = torch.from_numpy(parser.points).float()
                raw_init_data.sparse_rgbs = torch.from_numpy(
                    parser.points_rgb / 255.0
                ).float()

            return raw_init_data
        elif not parser.nerfbaselines_dataset["metadata"].get(
            "ivd_splat_dense_init", False
        ):
            _LOGGER.warning(
                "Nerfbaselines dataset does not indicate that the initialization data is dense. Initialization will proceed, but double check that everything is correct. Number of points: %d.",
                (
                    parser.points.shape[0]
                    if parser.points is not None
                    else "<Error: parser.points is None>"
                ),
            )
        elif config.dense_init.include_sparse:
            raise RuntimeError(
                "Config is set to include sparse points for dense initialization, but the Nerfbaselines dataset provides dense points by overriding the dataset's original point fields, so the sparse data is not available. Please check your configuration and dataset."
            )

    elif (
        config.init_type == "sparse"
        and isinstance(parser, NerfbaselinesParser)
        and parser.nerfbaselines_dataset["metadata"].get("ivd_splat_dense_init", False)
    ):
        raise RuntimeError(
            "Parser indicates that the initialization data is dense, but config.init_type is set to sparse. Please check your configuration and dataset."
        )

    return RawInitData(
        points=torch.from_numpy(parser.points).float(),
        rgbs=torch.from_numpy(parser.points_rgb / 255.0).float(),
    )


def _pick_dense_init_points(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    num_sparse_pts: int,
    config: Config,
) -> torch.Tensor:
    """
    Select a subset of points for dense initialization.
    Args:
        points: (N, 3) tensor of point positions.
        rgbs: (N, 3) tensor of point colors.
        config: Configuration object with dense_init parameters.
        scene_scale: The extent of the scene, as defined by the cameras.
    Returns:
        Indices of selected points.
    """
    assert config.init_type == "dense"

    target_num_pts = config.dense_init.target_num_points or points.shape[0]
    if config.dense_init.target_points_fraction is not None:
        _LOGGER.info(
            f"Selecting {config.dense_init.target_points_fraction} * {target_num_pts} points for dense initialization."
        )
        target_num_pts = int(target_num_pts * config.dense_init.target_points_fraction)

    target_num_pts -= num_sparse_pts

    if target_num_pts == points.shape[0]:
        _LOGGER.info("Using all points for dense initialization.")
        return torch.arange(points.shape[0])

    if target_num_pts >= points.shape[0]:
        raise RuntimeError(
            "Cannot pick more points than available in dense point cloud."
        )

    _LOGGER.info(
        f"Dense initialization will use {target_num_pts}/{points.shape[0]} points."
    )
    if config.dense_init.sampling == "uniform":
        _LOGGER.info("Dense initialization using uniform sampling.")
        indices = torch.randperm(points.shape[0])[:target_num_pts]
        return indices

    _LOGGER.info(
        "picking %s dense init points with adaptive sampling",
        target_num_pts,
    )

    indices = torch.arange(points.shape[0])
    torch_multinomial_max_input_size = 2**24
    if target_num_pts > torch_multinomial_max_input_size:
        raise RuntimeError(
            f"Adaptive sampling currently supports up to {torch_multinomial_max_input_size} points."
        )

    initial_subsample_target_mult = 200
    if (
        points.shape[0] > initial_subsample_target_mult * target_num_pts
        or points.shape[0] > torch_multinomial_max_input_size
    ):
        initial_sample_num_pts = min(
            initial_subsample_target_mult * target_num_pts,
            torch_multinomial_max_input_size,
        )
        _LOGGER.info(
            f"Performing initial uniform subsample from {points.shape[0]} to {initial_sample_num_pts} points."
        )
        indices = torch.randperm(points.shape[0])[:initial_sample_num_pts]
        points = points[indices]
        rgbs = rgbs[indices]

    _LOGGER.info(
        "Adaptive sampling using KNN and color-based probabilities on %d points.",
        points.shape[0],
    )

    knn_dists, knn_indices = knn(
        points, K=config.dense_init.knn_num_neighbors
    )  # [N, K]
    color_dists_squared = ((rgbs.unsqueeze(1) - rgbs[knn_indices]) ** 2).sum(
        dim=-1
    )  # [N, K]
    # Reduce sensitivity to color noise
    color_dists_squared = torch.clamp(
        color_dists_squared, min=config.dense_init.color_dist_thresh**2
    )

    q75 = large_tensor_quantile(knn_dists.view(-1), 0.75)

    clamped_knn_dists = torch.clamp(knn_dists, max=q75)

    prob = torch.softmax(
        (color_dists_squared * clamped_knn_dists).mean(dim=-1).to(torch.float64)
        / config.dense_init.softmax_temp,
        dim=0,
    )

    # # debug export pointcloud colored by probability
    # from shared.point_cloud_io import export_pointcloud_ply

    # print(prob.min(), prob.max(), (prob != prob).max())

    # prob_vis = prob.view([-1, 1]).repeat([1, 3]).cpu().numpy()
    # prob_vis = (prob_vis - prob_vis.min()) / (prob_vis.max() - prob_vis.min())
    # export_pointcloud_ply(
    #     points.cpu().numpy(),
    #     prob_vis,
    #     "adaptive_sampling_debug.ply",
    # )

    adaptive_indices = torch.multinomial(prob, target_num_pts, replacement=False)
    return indices[adaptive_indices]


def _get_floater_mask(points: torch.Tensor, config: Config) -> torch.Tensor:
    _LOGGER.info("Removing floaters from point cloud.")
    dist2_avg = (knn(points, 3)[0] ** 2).mean(dim=-1)  # [N,]

    threshold = torch.quantile(dist2_avg, config.init.floater_knn_distance_percentile)
    mask = dist2_avg <= threshold
    _LOGGER.info(
        f"Removed {torch.sum(~mask).item()} floaters out of {points.shape[0]} points in point cloud."
    )
    return mask


def _add_noise_to_init_points(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    config: Config,
    scene_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Add noise to initial points and colors.
    """
    if config.init.color_noise_std > 0.0:
        noise = torch.randn_like(rgbs) * config.init.color_noise_std
        rgbs = torch.clamp(rgbs + noise, 0.0, 1.0)

    if config.init.position_noise_std > 0.0:
        noise = torch.randn_like(points) * scene_scale * config.init.position_noise_std
        points = points + noise

    return points, rgbs


def point_cloud_init(
    raw_init_data: RawInitData, config: Config, scene_scale: float
) -> SplatData:
    """
    Create splats from point cloud as in base 3DGS.
    """
    points = raw_init_data.points
    rgbs = raw_init_data.rgbs
    sparse_points = raw_init_data.sparse_points
    sparse_rgbs = raw_init_data.sparse_rgbs
    _LOGGER.info(
        "initializing gaussians from point cloud with %d points", points.shape[0]
    )

    if points.shape[0] != rgbs.shape[0]:
        raise RuntimeError("Number of points and rgbs must be identical.")
    if (sparse_points is not None) != (sparse_rgbs is not None):
        raise RuntimeError(
            "sparse_points and sparse_rgbs must both be provided or both be None."
        )

    num_sparse_pts = 0
    if sparse_points is not None and sparse_rgbs is not None:
        if sparse_points.shape[0] != sparse_rgbs.shape[0]:
            raise RuntimeError(
                "Number of sparse points and sparse rgbs must be identical."
            )
        _LOGGER.info(
            "including %d sparse points from parser in addition to dense points for initialization.",
            sparse_points.shape[0],
        )
        num_sparse_pts = sparse_points.shape[0]

    if config.init_type == "dense":
        point_indices = _pick_dense_init_points(points, rgbs, num_sparse_pts, config)
        points = points[point_indices]
        rgbs = rgbs[point_indices]

    if sparse_points is not None and sparse_rgbs is not None:
        points = torch.cat([points, sparse_points], dim=0)
        rgbs = torch.cat([rgbs, sparse_rgbs], dim=0)

    points, rgbs = _add_noise_to_init_points(points, rgbs, config, scene_scale)

    if config.init.remove_floaters:
        mask = _get_floater_mask(points, config)
        points = points[mask]
        rgbs = rgbs[mask]

    N = points.shape[0]

    scales = default_init_scales(points, scene_scale, config)  # [N, 3]
    quats = torch.rand((N, 4))  # [N, 4]

    return InitResult(
        points=points, rgbs=rgbs, scales=scales, quats=quats
    ).to_splat_data(config)


def _get_splat_subset_inplace(splat: SplatData, config: Config) -> None:
    if config.dense_init.target_num_points is None:
        _LOGGER.info(
            "Using all pre-made splat points for initialization since target_num_points is None."
        )
        return

    target_num_pts = config.dense_init.target_num_points

    if config.dense_init.target_points_fraction is not None:
        _LOGGER.info(
            f"Selecting {config.dense_init.target_points_fraction} * {target_num_pts} splats for dense initialization."
        )
        target_num_pts = int(target_num_pts * config.dense_init.target_points_fraction)

    num_points = splat.means.shape[0]
    if target_num_pts >= num_points:
        _LOGGER.warning(
            f"Requested {target_num_pts} splats for initialization, but only {num_points} available in pre-made splat. Using all available splats."
        )
        return

    _LOGGER.info(
        f"Selecting {target_num_pts} of pre-made splat points for initialization."
    )

    splat.select_random_subset_inplace(target_num_pts)

    splat_fraction = target_num_pts / num_points
    if config.splat_init.increase_scale_with_fewer_splats:
        _LOGGER.info(
            f"increasing scale of pre-made splats by {1/splat_fraction} to compensate for fewer splats."
        )
        splat.scales = np.log(np.exp(splat.scales) * (1 / splat_fraction))


def transform_shs(shs_feat, rotation_matrix, beta_coef=-1.0):
    # Rotate SH values
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float()
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)

    # Construct rotation matrices for SH orders
    D_1 = o3.wigner_D(1, rot_angles[0], beta_coef * rot_angles[1], rot_angles[2])
    D_2 = o3.wigner_D(2, rot_angles[0], beta_coef * rot_angles[1], rot_angles[2])
    D_3 = o3.wigner_D(3, rot_angles[0], beta_coef * rot_angles[1], rot_angles[2])

    # Apply rotation to SH features
    return torch.cat(
        (D_1 @ shs_feat[:, :3], D_2 @ shs_feat[:, 3:8], D_3 @ shs_feat[:, 8:15]), dim=1
    )


def rotate_quaternions(quats, rotation_matrix):
    rot_quat = o3.matrix_to_quaternion(rotation_matrix)
    rot_quat = rot_quat / torch.norm(rot_quat)
    return o3.compose_quaternion(rot_quat, quats)


def load_splat_from_nerfbaselines_parser(config: Config, parser: Parser) -> SplatData:
    if not isinstance(parser, NerfbaselinesParser):
        raise RuntimeError(
            "Init with pre-made splat currently requires NerfbaselinesParser."
        )

    nb_metadata = parser.nerfbaselines_dataset["metadata"]
    if "ivd_splat_splat_init_path" not in nb_metadata:
        raise RuntimeError(
            "Nerfbaselines dataset does not contain initialization splat path."
        )

    splat_path = nb_metadata["ivd_splat_splat_init_path"]
    splat = load_splat_ply(splat_path)
    # Also increases scales if config.splat_init.increase_scale_with_fewer_splats is True
    _get_splat_subset_inplace(splat, config)

    rotation, _, scale = decompose_rotation_translation_and_uniform_scale(
        parser.transform
    )
    splat.means = transform_points(parser.transform, splat.means).to(torch.float32)
    splat.scales = torch.log(torch.exp(splat.scales) * scale).to(torch.float32)

    rot_torch = torch.from_numpy(rotation).float()

    splat.shN = transform_shs(splat.shN, rot_torch)
    splat.quats = rotate_quaternions(splat.quats, rot_torch)

    return splat
