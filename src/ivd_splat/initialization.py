import logging

import typing
import numpy as np
import torch
from ivd_splat.config import Config
from ivd_splat.datasets.colmap import Parser
from ivd_splat.datasets.normalize import transform_normals, transform_points
from ivd_splat.nerfbaselines_integration.parser import NerfbaselinesParser
from ivd_splat.utils.runner_utils import knn, rgb_to_sh

from shared.point_cloud_io import load_pointcloud_ply, load_normals
from shared.splat_ply_io import SplatData, load_splat_ply

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


def get_point_data_from_parser(
    config: Config,
    parser: Parser | NerfbaselinesParser,
) -> tuple[torch.Tensor, torch.Tensor]:
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
            return torch.from_numpy(points).float(), torch.from_numpy(rgbs).float()
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
    elif (
        config.init_type == "sparse"
        and isinstance(parser, NerfbaselinesParser)
        and parser.nerfbaselines_dataset["metadata"].get("ivd_splat_dense_init", False)
    ):
        raise RuntimeError(
            "Parser indicates that the initialization data is dense, but config.init_type is set to sparse. Please check your configuration and dataset."
        )

    return (
        torch.from_numpy(parser.points).float(),
        torch.from_numpy(parser.points_rgb / 255.0).float(),
    )


def _pick_dense_init_points(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """
    Select a subset of points for dense initialization.
    Args:
        points: (N, 3) tensor of point positions.
        rgbs: (N, 3) tensor of point colors.
        config: Configuration object with dense_init parameters.
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

    while points.shape[0] > 100 * target_num_pts or (
        points.shape[0] > torch_multinomial_max_input_size
    ):
        _LOGGER.info(
            f"Downsampling point cloud from {points.shape[0]} to {points.shape[0] // 2} points with random sampling."
        )
        perm = torch.randperm(points.shape[0])[: points.shape[0] // 2]
        points = points[perm]
        rgbs = rgbs[perm]
        indices = indices[perm]

    _LOGGER.info(
        "Adaptive sampling using KNN and color-based probabilities on %d points.",
        points.shape[0],
    )

    _, knn_indices = knn(points, K=config.dense_init.knn_num_neighbors)  # [N, K + 1]
    avg_color_dist2 = (
        ((rgbs.unsqueeze(1) - rgbs[knn_indices]) ** 2).sum(dim=-1).mean(dim=1)
    )  # [N,]

    prob = avg_color_dist2 / avg_color_dist2.max()
    prob = prob / prob.sum()

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
    points: torch.Tensor, rgbs: torch.Tensor, config: Config, scene_scale: float
) -> SplatData:
    """
    Create splats from point cloud as in base 3DGS.
    """
    _LOGGER.info(
        "initializing gaussians from point cloud with %d points", points.shape[0]
    )

    if points.shape[0] != rgbs.shape[0]:
        raise RuntimeError("Number of points and rgbs must be identical.")

    if config.init_type == "dense":
        point_indices = _pick_dense_init_points(points, rgbs, config)
        points = points[point_indices]
        rgbs = rgbs[point_indices]

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

    # TODO: this is only fine as long as the init method outputs isotropic covariances.
    # If we want to support anisotropic covariances we need to apply rotation too
    # splat.quats = rotate_quaternions(splat.quats, rotation)
    scales_are_isotropic = torch.allclose(
        splat.scales[:, 0], splat.scales[:, 1]
    ) and torch.allclose(splat.scales[:, 1], splat.scales[:, 2])
    if not scales_are_isotropic:
        raise NotImplementedError(
            "Transforming initial splats with anisotropic scales is not implemented!"
        )

    return splat
