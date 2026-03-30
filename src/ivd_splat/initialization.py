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


def rotation_quat_from_normal(normals: torch.Tensor) -> torch.Tensor:
    """Compute rotation quaternions that align the z-axis with the given normal vectors.

    Args:
        normals: A tensor of shape (3,) or (N, 3) representing the normal vector(s).

    Returns:
        A tensor of shape (4,) or (N, 4) representing the rotation quaternion(s).
    """
    if normals.ndim == 1:
        # If input is a single vector (3,), reshape to (1, 3)
        normals = normals.unsqueeze(0)
        is_single = True
    else:
        is_single = False

    device = normals.device
    N = normals.shape[0]

    z_axis = torch.tensor([0.0, 0.0, 1.0], device=device).expand(N, 3)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    # Dot product between z_axis and normals: (N,)
    cos_theta = (z_axis * normals).sum(dim=1)

    # Initialize quaternion tensor (w, x, y, z)
    quats = torch.zeros((N, 4), device=device)

    # Case 1: No rotation needed (cos_theta close to 1)
    mask_identity = torch.isclose(cos_theta, torch.tensor(1.0, device=device))
    quats[mask_identity] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    # Case 2: 180 degree rotation (cos_theta close to -1)
    # We rotate 180 degrees around the x-axis or y-axis (any axis perpendicular to z)
    mask_flip = torch.isclose(cos_theta, torch.tensor(-1.0, device=device))
    # Standard flip: 180 deg around X-axis -> q = [0, 1, 0, 0]
    quats[mask_flip] = torch.tensor([0.0, 1.0, 0.0, 0.0], device=device)

    # Case 3: Standard case
    mask_standard = ~(mask_identity | mask_flip)

    if mask_standard.any():
        # normals_standard shape: (M, 3), where M is number of standard cases
        normals_standard = normals[mask_standard]
        z_axis_standard = z_axis[mask_standard]
        cos_theta_standard = cos_theta[mask_standard]

        rotation_axis = torch.cross(z_axis_standard, normals_standard, dim=1)
        rotation_axis = rotation_axis / torch.norm(rotation_axis, dim=1, keepdim=True)
        angle = torch.acos(cos_theta_standard)

        half_angle = angle / 2.0
        sin_half_angle = torch.sin(half_angle)

        quats[mask_standard, 0] = torch.cos(half_angle)  # qw
        quats[mask_standard, 1:] = rotation_axis * sin_half_angle.unsqueeze(
            1
        )  # qx, qy, qz

    if is_single:
        return quats.squeeze(0)
    return quats


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


def init_load_pts_and_rgbs(
    config: Config, parser: Parser | NerfbaselinesParser
) -> tuple[torch.Tensor, torch.Tensor]:
    if config.init_type == "sparse":
        _LOGGER.info("using sparse points from parser")
        if parser.points_rgb is None:
            raise RuntimeError("Parser does not provide point colors for sparse init.")
        return (
            torch.from_numpy(parser.points).float(),
            torch.from_numpy(parser.points_rgb / 255.0).float(),
        )

    if config.init_type != "dense":
        raise RuntimeError(
            f"Unsupported init_type {config.init_type} for load_pts_and_rgbs"
        )

    # Dealing with "dense" init_type from here
    if not isinstance(parser, NerfbaselinesParser):
        raise RuntimeError(
            "Dense initialization currently requires NerfbaselinesParser."
        )

    _LOGGER.info("using dense points from nerfbaselines dataset")
    points, rgbs = load_pointcloud_ply(
        parser.nerfbaselines_dataset["dense_points3D_path"]
    )
    if rgbs is None:
        raise RuntimeError("Dense pointcloud does not contain colors.")

    points = transform_points(parser.transform, points)

    return (torch.from_numpy(points).float(), torch.from_numpy(rgbs).float())


def init_load_normals(
    config: Config, parser: Parser | NerfbaselinesParser
) -> torch.Tensor:

    path = None
    if config.init.use_normals is not None:
        _LOGGER.info("using normals from --init-normals-path")
        path = config.init.normals_path

    if isinstance(parser, NerfbaselinesParser) and path is None:
        _LOGGER.info("using normals from nerfbaselines dataset")
        path = parser.nerfbaselines_dataset.get("dense_points3D_normals_path", None)

    if path is None:
        raise RuntimeError(
            "Init with normals requires --init-normals-path or Nerfbaselines parser that provides normals."
        )

    _LOGGER.info(f"loading normals from {path}")
    normals = load_normals(path)
    normals = transform_normals(parser.transform, normals)

    return torch.from_numpy(normals).float()


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
    dist2_avg = (knn(points, 4)[0] ** 2).mean(dim=-1)  # [N,]

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


def init_without_normals(
    points: torch.Tensor, rgbs: torch.Tensor, config: Config, scene_scale: float
) -> SplatData:
    """
    Regular initialization without normals.
    """
    _LOGGER.info("initializing gaussians without using normals")

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

    dist2_avg = (knn(points, 4)[0] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = (dist_avg * config.init.scale_mult).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    if config.init.clamp_scales:
        scales = torch.clamp(scales, max=scene_scale / 100)
    scales = torch.log(scales)
    quats = torch.rand((N, 4))  # [N, 4]

    return InitResult(
        points=points, rgbs=rgbs, scales=scales, quats=quats
    ).to_splat_data(config)


def init_with_normals(
    points: torch.Tensor,
    rgbs: torch.Tensor,
    normals: torch.Tensor,
    config: Config,
    scene_scale: float,
):
    """
    Initialize gaussians using point normals.
    """

    _LOGGER.info(
        f"initializing gaussians with normals, small axis scale: {config.init.normal_init_small_axis_scale}"
    )

    if points.shape[0] != normals.shape[0]:
        raise RuntimeError("Number of points and normals must be identical.")

    if points.shape[0] != rgbs.shape[0]:
        raise RuntimeError("Number of points and rgbs must be identical.")

    if config.init_type == "dense":
        point_indices = _pick_dense_init_points(points, rgbs, config)
        points = points[point_indices]
        rgbs = rgbs[point_indices]
        normals = normals[point_indices]

    points, rgbs = _add_noise_to_init_points(points, rgbs, config, scene_scale)

    if config.init.remove_floaters:
        mask = _get_floater_mask(points, config)
        points = points[mask]
        rgbs = rgbs[mask]
        normals = normals[mask]

    dist2_avg = (knn(points, 4)[0] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = (dist_avg * config.init.scale_mult).unsqueeze(-1).repeat(1, 3)  # [N, 3]
    if config.init.clamp_scales:
        scales = torch.clamp(scales, max=scene_scale / 100)

    scales[:, 0] = scales[:, 0] * config.init.normal_init_small_axis_scale
    scales = torch.log(scales)

    quats = rotation_quat_from_normal(normals)  # [N, 4]

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

    if "initialization_splat_path" not in parser.nerfbaselines_dataset:
        raise RuntimeError(
            "Nerfbaselines dataset does not contain initialization splat path."
        )

    splat_path = parser.nerfbaselines_dataset["initialization_splat_path"]
    splat = load_splat_ply(splat_path)
    _get_splat_subset_inplace(splat, config)

    rotation, _, scale = decompose_rotation_translation_and_uniform_scale(
        parser.transform
    )
    splat.means = transform_points(parser.transform, splat.means).to(torch.float32)
    splat.scales = torch.log(torch.exp(splat.scales) * scale).to(torch.float32)

    # TODO: this is only fine as long as the init method outputs isotropic covariances.
    # If we want to support anisotropic covariances we need to apply rotation too
    # splat.quats = rotate_quaternions(splat.quats, rotation)
    scales_are_isotropic = torch.allclose(splat.scales[:, 0], splat.scales[:, 1]) and torch.allclose(splat.scales[:, 1], splat.scales[:, 2])
    if not scales_are_isotropic:
        raise NotImplementedError("Applying rotations to pre-made splats not implemented")

    return splat
