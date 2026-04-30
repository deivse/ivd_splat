from dataclasses import dataclass, field
import dataclasses
import logging
from typing import Optional, List, Literal, Tuple
import typing

from ivd_splat.strategies import DefaultWithGaussianCapStrategy, IVDSplatBaseStrategy
from shared.serializable_config import SerializableConfig

_LOGGER = logging.getLogger(__name__)


@dataclass
class DenseInitConfig(SerializableConfig):
    # Number of points to sample from dense point cloud for initialization
    # if None, use all points
    target_num_points: Optional[int] = None

    # If set, a fraction of target_num_points or total points (if target_num_points is None)
    # will be randomly sampled from the dense point cloud for initialization
    target_points_fraction: Optional[float] = None

    # Method to sample points from dense point cloud
    # Uniform - random uniform sampling
    # Adaptive - sample each point with probability proportional to color differences with it's K nearest neighbors
    sampling: Literal["uniform", "adaptive"] = "uniform"

    knn_num_neighbors: int = 3  # for adaptive sampling


@dataclass
class NanoGSConfig(SerializableConfig):
    # Merge Gs with edge cost below this value
    cost_threshold: float = 1.3
    # Number of merge iterations to perform
    iterations: int = 3

    # Before main merging loop, prune splats with opacity below min(preprune_opacity_threshold, all_opacities.median()).
    preprune_opacity_threshold: float = 0.1
    # Number of nearest neighbors to consider when building merge graph.
    knn_k: int = 16


@dataclass
class InitConfig(SerializableConfig):
    # Multiplier for position noise std relative to scene scale.
    # final std is scene_scale * position_noise_std
    position_noise_std: float = 0.0
    # Color noise stddev added to each point during initialization
    color_noise_std: float = 0.0

    # Clamp initial scales to be at most scene_scale / 100
    clamp_scales: bool = False
    # Initialize with normals, either dataset must provide normals or --md-init-path must be set
    use_normals: bool = False

    # Override/provide point normals used for initialization. Used if use_normals is True
    # Should corespond to the dataset's sfm or dense point cloud depending on init_type
    normals_path: Optional[str] = None

    # If true, points with average distance to knn_num_neighbors neighbors which is above
    # floater_knn_distance_percentile percentile will be removed
    remove_floaters: bool = False
    # Percentile threshold on average knn distance for floater removal
    floater_knn_distance_percentile: float = 0.0

    # Initial opacity of GS
    opacity: float = 0.1
    # Multipler for initial scale of GS relative to average distance to knn neighbors
    scale_mult: float = 1.0
    # For normal initialization, scale along small axis is set to normal_init_small_axis_scale * scale_mult * avg_knn_distance
    normal_init_small_axis_scale: float = 0.2


@dataclass
class RandomInitConfig(SerializableConfig):
    # Initial number of GSs. Ignored if using sfm or monodepthth
    num_points: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    extent: float = 3.0


@dataclass
class SplatInitConfig(SerializableConfig):
    # Whever to increase the scale of the splats when using fewer splats for initialization (based on dense_init.target_num_points), to keep the overall scene coverage similar.
    # ie if target_splat_fraction is 0.5 and increase_scale_with_fewer_splats is True,
    # then the scale of each splat will be multiplied by 1/0.5 = 2.0.
    # With EDGS init, it seemed to perform much better with AbsGS densification (tested on a couple scenes), without this
    # densification created a lot of useless splats at the start instead of growing them.
    increase_scale_with_fewer_splats: bool = True


@dataclass
class Config(SerializableConfig):
    CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
        "disable_viewer",
        "non_blocking_viewer",
        "port",
        "data_dir",
        "result_dir",
        "tb_every",
        "tb_save_image",
    }

    # Disable viewer
    disable_viewer: bool = False
    # Don't wait for user to close viewer after finishing training
    non_blocking_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None

    # NOTE: These dataset parameters are ignored when running with nerfbaselines.
    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8

    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1

    # Number of training steps
    max_steps: int = 30_000
    # Steps to evaluate the model
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    # Steps to save the model
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_final_ply: bool = True

    # Initialization strategy
    init_type: Literal["sparse", "dense", "splat", "random"] = "sparse"

    init: InitConfig = field(default_factory=InitConfig)
    dense_init: DenseInitConfig = field(default_factory=DenseInitConfig)
    random_init: RandomInitConfig = field(default_factory=RandomInitConfig)
    splat_init: SplatInitConfig = field(default_factory=SplatInitConfig)

    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: IVDSplatBaseStrategy = field(
        default_factory=DefaultWithGaussianCapStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_background: bool = False

    # If set to a positive number, will run NanoGS simplification after n steps of training.
    # Warning: Best to avoid doing this after densification begins with the current implementation (see TODOs.)
    nanogs_simplify_iter: int = -1
    # Configuration for NanoGS simplification.
    nanogs_config: Optional[NanoGSConfig] = field(default_factory=NanoGSConfig)

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Means initial learning rate
    means_lr_init: float = 1.6e-4
    # Means final learning rate
    means_lr_final: float = 1.6e-4 * 0.01

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 1e-5
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    # Network used for Learned Perceptual Image Patch Similarity (LPIPS) loss
    lpips_net: Literal["vgg", "alex"] = "alex"

    # ====== nerfbaselines extensions ======

    # Appearance optimization eval settings
    app_test_opt_steps: int = 128
    app_test_opt_lr: float = 0.1

    # Background color for rendering
    background_color: Optional[Tuple[float, float, float]] = None

    def __post_init__(self):
        strategy_overrides = self.strategy.get_default_config_overrides()
        for k, v in strategy_overrides.items():
            key_sequence = k.split(".")
            cfg = self
            try:
                for key in key_sequence[:-1]:
                    cfg = getattr(cfg, key)
                value = getattr(cfg, key_sequence[-1])
            except AttributeError as e:
                raise AttributeError(
                    f"Invalid config override key {k} from strategy {type(self.strategy).__name__}"
                ) from e

            # ensure cfg is a dataclass
            if not dataclasses.is_dataclass(cfg):
                raise ValueError(
                    f"Strategy config overrides only support dataclasses, but {type(cfg)} is not a dataclass (key {k} from strategy {type(self.strategy).__name__})"
                )
            field = [f for f in dataclasses.fields(cfg) if f.name == key_sequence[-1]][
                0
            ]
            default_value = (
                field.default
                if field.default is not dataclasses.MISSING
                else (
                    field.default_factory()
                    if field.default_factory is not dataclasses.MISSING
                    else None
                )
            )
            if value != default_value:
                _LOGGER.info(
                    f"Config field {k} is set to {value}, not overriding with strategy {type(self.strategy).__name__} default value {default_value}"
                )
            else:
                setattr(cfg, key_sequence[-1], v)
                _LOGGER.info(
                    f"Overriding config field {k} with strategy {type(self.strategy).__name__} default value {v}"
                )

    def to_dict(self):
        retval = super().to_dict()
        retval["strategy_"] = type(self.strategy).__name__
        return retval

    def to_flat_dict(self):
        retval = super().to_flat_dict()
        retval["strategy"] = retval.pop("strategy_")
        return retval
