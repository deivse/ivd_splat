from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union, Tuple
import typing
from typing_extensions import assert_never

from ivd_splat.strategies import (
    DefaultWithGaussianCapStrategy,
    DefaultWithoutADCStrategy,
    MCMCStrategy,
    IDHFRStrategy,
)
from shared.serializable_config import SerializableConfig


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

    knn_num_neighbors: int = 4  # for adaptive sampling


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

    # NOTE: These dataset parameters are ignored when runnin with nerfbaselines.
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
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

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
    strategy: Union[
        DefaultWithGaussianCapStrategy,
        DefaultWithoutADCStrategy,
        MCMCStrategy,
        IDHFRStrategy,
    ] = field(default_factory=DefaultWithGaussianCapStrategy)
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_background: bool = False

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

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(
            strategy,
            (DefaultWithGaussianCapStrategy, DefaultWithoutADCStrategy, IDHFRStrategy),
        ):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)

    def to_dict(self):
        retval = super().to_dict()
        if isinstance(self.strategy, DefaultWithGaussianCapStrategy):
            retval["strategy_"] = "DefaultWithGaussianCapStrategy"
        elif isinstance(self.strategy, DefaultWithoutADCStrategy):
            retval["strategy_"] = "DefaultWithoutADCStrategy"
        elif isinstance(self.strategy, MCMCStrategy):
            retval["strategy_"] = "MCMCStrategy"
        elif isinstance(self.strategy, IDHFRStrategy):
            retval["strategy_"] = "IDHFRStrategy"
        else:
            raise ValueError(f"Unknown strategy type: {type(self.strategy)}")
        return retval

    def to_flat_dict(self):
        retval = super().to_flat_dict()
        retval["strategy"] = retval.pop("strategy_")
        return retval
