from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Union
import typing

from torch.types import Device as TorchDevice

from monodepth.depth_alignment.config import (
    AlignmentConfig,
    DepthAlignmentStrategyEnum,
)
from monodepth.depth_prediction.interface import (
    DepthPredictor,
)

from monodepth.depth_subsampling.adaptive_subsampling import (
    AdaptiveDepthSubsampler,
)
from monodepth.depth_subsampling.config import (
    AdaptiveSubsamplingConfig,
    NumSfMPointsMaskConfig,
)
from monodepth.depth_subsampling.interface import DepthSubsampler
from monodepth.depth_subsampling.static_subsampler import StaticDepthSubsampler
from shared.serializable_config import SerializableConfig

from .depth_prediction.configs import (
    Metric3dV2Config,
    DepthAnythingV2Config,
    MogeConfig,
    UnidepthConfig,
)


@dataclass
class Config(SerializableConfig):
    CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
        "output_dir",
        "cache_dir",
        "debug_output",
        "save_depth_maps",
        "output_unmerged_points",
    }

    # dataset string in nerfbaselines format
    scene: str = "external://mipnerf360/garden"

    predictor: Literal["metric3d", "moge", "depth_anything_v2", "depth_pro"] = (
        "metric3d"
    )

    metric3d: Metric3dV2Config = field(default_factory=Metric3dV2Config)
    unidepth: UnidepthConfig = field(default_factory=UnidepthConfig)
    depthanything: DepthAnythingV2Config = field(default_factory=DepthAnythingV2Config)
    moge: MogeConfig = field(default_factory=MogeConfig)

    # determines how to align predicted depths to SfM points
    alignment_method: DepthAlignmentStrategyEnum = (
        DepthAlignmentStrategyEnum.depth_range_interp
    )
    # configuration of individual alignment methods
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    floater_removal: bool = True
    final_outlier_removal: Literal["lof", "nn", "pcd_stat"] | None = None

    # Only use SfM points with reprojection error below this percentile for depth alignment and final output.
    sfm_points_max_err_percentile: float | None = 99.0

    # if >0, images are resized so that their largest dimension is target_image_size.
    # This is so the same # of points is produced based on subsampling settings, irrespective of input size.
    # And to potentially speed up depth predictions for large images. Set to -1 to disable resizing.
    target_image_size: int = 1297

    subsample_factor: Union[int, Literal["adaptive"]] = "adaptive"
    # Configuration for adaptive subsampling. Ignored if not using "adaptive" subsampling.
    adaptive_subsampling: AdaptiveSubsamplingConfig = field(
        default_factory=AdaptiveSubsamplingConfig
    )
    # if set, use mask out depth map regions with depth gradient above this threshold. (To filter out depths at object boundaries)
    depth_grad_mask_thresh: Optional[float] = 0.02

    max_num_images: Optional[int] = 300

    # datasets with modified point clouds from monocular depth init are saved to this directory.
    output_dir: Path = Path("./output")
    # path which is used for caching depth predictions and other data, relative paths interpreted relative to base_output_dir
    cache_dir: Path = Path("<<output_dir>>/__monodepth_cache__")
    # ignore cached depth predictions and recompute them
    ignore_depth_cache: bool = False

    # whether to include SfM points in the final output point cloud
    include_sfm_points: bool = False
    # whether to perform outlier removal on SfM points before depth alignment
    sfm_outlier_removal: bool = False
    # number of neighbors to use in LOF for SfM outlier removal
    sfm_lof_n_neighbors: int = 100

    # Whether to use masking based on number of SfM points per image patch.
    use_num_sfm_points_mask: bool = False
    num_sfm_points_mask: NumSfMPointsMaskConfig = field(
        default_factory=NumSfMPointsMaskConfig
    )
    # export debug outputs such as aligned depth maps, masks, etc.
    debug_output: bool = False

    # Whether to save predicted depth maps to disk
    save_depth_maps: bool = False

    def process(self):
        self.output_dir = self.output_dir.absolute()
        if self.cache_dir == Path("<<output_dir>>/__monodepth_cache__"):
            self.cache_dir = self.output_dir / "__monodepth_cache__"
        else:
            self.cache_dir = self.cache_dir.absolute()

    def get_predictor_class(self):
        if self.predictor == "metric3d":
            from .depth_prediction.predictors.metric3d import Metric3d

            return Metric3d
        elif self.predictor == "depth_pro":
            from .depth_prediction.predictors.apple_depth_pro import AppleDepthPro

            return AppleDepthPro
        elif self.predictor == "moge":
            from .depth_prediction.predictors.moge import MoGe

            return MoGe
        elif self.predictor == "depth_anything_v2":
            from .depth_prediction.predictors.depth_anything_v2 import DepthAnythingV2

            return DepthAnythingV2
        else:
            raise ValueError(f"Unsupported monocular depth predictor: {self.predictor}")

    def instantiate_predictor(self, device: TorchDevice) -> DepthPredictor:
        predictor_class = self.get_predictor_class()
        return predictor_class(self, device)

    def instantiate_subsampler(self) -> DepthSubsampler:
        if self.subsample_factor == "adaptive":
            return AdaptiveDepthSubsampler(self.adaptive_subsampling)
        elif isinstance(self.subsample_factor, int):
            return StaticDepthSubsampler(self.subsample_factor)  # noqa: F821
        else:
            raise ValueError(f"Unsupported subsampling factor: {self.subsample_factor}")
