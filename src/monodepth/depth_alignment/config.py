from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

from shared.serializable_config import SerializableConfig


class DepthAlignmentStrategyEnum(str, Enum):
    """Enumeration of depth alignment strategies."""

    none = "none"
    lstsqrs = "lstsqrs"
    ransac = "ransac"
    msac = "msac"
    interp = "interp"
    depth_range_interp = "depth_range_interp"

    def get_implementation(self):
        if self == self.none:
            from .alignment.none import DepthAlignmentNone

            return DepthAlignmentNone
        elif self == self.lstsqrs:
            from .alignment.lstsqrs import DepthAlignmentLstSqrs

            return DepthAlignmentLstSqrs
        elif self == self.ransac:
            from .alignment.ransacs import DepthAlignmentRansac

            return DepthAlignmentRansac
        elif self == self.msac:
            from .alignment.ransacs import DepthAlignmentMsac

            return DepthAlignmentMsac
        elif self == self.interp:
            from .alignment.interp import DepthAlignmentInterpolate

            return DepthAlignmentInterpolate
        elif self == self.depth_range_interp:
            from .alignment.depth_range_interp import (
                DepthAlignmentDepthRangeInterpolate,
            )

            return DepthAlignmentDepthRangeInterpolate
        else:
            raise NotImplementedError(f"Unknown depth alignment strategy: {self}")


@dataclass
class RansacConfig(SerializableConfig):
    """Configuration for all RANSAC-based depth alignment. This also affects '*interp' if it uses RANSAC for initial estimation."""

    inlier_threshold: float = 0.01
    max_iters: int = 2500
    confidence: float = 0.999
    sample_size: int = 4
    min_iters: int = 0


@dataclass
class InterpConfig(SerializableConfig):
    """Configuration for 'interp' and 'depth_range_interp' depth alignment strategies."""

    init: Literal["lstsqrs", "ransac", "msac"] | None = "ransac"
    """If set, use this method to get an initial estimate of scale and shift before interpolation."""

    scale_outlier_removal: Literal["complex", "iqr"] | None = "complex"
    """
    If true, enable scale factor outlier removal using a custom method which detects outliers 
    in 2D coordinates, and scale, and only selects the points which are only outliers in scale.
    """


@dataclass
class AlignmentConfig(SerializableConfig):
    """Depth alignment configuration."""

    ransac: RansacConfig = field(default_factory=RansacConfig)
    """Configuration for all RANSAC-based depth alignment. This also affects 'interp' if it uses RANSAC for initial estimation."""

    interp: InterpConfig = field(default_factory=InterpConfig)
    """Configuration for 'interp' depth alignment strategy."""
