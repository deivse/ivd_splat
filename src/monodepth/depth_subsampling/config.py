from dataclasses import dataclass

from shared.serializable_config import SerializableConfig


@dataclass
class AdaptiveSubsamplingConfig(SerializableConfig):
    """
    Configures which heuristics to use for adaptive subsampling.
    """

    # Range of subsample factors to choose from.
    factor_range_min: int = 5
    factor_range_max: int = 15


@dataclass
class NumSfMPointsMaskConfig(SerializableConfig):
    """
    Configuration for masking based on number of SfM points per image patch.
    """

    # Number of patches along the smaller image axis.
    num_patches_small_axis: int = 20

    # Number of SfM points in a patch,
    # above which points won't be unprojected from that patch.
    threshold: int = 15
