from dataclasses import dataclass
from enum import Enum
from typing import Literal

from shared.serializable_config import SerializableConfig


class DepthAnythingV2Backbone(str, Enum):
    vits = "vits"
    vitb = "vitb"
    vitl = "vitl"


@dataclass
class DepthAnythingV2Config(SerializableConfig):
    """
    Configuration for the Depth Anything V2 monocular depth predictor.
    """

    backbone: DepthAnythingV2Backbone = DepthAnythingV2Backbone.vitl
    """ Backbone to use. """
    metric: bool = True
    """
    Whether to use the model fine tuned for metric depth (True) or relative depth model (False).
    If true, `metric_model_type` must be specified.
    """

    metric_model_type: Literal["indoor", "outdoor"] | None = "indoor"
    """
    Which metric model to use. "indoor" is trained on Hypersim and "outdoor" is trained on VKITTI.
    Only used if `metric` is True.
    """


class Metric3dBackbone(str, Enum):
    vits = "vits"
    vitl = "vitl"
    vitg = "vitg"


@dataclass
class Metric3dV2Config(SerializableConfig):
    """
    Configuration for the Metric3dV2 monocular depth predictor.
    """

    backbone: Metric3dBackbone = Metric3dBackbone.vitl


class MogeBackbone(str, Enum):
    vits = "vits"
    vitl = "vitl"
    vitg = "vitg"


@dataclass
class MogeConfig(SerializableConfig):
    """
    Configuration for the MoGe monocular depth predictor.
    """

    backbone: MogeBackbone = MogeBackbone.vitl


class UnidepthBackbone(str, Enum):
    vits = "vits"
    vitb = "vitb"
    vitl = "vitl"


@dataclass
class UnidepthConfig(SerializableConfig):
    """
    Configuration for the UniDepth monocular depth predictor.
    """

    backbone: UnidepthBackbone = UnidepthBackbone.vitl
