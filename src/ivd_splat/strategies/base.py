import abc
import typing
from typing import Any, Optional, Union

import torch
from gsplat.strategy.base import Dict, Strategy

from ivd_splat.datasets.colmap import Dataset
from shared.serializable_config import SerializableConfig


class IVDSplatBaseStrategy(Strategy, SerializableConfig, abc.ABC):
    """Customized strategy base class with better abstraction of common functionality."""

    class StepPreBackwardArgs(typing.NamedTuple):
        """Arguments for the `step_pre_backward` callback."""

        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]
        optimizers: Dict[str, torch.optim.Optimizer]
        state: Dict[str, Any]
        step: int
        info: Dict[str, Any]

    class StepPostBackwardArgs(typing.NamedTuple):
        """Arguments for the `step_post_backward` callback."""

        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict]
        optimizers: Dict[str, torch.optim.Optimizer]
        # Last args passed to gsplat.rendering.rasterization() in the current step.
        last_rasterization_args: Dict
        # The strategy state as created by `initialize_state`. Can be updated in-place across steps.
        state: Dict[str, Any]
        # The current step number.
        step: int
        # the last value returned by the `gsplat.rendering.rasterization()` call in the current step (meta).
        info: Dict[str, Any]
        # Whether packed mode is being used for rasterization (see `Config.packed`).
        packed: bool
        # Current step's LR for Gaussian means.
        lr: float

    def get_default_config_overrides(self) -> Dict[str, Any]:
        """Get default config overrides for this strategy.

        This can be used to set strategy-specific default values for the config.
        For example, MCMCStrategy uses this to set default values for the MCMC-specific config fields.
        The Config class calls this method during post-init (only values not changed from defaults are overridden).

        Note: for nested config fields, use dot notation to specify the field name. For example, to override `Config.init.opacity`, return a dict with key `init.opacity`.

        Returns:
            A (non-nested) dict mapping config field names to their default values for this strategy.
        """
        return {}

    @abc.abstractmethod
    def get_cap_max(self) -> Optional[int]:
        """
        Get the maximum number of GSs allowed by the strategy.
        If there is no limit, return None.
        """
        pass

    def should_step_optimizers(self, step: int) -> bool:
        """Whether to update the parameters in this step."""
        return True

    @abc.abstractmethod
    def initialize_state(self, scene_scale: float, dataset: Dataset) -> Dict[str, Any]:
        """Initialize the strategy state."""

    @abc.abstractmethod
    def step_pre_backward(self, args: StepPreBackwardArgs):
        """Callback function to be executed before the `loss.backward()` call."""
        pass

    @abc.abstractmethod
    def step_post_backward(
        self,
        args: StepPostBackwardArgs,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        pass
