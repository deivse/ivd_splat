from dataclasses import dataclass
import typing

from torch import Tensor
import torch
from gsplat.strategy import MCMCStrategy as GSplatMCMCStrategy
from gsplat.strategy.ops import remove

from ivd_splat.strategies.base import IVDSplatBaseStrategy


@dataclass
class MCMCModStrategy(GSplatMCMCStrategy, IVDSplatBaseStrategy):
    CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
        "verbose",
    }

    base_opacity_reg: float = 0.0025
    base_scale_reg: float = 0.0025
    view_opacity_reg: float = 0.01
    view_scale_reg: float = 0.01

    init_scale_mult: float = 0.1

    def get_cap_max(self):
        if self.cap_max == -1:
            return None
        return self.cap_max

    def initialize_state(self, *args, **kwargs) -> dict:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` callbacks.
        """
        return GSplatMCMCStrategy.initialize_state(self)

    def step_pre_backward(self, *args, **kwargs):
        pass

    def get_additional_loss_term(
        self, args: IVDSplatBaseStrategy.AdditionalLossArgs
    ) -> typing.Optional[Tensor]:
        """Get an additional loss term to be added to the main loss in the current step.

        This can be used to implement strategies that require an additional loss term besides the main image reconstruction loss.
        The returned loss term will be added to the main loss before calling `loss.backward()`.

        Args:
            args: An AdditionalLossArgs containing relevant information for computing the additional loss term.
        Returns:
            A scalar tensor representing the additional loss term to be added to the main loss. Return 0 or None to not add any additional loss term.
        """

        radii = args.info["radii"]
        valid_in_image: torch.Tensor = (radii > 0).all(dim=-1).squeeze()

        splat_opas = torch.abs(torch.sigmoid(args.splats["opacities"]))
        opacity_reg = (
            self.base_opacity_reg * splat_opas.mean()
            + self.view_opacity_reg * (splat_opas * valid_in_image).mean()
        )
        splat_scales = torch.abs(torch.exp(args.splats["scales"]))
        scale_reg = (
            self.base_scale_reg * splat_scales.mean()
            + self.view_scale_reg * (splat_scales * valid_in_image).mean()
        )
        return opacity_reg + scale_reg

    def step_post_backward(
        self, args: IVDSplatBaseStrategy.StepPostBackwardArgs
    ) -> None:
        GSplatMCMCStrategy.step_post_backward(
            self,
            args.params,
            args.optimizers,
            args.state,
            args.step,
            args.info,
            args.lr,
        )

    def get_default_config_overrides(self):
        return {
            "init.opacity": 0.5,
            "init.scale_mult": self.init_scale_mult,
        }
