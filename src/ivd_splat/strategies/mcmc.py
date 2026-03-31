from dataclasses import dataclass
import typing

from gsplat.strategy import MCMCStrategy as GSplatMCMCStrategy

from ivd_splat.strategies.base import IVDSplatBaseStrategy


@dataclass
class MCMCStrategy(GSplatMCMCStrategy, IVDSplatBaseStrategy):
    CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
        "verbose",
    }

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

    def step_post_backward(self, args: IVDSplatBaseStrategy.StepPostBackwardArgs):
        return GSplatMCMCStrategy.step_post_backward(
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
            "init.scale_mult": 0.1,
            "opacity_reg": 0.01,
            "scale_reg": 0.01,
        }
