from gsplat.strategy.base import Strategy
from shared.serializable_config import SerializableConfig


class IVDSplatBaseStrategy(Strategy, SerializableConfig):
    """Customized strategy for the IVD-Splat paper."""

    def should_step_optimizers(self, step: int) -> bool:
        """Whether to update the parameters in this step."""
        return True
