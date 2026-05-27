from dataclasses import dataclass

from ivd_splat.strategies.default_with_gaussian_cap import (
    DefaultWithGaussianCapStrategy,
)


@dataclass
class INRIAStrategy(DefaultWithGaussianCapStrategy):
    """Just a thin wrapper that sets absgrad to False by default, for convenience."""

    absgrad: bool = False
    grow_grad2d: float = 0.0002
