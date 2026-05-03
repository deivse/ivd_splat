from ivd_splat.strategies.default_with_gaussian_cap import (
    DefaultWithGaussianCapStrategy,
)


class INRIAStrategy(DefaultWithGaussianCapStrategy):
    """Just a thin wrapper that sets absgrad to False by default, for convenience."""

    absgrad = False
