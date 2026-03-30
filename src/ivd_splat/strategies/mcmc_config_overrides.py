from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ivd_splat.config import Config


def override_default_config_vals_for_mcmc(cfg: "Config"):
    """Override default config values for MCMC strategy."""
    cfg.init.opacity = 0.5
    cfg.init.scale_mult = 0.1
    cfg.opacity_reg = 0.01
    cfg.scale_reg = 0.01
