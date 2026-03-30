# License Notice: Based on gsplat examples, which is licensed under Apache 2.0.
import logging
import sys
import time

import torch
import tyro
from ivd_splat.config import Config
from ivd_splat.runner import Runner

from gsplat.distributed import cli
from ivd_splat.strategies import (
    DefaultWithGaussianCapStrategy,
    DefaultWithoutADCStrategy,
    MCMCStrategy,
)
from ivd_splat.strategies.mcmc_config_overrides import (
    override_default_config_vals_for_mcmc,
)


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

    if cfg.ckpt is not None:
        # run eval only
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
    else:
        runner.train()

    if not cfg.disable_viewer and not cfg.non_blocking_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


def run_with_config(cfg: Config):
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except ImportError:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    cli(main, cfg, verbose=True)


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultWithGaussianCapStrategy(verbose=True),
            ),
        ),
        "no_adc": (
            "Gaussian splatting training with pruning but no ADC heuristics.",
            Config(
                strategy=DefaultWithoutADCStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    override_default_config_vals_for_mcmc(configs["mcmc"][1])

    logging.basicConfig(level=logging.INFO)
    cfg = tyro.extras.overridable_config_cli(configs)
    run_with_config(cfg)
