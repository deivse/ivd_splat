from dataclasses import dataclass
import typing
from typing import Any, Tuple, Union, Dict

import torch

from ivd_splat.datasets.colmap import Dataset
from ivd_splat.strategies.base import IVDSplatBaseStrategy
from ivd_splat.utils.runner_utils import rgb_to_sh
from gsplat.strategy.ops import _update_param_with_optimizer, duplicate, split, remove

from fused_ssim import allowed_padding, FusedSSIMMap


def fused_ssim_no_mean(img1, img2, padding="same", train=True):
    C1 = 0.01**2
    C2 = 0.03**2

    assert padding in allowed_padding

    return FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)


@torch.no_grad()
def duplicate_with_opacity_correction(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, torch.Tensor],
    mask: torch.Tensor,
):
    """Inplace duplicate the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]

    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        if name == "opacities":
            new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
            p[sel] = torch.logit(new_opacities)

        p_new = torch.cat([p, p[sel]])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = torch.cat((v, v[sel]))

    # device = mask.device
    # sel = torch.where(mask)[0]
    # rest = torch.where(~mask)[0]

    # def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
    #     repeats = [2] + [1] * (p.dim() - 1)

    #     if name == "opacities":
    #         new_opacities = 1.0 - torch.sqrt(1.0 - torch.sigmoid(p[sel]))
    #         p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
    #     else:
    #         p_split = p[sel].repeat(repeats)

    #     p_new = torch.cat([p[rest], p_split])
    #     return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    # def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
    #     return torch.cat([v, torch.zeros((len(sel), *v.shape[1:]), device=device)])

    # # update the parameters and the state in the optimizers
    # _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # # update the extra running state
    # for k, v in state.items():
    #     if isinstance(v, torch.Tensor):
    #         state[k] = torch.cat((v, v[sel]))


@dataclass
class RevDGSStrategy(IVDSplatBaseStrategy):
    CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
        "verbose",
        "debug_exports",
    }

    # TODO: numbers from paper.
    verbose: bool = False
    debug_exports: bool = False
    cap_max: int = -1
    refine_start_iter: int = 500
    refine_end_iter: int = 27000
    refine_every: int = 100

    large_gs_prune_start_iter: int = 3000

    # Gaussian-distributed error threshold above which densification is triggered.
    err_thresh: float = 0.1
    prune_opa: float = 0.005
    grow_scale3d: float = 0.01
    prune_scale3d: float = 0.1

    # The max number of primitives added during densification iteration is max_grow_fraction * num_splats.
    max_grow_fraction: float = 0.05
    # This is different than opacity regularization.
    # Staying faithful to the paper, this is a fixed decrease that is applied directly to the opacities:
    # "Specifically, we decrease the opacity of each primitive by a fixed amount (we use 0.001) after each densification run"
    opacity_decrease: float = 0.001
    # Excerpt from paper: "To counteract this dynamics, we also regularize the residual probabilities of the
    # alpha-compositing (a.k.a. residual transmittance) to be zero for every pixel, by
    # simply minimizing their average value, weighted by a hyperparameter (here 0.1).
    residual_opacity_reg_weight: float = 0.1

    def get_cap_max(self):
        if self.cap_max == -1:
            return None
        return self.cap_max

    def initialize_state(self, scene_scale: float, _: Dataset) -> dict:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` callbacks.
        """
        return {"scene_scale": scene_scale}

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers."""

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    def get_extra_signals(
        self, splat_params: torch.ParameterDict, strategy_state: dict
    ):
        num_splats = splat_params["means"].shape[0]
        if strategy_state.get("extra_signals", None) is None:
            strategy_state["extra_signals"] = torch.zeros(
                (num_splats, 1), device=splat_params["means"].device, requires_grad=True
            )
            strategy_state["max_error_contribs"] = torch.full(
                (num_splats,),
                -torch.finfo(torch.float32).max,
                dtype=torch.float32,
                device=splat_params["means"].device,
            )
        return strategy_state["extra_signals"]

    def step_pre_backward(self, args: IVDSplatBaseStrategy.StepPreBackwardArgs):
        pass

    def get_additional_loss_term(self, args: IVDSplatBaseStrategy.AdditionalLossArgs):
        rendered_signals = args.info["render_extra_signals"]

        pixel_errs = 1 - (
            fused_ssim_no_mean(
                args.rendered_image.permute(0, 3, 1, 2),
                args.gt_image.permute(0, 3, 1, 2),
                padding="same",
                train=False,
            )
            .squeeze()
            .mean(dim=0)
            .detach()
        )
        L_aux = (pixel_errs * rendered_signals.squeeze()).sum()
        L_residual_opacity = (
            self.residual_opacity_reg_weight * (1 - args.rendered_opacity).mean()
        )

        return L_aux + L_residual_opacity

    @torch.no_grad()
    def step_post_backward(self, args: IVDSplatBaseStrategy.StepPostBackwardArgs):
        step, params, optimizers, state, info, packed = (
            args.step,
            args.params,
            args.optimizers,
            args.state,
            args.info,
            args.packed,
        )

        # Update error contribs
        state["max_error_contribs"] = torch.max(
            state["max_error_contribs"], state["extra_signals"].grad.squeeze()
        )
        state["extra_signals"].grad.zero_()

        if (
            step < self.refine_start_iter
            or step > self.refine_end_iter
            or step % self.refine_every != 0
        ):
            return

        # grow GSs
        n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
        if self.verbose:
            print(
                f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                f"Now having {len(params['means'])} GSs."
            )

        # prune GSs
        n_prune = self._prune_gs(params, optimizers, state, step)
        if self.verbose:
            print(
                f"Step {step}: {n_prune} GSs pruned. "
                f"Now having {len(params['means'])} GSs."
            )

        # Decrease opacity after each densification run.
        new_alpha = torch.sigmoid(params["opacities"]) - self.opacity_decrease
        eps = 1e-7
        params["opacities"].data.copy_(torch.logit(new_alpha.clamp(eps, 1 - eps)))

        if self.debug_exports:
            # visualize error contribs
            from matplotlib import cm
            from shared.splat_ply_io import export_splat_ply, SplatData

            values = args.state["max_error_contribs"].squeeze().cpu().numpy()
            colors = torch.from_numpy(
                cm.magma(values / (values.max() + 1e-8))[:, :3]
            ).to(
                device=args.params["sh0"].device,
                dtype=args.params["sh0"].dtype,
            )

            export_splat_ply(
                f"error_contribs_{args.step}.ply",
                SplatData(
                    means=args.params["means"].detach().cpu(),
                    scales=args.params["scales"].detach().cpu(),
                    quats=args.params["quats"].detach().cpu(),
                    opacities=args.params["opacities"].detach().cpu(),
                    sh0=rgb_to_sh(colors).unsqueeze(1).detach().cpu(),
                    shN=torch.zeros_like(args.params["shN"]).detach().cpu(),
                ),
            )

        state["extra_signals"].requires_grad_(True)
        # Reset max error contribs
        state["max_error_contribs"] = torch.full(
            (params["means"].shape[0],),
            -torch.finfo(torch.float32).max,
            dtype=torch.float32,
            device=params["means"].device,
        )

        torch.cuda.empty_cache()

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> Tuple[int, int]:
        error_contribs = state["max_error_contribs"]
        device = error_contribs.device

        curr_num_splats = len(params["means"])

        is_candidate = error_contribs > self.err_thresh
        num_splats_to_add = min(
            int(self.max_grow_fraction * curr_num_splats),
            is_candidate.sum().item(),
        )
        if self.cap_max != -1:
            remaining_capacity = max(self.cap_max - curr_num_splats, 0)
            num_splats_to_add = min(num_splats_to_add, remaining_capacity)

        if num_splats_to_add == 0:
            return 0, 0

        if num_splats_to_add < is_candidate.sum().item():
            # keep only the top-k splats based on gradient magnitude
            _, topk_indices = torch.topk(
                error_contribs[is_candidate], num_splats_to_add
            )
            new_is_candidate = torch.zeros_like(is_candidate, dtype=torch.bool)
            new_is_candidate[torch.where(is_candidate)[0][topk_indices]] = True
            is_candidate = new_is_candidate

        is_small = (
            torch.exp(params["scales"]).max(dim=-1).values
            <= self.grow_scale3d * state["scene_scale"]
        )
        is_dupli = is_candidate & is_small
        n_dupli = is_dupli.sum().item()

        is_large = ~is_small
        is_split = is_candidate & is_large
        n_split = is_split.sum().item()

        # first duplicate
        if n_dupli > 0:
            duplicate_with_opacity_correction(
                params=params, optimizers=optimizers, state=state, mask=is_dupli
            )

        # new GSs added by duplication will not be split
        is_split = torch.cat(
            [
                is_split,
                torch.zeros(n_dupli, dtype=torch.bool, device=device),
            ]
        )

        # then split
        if n_split > 0:
            split(
                params=params,
                optimizers=optimizers,
                state=state,
                mask=is_split,
                # Per RevDGS §3.3, this correction is for clone, not split. gsplat exposes the flag on split, which contradicts the paper.
                revised_opacity=False,
            )
        return n_dupli, n_split

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa

        # Unfortunately, the paper doesn't explicitly specify whether big Gs pruning is also delayed until iter 3000
        # as in INRIA ADC, but we choose to err on the side of being faithful to the that, as RevDGS authors specify that
        # "Other relevant hyper-parameters are left to the default values used in 3DGS..."
        if step > self.large_gs_prune_start_iter:
            is_too_big = (
                torch.exp(params["scales"]).max(dim=-1).values
                > self.prune_scale3d * state["scene_scale"]
            )

            is_prune = is_prune | is_too_big

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return int(n_prune)
