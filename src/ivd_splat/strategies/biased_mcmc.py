from dataclasses import dataclass
import typing
from typing import Any, Dict, Union

from torch import Tensor
import torch
from gsplat.strategy import MCMCStrategy as GSplatMCMCStrategy
from gsplat.strategy.ops import (
    _multinomial_sample,
    _update_param_with_optimizer,
    compute_relocation,
    inject_noise_to_position,
)

from ivd_splat.datasets.colmap import Dataset
from ivd_splat.strategies.base import IVDSplatBaseStrategy
from ivd_splat.strategies.revdgs import fused_ssim_no_mean


@torch.no_grad()
def relocate_custom_probs(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    probabilities: Tensor,  # [N,] tensor of probabilities for each Gaussian
    mask: Tensor,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    """Inplace relocate some dead Gaussians to the lives ones.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to indicates which Gaussians are dead.
    """
    # support "opacities" with shape [N,] or [N, 1]
    opacities = torch.sigmoid(params["opacities"])

    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n = len(dead_indices)

    # Sample for new GSs
    eps = torch.finfo(torch.float32).eps
    probs = probabilities[alive_indices].flatten()
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    sampled_idxs = alive_indices[sampled_idxs]
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p[dead_indices] = p[sampled_idxs]
        return torch.nn.Parameter(p, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v[sampled_idxs] = 0
        return v

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v[sampled_idxs] = 0


# We wouldn't need to re-implement this, but original sample_add has a bug in the
# state update which tries to get the shape of a non-tensor state value (binoms) and fails.
# This is a workaround to avoid that.
@torch.no_grad()
def sample_add_state_fix(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, Tensor],
    n: int,
    binoms: Tensor,
    min_opacity: float = 0.005,
):
    opacities = torch.sigmoid(params["opacities"])

    eps = torch.finfo(torch.float32).eps
    probs = opacities.flatten()
    sampled_idxs = _multinomial_sample(probs, n, replacement=True)
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[sampled_idxs],
        scales=torch.exp(params["scales"])[sampled_idxs],
        ratios=torch.bincount(sampled_idxs)[sampled_idxs] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, p: Tensor) -> Tensor:
        if name == "opacities":
            p[sampled_idxs] = torch.logit(new_opacities)
        elif name == "scales":
            p[sampled_idxs] = torch.log(new_scales)
        p_new = torch.cat([p, p[sampled_idxs]])
        return torch.nn.Parameter(p_new, requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: Tensor) -> Tensor:
        v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
        return torch.cat([v, v_new])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            v_new = torch.zeros((len(sampled_idxs), *v.shape[1:]), device=v.device)
            state[k] = torch.cat((v, v_new))


@dataclass
class NonSplatStateTensor:
    """
    A wrapper to avoid functions like relocate trying to adjust
    a state tensor that doesn't correspond to gaussians in the scene.
    """

    value: Tensor


@dataclass
class BiasedMCMCStrategy(GSplatMCMCStrategy, IVDSplatBaseStrategy):
    CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
        "verbose",
    }

    opacity_reg: float = 0.01
    scale_reg: float = 0.01
    init_scale_mult: float = 0.1
    key_for_gradient: str = "means2d"
    absgrad: bool = True
    revdgs_mode: bool = False
    prob_scaler_weight: float = 0.5

    def get_cap_max(self):
        if self.cap_max == -1:
            return None
        return self.cap_max

    def initialize_state(self, scene_scale: float, _: Dataset) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        state = GSplatMCMCStrategy.initialize_state(self)
        state["binoms"] = NonSplatStateTensor(state["binoms"])
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state = {**state, "grad2d": None, "count": None, "scene_scale": scene_scale}
        return state

    def get_extra_signals(
        self, splat_params: torch.ParameterDict, strategy_state: dict
    ):
        if not self.revdgs_mode:
            return None
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

    def get_additional_loss_term(self, args: IVDSplatBaseStrategy.AdditionalLossArgs):
        if not self.revdgs_mode:
            return 0.0

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

        return (pixel_errs * rendered_signals.squeeze()).sum()

    def step_pre_backward(
        self,
        args: IVDSplatBaseStrategy.StepPreBackwardArgs,
    ):
        """Callback function to be executed before the `loss.backward()` call."""
        assert (
            self.key_for_gradient in args.info
        ), "The 2D means of the Gaussians is required but missing."
        args.info[self.key_for_gradient].retain_grad()

    def step_post_backward(
        self, args: IVDSplatBaseStrategy.StepPostBackwardArgs
    ) -> None:
        step, params, optimizers, state, info, packed = (
            args.step,
            args.params,
            args.optimizers,
            args.state,
            args.info,
            args.packed,
        )

        # move to the correct device
        state["binoms"].value = state["binoms"].value.to(params["means"].device)
        binoms = state["binoms"].value

        if step < self.refine_stop_iter:
            self._update_state(params, state, info, packed=packed)
            if self.revdgs_mode:
                # Update error contribs
                state["max_error_contribs"] = torch.max(
                    state["max_error_contribs"], state["extra_signals"].grad.squeeze()
                )
                state["extra_signals"].grad.zero_()

        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # teleport GSs
            n_relocated_gs = self._relocate_gs(params, optimizers, binoms, state)
            if self.verbose:
                print(f"Step {step}: Relocated {n_relocated_gs} GSs.")

            # add new GSs
            n_new_gs = self._add_new_gs(params, optimizers, binoms, state)
            if self.verbose:
                print(
                    f"Step {step}: Added {n_new_gs} GSs. "
                    f"Now having {len(params['means'])} GSs."
                )

            torch.cuda.empty_cache()

            if self.revdgs_mode:
                state["extra_signals"].requires_grad_(True)
                # Reset max error contribs
                state["max_error_contribs"] = torch.full(
                    (params["means"].shape[0],),
                    -torch.finfo(torch.float32).max,
                    dtype=torch.float32,
                    device=params["means"].device,
                )

        # add noise to GSs (stop after noise_injection_stop_iter if set)
        noise_stop = (
            self.noise_injection_stop_iter
            if self.noise_injection_stop_iter >= 0
            else float("inf")
        )
        if step < noise_stop:
            inject_noise_to_position(
                params=params,
                optimizers=optimizers,
                state=state,
                scaler=args.lr * self.noise_lr,
            )

    def get_default_config_overrides(self):
        return {
            "init.opacity": 0.5,
            "init.scale_mult": self.init_scale_mult,
            "opacity_reg": self.opacity_reg,
            "scale_reg": self.scale_reg,
        }

    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        state: Dict[str, Any],
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= self.min_opacity
        n_gs = dead_mask.sum().item()

        if n_gs == 0:
            return 0

        probs_base = opacities.flatten()

        if not self.revdgs_mode:
            count = state["count"]
            prob_scaler = state["grad2d"] / count.clamp_min(1)
        else:
            prob_scaler = state["max_error_contribs"]

        prob_scaler_norm = prob_scaler / prob_scaler.max().clamp_min(1e-6)
        probs = (
            probs_base * (1 - self.prob_scaler_weight)
            + probs_base * prob_scaler_norm * self.prob_scaler_weight
        )

        relocate_custom_probs(
            params=params,
            optimizers=optimizers,
            state=state,
            probabilities=probs,
            mask=dead_mask,
            binoms=binoms,
            min_opacity=self.min_opacity,
        )
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
        state: Dict[str, Any],
    ) -> int:
        current_n_points = len(params["means"])
        n_target = min(self.cap_max, int(1.05 * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            sample_add_state_fix(
                params=params,
                optimizers=optimizers,
                state=state,
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    def _update_state(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        state: Dict[str, Any],
        info: Dict[str, Any],
        packed: bool = False,
    ):
        for key in [
            "width",
            "height",
            "n_cameras",
            "radii",
            "gaussian_ids",
            self.key_for_gradient,
        ]:
            assert key in info, f"{key} is required but missing."

        # normalize grads to [-1, 1] screen space
        if self.absgrad:
            grads = info[self.key_for_gradient].absgrad.clone()
        else:
            grads = info[self.key_for_gradient].grad.clone()
        grads[..., 0] *= info["width"] / 2.0 * info["n_cameras"]
        grads[..., 1] *= info["height"] / 2.0 * info["n_cameras"]

        # initialize state on the first run
        n_gaussian = len(list(params.values())[0])

        if state["grad2d"] is None:
            state["grad2d"] = torch.zeros(n_gaussian, device=grads.device)
        if state["count"] is None:
            state["count"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
