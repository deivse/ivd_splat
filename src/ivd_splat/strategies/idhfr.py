# License Notice:
# Some of the contents in this file are based on the official implementation https://github.com/XiaoBin2001/Improved-GS,
# which is licensed under the terms of the Gaussian Splatting license (https://github.com/XiaoBin2001/Improved-GS/blob/d008d11849d9dec0be8824484694a281add3c7dc/LICENSE.md).

from dataclasses import dataclass
import math
from typing import Any, Dict, Union, Literal
import typing

from PIL import ImageFilter
from PIL.Image import Image as PILImage
import torch
import torch.nn.functional as F
from torchvision.transforms import functional as visionF

from gsplat.strategy.ops import (
    _update_param_with_optimizer,
    normalized_quat_to_rotmat,
    remove,
    reset_opa,
)

from ivd_splat.datasets.colmap import Dataset
from ivd_splat.strategies.base import IVDSplatBaseStrategy
from ivd_splat.strategies.idhfr_stuff.raster_with_accum_weights import (
    rasterization_inria_wrapper_accum_weights,
)


def get_edges(image: torch.Tensor):
    image_pil: PILImage = visionF.to_pil_image(image)
    image_gray: PILImage = image_pil.convert("L")
    image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
    image_edges_tensor = visionF.to_tensor(image_edges)
    return image_edges_tensor.squeeze()  # Remove useless channel


def normalize(value_tensor):
    value_tensor[value_tensor.isnan()] = 0
    valid_indices = value_tensor > 0
    valid_value = value_tensor[valid_indices].to(torch.float32)
    ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
    ret_value[valid_indices] = valid_value / torch.mean(valid_value)

    return ret_value


@torch.no_grad()
def long_axis_split(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    state: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    split_dist_scale: float,
    opacity_reduction: float,
):
    """Inplace split the Gaussian with the given mask.

    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to split the Gaussians.
        split_dist_scale: The scale factor for splitting the Gaussians along their longest axis.
        opacity_reduction: The factor by which to reduce the opacity of the split Gaussians.
    """
    device = mask.device
    sel = torch.where(mask)[0]
    rest = torch.where(~mask)[0]

    scales = torch.exp(params["scales"][sel])
    max_scales, max_scale_indices = torch.max(scales, dim=1, keepdim=True)
    longest_axis_mask = torch.zeros_like(scales, dtype=torch.bool).scatter(
        dim=1, index=max_scale_indices, value=True
    )

    longest_axis_lens = scales * longest_axis_mask * 3.0

    quats = F.normalize(params["quats"][sel], dim=-1)
    rotmats = normalized_quat_to_rotmat(quats).repeat(2, 1, 1)  # [2N, 3, 3]

    offset = longest_axis_lens * split_dist_scale
    mean_offsets = torch.cat([offset, -offset], dim=0)  # [2N, 3]
    means_offsets = (
        torch.bmm(rotmats, mean_offsets.unsqueeze(-1))
        .squeeze(-1)
        .reshape(2, len(scales), 3)
    )  # [2, N, 3]

    scale_mult_longest = 1.0 - split_dist_scale
    scale_mult_other = math.sqrt(1.0 - split_dist_scale**2)
    scales = (
        scales.scatter(
            1, max_scale_indices, max_scales * scale_mult_longest / scale_mult_other
        )
        * scale_mult_other
    )

    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        repeats = [2] + [1] * (p.dim() - 1)
        if name == "means":
            p_split = (p[sel] + means_offsets).reshape(-1, 3)  # [2N, 3]
        elif name == "scales":
            p_split = torch.log(scales).repeat(repeats)  # [2N, 3]
        elif name == "opacities":
            # Each opacity multiplied by opacity_reduction
            new_opacities = torch.sigmoid(p[sel]) * opacity_reduction
            p_split = torch.logit(new_opacities).repeat(repeats)  # [2N]
        else:
            p_split = p[sel].repeat(repeats)
        p_new = torch.cat([p[rest], p_split])
        p_new = torch.nn.Parameter(p_new, requires_grad=p.requires_grad)
        return p_new

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        v_split = torch.zeros((2 * len(sel), *v.shape[1:]), device=device)
        return torch.cat([v[rest], v_split])

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            repeats = [2] + [1] * (v.dim() - 1)
            v_new = v[sel].repeat(repeats)
            state[k] = torch.cat((v[rest], v_new))


@dataclass
class IDHFRStrategy(IVDSplatBaseStrategy):
    """
    https://arxiv.org/pdf/2508.12313
    """

    CONFIG_SERIALIZATION_IGNORED_FIELDS: typing.ClassVar[set[str]] = {
        "verbose",
    }

    preprune_iter = 300
    preprune_opa = 0.02
    prune_opa: float = 0.005
    reset_opa: float = 0.05
    rap_opa_quantile: float = 0.2
    rap_opa_quantile_iter_offset: int = 300
    rap_opa_quantile_stop_iter: int = 9000

    grow_grad2d: float = 0.0003
    grow_scale3d: float = 0.01
    grow_scale2d: float = 0.05
    prune_scale3d: float = 0.1
    prune_scale2d: float = 0.15
    refine_scale2d_stop_iter: int = 0
    refine_start_iter: int = 500
    refine_stop_iter: int = 15_000
    reset_every: int = 3000
    refine_every: int = 100
    pause_refine_after_reset: int = 0
    absgrad: bool = True
    revised_opacity: bool = False
    verbose: bool = False
    key_for_gradient: Literal["means2d", "gradient_2dgs"] = "means2d"
    cap_max: int = 3_000_000  # Maximum number of GSs allowed.
    split_dist_scale = 0.45
    opacity_reduction = 0.6
    num_views_for_edge_score = 10

    def get_cap_max(self):
        if self.cap_max == -1:
            return None
        return self.cap_max

    def initialize_state(self, scene_scale: float, dataset: Dataset) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy.

        The returned state should be passed to the `step_pre_backward()` and
        `step_post_backward()` functions.
        """
        # Postpone the initialization of the state to the first step so that we can
        # put them on the correct device.
        # - grad2d: running accum of the norm of the image plane gradients for each GS.
        # - count: running accum of how many time each GS is visible.
        # - radii: the radii of the GSs (normalized by the image resolution).
        state: Dict[str, Any] = {
            "grad2d": None,
            "count": None,
            "scene_scale": scene_scale,
            "edge_activations": [],
            "dataset": dataset,
        }
        if self.refine_scale2d_stop_iter > 0:
            state["radii"] = None

        edge_activations: list[torch.Tensor] = state["edge_activations"]
        for i in range(len(dataset)):
            image: torch.Tensor = dataset[i]["image"]
            # Permute since get_edges expects image to be C, H, W
            edges_loss = get_edges(image.permute(2, 0, 1)).to(image.device)
            edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (
                torch.max(edges_loss) - torch.min(edges_loss)
            )
            edge_activations.append(edges_loss_norm.cpu())

        return state

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

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
        self,
        args: IVDSplatBaseStrategy.StepPostBackwardArgs,
    ):
        """Callback function to be executed after the `loss.backward()` call."""

        step, params, optimizers, state, info, last_rasterization_args, packed = (
            args.step,
            args.params,
            args.optimizers,
            args.state,
            args.info,
            args.last_rasterization_args,
            args.packed,
        )
        if step >= self.refine_stop_iter:
            return

        self._update_state(params, state, info, packed=packed)

        if step == self.preprune_iter:
            n_pruned = self._prune_gs_opacity(
                self.preprune_opa, params, optimizers, state
            )
            if self.verbose:
                print(
                    f"Pre-pruning at step {step}: {n_pruned} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_split = self._grow_gs(
                params, optimizers, state, step, last_rasterization_args
            )
            if self.verbose:
                print(
                    f"Step {step}: {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs_opacity(self.prune_opa, params, optimizers, state)
            if self.verbose:
                print(
                    f"Step {step}: {n_prune} GSs pruned. "
                    f"Now having {len(params['means'])} GSs."
                )

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0 & step > 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

        if (
            step % self.reset_every == self.rap_opa_quantile_iter_offset
            and step < self.rap_opa_quantile_stop_iter
        ):
            self._prune_gs_opacity_below_quantile(
                self.rap_opa_quantile, params, optimizers, state
            )

    def should_step_optimizers(self, step: int) -> bool:
        """Whether to update the parameters in this step."""
        if step <= self.refine_stop_iter:
            return True
        elif step <= 22500:  # TODO: magic numbers
            return step % 5 == 0
        else:
            return step % 20 == 0

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
        if self.refine_scale2d_stop_iter > 0 and state["radii"] is None:
            assert "radii" in info, "radii is required but missing."
            state["radii"] = torch.zeros(n_gaussian, device=grads.device)

        # update the running state
        if packed:
            # grads is [nnz, 2]
            gs_ids = info["gaussian_ids"]  # [nnz]
            radii = info["radii"].max(dim=-1).values  # [nnz]
        else:
            # grads is [C, N, 2]
            sel = (info["radii"] > 0.0).all(dim=-1)  # [C, N]
            gs_ids = torch.where(sel)[1]  # [nnz]
            grads = grads[sel]  # [nnz, 2]
            radii = info["radii"][sel].max(dim=-1).values  # [nnz]
        state["grad2d"].index_add_(0, gs_ids, grads.norm(dim=-1))
        state["count"].index_add_(
            0, gs_ids, torch.ones_like(gs_ids, dtype=torch.float32)
        )
        if self.refine_scale2d_stop_iter > 0:
            # Should be ideally using scatter max
            state["radii"][gs_ids] = torch.maximum(
                state["radii"][gs_ids],
                # normalize radii to [0, 1] screen space
                radii / float(max(info["width"], info["height"])),
            )

    @torch.no_grad()
    def _grow_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        last_rasterization_args: Dict[str, Any],
    ) -> int:
        count = state["count"]
        grads = state["grad2d"] / count.clamp_min(1)
        device = grads.device

        curr_num_splats = len(params["means"])

        assert self.cap_max > 0, "cap_max should be a positive integer."

        last_densification_stage = step > 14500

        grow_grad2d = (
            self.grow_grad2d / 1.5 if last_densification_stage else self.grow_grad2d
        )

        is_grad_high = grads > grow_grad2d
        startI = self.refine_start_iter
        endI = self.refine_stop_iter - 500  # Why 500?
        rate = (step - startI) / (endI - startI)
        if rate >= 1:
            cap_max = int(self.cap_max)
        else:
            cap_max = int(math.sqrt(rate) * self.cap_max)

        num_splats_to_add = min(
            max(cap_max - curr_num_splats, 0), is_grad_high.sum().item()
        )

        # Avoid computing edge scores if we can
        if num_splats_to_add == 0 or is_grad_high.sum().item() == 0:
            return 0

        if last_densification_stage:
            scores = grads
        else:
            scores = self._get_edge_scores(
                state,
                params,
                last_rasterization_args,
                device=params["means"].device,
                step=step,
            )
        scores[~is_grad_high] = 0.0

        edge_aware_split_votes = (scores > 0).sum().item()
        if num_splats_to_add > edge_aware_split_votes:
            num_splats_to_add = edge_aware_split_votes

        sampled_indices = torch.multinomial(
            scores.squeeze(), num_splats_to_add, replacement=False
        )
        mask = torch.zeros(curr_num_splats, dtype=torch.bool, device=device)
        mask[sampled_indices] = True

        long_axis_split(
            params=params,
            optimizers=optimizers,
            state=state,
            mask=mask,
            split_dist_scale=self.split_dist_scale,
            opacity_reduction=self.opacity_reduction,
        )

        return num_splats_to_add

    @torch.no_grad()
    def _get_edge_scores(
        self,
        state: Dict[str, Any],
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        last_rasterization_args: Dict,
        device,
        step: int,
    ):
        num_points = len(params["means"])
        gaussian_importance = torch.zeros(
            num_points, device=device, dtype=torch.float32
        )

        total_num_cams = len(state["edge_activations"])

        # TODO Magic numbers...
        if step % 3000 == 400 and step < 9000:
            num_views = total_num_cams
        else:
            num_views = self.num_views_for_edge_score
        view_indices = torch.randperm(total_num_cams)[:num_views]

        for i in range(num_views):
            data = state["dataset"][view_indices[i]]

            cam_to_worlds = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            height, width = data["image"].shape[:2]  # [H, W, C]

            last_rasterization_args["viewmats"] = torch.linalg.inv(
                cam_to_worlds
            ).reshape(1, 4, 4)
            last_rasterization_args["Ks"] = Ks.reshape(1, 3, 3)
            last_rasterization_args["height"] = height
            last_rasterization_args["width"] = width

            weights_per_g, visibility_filter = (
                rasterization_inria_wrapper_accum_weights(
                    **last_rasterization_args,
                    pixel_weights=state["edge_activations"][view_indices[i]],
                )
            )
            weights_per_g = normalize(weights_per_g)
            visibility_filter = visibility_filter.detach()

            gaussian_importance[visibility_filter] += (
                weights_per_g[visibility_filter] / num_views
            )
        # export_splat_ply(
        #     "tmp.ply",
        #     SplatData(
        #         means=params["means"].detach().cpu(),
        #         scales=params["scales"].detach().cpu(),
        #         quats=params["quats"].detach().cpu(),
        #         opacities=params["opacities"].detach().cpu(),
        #         sh0=rgb_to_sh(gaussian_importance.unsqueeze(-1).repeat(1, 3).cpu()),
        #         shN=torch.zeros((len(params["means"]), (3 + 1) ** 2 - 1, 3)),
        #     ),
        # )
        return gaussian_importance

    @torch.no_grad()
    def _prune_gs_opacity(
        self,
        prune_opa: float,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < prune_opa
        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return int(n_prune)

    @torch.no_grad()
    def _prune_gs_opacity_below_quantile(
        self,
        prune_opacity_quantile: float,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
    ) -> int:
        opacity_array = torch.sigmoid(params["opacities"].flatten())
        min_opacity = torch.quantile(opacity_array, prune_opacity_quantile)
        is_prune = opacity_array < min_opacity
        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return int(n_prune)
