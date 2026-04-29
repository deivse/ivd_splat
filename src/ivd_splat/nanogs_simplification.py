from typing import Dict, Union

import numpy as np
import torch
from gsplat.strategy.ops import _update_param_with_optimizer, remove
from gsplat.exporter import export_splats

from ivd_splat.config import NanoGSConfig

from nanogs.simplification import (
    CostParams,
    edge_costs,
    knn_indices,
    knn_undirected_edges,
    merge_pairs,
)
import nanogs.utils.splat_utils as nanogs_splat_utils


def _greedy_pairs_from_edges_threshold(
    edges: np.ndarray,  # (M,2) int32, u<v
    w: np.ndarray,  # (M,) float32 costs
    N: int,
    w_thresh: float,
) -> np.ndarray:
    """
    Sort all edges by weight and greedily pick disjoint pairs below a weight threshold.
    License notice: based on NanoGS code licensed under CC Attribution-NonCommercial 4.0 International (https://github.com/saliteta/NanoGS)
    """
    if edges.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int32)

    # filter invalid costs if any
    valid = np.isfinite(w)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32)

    idx = np.nonzero(valid)[0]
    order = idx[np.argsort(w[idx], kind="mergesort")]  # stable

    used = np.zeros(N, dtype=bool)
    pairs = []
    for ei in order:
        if w[ei] > w_thresh:
            break
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        if used[u] or used[v]:
            continue
        used[u] = True
        used[v] = True
        pairs.append((u, v))

    if not pairs:
        return np.zeros((0, 2), dtype=np.int32)
    return np.asarray(pairs, dtype=np.int32)


@torch.no_grad()
def merge_gs_gsplat(
    params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    optimizers: Dict[str, torch.optim.Optimizer],
    new_param_dict: Dict[str, torch.Tensor],
    pairs: torch.Tensor,  # (P, 2) long tensor of indices to merge, where pairs[:,0] and pairs[:,1] are the indices of the two GSs to merge
    state: Dict[str, torch.Tensor],
):
    """Replaces the parameters of the GSs in pairs[:,0] with new_param_dict, and removes the GSs in pairs[:,1].
    Args:
        params: A dictionary of parameters.
        optimizers: A dictionary of optimizers, each corresponding to a parameter.
        mask: A boolean mask to duplicate the Gaussians.
    """
    device = next(iter(params.values())).device

    to_overwrite = pairs[:, 0]
    to_remove = pairs[:, 1]
    to_keep = torch.ones(params["means"].shape[0], dtype=torch.bool, device=device)
    to_keep[to_remove] = False

    def param_fn(name: str, p: torch.Tensor) -> torch.Tensor:
        new_values = p.clone()
        new_values[to_overwrite] = new_param_dict[name]
        return torch.nn.Parameter(new_values[to_keep], requires_grad=p.requires_grad)

    def optimizer_fn(key: str, v: torch.Tensor) -> torch.Tensor:
        new_values = v.clone()
        new_values[to_overwrite] = torch.zeros(
            (len(to_overwrite), *v.shape[1:]), device=device
        )
        return new_values[to_keep]

    # update the parameters and the state in the optimizers
    _update_param_with_optimizer(param_fn, optimizer_fn, params, optimizers)
    # update the extra running state

    # We keep strategy state for Gs in pairs[:,0] intact, which is not strictly correct,
    # but is ok for now since we run this before any densification steps.

    # TODO: would be good to refactor the strategies so they can specify which state entries
    # should be updated and how, and specify merging/default construction of their
    # strategy state entries
    for k, v in state.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == len(to_keep):
            state[k] = v[to_keep]


# Exists due to issue with strategy state management as described above...
# TODO: refactor strategy state management and remove this hacky function
# This is also not always correct since for a non-per-gaussian tensor that happens to have
# as many values as there is Gaussians, we will incorrectly prune it.
def _remove_strategy_state_safe(strat_state, to_remove):
    # to_remove is a boolean mask of which Gs to remove
    keep_mask = ~to_remove
    for k, v in strat_state.items():
        if isinstance(v, torch.Tensor) and v.shape[0] == len(to_remove):
            strat_state[k] = v[keep_mask]


def _nanogs_simplify_impl(
    splat_params: torch.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    strategy_state: dict,
    config: NanoGSConfig,
) -> None:
    """
    NanoGS simplification adjusted to run during gsplat training and to use a fixed max edge
    cost threshold and number of iterations instead of targeting a specific number of Gaussians.

    Note the implementation is extremely suboptimal performance-wise and has some dirty hacks.

    License notice: based on NanoGS code licensed under CC Attribution-NonCommercial 4.0 International (https://github.com/saliteta/NanoGS)
    """
    cp: CostParams = CostParams()

    def load_from_splats():
        mu = splat_params["means"].detach().cpu().numpy()  # (N,3)
        sc_raw = splat_params["scales"].detach().cpu().numpy()  # (N,3)
        q_raw = splat_params["quats"].detach().cpu().numpy()  # (N,4)
        op_raw = splat_params["opacities"].detach().cpu().numpy()  # (N,)
        sh0 = splat_params["sh0"].detach().cpu().numpy()  # (N, 3)
        if "shN" in splat_params:
            shN = splat_params["shN"].detach().cpu().numpy()  # (N, :, 3)
            sh = np.concatenate([sh0, shN], axis=1)
        else:
            sh = sh0

        sh = sh.reshape(sh.shape[0], -1)

        op = nanogs_splat_utils.sigmoid(op_raw).astype(np.float32)
        sc = np.exp(np.clip(sc_raw, -30.0, 30.0)).astype(np.float32)
        q = nanogs_splat_utils.quat_normalize(q_raw).astype(np.float32)

        return mu, sc, q, op, sh

    opa_raw_tensor = splat_params["opacities"]
    threshold = min(
        nanogs_splat_utils.logit(config.preprune_opacity_threshold),
        opa_raw_tensor.median().item(),
    )
    print(
        f"Pruning splats with opacity below {nanogs_splat_utils.sigmoid(threshold):.4f}"
    )
    to_remove = opa_raw_tensor < torch.tensor(threshold, device=opa_raw_tensor.device)
    remove(
        splat_params,
        optimizers,
        state={},
        mask=to_remove,
    )
    _remove_strategy_state_safe(strategy_state, to_remove)
    print(
        f"Original count: {to_remove.shape[0]}, after opacity pruning: {splat_params['means'].shape[0]}"
    )

    print(
        f"Running NanoGS merging for up to {config.iterations} iterations.",
    )

    for i in range(config.iterations):
        mu, sc, q, op, sh = load_from_splats()
        N = int(mu.shape[0])

        print(f"Pass {i + 1}: {N} splats")

        k_eff = min(max(1, config.knn_k), max(1, N - 1))
        nbr = knn_indices(mu, k=k_eff)

        edges = knn_undirected_edges(nbr)
        w = edge_costs(edges, mu, sc, q, op, sh, cp)
        # merges_needed = N - target
        # P = min(merges_needed, p_cap) if merges_needed > 0 else None

        pairs = _greedy_pairs_from_edges_threshold(edges, w, N, config.cost_threshold)
        if pairs.shape[0] == 0:
            print("No pairs below cost threshold, stopping.")
            break

        print(f"  edges: {edges.shape[0]}, pairs: {pairs.shape[0]}")

        to_keep = np.ones(N, dtype=bool)
        to_keep[pairs[:, 0]] = False
        to_keep[pairs[:, 1]] = False
        num_to_keep = to_keep.sum()

        mu, sc, q, op, sh = merge_pairs(mu, sc, q, op, sh, pairs)

        sc_raw = np.log(np.maximum(sc, 1e-12))
        op_raw = nanogs_splat_utils.logit(op)

        new_param_dict = {
            "means": torch.from_numpy(mu[num_to_keep:]).to(splat_params["means"]),
            "scales": torch.from_numpy(sc_raw[num_to_keep:]).to(splat_params["scales"]),
            "quats": torch.from_numpy(q[num_to_keep:]).to(splat_params["quats"]),
            "opacities": torch.from_numpy(op_raw[num_to_keep:]).to(
                splat_params["opacities"]
            ),
            "sh0": torch.from_numpy(sh[num_to_keep:, :3])
            .to(splat_params["sh0"])
            .reshape(-1, 1, 3),
        }
        if "shN" in splat_params:
            new_param_dict["shN"] = (
                torch.from_numpy(sh[num_to_keep:, 3:])
                .to(splat_params["shN"])
                .reshape(-1, *splat_params["shN"].shape[1:])
            )

        merge_gs_gsplat(
            splat_params,
            optimizers,
            new_param_dict,
            pairs=torch.from_numpy(pairs).to(splat_params["means"].device),
            state=strategy_state,
        )

    print(f"Num splats after NanoGS simplification: {splat_params['means'].shape[0]}")


@torch.no_grad()
def nanogs_simplify(
    splat_params: torch.ParameterDict,
    optimizers: dict[str, torch.optim.Optimizer],
    strategy_state: dict,
    config: NanoGSConfig = NanoGSConfig(),
):
    export_splats(
        means=splat_params["means"],
        scales=splat_params["scales"],
        quats=splat_params["quats"],
        opacities=splat_params["opacities"],
        sh0=splat_params["sh0"],
        shN=splat_params["shN"],
        format="ply",
        save_to="splats_before_nanogs.ply",
    )

    _nanogs_simplify_impl(splat_params, optimizers, strategy_state, config)

    export_splats(
        means=splat_params["means"],
        scales=splat_params["scales"],
        quats=splat_params["quats"],
        opacities=splat_params["opacities"],
        sh0=splat_params["sh0"],
        shN=splat_params["shN"],
        format="ply",
        save_to="splats_after_nanogs.ply",
    )
