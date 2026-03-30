import math
from typing import Optional, Tuple
import warnings

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError:
    warnings.warn("diff_gaussian_rasterization not found, IDHFRStrategy will not work.")

from gsplat.utils import get_projection_matrix


@torch.no_grad()
def rasterization_inria_wrapper_accum_weights(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., N, D] or [..., N, K, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    pixel_weights: Tensor,
    near_plane: float = 0.01,
    far_plane: float = 100.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    backgrounds: Optional[Tensor] = None,
    **kwargs,
) -> Tuple[Tensor, Tensor]:
    """Wrapper for Inria's rasterization backend.

    .. warning::
        This function exists for comparison purpose only. Only rendered image is
        returned.

    .. warning::
        Inria's CUDA backend has its own LICENSE, so this function should be used with
        the respect to the original LICENSE at:
        https://github.com/graphdeco-inria/diff-gaussian-rasterization

    """

    assert eps2d == 0.3, "This is hard-coded in CUDA to be 0.3"
    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    N = means.shape[-2]
    B = math.prod(batch_dims)
    C = viewmats.shape[-3]
    device = means.device
    channels = colors.shape[-1]

    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape

    assert B == C and C == 1, "Only batch size of 1 and 1 view supported for now"
    # flatten all batch dimensions
    means = means.reshape(N, 3)
    quats = quats.reshape(N, 4)
    scales = scales.reshape(N, 3)
    opacities = opacities.reshape(N)
    viewmats = viewmats.reshape(4, 4)
    Ks = Ks.reshape(3, 3)
    if colors.dim() == num_batch_dims + 2:
        colors = colors.reshape(N, -1)
    elif colors.dim() == num_batch_dims + 3:
        colors = colors.reshape(C, N, -1)

    # rasterization from inria does not do normalization internally
    quats = F.normalize(quats, dim=-1)  # [N, 4]

    FoVx = 2 * math.atan(width / (2 * Ks[0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * Ks[1, 1].item()))
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    world_view_transform = viewmats.transpose(0, 1)
    projection_matrix = get_projection_matrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device
    ).transpose(0, 1)
    full_proj_transform = (
        world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
    ).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=torch.zeros(3, dtype=torch.float32, device=device),
        scale_modifier=1.0,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False,
        pixel_weights=pixel_weights.to(means.device),
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means2D = torch.zeros((N, 3), dtype=means.dtype, requires_grad=True, device=device)
    (
        rendered_image,
        radii,
        counts,
        lists,
        listsRender,
        listsDistance,
        centers,
        depths,
        my_radii,
        accum_weights,
        accum_count,
        accum_blend,
        accum_dist,
    ) = rasterizer(
        means3D=means,
        means2D=means2D,
        colors_precomp=torch.zeros(
            [N, channels],
            device=device,
        ),
        opacities=opacities[..., None],
        scales=scales,
        rotations=quats,
        cov3D_precomp=None,
    )
    return accum_weights, (radii > 0).nonzero()
