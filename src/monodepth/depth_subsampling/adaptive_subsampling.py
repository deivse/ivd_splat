from dataclasses import dataclass
from typing import Sequence, Tuple, Union
import torch
from shared.image_filtering import (
    gaussian_filter2d,
    spatial_gradient_first_order,
)
from monodepth.depth_subsampling.config import AdaptiveSubsamplingConfig
from monodepth.depth_subsampling.interface import DepthSubsampler


def _map_to_range(tensor: torch.Tensor, output_range=(0.0, 1.0), input_range=None):
    if input_range is None:
        input_range = (tensor.min(), tensor.max())
    tensor = tensor - input_range[0]
    tensor /= input_range[1] - input_range[0]
    return (output_range[1] - output_range[0]) * tensor + output_range[0]


def _map_to_closest_discrete_value(data: torch.Tensor, values: Sequence[int | float]):
    range = torch.tensor(values, device=data.device)
    dists = torch.abs(data[:, :, None] - range[None, None, :])
    return range[torch.argmin(torch.abs(dists), dim=-1)]


def _color_grad_intensity_map(
    rgb: torch.Tensor,
    grad_approx_gauss_sigma: float = 1.2,
) -> torch.Tensor:
    """
    Args:
        `rgb`                        input RGB image `[H, W, 3]`
        `possible_subsample_factors` range of possible subsample factors in increasing order
        `grad_approx_gauss_sigma`    sigma for gaussian kernel used to approximate
                                     gradient of the image
    """
    color_grad = (
        spatial_gradient_first_order(
            rgb.permute(2, 0, 1)[None], sigma=grad_approx_gauss_sigma
        )
        .sum(1)
        .sum(1)
    )
    intensity_0_to_1 = _map_to_range(color_grad.abs())
    return gaussian_filter2d(intensity_0_to_1[None], 5).squeeze()


def get_sample_mask(
    downsample_factor_map: torch.Tensor,
    image_size: Union[torch.Size, Tuple[int, int]],
) -> torch.Tensor:
    """
    Generates a tensor of boolean values indicating which pixel indices should be sampled
    based on the provided downsample factor map and the desired image size.
    Args:
        downsample_factor_map (torch.Tensor): A tensor representing the downsample factors
            for each pixel in the original image.
        image_size (Union[torch.Size, Tuple[int, int]]): The size of the image to which the
            downsample factor map should be interpolated.
    Returns:
        torch.Tensor: A boolean 1D tensor of length width * hight which can be used to index, e.g. img.view(-1, 3)
    """
    per_pixel_df: torch.Tensor = (
        torch.nn.functional.interpolate(
            downsample_factor_map[None, None].to(torch.float), size=image_size, mode="nearest"
        )
        .squeeze()
        .to(torch.int)
    )
    pixel_coords = torch.cartesian_prod(
        torch.arange(per_pixel_df.shape[0], device=per_pixel_df.device),
        torch.arange(per_pixel_df.shape[1], device=per_pixel_df.device),
    )

    per_pixel_df[per_pixel_df == 0] = 1
    return torch.logical_and(
        (pixel_coords[:, 0] % per_pixel_df.view(-1)) == 0,
        (pixel_coords[:, 1] % per_pixel_df.view(-1)) == 0,
    )


def iqr_outlier_bounds(data: torch.Tensor):
    q1 = torch.quantile(data, 0.25)
    q3 = torch.quantile(data, 0.75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def get_depth_multipler_map(depth: torch.Tensor, mask: torch.Tensor):
    masked_depth = depth[mask]
    outlier_bounds = iqr_outlier_bounds(masked_depth)
    input_range = (
        max(masked_depth.min(), outlier_bounds[0]),
        min(masked_depth.max(), outlier_bounds[1]),
    )
    multiplier_map = torch.clamp(_map_to_range(depth, input_range=input_range), 0, 1)
    multiplier_map[~mask] = 0.5
    return 1.0 - multiplier_map


@dataclass
class AdaptiveDepthSubsampler(DepthSubsampler):
    config: AdaptiveSubsamplingConfig

    def get_mask(self, rgb, depth, mask):
        multiplier_map = get_depth_multipler_map(depth, mask)
        factor_map = torch.clamp(
            _map_to_range(
                multiplier_map,
                output_range=(
                    self.config.factor_range_min,
                    self.config.factor_range_max,
                ),
                input_range=(0.0, 1.0),
            ),
            self.config.factor_range_min,
            self.config.factor_range_max,
        )
        return torch.logical_and(
            get_sample_mask(factor_map.to(int), rgb.shape[:2]),
            mask.view(-1),
        )
