import abc

import torch


class DepthSubsampler(abc.ABC):
    def get_mask(
        self, rgb: torch.Tensor, depth: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            `rgb`   input RGB image `[H, W, 3]`
            `depth` input depth map `[H, W]`
            `mask`  input boolean mask `[H, W]` indicating valid pixels
        Returns:
            Boolean sampling mask of same shape as flattened depth - [H * W].
            ! Pixels where `mask` is False will also be False.
        """
