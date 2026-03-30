import numpy as np
import math
import torch
import torch.nn.functional as F


def get_gausskernel_size(sigma, force_odd=True):
    ksize = 2 * math.ceil(sigma * 3.0) + 1
    if ksize % 2 == 0 and force_odd:
        ksize += 1
    return int(ksize)


def gaussian1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Function that computes values of a (1D) Gaussian with zero mean and variance sigma^2"""
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-x * x / (2 * sigma * sigma))


def gaussian_deriv1d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Function that computes values of a (1D) Gaussian derivative"""
    return -(x / (sigma * sigma)) * gaussian1d(x, sigma)


def box_blur_kernel(ksize: int) -> torch.Tensor:
    """Function that returns a box blur kernel of given size."""
    if ksize % 2 == 0:
        raise ValueError("Box blur kernel size must be odd.")
    return torch.ones((ksize, ksize)) / (ksize * ksize)


def filter2d(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Function that convolves a tensor with a kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input (torch.Tensor): the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel (torch.Tensor): the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(kH, kW)`.
    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input.
    """

    pad_x = kernel.size()[1] // 2
    pad_y = kernel.size()[0] // 2

    padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="replicate")
    kernel = kernel.flip(0, 1)[None, None]
    kernel = kernel.repeat(x.size(1), 1, 1, 1)

    return F.conv2d(padded, kernel.to(x.device), stride=1, groups=x.size(1))


def _separable_conv(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Function that applies a separable convolution kernel to a 2D tensor."""
    return filter2d(filter2d(x, kernel[:, None]), kernel[None, :])


def _gaussian_input_range(sigma: float) -> torch.Tensor:
    ksize = get_gausskernel_size(sigma)
    ksize_half = math.floor(ksize / 2)
    return torch.arange(-ksize_half, ksize_half + 1)


def gaussian_filter2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Function that blurs a tensor using a Gaussian filter.

    Arguments:
        sigma (Tuple[float, float]): the standard deviation of the kernel.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    """
    kernel_1d = gaussian1d(_gaussian_input_range(sigma), sigma)
    return _separable_conv(x, kernel_1d)


def box_blur2d(x: torch.Tensor, ksize: int) -> torch.Tensor:
    """Function that blurs a tensor using a box blur filter.

    Arguments:
        ksize (int): the size of the kernel. Must be odd.

    Returns:
        Tensor: the blurred tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    """
    kernel_2d = box_blur_kernel(ksize)
    return filter2d(x, kernel_2d)


def spatial_gradient_first_order(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Computes the first order image derivative in both x and y directions using Gaussian derivative

    Return:
        torch.Tensor: spatial gradients

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    """
    gaussian_input = _gaussian_input_range(sigma)
    gderiv_k = gaussian_deriv1d(gaussian_input, sigma).to(x.device)
    gaussian_k = gaussian1d(gaussian_input, sigma).to(x.device)

    blurred_y = filter2d(x, gaussian_k[:, None])
    blurred_x = filter2d(x, gaussian_k[None, :])

    return torch.stack(
        [
            filter2d(blurred_y, gderiv_k[None, :]),
            filter2d(blurred_x, gderiv_k[:, None]),
        ],
        dim=2,
    )
