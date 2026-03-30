from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
from monodepth.depth_alignment.interface import DepthAlignmentResult
from monodepth.depth_prediction.interface import PredictedDepth


def debug_export_alignment(
    predicted_depth: PredictedDepth,
    result: DepthAlignmentResult,
    debug_export_dir: Path,
):
    debug_export_dir.mkdir(parents=True, exist_ok=True)

    # Get depth values for visualization
    initial_depth = predicted_depth.depth.cpu().numpy()
    aligned_depth = result.aligned_depth.cpu().numpy()
    initial_mask = predicted_depth.mask.cpu().numpy()
    result_mask = result.mask.cpu().numpy()

    # Determine common depth range for consistent colormap
    valid_initial = initial_depth[initial_mask]
    valid_aligned = aligned_depth[result_mask]
    if len(valid_initial) > 0 and len(valid_aligned) > 0:
        depth_min = min(np.min(valid_initial), np.min(valid_aligned))
        depth_max = max(np.max(valid_initial), np.max(valid_aligned))
    else:
        depth_min, depth_max = 0, 1

    # Create visualization for initial depth
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(initial_depth, cmap="viridis", vmin=depth_min, vmax=depth_max)
    # Dim masked pixels by overlaying semi-transparent black
    masked_overlay = np.zeros((*initial_depth.shape, 4))
    masked_overlay[~initial_mask] = [0, 0, 0, 0.5]
    ax.imshow(masked_overlay)
    plt.colorbar(im, ax=ax)
    ax.set_title("Initial Predicted Depth")
    plt.savefig(debug_export_dir / "initial_depth.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Create visualization for aligned depth
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(aligned_depth, cmap="viridis", vmin=depth_min, vmax=depth_max)
    # Dim masked pixels
    masked_overlay = np.zeros((*aligned_depth.shape, 4))
    masked_overlay[~result_mask] = [0, 0, 0, 0.5]
    ax.imshow(masked_overlay)
    plt.colorbar(im, ax=ax)
    ax.set_title("Aligned Depth")
    plt.savefig(debug_export_dir / "aligned_depth.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Create difference heatmap
    depth_diff = aligned_depth - initial_depth
    # Only show differences where both masks are valid
    valid_diff_mask = initial_mask & result_mask
    depth_diff[~valid_diff_mask] = 0

    # Determine symmetric range for diverging colormap
    valid_diff = depth_diff[valid_diff_mask]
    if len(valid_diff) > 0:
        max_abs_diff = np.max(np.abs(valid_diff))
        vmin, vmax = -max_abs_diff, max_abs_diff
    else:
        vmin, vmax = -1, 1

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(depth_diff, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    # Dim invalid pixels
    masked_overlay = np.zeros((*depth_diff.shape, 4))
    masked_overlay[~valid_diff_mask] = [0, 0, 0, 0.7]
    ax.imshow(masked_overlay)
    plt.colorbar(im, ax=ax)
    ax.set_title("Depth Difference (Aligned - Initial)")
    plt.savefig(debug_export_dir / "depth_difference.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Create multiplicative factor heatmap
    mult_factor = np.ones_like(aligned_depth)
    # Only compute factors where both depths are valid and non-zero
    valid_mult_mask = valid_diff_mask & (initial_depth != 0)
    mult_factor[valid_mult_mask] = (
        aligned_depth[valid_mult_mask] / initial_depth[valid_mult_mask]
    )
    mult_factor[~valid_mult_mask] = 1.0  # Set invalid pixels to neutral factor

    # Determine symmetric range for diverging colormap centered at 1
    valid_mult = mult_factor[valid_mult_mask]
    if len(valid_mult) > 0:
        max_mult = np.max(valid_mult)
        min_mult = np.min(valid_mult)
        # Symmetric range around 1
        max_deviation = max(abs(max_mult - 1), abs(min_mult - 1))
        vmin, vmax = 1 - max_deviation, 1 + max_deviation
    else:
        vmin, vmax = 0.5, 1.5

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mult_factor, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    # Dim invalid pixels
    masked_overlay = np.zeros((*mult_factor.shape, 4))
    masked_overlay[~valid_mult_mask] = [0, 0, 0, 0.7]
    ax.imshow(masked_overlay)
    plt.colorbar(im, ax=ax, label="Multiplicative Factor")
    ax.set_title("Depth Multiplication Factor (Aligned / Initial)")
    plt.savefig(
        debug_export_dir / "depth_mult_factor.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
