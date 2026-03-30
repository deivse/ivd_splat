import numpy as np

def align_depth_ransac(
    depth: np.ndarray,
    gt_depth: np.ndarray,
    use_msac_loss: bool,
    sample_size: int,
    max_iters: int,
    inlier_threshold: float,
    confidence: float,
    min_iters: int,
) -> tuple[float, float, int, int, int]: ...

"""
Returns: A tuple containing:
    - scale (float): The estimated scale factor.
    - shift (float): The estimated shift value.
    - num_inliers_best_lo (int): The best number of inliers after local optimization.
    - num_inliers_best_pre_lo (int): The best number of inliers without local optimization.
    - iteration (int): The number of iterations performed.
"""
