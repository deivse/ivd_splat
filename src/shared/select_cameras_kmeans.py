import logging

import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans, vq

_LOGGER = logging.getLogger(__name__)


def select_cameras_kmeans(cameras, K):
    """
    Selects K cameras from a set using K-means clustering.

    Args:
        cameras: (N, 16), representing N cameras with flattened 4x4 homogeneous matrices.
        K: Number of cameras/clusters to select.

    Returns:
        selected_indices: List of indices of the cameras closest to the cluster centers.
    """
    if cameras.shape[1] != 16:
        raise ValueError(
            "Each camera must have 16 values corresponding to a flattened 4x4 matrix."
        )

    for _ in range(10):  # Try multiple times to avoid less than K unique clusters
        cluster_centers, _ = kmeans(cameras, K)
        if len(cluster_centers) == K:
            break

    if len(cluster_centers) != K:
        _LOGGER.warning(
            f"K-means converged to {len(cluster_centers)} unique clusters instead of {K}. Adjusting K to {len(cluster_centers)}."
        )

    cluster_assignments, _ = vq(cameras, cluster_centers)

    selected_indices = []
    for k in range(min(K, len(cluster_centers))):
        cluster_members = cameras[cluster_assignments == k]
        distances = cdist([cluster_centers[k]], cluster_members)[0]
        nearest_camera_idx = np.where(cluster_assignments == k)[0][np.argmin(distances)]
        selected_indices.append(nearest_camera_idx)

    return selected_indices
