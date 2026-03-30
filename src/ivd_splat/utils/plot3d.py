from matplotlib import pyplot as plt
import numpy as np


def plot3d(xyz: np.ndarray, color="b", ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    coords = xyz.reshape(-1, 3)

    ax.scatter(
        coords[:, 0].flatten(),
        coords[:, 1].flatten(),
        coords[:, 2].flatten(),
        s=1,
        c=color,
    )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])
