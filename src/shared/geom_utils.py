import torch


def project_points(
    points3D: torch.Tensor, P: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    points3D_h = torch.hstack(
        [points3D, torch.ones((points3D.shape[0], 1), device=points3D.device)]
    )
    points2D_h = points3D_h @ P.T
    points2D = points2D_h[:, :2] / points2D_h[:, 2:3]
    depth = points2D_h[:, 2]
    return points2D, depth
