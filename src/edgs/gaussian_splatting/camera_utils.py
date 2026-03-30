#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# NOTE: MODIFICATIONS HAVE BEEN MADE TO ACCOMODATE SIMPLE INTEGRATION OF EDGS INTO OUR CODEBASE.

import torch
from torch.types import Device as TorchDevice
from edgs.gaussian_splatting.camera import Camera
from edgs.gaussian_splatting.dataset_readers import CameraInfo
from edgs.gaussian_splatting.general_utils import PILtoTorch
from edgs.gaussian_splatting.graphics_utils import fov2focal


WARNED = False


def _loadCam_original(
    id: int,
    cam_info: CameraInfo,
    device: TorchDevice,
    is_test_dataset: bool,
):

    if cam_info.depth_path != "":
        raise NotImplementedError(
            "Depth map support removed from this version for simplicity, as not used by EDGS."
        )

    return Camera(
        cam_info.image.size,
        colmap_id=cam_info.uid,
        R=cam_info.R,
        T=cam_info.T,
        FoVx=cam_info.FovX,
        FoVy=cam_info.FovY,
        depth_params=cam_info.depth_params,
        image=cam_info.image,
        invdepthmap=None,
        image_name=cam_info.image_name,
        uid=id,
        data_device=device,
        train_test_exp=False,
        is_test_dataset=is_test_dataset,
        is_test_view=cam_info.is_test,
    )


def _getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


# Version from nerfbaselines with some fixes to include mask and cx, cy
def loadCam(id: int, cam_info: CameraInfo, device: TorchDevice, is_test_dataset: bool):
    camera = _loadCam_original(id, cam_info, device, is_test_dataset)

    mask = None
    if cam_info.mask is not None:
        mask = PILtoTorch(cam_info.mask, (camera.image_width, camera.image_height))
    setattr(camera, "mask", mask)
    setattr(camera, "_patched", True)

    # Fix cx, cy (ignored in gaussian-splatting)
    camera.focal_x = fov2focal(cam_info.FovX, camera.image_width)
    camera.focal_y = fov2focal(cam_info.FovY, camera.image_height)
    camera.cx = cam_info.cx
    camera.cy = cam_info.cy
    camera.projection_matrix = (
        _getProjectionMatrixFromOpenCV(
            camera.image_width,
            camera.image_height,
            camera.focal_x,
            camera.focal_y,
            camera.cx,
            camera.cy,
            camera.znear,
            camera.zfar,
        )
        .transpose(0, 1)
        .cuda()
    )
    camera.full_proj_transform = (
        camera.world_view_transform.unsqueeze(0).bmm(
            camera.projection_matrix.unsqueeze(0)
        )
    ).squeeze(0)

    return camera


def cameraList_from_camInfos(
    cam_infos: list[CameraInfo],
    is_test_dataset: bool,
    device: TorchDevice,
):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(
            loadCam(
                id,
                c,
                device,
                is_test_dataset,
            )
        )

    return camera_list
