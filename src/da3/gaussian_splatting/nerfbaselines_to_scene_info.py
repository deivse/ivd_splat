import dataclasses
import warnings
import logging
import os
import numpy as np
from PIL import Image
from nerfbaselines import (
    camera_model_to_int,
    Dataset,
)

import torch
import edgs.gaussian_splatting.camera_utils as camera_utils
from edgs.gaussian_splatting.dataset_readers import (
    CameraInfo,
    SceneInfo,
    fetchPly,
    getNerfppNorm,
    storePly,
)
from edgs.gaussian_splatting.graphics_utils import BasicPointCloud, focal2fov
from edgs.sh_utils import SH2RGB


# NOTE: MODIFICATIONS HAVE BEEN MADE TO ACCOMODATE SIMPLE INTEGRATION OF EDGS INTO OUR CODEBASE.


def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {
            f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)
        }
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator).items())
        else:
            flat[k] = v
    return flat


def _load_caminfo(
    idx,
    pose,
    intrinsics,
    image_name,
    image_size,
    image=None,
    image_path=None,
    mask=None,
    scale_coords=None,
):
    pose = np.copy(pose)
    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
    pose = np.linalg.inv(pose)
    R = pose[:3, :3]
    T = pose[:3, 3]
    if scale_coords is not None:
        T = T * scale_coords
    R = np.transpose(R)

    width, height = image_size
    fx, fy, cx, cy = intrinsics
    if image is None:
        image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    return CameraInfo(
        uid=idx,
        R=R,
        T=T,
        FovX=focal2fov(float(fx), float(width)),
        FovY=focal2fov(float(fy), float(height)),
        depth_params=None,
        image=image,
        image_path=image_path,
        image_name=image_name,
        width=int(width),
        height=int(height),
        mask=mask,
        depth_path="",
        is_test=False,  # dirty hack but this is only ever used for init with EDGS so it doesn't matter.
        cx=cx,
        cy=cy,
    )


def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
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


def nerfbaselines_dataset_to_3dgs_scene_info(
    dataset: Dataset,
    tempdir: str,
    white_background: bool = False,
    add_dataset_points_to_scene: bool = True,
    scale_coords=None,
) -> SceneInfo:

    if not np.all(dataset["cameras"].camera_models == camera_model_to_int("pinhole")):
        raise ValueError("Only pinhole cameras supported")

    cam_infos = []
    for idx, extr in enumerate(dataset["cameras"].poses):
        del extr
        intrinsics = dataset["cameras"].intrinsics[idx]
        pose = dataset["cameras"].poses[idx]
        image_path = (
            dataset["image_paths"][idx]
            if dataset["image_paths"] is not None
            else f"{idx:06d}.png"
        )
        image_name = (
            os.path.relpath(
                str(dataset["image_paths"][idx]), str(dataset["image_paths_root"])
            )
            if dataset["image_paths"] is not None
            and dataset["image_paths_root"] is not None
            else os.path.basename(image_path)
        )

        w, h = dataset["cameras"].image_sizes[idx]
        im_data = dataset["images"][idx][:h, :w]
        assert im_data.dtype == np.uint8, "Gaussian Splatting supports images as uint8"
        if im_data.shape[-1] == 4:
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = (
                norm_data[:, :, :3] * norm_data[:, :, 3:4]
                + (1 - norm_data[:, :, 3:4]) * bg
            )
            im_data = np.array(arr * 255.0, dtype=np.uint8)
        if not white_background and dataset["metadata"].get("id") == "blender":
            warnings.warn(
                "Blender scenes are expected to have white background. If the background is not white, please set white_background=True in the dataset loader."
            )
        elif white_background and dataset["metadata"].get("id") != "blender":
            warnings.warn(
                "white_background=True is set, but the dataset is not a blender scene. The background may not be white."
            )
        image = Image.fromarray(im_data)
        del im_data

        mask = None
        if dataset["masks"] is not None:
            mask = Image.fromarray((dataset["masks"][idx] * 255).astype(np.uint8))

        cam_info = _load_caminfo(
            idx,
            pose,
            intrinsics,
            image_name=image_name,
            image_path=image_path,
            image_size=(w, h),
            image=image,
            mask=mask,
            scale_coords=scale_coords,
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    if add_dataset_points_to_scene:
        points3D_xyz = dataset["points3D_xyz"]
        if scale_coords is not None:
            points3D_xyz = points3D_xyz * scale_coords
        points3D_rgb = dataset["points3D_rgb"]
        if points3D_xyz is None and dataset["metadata"].get("id", None) == "blender":
            # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L221C4-L221C4
            num_pts = 100_000
            logging.info(f"generating random point cloud ({num_pts})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            points3D_xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
            shs = np.random.random((num_pts, 3)) / 255.0
            points3D_rgb = (SH2RGB(shs) * 255).astype(np.uint8)

        ply_path = os.path.join(tempdir, "scene.ply")
        storePly(ply_path, points3D_xyz, points3D_rgb)
        pcd = fetchPly(ply_path)
    else:
        pcd = BasicPointCloud(points=np.zeros((0, 3)), colors=np.zeros((0, 3)), normals=np.zeros((0, 3)))
        ply_path = ""
        
    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=cam_infos,
        test_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
