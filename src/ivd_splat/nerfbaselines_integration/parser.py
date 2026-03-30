from enum import Enum
from inspect import isclass
import logging
import pprint
import dataclasses
import json
import io
import base64
import yaml
from contextlib import contextmanager
import warnings
from functools import partial
import numpy as np
import argparse
import os
import ast
from operator import attrgetter
import importlib.util
from typing import cast, Optional, List, Union
from nerfbaselines import (
    Method,
    MethodInfo,
    ModelInfo,
    Dataset,
    OptimizeEmbeddingOutput,
)
from nerfbaselines.utils import pad_poses, image_to_srgb, convert_image_dtype
from typing_extensions import Literal, get_origin, get_args

import torch  # type: ignore
from torch.nn import functional as F


def numpy_to_base64(array: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, array)
        return base64.b64encode(f.getvalue()).decode("ascii")


def numpy_from_base64(data: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(data)) as f:
        return np.load(f)


class NerfbaselinesParser:
    def __init__(
        self,
        data_dir: str,
        factor: int = 1,
        normalize: bool = False,
        test_every: int = 8,
        state=None,
        dataset: Optional[Dataset] = None,
    ):
        assert factor == 1, "Factor must be 1"
        del test_every, data_dir

        if state is not None:
            self.transform = numpy_from_base64(state["transform_base64"])
            self.scene_scale = state["scene_scale"]
            self.points = self.points_rgb = None
            if state["num_points"] is not None:
                self.points = np.zeros((state["num_points"], 3), dtype=np.float32)
                self.points_rgb = np.zeros((state["num_points"], 3), dtype=np.uint8)
            self.num_train_images = state["num_train_images"]
            self.dataset = None
            return

        assert dataset is not None, "Dataset must be provided"
        self.num_train_images = len(dataset.get("images"))

        # Optional normalize
        from ivd_splat.datasets.normalize import (  # type: ignore
            similarity_from_cameras,
            transform_cameras,
            transform_points,
            align_principle_axes,
        )

        if normalize:
            points = dataset.get("points3D_xyz")
            camtoworlds = pad_poses(dataset.get("cameras").poses)
            transform = T1 = similarity_from_cameras(camtoworlds)
            camtoworlds = transform_cameras(T1, camtoworlds)
            if points is not None:
                points = transform_points(T1, points)

                T2 = align_principle_axes(points)
                camtoworlds = transform_cameras(T2, camtoworlds)
                points = transform_points(T2, points)

                transform = cast(np.ndarray, T2 @ T1)

            # Apply transform to the dataset
            dataset = dataset.copy()
            dataset["cameras"] = dataset["cameras"].replace(
                poses=camtoworlds[..., :3, :4]
            )
            dataset["points3D_xyz"] = points
        else:
            transform = np.eye(4)
        self.transform = transform
        # TODO: there's some issue with parser.export() and further serialization when normalize is off

        # size of the scene measured by cameras
        camera_locations = dataset["cameras"].poses[:, :3, 3]
        scene_center = np.mean(camera_locations, axis=0)
        dists = np.linalg.norm(camera_locations - scene_center, axis=1)

        self.scene_scale = np.max(dists)
        self.points = dataset.get("points3D_xyz")
        self.points_rgb = dataset.get("points3D_rgb")
        self.dataset = dataset

        self.nerfbaselines_dataset = dataset

        # Compatibility with original parser
        self.image_names = [os.path.basename(p) for p in dataset["image_paths"]]
        self.point_indices = {
            self.image_names[i]: indices
            for i, indices in enumerate(dataset["images_points3D_indices"])
        }

    @property
    def dataset_name(self):
        meta = self.dataset["metadata"]
        return "_".join([meta["id"], meta["scene"]])

    def export(self):
        return {
            "scene_scale": self.scene_scale,
            "num_points": len(self.points) if self.points is not None else None,
            "transform": self.transform.tolist(),
            "transform_base64": numpy_to_base64(self.transform),
            "num_train_images": self.num_train_images,
        }
