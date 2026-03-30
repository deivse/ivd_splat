import abc
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import torch

from monodepth.depth_prediction.interface import (
    CameraIntrinsics,
    PredictedDepth,
)


class DepthAlignmentResult(NamedTuple):
    aligned_depth: torch.Tensor
    mask: torch.Tensor


class DepthAlignmentStrategy(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def align(
        cls,
        predicted_depth: PredictedDepth,
        sfm_points_camera_coords: torch.Tensor,
        sfm_points_depth: torch.Tensor,
        sfm_points_error: torch.Tensor,
        intrinsics: CameraIntrinsics,
        cam2world: torch.Tensor,
        config,  # : Config,
        debug_export_dir: Path | None,
    ) -> DepthAlignmentResult:
        """
        Estimate the alignment between predicted and ground truth depth maps and return the aligned depth map.

        Args:
            predicted_depth: The predicted depth map. Shape: [Width, Height]
            sfm_points_camera_coords: The (y, x) (in that order!) coordinates of the SfM points in the camera frame. Shape: [2, NumPoints]
            sfm_points_depth: The depth of the SfM points. Shape: [NumPoints]
            sfm_points_error: Reprojection error of the SfM points. Shape: [NumPoints]
            config: Config
            debug_export_dir: Path | None
        """
