from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Optional, NamedTuple

import torch

_LOGGER = logging.getLogger(__name__)


@dataclass
class PredictedDepth:
    depth: torch.Tensor
    """ Float tensor of shape (H, W) """
    mask: torch.Tensor
    """ Bool tensor indicating valid pixels. (H, W) """
    depth_confidence: Optional[torch.Tensor] = None
    """ Optional float tensor indicating confidence of each pixel. (H, W) """
    normal: Optional[torch.Tensor] = None
    """ Optional float tensor of shape (H, W, 3) """
    normal_confidence: Optional[torch.Tensor] = None
    """ Optional float tensor indicating confidence of each normal vector. (H, W, 3) """


class CameraIntrinsics(NamedTuple):
    data: torch.Tensor  # 4 elements: fx, fy, cx, cy

    @property
    def fx(self):
        return self.data[0].item()

    @property
    def fy(self):
        return self.data[1].item()

    @property
    def cx(self):
        return self.data[2].item()

    @property
    def cy(self):
        return self.data[3].item()

    @staticmethod
    def from_K(K: torch.Tensor) -> "CameraIntrinsics":
        fx = K[0, 0].item()
        fy = K[1, 1].item()
        cx = K[0, 2].item()
        cy = K[1, 2].item()
        return CameraIntrinsics(torch.tensor([fx, fy, cx, cy]))

    def get_K_matrix(self, device=None, dtype=torch.float32) -> torch.Tensor:
        K = torch.zeros((3, 3), device=device, dtype=dtype)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        K[2, 2] = 1.0
        return K


if torch.__version__ >= "2.4.0":
    torch.serialization.add_safe_globals([PredictedDepth])


class DepthPredictor(metaclass=ABCMeta):
    def __init__(self, config, device: torch.device | str):
        self.config = config
        self.device = device

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the predictor.
        """

    @abstractmethod
    def predict_depth(
        self, img: torch.Tensor, intrinsics: CameraIntrinsics
    ) -> PredictedDepth:
        """
        Predict depth from a single image.

        Args:
            img: tensor of shape (H, W, 3).
            intrinsics: Camera intrinsics from sparse reconstruction.

        Returns:
            Depth map.
        """

    def predict_depth_or_get_cached_depth(
        self,
        image: torch.Tensor,
        intrinsics: CameraIntrinsics,
        image_name: str,
        dataset_subdir: Path,
    ) -> PredictedDepth:
        assert (
            dataset_subdir.is_absolute() is False
        ), "dataset_subdir must be relative path"
        cache_dir = self.config.cache_dir / self.name / dataset_subdir

        cache_dir.mkdir(exist_ok=True, parents=True)
        width, height = image.shape[1], image.shape[0]
        cache_path = cache_dir / f"{image_name}_{width}x{height}.pth"

        depth = None
        if not self.config.ignore_depth_cache and cache_path.exists():
            try:
                depth = torch.load(cache_path)
            except Exception as e:
                _LOGGER.warning(
                    f"Failed to load cached depth for image {image_name}: {e}"
                )

        if depth is None:
            depth = self.predict_depth(image, intrinsics)
            try:
                torch.save(depth, cache_path)
            except KeyboardInterrupt:
                cache_path.unlink(missing_ok=True)
                raise
        return depth
