import logging
import torch

from monodepth.config import Config

from monodepth.depth_prediction.interface import (
    CameraIntrinsics,
    DepthPredictor,
    PredictedDepth,
)
from transformers import DepthProImageProcessorFast, DepthProForDepthEstimation

_LOGGER = logging.getLogger(__name__)


def _load_rgb(img: torch.Tensor, intrinsics: CameraIntrinsics):
    icc_profile = None
    _LOGGER.debug(f"abs(fx - fy) = {abs(intrinsics.fx - intrinsics.fy)}")
    # Should be equal, but may be slightly inconsistent
    f_px = torch.tensor([intrinsics.fx + intrinsics.fy]) / 2.0

    return img.cpu().numpy(), icc_profile, f_px


class AppleDepthPro(DepthPredictor):
    def __init__(self, config: Config, device: str):
        super().__init__(config, device)

        self.image_processor = DepthProImageProcessorFast.from_pretrained(
            "apple/DepthPro-hf"
        )
        self.model = DepthProForDepthEstimation.from_pretrained("apple/DepthPro-hf").to(
            device
        )

    def can_predict_points_directly(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "AppleDepthPro"

    def predict_depth(
        self, img: torch.Tensor, intrinsics: CameraIntrinsics
    ) -> PredictedDepth:
        inputs = self.image_processor(images=img, return_tensors="pt").to(
            self.model.device
        )

        # TODO: Pass in known FOV?

        with torch.no_grad():
            outputs = self.model(**inputs)

        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[img.shape[:2]],
        )

        depth = post_processed_output[0]["predicted_depth"]

        return PredictedDepth(
            depth=depth, mask=torch.ones_like(depth, dtype=torch.bool)
        )
