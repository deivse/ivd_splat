import logging

import cv2
import numpy as np
import torch

from monodepth.config import Config
from monodepth.depth_prediction.configs import Metric3dBackbone
from monodepth.depth_prediction.interface import (
    CameraIntrinsics,
    DepthPredictor,
    PredictedDepth,
)

_LOGGER = logging.getLogger(__name__)


class Metric3d(DepthPredictor):
    def __init__(self, config: Config, device: str):
        super().__init__(config, device)
        m3d_preset_to_checkpoint = {
            Metric3dBackbone.vits: "metric3d_vit_small",
            Metric3dBackbone.vitl: "metric3d_vit_large",
            Metric3dBackbone.vitg: "metric3d_vit_giant2",
        }
        checkpoint = m3d_preset_to_checkpoint[config.metric3d.backbone]
        self.__name = f"Metric3d_{config.metric3d.backbone.value}"
        self.__model = torch.hub.load(
            "yvanyin/metric3d",
            checkpoint,
            pretrain=True,
        )
        self.__model.to(device).eval()

    @property
    def name(self) -> str:
        return self.__name

    def predict_depth(
        self, img: torch.Tensor, intrinsics: CameraIntrinsics
    ) -> PredictedDepth:
        #### prepare data
        intrinsic = [intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]
        rgb_origin = (img * 255.0).cpu().numpy().astype(np.uint8)[:, :, ::-1]

        #### ajust input size to fit pretrained model
        # keep ratio resize
        input_size = (616, 1064)  # for vit model
        h, w = rgb_origin.shape[:2]
        scale = min(input_size[0] / h, input_size[1] / w)
        rgb = cv2.resize(
            rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
        )
        # remember to scale intrinsic, hold depth
        intrinsic = [
            intrinsic[0] * scale,
            intrinsic[1] * scale,
            intrinsic[2] * scale,
            intrinsic[3] * scale,
        ]
        # padding to input_size
        padding = [123.675, 116.28, 103.53]
        h, w = rgb.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        rgb = cv2.copyMakeBorder(
            rgb,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=padding,
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        #### normalize
        mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
        std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
        rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
        rgb = torch.div((rgb - mean), std)
        rgb = rgb[None, :, :, :].cuda()

        ###################### canonical camera space ######################
        # inference
        with torch.no_grad():
            pred_depth, confidence, output_dict = self.__model.inference({"input": rgb})
        pred_normal = output_dict["prediction_normal"][
            :, :3, :, :
        ]  # only available for Metric3Dv2 i.e., ViT models

        def to_og_size(tensor, pad_info, rgb_origin):
            # un pad
            tensor = tensor.squeeze()
            has_data_dim = len(tensor.shape) == 3
            if has_data_dim:
                tensor = tensor[
                    :,
                    pad_info[0] : tensor.shape[1] - pad_info[1],
                    pad_info[2] : tensor.shape[2] - pad_info[3],
                ]
                tensor = torch.nn.functional.interpolate(
                    tensor[None], rgb_origin.shape[:2], mode="bilinear"
                ).squeeze()
            else:
                tensor = tensor[
                    pad_info[0] : tensor.shape[0] - pad_info[1],
                    pad_info[2] : tensor.shape[1] - pad_info[3],
                ]
                tensor = torch.nn.functional.interpolate(
                    tensor[None, None], rgb_origin.shape[:2], mode="bilinear"
                ).squeeze()
            return tensor

        # un pad and upsample to original size
        pred_depth = to_og_size(pred_depth, pad_info, rgb_origin)
        confidence = to_og_size(confidence, pad_info, rgb_origin)
        pred_normal = to_og_size(pred_normal, pad_info, rgb_origin).permute(1, 2, 0)
        ###################### canonical camera space ######################

        #### de-canonical transform
        canonical_to_real_scale = (
            intrinsic[0] / 1000.0
        )  # 1000.0 is the focal length of canonical camera
        pred_depth = pred_depth * canonical_to_real_scale  # now the depth is metric
        pred_depth = torch.clamp(pred_depth, 0, 300)

        return PredictedDepth(
            depth=pred_depth,
            mask=torch.ones_like(pred_depth, dtype=torch.bool),
            depth_confidence=confidence,
            normal=pred_normal,
            normal_confidence=None,
        )
