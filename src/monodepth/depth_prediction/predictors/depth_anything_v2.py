# import logging
# from pathlib import Path

# import cv2
# import torch

# from monodepth.config import Config
# from common.download_with_tqdm import (
#     download_with_pbar,
# )

# from monodepth.third_party.depth_anything_v2.metric_depth.depth_anything_v2.dpt import (
#     DepthAnythingV2 as DepthAnythingV2Model,
# )

# from .depth_predictor_interface import DepthPredictor, PredictedDepth

# _LOGGER = logging.getLogger(__name__)


# class DepthAnythingV2(DepthPredictor):
#     __MODEL_CONFIGS = {
#         "vits": {
#             "encoder": "vits",
#             "features": 64,
#             "out_channels": [48, 96, 192, 384],
#         },
#         "vitb": {
#             "encoder": "vitb",
#             "features": 128,
#             "out_channels": [96, 192, 384, 768],
#         },
#         "vitl": {
#             "encoder": "vitl",
#             "features": 256,
#             "out_channels": [256, 512, 1024, 1024],
#         },
#         "vitg": {
#             "encoder": "vitg",
#             "features": 384,
#             "out_channels": [1536, 1536, 1536, 1536],
#         },
#     }

#     _MODEL_PARAMS_BY_TYPE = {
#         "indoor": {
#             "dataset": "hypersim",
#             "max_depth": 20,
#         },
#         "outdoor": {
#             "dataset": "vkitti",
#             "max_depth": 80,
#         },
#     }

#     def __init__(self, config: Config, device: str):
#         self.device = device
#         encoder = config.mdi.depthanything.backbone
#         is_metric = config.mdi.depthanything.metric
#         self.__is_metric = is_metric
#         metric_model_type = config.mdi.depthanything.metric_model_type

#         if is_metric:
#             self.__name = f"DepthAnythingV2_{encoder}_metric_{metric_model_type}"
#             try:
#                 metric_dataset = self._MODEL_PARAMS_BY_TYPE[metric_model_type][
#                     "dataset"
#                 ]
#                 max_depth = self._MODEL_PARAMS_BY_TYPE[metric_model_type]["max_depth"]
#             except KeyError as e:
#                 raise ValueError(
#                     f"Unsupported metric model type '{metric_model_type}' for DepthAnythingV2"
#                 ) from e

#             checkpoint_path = (
#                 Path(config.mdi.cache_dir)
#                 / f"checkpoints/depth_anything_v2_metric_{encoder}_{metric_dataset}.pt"
#             )
#         else:
#             self.__name = f"DepthAnythingV2_{encoder}_relative"
#             metric_dataset = None
#             max_depth = 20  # default max depth for relative model
#             checkpoint_path = (
#                 Path(config.mdi.cache_dir)
#                 / f"checkpoints/depth_anything_v2_relative_{encoder}.pt"
#             )

#         url = self.__get_checkpoint_url(encoder, is_metric, metric_dataset)
#         checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
#         # Download the checkpoint if it doesn't exist
#         download_with_pbar(url, checkpoint_path)

#         self.model = DepthAnythingV2Model(
#             **{
#                 **self.__MODEL_CONFIGS[encoder],
#                 "max_depth": max_depth,
#             }
#         )
#         self.model.load_state_dict(torch.load(str(checkpoint_path), map_location="cpu"))
#         self.model = self.model.to(device).eval()

#     @property
#     def name(self) -> str:
#         return self.__name

#     @staticmethod
#     def __get_checkpoint_url(
#         encoder: str, metric: bool, metric_dataset: str | None = None
#     ) -> str:
#         encoder_to_name = {
#             "vits": "Small",
#             "vitb": "Base",
#             "vitl": "Large",
#             # "vitg": "Giant", # Not available yet
#         }

#         if metric:
#             dataset_to_name = {
#                 "hypersim": "Hypersim",
#                 "vkitti": "VKITTI",
#             }
#             return (
#                 "https://huggingface.co/depth-anything/"
#                 f"Depth-Anything-V2-Metric-{dataset_to_name[metric_dataset]}-{encoder_to_name[encoder]}/"
#                 f"resolve/main/depth_anything_v2_metric_{metric_dataset}_{encoder}.pth?download=true"
#             )
#         else:
#             return (
#                 "https://huggingface.co/depth-anything/"
#                 f"Depth-Anything-V2-{encoder_to_name[encoder]}/"
#                 f"resolve/main/depth_anything_v2_{encoder}.pth?download=true"
#             )

#     def predict_depth(self, img: torch.Tensor, *_) -> PredictedDepth:
#         # `infer_image` expects image in BGR and in range [0, 255]
#         input_image = (img.cpu().numpy() * 255)[:, :, ::-1].astype("uint8")
#         output = self.model.infer_image(input_image)
#         output = torch.from_numpy(output).to(self.device)
#         if not self.__is_metric:
#             mask = torch.isfinite(output) & (output >= 1e-4)
#             output[~mask] = 0.0
#             # output in relative mode is disparity, convert to scale-invariant depth
#             return PredictedDepth(
#                 depth=1.0 / (output + 1e-6),
#                 mask=mask,
#                 depth_confidence=None,
#                 normal=None,
#                 normal_confidence=None,
#             )
#         else:
#             return PredictedDepth(
#                 depth=output,
#                 mask=torch.ones_like(output, dtype=torch.bool),
#                 depth_confidence=None,
#                 normal=None,
#                 normal_confidence=None,
#             )

# TODO
