# import torch

# from monodepth.third_party.MoGe.moge.model.v2 import MoGeModel
# from monodepth.config import Config
# from .depth_predictor_interface import DepthPredictor, PredictedDepth


# class MoGe(DepthPredictor):
#     def __init__(self, config: Config, device: str):
#         # Load the model from huggingface
#         self.__device = device
#         self.__model = MoGeModel.from_pretrained(
#             f"Ruicheng/moge-2-{config.mdi.moge.backbone}-normal"
#         ).to(device)
#         self.__name = f"MoGe_{config.mdi.moge.backbone.value}"

#     @property
#     def name(self) -> str:
#         return self.__name

#     def __preprocess(self, img: torch.Tensor):
#         assert img.ndim == 3
#         return img.permute(2, 0, 1).to(self.__device)

#     def predict_depth(self, img: torch.Tensor, *_):
#         result = self.__model.infer(self.__preprocess(img))
#         return PredictedDepth(
#             depth=result["depth"].clone(),
#             mask=result["mask"].clone(),
#             depth_confidence=None,
#             normal=result["normal"].clone(),
#             normal_confidence=None,
#         )

# TODO
