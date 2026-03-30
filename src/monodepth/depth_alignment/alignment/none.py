from monodepth.depth_alignment.interface import (
    DepthAlignmentResult,
    DepthAlignmentStrategy,
)
from monodepth.depth_prediction.interface import PredictedDepth


class DepthAlignmentNone(DepthAlignmentStrategy):
    @classmethod
    def align(
        cls,
        predicted_depth: PredictedDepth,
        *args,
        **kwargs,
    ) -> DepthAlignmentResult:
        return DepthAlignmentResult(
            aligned_depth=predicted_depth.depth,
            mask=predicted_depth.mask,
        )
