from .parameters import (
    NerfbaselinesJSONParameter,
    ParamOrdering,
    Parameter,
    TensorboardParameter,
    seconds_to_mins_secs_formatter,
)

# TODO: this is outdated, everything is logged using mlflow now and that should be used instead.

PARAMS: dict[str, Parameter] = {
    "psnr": NerfbaselinesJSONParameter(
        name="PSNR",
        json_path=["metrics", "psnr"],
        ordering=ParamOrdering.HIGHER_IS_BETTER,
    ),
    "ssim": NerfbaselinesJSONParameter(
        name="SSIM",
        json_path=["metrics", "ssim"],
        ordering=ParamOrdering.HIGHER_IS_BETTER,
    ),
    "lpips": NerfbaselinesJSONParameter(
        name="LPIPS",
        json_path=["metrics", "lpips"],
        ordering=ParamOrdering.LOWER_IS_BETTER,
    ),
    "lpips_vgg": NerfbaselinesJSONParameter(
        name="LPIPS(VGG)",
        json_path=["metrics", "lpips_vgg"],
        ordering=ParamOrdering.LOWER_IS_BETTER,
    ),
    "train_time": NerfbaselinesJSONParameter(
        name="Training Time",
        json_path=["nb_info", "total_train_time"],
        ordering=ParamOrdering.LOWER_IS_BETTER,
        formatter=seconds_to_mins_secs_formatter,
    ),
    "num_sfm_points": NerfbaselinesJSONParameter(  # for patches only
        name="Num Sfm Points",
        json_path=["num_sfm_points"],
        formatter=lambda val: f"{int(val):,}",
    ),
    "num_gaussians": TensorboardParameter(
        name="Num Gaussians",
        tensorboard_id="train/num-gaussians",
        formatter=lambda val: f"{int(float(val) / 1000):,}K",
        ordering=ParamOrdering.LOWER_IS_BETTER,
        should_highlight_best=False,
    ),
}
