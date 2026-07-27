import typing


def boolean_conversion(default: bool = False) -> typing.Callable[[typing.Any], bool]:
    def converter(x: typing.Any) -> bool:
        if x is None:
            return default
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            if x.lower() in {"true", "1", "yes"}:
                return True
            elif x.lower() in {"false", "0", "no"}:
                return False
        raise ValueError(f"Cannot convert value '{x}' to boolean.")

    return converter


T = typing.TypeVar("T")


def defaulter(
    default: typing.Optional[T] = None,
) -> typing.Callable[[typing.Any], typing.Optional[T]]:
    def converter(x: typing.Optional[T]) -> typing.Optional[T]:
        if x is None:
            return default
        return x

    return converter


def converter(conversion_func, default=None) -> typing.Callable[[typing.Any], str]:
    def converter(x: typing.Any) -> str:
        if x is None:
            return default
        return conversion_func(x)

    return converter


PARAM_CONVERSIONS: dict[str, typing.Callable[[typing.Any], typing.Any]] = {
    "train/num-gaussians": int,
    "train/total-train-time": converter(lambda x: float(x) / 60),
    "gaussian_cap_fraction": converter(str, default="1.0"),
    "splat_init.increase_scale_with_fewer_splats": boolean_conversion(default=True),
    "splat_init.target_splat_fraction": defaulter(default="1.0"),
    "dense_init.target_points_fraction": defaulter(default="1.0"),
    "means_lr_init": defaulter("0.00016"),
    "means_lr_final": defaulter("1.6000000000000001e-06"),
    "nanogs_simplify_iter": defaulter("-1"),
    "eval_iter": converter(int, default=0),
    "dense_init.softmax_temp": defaulter("0.0001"),
    "dense_init.color_dist_thresh": defaulter("0.01"),
    "dense_init.include_sparse": boolean_conversion(default=False),
    "init.target_median_scale": defaulter(None),
    "init.scale_color_dist_factor": defaulter(None),
    "splat_init.opacity_uniform_override": defaulter(None),
    "splat_init.opacity_noise_std": defaulter(None),
    "splat_init.init_scale_with_knn": defaulter("False"),
    "splat_init.init_scale_isotropic_mean": defaulter("False"),
    "splat_init.scale_noise_std_wrt_median": defaulter(None),
    "splat_init.rotation_noise_angle_std_deg": defaulter(None),
    "splat_init.color_noise_std": defaulter(None),
    "splat_init.simulate_point_init": defaulter("False"),
}

INIT_METHOD_PARAM_CONVERSIONS: dict[str, typing.Callable[[typing.Any], typing.Any]] = {
    "output_gaussians": boolean_conversion(default=False),
    "floater_removal": boolean_conversion(default=False),
    "full_sh_init": boolean_conversion(default=False),
}
