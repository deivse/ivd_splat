from dataclasses import dataclass, field
import itertools
import json
import logging
from pathlib import Path
from typing import Literal

from eval_scripts.common.results_dir import ResultsDirectory
from eval_scripts.common.typedefs import InitMethod
import tqdm

from shared.point_cloud_io import load_pointcloud_ply
from shared.splat_ply_io import load_splat_ply

from eval_scripts.common.dataset_scenes import (
    SCENES_PER_DATASET,
)
import tyro

from monodepth.proxy_dataset import POINTS_FILE_NAME as MONODEPTH_POINTS_FILE_NAME
from edgs.proxy_dataset import GAUSSIANS_FILE_NAME as EDGS_GAUSSIANS_FILE_NAME

_LOGGER = logging.getLogger(__name__)


def load_num_points_per_scene(
    path: str | None,
) -> dict[str, int]:
    if path is None:
        raise RuntimeError(
            "num_points_per_scene_file must be provided to use laser_scan initialization."
        )

    with open(path, "r") as f:
        data = json.load(f)

    return {str(k): int(v) for k, v in data.items()}


@dataclass
class Args:
    datasets: list[str] | None = None
    init_methods: list[Literal[InitMethod.edgs, InitMethod.monodepth]] = field(
        default_factory=lambda: [InitMethod.edgs, InitMethod.monodepth]
    )
    results_dir: Path = Path("results")
    gaussian_cap_per_scene_file: Path = Path("num_points_per_scene.json")
    gaussian_cap_fraction: float = 1.0
    method_configs: list[str] = field(
        default_factory=lambda: ["edgs=default", "monodepth=default"]
    )

    output: str = "real_init_min_pts_per_scene.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = tyro.cli(Args)
    if args.datasets is None:
        args.datasets = list(SCENES_PER_DATASET.keys())
    scenes = itertools.chain(*[SCENES_PER_DATASET[ds] for ds in args.datasets])

    config_name_per_method = {}
    for method_config_str in args.method_configs:
        method_str, config_name = method_config_str.split("=")
        method = InitMethod(method_str)
        config_name_per_method[method] = config_name

    gaussian_caps = load_num_points_per_scene(args.gaussian_cap_per_scene_file)

    results_dir = ResultsDirectory(args.results_dir)
    retval: dict[str, int] = {}
    for scene in tqdm.tqdm(list(scenes), desc="Processing scenes"):
        try:
            pts_for_methods = []
            for method in args.init_methods:
                config_name = config_name_per_method[method]
                out_dir = results_dir.get_init_method_output_dir(
                    scene, config_name, method
                )
                if method == InitMethod.monodepth:
                    points_file = out_dir / MONODEPTH_POINTS_FILE_NAME
                    if not points_file.exists():
                        raise FileNotFoundError(
                            f"Points file not found for scene {scene} and method {method} at expected location {points_file}"
                        )
                    num_pts = load_pointcloud_ply(points_file)[0].shape[0]
                elif method == InitMethod.edgs:
                    gaussians_file = out_dir / EDGS_GAUSSIANS_FILE_NAME
                    if not gaussians_file.exists():
                        raise FileNotFoundError(
                            f"Gaussians file not found for scene {scene} and method {method} at expected location {gaussians_file}"
                        )
                    num_pts = load_splat_ply(gaussians_file).means.shape[0]
                else:
                    raise ValueError(f"Unknown method: {method}")
                pts_for_methods.append(num_pts)
            if len(pts_for_methods) != len(args.init_methods):
                raise ValueError(
                    f"Number of methods ({len(args.init_methods)}) does not match number of pts counts ({len(pts_for_methods)}) for scene {scene}"
                )

            cap = cap = gaussian_caps[scene]
            cap = int(cap * args.gaussian_cap_fraction)

            retval[scene] = min(min(pts_for_methods), cap)
        except Exception as e:
            print(f"Error processing scene {scene}: {e}")

    with open(args.output, "w") as f:
        json.dump(retval, f, indent=4)
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
