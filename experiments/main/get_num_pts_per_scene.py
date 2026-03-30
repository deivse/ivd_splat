from dataclasses import dataclass
import itertools
import json
import logging

from eval_scripts.common.parameter_defs import PARAMS
from eval_scripts.common.results_dir import ResultsDirectory
from eval_scripts.common.typedefs import InitMethod
from eval_scripts.ivd_splat_runner import MAX_STEPS
from eval_scripts.common.dataset_scenes import SCENES_PER_DATASET
import tyro

_LOGGER = logging.getLogger(__name__)


@dataclass
class Args:
    results_dir: str
    datasets: list[str]

    init_method: InitMethod = InitMethod.sfm
    method: str = "ivd-splat"
    init_config: str = "default"
    method_config: str = "default"
    output: str = "num_points_per_scene.json"


def load_final_num_gaussians(
    results_dir: ResultsDirectory,
    scene: str,
    method: str,
    init_method: InitMethod,
    init_method_config_id: str,
    method_config_id: str,
    step: int = MAX_STEPS,
) -> int:
    dir = results_dir.get_method_output_dir(
        scene,
        method,
        init_method,
        init_method_config_id,
        method_config_id,
    )
    # TODO: this is outdated, should query via mlflow now, but no need to rewrite for now.
    param_instance = PARAMS["num_gaussians"].load(dir, step)
    _LOGGER.info(
        f"Loaded final num gaussians with SfM for {scene} at step {step} from {dir}: {param_instance.value}"
    )
    return int(param_instance.value)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = tyro.cli(Args)

    retval: dict[str, int] = {}
    for scene in itertools.chain(*[SCENES_PER_DATASET[ds] for ds in args.datasets]):
        try:
            num_points = load_final_num_gaussians(
                ResultsDirectory(args.results_dir),
                str(scene),
                args.method,
                args.init_method,
                args.init_config,
                args.method_config,
            )
            retval[str(scene)] = num_points
        except Exception as e:
            _LOGGER.error(f"Error loading num points for scene {scene}: {e}")

    with open(args.output, "w") as f:
        json.dump(retval, f, indent=2)


if __name__ == "__main__":
    main()
