"""
Runs training and evaluation for multiple scenes and initialization strategies, reports results.
"""

import functools
import itertools
import json
import logging
import os
import traceback
import sys
import typing
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path

from eval_scripts.common.config_strings import (
    ParamList,
    load_configs,
)
from eval_scripts.common.mlflow_setup import mlflow_runner_setup
from eval_scripts.common.slurm import select_task_subset_if_slurm
from eval_scripts.common.subprocess import subprocess_run_tee_stderr
import mlflow
import tyro
from ivd_splat.config import Config as IVDSplatConfig

from eval_scripts.common.ansi_escapes import ANSIEscapes, ansiesc_print
from eval_scripts.common.dataset_scenes import (
    get_scenes_from_args,
    scene_id_to_nerfbaselines_data_value,
)
from eval_scripts.common.results_dir import (
    ResultsDirectory,
    directory_exists_and_has_files,
    rename_old_dir_with_timestamp,
)
from eval_scripts.common.typedefs import InitMethod

_LOGGER = logging.getLogger(__name__)


@dataclass
class IVDRunnerArguments:
    init_methods: list[InitMethod] = field(
        default_factory=lambda: [InitMethod.gt_pointcloud]
    )
    # List of actions to perform. 'train' - training + evaluation, 'eval' - only evaluation.
    actions: list[typing.Literal["train", "eval"]] = field(
        default_factory=lambda: ["train"]
    )

    # MLflow experiment name. If not set and MLFLOW_EXPERIMENT_NAME env var not set, defaults to "Default".
    mlflow_experiment: typing.Optional[str] = None
    # Nerfbaselines method identifier, the method will be trained on the produced outputs.
    method: str = "ivd-splat"

    init_method_config: str = "default"

    # Json file containing mapping from scene id to number of points to use for initialization.
    # If set, will be used to set a per-scene cap on the number of gaussians for gt_pointcloud initialization.
    gaussian_cap_per_scene_file: str | None = None
    # Fraction of number of gaussians from `gaussian_cap_per_scene_file` to use as `strategy.cap_max`.
    gaussian_cap_fraction: float = 1.0
    # Json file containing mapping from scene id to number of SfM points, used for gt_pointcloud initialization.
    # If not set, gt_pointcloud init will use default settings (all points, I think).
    init_size_per_scene_file: str | None = None

    # Config strings specifying which configurations of the method to run.
    configs: list[str] = field(default_factory=lambda: [""])
    # Path to a file containing config strings to run, one per line.
    configs_file: str | None = None

    # Extra tags added to config name (to differentiate from other runs), and logged as mlflow params. Form: <name>=<value>
    extra_tags: list[str] = field(default_factory=list)

    # Extra config overrides for the method. If passsed in without -- or -, will be added automatically.
    config_overrides: list[str] = field(default_factory=list)
    # Extra args passed to nerfbaselines train such as --ignore-depth-cache, not treated as part of config string. If passed in without -- or -, will be added automatically.
    extra_train_args: list[str] = field(default_factory=list)

    # Output directory for results.
    output_dir: Path = Path("results")

    # Datasets to run on.
    datasets: list[str] = field(default_factory=list)
    # Scenes to run on, can be either dataset/scene format we use, or local paths.
    scenes: list[str] = field(default_factory=list)

    # Evaluation frequency in steps.
    eval_frequency: int = 8000
    # Force overwrite existing results instead of backing them up.
    force_overwrite: bool = False

    def get_cap_max_param_name(self) -> str:
        if self.method.replace("_", "-") == "ivd-splat":
            return "strategy.cap_max"
        elif self.method.replace("_", "-") == "mcmc":
            return "cap_max"
        else:
            logging.warning(
                f"cap_num_gaussians is set to True, but method {self.method} is not recognized. Defaulting to 'cap_max' as the parameter name for max number of gaussians. This may lead to errors if the method actually uses a different parameter name for max number of gaussians."
            )
            return "cap_max"


def get_cap_max_param_name_for_method(method: str) -> str:
    if method.replace("_", "-") == "ivd-splat":
        return "strategy.cap_max"
    elif method.replace("_", "-") == "mcmc":
        return "cap_max"
    else:
        logging.warning(
            f"Method {method} is not recognized. Defaulting to 'cap_max' as the parameter name for max number of gaussians. This may lead to errors if the method actually uses a different parameter name for max number of gaussians."
        )
        return "cap_max"


CONFIG_STR_PARAM_RENAMES: dict[str, str | None] = {}


def nb_output_dir_needs_overwrite(
    output_dir: Path,
    args: IVDRunnerArguments,
    eval_all_iters: list[int],
) -> bool:
    if args.force_overwrite:
        return True

    if not directory_exists_and_has_files(output_dir):
        return True

    for iter in eval_all_iters:
        if iter == 0:
            continue  # nerfbaselines never evals at 0

        if not (output_dir / f"results-{str(iter)}.json").exists():
            return True

    return False


def add_missing_dashes(arg: str) -> str:
    if arg.startswith("-"):
        return arg
    if len(arg) == 1:
        return "-" + arg
    return "--" + arg


def nerfbaselines_train(
    nerfbaselines_data_value: str,
    args: IVDRunnerArguments,
    config: ParamList,
    train_output_dir: Path,
    eval_all_iters: list[int],
    subprocess_env: dict[str, str],
):
    command = [
        "nerfbaselines",
        "train",
        f"--data={nerfbaselines_data_value}",
        f"--method={args.method}",
        f"--eval-all-iters={','.join(str(x) for x in eval_all_iters if x != 0)}",
        f"--output={train_output_dir}",
        "--logger=mlflow,tensorboard",
    ] + [add_missing_dashes(arg) for arg in args.extra_train_args]

    config_overrides = ["=".join(x) for x in config]
    for override in itertools.chain(config_overrides, args.config_overrides):
        command.append("--set")
        command.append(override)

    if args.method.replace("_", "-") == "ivd-splat":
        command.append("--backend=python")

    ansiesc_print("Running nerfbaselines/train:", "bold")
    print(command[0], " ".join([f'"{x}"' for x in command[1:]]))
    rc, stderr_output = subprocess_run_tee_stderr(command, env=subprocess_env)
    if stderr_output:
        mlflow.log_text(stderr_output, "nb_train_stderr.txt")
    if rc != 0:
        raise RuntimeError(
            f"Error during {args.method} training for {nerfbaselines_data_value}."
        )


def nerfbaselines_evaluate(
    nerfbaselines_data_value: str,
    args: IVDRunnerArguments,
    config: ParamList,
    curr_output_dir: Path,
    subprocess_env: dict[str, str],
):
    # evaluate --data external: // mipnerf360/garden - -output results/eval.json  results/predictions-2000.tar.gz
    output = f"{curr_output_dir}/{args.method}/results-{MAX_STEPS}.json"
    Path(output).unlink(missing_ok=True)
    command = [
        "nerfbaselines",
        "evaluate",
        f"--output={output}",
        f"--data={nerfbaselines_data_value}",
        f"{curr_output_dir}/{args.method}/predictions-{MAX_STEPS}.tar.gz",
    ]

    for override in itertools.chain(
        ["=".join(x) for x in config], args.config_overrides
    ):
        command.append("--set")
        command.append(override)

    if args.method.replace("_", "-") == "ivd-splat":
        command.append("--backend=python")

    ansiesc_print("Running nerfbaselines/evaluate:", "bold")
    print(command[0], " ".join([f'"{x}"' for x in command[1:]]))
    rc, stderr_output = subprocess_run_tee_stderr(command, env=subprocess_env)
    if stderr_output:
        mlflow.log_text(stderr_output, "evaluate_stderr.txt")
    if rc != 0:
        raise RuntimeError(
            f"Error during {args.method} evaluation for {nerfbaselines_data_value}."
        )


@functools.cache
def load_num_points_per_scene(
    path: str | None,
) -> dict[str, int]:
    if path is None:
        raise RuntimeError(
            "num_points_per_scene_file must be provided to use gt_pointcloud initialization."
        )

    with open(path, "r") as f:
        data = json.load(f)

    return {str(k): int(v) for k, v in data.items()}


def get_data_and_config_overrides_for_init_method(
    results_dir: ResultsDirectory,
    init_method: InitMethod,
    scene: str,
    args: IVDRunnerArguments,
) -> tuple[str, ParamList]:
    def append_target_num_points_if_needed(overrides: list[tuple[str, str]]):
        if args.init_size_per_scene_file is not None:
            target_num_points = load_num_points_per_scene(
                args.init_size_per_scene_file
            )[scene]
            overrides.append(("dense_init.target_num_points", str(target_num_points)))
        else:
            logging.warning(
                f"init_size_per_scene_file not provided for {init_method.value} initialization, using default settings for number of points (probably all points)."
            )

    if args.method.replace("_", "-") != "ivd-splat":
        logging.warning(
            f"Training with method other than ivd-splat, skipping config overrides for init method {init_method.value}."
        )
        return scene_id_to_nerfbaselines_data_value(scene), ParamList(())
    if init_method == InitMethod.sfm:
        return scene_id_to_nerfbaselines_data_value(scene), ParamList(
            (("init_type", "sparse"),)
        )
    if init_method == InitMethod.gt_pointcloud:
        overrides = [("init_type", "dense")]
        append_target_num_points_if_needed(overrides)

        return scene_id_to_nerfbaselines_data_value(scene), ParamList(overrides)
    if init_method in (InitMethod.monodepth, InitMethod.da3):
        overrides = [("init_type", "dense")]
        append_target_num_points_if_needed(overrides)

        init_dir = results_dir.get_init_method_output_dir(
            scene, args.init_method_config, init_method
        )
        return str(init_dir), ParamList(overrides)
    if init_method == InitMethod.edgs:
        overrides = [("init_type", "splat")]
        append_target_num_points_if_needed(overrides)

        edgs_output_dir = results_dir.get_edgs_output_dir(
            scene, args.init_method_config
        )
        return str(edgs_output_dir), ParamList(overrides)
    raise RuntimeError(f"Unknown init method: {init_method}")


def add_final_step_metric_to_mlflow_run(run: mlflow.ActiveRun):
    client = mlflow.tracking.MlflowClient()

    # get max step logged for metric "train/loss"
    metric_history = client.get_metric_history(run.info.run_id, "train/loss")
    if len(metric_history) == 0:
        # Run didn't have time to log any metrics.
        return
    max_step = max(metric_history, key=lambda x: x.step).step

    client.log_metric(run.info.run_id, "final_step", max_step)


def should_train(
    train_output_dir: Path, args: IVDRunnerArguments, scene: str, eval_all_iters
):
    if not directory_exists_and_has_files(train_output_dir) or args.force_overwrite:
        return True

    if not nb_output_dir_needs_overwrite(train_output_dir, args, eval_all_iters):
        ansiesc_print(
            f"Skipping training for {args.method} on {scene}. (Output exists)",
            ANSIEscapes.GREEN,
        )
        return False

    new_path = rename_old_dir_with_timestamp(
        train_output_dir, ResultsDirectory(args.output_dir)
    )
    ansiesc_print(
        f"Detected incomplete nerfbaselines output. Old output directory moved to: {new_path}",
        ANSIEscapes.YELLOW,
    )
    assert not train_output_dir.exists()

    return True


def process_combination(
    scene: str,
    init_method: InitMethod,
    config: ParamList,
    args: IVDRunnerArguments,
    eval_all_iters: list[int],
    subprocess_env: dict[str, str],
):
    print(
        ANSIEscapes.format("_" * 80, "bold"),
        ANSIEscapes.format("=" * 80 + "\n", "blue"),
        sep="\n",
    )

    if args.gaussian_cap_per_scene_file is not None:
        try:
            cap = load_num_points_per_scene(args.gaussian_cap_per_scene_file)[scene]
            cap = int(cap * args.gaussian_cap_fraction)
        except KeyError as e:
            raise RuntimeError(
                f"gaussian_cap_per_scene_file is provided, but no entry for scene {scene} was found."
            ) from e
        config = config.with_prepended_params(
            (
                (
                    get_cap_max_param_name_for_method(args.method),
                    str(cap),
                ),
            )
        )

    nerfbaselines_data_val, initialization_params = (
        get_data_and_config_overrides_for_init_method(
            ResultsDirectory(args.output_dir), init_method, scene, args
        )
    )
    init_params_included_in_name = [
        f"{k}={v}" for k, v in initialization_params if k != "init_type"
    ]

    # We don't include init_type in config name
    # because it's already in the output dir structure through init_method
    config_name = config.make_config_name(
        CONFIG_STR_PARAM_RENAMES,
        args.extra_tags + init_params_included_in_name,
    )
    config = config.with_prepended_params(initialization_params)

    output_dir = ResultsDirectory(args.output_dir).get_method_output_dir(
        scene,
        args.method,
        init_method,
        args.init_method_config,
        config_name,
    )

    ansiesc_print(
        f"Processing '{config_name}' on {scene}. (Outputting to: {output_dir})",
        ANSIEscapes.BLUE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    interrupted: typing.Optional[KeyboardInterrupt] = None

    def raise_if_interrupted():
        if interrupted is not None:
            raise KeyboardInterrupt() from interrupted

    def with_exception_logger(fun: typing.Callable[[], None], stage_name: str):
        nonlocal interrupted
        try:
            fun()
        except KeyboardInterrupt as e:
            print("Interrupted by user, will exit after mlflow housekeeping.")
            interrupted = e
        except Exception as e:
            print(
                f"Error during {stage_name} with {init_method.value} initialization for {config_name} on {scene}: {e}"
            )

    if "train" in args.actions and should_train(
        output_dir, args, scene, eval_all_iters
    ):
        if os.environ.get("MLFLOW_RUN_ID") is not None:
            raise RuntimeError("MLFLOW_RUN_ID is already set in the environment.")

        mlflow_run_name = str(output_dir.relative_to(args.output_dir))

        with mlflow.start_run(run_name=mlflow_run_name) as mlflow_run:
            # Make sure final_step is always logged, even if training is interrupted before any steps are logged.
            # Since mlflow doesn't filter well when metrics are missing, we log final_step = -1 here,
            # and update it at the end of training.
            mlflow.MlflowClient().log_metric(mlflow_run.info.run_id, "final_step", -1)

            print(f"MLflow run ID: {mlflow_run.info.run_id}")
            subprocess_env["MLFLOW_RUN_ID"] = mlflow_run.info.run_id

            mlflow.log_param("scene", scene)
            mlflow.log_param("method", args.method)
            mlflow.log_param("init_method", init_method.value)
            mlflow.log_param("init_method_config", args.init_method_config)
            mlflow.log_param("gaussian_cap_fraction", args.gaussian_cap_fraction)
            for param in args.extra_tags:
                try:
                    name, value = param.split("=")
                    mlflow.log_param(name.strip(), value.strip())
                except Exception:
                    logging.exception(
                        f"Failed to add extra mlflow parameter, arg string: {param}"
                    )

            with_exception_logger(
                lambda: nerfbaselines_train(
                    nerfbaselines_data_val,
                    args,
                    config,
                    output_dir,
                    eval_all_iters,
                    subprocess_env,
                ),
                "training",
            )

            add_final_step_metric_to_mlflow_run(mlflow_run)
        raise_if_interrupted()

    if "eval" in args.actions:
        with_exception_logger(
            lambda: nerfbaselines_evaluate(
                scene,
                args,
                config,
                output_dir,
                subprocess_env,
            ),
            "evaluation",
        )
        raise_if_interrupted()


MAX_STEPS = 30000


def get_eval_it_list(args: IVDRunnerArguments):
    eval_all_iters = list(range(0, MAX_STEPS + 1, args.eval_frequency))
    if MAX_STEPS not in eval_all_iters:
        eval_all_iters.append(MAX_STEPS)
    return eval_all_iters


def main():
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO)

    args = tyro.cli(IVDRunnerArguments)

    configs = load_configs(args.configs, args.configs_file, IVDSplatConfig)
    scenes = get_scenes_from_args(args.scenes, args.datasets)

    combinations = sorted(list(product(scenes, args.init_methods, configs)))
    combinations = select_task_subset_if_slurm(combinations)

    subprocess_env = mlflow_runner_setup(args.output_dir, args.mlflow_experiment)

    scenes, init_methods, configs = (
        zip(*combinations) if len(combinations) > 0 else ([], [], [])
    )

    print(
        ANSIEscapes.format("_" * 80, "bold"),
        ANSIEscapes.format(
            f"Will {', '.join(args.actions)} {len(combinations)} combinations.", "bold"
        ),
        ANSIEscapes.format("Settings:", "bold"),
        f"\tOutput directory: {ANSIEscapes.format(args.output_dir, 'cyan')}",
        f"\tEvaluation frequency: {ANSIEscapes.format(args.eval_frequency, 'cyan')}",
        "\tConfigs: "
        + ANSIEscapes.format(
            "\n\t          ".join(
                c.make_config_name(CONFIG_STR_PARAM_RENAMES, args.extra_tags)
                for c in configs
            ),
            "cyan",
        ),
        f"\tScenes: {ANSIEscapes.format(scenes, 'cyan')}",
        f"\tInit methods: {ANSIEscapes.format(init_methods, 'cyan')}",
        sep="\n",
    )

    for scene, init_method, config in combinations:
        try:
            process_combination(
                scene, init_method, config, args, get_eval_it_list(args), subprocess_env
            )
        except Exception as e:
            logging.error(
                f"Error processing combination of scene {scene}, init method {init_method.value}, config {config}: {e}"
            )
            logging.error(traceback.format_exc())
            print(
                ANSIEscapes.format(
                    "Proceeding to next combination due to exception.",
                    ANSIEscapes.RED,
                )
            )


if __name__ == "__main__":
    main()
