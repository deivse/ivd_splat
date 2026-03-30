"""
Runs training and evaluation for multiple scenes and initialization strategies, reports results.
"""

from datetime import datetime
import logging
import os
import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Literal

from eval_scripts.common.mlflow_setup import mlflow_runner_setup
from eval_scripts.common.slurm import select_task_subset_if_slurm
from eval_scripts.common.subprocess import subprocess_run_tee_stderr
from eval_scripts.common.typedefs import InitMethod
import mlflow
import tyro
from edgs.config import EDGSConfig
from monodepth.config import Config as MdTsdfFusionConfig
from da3.config import DA3Config
from monodepth.proxy_dataset import POINTS_FILE_NAME as MONODEPTH_POINTS_FILE_NAME
from monodepth.proxy_dataset import NB_META_FILE_NAME as MONODEPTH_NB_META_FILE_NAME
from da3.proxy_dataset import POINTS_FILE_NAME as DA3_POINTS_FILE_NAME
from da3.proxy_dataset import NB_META_FILE_NAME as DA3_NB_META_FILE_NAME
from edgs.proxy_dataset import GAUSSIANS_FILE_NAME as EDGS_GAUSSIANS_FILE_NAME
from edgs.proxy_dataset import NB_META_FILE_NAME as EDGS_NB_META_FILE_NAME

from eval_scripts.common.ansi_escapes import ANSIEscapes, ansiesc_print
from eval_scripts.common.config_strings import ParamList, load_configs
from eval_scripts.common.dataset_scenes import get_scenes_from_args
from eval_scripts.common.results_dir import (
    ResultsDirectory,
    directory_exists_and_has_files,
    rename_old_dir_with_timestamp,
)


@dataclass
class InitRunnerArguments:
    method: Literal[InitMethod.monodepth, InitMethod.edgs, InitMethod.da3] = InitMethod.monodepth
    # Config strings specifying which configurations to run.
    configs: list[str] = field(default_factory=lambda: [""])
    # Path to a file containing config strings to run, one per line.
    configs_file: str | None = None
    # Extra args passed to init method such as --ignore-depth-cache, not treated as part of config string. If passsed in without -- or -, will be added automatically.
    extra_args: list[str] = field(default_factory=list)
    # Output directory for results.
    output_dir: Path = Path("results")
    # Datasets to run on.
    datasets: list[str] = field(default_factory=list)
    # Scenes to run on, can be either dataset/scene format we use, or local paths.
    scenes: list[str] = field(default_factory=list)
    # Force overwrite existing results instead of backing them up.
    force_overwrite: bool = False

    # MLFlow experiment name to log to. If not provided, will use `{method}_init`.
    mlflow_experiment: str | None = None

    def get_method_config_class(self):
        if self.method == "monodepth":
            return MdTsdfFusionConfig
        elif self.method == "edgs":
            return EDGSConfig
        elif self.method == "da3":
            return DA3Config
        else:
            raise ValueError(f"Unknown init_method: {self.method}")

    def get_executable_name_for_init_method(self) -> str:
        if self.method == InitMethod.monodepth:
            return "monodepth"
        elif self.method == InitMethod.edgs:
            return "edgs"
        elif self.method == InitMethod.da3:
            return "da3_init"
        else:
            raise ValueError(f"Unknown init_method: {self.method}")


CONFIG_STR_FORBIDDEN_PARAM_NAMES = {
    "scene",
    "cache_dir",
    "base_output_dir",
    "ignore_depth_cache",
    "debug_output",
}

CONFIG_NAME_RENAMES = {
    "mdi.predictor": None,
    "mdi.depth-alignment-strategy": "align",
    "mdi.subsample-factor": "subsample",
}


def init_method_dir_needs_overwrite(
    output_dir: Path,
    args: InitRunnerArguments,
) -> bool:
    if args.force_overwrite:
        return True

    if not directory_exists_and_has_files(output_dir):
        return True

    required_files = {
        InitMethod.monodepth: [
            MONODEPTH_POINTS_FILE_NAME,
            MONODEPTH_NB_META_FILE_NAME,
        ],
        InitMethod.edgs: [
            EDGS_GAUSSIANS_FILE_NAME,
            EDGS_NB_META_FILE_NAME,
        ],
        InitMethod.da3: [
            DA3_POINTS_FILE_NAME,
            DA3_NB_META_FILE_NAME,
        ],
    }
    for file in required_files[args.method]:
        if not (output_dir / file).exists():
            return True

    return False


def add_missing_dashes(arg: str) -> str:
    if arg.startswith("-"):
        return arg
    if len(arg) == 1:
        return "-" + arg
    return "--" + arg


def run_init_method(
    scene: str,
    config: ParamList,
    args: InitRunnerArguments,
    results_dir: ResultsDirectory,
    subprocess_env: dict[str, str],
):
    config_name = config.make_config_name(CONFIG_NAME_RENAMES)

    init_out_dir = results_dir.get_init_method_output_dir(
        scene, config_name, args.method
    )
    init_out_dir.mkdir(parents=True, exist_ok=True)

    executable_name = args.get_executable_name_for_init_method()
    ansiesc_print(
        f"Running {executable_name} for '{config_name}' on {scene}. (Output dir: {init_out_dir})",
        ANSIEscapes.BLUE,
    )

    if directory_exists_and_has_files(init_out_dir) and not args.force_overwrite:
        if not init_method_dir_needs_overwrite(init_out_dir, args):
            ansiesc_print(
                f"Skipping point generation for {config_name} on {scene}. (Output exists)",
                ANSIEscapes.GREEN,
            )
            return
        else:
            new_path = rename_old_dir_with_timestamp(init_out_dir, results_dir)
            ansiesc_print(
                f"Detected incomplete point generation output. Old output directory moved to: {new_path}",
                ANSIEscapes.YELLOW,
            )
            assert not init_out_dir.exists()

    command = [
        executable_name,
        f"--scene={scene}",
        f"--output-dir={init_out_dir}",
    ] + [add_missing_dashes(arg) for arg in args.extra_args]

    if args.method == InitMethod.monodepth:
        command.append(f"--cache-dir={results_dir.get_monodepth_cache_dir()}")

    for name, value in config:
        command.append(f"--{name}")
        command.append(f"{value}")

    if os.environ.get("MLFLOW_RUN_ID") is not None:
        raise RuntimeError("MLFLOW_RUN_ID is already set in the environment.")
    mlflow_run_name = str(init_out_dir.relative_to(args.output_dir))

    with mlflow.start_run(run_name=mlflow_run_name) as mlflow_run:
        print(f"MLflow run ID: {mlflow_run.info.run_id}")
        subprocess_env["MLFLOW_RUN_ID"] = mlflow_run.info.run_id

        command_str_for_logging = (
            command[0] + " " + " ".join([f'"{x}"' for x in command[1:]])
        )

        ansiesc_print(f"Running {args.method}:", "bold")
        print(command_str_for_logging)
        mlflow.log_param("init_method_invocation", command_str_for_logging)

        start_time = datetime.now()
        rc, stderr_output = subprocess_run_tee_stderr(command, env=subprocess_env)
        end_time = datetime.now()

        mlflow.log_metric(
            "total_runtime_seconds", (end_time - start_time).total_seconds()
        )
        if stderr_output:
            mlflow.log_text(stderr_output, "nb_train_stderr.txt")
        if rc != 0:
            raise RuntimeError(
                f"Error during point generation for {config_name} on {scene}. (Exited with code {rc})"
            )


def get_method_config_overrides_suffix(method_config_overrides: list[str]) -> str:
    if len(method_config_overrides) == 0:
        return ""
    out = []
    for override in method_config_overrides:
        clean_override = override.replace("=", "-").replace(".", "-").replace(" ", "_")
        out.append(clean_override)
    return "_" + "_".join(out)


def process_combination(
    scene: str,
    param_list: ParamList,
    args: InitRunnerArguments,
    subprocess_env: dict[str, str],
):
    print(
        ANSIEscapes.format("_" * 80, "bold"),
        ANSIEscapes.format("=" * 80 + "\n", "blue"),
        sep="\n",
    )
    param_list.validate(CONFIG_STR_FORBIDDEN_PARAM_NAMES)
    results_dir = ResultsDirectory(args.output_dir)

    try:
        run_init_method(scene, param_list, args, results_dir, subprocess_env)
    except Exception as e:
        ansiesc_print(
            f"Exception occurred while processing {scene} with config {param_list.make_config_name(CONFIG_NAME_RENAMES)}: {e}",
            "red",
        )
        ansiesc_print("Proceeding to next combination.", "yellow")


def main():
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO)

    args = tyro.cli(InitRunnerArguments)

    subprocess_env = mlflow_runner_setup(
        args.output_dir, args.mlflow_experiment or f"{args.method.value}_init"
    )

    configs = load_configs(
        args.configs, args.configs_file, args.get_method_config_class()
    )
    scenes = get_scenes_from_args(args.scenes, args.datasets)

    combinations = sorted(list(product(scenes, configs)))
    combinations = select_task_subset_if_slurm(combinations)
    configs_for_print = {cfg for _, cfg in combinations}
    scenes_for_print = {str(scene) for scene, _ in combinations}
    print(
        ANSIEscapes.format("_" * 80, "bold"),
        ANSIEscapes.format(
            f"Will prepare point clouds for {len(combinations)} combinations using method {args.method}.",
            "bold",
        ),
        ANSIEscapes.format("Settings:", "bold"),
        f"\tOutput directory: {ANSIEscapes.format(args.output_dir, 'cyan')}",
        "\tConfigs: "
        + ANSIEscapes.format(
            "\n\t          ".join(
                c.make_config_name(CONFIG_NAME_RENAMES) for c in configs_for_print
            ),
            "cyan",
        ),
        f"\tScenes: {ANSIEscapes.format(scenes_for_print, 'cyan')}",
        sep="\n",
    )

    for scene, param_list in combinations:
        process_combination(scene, param_list, args, subprocess_env)


if __name__ == "__main__":
    main()
