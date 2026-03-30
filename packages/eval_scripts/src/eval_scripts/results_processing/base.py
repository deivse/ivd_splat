from dataclasses import dataclass
import datetime
import itertools
import typing
from eval_scripts.ivd_splat_runner import MAX_STEPS
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.entities
import numpy as np
import pandas as pd
import logging

TABLE_METRICS = [f"eval-all-test/{metric}" for metric in ["psnr", "ssim", "lpips"]] + [
    "train/num-gaussians",
    "train/total-train-time",
]

METRIC_PRETTY_NAMES = {
    "eval-all-test/psnr": "PSNR ↑",
    "eval-all-test/ssim": "SSIM ↑",
    "eval-all-test/lpips": "LPIPS ↓",
    "train/num-gaussians": "Num Gaussians",
    "train/total-train-time": "Total Train Time (min)",
}
PER_SCENE_VARYING_PARAMS = {"scene", "dense_init.target_num_points", "strategy.cap_max"}

CONVERSIONS: dict[str, typing.Callable[[typing.Any], typing.Any]] = {
    "train/num-gaussians": int,
    "train/total-train-time": lambda x: float(x) / 60 if x is not None else None,
    "gaussian_cap_fraction": lambda x: str(x) if x is not None else str(1.0),
    "init_size_same_as_sfm": lambda x: x == "true" if x is not None else False,
    "splat_init.increase_scale_with_fewer_splats": lambda x: (
        x.lower() == "true" if x is not None else False
    ),
    "splat_init.target_splat_fraction": lambda x: x if x is not None else str(1.0),
    "dense_init.target_points_fraction": lambda x: x if x is not None else str(1.0),
    "means_lr_init": lambda x: x if x is not None else "0.00016",
    "means_lr_final": lambda x: x if x is not None else "1.6000000000000001e-06",
}


@dataclass
class RunsInfo:
    df: pd.DataFrame
    param_names: set[str]
    metric_names: set[str]

    def get_runs_with_params(self, params: dict[str, typing.Any]):
        filtered_runs = self.df
        for param_name, param_value in params.items():
            if param_value is not None:
                if param_name not in filtered_runs.columns:
                    logging.warning(
                        "Parameter '%s' not found in runs dataframe columns. Ignoring this parameter in filtering.",
                        param_name,
                    )
                    continue
                filtered_runs = pd.DataFrame(
                    filtered_runs[filtered_runs[param_name] == param_value]
                )
            elif param_name in filtered_runs.columns:
                filtered_runs = pd.DataFrame(
                    filtered_runs[filtered_runs[param_name].isnull()]
                )
            else:
                # Copy to avoid potentially mutating self.df
                filtered_runs = filtered_runs.copy()
        return RunsInfo(
            df=filtered_runs,
            param_names=self.param_names,
            metric_names=self.metric_names,
        )

    def copy(self) -> "RunsInfo":
        """Returns a copy of this RunsInfo with a copy of the underlying dataframe."""
        return RunsInfo(
            df=self.df.copy(),
            param_names=self.param_names.copy(),
            metric_names=self.metric_names.copy(),
        )

    def get_params_differing_across_runs(
        self,
    ) -> dict[str, list[str]]:
        """
        Returns a dictionary that maps parameter names to lists of unique values for parameters that differ across any of the runs in self.
        """
        unique_hyperparam_combinations = self.df[
            list(self.param_names.difference(PER_SCENE_VARYING_PARAMS))
        ].drop_duplicates()
        differing_params: dict[str, list[str]] = {}
        for param in unique_hyperparam_combinations.columns:
            vals = unique_hyperparam_combinations[param].unique()
            if len(vals) > 1:
                differing_params[param] = vals.tolist()
        return differing_params

    def describe(self) -> None:
        """Prints a summary of the available runs, including the number of runs, parameter names, metric names, and unique values for parameters that differ across runs."""
        scenes: set[str] = set(self.df["scene"].unique())

        print(f"Available runs: {len(self.df)}")
        print("Datasets:")
        datasets = set(scene.split("/")[0] for scene in scenes)
        max_scenes_to_print = 10
        for dataset in datasets:
            dataset_scenes = sorted(
                scene.split("/")[1]
                for scene in scenes
                if scene.startswith(dataset + "/")
            )
            print(
                f"- {dataset} ({len(dataset_scenes)}): {', '.join(dataset_scenes[:max_scenes_to_print])}{'...' if len(dataset_scenes) > max_scenes_to_print else ''}"
            )

        print("Differing parameters:")
        differing_params = self.get_params_differing_across_runs()
        for param, vals in differing_params.items():
            try:
                vals = sorted(vals)
            except TypeError:
                pass  # If values are not sortable, just print them as they are
            print(f"- {param}: {vals}")

    def get_per_scene_metrics_for_params(
        self,
        params: dict[str, typing.Any],
        metrics=TABLE_METRICS,
        ignore_differing_params: set[str] | None = None,
    ) -> pd.DataFrame:
        """
        For a given set of parameters, returns a table with one row per scene and columns for each metric in TABLE_METRICS,
        containing the average metric value across all runs with those parameters for that scene.
        """
        runs_with_params = self.get_runs_with_params(params)
        if runs_with_params.df.empty:
            logging.warning(
                "No runs found with parameters %s. Returning empty dataframe.",
                params,
            )
            return pd.DataFrame(columns=["scene"] + metrics)

        differing_param_values = runs_with_params.get_params_differing_across_runs()
        if ignore_differing_params is not None:
            differing_param_values = {
                param: vals
                for param, vals in differing_param_values.items()
                if param not in ignore_differing_params
            }
        if differing_param_values != {}:
            logging.error(
                f"Failed to get per-scene metrics - some parameters have differing values across runs:\n{differing_param_values}"
            )
            raise ValueError(
                f"Failed to get per-scene metrics. Runs with query {params} do not have identical hyperparameter combinations."
            )

        # Check same number of runs for each scene
        num_runs_per_scene = runs_with_params.df["scene"].value_counts()
        if (
            num_runs_per_scene.size > 0
            and not (num_runs_per_scene == num_runs_per_scene.iloc[0]).all()
        ):
            logging.warning(
                "Different number of runs per scene detected: %s",
                num_runs_per_scene.unique(),
            )

        # Table with scenes in first column and metrics in other columns
        per_scene_metrics = pd.DataFrame()

        per_scene_metrics["scene"] = runs_with_params.df["scene"].unique()
        for metric in metrics:
            per_scene_metrics[metric] = per_scene_metrics["scene"].apply(
                lambda scene: (
                    runs_with_params.df[runs_with_params.df["scene"] == scene][
                        metric
                    ].item()
                    if len(runs_with_params.df[runs_with_params.df["scene"] == scene])
                    > 0
                    else np.nan
                )
            )
        per_scene_metrics = per_scene_metrics.set_index("scene")

        return per_scene_metrics


def get_common_scenes(
    *per_scene_data: pd.DataFrame,
) -> set[str]:
    scene_sets = [set(data.index) for data in per_scene_data]
    common_scenes = set.intersection(*scene_sets)
    all_scenes = set.union(*scene_sets)
    if len(common_scenes) < len(all_scenes):
        logging.warning(
            "Not all runs have the same scenes. Common scenes: %d, total unique scenes: %d",
            len(common_scenes),
            len(all_scenes),
        )
        logging.warning(
            "Missing scenes per dataframe:\n%s",
            "\n".join(
                f"\tDataframe {i}: {all_scenes - scene_set}"
                for i, scene_set in enumerate(scene_sets)
                if all_scenes - scene_set
            ),
        )
    return common_scenes


def drop_scenes_not_present_in_all(
    *per_scene_data: pd.DataFrame,
) -> tuple[set[str], set[str]]:
    """
    Drops data for scenes that are not present in all provided dataframes.
    Modifies the dataframes in-place.
    Returns: tuple containing:
        - set of scenes that are present in all dataframes (common scenes)
        - set of scenes that were dropped (not common scenes)
    """
    common_scenes = get_common_scenes(*per_scene_data)
    if not common_scenes:
        logging.error(
            "No common scenes found across runs. Scenes per dataframe:\n%s",
            "\n".join(
                f"Dataframe {i}: {set(data.index)}"
                for i, data in enumerate(per_scene_data)
            ),
        )
        raise ValueError("No common scenes found across runs.")

    print(f"Common scenes ({len(common_scenes)}): " + ", ".join(common_scenes))

    # in-place filtering of each dataframe to only include common scenes
    for data in per_scene_data:
        data.drop(index=data.index.difference(list(common_scenes)), inplace=True)
    dropped_scenes = (
        set.union(*[set(data.index) for data in per_scene_data]) - common_scenes
    )
    return (common_scenes, dropped_scenes)


def load_runs(
    query: str | None = None,
    experiment_name: str = "gt_pointclouds",
    tracking_uri: str = "http://localhost:6069",
    finished_run_step: int = MAX_STEPS,
    finished_only: bool = True,
    created_after: datetime.datetime | None = None,
) -> RunsInfo:
    mlflow.set_tracking_uri(tracking_uri)

    # Initialize MLflow client
    client = MlflowClient()
    # Specify the experiment name

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found with tracking URI '{tracking_uri}'"
        )

    runs: list[mlflow.entities.Run] = list(
        client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(query or ""),
            max_results=np.iinfo(np.int16).max,
        )
    )

    finished_runs = [
        run
        for run in runs
        if run.info.status == "FINISHED"
        and run.data.metrics["final_step"] == finished_run_step
        and (
            created_after is None
            or datetime.datetime.fromtimestamp(run.info.start_time / 1000)
            > created_after
        )
    ]

    if finished_only:
        logging.info(f"Loading only finished runs ({len(finished_runs)}/{len(runs)})")
        runs = finished_runs
    else:
        logging.info(f"Loading all runs ({len(runs)})")

    param_names = {k for run in finished_runs for k in run.data.params.keys()}
    metric_names = {k for run in finished_runs for k in run.data.metrics.keys()}

    runs_dataframe = pd.DataFrame(
        [
            dict(**run.data.params, **run.data.metrics, run_id=run.info.run_id)
            for run in runs
        ]
    )

    def _make_resilient_converter(
        conversion_fn: typing.Callable[[typing.Any], typing.Any],
    ) -> typing.Callable[[typing.Any], typing.Any]:
        def converter(value: typing.Any) -> typing.Any:
            if (
                value is None
                or (isinstance(value, float) and value != value)  # NaN check
                or (isinstance(value, str) and value.lower() == "none")
            ):
                return conversion_fn(None)
            return conversion_fn(value)

        return converter

    for column_name, conversion_fn in CONVERSIONS.items():
        if column_name in runs_dataframe.columns:
            try:
                runs_dataframe[column_name] = runs_dataframe[column_name].apply(
                    _make_resilient_converter(conversion_fn)
                )
            except Exception as e:
                raise ValueError(
                    f"Failed to convert column '{column_name}' using {conversion_fn}: {e}"
                ) from e

    return RunsInfo(
        df=runs_dataframe, param_names=param_names, metric_names=metric_names
    )


def load_init_method_runs(
    experiment_name: str,
    query: str | None = None,
    tracking_uri: str = "http://localhost:6069",
    finished_only: bool = True,
    created_after: datetime.datetime | None = None,
) -> RunsInfo:
    mlflow.set_tracking_uri(tracking_uri)

    # Initialize MLflow client
    client = MlflowClient()
    # Specify the experiment name

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' not found with tracking URI '{tracking_uri}'"
        )

    runs: list[mlflow.entities.Run] = list(
        client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(query or ""),
            max_results=np.iinfo(np.int16).max,
        )
    )

    finished_runs = [
        run
        for run in runs
        if run.info.status == "FINISHED"
        and (
            created_after is None
            or datetime.datetime.fromtimestamp(run.info.start_time / 1000)
            > created_after
        )
    ]

    if finished_only:
        logging.info(f"Loading only finished runs ({len(finished_runs)}/{len(runs)})")
        runs = finished_runs
    else:
        logging.info(f"Loading all runs ({len(runs)})")

    param_names = {k for run in finished_runs for k in run.data.params.keys()}
    metric_names = {k for run in finished_runs for k in run.data.metrics.keys()}

    runs_dataframe = pd.DataFrame(
        [
            dict(**run.data.params, **run.data.metrics, run_id=run.info.run_id)
            for run in runs
        ]
    )

    return RunsInfo(
        df=runs_dataframe, param_names=param_names, metric_names=metric_names
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    runs = load_runs()
