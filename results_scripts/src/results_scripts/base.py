from dataclasses import dataclass
import typing
from eval_scripts.ivd_splat_runner import MAX_STEPS
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.entities
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import pickle

from results_scripts.constants import (
    DATASET_SCENE_SELECTION,
    DEFAULT_TABLE_METRICS,
    PER_SCENE_VARYING_PARAMS,
    get_default_strategy_args,
)
from results_scripts.param_conversions import (
    INIT_METHOD_PARAM_CONVERSIONS,
    PARAM_CONVERSIONS,
    boolean_conversion,
)
from results_scripts.utils import name_to_path


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
            if filtered_runs.empty:
                logging.warning(
                    "Filtered runs empty after filtering for parameter '%s' = %s.",
                    param_name,
                    param_value,
                )
                break
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
        metrics=DEFAULT_TABLE_METRICS,
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
        differing_param_values.pop("eval_iter", None)

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
                "Different number of runs per scene detected for scenes: %s",
                num_runs_per_scene.unique(),
            )
            logging.warning("Params: %s", params)
            baseline_num_runs = num_runs_per_scene.max()
            logging.warning(
                "Scenes with fewer_runs than baseline (%d):", baseline_num_runs
            )
            for scene, count in num_runs_per_scene.items():
                if count < baseline_num_runs:
                    logging.warning(" - %s: %d runs", scene, count)

        if "eval_iter" in runs_with_params.df.columns:
            num_runs_per_eval_iter = runs_with_params.df["eval_iter"].value_counts()
            if (
                num_runs_per_eval_iter.size > 0
                and not (num_runs_per_eval_iter == num_runs_per_eval_iter.iloc[0]).all()
            ):
                logging.warning(
                    "Different number of runs per eval_iter detected: %s",
                    num_runs_per_eval_iter.unique(),
                )
        else:
            logging.warning("'eval_iter' column not found in runs dataframe.")

        # Table with scenes in first column and metrics in other columns
        per_scene_metrics = pd.DataFrame()

        per_scene_metrics["scene"] = runs_with_params.df["scene"].unique()

        def get_metric_for_scene(scene: str, metric: str) -> float:
            scene_runs = runs_with_params.df[runs_with_params.df["scene"] == scene]
            if len(scene_runs) == 0:
                return np.nan
            # Sort by seed so per-scene value arrays share a consistent eval_iter
            # ordering across columns, enabling seed-paired comparisons downstream.
            if "eval_iter" in scene_runs.columns:
                scene_runs = scene_runs.sort_values("eval_iter")
            return scene_runs[metric].to_numpy()

        for metric in metrics:
            per_scene_metrics[metric] = per_scene_metrics["scene"].apply(
                lambda scene: get_metric_for_scene(scene, metric)
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
    *per_scene_data: pd.DataFrame, debug_out: bool = True
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

    if debug_out:
        print(f"Common scenes ({len(common_scenes)}): " + ", ".join(common_scenes))

    # in-place filtering of each dataframe to only include common scenes
    for data in per_scene_data:
        data.drop(index=data.index.difference(list(common_scenes)), inplace=True)
    dropped_scenes = (
        set.union(*[set(data.index) for data in per_scene_data]) - common_scenes
    )
    return (common_scenes, dropped_scenes)


def _build_runs_query(
    input_query: str | None, finished_only: bool, finished_query: str
) -> str:
    query = input_query or ""
    if finished_only:
        if query != "":
            query += " and "
        query += finished_query
    return query


def _get_full_runs_list(
    client: MlflowClient, experiment: mlflow.entities.Experiment, query: str | None
) -> list[mlflow.entities.Run]:
    runs: list[mlflow.entities.Run] = []
    while True:
        paged_list = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(query or ""),
            max_results=np.iinfo(np.int16).max,
        )
        runs.extend(paged_list)
        if not paged_list.token:
            break
        logging.info(
            f"Fetched {len(runs)} runs so far, fetching more with token {paged_list.token}..."
        )
        paged_list = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=(query or ""),
            max_results=np.iinfo(np.int16).max,
            page_token=paged_list.token,
        )
        if not paged_list:
            break
    return runs


def load_runs(
    query: str | None = None,
    experiment_name: str = "main",
    tracking_uri: str = "http://localhost:6069",
    finished_run_step: int = MAX_STEPS,
    finished_only: bool = True,
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

    query = _build_runs_query(
        query,
        finished_only,
        f"metrics.final_step = {finished_run_step} and attributes.status = 'FINISHED'",
    )
    runs = _get_full_runs_list(client, experiment, query)

    param_names = {k for run in runs for k in run.data.params.keys()}
    metric_names = {k for run in runs for k in run.data.metrics.keys()}

    runs_dataframe = pd.DataFrame(
        [
            dict(
                **run.data.params,
                **run.data.metrics,
                run_id=run.info.run_id,
            )
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

    for column_name, conversion_fn in PARAM_CONVERSIONS.items():
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

    query = _build_runs_query(
        query,
        finished_only,
        "attributes.status = 'FINISHED'",
    )
    runs = _get_full_runs_list(client, experiment, query)

    param_names = {k for run in runs for k in run.data.params.keys()}
    metric_names = {k for run in runs for k in run.data.metrics.keys()}

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

    for column_name, conversion_fn in INIT_METHOD_PARAM_CONVERSIONS.items():
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


def get_cache_dir(script_dir: Path, tracking_uri: str) -> Path:
    return script_dir / "cache" / name_to_path(tracking_uri, allow_subdirs=False)


def load_cached_runs(cache_path: Path) -> RunsInfo:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    return RunsInfo(
        df=payload["df"],
        param_names=set(payload["param_names"]),
        metric_names=set(payload["metric_names"]),
    )


def save_cached_runs(cache_path: Path, runs: RunsInfo) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as handle:
        pickle.dump(
            {
                "df": runs.df,
                "param_names": sorted(runs.param_names),
                "metric_names": sorted(runs.metric_names),
            },
            handle,
        )


def load_or_download_runs(
    cache_path: Path,
    loader,
    download: bool,
    label: str,
) -> RunsInfo:
    if cache_path.exists() and not download:
        print(f"Loading cached {label} from {cache_path}")
        return load_cached_runs(cache_path)

    print(f"Downloading {label}")
    runs = loader()
    save_cached_runs(cache_path, runs)
    print(f"Cached {label} at {cache_path}")
    return runs


class MLFlowTagger:
    def __init__(self, tracking_uri: str | None = None):
        self.client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    def set_tag(
        self,
        run_id: str | typing.Collection[str] | pd.DataFrame,
        tag_key: str,
        tag_value: str,
    ):
        if isinstance(run_id, pd.DataFrame):
            run_id = run_id["run_id"]

        if isinstance(run_id, str):
            print(f"Setting tag {tag_key}={tag_value} for run {run_id}")
            self.client.set_tag(run_id, tag_key, tag_value)
        else:
            print(f"Setting tag {tag_key}={tag_value} for {len(run_id)} runs.")
            for rid in run_id:
                self.client.set_tag(rid, tag_key, tag_value)

    def delete_tag(
        self,
        run_id: str | typing.Collection[str] | pd.DataFrame,
        tag_key: str,
        should_print=True,
    ):
        if isinstance(run_id, pd.DataFrame):
            run_id = run_id["run_id"]

        if isinstance(run_id, str):
            should_print and print(f"Deleting tag {tag_key} for run {run_id}")
            try:
                self.client.delete_tag(run_id, tag_key)
            except mlflow.exceptions.RestException as e:
                if e.error_code != "RESOURCE_DOES_NOT_EXIST":
                    raise
        else:
            should_print and print(f"Deleting tag {tag_key} for {len(run_id)} runs.")
            for rid in run_id:
                self.delete_tag(rid, tag_key, should_print=False)


def filter_and_tag_runs(
    runs: RunsInfo,
    tracking_uri: str | None,
    gmax_per_scene: dict[str, int],
    sfm_init_num_pts_per_scene: dict[str, int],
    real_init_num_pts_per_scene: dict[str, int],
    get_default_strategy_args: typing.Callable[[str, str], dict[str, str]],
    tag_in_db: bool = False,
) -> RunsInfo:
    """
    Tags runs with the following tags:
        - init_size_matches_sfm: whether the target number of points in the dense initialization matches the number of points in the SFM initialization (or if the init method is sfm, in which case we consider it a match)
        - init_size_matches_real_init: whether the target number of points in the dense initialization matches the provided real init size for the scene
        - init_size_matches_gmax: whether the target number of points in the dense initialization matches the gmax for the scene
        - anomaly_type: if the run is not a base sfm run with no cap and either the cap max or the init size does not match the expected values, then this tag indicates whether it's a gmax mismatch, an init size mismatch, or both
        - is_default_strategy_config: whether the run uses the default strategy config for its strategy (e.g. for MCMCStrategy, this means not having overridden any of the default values for the strategy's config)
        - init_group: currently only tags as "sfm_baseline" for runs which are the base SfM run with a given strategy (without cap for AbsGS and with GMax cap for others)

    Arguments:
        runs: the runs to tag
        tracking_uri: the tracking URI of the MLFlow server where the runs are logged, used to set tags in the database if tag_in_db is True
        gmax_per_scene: a dictionary mapping each scene to its gmax value (strategy.cap_max)
        sfm_init_num_pts_per_scene: a dictionary mapping each scene to the number of points produced by SfM init
        real_init_num_pts_per_scene: a dictionary mapping each scene to the real init size min(monodepth_size, edgs_size, gmax)
        get_default_strategy_args: (strategy_name, dataset) -> default args for the strategy.
        tag_in_db: whether to set the tags in the MLFlow database (in addition to tagging the runs.df).
                   Setting tags in the database allows them to be visible in the MLFlow UI and used for filtering runs there,
                   but is also much slower than just tagging the runs.df.
    """
    tagger = MLFlowTagger(tracking_uri=tracking_uri)
    _CONVERSIONS = {
        "init_size_matches_sfm": boolean_conversion(default=False),
        "init_size_matches_real_init": boolean_conversion(default=False),
        "init_size_matches_gmax": boolean_conversion(default=False),
        "is_default_strategy_config": boolean_conversion(default=False),
        "is_default_init_config": boolean_conversion(default=False),
    }

    def tag(df: pd.DataFrame, selection: pd.Series, tag_key: str, tag_value: str):
        df.loc[selection, tag_key] = _CONVERSIONS.get(tag_key, lambda x: x)(tag_value)

        if tag_in_db:
            tagger.set_tag(df.loc[selection, "run_id"], tag_key, tag_value)

    df = runs.df.copy()

    is_base_sfm_with_no_cap = (
        (df["init_method"] == "sfm")
        & (df["strategy"] == "DefaultWithGaussianCapStrategy")
        & (df["strategy.cap_max"] == "-1")
    )

    cap_max_matches = df["strategy.cap_max"].isna() | (
        df["strategy.cap_max"].fillna("-1").astype(int)
        == (
            df["scene"].map(gmax_per_scene)
            * df["gaussian_cap_fraction"].fillna("1.0").astype(float)
        ).astype(int)
    )

    target_num_pts_int = (
        df["dense_init.target_num_points"].replace("None", "-1").astype(int)
    )
    init_size_matches_sfm = (
        target_num_pts_int == df["scene"].map(sfm_init_num_pts_per_scene)
    ) | (df["init_method"] == "sfm")

    init_size_matches_real_init = target_num_pts_int == df["scene"].map(
        real_init_num_pts_per_scene
    )
    init_size_matches_gmax = target_num_pts_int == df["scene"].map(gmax_per_scene)

    init_pts_matches = (
        init_size_matches_sfm | init_size_matches_real_init | init_size_matches_gmax
    )

    # Tag init size matches
    tag(df, init_size_matches_sfm, "init_size_matches_sfm", "1")
    tag(df, init_size_matches_real_init, "init_size_matches_real_init", "1")
    tag(df, init_size_matches_gmax, "init_size_matches_gmax", "1")

    # Tag anomaly types
    tag(
        df, ~is_base_sfm_with_no_cap & ~cap_max_matches, "anomaly_type", "gmax_mismatch"
    )
    tag(
        df,
        ~is_base_sfm_with_no_cap & ~init_pts_matches,
        "anomaly_type",
        "init_size_mismatch",
    )
    tag(
        df,
        ~is_base_sfm_with_no_cap & ~cap_max_matches & ~init_pts_matches,
        "anomaly_type",
        "gmax_and_init_size_mismatch",
    )

    # Filter out runs marked as anomalous (gmax mismatch/init size mismatch)
    is_anomalous = df["anomaly_type"].notna()
    print(f"Filtering out {is_anomalous.sum()} anomalous runs.")
    df = df[~is_anomalous].copy()
    runs.df = df

    # dataset is first part of scene before '/'
    datasets = df["scene"].apply(lambda x: x.split("/")[0]).unique()
    for dataset in datasets:
        for strategy in df["strategy"].unique():
            default_strategy_runs = runs.get_runs_with_params(
                get_default_strategy_args(strategy, dataset)
            )
            default_ids = default_strategy_runs.df["run_id"].copy()
            subset_mask = df["strategy"].eq(strategy) & df["scene"].str.startswith(
                dataset + "/"
            )
            is_in_default = df["run_id"].isin(default_ids)
            default_mask = is_in_default & subset_mask
            nondefault_mask = ~is_in_default & subset_mask
            tag(
                df,
                default_mask,
                "is_default_strategy_config",
                "1",
            )
            tag(
                df,
                nondefault_mask,
                "is_default_strategy_config",
                "0",
            )

    def col_or_nones(df: pd.DataFrame, col_name: str) -> pd.Series:
        if col_name in df.columns:
            return df[col_name]
        else:
            return pd.Series([None] * len(df), index=df.index)

    is_default_init_config = (
        (df["dense_init.sampling"] == "uniform")
        & col_or_nones(df, "init.scale_color_dist_factor").isna()
        & col_or_nones(df, "init.target_median_scale").isna()
        & col_or_nones(df, "splat_init.opacity_uniform_override").isna()
        & col_or_nones(df, "splat_init.opacity_noise_std").isna()
        & col_or_nones(df, "splat_init.init_scale_with_knn").fillna("False").eq("False")
        & col_or_nones(df, "splat_init.init_scale_isotropic_mean")
        .fillna("False")
        .eq("False")
        & col_or_nones(df, "splat_init.simulate_point_init").fillna("False").eq("False")
        & col_or_nones(df, "splat_init.scale_noise_std_wrt_median").isna()
        & col_or_nones(df, "splat_init.rotation_noise_angle_std_deg").isna()
        & col_or_nones(df, "splat_init.color_noise_std").isna()
    )
    tag(df, ~is_default_init_config, "is_default_init_config", "0")
    tag(df, is_default_init_config, "is_default_init_config", "1")

    is_sfm_baseline = (
        df["init_method"].eq("sfm")
        & df["is_default_strategy_config"].eq(True)
        & df["gaussian_cap_fraction"].eq("1.0")
        & df["is_default_init_config"].eq(True)
    )

    tag(df, is_sfm_baseline, "init_group", "sfm_baseline")
    tag(df, ~is_sfm_baseline, "init_group", "None")
    runs.df = df

    return runs


def load_and_prepare_dataset_runs(
    dataset: str,
    tracking_uri: str,
    main_experiment_name: str,
    num_pts_per_scene: dict[str, int],
    sfm_init_num_pts_per_scene: dict[str, int],
    real_init_num_pts_per_scene: dict[str, int],
    max_eval_iter: int | None = None,
) -> RunsInfo:
    runs = load_runs(
        query=f"params.scene like '{dataset}%'",
        tracking_uri=tracking_uri,
        experiment_name=main_experiment_name,
    )

    if dataset in DATASET_SCENE_SELECTION:
        scenes = DATASET_SCENE_SELECTION[dataset]
        runs.df = runs.df[runs.df["scene"].isin(scenes)]

    runs.df["init_method"] = runs.df["init_method"].replace(
        "gt_pointcloud", "laser_scan"
    )
    runs.df["dense_init.knn_num_neighbors"] = runs.df[
        "dense_init.knn_num_neighbors"
    ].replace("4", "3")

    # legacy, steps_scaler was never used, but the default changed or smth
    params_to_remove = ["init_size_same_as_sfm", "steps_scaler"]
    params_to_remove += ["gsplat_version"]  # identical across all used data for now
    params_to_remove += ["random_seed"]  # correlates with eval_iter directly
    # default changed, adaptive sampling not used anyways
    params_to_remove += ["dense_init.knn_num_neighbors"]
    params_to_remove += [
        param for param in runs.param_names if param.startswith("nanogs_config.")
    ]
    runs.param_names = runs.param_names - set(params_to_remove)
    runs.df = runs.df.loc[:, ~runs.df.columns.isin(params_to_remove)]

    if max_eval_iter is not None:
        runs.df = runs.df[runs.df["eval_iter"] <= max_eval_iter]

    runs.df.loc[
        runs.df["init_type"] != "splat",
        "splat_init.increase_scale_with_fewer_splats",
    ] = True

    runs = filter_and_tag_runs(
        runs,
        tracking_uri=tracking_uri,
        gmax_per_scene=num_pts_per_scene,
        sfm_init_num_pts_per_scene=sfm_init_num_pts_per_scene,
        real_init_num_pts_per_scene=real_init_num_pts_per_scene,
        get_default_strategy_args=get_default_strategy_args,
    )

    params_per_run = runs.df[list(runs.param_names)]
    identical_runs = params_per_run.duplicated(keep="last")
    if identical_runs.any():
        print("Num duplicates: ", identical_runs.sum())
    runs.df = runs.df[~identical_runs]
    print("Num runs after dropping duplicates: ", len(runs.df))
    print(
        f"Dataset: {dataset}, num EDGS runs: {(runs.df['init_method'] == 'edgs').sum()}, "
        f"num monodepth runs: {(runs.df['init_method'] == 'monodepth').sum()}"
    )
    return runs
