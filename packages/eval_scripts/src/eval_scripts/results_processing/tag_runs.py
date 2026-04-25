from typing import Callable, Collection

from eval_scripts.results_processing.base import RunsInfo, boolean_conversion
import mlflow
import pandas as pd


class MLFlowTagger:
    def __init__(self, tracking_uri: str | None = None):
        self.client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    def set_tag(
        self, run_id: str | Collection[str] | pd.DataFrame, tag_key: str, tag_value: str
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
        run_id: str | Collection[str] | pd.DataFrame,
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


_CONVERSIONS = {
    "init_size_matches_sfm": boolean_conversion(default=False),
    "init_size_matches_real_init": boolean_conversion(default=False),
    "init_size_matches_gmax": boolean_conversion(default=False),
    "is_default_strategy_config": boolean_conversion(default=False),
}


def filter_and_tag_runs(
    runs: RunsInfo,
    tracking_uri: str | None,
    gmax_per_scene: dict[str, int],
    sfm_init_num_pts_per_scene: dict[str, int],
    real_init_num_pts_per_scene: dict[str, int],
    get_default_strategy_args: Callable[[str, str], dict[str, str]],
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

    def tag(selection: pd.Series, tag_key: str, tag_value: str):
        runs.df.loc[selection, tag_key] = _CONVERSIONS.get(tag_key, lambda x: x)(
            tag_value
        )

        if tag_in_db:
            tagger.set_tag(runs.df.loc[selection, "run_id"], tag_key, tag_value)

    df = runs.df
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
    tag(init_size_matches_sfm, "init_size_matches_sfm", "1")
    tag(init_size_matches_real_init, "init_size_matches_real_init", "1")
    tag(init_size_matches_gmax, "init_size_matches_gmax", "1")

    # Tag anomaly types
    tag(~is_base_sfm_with_no_cap & ~cap_max_matches, "anomaly_type", "gmax_mismatch")
    tag(
        ~is_base_sfm_with_no_cap & ~init_pts_matches,
        "anomaly_type",
        "init_size_mismatch",
    )
    tag(
        ~is_base_sfm_with_no_cap & ~cap_max_matches & ~init_pts_matches,
        "anomaly_type",
        "gmax_and_init_size_mismatch",
    )

    # Filter out runs marked as anomalous (gmax mismatch/init size mismatch)
    is_anomalous = df["anomaly_type"].notna()
    print(f"Filtering out {is_anomalous.sum()} anomalous runs.")
    df = df[~is_anomalous]
    runs.df = df

    # dataset is first part of scene before '/'
    datasets = df["scene"].apply(lambda x: x.split("/")[0]).unique()
    for dataset in datasets:
        for strategy in df["strategy"].unique():
            default_strategy_runs = runs.get_runs_with_params(
                get_default_strategy_args(strategy, dataset)
            )
            tag(
                df["run_id"].isin(default_strategy_runs.df["run_id"])
                & df["strategy"].eq(strategy)
                & df["scene"].str.startswith(dataset + "/"),
                "is_default_strategy_config",
                "1",
            )

    is_sfm_baseline = (
        df["init_method"].eq("sfm")
        & df["is_default_strategy_config"].eq(True)
        & df["gaussian_cap_fraction"].eq("1.0")
    )
    tag(is_sfm_baseline, "init_group", "sfm_baseline")

    return runs
