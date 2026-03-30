from eval_scripts.results_processing.base import TABLE_METRICS
import pandas as pd


def add_metrics_per_x_gaussians(
    *data: pd.DataFrame,
    x=1e6,
    metrics=["eval-all-test/psnr", "eval-all-test/ssim", "eval-all-test/lpips"],
) -> list[str]:
    """
    Adds new columns to the dataframe for each metric divided by the number of gaussians (in millions).
    Modifies the dataframe in-place.

    Returns the list of new metric names added.
    """
    if x % 1e6 == 0:
        x_str = f"{int(x / 1e6)}M"
    elif x % 1e3 == 0:
        x_str = f"{int(x / 1e3)}K"
    else:
        x_str = f"{x:.0e}"

    def make_metric_name(metric_id: str) -> str:
        base_name = metric_id.split("/")[-1]
        if base_name == "lpips":
            base_name = "(1 - LPIPS)"
        return f"{base_name.upper()} per {x_str} Gaussians ↑"

    new_metric_names = []
    for metric in metrics:
        metric_name = make_metric_name(metric)
        new_metric_names.append(metric_name)
        for df in data:
            if "LPIPS" in metric_name:
                df[metric_name] = (1 - df[metric]) * x / df["train/num-gaussians"]
            else:
                df[metric_name] = df[metric] * x / df["train/num-gaussians"]
    return new_metric_names


def get_metrics_as_fraction_of_sfm(
    sfm: pd.DataFrame, *other_data: pd.DataFrame, metrics=TABLE_METRICS
) -> list[pd.DataFrame]:
    """
    For each metric, divides the values in the other_data dataframes by the corresponding value in the sfm dataframe, to get a percentage of sfm performance.
    Returns a list of new dataframes with the same metrics as original but with values as percentage of sfm.
    """
    retval: list[pd.DataFrame] = []
    for df in other_data:
        new_df = pd.DataFrame(index=df.index)
        for metric in metrics:
            if metric in df.index:
                new_df[metric] = df[metric] / sfm[metric]
            else:
                print(f"Warning: Metric {metric} not found in dataframe. Skipping.")
        retval.append(new_df)
    return retval
