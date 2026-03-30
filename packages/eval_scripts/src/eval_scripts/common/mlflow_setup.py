import os
import logging
from pathlib import Path
import subprocess

import mlflow

_LOGGER = logging.getLogger(__name__)


def mlflow_runner_setup(
    output_dir: Path | str,
    mlflow_experiment: str | None = None,
) -> dict[str, str]:
    """
    Sets up MLflow tracking for a script that will be run as a subprocess. This includes:
      - Setting the MLFLOW_TRACKING_URI to a SQLite database in the output directory if it's
        not already set in the environment.
      - Creating an MLflow experiment with the given name (or a default name) if it doesn't
        already exist, and setting the MLFLOW_EXPERIMENT_ID in the environment.
      - Handling the case where an experiment with the given name was previously deleted by running
        `mlflow gc` to clean it up so it can be recreated.

    Args:
        output_dir: The directory where MLflow artifacts and tracking database will be stored (the base output dir for ivd_splat_runner and init_runner).
        mlflow_experiment: Optional name of the MLflow experiment to use. If None, will use the MLFLOW_EXPERIMENT_NAME environment variable or "Default" if that is not set.
    Returns:
        A dictionary of environment variables that should be set for the subprocess to ensure it uses the correct MLflow tracking URI and experiment.
    """
    output_dir = Path(output_dir)

    subprocess_env = {**os.environ}
    if "MLFLOW_TRACKING_URI" not in os.environ:
        mlflow_tracking_uri = f"sqlite:///{output_dir.absolute()}/mlflow.db"
        subprocess_env["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    artifact_dir = output_dir.absolute() / "mlruns"

    if mlflow_experiment is not None:
        experiment_name = mlflow_experiment
    else:
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "Default")

    existing_experiment = mlflow.get_experiment_by_name(experiment_name)

    if (
        existing_experiment is not None
        and existing_experiment.lifecycle_stage == "deleted"
    ):
        _LOGGER.info(
            f"MLflow experiment {experiment_name} existed previously, but was deleted. Running mlflow gc to clean up."
        )
        # Clean up old deleted experiment to allow recreation.
        subprocess.run(
            ["mlflow", "gc", "--experiment-ids", existing_experiment.experiment_id],
            env=subprocess_env,
            check=True,
        )

    if existing_experiment is None:
        _LOGGER.info(
            f"Creating new MLflow experiment: {experiment_name} with artifacts stored in {artifact_dir}"
        )
        mlflow.create_experiment(
            name=experiment_name, artifact_location=artifact_dir.as_uri()
        )

    experiment = mlflow.set_experiment(experiment_name=experiment_name)
    subprocess_env["MLFLOW_EXPERIMENT_ID"] = experiment.experiment_id

    return subprocess_env
