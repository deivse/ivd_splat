import contextlib
import mlflow
from nerfbaselines.logging import BaseLogger, BaseLoggerEvent
from nerfbaselines.utils import convert_image_dtype

from typing import Any, Dict, Optional, Union
import numpy as np

from mlflow_nerfbaselines_logger.patch_mlflow_log_image import patch_mlflow_log_image


class MLflowLoggerEvent(BaseLoggerEvent):
    def __init__(self, run, step):
        self._run = run
        self._step = step

    def add_scalar(self, tag: str, value: Union[float, int]) -> None:
        mlflow.log_metric(tag, value, step=self._step)

    def add_text(self, tag: str, text: str) -> None:
        mlflow.log_text(text, f"{tag}_step_{self._step}.txt")

    def add_image(
        self,
        tag: str,
        image: np.ndarray,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        **kwargs,
    ) -> None:
        image = convert_image_dtype(image, np.uint8)
        # Apparently both the patch and this replace are required for image logging to work....
        mlflow.log_image(image, key=tag.replace("/", "_"), step=self._step)

    def add_histogram(self, tag, values, *, num_bins=None):
        raise NotImplementedError("MLflowLogger does not support histograms.")


class MLflowLogger(BaseLogger):
    def __init__(self, *args, **kwargs):
        # thx, mlflow https://github.com/mlflow/mlflow/issues/12151
        patch_mlflow_log_image()

        # Runs are started in ivd_splat_runner.py

        # If some other part of the code already resumed it, then use that.
        self._run = mlflow.active_run()
        # Otherwise, resume the run ourselves.
        if self._run is None:
            self._run = mlflow.start_run()

    @contextlib.contextmanager
    def add_event(self, step: int):
        yield MLflowLoggerEvent(self._run, step)

    def add_hparams(self, hparams: Dict[str, Any]):
        raise NotImplementedError("MLflowLogger does not support hparams.")

    def __str__(self):
        return "mlflow"
