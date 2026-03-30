import dataclasses
import os
import logging

import mlflow

_LOGGER = logging.getLogger(__name__)


class SerializableConfig:
    """
    Utility class for configs that consist of (potentially nested) dataclasses and
    should be serializable into dictionaries. Provides methods for serialization and
    for getting a flat dictionary of stringified parameters (e.g. for logging to MLflow).
    """

    def to_dict(self) -> dict:
        """Serialize the config to a dictionary."""
        ignored_fields: set[str] = getattr(
            self, "CONFIG_SERIALIZATION_IGNORED_FIELDS", set()
        )

        retval = {}
        for field in dataclasses.fields(self):  # type: ignore
            if field.name in ignored_fields:
                continue
            value = getattr(self, field.name)
            if dataclasses.is_dataclass(value):
                if isinstance(value, SerializableConfig):
                    retval[field.name] = value.to_dict()
                else:
                    raise ValueError(
                        f"Field {field.name} is a dataclass but not a SerializableConfig"
                    )
            else:
                retval[field.name] = value
        return retval

    def to_flat_dict(self) -> dict[str, str]:
        """Get a flat dictionary of stringified parameters, e.g. for logging to MLflow."""
        serialized = self.to_dict()
        param_dict = {}

        def _recursive(dictionary: dict, prefix=""):
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    _recursive(value, prefix + key + ".")
                else:
                    param_dict[prefix + key] = str(value)

        _recursive(serialized)
        return param_dict


def mlflow_log_config_params(cfg: SerializableConfig) -> mlflow.ActiveRun | None:
    if "MLFLOW_RUN_ID" not in os.environ:
        _LOGGER.warning(
            "MLFLOW_RUN_ID not found in environment. Skipping logging config parameters to MLflow."
        )
        return None
    active_run = mlflow.start_run()  # Continue existing run started by runner program
    for key, value in cfg.to_flat_dict().items():
        mlflow.log_param(key, value)
    return active_run
