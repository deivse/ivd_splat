import re
import uuid
import string
import random
from typing import Tuple


def generate_mlflow_image_filenames(
    sanitized_key: str, step: int, timestamp: int
) -> Tuple[str, str]:
    """
    Generate MLflow image filenames with proper delimiters.

    This function fixes the bug in MLflow where % delimiters are used in filenames,
    which causes issues with URL encoding. We use + delimiters instead.

    Original MLflow issue: https://github.com/mlflow/mlflow/issues/14136

    Args:
        sanitized_key: The sanitized image key (with / replaced by #)
        step: The training step
        timestamp: The timestamp when the image was logged

    Returns:
        Tuple of (uncompressed_filename, compressed_filename) without extensions
    """
    filename_uuid = str(uuid.uuid4())
    # Construct a filename uuid that does not start with hex digits
    filename_uuid = f"{random.choice(string.ascii_lowercase[6:])}{filename_uuid[1:]}"

    # Use + delimiters instead of % to avoid URL encoding issues
    uncompressed_filename = (
        f"images/{sanitized_key}+step+{step}+timestamp+{timestamp}+{filename_uuid}"
    )
    compressed_filename = f"{uncompressed_filename}+compressed"

    return uncompressed_filename, compressed_filename


def patch_mlflow_log_image():
    """
    Monkey patch MLflow's log_image method to use the fixed filename generation.

    Call this function once at the start of your program to apply the fix
    without modifying the MLflow source code.
    """
    from mlflow.tracking.client import MlflowClient
    from mlflow.utils.time import get_current_time_millis
    from mlflow.tracking.multimedia import (
        compress_image_size,
        convert_to_pil_image,
        Image,
    )
    from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_IMAGES

    # Store the original method
    original_log_image = MlflowClient.log_image

    def patched_log_image(
        self,
        run_id: str,
        image,
        artifact_file=None,
        key=None,
        step=None,
        timestamp=None,
        synchronous=None,
    ):
        """Patched version of log_image with fixed filename delimiters."""
        from mlflow.environment_variables import MLFLOW_ENABLE_ASYNC_LOGGING
        import numpy as np
        import PIL.Image

        synchronous = (
            synchronous
            if synchronous is not None
            else not MLFLOW_ENABLE_ASYNC_LOGGING.get()
        )

        if artifact_file is not None and any(
            arg is not None for arg in [key, step, timestamp]
        ):
            raise TypeError(
                "The `artifact_file` parameter cannot be used in conjunction with `key`, "
                "`step`, or `timestamp` parameters. Please ensure that `artifact_file` is "
                "specified alone, without any of these conflicting parameters."
            )
        elif artifact_file is None and key is None:
            raise TypeError(
                "Invalid arguments: Please specify exactly one of `artifact_file` or `key`. Use "
                "`key` to log dynamic image charts or `artifact_file` for saving static images. "
            )

        # Convert image type to PIL if its a numpy array
        if isinstance(image, np.ndarray):
            image = convert_to_pil_image(image)
        elif isinstance(image, Image):
            image = image.to_pil()
        else:
            if not isinstance(image, PIL.Image.Image):
                raise TypeError(
                    f"Unsupported image object type: {type(image)}. "
                    "`image` must be one of numpy.ndarray, "
                    "PIL.Image.Image, and mlflow.Image."
                )

        if artifact_file is not None:
            with self._log_artifact_helper(run_id, artifact_file) as tmp_path:
                image.save(tmp_path)

        elif key is not None:
            # Check image key for invalid characters
            if not re.match(r"^[a-zA-Z0-9_\-./ ]+$", key):
                raise ValueError(
                    "The `key` parameter may only contain alphanumerics, underscores (_), "
                    "dashes (-), periods (.), spaces ( ), and slashes (/)."
                    f"The provided key `{key}` contains invalid characters."
                )

            step = step or 0
            timestamp = timestamp or get_current_time_millis()

            # Sanitize key to use in filename (replace / with # to avoid subdirectories)
            sanitized_key = re.sub(r"/", "#", key)

            # Use our fixed filename generation
            uncompressed_filename, compressed_filename = (
                generate_mlflow_image_filenames(sanitized_key, step, timestamp)
            )

            # Save full-resolution image
            image_filepath = f"{uncompressed_filename}.png"
            compressed_image_filepath = f"{compressed_filename}.webp"

            # Need to make a resize copy before running thread for thread safety
            # If further optimization is needed, we can move this resize to async queue.
            compressed_image = compress_image_size(image)

            if synchronous:
                with self._log_artifact_helper(run_id, image_filepath) as tmp_path:
                    image.save(tmp_path)
            else:
                self._log_artifact_async_helper(run_id, image_filepath, image)

            if synchronous:
                with self._log_artifact_helper(
                    run_id, compressed_image_filepath
                ) as tmp_path:
                    compressed_image.save(tmp_path)
            else:
                self._log_artifact_async_helper(
                    run_id, compressed_image_filepath, compressed_image
                )

            # Log tag indicating that the run includes logged image
            self.set_tag(run_id, MLFLOW_LOGGED_IMAGES, True, synchronous)

    # Apply the patch
    MlflowClient.log_image = patched_log_image
    print("MLflow log_image method patched to use + delimiters instead of % delimiters")
