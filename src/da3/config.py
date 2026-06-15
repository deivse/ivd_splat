from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from depth_anything_3.utils.constants import DEFAULT_MODEL
from shared.serializable_config import SerializableConfig


@dataclass
class DA3Config(SerializableConfig):
    # Directory to save the output point cloud and proxy dataset to.
    output_dir: Path
    # Scene in our <dataset_id>/<scene_id> format, e.g. "mipnerf360/garden" or "scannet++/<unreadable_hex_string>"
    scene: str

    # If set, at most max_num_images images will be used for DA3 inference.
    # Images will be selected using K-Means as per EDGS implementation.
    max_num_images: int | None = 300

    # Use floater removal inspired by OpsiClear Depth Densifier implementation (same as monodepth).
    floater_removal: bool = False

    # Model directory path (can be on huggingface or a local path).
    model_dir: str = "depth-anything/DA3-GIANT-1.1"
    process_res: int = 504
    process_res_method: str = "upper_bound_resize"
    # Reference view selection strategy
    ref_view_strategy: Literal[
        "empty", "first", "middle", "saddle_balanced", "saddle_sim_range"
    ] = "saddle_balanced"
    # Use ray-based pose estimation instead of camera decoder
    use_ray_pose: bool = False
