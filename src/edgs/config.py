from dataclasses import dataclass
from pathlib import Path
from typing_extensions import Literal
from shared.serializable_config import SerializableConfig


@dataclass
class EDGSConfig(SerializableConfig):
    # Directory to save the output gaussian data and proxy dataset to.
    output_dir: Path
    # Scene in our <dataset_id>/<scene_id> format, e.g. "mipnerf360/garden" or "scannet++/<unreadable_hex_string>"
    scene: str

    # If set, will limit the number of images loaded from the dataset, which can be useful for debugging or
    # if running out of RAM with many images. If set, will select a random subset of this many images to load.
    max_images_to_load: int | None = None

    # number of matches per reference
    matches_per_ref: int = 15_000
    # number of reference images
    num_refs: int = 180
    # number of nearest neighbors per reference
    nns_per_ref: int = 3
    # Default in EDGS is 0.001, but they further multiply it by 0.5 in another function afterwards
    scaling_factor: float = 0.0005
    proj_err_tolerance: float = 0.01
    add_sfm_init: bool = False

    init_sh_order: int = 3
    # Original code uses the analog of our "invisible" option, but comment states training performance
    # is the same, which was confirmed independently (though only on 1 scene)
    # We use this mode because having Gs with almost zero opacity will make selecting a random subset of given size for training invalid,
    # as the "effective size" will potentially be lower due to invisible Gaussians.
    bad_points_mode: Literal["invisible", "remove"] = "remove"

    # Whether to export the output point cloud as a ply file for visualization. The point cloud will be saved to {output_dir}/edgs_points3D.ply.
    output_pointcloud: bool = True
