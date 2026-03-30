import datetime
import os
from pathlib import Path

from eval_scripts.common.typedefs import InitMethod
from shared.scene_subdir import get_scene_subdir


def get_default_results_dir() -> Path:
    if os.environ.get("RESULTS_DIR"):
        return Path(os.environ["RESULTS_DIR"])
    return Path("./results")


class ResultsDirectory:
    TSDF_FUSION_DIR_NAME = "monodepth"
    EDGS_DIR_NAME = "edgs"
    TSDF_FUSION_CACHE_DIR_NAME = "__monodepth_cache__"

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)

    def get_scene_dir(self, scene: str) -> Path:
        """
        Get the directory for a specific scene.
        Args:
            scene: The scene id in form dataset/scene, as supported by scene_id_to_nerfbaselines_data_value.
        Returns:
            Path to the scene directory.
        """
        return self.base_dir / get_scene_subdir(scene)

    def get_monodepth_output_dir(self, scene: str, monodepth_config_name: str) -> Path:
        """
        Get the directory for md-tsdf-fusion outputs for a specific scene and config.
        Args:
            scene: The scene id in form dataset/scene, as supported by scene_id_to_nerfbaselines_data_value.
            config_name: The config string used for md-tsdf-fusion, sanitized for filesystem usage.
        Returns:
            Path to the md-tsdf-fusion output directory.
        """
        return (
            self.get_scene_dir(scene)
            / ResultsDirectory.TSDF_FUSION_DIR_NAME
            / monodepth_config_name
        )

    def get_edgs_output_dir(self, scene: str, edgs_config_name: str) -> Path:
        """
        Get the directory for EDGS outputs for a specific scene.
        Args:
            scene: The scene id in form dataset/scene, as supported by scene_id_to_nerfbaselines_data_value.
            edgs_config_name: The config string used for EDGS, sanitized for filesystem usage.
        Returns:
            Path to the EDGS output directory.
        """
        return (
            self.get_scene_dir(scene)
            / ResultsDirectory.EDGS_DIR_NAME
            / edgs_config_name
        )

    def get_da3_output_dir(self, scene: str, da3_config_name: str) -> Path:
        """
        Get the directory for DA3 outputs for a specific scene.
        Args:
            scene: The scene id in form dataset/scene, as supported by scene_id_to_nerfbaselines_data_value.
            da3_config_name: The config string used for DA3, sanitized for filesystem usage.
        Returns:
            Path to the DA3 output directory.
        """
        return self.get_scene_dir(scene) / "da3" / da3_config_name

    def get_init_method_output_dir(
        self, scene, config_name, init_method: InitMethod
    ) -> Path:
        if init_method == InitMethod.monodepth:
            return self.get_monodepth_output_dir(scene, config_name)
        elif init_method == InitMethod.edgs:
            return self.get_edgs_output_dir(scene, config_name)
        elif init_method == InitMethod.da3:
            return self.get_da3_output_dir(scene, config_name)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

    def get_method_output_dir(
        self,
        scene: str,
        method: str,
        init_method: InitMethod,
        init_method_config_id: str | None = None,
        method_config_id: str | None = None,
    ) -> Path:
        """
        Get the directory for nerfbaselines method outputs for a specific scene and method.
        Args:
            scene: The scene id in form dataset/scene, as supported by scene_id_to_nerfbaselines_data_value.
            method: The nerfbaselines method name.
            init_method: The initialization method used.
            init_method_config_id: Optional identifier for the initialization method configuration, e.g. md-tsdf-fusion config string.
            method_config_id: Optional identifier for the method configuration.
        Returns:
            Path to the nerfbaselines method output directory.
        """
        dir = self.get_scene_dir(scene) / method / init_method.value
        init_method_config_id = init_method_config_id or "default"
        method_config_id = method_config_id or "default"

        return dir / init_method_config_id / method_config_id

    def get_monodepth_cache_dir(self) -> Path:
        """
        Get the directory for md-tsdf-fusion cache
        """
        return self.base_dir / ResultsDirectory.TSDF_FUSION_CACHE_DIR_NAME


def rename_old_dir_with_timestamp(dir: Path, results_dir: ResultsDirectory) -> Path:
    """
    Appends a timestamp to the directory name to avoid conflicts
    when the directory already exists.

    `dir` is not modified, a new Path object is returned.
    """
    last_edit_time = max(f.stat().st_mtime for f in dir.rglob("*"))
    last_edit_time_str = datetime.datetime.fromtimestamp(last_edit_time).strftime(
        "_%d-%m-%Y_%H:%M:%S"
    )
    new_old_dir_name = dir.name + last_edit_time_str

    backup_results_dir_path = (
        results_dir.base_dir.parent / f"{results_dir.base_dir.name}_backup"
    )

    new_relative_path = dir.relative_to(results_dir.base_dir)
    new_relative_path = new_relative_path.parent / new_old_dir_name

    new_path = backup_results_dir_path / new_relative_path
    new_path.parent.mkdir(parents=True, exist_ok=True)

    # This doesn't point dir to the new location
    # (Which is what we want)
    return dir.rename(backup_results_dir_path / new_relative_path)


def directory_exists_and_has_files(dir: Path) -> bool:
    if not dir.exists():
        return False
    for d in dir.rglob("*"):
        if d.is_file():
            return True
    return False
