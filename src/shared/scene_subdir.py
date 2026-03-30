from pathlib import Path

from eval_scripts.common.dataset_scenes import SCENES_PER_DATASET


def get_scene_subdir(scene: str | Path) -> Path:
    if any(scene.startswith(dataset) for dataset in SCENES_PER_DATASET.keys()):
        return Path(*scene.split("/"))

    scene_path = Path(scene)
    if scene_path.is_relative_to(Path.cwd()):
        path_relative_to_wd = scene_path.relative_to(Path.cwd())
        return Path(path_relative_to_wd.parent, path_relative_to_wd.stem)

    return Path(str(scene_path.parent).replace("/", "_")) / scene_path.stem
