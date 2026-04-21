import logging
import os
from pathlib import Path
from typing import Literal
from nerfbaselines._registry import get_dataset_spec

_LOGGER = logging.getLogger(__name__)


def get_scannetpp_subdir(dataset_id: str) -> str:
    if dataset_id == "scannet++":
        return "scannetpp"
    elif dataset_id == "eval_on_train_set_scannet++":
        return "scannetpp_eval_on_train_set"
    else:
        raise ValueError(f"Unknown scannet++ variation dataset_id: {dataset_id}")


def is_scannetpp_dataset(dataset_id: str) -> bool:
    return dataset_id in ["scannet++", "eval_on_train_set_scannet++"]


def get_dataset_scenes(dataset_id: str, exclude_list) -> list[str] | list[Path]:
    if is_scannetpp_dataset(dataset_id):
        if os.environ.get("SCANNETPP_SCENES", None) is not None:
            return [
                f"{dataset_id}/{scene.strip()}"
                for scene in os.environ["SCANNETPP_SCENES"].split(",")
            ]
        if os.environ.get("SCANNETPP_PATH", None) is None:
            _LOGGER.warning(
                "SCANNETPP_PATH environment variable not set. dataset_scenes.SCENES_PER_DATASET will not contain scannet++ scenes."
            )
            return []
        return [
            f"{dataset_id}/{d.name}"
            for d in Path(
                os.environ["SCANNETPP_PATH"], get_scannetpp_subdir(dataset_id)
            ).glob("*")
        ]

    if dataset_id == "eth3d":
        if os.environ.get("ETH3D_SCENES", None) is not None:
            return [
                f"{dataset_id}/{scene.strip()}"
                for scene in os.environ["ETH3D_SCENES"].split(",")
            ]
        if os.environ.get("ETH3D_PATH", None) is None:
            _LOGGER.warning(
                "ETH3D_PATH environment variable not set. dataset_scenes.SCENES_PER_DATASET will not contain eth3d scenes."
            )
            return []
        return [
            f"eth3d/{d.name}"
            for d in Path(os.environ["ETH3D_PATH"]).glob("*")
            if d.is_dir()
        ]

    dataset_scenes_env_var = f"{dataset_id.upper()}_SCENES"
    if os.environ.get(dataset_scenes_env_var, None) is not None:
        return [
            f"{dataset_id}/{scene.strip()}"
            for scene in os.environ[dataset_scenes_env_var].split(",")
        ]

    scenes = get_dataset_spec(dataset_id)["metadata"]["scenes"]

    def excluded(scene_id):
        for block in exclude_list:
            if block in scene_id:
                return True
        return False

    return [
        f"{dataset_id}/{scene['id']}" for scene in scenes if not excluded(scene["id"])
    ]


def get_scannetpp_scene_path(dataset_id: str, scene_id: str) -> str:
    if os.environ.get("SCANNETPP_PATH", None) is None:
        raise RuntimeError("SCANNETPP_PATH environment variable not set.")
    return str(
        Path(os.environ["SCANNETPP_PATH"], get_scannetpp_subdir(dataset_id), scene_id)
    )


NATIVE_NB_DATASETS = ["mipnerf360", "tanksandtemples"]

SCENES_PER_DATASET = {
    dataset: get_dataset_scenes(dataset, [])
    for dataset in [
        "mipnerf360",
        "tanksandtemples",
        "scannet++",
        "eval_on_train_set_scannet++",
        "eth3d",
    ]
}

print(SCENES_PER_DATASET)  # For debugging


def scene_id_to_nerfbaselines_data_value(scene: str | Path) -> str:
    scene = str(scene)
    split = scene.split("/")

    if len(split) != 2:
        logging.info(f"Treating scene id {scene} as a local path.")
        return str(scene)

    dataset_id, scene_id = scene.split("/")
    if dataset_id in NATIVE_NB_DATASETS:
        return f"external://{dataset_id}/{scene_id}"
    elif is_scannetpp_dataset(dataset_id):
        if scene not in SCENES_PER_DATASET[dataset_id]:
            raise RuntimeError(f"Unknown scannet++ scene_id: {scene_id}")
        return get_scannetpp_scene_path(dataset_id, scene_id)
    elif dataset_id == "eth3d":
        if scene not in SCENES_PER_DATASET[dataset_id]:
            raise RuntimeError(f"Unknown eth3d scene_id: {scene_id}")
        return str(Path(os.environ["ETH3D_PATH"], scene_id))
    else:
        try:
            get_dataset_scenes(dataset_id, [])[0]
        except Exception:
            logging.info(f"Treating scene id {scene} as a local path.")
            return str(scene)
        logging.info(f"Treating scene id {scene} as a nerfbaselines dataset.")
        return f"external://{dataset_id}/{scene_id}"


def get_scenes_from_args(
    scenes: list[str | Path], datasets: list[str]
) -> list[str | Path]:
    if len(scenes) > 0:
        return scenes

    scenes = []
    if datasets is None or datasets == []:
        datasets = [
            key for key, values in SCENES_PER_DATASET.items() if len(values) > 0
        ]

    for dataset in datasets:
        if dataset not in SCENES_PER_DATASET:
            raise ValueError(f"Unknown dataset specified: {dataset}")
        scenes.extend(SCENES_PER_DATASET[dataset])
    return scenes


def is_scene_indoors_or_outdoors(scene_id: str) -> Literal["indoors", "outdoors"]:
    inside: Literal["indoors"] = "indoors"
    outside: Literal["outdoors"] = "outdoors"

    dataset, scene = scene_id.split("/")
    if dataset == "mipnerf360":
        mipnerf360_types: dict[str, Literal["indoors", "outdoors"]] = {
            "garden": outside,
            "bicycle": outside,
            "flowers": outside,
            "treehill": outside,
            "stump": outside,
            "kitchen": inside,
            "bonsai": inside,
            "counter": inside,
            "room": inside,
        }
        return mipnerf360_types[scene]
    if dataset == "tanksandtemples":
        tanksandtemples_types: dict[str, Literal["indoors", "outdoors"]] = {
            "auditorium": inside,
            "ballroom": inside,
            "courtroom": inside,
            "museum": inside,
            "m60": inside,
            "panther": inside,
            "church": inside,
            "meetingroom": inside,
            "playground": outside,
            "palace": outside,
            "temple": outside,
            "family": outside,
            "francis": outside,
            "horse": outside,
            "lighthouse": outside,
            "train": outside,
            "barn": outside,
            "caterpillar": outside,
            "courthouse": outside,
            "ignatius": outside,
            "truck": outside,
        }
        return tanksandtemples_types[scene]
    if dataset in ["scannet++", "eval_on_train_set_scannet++"]:
        return inside
    if dataset == "eth3d":
        eth3d_types: dict[str, Literal["indoors", "outdoors"]] = {
            "courtyard": outside,
            "delivery_area": inside,
            "electro": outside,
            "facade": outside,
            "kicker": inside,
            "meadow": outside,
            "office": inside,
            "pipes": inside,
            "playground": outside,
            "relief": inside,
            "relief_2": inside,
            "terrace": outside,
            "terrains": inside,
        }
        return eth3d_types[scene]
    raise ValueError(f"Unknown dataset: {dataset}")
