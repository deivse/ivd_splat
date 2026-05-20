from dataclasses import dataclass
import itertools
import json
import logging

from eval_scripts.common.dataset_scenes import (
    SCENES_PER_DATASET,
    scene_id_to_nerfbaselines_data_value,
)
from nerfbaselines.datasets import load_dataset
import tyro


@dataclass
class Args:
    datasets: list[str] | None = None

    output: str = "init_sfm_pts_per_scene.json"


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    args = tyro.cli(Args)
    if args.datasets is None:
        args.datasets = list(SCENES_PER_DATASET.keys())
    scenes = itertools.chain(*[SCENES_PER_DATASET[ds] for ds in args.datasets])
    retval: dict[str, int] = {}
    for scene in scenes:
        try:
            nerfbaselines_data_val = scene_id_to_nerfbaselines_data_value(scene)
            dataset = load_dataset(
                nerfbaselines_data_val, "train", features=["points3D_xyz"]
            )
            num_points = len(dataset["points3D_xyz"])
            print(f"{scene}: {num_points} SfM points")
            retval[scene] = num_points
        except Exception as e:
            print(f"Error processing scene {scene}: {e}")

    with open(args.output, "w") as f:
        json.dump(retval, f, indent=4)
    print(f"Wrote output to {args.output}")


if __name__ == "__main__":
    main()
