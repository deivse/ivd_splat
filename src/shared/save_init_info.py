import json
from pathlib import Path


INIT_INFO_JSON_FILENAME = "init_info.json"


def save_init_info_json(
    output_dir: Path,
    ivd_splat_init_type: str,
    required_files: list[str],
) -> None:
    """
    Save a JSON file containing information about the initialization method and required files for a given scene and initialization method.

    Args:
        output_dir: The directory where the JSON file will be saved.
        ivd_splat_init_type: The type of ivd-splat initialization used, e.g. "splat", "sparse", "dense".
        required_files: A list of file names that are required for this initialization method to be considered complete.
    """
    init_info = {
        "init_type": ivd_splat_init_type,
        "required_files": required_files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / INIT_INFO_JSON_FILENAME).open("w") as f:
        json.dump(init_info, f, indent=4)
