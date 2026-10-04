#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from nerfbaselines import NB_PREFIX


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a metadata-only dataset whose scenes refer to an existing "
            "downloaded NerfBaselines dataset."
        )
    )
    parser.add_argument("source_dataset", help="Downloaded source dataset ID, e.g. mipnerf360")
    parser.add_argument("dataset", help="ID of the derived dataset, e.g. mipnerf360-sparsified")
    parser.add_argument("loader", help="Custom loader as a registered ID or module:function")
    parser.add_argument(
        "--scene",
        dest="scenes",
        action="append",
        help="Source scene to include. May be repeated; defaults to every downloaded scene.",
    )
    parser.add_argument(
        "--source-path-key",
        default="source_path",
        help="loader_kwargs key receiving the source scene path (default: source_path)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory (default: <NERFBASELINES_PREFIX>/datasets/<dataset>)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing scene metadata")
    parser.add_argument("--dry-run", action="store_true", help="Print planned scene mappings only")
    return parser.parse_args()


def _find_scenes(source_root: Path, requested_scenes: Optional[Iterable[str]]) -> List[str]:
    if not source_root.is_dir():
        raise RuntimeError(
            f"Source dataset is not downloaded at {source_root}. Load or download it first."
        )

    if requested_scenes:
        scenes = list(dict.fromkeys(requested_scenes))
    else:
        scenes = sorted(
            path.name for path in source_root.iterdir()
            if path.is_dir() and (path / "nb-info.json").is_file()
        )

    if not scenes:
        raise RuntimeError(f"No downloaded scenes with nb-info.json found in {source_root}")

    for scene in scenes:
        metadata_path = source_root / scene / "nb-info.json"
        if not metadata_path.is_file():
            raise RuntimeError(f"Source scene metadata does not exist: {metadata_path}")
    return scenes


def _build_metadata(
    source_scene: Path,
    dataset: str,
    loader: str,
    source_path_key: str,
) -> Dict[str, Any]:
    with (source_scene / "nb-info.json").open("r", encoding="utf8") as file:
        metadata = json.load(file)

    loader_kwargs = dict(metadata.get("loader_kwargs") or {})
    loader_kwargs[source_path_key] = str(source_scene.resolve())
    metadata.update({
        "id": dataset,
        "loader": loader,
        "loader_kwargs": loader_kwargs,
    })
    return metadata


def main() -> None:
    args = _parse_args()
    datasets_root = Path(NB_PREFIX) / "datasets"
    source_root = datasets_root / args.source_dataset
    output_root = args.output or datasets_root / args.dataset
    if source_root.resolve() == output_root.resolve():
        raise RuntimeError("Source and output datasets must be different")

    scenes = _find_scenes(source_root, args.scenes)
    for scene in scenes:
        source_scene = source_root / scene
        output_metadata = output_root / scene / "nb-info.json"
        metadata = _build_metadata(
            source_scene,
            args.dataset,
            args.loader,
            args.source_path_key,
        )
        if args.dry_run:
            print(f"{output_metadata} -> {source_scene.resolve()}")
            continue
        if output_metadata.exists() and not args.force:
            raise RuntimeError(f"Output already exists: {output_metadata} (use --force to overwrite)")
        output_metadata.parent.mkdir(parents=True, exist_ok=True)
        with output_metadata.open("w", encoding="utf8") as file:
            json.dump(metadata, file, indent=2, ensure_ascii=False)
            file.write("\n")
        print(f"Created {output_metadata}")


if __name__ == "__main__":
    main()
