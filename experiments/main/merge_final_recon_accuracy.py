#!/usr/bin/env python3

"""
Merge multiple reconstruction-accuracy JSON result files into one.
Useful to produce a table including both hybrid init and non-hybrid init laser scan results,
since the eval_final_recon_accuracy_*.py script CLI API doesn't allow to invoke both in one run currently.
"""

import argparse
import json
from pathlib import Path


def merge_results(input_paths: list[Path]) -> dict:
    inputs = [json.loads(path.read_text()) for path in input_paths]
    merged = {
        key: inputs[0][key] for key in ("dataset", "scenes", "fscore_threshold_meters")
    }
    merged["columns"] = []
    merged["resolved_runs"] = {scene: [] for scene in merged["scenes"]}

    seen_runs: set[tuple[str, str, str]] = set()
    for path, data in zip(input_paths, inputs):
        for key in ("dataset", "scenes", "fscore_threshold_meters"):
            if data[key] != merged[key]:
                raise ValueError(f"{path}: incompatible {key!r}")

        for column in data["columns"]:
            if column not in merged["columns"]:
                merged["columns"].append(column)

        for scene in merged["scenes"]:
            for entry in data["resolved_runs"].get(scene, []):
                run_key = (scene, entry["column"], entry["strategy"])
                if run_key in seen_runs:
                    raise ValueError(
                        f"{path}: duplicate run for scene={scene!r}, "
                        f"column={entry['column']!r}, strategy={entry['strategy']!r}"
                    )
                seen_runs.add(run_key)
                merged["resolved_runs"][scene].append(entry)

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge reconstruction-accuracy JSON result files."
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="JSON files to merge")
    parser.add_argument("-o", "--output", type=Path, required=True)
    args = parser.parse_args()

    merged = merge_results(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")


if __name__ == "__main__":
    main()
