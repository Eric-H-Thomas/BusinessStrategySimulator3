#!/usr/bin/env python3
"""
Scan a directory tree for files named 'evaluation_metrics.json',
extract the value of key 'mean reward', and append results to a CSV.

Usage:
    python collect_mean_rewards.py /path/to/search /path/to/output.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def find_json_files(root_dir: Path, target_name: str = "evaluation_metrics.json"):
    """Yield all files named `target_name` under `root_dir` (recursively)."""
    for path in root_dir.rglob(target_name):
        if path.is_file():
            yield path


def extract_mean_reward(json_path: Path, key: str = "mean_reward"):
    """Return the 'mean reward' value from a JSON file, or None if missing/invalid."""
    try:
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: could not read/parse {json_path}: {e}", file=sys.stderr)
        return None

    if key not in data:
        print(f"Warning: key '{key}' not found in {json_path}", file=sys.stderr)
        return None

    return data[key]


def append_to_csv(csv_path: Path, rows):
    """
    Append rows of (json_path, mean_reward) to CSV.
    If file does not exist, create it and write header first.
    """
    file_exists = csv_path.exists()

    # Ensure parent directory exists
    if csv_path.parent and not csv_path.parent.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if file_exists else "w"
    with csv_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["json_path", "mean_reward", "group", "rank_within_group"])
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Collect 'mean reward' from evaluation_metrics.json files into a CSV."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory to recursively search for evaluation_metrics.json",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to output CSV file (will be created or appended to).",
    )

    args = parser.parse_args()

    if not args.root_dir.exists() or not args.root_dir.is_dir():
        print(f"Error: root_dir '{args.root_dir}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    rows_to_append = []
    for json_path in find_json_files(args.root_dir):
        mean_reward = extract_mean_reward(json_path)
        if mean_reward is not None:
            relative_parts = json_path.relative_to(args.root_dir).parts
            group = relative_parts[0] if relative_parts else args.root_dir.name
            rows_to_append.append(
                {"json_path": str(json_path), "mean_reward": mean_reward, "group": group}
            )

    if not rows_to_append:
        print("No valid 'mean reward' values found; nothing to write.")
        return

    group_rows = {}
    for row in rows_to_append:
        group_rows.setdefault(row["group"], []).append(row)

    formatted_rows = []
    for group, rows in group_rows.items():
        sorted_rows = sorted(rows, key=lambda row: row["mean_reward"], reverse=True)
        rank = 0
        last_reward = None
        for row in sorted_rows:
            if last_reward is None or row["mean_reward"] != last_reward:
                rank += 1
                last_reward = row["mean_reward"]
            formatted_rows.append((row["json_path"], row["mean_reward"], group, rank))

    append_to_csv(args.output_csv, formatted_rows)
    print(f"Wrote {len(rows_to_append)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
