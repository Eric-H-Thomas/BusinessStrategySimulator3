#!/usr/bin/env python3
"""Extract first and final timestep rows from MasterOutput.csv files.

Recursively searches a directory for ``MasterOutput.csv`` files and writes
``MasterOutputEnds.csv`` next to each source file. For each simulation,
rows from both the first and final recorded timestep are kept.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

MASTER_OUTPUT_FILENAME = "MasterOutput.csv"
ENDS_FILENAME = "MasterOutputEnds.csv"


def extract_ends(master_output_path: Path) -> Path:
    """Write a MasterOutputEnds.csv next to a MasterOutput.csv file."""
    with master_output_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if fieldnames is None:
            raise ValueError(f"{master_output_path} has no header row.")
        if "Sim" not in fieldnames or "Step" not in fieldnames:
            raise ValueError(
                f"{master_output_path} must contain 'Sim' and 'Step' columns. "
                f"Found columns: {fieldnames}"
            )
        rows = list(reader)

    min_step_by_sim: dict[str, float] = {}
    max_step_by_sim: dict[str, float] = {}
    for row in rows:
        sim = row["Sim"]
        step = float(row["Step"])
        prev_min = min_step_by_sim.get(sim)
        prev_max = max_step_by_sim.get(sim)
        if prev_min is None or step < prev_min:
            min_step_by_sim[sim] = step
        if prev_max is None or step > prev_max:
            max_step_by_sim[sim] = step

    end_rows = [
        row
        for row in rows
        if float(row["Step"]) in (min_step_by_sim[row["Sim"]], max_step_by_sim[row["Sim"]])
    ]

    output_path = master_output_path.with_name(ENDS_FILENAME)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(end_rows)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find MasterOutput.csv files and write first/final-timestep "
            "rows to MasterOutputEnds.csv next to each file."
        )
    )
    parser.add_argument(
        "search_dir",
        type=Path,
        help="Directory to search recursively for MasterOutput.csv files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    search_dir = args.search_dir

    if not search_dir.exists() or not search_dir.is_dir():
        raise SystemExit(f"Not a directory: {search_dir}")

    master_paths = sorted(search_dir.rglob(MASTER_OUTPUT_FILENAME))
    if not master_paths:
        print(f"No {MASTER_OUTPUT_FILENAME} files found under {search_dir}")
        return

    written = 0
    for path in master_paths:
        output_path = extract_ends(path)
        written += 1
        print(f"Wrote: {output_path}")

    print(f"Done. Wrote {written} {ENDS_FILENAME} files.")


if __name__ == "__main__":
    main()
