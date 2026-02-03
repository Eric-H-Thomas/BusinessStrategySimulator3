"""Extract the first N simulations from a master CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract the first N simulations from a master output CSV and save "
            "a filtered CSV alongside it."
        )
    )
    parser.add_argument("csv_path", type=Path, help="Path to the master output CSV file.")
    parser.add_argument(
        "num_simulations",
        type=int,
        help="Number of simulations to extract (N).",
    )
    return parser.parse_args()


def build_output_path(csv_path: Path, num_simulations: int) -> Path:
    return csv_path.with_name(
        f"{csv_path.stem}_first_{num_simulations}_sims{csv_path.suffix}"
    )


def extract_first_simulations(csv_path: Path, num_simulations: int) -> Path:
    if num_simulations <= 0:
        raise ValueError("Number of simulations must be a positive integer.")

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    output_path = build_output_path(csv_path, num_simulations)

    with csv_path.open(newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError("CSV file does not contain a header row.")
        if "Sim" not in reader.fieldnames:
            raise ValueError("CSV file must include a 'Sim' column.")

        with output_path.open("w", newline="", encoding="utf-8") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
            writer.writeheader()

            seen_simulations: list[str] = []
            for row in reader:
                sim_id = row.get("Sim")
                if sim_id is None:
                    continue
                if sim_id not in seen_simulations:
                    seen_simulations.append(sim_id)
                if sim_id in seen_simulations[:num_simulations]:
                    writer.writerow(row)

    return output_path


def main() -> None:
    args = parse_args()
    output_path = extract_first_simulations(args.csv_path, args.num_simulations)
    print(f"Wrote filtered CSV to: {output_path}")


if __name__ == "__main__":
    main()
