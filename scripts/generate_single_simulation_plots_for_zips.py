#!/usr/bin/env python3
"""Generate single-simulation plots for every ZIP archive under a directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MASTER_OUTPUT_FILE_NAME = "MasterOutput.csv"
MARKET_OVERLAP_FILE_NAME = "MarketOverlap.csv"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Recursively search for ZIP archives and generate plots for each "
            "requested simulation number."
        )
    )
    parser.add_argument(
        "search_dir",
        type=Path,
        help="Directory to recursively search for ZIP archives.",
    )
    parser.add_argument(
        "simulation_numbers",
        type=int,
        nargs="+",
        help="Simulation numbers to pass to --single-simulation.",
    )
    return parser.parse_args()


def output_dir_for_zip(zip_path: Path) -> Path:
    """Return the directory used to store generated plots for a ZIP file."""
    return zip_path.parent / f"{zip_path.stem}_plots"


def run_plot_command(zip_path: Path, simulation_number: int, destination: Path) -> None:
    """Execute the plotting script for a single ZIP and simulation number."""
    destination.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "Analysis/business_strategy_plots.py",
        "--zip-path",
        str(zip_path),
        "--master-output-file-name",
        MASTER_OUTPUT_FILE_NAME,
        "--market-overlap-file-name",
        MARKET_OVERLAP_FILE_NAME,
        "--single-simulation",
        str(simulation_number),
        "--output-dir",
        str(destination),
    ]

    subprocess.run(command, check=True)


def main() -> None:
    """Generate single-simulation plots for each ZIP archive found."""
    args = parse_args()

    zip_paths = sorted(args.search_dir.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No ZIP files found under {args.search_dir}")

    for zip_path in zip_paths:
        base_output_dir = output_dir_for_zip(zip_path)
        for simulation_number in args.simulation_numbers:
            simulation_output_dir = base_output_dir / f"sim_{simulation_number}"
            print(
                f"Generating plots for {zip_path} (simulation {simulation_number}) -> "
                f"{simulation_output_dir}"
            )
            run_plot_command(zip_path, simulation_number, simulation_output_dir)


if __name__ == "__main__":
    main()
