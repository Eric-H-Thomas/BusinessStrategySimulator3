#!/usr/bin/env python3
"""Generate plot folders for every ZIP archive under a directory."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Analysis import business_strategy_plots as plots


MASTER_OUTPUT_FILENAMES = ("MasterOutput.csv",)
MARKET_OVERLAP_FILENAMES = ("MarketOverlap.csv",)


def find_member(zip_path: Path, candidates: tuple[str, ...]) -> str:
    """Return the first member name matching one of the candidate basenames."""
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.namelist()

    for candidate in candidates:
        for member in members:
            if Path(member).name == candidate:
                return member
        for member in members:
            if Path(member).name.lower() == candidate.lower():
                return member

    raise ValueError(f"Could not find {candidates} inside {zip_path}")

def generate_plots(parent_name: Path, zip_path: Path, output_dir: Path) -> None:
    """Generate plots for a ZIP archive and save them into output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    master_output = find_member(zip_path, MASTER_OUTPUT_FILENAMES)
    _market_overlap = find_member(zip_path, MARKET_OVERLAP_FILENAMES)

    df = plots.load_data(zip_path, master_output)

    if str(parent_name) == "VariousSophisticatedAgents":
        df = plots.sort_by_agent_type_various_sophisticated_agent_types(df)
        fig1 = plots.avg_bankruptcy_various_sophisticated_agent_types(df, clear_previous=True)
        fig2 = plots.plot_cumulative_capital_various_sophisticated_agent_types(df, clear_previous=False)

        figures = {
            "avg_bankruptcy.png": fig1,
            "cumulative_capital.png": fig2
        } # TODO: hotfix here; figure out the bug with plot 3 on the various sophisticated agents

    else:
        df = plots.sort_by_agent_type(df)
        fig1 = plots.avg_bankruptcy(df, clear_previous=True)
        fig2 = plots.plot_cumulative_capital(df, clear_previous=False)
        fig3 = plots.performance_summary_std_error(df, clear_previous=False)

        figures = {
            "avg_bankruptcy.png": fig1,
            "cumulative_capital.png": fig2,
            "performance_summary_std_error.png": fig3,
        }



    for filename, figure in figures.items():
        figure.savefig(output_dir / filename, dpi=300)

    plt.close("all")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively search for ZIP archives and generate plot folders for each."
        )
    )
    parser.add_argument(
        "search_dir",
        type=Path,
        help="Directory to recursively search for ZIP archives.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory to write the folder-of-folders of plots.",
    )
    args = parser.parse_args()

    zip_paths = sorted(args.search_dir.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No ZIP files found under {args.search_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for zip_path in zip_paths:
        parent_name = zip_path.parent.name
        destination = args.output_dir / parent_name
        generate_plots(parent_name, zip_path, destination)


if __name__ == "__main__":
    main()
