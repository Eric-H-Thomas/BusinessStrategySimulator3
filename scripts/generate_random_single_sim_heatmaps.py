#!/usr/bin/env python3
"""Generate random single-simulation market presence heatmaps from master output CSV."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Analysis import business_strategy_plots as plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly sample simulations from a master output CSV and generate "
            "one market-presence-over-time heatmap per sampled simulation."
        )
    )
    parser.add_argument(
        "master_output_csv",
        type=Path,
        help="Path to MasterOutput.csv (or compatible master output CSV).",
    )
    parser.add_argument(
        "num_plots",
        type=int,
        help="Number of simulations/heatmaps to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("random_single_sim_heatmaps"),
        help="Directory where generated heatmaps are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible sampling.",
    )
    return parser.parse_args()


def validate_inputs(df: pd.DataFrame, num_plots: int) -> list[int]:
    if num_plots <= 0:
        raise ValueError("num_plots must be a positive integer.")

    if "Sim" not in df.columns:
        raise ValueError("Input CSV must include a 'Sim' column.")

    unique_sims = sorted(df["Sim"].dropna().astype(int).unique().tolist())
    if not unique_sims:
        raise ValueError("No simulations found in the input CSV.")

    if num_plots > len(unique_sims):
        raise ValueError(
            f"Requested {num_plots} plots but only found {len(unique_sims)} unique simulations."
        )

    return unique_sims


def save_heatmap_for_simulation(df: pd.DataFrame, sim_id: int, output_path: Path) -> None:
    sim_df = df[df["Sim"] == sim_id].copy()
    plots.plot_firm_market_heatmap(sim_df, sim=sim_id)
    plt.gcf().savefig(output_path, dpi=300)
    plt.close(plt.gcf())


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.master_output_csv)
    unique_sims = validate_inputs(df, args.num_plots)

    rng = random.Random(args.seed)
    sampled_sims = sorted(rng.sample(unique_sims, args.num_plots))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sim_id in sampled_sims:
        output_path = args.output_dir / f"sim_{sim_id}_firm_market_heatmap.png"
        save_heatmap_for_simulation(df, sim_id, output_path)
        print(f"Saved heatmap for simulation {sim_id} -> {output_path}")

    print(f"Done. Generated {len(sampled_sims)} heatmap(s) in {args.output_dir}")


if __name__ == "__main__":
    main()
