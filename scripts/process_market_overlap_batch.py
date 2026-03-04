#!/usr/bin/env python3
"""Batch-process market-overlap ZIP outputs for a top-level folder A.

For each immediate child folder B under folder A, this script:
1) finds all ``MasterOutput_MarketOverlap.zip`` files recursively,
2) runs ``Analysis/business_strategy_plots.py`` with ``--stats-only`` for each ZIP,
3) collects per-ZIP summary CSVs,
4) writes an averaged summary CSV for folder A,
5) writes folder-level average bankruptcy and cumulative-capital plots with 95% CIs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Import plotting helpers for consistent label mapping and aggregation behavior.
from Analysis.business_strategy_plots import (  # pylint: disable=import-error
    aggregate_data_with_std,
    bankruptcy_rate_by_agent_type,
    sort_by_agent_type,
    sort_by_agent_type_various_sophisticated_agent_types,
)

MASTER_OUTPUT_ZIP_NAME = "MasterOutput_MarketOverlap.zip"
MASTER_OUTPUT_FILE_NAME = "MasterOutput.csv"
MARKET_OVERLAP_FILE_NAME = "MarketOverlap.csv"
SUMMARY_SUFFIX = "_summary_statistics.csv"


@dataclass
class ZipResult:
    """Container for outputs derived from one ZIP archive."""

    zip_path: Path
    various_sophisticated_agent_types: bool
    summary_csv_path: Path
    bankruptcy_by_agent_type: pd.Series
    capital_by_step_agent_type: pd.DataFrame


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Process each folder B under folder A, run business_strategy_plots.py "
            "with --stats-only for each MasterOutput_MarketOverlap.zip, and "
            "produce folder-A aggregates and confidence-interval plots."
        )
    )
    parser.add_argument(
        "folder_a",
        type=Path,
        help="Folder A containing immediate child folders B.",
    )
    return parser.parse_args()


def folder_uses_various_sophisticated_agent_types(folder: Path) -> bool:
    """Return whether the folder indicates various sophisticated agent types."""
    return "varioussophisticatedagents" in str(folder).lower()


def run_stats_only_for_zip(zip_path: Path, various_types: bool) -> Path:
    """Run business_strategy_plots.py with --stats-only for one ZIP archive."""
    command = [
        sys.executable,
        "Analysis/business_strategy_plots.py",
        "--zip-path",
        str(zip_path),
        "--master-output-file-name",
        MASTER_OUTPUT_FILE_NAME,
        "--market-overlap-file-name",
        MARKET_OVERLAP_FILE_NAME,
        "--stats-only",
    ]
    if various_types:
        command.append("--various-sophisticated-agent-types")

    subprocess.run(command, check=True)
    return zip_path.parent / f"{zip_path.stem}{SUMMARY_SUFFIX}"


def load_master_output_from_zip(zip_path: Path) -> pd.DataFrame:
    """Load MasterOutput.csv from a ZIP archive."""
    with zipfile.ZipFile(zip_path) as zip_ref:
        with zip_ref.open(MASTER_OUTPUT_FILE_NAME) as csv_file:
            return pd.read_csv(csv_file)


def extract_zip_metrics(zip_path: Path, various_types: bool, summary_csv_path: Path) -> ZipResult:
    """Extract per-ZIP bankruptcy and capital trajectories by agent type."""
    df = load_master_output_from_zip(zip_path)
    df = (
        sort_by_agent_type_various_sophisticated_agent_types(df)
        if various_types
        else sort_by_agent_type(df)
    )

    bankruptcy = bankruptcy_rate_by_agent_type(df)

    capital_df = aggregate_data_with_std(df)[["Step", "Agent Type", "Capital_mean"]].copy()
    capital_df.rename(columns={"Capital_mean": "Capital"}, inplace=True)

    return ZipResult(
        zip_path=zip_path,
        various_sophisticated_agent_types=various_types,
        summary_csv_path=summary_csv_path,
        bankruptcy_by_agent_type=bankruptcy,
        capital_by_step_agent_type=capital_df,
    )


def confidence_interval_95(values: pd.Series) -> float:
    """Return the half-width of the 95% confidence interval."""
    n = values.count()
    if n <= 1:
        return 0.0
    return float(1.96 * values.std(ddof=1) / np.sqrt(n))


def write_average_summary_csv(summary_paths: list[Path], output_path: Path) -> None:
    """Average summary-stat rows across all per-ZIP summary CSV files."""
    summary_frames = [pd.read_csv(path) for path in summary_paths]
    combined = pd.concat(summary_frames, ignore_index=True)
    averaged = (
        combined.groupby("Statistic", as_index=False)["Value"]
        .mean()
        .sort_values("Statistic")
    )
    averaged.to_csv(output_path, index=False)


def build_bankruptcy_ci_df(results: list[ZipResult]) -> pd.DataFrame:
    """Build bankruptcy mean and CI table across ZIPs (CI aligned by index)."""
    rows: list[dict[str, float | str]] = []
    for result in results:
        for agent_type, value in result.bankruptcy_by_agent_type.items():
            rows.append(
                {
                    "Zip": str(result.zip_path),
                    "Agent Type": agent_type,
                    "Bankruptcy Rate": float(value),
                }
            )

    per_zip = pd.DataFrame(rows)

    grouped = per_zip.groupby("Agent Type")["Bankruptcy Rate"]

    # Compute aggregates; keep Agent Type as index for safe alignment.
    summary = grouped.agg(Mean="mean", N="count")

    # Compute CI as a Series indexed by Agent Type and align by index on assignment.
    summary["CI95"] = grouped.apply(confidence_interval_95)

    # Back to a regular column layout
    return summary.reset_index()


def build_capital_ci_df(results: list[ZipResult]) -> pd.DataFrame:
    """Build timestep-wise capital mean and CI table across ZIPs (CI aligned by index)."""
    frames: list[pd.DataFrame] = []
    for result in results:
        tmp = result.capital_by_step_agent_type.copy()
        tmp["Zip"] = str(result.zip_path)
        frames.append(tmp)

    per_zip = pd.concat(frames, ignore_index=True)

    grouped = per_zip.groupby(["Step", "Agent Type"])["Capital"]

    # Compute aggregates; keep (Step, Agent Type) as a MultiIndex for safe alignment.
    summary = grouped.agg(Mean="mean", N="count")

    # CI is a Series with the same MultiIndex; assignment aligns correctly.
    summary["CI95"] = grouped.apply(confidence_interval_95)

    return summary.reset_index()


def plot_average_bankruptcy_with_ci(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot average bankruptcy rates by agent type with 95% CI bars."""
    ordered = summary_df.sort_values("Agent Type")
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(ordered))
    means_pct = ordered["Mean"].to_numpy() * 100
    ci_pct = ordered["CI95"].to_numpy() * 100

    ax.bar(x, means_pct, color="#79AEA3", alpha=0.85)
    ax.errorbar(x, means_pct, yerr=ci_pct, fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["Agent Type"], rotation=0)
    ax.set_ylabel("Average Bankruptcy Rate (%)")
    ax.set_title("Average Bankruptcy Rate Across Agent-Level Means (95% CI)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_average_capital_with_ci(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot average cumulative capital trajectories with shaded 95% CIs."""
    type_to_color = {
        "AI": "#79AEA3",
        "Naive": "#9E4770",
        "Sophisticated": "#1446A0",
        "Sophisticated A": "#1446A0",
        "Sophisticated B": "#2166C4",
        "Sophisticated C": "#0094C8",
        "Sophisticated D": "#00B894",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    for agent_type in sorted(summary_df["Agent Type"].unique()):
        agent_df = summary_df[summary_df["Agent Type"] == agent_type].sort_values("Step")
        x = agent_df["Step"].to_numpy()
        y = agent_df["Mean"].to_numpy()
        ci = agent_df["CI95"].to_numpy()
        color = type_to_color.get(agent_type, "#999999")

        ax.plot(x, y, label=agent_type, color=color, linewidth=2)
        ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.2)

    ax.set_ylabel("Avg. Capital")
    ax.set_xlabel("Step")
    ax.set_title("Average Cumulative Capital Across Agent-Level Means (95% CI)")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    """Run batch processing workflow for folder A."""
    args = parse_args()
    folder_a = args.folder_a.resolve()

    if not folder_a.is_dir():
        raise SystemExit(f"Folder A does not exist or is not a directory: {folder_a}")

    folder_b_dirs = sorted(path for path in folder_a.iterdir() if path.is_dir())
    if not folder_b_dirs:
        raise SystemExit(f"No folder B directories found under folder A: {folder_a}")

    results: list[ZipResult] = []
    various_types = folder_uses_various_sophisticated_agent_types(folder_a)

    for folder_b in folder_b_dirs:

        zip_paths = sorted(folder_b.rglob(MASTER_OUTPUT_ZIP_NAME))
        if not zip_paths:
            continue

        for zip_path in zip_paths:
            print(f"Processing ZIP: {zip_path}")
            summary_csv_path = run_stats_only_for_zip(zip_path, various_types)
            result = extract_zip_metrics(zip_path, various_types, summary_csv_path)
            results.append(result)

    if not results:
        raise SystemExit(
            f"No {MASTER_OUTPUT_ZIP_NAME} files found in any folder B under {folder_a}"
        )

    summary_paths = [result.summary_csv_path for result in results]
    average_summary_path = folder_a / "average_summary_statistics.csv"
    write_average_summary_csv(summary_paths, average_summary_path)

    bankruptcy_ci_df = build_bankruptcy_ci_df(results)
    capital_ci_df = build_capital_ci_df(results)

    bankruptcy_ci_df.to_csv(folder_a / "average_bankruptcy_95ci.csv", index=False)
    capital_ci_df.to_csv(folder_a / "average_cumulative_capital_95ci.csv", index=False)

    plot_average_bankruptcy_with_ci(
        bankruptcy_ci_df,
        folder_a / "average_bankruptcy_95ci.png",
    )
    plot_average_capital_with_ci(
        capital_ci_df,
        folder_a / "average_cumulative_capital_95ci.png",
    )

    print(f"Wrote: {average_summary_path}")
    print(f"Wrote: {folder_a / 'average_bankruptcy_95ci.csv'}")
    print(f"Wrote: {folder_a / 'average_cumulative_capital_95ci.csv'}")
    print(f"Wrote: {folder_a / 'average_bankruptcy_95ci.png'}")
    print(f"Wrote: {folder_a / 'average_cumulative_capital_95ci.png'}")


if __name__ == "__main__":
    main()
