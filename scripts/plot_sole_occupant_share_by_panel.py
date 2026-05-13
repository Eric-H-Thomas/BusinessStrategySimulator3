#!/usr/bin/env python3
# This code was AI-generated and still requires verification by a human researcher. Remove this comment when done.
"""Plot sole-occupant portfolio-time share by agent type and AI win/loss panel.

For each trained-agent dataset, this script streams ``MasterOutput.csv`` from
``MasterOutput_MarketOverlap.zip`` and computes, separately for AI-win and
AI-loss simulations, the share of portfolio-time where a firm was the only firm
present in a market. The final plot shows the mean across trained-agent
datasets with 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "business_strategy_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MASTER_OUTPUT_ZIP_NAME = "MasterOutput_MarketOverlap.zip"
MASTER_OUTPUT_FILE_NAME = "MasterOutput.csv"
REQUIRED_COLUMNS = [
    "Sim",
    "Step",
    "Firm",
    "Agent Type",
    "Market",
    "Capital",
    "In Market",
]
AGENT_TYPE_ORDER = ["AI", "Sophisticated", "Naive"]
OUTCOME_ORDER = ["wins", "losses"]
OUTCOME_LABELS = {
    "wins": "AI wins panel",
    "losses": "AI loses panel",
}
TYPE_TO_COLOR = {
    "AI": "#79AEA3",
    "Sophisticated": "#1446A0",
    "Naive": "#9E4770",
}


@dataclass
class SimulationAccumulator:
    """Rows collected for one simulation."""

    zip_path: Path
    raw_sim: int
    steps_by_key: dict[tuple[int, int], list[int]]
    presence_by_key: dict[tuple[int, int], list[int]]
    agent_type_by_firm: dict[int, str]
    final_capital_by_firm: dict[int, float]

    @classmethod
    def create(cls, zip_path: Path, raw_sim: int) -> "SimulationAccumulator":
        return cls(
            zip_path=zip_path,
            raw_sim=raw_sim,
            steps_by_key={},
            presence_by_key={},
            agent_type_by_firm={},
            final_capital_by_firm={},
        )


@dataclass
class ZipTotals:
    """Sole-occupant numerator and portfolio-time denominator for one ZIP."""

    zip_path: Path
    sole_counts: dict[str, Counter[str]]
    portfolio_counts: dict[str, Counter[str]]
    outcome_run_counts: Counter[str]


def log(message: str) -> None:
    """Print progress to stderr."""
    print(f"[progress] {message}", file=sys.stderr, flush=True)


def normalize_agent_type(agent_type: str) -> str:
    """Map simulator agent labels to plot labels."""
    value = str(agent_type).strip()
    lowered = value.lower()

    if "stablebaselines3" in lowered or value.endswith("AI"):
        return "AI"
    if "highestoverlap" in lowered or value.endswith("S"):
        return "Sophisticated"
    if lowered == "all" or "naive" in lowered or value.endswith("N"):
        return "Naive"

    return value


def discover_zip_paths(economy_dir: Path) -> list[Path]:
    """Find expected zip outputs under immediate trained-agent folders."""
    direct_matches = sorted(economy_dir.glob(f"*/{MASTER_OUTPUT_ZIP_NAME}"))
    if direct_matches:
        return direct_matches
    return sorted(economy_dir.rglob(MASTER_OUTPUT_ZIP_NAME))


def is_readable_zip(zip_path: Path) -> bool:
    """Return True when the path is a non-empty readable ZIP archive."""
    return zip_path.is_file() and zip_path.stat().st_size > 0 and zipfile.is_zipfile(zip_path)


def get_master_output_member(zip_path: Path) -> str:
    """Return the archive member name for MasterOutput.csv."""
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if Path(member).name == MASTER_OUTPUT_FILE_NAME:
                return member
        for member in archive.namelist():
            if Path(member).name.lower() == MASTER_OUTPUT_FILE_NAME.lower():
                return member

    raise ValueError(f"{MASTER_OUTPUT_FILE_NAME} not found inside {zip_path}")


def append_rows(accumulator: SimulationAccumulator, frame: pd.DataFrame) -> None:
    """Append a chunk belonging to one simulation into the accumulator."""
    for (firm, market), group in frame.groupby(["Firm", "Market"], sort=False):
        key = (int(firm), int(market))
        accumulator.steps_by_key.setdefault(key, []).extend(group["Step"].astype(int).tolist())
        accumulator.presence_by_key.setdefault(key, []).extend(
            group["In Market"].astype(np.int8).tolist()
        )

    final_rows = frame.loc[
        frame.groupby("Firm", sort=False)["Step"].idxmax(),
        ["Firm", "Agent Type", "Capital"],
    ]
    for firm, agent_type, capital in final_rows.itertuples(index=False, name=None):
        firm = int(firm)
        accumulator.agent_type_by_firm[firm] = normalize_agent_type(agent_type)
        accumulator.final_capital_by_firm[firm] = float(capital)


def finalize_simulation(
    accumulator: SimulationAccumulator,
    sole_counts: dict[str, Counter[str]],
    portfolio_counts: dict[str, Counter[str]],
    outcome_run_counts: Counter[str],
) -> None:
    """Classify one run and update sole-occupant portfolio-time totals."""
    if not accumulator.final_capital_by_firm:
        return

    max_capital = max(accumulator.final_capital_by_firm.values())
    ai_firms = [
        firm
        for firm, agent_type in accumulator.agent_type_by_firm.items()
        if agent_type == "AI"
    ]
    if not ai_firms:
        raise ValueError(
            f"No AI firm found in {accumulator.zip_path} simulation {accumulator.raw_sim}"
        )

    ai_won = any(
        np.isclose(accumulator.final_capital_by_firm[firm], max_capital)
        for firm in ai_firms
    )
    outcome = "wins" if ai_won else "losses"
    outcome_run_counts[outcome] += 1

    firms = sorted({firm for firm, _ in accumulator.presence_by_key})
    markets = sorted({market for _, market in accumulator.presence_by_key})
    if not firms or not markets:
        return

    for market in markets:
        step_union = sorted(
            {
                step
                for firm in firms
                for step in accumulator.steps_by_key.get((firm, market), [])
            }
        )
        if not step_union:
            continue

        step_position = {step: index for index, step in enumerate(step_union)}
        market_presence = np.zeros((len(firms), len(step_union)), dtype=np.int8)

        for firm_index, firm in enumerate(firms):
            steps = accumulator.steps_by_key.get((firm, market))
            presence = accumulator.presence_by_key.get((firm, market))
            if not steps or not presence:
                continue
            positions = [step_position[step] for step in steps]
            market_presence[firm_index, positions] = presence

        total_present = market_presence.sum(axis=0)
        for firm_index, firm in enumerate(firms):
            series = market_presence[firm_index]
            portfolio_mask = series == 1
            portfolio_time = int(portfolio_mask.sum())
            if portfolio_time == 0:
                continue

            sole_time = int(np.logical_and(portfolio_mask, total_present == 1).sum())
            agent_type = accumulator.agent_type_by_firm.get(firm, str(firm))
            portfolio_counts[outcome][agent_type] += portfolio_time
            sole_counts[outcome][agent_type] += sole_time


def stream_zip(zip_path: Path, chunksize: int) -> ZipTotals:
    """Stream one ZIP archive and return its panel-level totals."""
    master_member = get_master_output_member(zip_path)
    current: SimulationAccumulator | None = None
    processed_runs = 0
    sole_counts: dict[str, Counter[str]] = defaultdict(Counter)
    portfolio_counts: dict[str, Counter[str]] = defaultdict(Counter)
    outcome_run_counts: Counter[str] = Counter()

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(master_member) as handle:
            reader = pd.read_csv(
                handle,
                usecols=REQUIRED_COLUMNS,
                chunksize=chunksize,
            )
            for chunk in reader:
                chunk = chunk.sort_values(["Sim", "Firm", "Market", "Step"])
                for sim, sim_frame in chunk.groupby("Sim", sort=False):
                    sim_id = int(sim)
                    if current is None:
                        current = SimulationAccumulator.create(zip_path, sim_id)
                    elif sim_id != current.raw_sim:
                        finalize_simulation(
                            current,
                            sole_counts,
                            portfolio_counts,
                            outcome_run_counts,
                        )
                        processed_runs += 1
                        if processed_runs % 100 == 0:
                            log(f"{zip_path.parent.name}: processed {processed_runs} runs")
                        current = SimulationAccumulator.create(zip_path, sim_id)

                    append_rows(current, sim_frame)

    if current is not None:
        finalize_simulation(
            current,
            sole_counts,
            portfolio_counts,
            outcome_run_counts,
        )
        processed_runs += 1

    log(f"Finished {zip_path.parent.name}: {processed_runs:,} runs")
    return ZipTotals(
        zip_path=zip_path,
        sole_counts=sole_counts,
        portfolio_counts=portfolio_counts,
        outcome_run_counts=outcome_run_counts,
    )


def build_zip_level_rows(zip_totals: list[ZipTotals]) -> pd.DataFrame:
    """Convert per-ZIP totals into per-trained-agent means."""
    rows: list[dict[str, object]] = []
    for totals in zip_totals:
        for outcome in OUTCOME_ORDER:
            for agent_type in AGENT_TYPE_ORDER:
                denominator = totals.portfolio_counts[outcome].get(agent_type, 0)
                numerator = totals.sole_counts[outcome].get(agent_type, 0)
                rows.append(
                    {
                        "zip_path": str(totals.zip_path),
                        "trained_agent": totals.zip_path.parent.name,
                        "Outcome": outcome,
                        "Panel": OUTCOME_LABELS[outcome],
                        "Agent Type": agent_type,
                        "AI Panel Run Count": totals.outcome_run_counts[outcome],
                        "Sole Occupant Portfolio-Time": numerator,
                        "Portfolio-Time": denominator,
                        "Sole Occupant Share": numerator / denominator
                        if denominator
                        else np.nan,
                    }
                )

    return pd.DataFrame(rows)


def confidence_interval_95(values: pd.Series) -> float:
    """Return the 95% confidence interval half-width."""
    clean_values = values.dropna()
    count = clean_values.count()
    if count <= 1:
        return 0.0
    return float(1.96 * clean_values.std(ddof=1) / np.sqrt(count))


def build_summary_df(zip_level_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ZIP-level means into plot means and CIs."""
    grouped = zip_level_df.groupby(["Outcome", "Panel", "Agent Type"])["Sole Occupant Share"]
    summary = grouped.agg(Mean="mean", N="count").reset_index()
    summary["CI95"] = grouped.apply(confidence_interval_95).to_numpy()
    return summary


def lighten_color(color: str, amount: float = 0.45) -> tuple[float, float, float]:
    """Blend a color toward white by ``amount``."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot sole-occupant shares with 95% CIs across trained-agent means."""
    fig, ax = plt.subplots(figsize=(16, 5))
    x = np.arange(len(AGENT_TYPE_ORDER))
    width = 0.34
    offsets = {
        "wins": -width / 2,
        "losses": width / 2,
    }

    max_height = 0.0
    for outcome in OUTCOME_ORDER:
        means: list[float] = []
        cis: list[float] = []
        colors: list[tuple[float, float, float] | str] = []
        for agent_type in AGENT_TYPE_ORDER:
            row = summary_df[
                (summary_df["Outcome"] == outcome)
                & (summary_df["Agent Type"] == agent_type)
            ]
            if row.empty:
                means.append(0.0)
                cis.append(0.0)
            else:
                means.append(float(row["Mean"].iloc[0]))
                cis.append(float(row["CI95"].iloc[0]))
            base_color = TYPE_TO_COLOR[agent_type]
            colors.append(base_color if outcome == "wins" else lighten_color(base_color))

        max_height = max(max_height, max((mean + ci for mean, ci in zip(means, cis)), default=0.0))
        ax.bar(
            x + offsets[outcome],
            means,
            width=width,
            color=colors,
            alpha=0.9,
        )
        ax.errorbar(
            x + offsets[outcome],
            means,
            yerr=cis,
            fmt="none",
            ecolor="#333333",
            elinewidth=1.5,
            capsize=4,
        )

    ax.set_title(
        "Sole-occupant share by agent type and panel",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_ylabel("Share of portfolio-time as sole occupant", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_TYPE_ORDER, fontsize=12)
    ax.set_ylim(0, min(1.0, max(0.16, max_height * 1.25)))
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(facecolor="#7f7f7f", label=OUTCOME_LABELS["wins"]),
        mpatches.Patch(facecolor="#bdbdbd", label=OUTCOME_LABELS["losses"]),
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="upper right", fontsize=12)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stream trained-agent MasterOutput_MarketOverlap.zip files under an "
            "economy folder and plot sole-occupant portfolio-time shares by "
            "agent type and AI win/loss panel."
        )
    )
    parser.add_argument(
        "economy_dir",
        type=Path,
        help="Folder containing one child folder per trained agent.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG and CSV outputs (default: economy_dir/sole_occupant_share_analysis).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per pandas streaming chunk (default: 1,000,000).",
    )
    return parser.parse_args()


def main() -> None:
    """Run analysis and write outputs."""
    args = parse_args()
    economy_dir = args.economy_dir.resolve()
    if not economy_dir.is_dir():
        raise SystemExit(f"Not a directory: {economy_dir}")
    if args.chunksize <= 0:
        raise SystemExit("--chunksize must be positive")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else economy_dir / "sole_occupant_share_analysis"
    )

    zip_paths = discover_zip_paths(economy_dir)
    if not zip_paths:
        raise SystemExit(f"No {MASTER_OUTPUT_ZIP_NAME} files found under {economy_dir}")

    readable_zip_paths = [zip_path for zip_path in zip_paths if is_readable_zip(zip_path)]
    skipped_zip_paths = [zip_path for zip_path in zip_paths if zip_path not in readable_zip_paths]
    if not readable_zip_paths:
        raise SystemExit(f"No readable {MASTER_OUTPUT_ZIP_NAME} files found under {economy_dir}")

    zip_totals: list[ZipTotals] = []
    for index, zip_path in enumerate(readable_zip_paths, start=1):
        log(f"Processing ZIP {index}/{len(readable_zip_paths)}: {zip_path}")
        zip_totals.append(stream_zip(zip_path, args.chunksize))

    zip_level_df = build_zip_level_rows(zip_totals)
    summary_df = build_summary_df(zip_level_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    zip_level_csv = output_dir / "sole_occupant_share_by_trained_agent.csv"
    summary_csv = output_dir / "sole_occupant_share_95ci.csv"
    plot_png = output_dir / "sole_occupant_share_by_panel.png"
    zip_level_df.to_csv(zip_level_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    plot_summary(summary_df, plot_png)

    total_runs = sum(sum(totals.outcome_run_counts.values()) for totals in zip_totals)
    total_wins = sum(totals.outcome_run_counts["wins"] for totals in zip_totals)
    total_losses = sum(totals.outcome_run_counts["losses"] for totals in zip_totals)
    print(f"Processed trained-agent datasets: {len(zip_totals):,}")
    print(f"Processed runs: {total_runs:,}")
    print(f"AI wins: {total_wins:,}")
    print(f"AI losses: {total_losses:,}")
    print(f"Wrote trained-agent means: {zip_level_csv}")
    print(f"Wrote 95% CI summary: {summary_csv}")
    print(f"Wrote plot: {plot_png}")
    if skipped_zip_paths:
        print("Skipped unreadable ZIPs:")
        for zip_path in skipped_zip_paths:
            print(f"  {zip_path}")


if __name__ == "__main__":
    main()
