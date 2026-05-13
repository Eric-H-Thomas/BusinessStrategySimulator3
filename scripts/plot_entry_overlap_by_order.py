#!/usr/bin/env python3
# This code was AI-generated and still requires verification by a human researcher. Remove this comment when done.
"""Plot entry-market overlap by entry order with trained-agent 95% CIs.

The simulator output does not include each firm's full capability portfolio at
entry time. As an approximation, this script measures each entry market's
highest pairwise capability overlap with any market already in the firm's
portfolio immediately before entry. It then averages by trained-agent dataset,
agent type, AI win/loss panel, and entry order, and plots the mean with a 95%
confidence interval across the trained-agent datasets.
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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MASTER_OUTPUT_ZIP_NAME = "MasterOutput_MarketOverlap.zip"
MASTER_OUTPUT_FILE_NAME = "MasterOutput.csv"
MARKET_OVERLAP_FILE_NAME = "MarketOverlap.csv"
MARKET_OVERLAP_COLUMN = "Percentage Cost Overlap (A^B/A)"
REQUIRED_MASTER_COLUMNS = [
    "Sim",
    "Step",
    "Firm",
    "Agent Type",
    "Market",
    "Capital",
    "In Market",
]
REQUIRED_OVERLAP_COLUMNS = [
    "Sim",
    "Market A",
    "Market B",
    MARKET_OVERLAP_COLUMN,
]
AGENT_TYPE_ORDER = ["AI", "Sophisticated", "Naive"]
OUTCOME_ORDER = ["wins", "losses"]
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
class ZipResult:
    """Per-ZIP entry-order results."""

    zip_path: Path
    entry_rows: list[dict[str, object]]
    run_count: int
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


def get_member(zip_path: Path, basename: str) -> str:
    """Return an archive member by basename, case-insensitively if needed."""
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.namelist():
            if Path(member).name == basename:
                return member
        for member in archive.namelist():
            if Path(member).name.lower() == basename.lower():
                return member

    raise ValueError(f"{basename} not found inside {zip_path}")


def load_overlap_lookup(zip_path: Path) -> dict[int, dict[tuple[int, int], float]]:
    """Load MarketOverlap.csv into a per-simulation pairwise overlap lookup."""
    overlap_member = get_member(zip_path, MARKET_OVERLAP_FILE_NAME)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(overlap_member) as handle:
            frame = pd.read_csv(handle, usecols=REQUIRED_OVERLAP_COLUMNS)

    lookup: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
    for sim, market_a, market_b, overlap in frame.itertuples(index=False, name=None):
        lookup[int(sim)][(int(market_a), int(market_b))] = float(overlap)

    return lookup


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


def classify_outcome(accumulator: SimulationAccumulator) -> str:
    """Return wins/losses depending on whether AI finished with top capital."""
    if not accumulator.final_capital_by_firm:
        return "losses"

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
    return "wins" if ai_won else "losses"


def finalize_simulation(
    accumulator: SimulationAccumulator,
    overlap_lookup: dict[int, dict[tuple[int, int], float]],
    entry_rows: list[dict[str, object]],
    entry_order_mode: str,
    max_entry_order: int | None,
) -> str:
    """Classify one run and collect entry overlap by order."""
    outcome = classify_outcome(accumulator)
    firms = sorted({firm for firm, _ in accumulator.presence_by_key})
    markets = sorted({market for _, market in accumulator.presence_by_key})
    if not firms or not markets:
        return outcome

    sim_overlap_lookup = overlap_lookup.get(accumulator.raw_sim, {})
    market_index_by_id = {market: index for index, market in enumerate(markets)}

    for firm in firms:
        step_union = sorted(
            {
                step
                for market in markets
                for step in accumulator.steps_by_key.get((firm, market), [])
            }
        )
        if not step_union:
            continue

        step_position = {step: index for index, step in enumerate(step_union)}
        firm_presence = np.zeros((len(markets), len(step_union)), dtype=np.int8)

        for market in markets:
            steps = accumulator.steps_by_key.get((firm, market))
            presence = accumulator.presence_by_key.get((firm, market))
            if not steps or not presence:
                continue
            positions = [step_position[step] for step in steps]
            firm_presence[market_index_by_id[market], positions] = presence

        raw_entry_events: list[tuple[int, int, int]] = []
        for entry_market in markets:
            entry_market_index = market_index_by_id[entry_market]
            series = firm_presence[entry_market_index]
            previous_presence = np.concatenate(([0], series[:-1]))
            entry_positions = np.flatnonzero((series == 1) & (previous_presence == 0))
            raw_entry_events.extend(
                (int(position), int(step_union[position]), int(entry_market))
                for position in entry_positions
            )

        raw_entry_events.sort(key=lambda item: (item[1], item[2]))
        entry_order = 0
        seen_markets: set[int] = set()
        agent_type = accumulator.agent_type_by_firm.get(firm, str(firm))

        for position, step, entry_market in raw_entry_events:
            if entry_order_mode == "distinct-markets" and entry_market in seen_markets:
                continue
            seen_markets.add(entry_market)
            entry_order += 1

            if max_entry_order is not None and entry_order > max_entry_order:
                continue
            if position == 0:
                continue

            portfolio_indices = np.flatnonzero(firm_presence[:, position - 1] == 1)
            portfolio_markets = [
                markets[index]
                for index in portfolio_indices
                if markets[index] != entry_market
            ]
            if not portfolio_markets:
                continue

            max_overlap = max(
                sim_overlap_lookup.get((entry_market, portfolio_market), np.nan)
                for portfolio_market in portfolio_markets
            )
            if np.isnan(max_overlap):
                continue

            entry_rows.append(
                {
                    "zip_path": str(accumulator.zip_path),
                    "trained_agent": accumulator.zip_path.parent.name,
                    "Sim": accumulator.raw_sim,
                    "Outcome": outcome,
                    "Agent Type": agent_type,
                    "Firm": firm,
                    "Step": step,
                    "Entry Market": entry_market,
                    "Entry Order": entry_order,
                    "Portfolio Size Before Entry": len(portfolio_markets),
                    "Max Overlap With Portfolio Market": float(max_overlap),
                }
            )

    return outcome


def stream_zip(
    zip_path: Path,
    chunksize: int,
    entry_order_mode: str,
    max_entry_order: int | None,
) -> ZipResult:
    """Stream one ZIP archive and return entry-order observations."""
    overlap_lookup = load_overlap_lookup(zip_path)
    master_member = get_member(zip_path, MASTER_OUTPUT_FILE_NAME)
    current: SimulationAccumulator | None = None
    processed_runs = 0
    outcome_run_counts: Counter[str] = Counter()
    entry_rows: list[dict[str, object]] = []

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(master_member) as handle:
            reader = pd.read_csv(
                handle,
                usecols=REQUIRED_MASTER_COLUMNS,
                chunksize=chunksize,
            )
            for chunk in reader:
                chunk = chunk.sort_values(["Sim", "Firm", "Market", "Step"])
                for sim, sim_frame in chunk.groupby("Sim", sort=False):
                    sim_id = int(sim)
                    if current is None:
                        current = SimulationAccumulator.create(zip_path, sim_id)
                    elif sim_id != current.raw_sim:
                        outcome = finalize_simulation(
                            current,
                            overlap_lookup,
                            entry_rows,
                            entry_order_mode,
                            max_entry_order,
                        )
                        outcome_run_counts[outcome] += 1
                        processed_runs += 1
                        if processed_runs % 100 == 0:
                            log(f"{zip_path.parent.name}: processed {processed_runs} runs")
                        current = SimulationAccumulator.create(zip_path, sim_id)

                    append_rows(current, sim_frame)

    if current is not None:
        outcome = finalize_simulation(
            current,
            overlap_lookup,
            entry_rows,
            entry_order_mode,
            max_entry_order,
        )
        outcome_run_counts[outcome] += 1
        processed_runs += 1

    log(f"Finished {zip_path.parent.name}: {processed_runs:,} runs")
    return ZipResult(
        zip_path=zip_path,
        entry_rows=entry_rows,
        run_count=processed_runs,
        outcome_run_counts=outcome_run_counts,
    )


def build_trained_agent_means(entry_df: pd.DataFrame) -> pd.DataFrame:
    """Average entry observations within each trained-agent dataset."""
    grouped_columns = [
        "zip_path",
        "trained_agent",
        "Outcome",
        "Agent Type",
        "Entry Order",
    ]
    grouped = entry_df.groupby(grouped_columns)["Max Overlap With Portfolio Market"]
    means = grouped.agg(
        Entry_Count="count",
        Mean_Entry_Overlap="mean",
    ).reset_index()
    return means


def confidence_interval_95(values: pd.Series) -> float:
    """Return the 95% confidence interval half-width."""
    clean_values = values.dropna()
    count = clean_values.count()
    if count <= 1:
        return 0.0
    return float(1.96 * clean_values.std(ddof=1) / np.sqrt(count))


def build_summary(trained_agent_means: pd.DataFrame) -> pd.DataFrame:
    """Summarize trained-agent means into plot means and CIs."""
    grouped_columns = ["Outcome", "Agent Type", "Entry Order"]
    grouped = trained_agent_means.groupby(grouped_columns)["Mean_Entry_Overlap"]
    summary = grouped.agg(Mean="mean", N="count").reset_index()
    summary["CI95"] = grouped.apply(confidence_interval_95).to_numpy()
    return summary


def plot_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Plot entry-overlap means by entry order with 95% CI ribbons."""
    fig, ax = plt.subplots(figsize=(8, 6))
    linestyle_by_outcome = {
        "wins": "-",
        "losses": "--",
    }

    for agent_type in AGENT_TYPE_ORDER:
        for outcome in OUTCOME_ORDER:
            series = summary_df[
                (summary_df["Agent Type"] == agent_type)
                & (summary_df["Outcome"] == outcome)
            ].sort_values("Entry Order")
            if series.empty:
                continue

            x = series["Entry Order"].to_numpy(dtype=float)
            y = series["Mean"].to_numpy(dtype=float)
            ci = series["CI95"].to_numpy(dtype=float)
            color = TYPE_TO_COLOR[agent_type]
            label = f"{agent_type} ({outcome})"
            ax.plot(
                x,
                y,
                linestyle=linestyle_by_outcome[outcome],
                color=color,
                linewidth=2,
                label=label,
            )
            ax.fill_between(x, y - ci, y + ci, color=color, alpha=0.12)

    ax.set_title(
        "Overlap at entry by entry order",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_xlabel("Entry order (1 = first market entered)", fontsize=13)
    ax.set_ylabel("Mean capability overlap at entry", fontsize=13)
    min_order = int(summary_df["Entry Order"].min())
    max_order = int(summary_df["Entry Order"].max())
    ax.set_xticks(np.arange(min_order, max_order + 1))
    ax.set_ylim(0, 1.0)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=10, ncol=2, loc="upper right")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stream trained-agent MasterOutput_MarketOverlap.zip files under an "
            "economy folder and plot market-entry overlap by entry order."
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
        help="Directory for PNG and CSV outputs (default: economy_dir/entry_overlap_by_order_analysis).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per pandas streaming chunk (default: 1,000,000).",
    )
    parser.add_argument(
        "--entry-order-mode",
        choices=("distinct-markets", "all-events"),
        default="distinct-markets",
        help=(
            "How to count entry order. distinct-markets counts only the first "
            "time a firm enters each market; all-events counts re-entries too. "
            "Default: distinct-markets."
        ),
    )
    parser.add_argument(
        "--max-entry-order",
        type=int,
        default=10,
        help="Maximum entry order to include after excluding undefined first entries (default: 10).",
    )
    parser.add_argument(
        "--write-entry-level",
        action="store_true",
        help="Also write one row per entry event with a computable overlap.",
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
    if args.max_entry_order is not None and args.max_entry_order < 2:
        raise SystemExit("--max-entry-order must be at least 2")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else economy_dir / "entry_overlap_by_order_analysis"
    )

    zip_paths = discover_zip_paths(economy_dir)
    if not zip_paths:
        raise SystemExit(f"No {MASTER_OUTPUT_ZIP_NAME} files found under {economy_dir}")

    readable_zip_paths = [zip_path for zip_path in zip_paths if is_readable_zip(zip_path)]
    skipped_zip_paths = [zip_path for zip_path in zip_paths if zip_path not in readable_zip_paths]
    if not readable_zip_paths:
        raise SystemExit(f"No readable {MASTER_OUTPUT_ZIP_NAME} files found under {economy_dir}")

    results: list[ZipResult] = []
    for index, zip_path in enumerate(readable_zip_paths, start=1):
        log(f"Processing ZIP {index}/{len(readable_zip_paths)}: {zip_path}")
        results.append(
            stream_zip(
                zip_path,
                args.chunksize,
                args.entry_order_mode,
                args.max_entry_order,
            )
        )

    entry_rows = [row for result in results for row in result.entry_rows]
    if not entry_rows:
        raise SystemExit("No computable entry-overlap observations were produced.")

    entry_df = pd.DataFrame(entry_rows)
    trained_agent_means = build_trained_agent_means(entry_df)
    summary_df = build_summary(trained_agent_means)

    output_dir.mkdir(parents=True, exist_ok=True)
    trained_agent_csv = output_dir / "entry_overlap_by_order_trained_agent_means.csv"
    summary_csv = output_dir / "entry_overlap_by_order_95ci.csv"
    plot_png = output_dir / "entry_overlap_by_order.png"
    trained_agent_means.to_csv(trained_agent_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    plot_summary(summary_df, plot_png)

    entry_csv = None
    if args.write_entry_level:
        entry_csv = output_dir / "entry_overlap_by_order_entry_level.csv"
        entry_df.to_csv(entry_csv, index=False)

    total_runs = sum(result.run_count for result in results)
    total_wins = sum(result.outcome_run_counts["wins"] for result in results)
    total_losses = sum(result.outcome_run_counts["losses"] for result in results)
    print(f"Processed trained-agent datasets: {len(results):,}")
    print(f"Processed runs: {total_runs:,}")
    print(f"AI wins: {total_wins:,}")
    print(f"AI losses: {total_losses:,}")
    print(f"Wrote trained-agent means: {trained_agent_csv}")
    print(f"Wrote 95% CI summary: {summary_csv}")
    print(f"Wrote plot: {plot_png}")
    if entry_csv is not None:
        print(f"Wrote entry-level observations: {entry_csv}")
    if skipped_zip_paths:
        print("Skipped unreadable ZIPs:")
        for zip_path in skipped_zip_paths:
            print(f"  {zip_path}")


if __name__ == "__main__":
    main()
