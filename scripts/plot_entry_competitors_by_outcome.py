#!/usr/bin/env python3
# This code was AI-generated and still requires verification by a human researcher. Remove this comment when done.
"""Plot market-entry competitor counts split by AI win/loss outcomes.

The input is an economy folder containing one child folder per trained agent.
Each child folder is expected to contain ``MasterOutput_MarketOverlap.zip`` with
``MasterOutput.csv`` inside it. The master CSV can be very large, so this script
streams it in chunks and only keeps one simulation's compact state in memory.
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
COMPETITOR_BINS = ["0", "1", "2", "3", "4+"]
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
class AnalysisResult:
    """Aggregated output from all processed simulations."""

    counts: dict[str, dict[str, Counter[str]]]
    outcome_run_counts: Counter[str]
    processed_runs: int
    skipped_zip_paths: list[Path]


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


def bin_competitor_count(count: int) -> str:
    """Collapse competitor counts into 0, 1, 2, 3, and 4+ bins."""
    return "4+" if count >= 4 else str(count)


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
    counts: dict[str, dict[str, Counter[str]]],
    outcome_run_counts: Counter[str],
) -> None:
    """Classify one run and add its market-entry counts to the aggregate."""
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
            previous_presence = np.concatenate(([0], series[:-1]))
            entry_positions = np.flatnonzero((series == 1) & (previous_presence == 0))
            if entry_positions.size == 0:
                continue

            agent_type = accumulator.agent_type_by_firm.get(firm, str(firm))
            for position in entry_positions:
                # Competitors are measured immediately before the entry action.
                if position == 0:
                    competitors_present = 0
                else:
                    competitors_present = int(total_present[position - 1] - series[position - 1])
                competitor_bin = bin_competitor_count(competitors_present)
                counts[outcome][agent_type][competitor_bin] += 1


def stream_zip(
    zip_path: Path,
    counts: dict[str, dict[str, Counter[str]]],
    outcome_run_counts: Counter[str],
    chunksize: int,
) -> int:
    """Stream one ZIP archive and return the number of simulations processed."""
    master_member = get_master_output_member(zip_path)
    processed_runs = 0
    current: SimulationAccumulator | None = None

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
                        finalize_simulation(current, counts, outcome_run_counts)
                        processed_runs += 1
                        if processed_runs % 100 == 0:
                            log(f"{zip_path.parent.name}: processed {processed_runs} runs")
                        current = SimulationAccumulator.create(zip_path, sim_id)

                    append_rows(current, sim_frame)

    if current is not None:
        finalize_simulation(current, counts, outcome_run_counts)
        processed_runs += 1

    return processed_runs


def write_summary_csv(counts: dict[str, dict[str, Counter[str]]], output_path: Path) -> None:
    """Write long-format entry shares/counts used by the plot."""
    rows: list[dict[str, object]] = []
    for outcome in ["wins", "losses"]:
        for agent_type in AGENT_TYPE_ORDER:
            agent_counts = counts[outcome].get(agent_type, Counter())
            total = sum(agent_counts.values())
            for competitor_bin in COMPETITOR_BINS:
                entry_count = agent_counts.get(competitor_bin, 0)
                rows.append(
                    {
                        "Outcome": outcome,
                        "Agent Type": agent_type,
                        "Competitors Present": competitor_bin,
                        "Entry Count": entry_count,
                        "Share of Entries": entry_count / total if total else 0.0,
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def plot_counts(
    counts: dict[str, dict[str, Counter[str]]],
    outcome_run_counts: Counter[str],
    output_path: Path,
) -> None:
    """Create the two-panel win/loss grouped-bar plot."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    x = np.arange(len(COMPETITOR_BINS))
    bar_width = 0.25
    max_share = 0.0
    offsets = {
        "AI": -bar_width,
        "Sophisticated": 0.0,
        "Naive": bar_width,
    }

    for ax, outcome, title_label in zip(
        axes,
        ["wins", "losses"],
        ["AI wins", "AI losses"],
    ):
        for agent_type in AGENT_TYPE_ORDER:
            agent_counts = counts[outcome].get(agent_type, Counter())
            total_entries = sum(agent_counts.values())
            shares = [
                agent_counts.get(competitor_bin, 0) / total_entries
                if total_entries
                else 0.0
                for competitor_bin in COMPETITOR_BINS
            ]
            max_share = max(max_share, max(shares, default=0.0))
            ax.bar(
                x + offsets[agent_type],
                shares,
                width=bar_width,
                color=TYPE_TO_COLOR[agent_type],
                label=agent_type,
                alpha=0.9,
            )

        ax.set_title(
            f"{title_label} (n = {outcome_run_counts[outcome]:,} runs)",
            fontsize=17,
            fontweight="bold",
        )
        ax.set_xlabel("Competitors present in market at moment of entry", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(COMPETITOR_BINS, fontsize=12)
        ax.grid(axis="both", linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel("Share of entries", fontsize=14)
    axes[0].legend(frameon=False, fontsize=12, loc="upper right")
    axes[0].set_ylim(0, min(1.0, max(0.55, max_share * 1.15)))

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def analyze_economy_dir(economy_dir: Path, chunksize: int) -> AnalysisResult:
    """Process all trained-agent ZIPs below one economy directory."""
    zip_paths = discover_zip_paths(economy_dir)
    if not zip_paths:
        raise SystemExit(f"No {MASTER_OUTPUT_ZIP_NAME} files found under {economy_dir}")

    counts: dict[str, dict[str, Counter[str]]] = {
        "wins": defaultdict(Counter),
        "losses": defaultdict(Counter),
    }
    outcome_run_counts: Counter[str] = Counter()
    skipped_zip_paths: list[Path] = []
    processed_runs = 0

    readable_zip_paths = []
    for zip_path in zip_paths:
        if is_readable_zip(zip_path):
            readable_zip_paths.append(zip_path)
        else:
            skipped_zip_paths.append(zip_path)

    if not readable_zip_paths:
        raise SystemExit(f"No readable {MASTER_OUTPUT_ZIP_NAME} files found under {economy_dir}")

    for index, zip_path in enumerate(readable_zip_paths, start=1):
        log(f"Processing ZIP {index}/{len(readable_zip_paths)}: {zip_path}")
        processed_for_zip = stream_zip(zip_path, counts, outcome_run_counts, chunksize)
        processed_runs += processed_for_zip
        log(f"Finished {zip_path.parent.name}: {processed_for_zip:,} runs")

    return AnalysisResult(
        counts=counts,
        outcome_run_counts=outcome_run_counts,
        processed_runs=processed_runs,
        skipped_zip_paths=skipped_zip_paths,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stream trained-agent MasterOutput_MarketOverlap.zip files under an "
            "economy folder and plot market-entry competitor counts split by AI "
            "wins/losses."
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
        help="Directory for the PNG and CSV outputs (default: economy_dir/entry_competitor_analysis).",
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
        else economy_dir / "entry_competitor_analysis"
    )

    result = analyze_economy_dir(economy_dir, args.chunksize)

    summary_csv = output_dir / "entry_competitors_by_outcome.csv"
    plot_png = output_dir / "entry_competitors_by_outcome.png"
    write_summary_csv(result.counts, summary_csv)
    plot_counts(result.counts, result.outcome_run_counts, plot_png)

    print(f"Processed runs: {result.processed_runs:,}")
    print(f"AI wins: {result.outcome_run_counts['wins']:,}")
    print(f"AI losses: {result.outcome_run_counts['losses']:,}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote plot: {plot_png}")
    if result.skipped_zip_paths:
        print("Skipped unreadable ZIPs:")
        for zip_path in result.skipped_zip_paths:
            print(f"  {zip_path}")


if __name__ == "__main__":
    main()
