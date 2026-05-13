#!/usr/bin/env python3
# This code was AI-generated and still requires verification by a human researcher. Remove this comment when done.
"""Plot entry-market overlap with the closest market already in portfolio.

The simulator output does not include each firm's full capability portfolio at
entry time. As an approximation, this script measures each entry market's
highest pairwise capability overlap with any market already in the firm's
portfolio immediately before entry, then plots per-run means by agent type and
AI win/loss panel.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "business_strategy_matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
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
    """Per-ZIP results."""

    zip_path: Path
    run_rows: list[dict[str, object]]
    entry_rows: list[dict[str, object]]
    run_count: int
    win_count: int
    loss_count: int


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
    for row in frame.itertuples(index=False, name=None):
        sim, market_a, market_b, overlap = row
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
    empty_portfolio: str,
    run_rows: list[dict[str, object]],
    entry_rows: list[dict[str, object]],
) -> str:
    """Classify one run and collect entry-overlap observations."""
    outcome = classify_outcome(accumulator)
    firms = sorted({firm for firm, _ in accumulator.presence_by_key})
    markets = sorted({market for _, market in accumulator.presence_by_key})
    if not firms or not markets:
        return outcome

    sim_overlap_lookup = overlap_lookup.get(accumulator.raw_sim, {})
    overlap_values_by_agent_type: dict[str, list[float]] = defaultdict(list)

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

        agent_type = accumulator.agent_type_by_firm.get(firm, str(firm))
        for entry_market in markets:
            entry_market_index = market_index_by_id[entry_market]
            series = firm_presence[entry_market_index]
            previous_presence = np.concatenate(([0], series[:-1]))
            entry_positions = np.flatnonzero((series == 1) & (previous_presence == 0))
            for position in entry_positions:
                if position == 0:
                    portfolio_markets: list[int] = []
                else:
                    portfolio_indices = np.flatnonzero(firm_presence[:, position - 1] == 1)
                    portfolio_markets = [
                        markets[index]
                        for index in portfolio_indices
                        if markets[index] != entry_market
                    ]

                if not portfolio_markets:
                    if empty_portfolio == "skip":
                        continue
                    max_overlap = 0.0
                else:
                    max_overlap = max(
                        sim_overlap_lookup.get((entry_market, portfolio_market), np.nan)
                        for portfolio_market in portfolio_markets
                    )
                    if np.isnan(max_overlap):
                        continue

                overlap_values_by_agent_type[agent_type].append(max_overlap)
                entry_rows.append(
                    {
                        "zip_path": str(accumulator.zip_path),
                        "trained_agent": accumulator.zip_path.parent.name,
                        "Sim": accumulator.raw_sim,
                        "Outcome": outcome,
                        "Agent Type": agent_type,
                        "Firm": firm,
                        "Step": step_union[position],
                        "Entry Market": entry_market,
                        "Portfolio Size Before Entry": len(portfolio_markets),
                        "Max Overlap With Portfolio Market": max_overlap,
                    }
                )

    for agent_type in AGENT_TYPE_ORDER:
        values = overlap_values_by_agent_type.get(agent_type, [])
        if not values:
            continue
        run_rows.append(
            {
                "zip_path": str(accumulator.zip_path),
                "trained_agent": accumulator.zip_path.parent.name,
                "Sim": accumulator.raw_sim,
                "Outcome": outcome,
                "Agent Type": agent_type,
                "Entry Count": len(values),
                "Mean Entry Overlap": float(np.mean(values)),
            }
        )

    return outcome


def stream_zip(zip_path: Path, chunksize: int, empty_portfolio: str) -> ZipResult:
    """Stream one ZIP archive and return per-run entry-overlap means."""
    overlap_lookup = load_overlap_lookup(zip_path)
    master_member = get_member(zip_path, MASTER_OUTPUT_FILE_NAME)
    current: SimulationAccumulator | None = None
    processed_runs = 0
    win_count = 0
    loss_count = 0
    run_rows: list[dict[str, object]] = []
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
                            empty_portfolio,
                            run_rows,
                            entry_rows,
                        )
                        if outcome == "wins":
                            win_count += 1
                        else:
                            loss_count += 1
                        processed_runs += 1
                        if processed_runs % 100 == 0:
                            log(f"{zip_path.parent.name}: processed {processed_runs} runs")
                        current = SimulationAccumulator.create(zip_path, sim_id)

                    append_rows(current, sim_frame)

    if current is not None:
        outcome = finalize_simulation(
            current,
            overlap_lookup,
            empty_portfolio,
            run_rows,
            entry_rows,
        )
        if outcome == "wins":
            win_count += 1
        else:
            loss_count += 1
        processed_runs += 1

    log(f"Finished {zip_path.parent.name}: {processed_runs:,} runs")
    return ZipResult(
        zip_path=zip_path,
        run_rows=run_rows,
        entry_rows=entry_rows,
        run_count=processed_runs,
        win_count=win_count,
        loss_count=loss_count,
    )


def lighten_color(color: str, amount: float = 0.45) -> tuple[float, float, float]:
    """Blend a color toward white by ``amount``."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def plot_boxplot(run_df: pd.DataFrame, output_path: Path) -> None:
    """Create grouped box-and-whisker plot for per-run mean entry overlap."""
    fig, ax = plt.subplots(figsize=(7, 7))
    positions = {
        ("AI", "wins"): 1.0,
        ("AI", "losses"): 2.0,
        ("Sophisticated", "wins"): 4.0,
        ("Sophisticated", "losses"): 5.0,
        ("Naive", "wins"): 7.0,
        ("Naive", "losses"): 8.0,
    }

    data: list[np.ndarray] = []
    plot_positions: list[float] = []
    colors: list[tuple[float, float, float] | str] = []
    for agent_type in AGENT_TYPE_ORDER:
        for outcome in OUTCOME_ORDER:
            values = run_df[
                (run_df["Agent Type"] == agent_type)
                & (run_df["Outcome"] == outcome)
            ]["Mean Entry Overlap"].dropna()
            data.append(values.to_numpy())
            plot_positions.append(positions[(agent_type, outcome)])
            base_color = TYPE_TO_COLOR[agent_type]
            colors.append(base_color if outcome == "wins" else lighten_color(base_color))

    boxplot = ax.boxplot(
        data,
        positions=plot_positions,
        widths=0.65,
        patch_artist=True,
        showfliers=True,
        medianprops={"color": "black", "linewidth": 1.8},
        boxprops={"linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
        flierprops={
            "marker": "o",
            "markerfacecolor": "#777777",
            "markeredgecolor": "#777777",
            "markersize": 2.4,
            "alpha": 0.35,
        },
    )

    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.9)

    ax.set_ylabel("Mean capability overlap at entry (per run)", fontsize=13)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0.5, 8.5)
    ax.set_xticks(plot_positions)
    ax.set_xticklabels(["wins", "loses"] * 3, fontsize=10)
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for center, label in zip([1.5, 4.5, 7.5], AGENT_TYPE_ORDER):
        ax.text(
            center,
            -0.085,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_summary(run_df: pd.DataFrame, output_path: Path) -> None:
    """Write descriptive summary by agent type and outcome."""
    grouped = run_df.groupby(["Agent Type", "Outcome"])["Mean Entry Overlap"]
    summary = grouped.agg(
        Runs="count",
        Mean="mean",
        Median="median",
        StdDev=lambda values: values.std(ddof=1),
        Q1=lambda values: values.quantile(0.25),
        Q3=lambda values: values.quantile(0.75),
    ).reset_index()
    summary.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stream trained-agent MasterOutput_MarketOverlap.zip files under an "
            "economy folder and plot, for each run, the mean maximum overlap "
            "between entry markets and markets already in the firm's portfolio."
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
        help="Directory for PNG and CSV outputs (default: economy_dir/entry_overlap_analysis).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per pandas streaming chunk (default: 1,000,000).",
    )
    parser.add_argument(
        "--empty-portfolio",
        choices=("skip", "zero"),
        default="skip",
        help=(
            "How to handle entries when the firm had no markets in its portfolio "
            "immediately before entry. Default: skip."
        ),
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

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else economy_dir / "entry_overlap_analysis"
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
        results.append(stream_zip(zip_path, args.chunksize, args.empty_portfolio))

    run_rows = [row for result in results for row in result.run_rows]
    if not run_rows:
        raise SystemExit("No entry-overlap observations were produced.")

    run_df = pd.DataFrame(run_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_csv = output_dir / "entry_overlap_with_portfolio_per_run.csv"
    summary_csv = output_dir / "entry_overlap_with_portfolio_summary.csv"
    plot_png = output_dir / "entry_overlap_with_portfolio_boxplot.png"
    run_df.to_csv(run_csv, index=False)
    write_summary(run_df, summary_csv)
    plot_boxplot(run_df, plot_png)

    entry_csv = None
    if args.write_entry_level:
        entry_rows = [row for result in results for row in result.entry_rows]
        entry_csv = output_dir / "entry_overlap_with_portfolio_entry_level.csv"
        pd.DataFrame(entry_rows).to_csv(entry_csv, index=False)

    total_runs = sum(result.run_count for result in results)
    total_wins = sum(result.win_count for result in results)
    total_losses = sum(result.loss_count for result in results)
    print(f"Processed trained-agent datasets: {len(results):,}")
    print(f"Processed runs: {total_runs:,}")
    print(f"AI wins: {total_wins:,}")
    print(f"AI losses: {total_losses:,}")
    print(f"Wrote per-run means: {run_csv}")
    print(f"Wrote summary: {summary_csv}")
    print(f"Wrote plot: {plot_png}")
    if entry_csv is not None:
        print(f"Wrote entry-level observations: {entry_csv}")
    if skipped_zip_paths:
        print("Skipped unreadable ZIPs:")
        for zip_path in skipped_zip_paths:
            print(f"  {zip_path}")


if __name__ == "__main__":
    main()
