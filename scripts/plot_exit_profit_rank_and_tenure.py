#!/usr/bin/env python3
# This code was AI-generated and still requires verification by a human researcher. Remove this comment when done.
"""Plot exit-market profit rank and market tenure before exit.

For each exit event, this script ranks the exited market by its previous-step
profit within the firm's current portfolio. It also measures how many micro
timesteps the firm held that market before exit. The tenure plot averages first
within each trained-agent dataset, then shows 95% CIs across those dataset-level
means.
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
    "Rev",
    "Fix Cost",
    "Var Cost",
    "Quantity",
    "In Market",
]
BANKRUPTCY_SENTINEL_CAPITAL = -1e-9
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
    profit_by_key: dict[tuple[int, int], list[float]]
    capital_by_firm_step: dict[tuple[int, int], float]
    agent_type_by_firm: dict[int, str]
    final_capital_by_firm: dict[int, float]

    @classmethod
    def create(cls, zip_path: Path, raw_sim: int) -> "SimulationAccumulator":
        return cls(
            zip_path=zip_path,
            raw_sim=raw_sim,
            steps_by_key={},
            presence_by_key={},
            profit_by_key={},
            capital_by_firm_step={},
            agent_type_by_firm={},
            final_capital_by_firm={},
        )


@dataclass
class ZipResult:
    """Per-ZIP exit analysis output."""

    zip_path: Path
    exit_rows: list[dict[str, object]]
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


def lighten_color(color: str, amount: float = 0.45) -> tuple[float, float, float]:
    """Blend a color toward white by ``amount``."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def color_for(agent_type: str, outcome: str) -> str | tuple[float, float, float]:
    """Return the shared six-color encoding used by the exit plots."""
    base_color = TYPE_TO_COLOR[agent_type]
    return base_color if outcome == "wins" else lighten_color(base_color)


def label_for(agent_type: str, outcome: str) -> str:
    """Return a legend label for an agent type and panel."""
    return f"{agent_type} ({outcome})"


def legend_handles() -> list[mpatches.Patch]:
    """Return the shared six-color legend handles."""
    return [
        mpatches.Patch(
            facecolor=color_for(agent_type, outcome),
            label=label_for(agent_type, outcome),
            alpha=0.9,
        )
        for agent_type in AGENT_TYPE_ORDER
        for outcome in OUTCOME_ORDER
    ]


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


def is_bankruptcy_capital(capital: float) -> bool:
    """Return True when capital equals the simulator's bankruptcy sentinel."""
    return np.isclose(capital, BANKRUPTCY_SENTINEL_CAPITAL, atol=1e-12, rtol=0.0)


def append_rows(accumulator: SimulationAccumulator, frame: pd.DataFrame) -> None:
    """Append a chunk belonging to one simulation into the accumulator."""
    frame = frame.copy()
    frame["Profit"] = frame["Rev"] - frame["Fix Cost"] - (
        frame["Var Cost"] * frame["Quantity"]
    )
    for (firm, market), group in frame.groupby(["Firm", "Market"], sort=False):
        key = (int(firm), int(market))
        accumulator.steps_by_key.setdefault(key, []).extend(group["Step"].astype(int).tolist())
        accumulator.presence_by_key.setdefault(key, []).extend(
            group["In Market"].astype(np.int8).tolist()
        )
        accumulator.profit_by_key.setdefault(key, []).extend(group["Profit"].astype(float).tolist())

    capital_rows = frame.loc[
        frame.groupby(["Firm", "Step"], sort=False)["Market"].idxmax(),
        ["Firm", "Step", "Capital"],
    ]
    for firm, step, capital in capital_rows.itertuples(index=False, name=None):
        accumulator.capital_by_firm_step[(int(firm), int(step))] = float(capital)

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


def ordinal_profit_rank(
    portfolio_markets: list[int],
    profit_by_market: dict[int, float],
    exited_market: int,
) -> int:
    """Rank markets from lowest to highest previous-step profit."""
    sorted_markets = sorted(
        portfolio_markets,
        key=lambda market: (profit_by_market[market], market),
    )
    return sorted_markets.index(exited_market) + 1


def finalize_simulation(
    accumulator: SimulationAccumulator,
    exit_rows: list[dict[str, object]],
) -> str:
    """Classify one run and collect exit events."""
    outcome = classify_outcome(accumulator)
    firms = sorted({firm for firm, _ in accumulator.presence_by_key})
    markets = sorted({market for _, market in accumulator.presence_by_key})
    if not firms or not markets:
        return outcome

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
        firm_profit = np.zeros((len(markets), len(step_union)), dtype=float)

        for market in markets:
            key = (firm, market)
            steps = accumulator.steps_by_key.get(key)
            presence = accumulator.presence_by_key.get(key)
            profit = accumulator.profit_by_key.get(key)
            if not steps or not presence or not profit:
                continue
            positions = [step_position[step] for step in steps]
            market_index = market_index_by_id[market]
            firm_presence[market_index, positions] = presence
            firm_profit[market_index, positions] = profit

        agent_type = accumulator.agent_type_by_firm.get(firm, str(firm))
        for exit_market in markets:
            market_index = market_index_by_id[exit_market]
            series = firm_presence[market_index]
            previous_presence = np.concatenate(([0], series[:-1]))
            exit_positions = np.flatnonzero((series == 0) & (previous_presence == 1))
            entry_positions = np.flatnonzero((series == 1) & (previous_presence == 0))
            entry_pointer = 0
            current_entry_step: int | None = None

            for exit_position in exit_positions:
                while (
                    entry_pointer < len(entry_positions)
                    and entry_positions[entry_pointer] < exit_position
                ):
                    current_entry_step = int(step_union[entry_positions[entry_pointer]])
                    entry_pointer += 1

                if current_entry_step is None or exit_position == 0:
                    continue

                exit_step = int(step_union[exit_position])
                exit_capital = accumulator.capital_by_firm_step.get((firm, exit_step))
                if exit_capital is not None and is_bankruptcy_capital(exit_capital):
                    continue

                previous_position = exit_position - 1
                portfolio_indices = np.flatnonzero(firm_presence[:, previous_position] == 1)
                portfolio_markets = [markets[index] for index in portfolio_indices]
                if exit_market not in portfolio_markets:
                    continue

                profit_by_market = {
                    market: float(firm_profit[market_index_by_id[market], previous_position])
                    for market in portfolio_markets
                }
                rank = ordinal_profit_rank(portfolio_markets, profit_by_market, exit_market)
                exit_rows.append(
                    {
                        "zip_path": str(accumulator.zip_path),
                        "trained_agent": accumulator.zip_path.parent.name,
                        "Sim": accumulator.raw_sim,
                        "Outcome": outcome,
                        "Agent Type": agent_type,
                        "Firm": firm,
                        "Exit Step": exit_step,
                        "Entry Step": current_entry_step,
                        "Market": exit_market,
                        "Tenure Before Exit": exit_step - current_entry_step,
                        "Portfolio Size Before Exit": len(portfolio_markets),
                        "Exited Market Profit Previous Step": profit_by_market[exit_market],
                        "Exited Market Profit Rank": rank,
                    }
                )

    return outcome


def stream_zip(zip_path: Path, chunksize: int) -> ZipResult:
    """Stream one ZIP archive and return exit-event rows."""
    master_member = get_master_output_member(zip_path)
    current: SimulationAccumulator | None = None
    processed_runs = 0
    outcome_run_counts: Counter[str] = Counter()
    exit_rows: list[dict[str, object]] = []

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
                        outcome = finalize_simulation(current, exit_rows)
                        outcome_run_counts[outcome] += 1
                        processed_runs += 1
                        if processed_runs % 100 == 0:
                            log(f"{zip_path.parent.name}: processed {processed_runs} runs")
                        current = SimulationAccumulator.create(zip_path, sim_id)

                    append_rows(current, sim_frame)

    if current is not None:
        outcome = finalize_simulation(current, exit_rows)
        outcome_run_counts[outcome] += 1
        processed_runs += 1

    log(f"Finished {zip_path.parent.name}: {processed_runs:,} runs")
    return ZipResult(
        zip_path=zip_path,
        exit_rows=exit_rows,
        run_count=processed_runs,
        outcome_run_counts=outcome_run_counts,
    )


def build_profit_rank_distribution(exit_df: pd.DataFrame, max_rank: int) -> pd.DataFrame:
    """Build share of exits by profit rank for each agent/outcome group."""
    rows: list[dict[str, object]] = []
    for agent_type in AGENT_TYPE_ORDER:
        for outcome in OUTCOME_ORDER:
            subset = exit_df[
                (exit_df["Agent Type"] == agent_type)
                & (exit_df["Outcome"] == outcome)
            ]
            total = len(subset)
            counts = subset["Exited Market Profit Rank"].value_counts().to_dict()
            for rank in range(1, max_rank + 1):
                count = int(counts.get(rank, 0))
                rows.append(
                    {
                        "Agent Type": agent_type,
                        "Outcome": outcome,
                        "Profit Rank": rank,
                        "Exit Count": count,
                        "Share of Exits": count / total if total else 0.0,
                    }
                )

    return pd.DataFrame(rows)


def build_tenure_trained_agent_means(exit_df: pd.DataFrame) -> pd.DataFrame:
    """Average tenure within each trained-agent dataset."""
    grouped = exit_df.groupby(["zip_path", "trained_agent", "Agent Type", "Outcome"])[
        "Tenure Before Exit"
    ]
    return grouped.agg(
        Exit_Count="count",
        Mean_Tenure_Before_Exit="mean",
    ).reset_index()


def confidence_interval_95(values: pd.Series) -> float:
    """Return the 95% confidence interval half-width."""
    clean_values = values.dropna()
    count = clean_values.count()
    if count <= 1:
        return 0.0
    return float(1.96 * clean_values.std(ddof=1) / np.sqrt(count))


def build_tenure_summary(trained_agent_means: pd.DataFrame) -> pd.DataFrame:
    """Summarize trained-agent tenure means into plot means and CIs."""
    grouped = trained_agent_means.groupby(["Agent Type", "Outcome"])[
        "Mean_Tenure_Before_Exit"
    ]
    summary = grouped.agg(Mean="mean", N="count").reset_index()
    summary["CI95"] = grouped.apply(confidence_interval_95).to_numpy()
    return summary


def plot_profit_rank_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    """Plot share of exits by exited-market profit rank."""
    max_rank = int(distribution["Profit Rank"].max())
    x = np.arange(1, max_rank + 1)
    groups = [
        (agent_type, outcome)
        for agent_type in AGENT_TYPE_ORDER
        for outcome in OUTCOME_ORDER
    ]
    width = 0.12
    offsets = (np.arange(len(groups)) - (len(groups) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(9, 6))
    for offset, (agent_type, outcome) in zip(offsets, groups):
        subset = distribution[
            (distribution["Agent Type"] == agent_type)
            & (distribution["Outcome"] == outcome)
        ].sort_values("Profit Rank")
        ax.bar(
            x + offset,
            subset["Share of Exits"].to_numpy(),
            width=width,
            color=color_for(agent_type, outcome),
            label=label_for(agent_type, outcome),
            alpha=0.9,
        )

    tick_labels = [str(rank) for rank in x]
    tick_labels[0] = "1\n(lowest)"
    tick_labels[-1] = f"{max_rank}\n(highest)"
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_xlabel("Profit rank of exited market within portfolio at exit", fontsize=13)
    ax.set_ylabel("Share of exits", fontsize=13)
    ax.set_ylim(0, min(1.0, max(0.2, distribution["Share of Exits"].max() * 1.15)))
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=legend_handles(),
        frameon=False,
        fontsize=10,
        ncol=2,
        loc="upper right",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_tenure_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Plot mean tenure before exit with 95% CI whiskers."""
    x = np.arange(len(AGENT_TYPE_ORDER))
    width = 0.34
    offsets = {
        "wins": -width / 2,
        "losses": width / 2,
    }

    fig, ax = plt.subplots(figsize=(7.5, 6))
    max_height = 0.0
    for agent_index, agent_type in enumerate(AGENT_TYPE_ORDER):
        for outcome in OUTCOME_ORDER:
            row = summary[
                (summary["Agent Type"] == agent_type)
                & (summary["Outcome"] == outcome)
            ]
            if row.empty:
                mean = 0.0
                ci = 0.0
            else:
                mean = float(row["Mean"].iloc[0])
                ci = float(row["CI95"].iloc[0])
            max_height = max(max_height, mean + ci)
            ax.bar(
                x[agent_index] + offsets[outcome],
                mean,
                width=width,
                color=color_for(agent_type, outcome),
                alpha=0.9,
                label=label_for(agent_type, outcome),
            )
            ax.errorbar(
                x[agent_index] + offsets[outcome],
                mean,
                yerr=ci,
                fmt="none",
                ecolor="#333333",
                elinewidth=1.4,
                capsize=4,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(AGENT_TYPE_ORDER, fontsize=12)
    ax.set_ylabel("Mean tenure in market before exit\n(micro time steps)", fontsize=13)
    ax.set_ylim(0, max(10.0, max_height * 1.18))
    ax.grid(axis="both", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=legend_handles(),
        frameon=False,
        fontsize=10,
        ncol=2,
        loc="upper right",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Stream trained-agent MasterOutput_MarketOverlap.zip files under an "
            "economy folder and plot exit profit rank plus market tenure before exit."
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
        help="Directory for PNG and CSV outputs (default: economy_dir/exit_analysis).",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1_000_000,
        help="Rows per pandas streaming chunk (default: 1,000,000).",
    )
    parser.add_argument(
        "--max-rank",
        type=int,
        default=10,
        help="Maximum profit rank to show in the rank-distribution plot (default: 10).",
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
    if args.max_rank <= 0:
        raise SystemExit("--max-rank must be positive")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else economy_dir / "exit_analysis"
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
        results.append(stream_zip(zip_path, args.chunksize))

    exit_rows = [row for result in results for row in result.exit_rows]
    if not exit_rows:
        raise SystemExit("No exit events were produced.")

    exit_df = pd.DataFrame(exit_rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    exit_events_csv = output_dir / "exit_events_with_profit_rank_and_tenure.csv"
    rank_distribution_csv = output_dir / "exit_profit_rank_distribution.csv"
    tenure_agent_means_csv = output_dir / "exit_tenure_trained_agent_means.csv"
    tenure_summary_csv = output_dir / "exit_tenure_95ci.csv"
    rank_plot_png = output_dir / "exit_profit_rank_distribution.png"
    tenure_plot_png = output_dir / "exit_tenure_by_panel.png"

    exit_df.to_csv(exit_events_csv, index=False)
    rank_distribution = build_profit_rank_distribution(exit_df, args.max_rank)
    rank_distribution.to_csv(rank_distribution_csv, index=False)
    tenure_agent_means = build_tenure_trained_agent_means(exit_df)
    tenure_agent_means.to_csv(tenure_agent_means_csv, index=False)
    tenure_summary = build_tenure_summary(tenure_agent_means)
    tenure_summary.to_csv(tenure_summary_csv, index=False)

    plot_profit_rank_distribution(rank_distribution, rank_plot_png)
    plot_tenure_summary(tenure_summary, tenure_plot_png)

    total_runs = sum(result.run_count for result in results)
    total_wins = sum(result.outcome_run_counts["wins"] for result in results)
    total_losses = sum(result.outcome_run_counts["losses"] for result in results)
    print(f"Processed trained-agent datasets: {len(results):,}")
    print(f"Processed runs: {total_runs:,}")
    print(f"AI wins: {total_wins:,}")
    print(f"AI losses: {total_losses:,}")
    print(f"Wrote exit-event audit CSV: {exit_events_csv}")
    print(f"Wrote rank distribution: {rank_distribution_csv}")
    print(f"Wrote tenure trained-agent means: {tenure_agent_means_csv}")
    print(f"Wrote tenure 95% CI summary: {tenure_summary_csv}")
    print(f"Wrote profit-rank plot: {rank_plot_png}")
    print(f"Wrote tenure plot: {tenure_plot_png}")
    if skipped_zip_paths:
        print("Skipped unreadable ZIPs:")
        for zip_path in skipped_zip_paths:
            print(f"  {zip_path}")


if __name__ == "__main__":
    main()
