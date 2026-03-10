#!/usr/bin/env python3
"""Summarize bankruptcy and capital growth from MasterOutputEnds.csv files."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ENDS_FILENAME = "MasterOutputEnds.csv"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev_sample(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((v - mu) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        return list(reader), fieldnames


def summarize_ends_file(
    path: Path,
    bankruptcy_value: float,
    bankruptcy_tol: float,
) -> list[dict[str, object]]:
    rows, fieldnames = read_csv_rows(path)
    required = {"Sim", "Step", "Agent Type", "Capital"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    # Identify rows at min/max step for each simulation.
    min_step_by_sim: dict[str, float] = {}
    max_step_by_sim: dict[str, float] = {}
    for row in rows:
        sim = row["Sim"]
        step = float(row["Step"])
        min_step_by_sim[sim] = step if sim not in min_step_by_sim else min(min_step_by_sim[sim], step)
        max_step_by_sim[sim] = step if sim not in max_step_by_sim else max(max_step_by_sim[sim], step)

    # Group to simulation+agent to derive start/end capital for growth calculations.
    sim_agent_starts: dict[tuple[str, str], list[float]] = {}
    sim_agent_ends: dict[tuple[str, str], list[float]] = {}

    # Track final-row bankruptcy flags at row granularity for each agent type.
    final_bankruptcy_flags_by_agent: dict[str, list[float]] = {}

    for row in rows:
        sim = row["Sim"]
        agent_type = row["Agent Type"]
        step = float(row["Step"])
        capital = float(row["Capital"])

        if step == min_step_by_sim[sim]:
            sim_agent_starts.setdefault((sim, agent_type), []).append(capital)

        if step == max_step_by_sim[sim]:
            sim_agent_ends.setdefault((sim, agent_type), []).append(capital)
            is_bankrupt = abs(capital - bankruptcy_value) <= bankruptcy_tol
            final_bankruptcy_flags_by_agent.setdefault(agent_type, []).append(1.0 if is_bankrupt else 0.0)

    growth_rates_by_agent: dict[str, list[float]] = {}
    for key, end_vals in sim_agent_ends.items():
        sim, agent_type = key
        start_vals = sim_agent_starts.get((sim, agent_type), [])
        if not start_vals:
            continue

        start_capital = mean(start_vals)
        end_capital = mean(end_vals)
        if start_capital == 0.0:
            continue
        growth = (end_capital - start_capital) / start_capital
        growth_rates_by_agent.setdefault(agent_type, []).append(growth)

    all_agent_types = sorted(set(final_bankruptcy_flags_by_agent) | set(growth_rates_by_agent))
    output_rows: list[dict[str, object]] = []
    for agent_type in all_agent_types:
        bankruptcy_flags = final_bankruptcy_flags_by_agent.get(agent_type, [])
        growth_rates = growth_rates_by_agent.get(agent_type, [])
        output_rows.append(
            {
                "ends_file": str(path),
                "Agent Type": agent_type,
                "bankruptcy_rate": mean(bankruptcy_flags),
                "average_capital_growth_rate": mean(growth_rates),
                "bankruptcy_sample_count": len(bankruptcy_flags),
                "growth_sample_count": len(growth_rates),
            }
        )

    return output_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each immediate child folder of a top-level directory, recursively "
            "summarize MasterOutputEnds.csv files by agent type and aggregate "
            "folder-level and top-level statistics."
        )
    )
    parser.add_argument("folder_a", type=Path, help="Top-level folder A.")
    parser.add_argument(
        "--bankruptcy-value",
        type=float,
        default=-1e-09,
        help="Capital value used to mark bankruptcy (default: -1e-09).",
    )
    parser.add_argument(
        "--bankruptcy-tol",
        type=float,
        default=1e-12,
        help="Absolute tolerance for bankruptcy-value comparisons.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_a = args.folder_a

    if not folder_a.exists() or not folder_a.is_dir():
        raise SystemExit(f"Not a directory: {folder_a}")

    child_folders = sorted(p for p in folder_a.iterdir() if p.is_dir())
    if not child_folders:
        print(f"No child folders found under {folder_a}")
        return

    all_folder_means: list[dict[str, object]] = []

    for folder_b in child_folders:
        ends_paths = sorted(folder_b.rglob(ENDS_FILENAME))
        if not ends_paths:
            print(f"No {ENDS_FILENAME} files in {folder_b}; skipping.")
            continue

        per_file_rows: list[dict[str, object]] = []
        for ends_path in ends_paths:
            per_file_rows.extend(
                summarize_ends_file(
                    ends_path,
                    bankruptcy_value=args.bankruptcy_value,
                    bankruptcy_tol=args.bankruptcy_tol,
                )
            )

        per_file_out = folder_b / "ends_per_file_summary.csv"
        write_csv(
            per_file_out,
            per_file_rows,
            [
                "ends_file",
                "Agent Type",
                "bankruptcy_rate",
                "average_capital_growth_rate",
                "bankruptcy_sample_count",
                "growth_sample_count",
            ],
        )

        grouped: dict[str, dict[str, list[float]]] = {}
        for row in per_file_rows:
            agent_type = str(row["Agent Type"])
            grouped.setdefault(agent_type, {"bankruptcy": [], "growth": []})
            grouped[agent_type]["bankruptcy"].append(float(row["bankruptcy_rate"]))
            grouped[agent_type]["growth"].append(float(row["average_capital_growth_rate"]))

        folder_mean_rows: list[dict[str, object]] = []
        for agent_type in sorted(grouped):
            bankruptcy_values = grouped[agent_type]["bankruptcy"]
            growth_values = grouped[agent_type]["growth"]
            row = {
                "folder_b": str(folder_b),
                "Agent Type": agent_type,
                "bankruptcy_rate_mean": mean(bankruptcy_values),
                "average_capital_growth_rate_mean": mean(growth_values),
            }
            folder_mean_rows.append(row)
            all_folder_means.append(row)

        folder_mean_out = folder_b / "ends_folder_means.csv"
        write_csv(
            folder_mean_out,
            folder_mean_rows,
            [
                "folder_b",
                "Agent Type",
                "bankruptcy_rate_mean",
                "average_capital_growth_rate_mean",
            ],
        )

        print(f"Wrote: {per_file_out}")
        print(f"Wrote: {folder_mean_out}")

    if not all_folder_means:
        print("No folders produced summaries; nothing to aggregate at top level.")
        return

    grouped_means: dict[str, dict[str, list[float]]] = {}
    for row in all_folder_means:
        agent_type = str(row["Agent Type"])
        grouped_means.setdefault(agent_type, {"bankruptcy": [], "growth": []})
        grouped_means[agent_type]["bankruptcy"].append(float(row["bankruptcy_rate_mean"]))
        grouped_means[agent_type]["growth"].append(float(row["average_capital_growth_rate_mean"]))

    top_rows: list[dict[str, object]] = []
    for agent_type in sorted(grouped_means):
        bankruptcy_values = grouped_means[agent_type]["bankruptcy"]
        growth_values = grouped_means[agent_type]["growth"]
        top_rows.append(
            {
                "Agent Type": agent_type,
                "bankruptcy_rate_mean_of_means": mean(bankruptcy_values),
                "bankruptcy_rate_std_of_means": stddev_sample(bankruptcy_values),
                "capital_growth_rate_mean_of_means": mean(growth_values),
                "capital_growth_rate_std_of_means": stddev_sample(growth_values),
            }
        )

    top_out = folder_a / "ends_top_level_mean_std.csv"
    write_csv(
        top_out,
        top_rows,
        [
            "Agent Type",
            "bankruptcy_rate_mean_of_means",
            "bankruptcy_rate_std_of_means",
            "capital_growth_rate_mean_of_means",
            "capital_growth_rate_std_of_means",
        ],
    )
    print(f"Wrote: {top_out}")


if __name__ == "__main__":
    main()
