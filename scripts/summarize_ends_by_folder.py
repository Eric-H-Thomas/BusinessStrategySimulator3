#!/usr/bin/env python3
"""Summarize ending outcomes from MasterOutputEndings.csv files by folder."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

ENDS_FILENAME = "MasterOutputEndings.csv"
PER_FILE_SUMMARY_NAME = "endings_per_file_summary.csv"
FOLDER_SUMMARY_NAME = "endings_folder_summary_mean_std.csv"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def stddev_sample(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    variance = sum((value - mu) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def read_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        return list(reader), fieldnames


def normalize_agent_type(agent_type: str, collapse_by_agent_type: bool) -> str:
    if not collapse_by_agent_type:
        return agent_type
    return "".join(char for char in agent_type if not char.isdigit())


def summarize_endings_file(
    path: Path,
    starting_capital: float,
    bankruptcy_value: float,
    bankruptcy_tol: float,
    collapse_by_agent_type: bool,
) -> list[dict[str, object]]:
    rows, fieldnames = read_csv_rows(path)
    required = {"Sim", "Step", "Firm", "Agent Type", "Capital"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    # Collapse duplicate rows so each sim/step/firm/agent contributes once.
    dedup_buckets: dict[tuple[str, float, str, str], list[float]] = {}
    for row in rows:
        key = (
            row["Sim"],
            float(row["Step"]),
            row["Firm"],
            normalize_agent_type(row["Agent Type"], collapse_by_agent_type),
        )
        dedup_buckets.setdefault(key, []).append(float(row["Capital"]))

    dedup_rows: list[tuple[str, float, str, str, float]] = [
        (sim, step, firm, agent_type, mean(capitals))
        for (sim, step, firm, agent_type), capitals in dedup_buckets.items()
    ]

    max_step_by_firm: dict[tuple[str, str, str], float] = {}
    for sim, step, firm, agent_type, _ in dedup_rows:
        key = (sim, firm, agent_type)
        max_step_by_firm[key] = step if key not in max_step_by_firm else max(max_step_by_firm[key], step)

    end_capital_by_firm: dict[tuple[str, str, str], float] = {}
    bankruptcy_flags_by_agent: dict[str, list[float]] = {}

    for sim, step, firm, agent_type, capital in dedup_rows:
        firm_key = (sim, firm, agent_type)

        if step == max_step_by_firm[firm_key]:
            end_capital_by_firm[firm_key] = capital
            is_bankrupt = abs(capital - bankruptcy_value) <= bankruptcy_tol
            bankruptcy_flags_by_agent.setdefault(agent_type, []).append(1.0 if is_bankrupt else 0.0)

    growth_rates_by_agent: dict[str, list[float]] = {}
    for firm_key, end_capital in end_capital_by_firm.items():
        _, _, agent_type = firm_key
        if starting_capital == 0.0:
            continue
        growth = (end_capital - starting_capital) / starting_capital
        growth_rates_by_agent.setdefault(agent_type, []).append(growth)

    all_agent_types = sorted(set(bankruptcy_flags_by_agent) | set(growth_rates_by_agent))
    summary_rows: list[dict[str, object]] = []
    for agent_type in all_agent_types:
        bankruptcy_values = bankruptcy_flags_by_agent.get(agent_type, [])
        growth_values = growth_rates_by_agent.get(agent_type, [])
        summary_rows.append(
            {
                "endings_file": str(path),
                "Agent Type": agent_type,
                "bankruptcy_rate": mean(bankruptcy_values),
                "average_capital_growth_rate": mean(growth_values),
                "bankruptcy_sample_count": len(bankruptcy_values),
                "growth_sample_count": len(growth_values),
            }
        )

    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Given top-level folder A, process each immediate child folder B by "
            "finding MasterOutputEndings.csv files recursively, writing per-file "
            "agent summaries, and writing folder-level mean/std summaries across files."
        )
    )
    parser.add_argument("folder_a", type=Path, help="Top-level folder A.")
    parser.add_argument(
        "starting_capital",
        type=float,
        help="Starting capital at the beginning of the simulation.",
    )
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
    parser.add_argument(
        "--collapse-by-agent-type",
        action="store_true",
        help="Strip digits from agent types before aggregating (e.g., 0S/1S -> S).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_a: Path = args.folder_a

    if not folder_a.exists() or not folder_a.is_dir():
        raise SystemExit(f"Not a directory: {folder_a}")

    child_folders = sorted(path for path in folder_a.iterdir() if path.is_dir())
    if not child_folders:
        print(f"No child folders found under {folder_a}")
        return

    for folder_b in child_folders:
        endings_paths = sorted(folder_b.rglob(ENDS_FILENAME))
        if not endings_paths:
            print(f"No {ENDS_FILENAME} files in {folder_b}; skipping.")
            continue

        per_file_rows: list[dict[str, object]] = []
        for endings_path in endings_paths:
            per_file_rows.extend(
                summarize_endings_file(
                    endings_path,
                    starting_capital=args.starting_capital,
                    bankruptcy_value=args.bankruptcy_value,
                    bankruptcy_tol=args.bankruptcy_tol,
                    collapse_by_agent_type=args.collapse_by_agent_type,
                )
            )

        per_file_path = folder_b / PER_FILE_SUMMARY_NAME
        write_csv(
            per_file_path,
            per_file_rows,
            [
                "endings_file",
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

        folder_rows: list[dict[str, object]] = []
        for agent_type in sorted(grouped):
            bankruptcy_values = grouped[agent_type]["bankruptcy"]
            growth_values = grouped[agent_type]["growth"]
            folder_rows.append(
                {
                    "folder_b": str(folder_b),
                    "Agent Type": agent_type,
                    "files_count": len(bankruptcy_values),
                    "bankruptcy_rate_mean": mean(bankruptcy_values),
                    "bankruptcy_rate_std": stddev_sample(bankruptcy_values),
                    "average_capital_growth_rate_mean": mean(growth_values),
                    "average_capital_growth_rate_std": stddev_sample(growth_values),
                }
            )

        folder_summary_path = folder_b / FOLDER_SUMMARY_NAME
        write_csv(
            folder_summary_path,
            folder_rows,
            [
                "folder_b",
                "Agent Type",
                "files_count",
                "bankruptcy_rate_mean",
                "bankruptcy_rate_std",
                "average_capital_growth_rate_mean",
                "average_capital_growth_rate_std",
            ],
        )

        print(f"Wrote: {per_file_path}")
        print(f"Wrote: {folder_summary_path}")


if __name__ == "__main__":
    main()
