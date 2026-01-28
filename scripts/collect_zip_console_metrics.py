#!/usr/bin/env python3
"""Collect console summary metrics from business_strategy_plots across ZIPs."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path


MASTER_OUTPUT_FILENAMES = ("MasterOutput.csv",)
MARKET_OVERLAP_FILENAMES = ("MarketOverlap.csv",)


SUMMARY_PATTERN = re.compile(
    r"^(?P<agent>.+?): (?P<growth>[-\d\.]+)% \(avg bankruptcy rate: (?P<bankruptcy>[-\d\.]+)%\)$"
)


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


def normalize_agent_type(agent_type: str) -> str | None:
    """Normalize agent type labels down to AI/Naive/Sophisticated."""
    agent_type = agent_type.strip()
    if agent_type.startswith("AI"):
        return "AI"
    if agent_type.startswith("Naive"):
        return "Naive"
    if agent_type.startswith("Sophisticated"):
        return "Sophisticated"
    return None


def parse_summary_output(output: str) -> dict[str, tuple[str, str]]:
    """Parse growth and bankruptcy rates from console output."""
    results: dict[str, tuple[str, str]] = {}
    for line in output.splitlines():
        match = SUMMARY_PATTERN.match(line.strip())
        if not match:
            continue
        normalized = normalize_agent_type(match.group("agent"))
        if not normalized:
            continue
        results[normalized] = (match.group("growth"), match.group("bankruptcy"))
    return results


def run_plots(zip_path: Path, master_output: str, market_overlap: str, various_flag: bool) -> str:
    """Run business_strategy_plots and return its stdout."""
    cmd = [
        sys.executable,
        str(Path("Analysis") / "business_strategy_plots.py"),
        "--zip-path",
        str(zip_path),
        "--master-output-file-name",
        master_output,
        "--market-overlap-file-name",
        market_overlap,
    ]
    if various_flag:
        cmd.append("--various-sophisticated-agent-types")

    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"

    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        raise RuntimeError(
            "\n".join(
                [
                    f"business_strategy_plots.py failed for {zip_path}",
                    result.stdout.strip(),
                    result.stderr.strip(),
                ]
            ).strip()
        )
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively search for ZIP archives and collect console summary metrics."
        )
    )
    parser.add_argument(
        "search_dir",
        type=Path,
        help="Directory to recursively search for ZIP archives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("zip_console_metrics.csv"),
        help="Destination CSV path for summary metrics.",
    )
    args = parser.parse_args()

    zip_paths = sorted(args.search_dir.rglob("*.zip"))
    if not zip_paths:
        raise SystemExit(f"No ZIP files found under {args.search_dir}")

    rows = []
    for zip_path in zip_paths:
        try:
            master_output = find_member(zip_path, MASTER_OUTPUT_FILENAMES)
            market_overlap = find_member(zip_path, MARKET_OVERLAP_FILENAMES)
        except ValueError as exc:
            print(f"Skipping {zip_path}: {exc}", file=sys.stderr)
            continue

        various_flag = zip_path.parent.name == "VariousSophisticatedAgents"
        try:
            output = run_plots(zip_path, master_output, market_overlap, various_flag)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            continue

        metrics = parse_summary_output(output)
        rows.append(
            {
                "zip_path": str(zip_path),
                "ai_growth_rate": metrics.get("AI", ("", ""))[0],
                "ai_bankruptcy_rate": metrics.get("AI", ("", ""))[1],
                "naive_growth_rate": metrics.get("Naive", ("", ""))[0],
                "naive_bankruptcy_rate": metrics.get("Naive", ("", ""))[1],
                "sophisticated_growth_rate": metrics.get("Sophisticated", ("", ""))[0],
                "sophisticated_bankruptcy_rate": metrics.get("Sophisticated", ("", ""))[1],
            }
        )

    fieldnames = [
        "zip_path",
        "ai_growth_rate",
        "ai_bankruptcy_rate",
        "naive_growth_rate",
        "naive_bankruptcy_rate",
        "sophisticated_growth_rate",
        "sophisticated_bankruptcy_rate",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
