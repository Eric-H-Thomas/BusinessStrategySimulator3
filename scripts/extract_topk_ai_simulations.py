#!/usr/bin/env python3
"""Extract top-k AI simulations by final-step capital from a ZIP archive."""

from __future__ import annotations

import argparse
import csv
import io
import zipfile
from pathlib import Path

MASTER_OUTPUT_FILE_NAME = "MasterOutput.csv"
MARKET_OVERLAP_FILE_NAME = "MarketOverlap.csv"
TOPK_MASTER_OUTPUT_FILE_NAME = "TopKMasterOutput.csv"
OUTPUT_ZIP_FILE_NAME = "TopK_MasterOutput_MarketOverlap.zip"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Read a ZIP containing MasterOutput.csv and MarketOverlap.csv, find the "
            "top-k simulations where AI did best at the final time step, and write "
            "TopKMasterOutput.csv zipped together with MarketOverlap.csv."
        )
    )
    parser.add_argument(
        "zip_path",
        type=Path,
        help="Path to an input ZIP containing MasterOutput.csv and MarketOverlap.csv.",
    )
    parser.add_argument(
        "k",
        type=int,
        help="Number of top simulations to retain.",
    )
    parser.add_argument(
        "--ai-agent-label",
        default="AI",
        help=(
            "Substring used to identify AI rows in the 'Agent Type' column "
            "(case-insensitive, default: 'AI')."
        ),
    )
    return parser.parse_args()


def parse_master_rows(master_csv_bytes: bytes) -> tuple[list[dict[str, str]], list[str]]:
    """Parse CSV bytes into rows and return rows + field names."""
    text_stream = io.StringIO(master_csv_bytes.decode("utf-8-sig"))
    reader = csv.DictReader(text_stream)
    if reader.fieldnames is None:
        raise ValueError("MasterOutput.csv does not contain a header row.")

    required_columns = {"Sim", "Step", "Capital", "Agent Type"}
    missing = required_columns - set(reader.fieldnames)
    if missing:
        missing_string = ", ".join(sorted(missing))
        raise ValueError(f"Master output is missing required column(s): {missing_string}")

    return list(reader), reader.fieldnames


def load_required_inputs(zip_path: Path) -> tuple[list[dict[str, str]], list[str], bytes]:
    """Load MasterOutput.csv rows and MarketOverlap.csv bytes from ZIP."""
    with zipfile.ZipFile(zip_path, "r") as archive:
        namelist = set(archive.namelist())
        if MASTER_OUTPUT_FILE_NAME not in namelist:
            raise ValueError(f"{MASTER_OUTPUT_FILE_NAME} not found in ZIP: {zip_path}")
        if MARKET_OVERLAP_FILE_NAME not in namelist:
            raise ValueError(f"{MARKET_OVERLAP_FILE_NAME} not found in ZIP: {zip_path}")

        master_csv_bytes = archive.read(MASTER_OUTPUT_FILE_NAME)
        rows, fieldnames = parse_master_rows(master_csv_bytes)
        market_overlap_bytes = archive.read(MARKET_OVERLAP_FILE_NAME)

    return rows, fieldnames, market_overlap_bytes


def _to_float(value: str, column_name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse {column_name} value as number: {value!r}") from exc


def select_top_k_simulations(rows: list[dict[str, str]], k: int, ai_agent_label: str) -> list[str]:
    """Return simulation IDs for the top-k AI final-capital simulations."""
    max_step_by_sim: dict[str, float] = {}
    for row in rows:
        sim_id = row["Sim"]
        step = _to_float(row["Step"], "Step")
        current_max = max_step_by_sim.get(sim_id)
        if current_max is None or step > current_max:
            max_step_by_sim[sim_id] = step

    final_ai_capitals: dict[str, list[float]] = {}
    lowered_label = ai_agent_label.lower()
    for row in rows:
        sim_id = row["Sim"]
        step = _to_float(row["Step"], "Step")
        if step != max_step_by_sim[sim_id]:
            continue
        agent_type = row["Agent Type"]
        if lowered_label not in agent_type.lower():
            continue

        capital = _to_float(row["Capital"], "Capital")
        final_ai_capitals.setdefault(sim_id, []).append(capital)

    if not final_ai_capitals:
        raise ValueError(
            "No AI rows found at final step. "
            "Check --ai-agent-label or verify the 'Agent Type' values."
        )

    scored = []
    for sim_id, capital_values in final_ai_capitals.items():
        score = sum(capital_values) / len(capital_values)
        scored.append((sim_id, score))

    scored.sort(key=lambda item: item[1], reverse=True)
    return [sim_id for sim_id, _ in scored[:k]]


def build_topk_rows(rows: list[dict[str, str]], top_sim_ids: list[str]) -> list[dict[str, str]]:
    """Return rows belonging to top simulation IDs."""
    top_ids = set(top_sim_ids)
    return [row for row in rows if row["Sim"] in top_ids]


def write_outputs(
    output_dir: Path,
    fieldnames: list[str],
    topk_rows: list[dict[str, str]],
    market_overlap_bytes: bytes,
) -> tuple[Path, Path]:
    """Write TopKMasterOutput.csv and TopK_MasterOutput_MarketOverlap.zip."""
    topk_csv_path = output_dir / TOPK_MASTER_OUTPUT_FILE_NAME
    output_zip_path = output_dir / OUTPUT_ZIP_FILE_NAME

    with topk_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(topk_rows)

    with zipfile.ZipFile(output_zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(topk_csv_path, arcname=TOPK_MASTER_OUTPUT_FILE_NAME)
        archive.writestr(MARKET_OVERLAP_FILE_NAME, market_overlap_bytes)

    return topk_csv_path, output_zip_path


def main() -> None:
    """Execute top-k extraction workflow."""
    args = parse_args()

    if args.k <= 0:
        raise ValueError("k must be a positive integer.")
    if not args.zip_path.exists():
        raise FileNotFoundError(f"Input ZIP not found: {args.zip_path}")

    rows, fieldnames, market_overlap_bytes = load_required_inputs(args.zip_path)
    top_sim_ids = select_top_k_simulations(rows, args.k, args.ai_agent_label)
    topk_rows = build_topk_rows(rows, top_sim_ids)

    topk_csv_path, output_zip_path = write_outputs(
        args.zip_path.parent,
        fieldnames,
        topk_rows,
        market_overlap_bytes,
    )

    print(f"Selected simulations: {top_sim_ids}")
    print(f"Wrote top-k CSV: {topk_csv_path}")
    print(f"Wrote ZIP bundle: {output_zip_path}")


if __name__ == "__main__":
    main()
