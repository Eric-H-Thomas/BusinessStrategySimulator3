#!/usr/bin/env python3
"""Aggregate A2C hyperparameter sweep results into a single CSV summary."""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON file: {path}\n{exc}") from exc


def gather_run_rows(run_dir: Path) -> Optional[Dict[str, Any]]:
    hyper_path = run_dir / "hyperparameters.json"
    eval_path = run_dir / "evaluation_metrics.json"

    if not hyper_path.exists():
        return None

    row: Dict[str, Any] = {"run": run_dir.name}
    hyperparameters = load_json(hyper_path)
    row.update({str(key): hyperparameters[key] for key in sorted(hyperparameters)})

    if eval_path.exists():
        evaluation = load_json(eval_path)
        for key, value in evaluation.items():
            if isinstance(value, (int, float, str)):
                row[f"eval_{key}"] = value
    else:
        row["status"] = "missing_evaluation"

    return row


def _normalise_value(value: Any) -> Any:
    """Tweak numeric values so spreadsheet tools treat them as numbers."""

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"

        return round(value, 12)

    return value


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        output_path.write_text("")
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            normalised_row = {key: _normalise_value(value) for key, value in row.items()}
            writer.writerow(normalised_row)


def cleanup_slurm_jobs(slurm_jobs_dir: Path) -> None:
    """Remove generated SLURM job scripts so future sweeps start clean."""

    if not slurm_jobs_dir.exists():
        print(f"No SLURM job directory found at {slurm_jobs_dir}; skipping cleanup.")
        return

    removed_entries = 0
    for entry in slurm_jobs_dir.iterdir():
        try:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
            removed_entries += 1
        except OSError as exc:
            print(f"Warning: failed to remove {entry}: {exc}")

    try:
        slurm_jobs_dir.rmdir()
    except OSError:
        pass

    if removed_entries:
        print(f"Cleared {removed_entries} item(s) from {slurm_jobs_dir}.")
    else:
        print(f"SLURM job directory {slurm_jobs_dir} was already empty.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine A2C sweep hyperparameters and evaluation metrics into a CSV table.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("WorkingFiles/Sweeps/a2c_slurm"),
        help="Directory produced by submit_slurm_a2c_sweep.py (default: WorkingFiles/Sweeps/a2c_slurm).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the aggregated CSV (default: <input-dir>/summary.csv).",
    )
    parser.add_argument(
        "--slurm-jobs-dir",
        type=Path,
        default=Path("WorkingFiles/SlurmJobs"),
        help="Directory containing generated sbatch files to purge after aggregation.",
    )
    parser.add_argument(
        "--skip-cleanup",
        action="store_true",
        help="Do not remove generated SLURM job scripts after summarising results.",
    )

    args = parser.parse_args()

    input_dir: Path = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Sweep directory not found: {input_dir}")

    run_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        row = gather_run_rows(run_dir)
        if row is not None:
            rows.append(row)

    output_path = args.output or (input_dir / "summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")

    if not args.skip_cleanup:
        cleanup_slurm_jobs(args.slurm_jobs_dir)
    else:
        print("Skipping SLURM job cleanup as requested.")


if __name__ == "__main__":
    main()
