#!/usr/bin/env python3
"""Aggregate hyperparameter sweep results into per-run and per-combo CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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

    row: Dict[str, Any] = {"run": str(run_dir)}
    hyperparameters = load_json(hyper_path)
    row.update({str(key): hyperparameters[key] for key in sorted(hyperparameters)})

    if eval_path.exists():
        evaluation = load_json(eval_path)
        mean_reward = evaluation.get("mean_reward")
        if isinstance(mean_reward, (int, float)):
            row["mean_reward"] = mean_reward
    else:
        row["status"] = "missing_evaluation"

    return row


_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(
    r"^[+-]?((\d+\.\d*)|(\d*\.\d+)|\d+)([eE][+-]?\d+)?$"
)


def _normalise_value(value: Any) -> Any:
    """Tweak numeric values so spreadsheet tools treat them as numbers."""

    if isinstance(value, bool):
        # bool is a subclass of int, but we want to keep True/False values unchanged.
        return value

    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"

        # Excel treats floats with long binary tails as text. Rounding curbs this.
        return round(value, 12)

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value

        if _INTEGER_RE.match(stripped):
            try:
                return int(stripped)
            except ValueError:
                # Value exceeds Python's int range; leave it as text.
                return value

        if _FLOAT_RE.match(stripped):
            try:
                numeric = float(stripped)
            except ValueError:
                return value

            if math.isnan(numeric):
                return "nan"
            if math.isinf(numeric):
                return "inf" if numeric > 0 else "-inf"

            return round(numeric, 12)

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


def iter_run_dirs(root_dir: Path) -> Iterable[Path]:
    for hyper_path in root_dir.rglob("hyperparameters.json"):
        run_dir = hyper_path.parent
        if run_dir.is_dir():
            yield run_dir


def _combo_group_key(row: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    excluded = {"run", "agent_index", "combo_id", "status", "mean_reward"}
    items = tuple(
        (key, row[key])
        for key in sorted(row.keys())
        if key not in excluded and not key.startswith("eval_")
    )
    return items


def summarise_by_combo(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Tuple[str, Any], ...], List[float]] = {}
    for row in rows:
        mean_reward = row.get("mean_reward")
        if not isinstance(mean_reward, (int, float)):
            continue
        key = _combo_group_key(row)
        grouped.setdefault(key, []).append(float(mean_reward))

    summary_rows: List[Dict[str, Any]] = []
    for key, rewards in grouped.items():
        if not rewards:
            continue
        combo_row = {k: v for k, v in key}
        combo_row["mean_reward_avg"] = sum(rewards) / len(rewards)
        combo_row["num_runs"] = len(rewards)
        summary_rows.append(combo_row)

    return summary_rows


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
        # Directory may still contain files we could not delete; leave it in place.
        pass

    if removed_entries:
        print(f"Cleared {removed_entries} item(s) from {slurm_jobs_dir}.")
    else:
        print(f"SLURM job directory {slurm_jobs_dir} was already empty.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine PPO sweep hyperparameters and evaluation metrics into a CSV table.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("WorkingFiles/Sweeps/ppo_slurm"),
        help="Directory produced by submit_slurm_ppo_sweep.py (default: WorkingFiles/Sweeps/ppo_slurm).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the per-run CSV (default: <input-dir>/summary.csv).",
    )
    parser.add_argument(
        "--combo-output",
        type=Path,
        help=(
            "Where to write the per-combo averaged CSV "
            "(default: <input-dir>/summary_by_combo.csv)."
        ),
    )
    parser.add_argument(
        "--slurm-jobs-dir",
        type=Path,
        default=Path("WorkingFiles/SlurmJobs/ppo"),
        help="Directory containing generated sbatch files to purge after aggregation (default: WorkingFiles/SlurmJobs/ppo).",
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

    run_dirs = sorted(iter_run_dirs(input_dir))
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        row = gather_run_rows(run_dir)
        if row is not None:
            rows.append(row)

    output_path = args.output or (input_dir / "summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_path)
    print(f"Wrote {len(rows)} rows to {output_path}")

    combo_rows = summarise_by_combo(rows)
    combo_output_path = args.combo_output or (input_dir / "summary_by_combo.csv")
    combo_output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv(combo_rows, combo_output_path)
    print(f"Wrote {len(combo_rows)} rows to {combo_output_path}")

    if not args.skip_cleanup:
        cleanup_slurm_jobs(args.slurm_jobs_dir)
    else:
        print("Skipping SLURM job cleanup as requested.")


if __name__ == "__main__":
    main()
