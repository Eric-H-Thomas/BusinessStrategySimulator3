#!/usr/bin/env python3
"""Submit PPO, DQN, and A2C hyperparameter sweeps in one command."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


def _expand_args(chunks: Sequence[str]) -> List[str]:
    expanded: List[str] = []
    for chunk in chunks:
        expanded.extend(shlex.split(chunk))
    return expanded


def _run_sweep(script_path: Path, args: Sequence[str]) -> None:
    cmd = [sys.executable, str(script_path), *args]
    quoted = " ".join(shlex.quote(part) for part in cmd)
    print(f"Launching: {quoted}")
    subprocess.run(cmd, check=True)


def main() -> None:
    base_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Submit PPO, DQN, and A2C sweeps sequentially.",
        epilog=(
            "Use --common-args to specify options shared by all sweeps and "
            "--<algo>-args to pass additional parameters to individual sweeps."
        ),
    )
    parser.add_argument(
        "--scripts-dir",
        type=Path,
        default=base_dir,
        help="Directory containing the sweep submission scripts.",
    )
    parser.add_argument(
        "--common-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Additional arguments forwarded to every sweep (may be provided multiple times).",
    )
    parser.add_argument(
        "--ppo-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Arguments forwarded only to the PPO sweep (may be provided multiple times).",
    )
    parser.add_argument(
        "--dqn-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Arguments forwarded only to the DQN sweep (may be provided multiple times).",
    )
    parser.add_argument(
        "--a2c-args",
        action="append",
        default=[],
        metavar="ARGS",
        help="Arguments forwarded only to the A2C sweep (may be provided multiple times).",
    )
    parser.add_argument(
        "--skip-ppo",
        action="store_true",
        help="Do not submit the PPO sweep.",
    )
    parser.add_argument(
        "--skip-dqn",
        action="store_true",
        help="Do not submit the DQN sweep.",
    )
    parser.add_argument(
        "--skip-a2c",
        action="store_true",
        help="Do not submit the A2C sweep.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Append --dry-run to any sweep arguments that do not already include it.",
    )

    args = parser.parse_args()

    scripts_dir: Path = args.scripts_dir
    if not scripts_dir.exists():
        raise FileNotFoundError(f"Sweep scripts directory not found: {scripts_dir}")

    common_args = _expand_args(args.common_args)
    if args.dry_run and "--dry-run" not in common_args:
        common_args.append("--dry-run")

    sweep_scripts = [
        ("submit_slurm_ppo_sweep.py", args.skip_ppo, _expand_args(args.ppo_args)),
        ("submit_slurm_dqn_sweep.py", args.skip_dqn, _expand_args(args.dqn_args)),
        ("submit_slurm_a2c_sweep.py", args.skip_a2c, _expand_args(args.a2c_args)),
    ]

    for script_name, skip, specific_args in sweep_scripts:
        if skip:
            print(f"Skipping {script_name} as requested.")
            continue

        script_path = scripts_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"Required sweep script not found: {script_path}")

        sweep_args = list(common_args)
        if args.dry_run and "--dry-run" not in specific_args:
            sweep_args.append("--dry-run")
        sweep_args.extend(specific_args)

        _run_sweep(script_path, sweep_args)


if __name__ == "__main__":
    main()
