#!/usr/bin/env python3
"""Submit a DQN hyperparameter sweep where each configuration runs as a SLURM job."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def cartesian_product(space: Dict[str, Sequence[object]]) -> Iterable[Dict[str, object]]:
    """Yield dictionaries representing every point in a hyperparameter grid."""

    keys = list(space.keys())
    for values in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, values))


def build_array_submit_command(
    submit_script: Path,
    args: argparse.Namespace,
    job_name: str,
    manifest_path: Path,
) -> List[str]:
    """Create the command used to submit a SLURM array job for all configurations."""

    cmd: List[str] = [
        str(submit_script),
        "--manifest",
        str(manifest_path),
        "--num-updates",
        str(args.num_updates),
        "--num-envs",
        str(args.num_envs),
        "--job-name",
        job_name,
    ]

    if args.time is not None:
        cmd.extend(["--time", args.time])
    if args.partition is not None:
        cmd.extend(["--partition", args.partition])
    if args.account is not None:
        cmd.extend(["--account", args.account])
    if args.qos is not None:
        cmd.extend(["--qos", args.qos])
    if args.nodes is not None:
        cmd.extend(["--nodes", str(args.nodes)])
    if args.cpus_per_task is not None:
        cmd.extend(["--cpus-per-task", str(args.cpus_per_task)])
    if args.memory is not None:
        cmd.extend(["--mem", args.memory])
    if args.gres is not None:
        cmd.extend(["--gres", args.gres])
    if args.dependency is not None:
        cmd.extend(["--dependency", args.dependency])
    if args.venv is not None:
        cmd.extend(["--venv", str(args.venv)])
    if args.build_dir is not None:
        cmd.extend(["--build-dir", str(args.build_dir)])
    if args.job_script_dir is not None:
        job_script = args.job_script_dir / f"{job_name}.sbatch"
        cmd.extend(["--job-script", str(job_script)])
    if args.use_gpu:
        cmd.append("--use-gpu")
    for extra in args.extra_sbatch:
        cmd.extend(["--extra-sbatch", extra])
    if args.array_max_concurrent is not None:
        cmd.extend(["--array-max-concurrent", str(args.array_max_concurrent)])
    if args.dry_run:
        cmd.append("--dry-run")

    return cmd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Submit a SLURM job per DQN hyperparameter configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=base_dir / "WorkingFiles" / "Config" / "default.json",
        help="Simulator configuration to use for every run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "WorkingFiles" / "Sweeps" / "dqn_slurm",
        help="Directory where run outputs and metadata will be stored.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to request per run.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=600,
        help="Number of training updates performed by each job.",
    )
    parser.add_argument(
        "--job-name-prefix",
        default="dqn-sweep",
        help="Prefix used when naming SLURM jobs.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional limit on how many hyperparameter combinations to submit.",
    )
    parser.add_argument(
        "--submit-script",
        type=Path,
        default=base_dir / "scripts" / "submit_slurm_training_array.sh",
        help="Path to the helper that generates the array sbatch file.",
    )
    parser.add_argument(
        "--time",
        help="Requested walltime for each job (e.g., 04:00:00).",
    )
    parser.add_argument("--partition", help="SLURM partition/queue name.")
    parser.add_argument("--account", help="SLURM account to charge.")
    parser.add_argument("--qos", help="QoS name to request.")
    parser.add_argument("--nodes", type=int, help="Nodes per job.")
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        dest="cpus_per_task",
        help="CPUs per task.",
    )
    parser.add_argument("--mem", dest="memory", help="Memory request (e.g., 16G).")
    parser.add_argument("--gres", help="Generic resource request (e.g., gpu:1).")
    parser.add_argument("--dependency", help="SLURM dependency specification.")
    parser.add_argument(
        "--extra-sbatch",
        action="append",
        default=[],
        help="Additional raw #SBATCH lines to include.",
    )
    parser.add_argument(
        "--venv",
        type=Path,
        help="Virtual environment activated inside each job.",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        help="Build directory appended to PYTHONPATH inside the job.",
    )
    parser.add_argument(
        "--job-script-dir",
        type=Path,
        help="Directory where generated sbatch scripts should be stored.",
    )
    parser.add_argument(
        "--array-max-concurrent",
        type=int,
        default=50,
        help="Maximum number of array tasks to run concurrently.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Request GPU resources and enable GPU training.",
    )
    parser.add_argument(
        "--disable-obs-normalization",
        action="store_true",
        help="Disable observation normalization in the environment.",
    )
    parser.add_argument(
        "--disable-reward-normalization",
        action="store_true",
        help="Disable reward normalization during training.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate job scripts without submitting to SLURM.",
    )

    args = parser.parse_args()

    if args.job_script_dir is None:
        args.job_script_dir = base_dir / "WorkingFiles" / "SlurmJobs" / "dqn"

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.submit_script.exists():
        raise FileNotFoundError(f"Submission helper not found: {args.submit_script}")

    base_config = json.loads(args.config.read_text())
    hyperparameter_space: Dict[str, Sequence[object]] = {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "gamma": [0.98, 0.99, 0.995],
        "buffer_size": [50000, 100000],
        "batch_size": [64, 128],
        "exploration_fraction": [0.1, 0.2],
        "exploration_final_eps": [0.01, 0.02],
    }

    training_schedules: Sequence[Dict[str, int]] = (
        {"train_freq": 1, "gradient_steps": 1, "target_update_interval": 1000},
        {"train_freq": 4, "gradient_steps": 2, "target_update_interval": 4000},
    )

    run_counter = 0
    manifest_entries: List[Dict[str, object]] = []

    for schedule in training_schedules:
        grid = dict(hyperparameter_space)
        grid.update({key: [value] for key, value in schedule.items()})

        for combo in cartesian_product(grid):
            if args.max_runs is not None and run_counter >= args.max_runs:
                break

            run_id = run_counter
            run_counter += 1

            combo_dir = output_dir / f"run_{run_id:03d}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = combo_dir / "hyperparameters.json"
            combo_with_algo = dict(combo)
            combo_with_algo["algorithm"] = "dqn"
            metadata_path.write_text(json.dumps(combo_with_algo, indent=2))

            output_path = combo_dir / "Agent.zip"
            run_results_dir = combo_dir.resolve()

            combo_config = copy.deepcopy(base_config)
            sim_params = combo_config.setdefault("simulation_parameters", {})
            sim_params["results_dir"] = str(run_results_dir)
            for agent in combo_config.get("ai_agents", []):
                agent["path_to_agent"] = str(output_path.resolve())

            combo_config_path = combo_dir / "config.json"
            combo_config_path.write_text(json.dumps(combo_config, indent=2))

            extra_args: List[str] = []
            for key, value in combo_with_algo.items():
                option = f"--{key.replace('_', '-')}"
                extra_args.extend([option, str(value)])

            if args.disable_obs_normalization:
                extra_args.append("--normalize_obs")
            if args.disable_reward_normalization:
                extra_args.append("--normalize_reward")

            manifest_entries.append(
                {
                    "config": str(combo_config_path),
                    "output": str(output_path),
                    "extra_args": extra_args,
                    "num_updates": args.num_updates,
                    "num_envs": args.num_envs,
                }
            )
            print(f"[run {run_id:03d}] queued for array submission")

        if args.max_runs is not None and run_counter >= args.max_runs:
            break

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = output_dir / f"{args.job_name_prefix}_manifest_{timestamp}.jsonl"
    with manifest_path.open("w") as handle:
        for entry in manifest_entries:
            handle.write(json.dumps(entry))
            handle.write("\n")

    array_job_name = f"{args.job_name_prefix}-array"
    cmd = build_array_submit_command(
        args.submit_script,
        args,
        array_job_name,
        manifest_path,
    )

    quoted = " ".join(shlex.quote(part) for part in cmd)
    print(f"Submitting array job with {run_counter} tasks: {quoted}")
    subprocess.run(cmd, check=True)

    if args.dry_run:
        print("Dry run complete; inspect the generated sbatch script before submitting.")
    else:
        print(f"Submitted {run_counter} array tasks via {args.submit_script}.")


if __name__ == "__main__":
    main()
