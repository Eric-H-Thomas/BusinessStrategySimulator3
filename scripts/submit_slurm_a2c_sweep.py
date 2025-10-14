#!/usr/bin/env python3
"""Submit an A2C hyperparameter sweep where each configuration runs as a SLURM job."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def cartesian_product(space: Dict[str, Sequence[object]]) -> Iterable[Dict[str, object]]:
    """Yield dictionaries representing every point in a hyperparameter grid."""

    keys = list(space.keys())
    for values in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, values))


def build_submit_command(
    submit_script: Path,
    args: argparse.Namespace,
    job_name: str,
    output_path: Path,
    combo: Dict[str, object],
    config_path: Path,
) -> List[str]:
    """Create the command used to submit a SLURM job for a single configuration."""

    cmd: List[str] = [
        str(submit_script),
        "--config",
        str(config_path),
        "--output",
        str(output_path),
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
    if args.dry_run:
        cmd.append("--dry-run")

    cmd.append("--")
    for key, value in combo.items():
        option = f"--{key.replace('_', '-')}"
        cmd.extend([option, str(value)])

    if args.disable_obs_normalization:
        cmd.append("--normalize_obs")
    if args.disable_reward_normalization:
        cmd.append("--normalize_reward")

    return cmd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Submit a SLURM job per A2C hyperparameter configuration.",
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
        default=base_dir / "WorkingFiles" / "Sweeps" / "a2c_slurm",
        help="Directory where run outputs and metadata will be stored.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments to request per run.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=500,
        help="Number of training updates performed by each job.",
    )
    parser.add_argument(
        "--job-name-prefix",
        default="a2c-sweep",
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
        default=base_dir / "scripts" / "submit_slurm_training_job.sh",
        help="Path to the helper that generates the sbatch file.",
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

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.submit_script.exists():
        raise FileNotFoundError(f"Submission helper not found: {args.submit_script}")

    base_config = json.loads(args.config.read_text())
    hyperparameter_space: Dict[str, Sequence[object]] = {
        "learning_rate": [3e-4, 7e-4, 1e-3],
        "gamma": [0.99, 0.995],
        "gae_lambda": [0.9, 0.95, 0.99],
        "ent_coef": [0.0, 0.01],
        "vf_coef": [0.5, 1.0],
        "max_grad_norm": [0.5, 0.75],
        "rms_prop_eps": [1e-5, 1e-4],
    }

    rollout_settings: Sequence[Tuple[int, int]] = (
        (5, 40),
        (20, 160),
    )

    run_counter = 0

    for n_steps, batch_size in rollout_settings:
        grid = dict(hyperparameter_space)
        grid.update({"n_steps": [n_steps], "batch_size": [batch_size]})

        for combo in cartesian_product(grid):
            if args.max_runs is not None and run_counter >= args.max_runs:
                break

            run_id = run_counter
            run_counter += 1

            combo_dir = output_dir / f"run_{run_id:03d}"
            combo_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = combo_dir / "hyperparameters.json"
            metadata = dict(combo)
            metadata["algorithm"] = "a2c"
            metadata_path.write_text(json.dumps(metadata, indent=2))

            output_path = combo_dir / "Agent.zip"
            run_results_dir = combo_dir.resolve()

            combo_config = copy.deepcopy(base_config)
            sim_params = combo_config.setdefault("simulation_parameters", {})
            sim_params["results_dir"] = str(run_results_dir)
            for agent in combo_config.get("ai_agents", []):
                agent["path_to_agent"] = str(output_path.resolve())

            combo_config_path = combo_dir / "config.json"
            combo_config_path.write_text(json.dumps(combo_config, indent=2))

            job_name = f"{args.job_name_prefix}-{run_id:03d}"

            cmd = build_submit_command(
                args.submit_script,
                args,
                job_name,
                output_path,
                combo,
                combo_config_path,
            )

            quoted = " ".join(shlex.quote(part) for part in cmd)
            print(f"[run {run_id:03d}] {quoted}")

            subprocess.run(cmd, check=True)

        if args.max_runs is not None and run_counter >= args.max_runs:
            break

    if args.dry_run:
        print("Dry run complete; inspect the generated sbatch scripts before submitting.")
    else:
        print(f"Submitted {run_counter} jobs via {args.submit_script}.")


if __name__ == "__main__":
    main()
