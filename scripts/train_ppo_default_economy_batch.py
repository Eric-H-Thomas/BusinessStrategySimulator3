#!/usr/bin/env python3
"""Submit PPO jobs to train multiple agents on the default economy config."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Dict, List


def build_array_submit_command(
    submit_script: Path,
    args: argparse.Namespace,
    job_name: str,
    manifest_path: Path,
) -> List[str]:
    """Create the command used to submit a SLURM array job for many training runs."""

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
    base_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Submit SLURM jobs to train many agents on the default economy configuration."
        )
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=100,
        help="Number of independent agents to train.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=base_dir / "WorkingFiles" / "Config" / "TestBench" / "DefaultEconomy.json",
        help="Path to the simulator configuration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "WorkingFiles" / "Sweeps" / "ppo_default_economy_batch",
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
        default=400,
        help="Number of PPO updates performed by each job.",
    )
    parser.add_argument(
        "--job-name-prefix",
        default="ppo-default-economy",
        help="Prefix used when naming SLURM jobs.",
    )
    parser.add_argument(
        "--submit-script",
        type=Path,
        default=base_dir / "scripts" / "submit_slurm_training_array.sh",
        help="Path to the helper that generates the array sbatch file.",
    )
    parser.add_argument(
        "--time",
        default="23:00:00",
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
    parser.add_argument(
        "--mem",
        dest="memory",
        default="16G",
        help="Memory request (e.g., 16G).",
    )
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
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate passed to PPO.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards.",
    )
    parser.add_argument(
        "--gae-lambda",
        dest="gae_lambda",
        type=float,
        default=0.9,
        help="Generalized Advantage Estimation lambda parameter.",
    )
    parser.add_argument(
        "--clip-range",
        dest="clip_range",
        type=float,
        default=0.15,
        help="Clipping range for the PPO policy objective.",
    )
    parser.add_argument(
        "--ent-coef",
        dest="ent_coef",
        type=float,
        default=0.02,
        help="Entropy bonus coefficient.",
    )
    parser.add_argument(
        "--vf-coef",
        dest="vf_coef",
        type=float,
        default=0.5,
        help="Value function loss coefficient.",
    )
    parser.add_argument(
        "--n-steps",
        dest="n_steps",
        type=int,
        default=1024,
        help="Rollout length (environment steps) collected before each PPO update.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=256,
        help="Mini-batch size used during gradient updates.",
    )

    args = parser.parse_args()

    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.job_script_dir is None:
        args.job_script_dir = base_dir / "WorkingFiles" / "SlurmJobs" / "ppo_default_economy"

    if not args.submit_script.exists():
        raise FileNotFoundError(f"Submission helper not found: {args.submit_script}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    base_config = json.loads(config_path.read_text())

    hyperparameters: Dict[str, float | int | str] = {
        "algorithm": "ppo",
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
    }

    manifest_entries: List[Dict[str, object]] = []

    for index in range(args.num_agents):
        run_dir = output_dir / f"{index:03d}_DefaultEconomy"
        run_dir.mkdir(parents=True, exist_ok=True)

        output_path = run_dir / "Agent.zip"

        metadata = dict(hyperparameters)
        metadata["config_source"] = str(config_path)
        (run_dir / "hyperparameters.json").write_text(json.dumps(metadata, indent=2))

        config_data = deepcopy(base_config)
        sim_params = config_data.setdefault("simulation_parameters", {})
        sim_params["results_dir"] = str(run_dir.resolve())
        for agent in config_data.get("ai_agents", []):
            agent["path_to_agent"] = str(output_path.resolve())

        run_config_path = run_dir / "config.json"
        run_config_path.write_text(json.dumps(config_data, indent=2))

        extra_training_args: List[str] = []
        for key, value in hyperparameters.items():
            option = f"--{key.replace('_', '-')}"
            extra_training_args.extend([option, str(value)])

        if args.disable_obs_normalization:
            extra_training_args.append("--normalize_obs")
        if args.disable_reward_normalization:
            extra_training_args.append("--normalize_reward")

        manifest_entries.append(
            {
                "config": str(run_config_path),
                "output": str(output_path),
                "extra_args": extra_training_args,
                "num_updates": args.num_updates,
                "num_envs": args.num_envs,
            }
        )
        print(f"[agent {index:03d}] queued for array submission")

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
    print(f"Submitting array job with {args.num_agents} tasks: {quoted}")
    subprocess.run(cmd, check=True)

    if args.dry_run:
        print("Dry run complete; inspect the generated sbatch script before submitting.")
    else:
        print(f"Submitted {args.num_agents} array tasks via {args.submit_script}.")


if __name__ == "__main__":
    main()
