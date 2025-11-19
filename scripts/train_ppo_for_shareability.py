#!/usr/bin/env python3
"""Submit PPO training jobs for each shareability configuration.

This script mirrors :mod:`train_ppo_for_config_batch` but specifically targets the
configuration JSON files stored in ``WorkingFiles/Config/TestBench/VaryingShareability``.
Use it to launch a fresh PPO training run for each scenario in that folder so you
can study how the shareability parameter influences performance.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List

from scripts.train_ppo_for_config_batch import _collect_config_files, build_submit_command


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    shareability_dir = base_dir / "WorkingFiles" / "Config" / "TestBench" / "VaryingShareability"

    parser = argparse.ArgumentParser(
        description=(
            "Submit a SLURM job for every shareability configuration JSON in "
            "WorkingFiles/Config/TestBench/VaryingShareability."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "WorkingFiles" / "Sweeps" / "ppo_shareability",
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
        default="ppo-shareability",
        help="Prefix used when naming SLURM jobs.",
    )
    parser.add_argument(
        "--submit-script",
        type=Path,
        default=base_dir / "scripts" / "submit_slurm_training_job.sh",
        help="Path to the helper that generates the sbatch file.",
    )
    parser.add_argument("--time", help="Requested walltime for each job (e.g., 04:00:00).")
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
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate passed to PPO.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.999,
        help="Discount factor for future rewards.",
    )
    parser.add_argument(
        "--gae-lambda",
        dest="gae_lambda",
        type=float,
        default=0.95,
        help="Generalized Advantage Estimation lambda parameter.",
    )
    parser.add_argument(
        "--clip-range",
        dest="clip_range",
        type=float,
        default=0.2,
        help="Clipping range for the PPO policy objective.",
    )
    parser.add_argument(
        "--ent-coef",
        dest="ent_coef",
        type=float,
        default=0.0,
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
        default=512,
        help="Rollout length (environment steps) collected before each PPO update.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Mini-batch size used during gradient updates.",
    )

    args = parser.parse_args()

    config_dir = shareability_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.job_script_dir is None:
        args.job_script_dir = base_dir / "WorkingFiles" / "SlurmJobs" / "ppo_shareability"

    if not args.submit_script.exists():
        raise FileNotFoundError(f"Submission helper not found: {args.submit_script}")
    if not config_dir.exists():
        raise FileNotFoundError(
            "Shareability configuration directory does not exist: " f"{config_dir}"
        )

    configs = _collect_config_files(config_dir)

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

    for index, config_path in enumerate(configs):
        run_dir = output_dir / f"{index:03d}_{config_path.stem}"
        run_dir.mkdir(parents=True, exist_ok=True)

        output_path = run_dir / "Agent.zip"

        metadata = dict(hyperparameters)
        metadata["config_source"] = str(config_path.resolve())
        (run_dir / "hyperparameters.json").write_text(json.dumps(metadata, indent=2))

        config_data = json.loads(config_path.read_text())
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

        job_name = f"{args.job_name_prefix}-{index:03d}"

        cmd = build_submit_command(
            args.submit_script,
            args,
            job_name,
            output_path,
            run_config_path,
            extra_training_args,
        )

        quoted = " ".join(shlex.quote(part) for part in cmd)
        print(f"[config {index:03d}] {quoted}")

        subprocess.run(cmd, check=True)

    if args.dry_run:
        print("Dry run complete; inspect the generated sbatch scripts before submitting.")
    else:
        print(f"Submitted {len(configs)} jobs via {args.submit_script}.")


if __name__ == "__main__":
    main()
