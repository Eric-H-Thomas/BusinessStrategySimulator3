#!/usr/bin/env python3
"""Submit PPO training jobs for every simulator configuration in a directory."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


def _collect_config_files(config_dir: Path) -> List[Path]:
    """Return all JSON configuration files inside ``config_dir`` recursively sorted by name."""
    if not config_dir.exists():
        raise FileNotFoundError(f"Config directory does not exist: {config_dir}")
    if not config_dir.is_dir():
        raise NotADirectoryError(f"Config path is not a directory: {config_dir}")

    configs = sorted(
        p
        for p in config_dir.rglob("*.json")
        if p.is_file() and p.suffix.lower() == ".json"
    )
    if not configs:
        raise FileNotFoundError(f"No JSON configuration files found in {config_dir}")
    return configs


def build_submit_command(
    submit_script: Path,
    args: argparse.Namespace,
    job_name: str,
    output_path: Path,
    config_path: Path,
    extra_training_args: Iterable[str],
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
    cmd.extend(str(arg) for arg in extra_training_args)

    if args.disable_obs_normalization:
        cmd.append("--normalize_obs")
    if args.disable_reward_normalization:
        cmd.append("--normalize_reward")

    return cmd


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Submit a SLURM job for every simulator configuration JSON in a directory."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Directory that contains the simulator configuration JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "WorkingFiles" / "Sweeps" / "ppo_config_batch",
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
        default="ppo-config",
        help="Prefix used when naming SLURM jobs.",
    )
    parser.add_argument(
        "--num-agents-per-config-file",
        type=int,
        default=10,
        help="Number of independent training runs to launch per config file.",
    )
    parser.add_argument(
        "--shareability-test",
        action="store_true",
        help="Generate shareability summary CSV and boxplot after training.",
    )
    parser.add_argument(
        "--analysis-only",
        type=Path,
        help=(
            "Path to a directory to search for evaluation_metrics.json files "
            "and generate shareability summaries without submitting jobs."
        ),
    )
    parser.add_argument(
        "--submit-script",
        type=Path,
        default=base_dir / "scripts" / "submit_slurm_training_job.sh",
        help="Path to the helper that generates the sbatch file.",
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
        default=0.995,
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
        default=0.25,
        help="Clipping range for the PPO policy objective.",
    )
    parser.add_argument(
        "--ent-coef",
        dest="ent_coef",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient.",
    )
    parser.add_argument(
        "--vf-coef",
        dest="vf_coef",
        type=float,
        default=1.0,
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

    if args.analysis_only is not None:
        analysis_dir = args.analysis_only.resolve()
        if not analysis_dir.exists():
            raise FileNotFoundError(f"Analysis directory does not exist: {analysis_dir}")
        if not analysis_dir.is_dir():
            raise NotADirectoryError(
                f"Analysis path is not a directory: {analysis_dir}"
            )

        metrics = collect_mean_rewards(analysis_dir)
        if not metrics:
            print(
                "No evaluation_metrics.json files found. "
                "Check the analysis path and try again."
            )
            return

        csv_path = analysis_dir / "shareability_mean_rewards.csv"
        write_mean_reward_csv(metrics, csv_path)
        plot_path = analysis_dir / "shareability_mean_rewards_boxplot.png"
        write_mean_reward_boxplot(metrics, plot_path)
        print(f"Wrote mean reward CSV to {csv_path}")
        print(f"Wrote box-and-whisker plot to {plot_path}")
        return

    if args.config_dir is None:
        raise ValueError("--config-dir is required unless --analysis-only is set.")

    config_dir = args.config_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.job_script_dir is None:
        args.job_script_dir = base_dir / "WorkingFiles" / "SlurmJobs" / "ppo_config"

    if not args.submit_script.exists():
        raise FileNotFoundError(f"Submission helper not found: {args.submit_script}")

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
        shareability_dir = output_dir / f"{index:03d}_{config_path.stem}"
        shareability_dir.mkdir(parents=True, exist_ok=True)

        for agent_index in range(args.num_agents_per_config_file):
            run_dir = shareability_dir / f"agent_{agent_index:02d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            output_path = run_dir / "Agent.zip"

            metadata = dict(hyperparameters)
            metadata["config_source"] = str(config_path.resolve())
            metadata["agent_index"] = agent_index
            (run_dir / "hyperparameters.json").write_text(
                json.dumps(metadata, indent=2)
            )

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

            job_name = f"{args.job_name_prefix}-{index:03d}-a{agent_index:02d}"

            cmd = build_submit_command(
                args.submit_script,
                args,
                job_name,
                output_path,
                run_config_path,
                extra_training_args,
            )

            quoted = " ".join(shlex.quote(part) for part in cmd)
            print(f"[config {index:03d} agent {agent_index:02d}] {quoted}")

            subprocess.run(cmd, check=True)

    if args.dry_run:
        print("Dry run complete; inspect the generated sbatch scripts before submitting.")
    else:
        total_jobs = len(configs) * args.num_agents_per_config_file
        print(f"Submitted {total_jobs} jobs via {args.submit_script}.")

    if args.shareability_test:
        metrics = collect_mean_rewards(output_dir)
        if not metrics:
            print(
                "No evaluation_metrics.json files found. "
                "Run this script again after training completes to generate plots."
            )
            return

        csv_path = output_dir / "shareability_mean_rewards.csv"
        write_mean_reward_csv(metrics, csv_path)
        plot_path = output_dir / "shareability_mean_rewards_boxplot.png"
        write_mean_reward_boxplot(metrics, plot_path)
        print(f"Wrote mean reward CSV to {csv_path}")
        print(f"Wrote box-and-whisker plot to {plot_path}")


def collect_mean_rewards(output_dir: Path) -> List[Tuple[int, str, float]]:
    """Collect mean rewards from evaluation metrics files under the output directory."""
    results: List[Tuple[int, str, float]] = []
    for metrics_path in output_dir.rglob("evaluation_metrics.json"):
        try:
            data = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON: {metrics_path}")
            continue

        mean_reward = data.get("mean_reward")
        if mean_reward is None:
            print(f"Skipping metrics without mean_reward: {metrics_path}")
            continue

        group_dir = metrics_path.parent.parent
        index, shareability = parse_shareability_dir(group_dir.name)
        results.append((index, shareability, float(mean_reward)))
    return results


def parse_shareability_dir(name: str) -> Tuple[int, str]:
    """Parse a shareability directory name like '000_HighShareability'."""
    prefix, _, remainder = name.partition("_")
    if prefix.isdigit() and remainder:
        return int(prefix), remainder
    return 9999, name


def write_mean_reward_csv(
    metrics: List[Tuple[int, str, float]], output_path: Path
) -> None:
    """Write per-run mean rewards to CSV."""
    metrics_sorted = sorted(metrics, key=lambda item: (item[0], item[1]))
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shareability_level", "mean_reward"])
        for _, shareability, mean_reward in metrics_sorted:
            writer.writerow([shareability, mean_reward])


def write_mean_reward_boxplot(
    metrics: List[Tuple[int, str, float]], output_path: Path
) -> None:
    """Create a box-and-whisker plot for mean rewards per shareability level."""
    grouped: Dict[Tuple[int, str], List[float]] = {}
    for index, shareability, mean_reward in metrics:
        grouped.setdefault((index, shareability), []).append(mean_reward)

    shareability_order = [
        "NoShareability",
        "LowConstantShareability",
        "LowVaryingShareability",
        "ModerateConstantShareability",
        "ModerateVaryingShareability",
        "HighConstantShareability",
        "HighVaryingShareability",
        "PerfectShareability",
    ]
    order_index = {name: position for position, name in enumerate(shareability_order)}
    ordered = sorted(
        grouped.items(),
        key=lambda item: (
            order_index.get(item[0][1], len(order_index)),
            item[0][0],
            item[0][1],
        ),
    )
    labels = [shareability for (_, shareability), _ in ordered]
    data = [values for _, values in ordered]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, patch_artist=True)
    plt.xlabel("Shareability level")
    plt.ylabel("Mean reward")
    plt.title("Mean reward distribution by shareability level")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
