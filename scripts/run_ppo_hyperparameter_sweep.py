"""Utility for running a PPO hyperparameter sweep on the BusinessStrategyEnv.

The script mirrors the single-job training flow in ``business_strategy_gym_env.py``
while iterating over a grid of PPO hyperparameters. After each training run the
resulting policy is evaluated over several rollouts so that learning progress
can be compared across configurations.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env import VecNormalize

from business_strategy_gym_env import BusinessStrategyEnv, make_env


@dataclass
class SweepResult:
    """Container for storing metadata and evaluation results for a run."""

    run_id: int
    hyperparameters: Dict[str, float]
    mean_reward: float
    reward_std: float
    total_timesteps: int
    training_time_sec: float
    model_path: str

    def to_serializable(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["hyperparameters"] = dict(self.hyperparameters)
        return payload


def cartesian_product(space: Dict[str, Sequence[object]]) -> Iterable[Dict[str, object]]:
    """Generate points from a hyperparameter grid."""

    keys = list(space.keys())
    for values in itertools.product(*(space[key] for key in keys)):
        yield dict(zip(keys, values))


def build_vec_env(config_path: Path, num_envs: int, seed: int, normalize_obs: bool) -> VecEnv:
    env_fns = [make_env(config_path, seed + idx, normalize_observations=normalize_obs) for idx in range(num_envs)]
    return SubprocVecEnv(env_fns)


def evaluate_model(
    model: PPO,
    config_path: Path,
    episodes: int,
    seed: int,
    normalize_obs: bool,
) -> Tuple[float, float]:
    """Roll out ``episodes`` trajectories and return the mean and std reward."""

    eval_env = Monitor(BusinessStrategyEnv(str(config_path), normalize_observations=normalize_obs))
    try:
        eval_env.reset(seed=seed)
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=episodes,
            deterministic=True,
        )
    finally:
        eval_env.close()
    return float(mean_reward), float(std_reward)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a PPO hyperparameter sweep.")
    base_dir = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--config",
        type=Path,
        default=base_dir / "WorkingFiles" / "Config" / "default.json",
        help="Path to the simulator configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "WorkingFiles" / "Sweeps" / "ppo",
        help="Where sweep artifacts (models, metrics) will be stored.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments to use during training.",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=400,
        help="Number of PPO update iterations (higher = longer training).",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="How many evaluation episodes to average over for each run.",
    )
    parser.add_argument(
        "--normalize-obs",
        dest="normalize_obs",
        action="store_true",
        help="Enable observation normalization inside the environment.",
    )
    parser.add_argument(
        "--no-normalize-obs",
        dest="normalize_obs",
        action="store_false",
        help="Disable observation normalization inside the environment.",
    )
    parser.set_defaults(normalize_obs=True)
    parser.add_argument(
        "--disable-reward-normalization",
        action="store_true",
        help="Disable VecNormalize reward scaling during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used for training and evaluation.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional limit on how many hyperparameter combinations to execute.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run training on (e.g., 'cpu', 'cuda', or 'auto').",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="Verbosity level passed to Stable-Baselines3.",
    )

    args = parser.parse_args()

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define a compact but expressive hyperparameter grid.
    # ``n_steps`` and ``batch_size`` are paired to ensure divisibility constraints.
    paired_rollout_settings: Sequence[Tuple[int, int]] = [(1024, 256), (2048, 512)]
    hyperparameter_space: Dict[str, Sequence[object]] = {
        "learning_rate": [1e-4, 3e-4, 1e-3],
        "gamma": [0.99, 0.995, 0.999],
        "gae_lambda": [0.9, 0.95],
        "clip_range": [0.15, 0.25],
        "ent_coef": [0.0, 0.01],
        "vf_coef": [0.5, 1.0],
    }

    # Prepare CSV and JSON writers for aggregate results.
    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    csv_file = csv_path.open("w", newline="")
    csv_writer = None
    all_results: List[SweepResult] = []

    try:
        run_counter = 0
        for (n_steps, batch_size) in paired_rollout_settings:
            grid = dict(hyperparameter_space)
            grid.update({"n_steps": [n_steps], "batch_size": [batch_size]})

            for combo in cartesian_product(grid):
                if args.max_runs is not None and run_counter >= args.max_runs:
                    break

                run_id = run_counter
                run_counter += 1

                combo_dir = output_dir / f"run_{run_id:03d}"
                combo_dir.mkdir(parents=True, exist_ok=True)

                total_timesteps = int(n_steps * args.num_envs * args.num_updates)
                device = args.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else (
                        "mps" if torch.backends.mps.is_available() else "cpu"
                    )

                train_env = build_vec_env(args.config, args.num_envs, args.seed + run_id * 10, args.normalize_obs)
                if not args.disable_reward_normalization:
                    train_env = VecNormalize(
                        train_env,
                        norm_obs=False,
                        norm_reward=True,
                        clip_obs=10.0,
                        clip_reward=10.0,
                    )

                model = PPO(
                    "MlpPolicy",
                    train_env,
                    verbose=args.verbose,
                    device=device,
                    tensorboard_log=str(combo_dir / "tensorboard"),
                    **combo,
                )

                start_time = time.perf_counter()
                model.learn(total_timesteps=total_timesteps)
                training_time = time.perf_counter() - start_time

                model_path = combo_dir / "model"
                model.save(str(model_path))

                if isinstance(train_env, VecNormalize):
                    # Save normalization statistics so evaluations can be reproduced if desired.
                    train_env.save(str(combo_dir / "vecnormalize.pkl"))

                mean_reward, std_reward = evaluate_model(
                    model,
                    args.config,
                    episodes=args.eval_episodes,
                    seed=args.seed + run_id,
                    normalize_obs=args.normalize_obs,
                )

                result = SweepResult(
                    run_id=run_id,
                    hyperparameters=combo,
                    mean_reward=mean_reward,
                    reward_std=std_reward,
                    total_timesteps=total_timesteps,
                    training_time_sec=training_time,
                    model_path=str(model_path.with_suffix(".zip")),
                )

                # Save run-specific metrics.
                metrics_path = combo_dir / "metrics.json"
                metrics_path.write_text(json.dumps(result.to_serializable(), indent=2))

                all_results.append(result)

                if csv_writer is None:
                    csv_writer = csv.DictWriter(
                        csv_file,
                        fieldnames=list(result.to_serializable().keys()),
                    )
                    csv_writer.writeheader()

                csv_writer.writerow(result.to_serializable())
                csv_file.flush()

                print(
                    f"[run {run_id:03d}] mean_reward={mean_reward:.2f} ± {std_reward:.2f} "
                    f"(timesteps={total_timesteps}, time={training_time/60:.1f} min)"
                )

                # Free resources associated with the model and environments before the next run.
                model.env.close()
                if hasattr(model, "eval_env") and model.eval_env is not None:
                    model.eval_env.close()
                del model

            if args.max_runs is not None and run_counter >= args.max_runs:
                break

    finally:
        csv_file.close()

    # Persist aggregated JSON summary.
    json_path.write_text(
        json.dumps([result.to_serializable() for result in all_results], indent=2)
    )

    print(f"Saved sweep results to {csv_path} and {json_path}.")


if __name__ == "__main__":
    main()
