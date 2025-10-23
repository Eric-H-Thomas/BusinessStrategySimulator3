#!/usr/bin/env python3
"""Train a PPO agent for every simulator configuration in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from business_strategy_gym_env import make_env


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


def _build_vector_env(
    *,
    config_path: Path,
    num_envs: int,
    seed: int,
    normalize_obs: bool,
    normalize_reward: bool,
) -> VecNormalize | DummyVecEnv | SubprocVecEnv:
    env_fns: Iterable = [
        make_env(config_path, seed + idx, normalize_observations=normalize_obs)
        for idx in range(num_envs)
    ]

    if num_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns)

    if normalize_reward:
        env = VecNormalize(
            env,
            norm_obs=False,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0,
        )

    return env


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description=(
            "Train a PPO agent for every simulator configuration JSON in a directory."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory that contains the simulator configuration JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=base_dir / "AgentFiles" / "ppo_batch",
        help="Directory where trained PPO checkpoints will be stored.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Total number of training timesteps per configuration.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel environments to use while training each agent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed used to initialize environments and policies.",
    )
    parser.add_argument(
        "--no-normalize-obs",
        action="store_true",
        help="Disable observation normalization inside the environment wrapper.",
    )
    parser.add_argument(
        "--no-normalize-reward",
        action="store_true",
        help="Disable reward normalization using VecNormalize.",
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
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use the MPS GPU if available instead of the CPU.",
    )

    args = parser.parse_args()

    config_dir = args.config_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _collect_config_files(config_dir)

    device = (
        torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if args.use_gpu
        else "cpu"
    )

    normalize_obs = not args.no_normalize_obs
    normalize_reward = not args.no_normalize_reward

    for index, config_path in enumerate(configs):
        print(f"Training PPO agent for {config_path.name}...")
        seed = args.seed + index * args.num_envs

        env = _build_vector_env(
            config_path=config_path,
            num_envs=args.num_envs,
            seed=seed,
            normalize_obs=normalize_obs,
            normalize_reward=normalize_reward,
        )

        rollout_cap = max(1, args.timesteps // max(1, args.num_envs))
        rollout_steps = min(args.n_steps, rollout_cap)
        effective_batch_size = max(1, min(args.batch_size, rollout_steps * args.num_envs))

        model = PPO(
            "MlpPolicy",
            env,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            n_steps=rollout_steps,
            batch_size=effective_batch_size,
            device=device,
            verbose=1,
        )

        try:
            model.learn(total_timesteps=args.timesteps)
        finally:
            env.close()

        checkpoint_path = output_dir / f"{config_path.stem}.zip"
        model.save(str(checkpoint_path))
        print(f"Saved PPO checkpoint to {checkpoint_path}")

    print("Finished training PPO agents for all configurations.")


if __name__ == "__main__":
    main()
