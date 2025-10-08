import json
import simulator_module
import gymnasium as gym
import numpy as np

from typing import Callable

from stable_baselines3.common.monitor import Monitor


class BusinessStrategyEnv(gym.Env):
    def __init__(self, path_to_config_file, normalize_observations: bool = False):
        super().__init__()
        # INITIALIZE THE PYTHON API AND THE SIMULATOR
        self.python_API = simulator_module.PythonAPI()
        self.python_API.init_simulator(path_to_config_file)

        self.num_agents = self.python_API.get_num_agents()
        self.num_markets = self.python_API.get_num_markets()
        self.normalize_obs = normalize_observations

        # Pre-compute slices for each observation component
        start = 0
        self.capital_slice = slice(start, start + self.num_agents)
        start += self.num_agents
        self.overlap_slice = slice(start, start + self.num_markets**2)
        start += self.num_markets**2
        self.variable_cost_slice = slice(start, start + self.num_agents * self.num_markets)
        start += self.num_agents * self.num_markets
        self.fixed_cost_slice = slice(start, start + self.num_agents * self.num_markets)
        start += self.num_agents * self.num_markets
        self.market_portfolio_slice = slice(start, start + self.num_agents * self.num_markets)
        start += self.num_agents * self.num_markets
        self.entry_cost_slice = slice(start, start + self.num_agents * self.num_markets)
        start += self.num_agents * self.num_markets
        self.demand_intercept_slice = slice(start, start + self.num_markets)
        start += self.num_markets
        self.demand_slope_slice = slice(start, start + self.num_markets)
        start += self.num_markets
        self.quantity_slice = slice(start, start + self.num_agents * self.num_markets)
        start += self.num_agents * self.num_markets
        self.price_slice = slice(start, start + self.num_agents * self.num_markets)

        self.obs_scale = np.ones(self.price_slice.stop, dtype=np.float32)

        # DEFINE THE ACTION SPACE
        self.action_space = gym.spaces.Discrete(self.num_markets + 1)

        # DEFINE THE OBSERVATION SPACE
        if self.normalize_obs:
            obs_low = np.zeros(self.price_slice.stop, dtype=np.float32)
            obs_high = np.ones(self.price_slice.stop, dtype=np.float32)
            obs_low[self.demand_slope_slice] = -1.0
            obs_high[self.demand_slope_slice] = 1.0
        else:
            # Capital of all agents
            obs_low = np.zeros(self.num_agents)
            obs_high = np.full(self.num_agents, np.finfo(np.float32).max)

            # Market overlap structure
            obs_low = np.append(obs_low, np.zeros(self.num_markets**2))
            obs_high = np.append(obs_high, np.ones(self.num_markets**2))

            # Variable costs for all firm-market combinations
            obs_low = np.append(obs_low, np.zeros(self.num_agents * self.num_markets))
            obs_high = np.append(obs_high, np.full(self.num_agents * self.num_markets, np.finfo(np.float32).max))

            # Fixed cost for each firm-market combination
            obs_low = np.append(obs_low, np.zeros(self.num_agents * self.num_markets))
            obs_high = np.append(obs_high, np.full(self.num_agents * self.num_markets, np.finfo(np.float32).max))

            # Market portfolio of all firms
            obs_low = np.append(obs_low, np.zeros(self.num_agents * self.num_markets))
            obs_high = np.append(obs_high, np.ones(self.num_agents * self.num_markets))

            # Entry cost for every agent-firm combination
            obs_low = np.append(obs_low, np.zeros(self.num_agents * self.num_markets))
            obs_high = np.append(obs_high, np.full(self.num_agents * self.num_markets, np.finfo(np.float32).max))

            # Demand intercept and slope in each market
            obs_low = np.append(obs_low, np.zeros(2 * self.num_markets))
            obs_high = np.append(obs_high, np.full(2 * self.num_markets, np.finfo(np.float32).max))

            # Most recent quantity and price for each firm-market combination
            obs_low = np.append(obs_low, np.zeros(2 * self.num_agents * self.num_markets))
            obs_high = np.append(obs_high, np.full(2 * self.num_agents * self.num_markets, np.finfo(np.float32).max))

            # Convert the observation space's lower and upper bounds to np.float32
            obs_low = obs_low.astype(np.float32)
            obs_high = obs_high.astype(np.float32)

        obs_space = gym.spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.observation_space = obs_space

    def _compute_obs_scale(self, observation: np.ndarray) -> None:
        """Compute scaling factors for each observation component."""
        self.obs_scale = np.ones_like(observation, dtype=np.float32)
        # Capital scaled by initial capital
        capitals = observation[self.capital_slice]
        self.obs_scale[self.capital_slice] = np.where(capitals != 0, capitals, 1.0)

        # Variable costs scaled by their max
        var_costs = observation[self.variable_cost_slice]
        max_var = np.max(var_costs)
        self.obs_scale[self.variable_cost_slice] = max_var if max_var > 0 else 1.0

        # Fixed costs scaled by their max
        fixed_costs = observation[self.fixed_cost_slice]
        max_fixed = np.max(fixed_costs)
        self.obs_scale[self.fixed_cost_slice] = max_fixed if max_fixed > 0 else 1.0

        # Entry costs scaled by their max
        entry_costs = observation[self.entry_cost_slice]
        max_entry = np.max(entry_costs)
        self.obs_scale[self.entry_cost_slice] = max_entry if max_entry > 0 else 1.0

        # Demand intercepts scaled by their max
        intercepts = observation[self.demand_intercept_slice]
        max_intercept = np.max(intercepts)
        self.obs_scale[self.demand_intercept_slice] = max_intercept if max_intercept > 0 else 1.0

        # Demand slopes scaled by max absolute slope
        slopes = observation[self.demand_slope_slice]
        max_slope = np.max(np.abs(slopes))
        self.obs_scale[self.demand_slope_slice] = max_slope if max_slope > 0 else 1.0

        # Quantities scaled by their max
        quantities = observation[self.quantity_slice]
        max_quantity = np.max(quantities)
        self.obs_scale[self.quantity_slice] = max_quantity if max_quantity > 0 else 1.0

        # Prices scaled by their max
        prices = observation[self.price_slice]
        max_price = np.max(prices)
        self.obs_scale[self.price_slice] = max_price if max_price > 0 else 1.0

    def _normalize_observation(self, observation: np.ndarray) -> np.ndarray:
        return observation / self.obs_scale

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        observation = np.array(self.python_API.reset(), dtype=np.float32)
        if self.normalize_obs:
            self._compute_obs_scale(observation)
            observation = self._normalize_observation(observation)
        info = {}
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated = self.python_API.step(action)
        observation = np.array(observation, dtype=np.float32)
        if self.normalize_obs:
            observation = self._normalize_observation(observation)
        reward = float(reward)
        info = {}
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        self.python_API.close()


def make_env(config_path, seed: int, normalize_observations: bool = False) -> Callable[[], BusinessStrategyEnv]:
    def _init():
        _env = BusinessStrategyEnv(str(config_path), normalize_observations=normalize_observations)
        _env = Monitor(_env)
        _env.reset(seed=seed)
        return _env

    return _init


if __name__ == "__main__":
    from pathlib import Path
    import argparse
    import torch
    import stable_baselines3
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from stable_baselines3.common.vec_env import VecNormalize  # NEW: for reward normalization
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.evaluation import evaluate_policy

    parser = argparse.ArgumentParser(description="Train a PPO agent in the BusinessStrategyEnv.")
    base_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--config",
        type=Path,
        default=base_dir / "WorkingFiles" / "Config" / "default.json",
        help="Path to the simulator configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base_dir / "AgentFiles" / "Agent.zip",
        help="Path where the trained model will be saved.",
    )
    parser.add_argument(
        "--num_updates", type=int, default=500, help="Number of PPO update iterations."
    )
    parser.add_argument(
        "--num_envs", type=int, default=10, help="Number of parallel environments."
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="Use the MPS GPU if available instead of the CPU.",
    )
    # CHANGED: make this a normal on/off flag (default True)
    parser.add_argument(
        "--normalize_obs",
        action="store_false",
        help="Normalize observations to the [0,1] range inside the env.",
    )
    # NEW: reward normalization flag (default True to preserve old behavior)
    parser.add_argument(
        "--normalize_reward",
        action="store_false",
        help="Normalize rewards using VecNormalize (recommended).",
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
        help="Clipping range for the policy objective.",
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
        default=2048,
        help="Number of environment steps to run per update.",
    )
    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Mini-batch size used during PPO updates.",
    )
    parser.add_argument(
        "--eval-episodes",
        dest="eval_episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes to run after training completes (0 to disable).",
    )

    args = parser.parse_args()

    device = (
        torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if args.use_gpu
        else "cpu"
    )

    total_steps = args.n_steps * args.num_envs * args.num_updates

    # Build vectorized envs
    env_fns = [make_env(args.config, i, normalize_observations=args.normalize_obs) for i in range(args.num_envs)]
    env = SubprocVecEnv(env_fns)

    # Wrap with VecNormalize ONLY for reward normalization (avoid double obs normalization)
    if args.normalize_reward:
        # norm_obs=False because the env already supports its own observation normalization
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = stable_baselines3.PPO(
        "MlpPolicy",
        env,
        gamma=args.gamma,
        learning_rate=args.learning_rate, # Default is 0.0003
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        device=device,
        verbose=1,
    )

    model.learn(total_timesteps=total_steps)

    # Save model (and VecNormalize stats if used)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))

    # These statistics are only necessary if comparing rewards at eval time. Since we are just running agents through
    # simulations and plotting results (agnostic to however rewards were calculated during training), we don't need
    # to do anything with these stats. But we still save this code in case we decide to use them in the future.
    # if args.normalize_reward:
    #     # Save the running mean/var so you can recreate the same normalization at eval time
    #     env.save(args.output.parent / "vecnormalize.pkl")

    if args.eval_episodes > 0:
        eval_env = DummyVecEnv(
            [
                make_env(
                    args.config,
                    seed=args.num_envs + 12345 + i,
                    normalize_observations=args.normalize_obs,
                )
                for i in range(1)
            ]
        )
        try:
            episode_rewards, episode_lengths = evaluate_policy(
                model,
                eval_env,
                n_eval_episodes=args.eval_episodes,
                deterministic=True,
                return_episode_rewards=True,
            )
        finally:
            eval_env.close()

        rewards = [float(r) for r in episode_rewards]
        lengths = [int(l) for l in episode_lengths]

        evaluation_summary = {
            "num_episodes": len(rewards),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            "min_reward": float(np.min(rewards)) if rewards else 0.0,
            "max_reward": float(np.max(rewards)) if rewards else 0.0,
            "mean_length": float(np.mean(lengths)) if lengths else 0.0,
            "min_length": int(np.min(lengths)) if lengths else 0,
            "max_length": int(np.max(lengths)) if lengths else 0,
        }

        metrics_path = args.output.parent / "evaluation_metrics.json"
        metrics_path.write_text(json.dumps(evaluation_summary, indent=2))
        print(f"Saved evaluation metrics to {metrics_path}")
    else:
        print("Evaluation skipped because --eval-episodes was set to 0.")

    env.close()
