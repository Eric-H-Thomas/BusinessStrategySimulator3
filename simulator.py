from stable_baselines3 import PPO
import numpy as np
import os
import json
from typing import Optional

# Globals for the loaded model and normalization parameters
model = None
_obs_scale = None
_slices_initialized = False
_num_agents = None
_num_markets = None
_num_possible_capabilities = None


def _init_slices(observation: np.ndarray):
    """Infer slice indices for each observation component."""
    global capital_slice, overlap_slice, variable_cost_slice, fixed_cost_slice
    global market_portfolio_slice, entry_cost_slice, demand_intercept_slice
    global demand_slope_slice, quantity_slice, price_slice, _slices_initialized
    global shareability_slice
    global _num_agents, _num_markets, _num_possible_capabilities

    if _slices_initialized:
        return

    if _num_agents is None or _num_markets is None or _num_possible_capabilities is None:
        raise ValueError("Agent, market, and capability counts not initialized")

    expected_len = (
        _num_agents
        + _num_markets ** 2
        + _num_possible_capabilities
        + 6 * _num_agents * _num_markets
        + 2 * _num_markets
    )
    if observation.shape[0] != expected_len:
        raise ValueError(
            f"Observation length {observation.shape[0]} does not match expected {expected_len}"
        )

    start = 0
    capital_slice = slice(start, start + _num_agents)
    start += _num_agents
    overlap_slice = slice(start, start + _num_markets ** 2)
    start += _num_markets ** 2
    shareability_slice = slice(start, start + _num_possible_capabilities)
    start += _num_possible_capabilities
    variable_cost_slice = slice(start, start + _num_agents * _num_markets)
    start += _num_agents * _num_markets
    fixed_cost_slice = slice(start, start + _num_agents * _num_markets)
    start += _num_agents * _num_markets
    market_portfolio_slice = slice(start, start + _num_agents * _num_markets)
    start += _num_agents * _num_markets
    entry_cost_slice = slice(start, start + _num_agents * _num_markets)
    start += _num_agents * _num_markets
    demand_intercept_slice = slice(start, start + _num_markets)
    start += _num_markets
    demand_slope_slice = slice(start, start + _num_markets)
    start += _num_markets
    quantity_slice = slice(start, start + _num_agents * _num_markets)
    start += _num_agents * _num_markets
    price_slice = slice(start, start + _num_agents * _num_markets)

    _slices_initialized = True


def _compute_obs_scale(observation: np.ndarray) -> None:
    """Compute per-component scaling factors matching training-time normalization."""
    global _obs_scale
    _obs_scale = np.ones_like(observation, dtype=np.float32)

    capitals = observation[capital_slice]
    _obs_scale[capital_slice] = np.where(capitals != 0, capitals, 1.0)

    var_costs = observation[variable_cost_slice]
    max_var = np.max(var_costs)
    _obs_scale[variable_cost_slice] = max_var if max_var > 0 else 1.0

    fixed_costs = observation[fixed_cost_slice]
    max_fixed = np.max(fixed_costs)
    _obs_scale[fixed_cost_slice] = max_fixed if max_fixed > 0 else 1.0

    entry_costs = observation[entry_cost_slice]
    max_entry = np.max(entry_costs)
    _obs_scale[entry_cost_slice] = max_entry if max_entry > 0 else 1.0

    intercepts = observation[demand_intercept_slice]
    max_intercept = np.max(intercepts)
    _obs_scale[demand_intercept_slice] = max_intercept if max_intercept > 0 else 1.0

    slopes = observation[demand_slope_slice]
    max_slope = np.max(np.abs(slopes))
    _obs_scale[demand_slope_slice] = max_slope if max_slope > 0 else 1.0

    quantities = observation[quantity_slice]
    max_quantity = np.max(quantities)
    _obs_scale[quantity_slice] = max_quantity if max_quantity > 0 else 1.0

    prices = observation[price_slice]
    max_price = np.max(prices)
    _obs_scale[price_slice] = max_price if max_price > 0 else 1.0


def reset_observation_normalization(config_path: Optional[str] = None) -> None:
    """Reset normalization so it will be recomputed on next call to simulate."""
    global _obs_scale, _slices_initialized, _num_agents, _num_markets
    global _num_possible_capabilities
    _obs_scale = None
    _slices_initialized = False
    _num_agents = None
    _num_markets = None
    _num_possible_capabilities = None
    if config_path is not None:
        with open(config_path, "r") as f:
            cfg = json.load(f)
        _num_agents = len(cfg.get("control_agents", [])) + len(cfg.get("ai_agents", []))
        econ = cfg.get("default_economy_parameters", {})
        _num_markets = econ.get("total_markets")
        _num_possible_capabilities = econ.get("possible_capabilities")


def simulate(path, obs: tuple):
    global model, _obs_scale

    if model is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        model = PPO.load(path)

    converted_obs = np.array(obs, dtype=np.float32)

    if _obs_scale is None:
        _init_slices(converted_obs)
        _compute_obs_scale(converted_obs)

    normalized_obs = converted_obs / _obs_scale
    action, _hidden = model.predict(normalized_obs, deterministic=True)
    return int(action)
