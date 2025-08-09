from stable_baselines3 import PPO
import numpy as np
import os

# Global variable to store the model
model = None


def simulate(path, obs: tuple):
    global model
    if model is None:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model file not found at: {path}")
        model = PPO.load(path)
    converted_obs = np.array(obs, dtype=np.float32)
    action, _hidden = model.predict(converted_obs, deterministic=True)
    return int(action)
