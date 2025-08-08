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
    convertedObs = np.array(obs)
    action, _hidden = model.predict(convertedObs)
    return action
