# BusinessStrategySimulator3

BusinessStrategySimulator3 is a research playground for studying market entry and exit strategies using reinforcement learning. The project couples a C++ simulator with a Gymnasium-compatible Python interface so agents can learn how firms should enter and leave markets to maximize capital. In our experiments, RL agents outperform simple rule-based strategies in a stylized economic game, demonstrating that AI tools can offer insight into microeconomic strategy.

## Overview

Firms often expand into related markets where they can reuse materials, skills, and infrastructure. This simulator captures those relationships and lets agents decide when to enter or exit markets with different overlaps in required competencies. For example, Amazon leveraged its e-commerce expertise to enter cloud computing, while struggles in unrelated spaces like smartphones highlight the challenges of lacking required capabilities.

## Building the simulator

1. Ensure submodules are fetched:
   ```
   git submodule update --init --recursive
   ```
2. Build the C++ simulator and Python bindings:
   ```
   cmake -S . -B build
   cmake --build build
   ```
   This produces the `simulator_module` Python extension used by the environment.

## Training an agent

Install Python dependencies (e.g., `gymnasium`, `stable-baselines3`, `torch`, and `numpy`). Then train a PPO agent with:

```
python business_strategy_gym_env.py --config WorkingFiles/Config/default.json --output AgentFiles/Agent.zip
```

## Using a trained model

The `simulator.py` helper loads a saved model and returns an action for a given observation:

```
from simulator import simulate
action = simulate("AgentFiles/Agent.zip", observation)
```

The repository serves as a foundation for exploring how reinforcement learning can guide firms when entering or exiting markets.
