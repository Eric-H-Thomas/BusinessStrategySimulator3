# BusinessStrategySimulator3

BusinessStrategySimulator3 is a research playground for studying market entry and exit strategies using reinforcement learning. The project couples a C++ simulator with a Gymnasium-compatible Python interface so agents can learn how firms should enter and leave markets to maximize capital. In our experiments, RL agents outperform simple rule-based strategies in a stylized economic game, demonstrating that AI tools can offer insight into microeconomic strategy.

## Overview

Firms often expand into related product markets where they can reuse materials, skills, and infrastructure (which we refer to, collectively, as *capabilities*). For example, Amazon leveraged its e-commerce expertise to enter cloud computing, while struggles in unrelated spaces like smartphones highlight the challenges of lacking required capabilities.

This simulator models market similarities and lets agents decide when to enter or exit markets with different overlaps in required capabilities. 

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

## Running on a compute cluster

When launching a single training job on a shared compute resource:

1. Ensure the simulator module is discoverable. Either copy the compiled extension into your project directory or add the build
   output to your Python path:
   ```
   export PYTHONPATH=/path/to/BusinessStrategySimulator3/cmake-build-debug:$PYTHONPATH
   ```
2. Activate the virtual environment used for experiments:
   ```
   source /path/to/BusinessStrategySimulator3/.venv/bin/activate
   ```
3. Choose the command-line flags that match your experiment. Available options are summarized below.
4. Launch training:
   ```
   python business_strategy_gym_env.py [flags]
   ```

### Training script flags

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--config` | Path | `WorkingFiles/Config/default.json` | Path to the simulator configuration file. |
| `--output` | Path | `AgentFiles/Agent.zip` | Location where the trained model checkpoint will be saved. |
| `--num_updates` | int | `500` | Number of PPO update iterations to perform. |
| `--n-envs` | int | `10` | Number of parallel environments to run during training. |
| `--use_gpu` | flag | _disabled_ | Use an available GPU (e.g., Apple MPS) instead of the CPU. |

Example cluster command:

```
python business_strategy_gym_env.py \
    --config /path/to/BusinessStrategySimulator3/WorkingFiles/Config/default.json \
    --output /path/to/BusinessStrategySimulator3/AgentFiles/Agent.zip \
    --num_updates 3 \
    --n-envs 5 \
    --use_gpu
```

## Using a trained model

The `simulator.py` helper loads a saved model and returns an action for a given observation:

```
from simulator import simulate
action = simulate("AgentFiles/Agent.zip", observation)
```

The repository serves as a foundation for exploring how reinforcement learning can guide firms when entering or exiting markets.
