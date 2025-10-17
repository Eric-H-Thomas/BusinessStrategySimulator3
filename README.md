# BusinessStrategySimulator3

BusinessStrategySimulator3 is a research playground for studying market entry and exit strategies with reinforcement learning. The project couples a C++ simulator with a Gymnasium-compatible Python interface so agents can learn how firms should enter and leave markets to maximize capital.

## Table of Contents
- [Overview](#overview)
- [Quickstart](#quickstart)
  - [Build the simulator](#build-the-simulator)
  - [Train your first agent](#train-your-first-agent)
  - [Use a trained model](#use-a-trained-model)
- [Batch training helpers](#batch-training-helpers)
- [Project layout](#project-layout)
- [Further reading](#further-reading)

## Overview

Firms often expand into related product markets where they can reuse materials, skills, and infrastructure (collectively, *capabilities*). This simulator models market similarities and lets agents decide when to enter or exit markets with different overlaps in required capabilities. In experiments, RL agents outperform simple rule-based strategies in a stylized economic game, illustrating how AI tools can guide microeconomic strategy.

## Quickstart

### Build the simulator

1. Ensure submodules are fetched:
   ```bash
   git submodule update --init --recursive
   ```
2. Build the C++ simulator and Python bindings:
   ```bash
   cmake --build cmake-build-debug
   ```
   This produces the `simulator_module` Python extension used by the environment. If you are on a managed cluster, load CMake first (for example `module load cmake/<version>`).

### Train your first agent

Install Python dependencies such as `gymnasium`, `stable-baselines3`, `torch`, and `numpy`, then launch training:

```bash
python business_strategy_gym_env.py --config WorkingFiles/Config/default.json --output AgentFiles/Agent.zip
```

The script trains a PPO policy by default. Switch algorithms with `--algorithm`:

```bash
python business_strategy_gym_env.py --algorithm dqn --num_envs 1 --config WorkingFiles/Config/default.json
```

### Use a trained model

Load a saved checkpoint and request an action from the helper in `simulator.py`:

```python
from simulator import simulate

action = simulate("AgentFiles/Agent.zip", observation)
```

## Batch training helpers

* `scripts/train_ppo_for_config_batch.py` trains separate PPO policies for every `.json` configuration in a directory. Set `--config-dir` to the scenario folder and `--output-dir` to the destination for checkpoints.
* `scripts/submit_slurm_training_job.sh` generates SLURM job scripts that activate the correct environment, export `PYTHONPATH`, and launch `business_strategy_gym_env.py`. Use `--dry-run` to preview the generated script.
* Sweep utilities (`submit_slurm_ppo_sweep.py`, `submit_slurm_dqn_sweep.py`, `submit_slurm_a2c_sweep.py`, and `submit_all_sweeps.py`) expand search grids into per-run jobs and collect results. See [Hyperparameter sweep utilities](docs/hyperparameter_sweeps.md) for the full workflow.

## Project layout

```
BusinessStrategySimulator3/
├── Analysis/                # Notebooks and analysis assets
├── JSONReader/              # C++ JSON parsing utilities
├── WorkingFiles/            # Default configs, sweep outputs, and SLURM job artifacts
├── scripts/                 # Training, sweep, and cluster submission helpers
├── simulator.py             # Python helper for running the trained simulator module
└── business_strategy_gym_env.py  # Gymnasium environment entry point
```

## Further reading

* [Running on a compute cluster](docs/cluster.md)
* [Hyperparameter sweep utilities](docs/hyperparameter_sweeps.md)

The repository serves as a foundation for exploring how reinforcement learning can guide firms when entering or exiting markets.
