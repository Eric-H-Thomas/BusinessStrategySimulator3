# Running on a Compute Cluster

This guide summarizes how to use the training scripts on shared compute resources such as SLURM clusters.

## Preparing the Environment

1. **Expose the simulator module** so Python can import the compiled extension:
   ```bash
   export PYTHONPATH=/path/to/BusinessStrategySimulator3/cmake-build-debug:$PYTHONPATH
   ```
2. **Activate the experiment environment** (for example, a virtual environment):
   ```bash
   source /path/to/BusinessStrategySimulator3/.venv/bin/activate
   ```
3. **Choose the CLI flags** that match your experiment, as summarized below.

## Training Script Flags

| Flag | Type | Default | Description |
| --- | --- | --- | --- |
| `--config` | Path | `WorkingFiles/Config/default.json` | Path to the simulator configuration file. |
| `--output` | Path | `AgentFiles/Agent.zip` | Location where the trained model checkpoint will be saved. |
| `--algorithm` | str | `ppo` | Algorithm to train (`ppo`, `dqn`, or `a2c`). |
| `--num_updates` | int | `500` | Number of update iterations (on-policy algorithms multiply this by `--n_steps` and `--num_envs`; DQN multiplies it by `--train-freq`). |
| `--num_envs` | int | `10` | Number of parallel environments to run during training. |
| `--use_gpu` | flag | _disabled_ | Use an available GPU (e.g., Apple MPS) instead of the CPU. |

### Example Command

```bash
python business_strategy_gym_env.py \
    --config /path/to/BusinessStrategySimulator3/WorkingFiles/Config/default.json \
    --output /path/to/BusinessStrategySimulator3/AgentFiles/Agent.zip \
    --num_updates 3 \
    --num_envs 5 \
    --use_gpu
```

## SLURM Helpers

Use `scripts/submit_slurm_training_job.sh` to create or submit SLURM jobs. The script fills in environment setup (virtual environment activation and `PYTHONPATH` adjustments) and accepts both training flags and scheduler options.

```bash
scripts/submit_slurm_training_job.sh \
    --partition compute \
    --account your-allocation \
    --gres gpu:1 \
    --time 06:00:00 \
    --output /path/to/output/Agent.zip
```

Key flags mirror `business_strategy_gym_env.py` plus SLURM options. Use `--dry-run` to generate the batch script without submitting, or append extra training arguments after `--` (for example `--num_updates 1000`). Generated scripts are stored in `WorkingFiles/SlurmJobs/` by default.

## Batch Sweeps on SLURM

For PPO, DQN, and A2C sweeps on a cluster, refer to [docs/hyperparameter_sweeps.md](hyperparameter_sweeps.md) for detailed usage patterns.
