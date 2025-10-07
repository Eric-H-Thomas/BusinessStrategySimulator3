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
   cmake --build cmake-build-debug
   ```
   This produces the `simulator_module` Python extension used by the environment.

   Note that if you're on the supercomputer, you might first have to make cmake available via
   ```
   module avail cmake (to check the available versions)
   module load cmake/[version number]
   ```

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
    --num_envs 5 \
    --use_gpu
```

### Submitting a SLURM training job

For recurring experiments on a SLURM-managed cluster, use the helper script in
`scripts/submit_slurm_training_job.sh`. The script generates an `sbatch` file,
fills in the required environment setup (virtual environment activation and
`PYTHONPATH` adjustments), and optionally submits it immediately. Run the
script from the repository root:

```
scripts/submit_slurm_training_job.sh \
    --partition compute \
    --account your-allocation \
    --gres gpu:1 \
    --time 06:00:00 \
    --output /path/to/output/Agent.zip
```

Key flags mirror the options of `business_strategy_gym_env.py` and SLURM. Use
`--dry-run` to generate the batch script without submitting it, or append
additional training arguments after `--`, for example to change the number of
updates:

```
scripts/submit_slurm_training_job.sh --dry-run -- --num_updates 1000
```

The generated scripts are stored in `WorkingFiles/SlurmJobs/` by default so you
can re-submit or audit past experiments.

## Hyperparameter sweeps

`scripts/submit_slurm_ppo_sweep.py` automates PPO sweeps by generating one
SLURM job per hyperparameter combination. The script wraps
`scripts/submit_slurm_training_job.sh`, so make sure you can launch a single
training job before scheduling a sweep.

### Step-by-step workflow

1. **Inspect (and optionally edit) the search space.** The sweep script defines
   two rollout settings in `paired_rollout_settings` and the remaining PPO
   hyperparameters in `hyperparameter_space`. Edit those lists if you want to
   expand or shrink the grid, or comment out entries while testing. You can also
   narrow the run count with `--max-runs` to sample the first *n* combinations
   without editing the file.
2. **Choose the simulator configuration.** By default the script copies the
   JSON in `WorkingFiles/Config/default.json`, injects the `results_dir` for each
   run, and rewrites any agent checkpoints to point at the sweep output
   directory. Pass a different file with `--config` if your experiment requires
   a custom scenario.
3. **Decide where results should live.** Each run gets its own folder inside the
   directory supplied by `--output-dir` (default:
   `WorkingFiles/Sweeps/ppo_slurm`). That folder will contain:
   - `hyperparameters.json`: the exact PPO settings for that job.
   - `config.json`: the generated simulator config handed to the training job.
   - `Agent.zip`: the trained model checkpoint saved by Stable-Baselines3.
   - `simulation_output/`: simulator-level CSV logs written during evaluation.
4. **Launch the sweep.** Provide whatever SLURM options you normally use when
   calling the single-job helper. The example below requests a GPU partition but
   otherwise relies on defaults:

   ```bash
   python scripts/submit_slurm_ppo_sweep.py \
       --config WorkingFiles/Config/default.json \
       --num-envs 8 \
       --num-updates 400 \
       --partition compute \
       --gres gpu:1 \
       --max-runs 10
   ```

5. **Monitor submissions.** The script prints the fully expanded command for
   each run and then invokes the SLURM helper. Use `--dry-run` to generate the
   sbatch scripts without submitting, or pass additional scheduler settings via
   dedicated flags such as `--time 04:00:00`, `--account`, `--qos`, or repeated
   `--extra-sbatch "#SBATCH --constraint=..."` options.

### After the sweep finishes

Aggregate the results once the jobs have completed:

```bash
python scripts/collect_ppo_sweep_results.py
```

The collector writes a `summary.csv` inside the sweep directory and, by default,
cleans out `WorkingFiles/SlurmJobs` so stale `sbatch` scripts do not pile up
between experiments. Pass `--skip-cleanup` if you want to keep those generated
job files around for auditing.

If you need per-run environment customization (e.g., pointing to a different
virtual environment or simulator build), forward the corresponding flags
(`--venv`, `--build-dir`, `--use-gpu`, etc.) exactly as you would when calling
`submit_slurm_training_job.sh` directly. The sweep script forwards those values
to every scheduled job.

## Using a trained model

The `simulator.py` helper loads a saved model and returns an action for a given observation:

```
from simulator import simulate
action = simulate("AgentFiles/Agent.zip", observation)
```

The repository serves as a foundation for exploring how reinforcement learning can guide firms when entering or exiting markets.
