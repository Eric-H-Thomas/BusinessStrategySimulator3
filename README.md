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

Install Python dependencies (e.g., `gymnasium`, `stable-baselines3`, `torch`, and `numpy`). Then train an agent with:

```
python business_strategy_gym_env.py --config WorkingFiles/Config/default.json --output AgentFiles/Agent.zip
```

By default the script trains a PPO policy. Switch algorithms with `--algorithm`:

```bash
python business_strategy_gym_env.py --algorithm dqn --num_envs 1 --config WorkingFiles/Config/default.json
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
| `--algorithm` | str | `ppo` | Algorithm to train (`ppo`, `dqn`, or `a2c`). |
| `--num_updates` | int | `500` | Number of update iterations (on-policy algorithms multiply this by `--n_steps` and `--num_envs`; DQN multiplies it by `--train-freq`). |
| `--num_envs` | int | `10` | Number of parallel environments to run during training. |
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
   - Any simulator outputs written during evaluation (if produced).
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

### DQN sweeps

`scripts/submit_slurm_dqn_sweep.py` mirrors the PPO workflow but targets the
DQN implementation. It iterates over the cartesian product of the
`hyperparameter_space` dictionary and the `training_schedules` tuples defined at
the bottom of the script.

The usage pattern matches the PPO helper:

1. Confirm the search grid aligns with your experiment. Edit
   `hyperparameter_space` to adjust optimizer and exploration settings, and
   tweak `training_schedules` if you want different `train_freq`,
   `gradient_steps`, or `target_update_interval` values.
2. Provide a simulator configuration and output directory. Each run gets its own
   folder (`WorkingFiles/Sweeps/dqn_slurm` by default) populated with
   `hyperparameters.json`, the generated `config.json`, and the resulting
   `Agent.zip` checkpoint.
3. Launch the sweep with the same SLURM options you would pass to
   `submit_slurm_training_job.sh`:

   ```bash
   python scripts/submit_slurm_dqn_sweep.py \
       --config WorkingFiles/Config/default.json \
       --num-envs 4 \
       --num-updates 500 \
       --partition compute \
       --mem 32G \
       --max-runs 8
   ```

Collect the metrics after all jobs complete with:

```bash
python scripts/collect_dqn_sweep_results.py
```

### A2C sweeps

Use `scripts/submit_slurm_a2c_sweep.py` to sweep the A2C hyperparameters. The
script exposes the same command-line flags as the PPO and DQN helpers while
iterating over the `hyperparameter_space` and `rollout_settings` combinations
near the bottom of the file.

1. Audit `hyperparameter_space` (learning rate, GAE λ, entropy/vf coefficients,
   etc.) and the paired `rollout_settings` that specify `n_steps` and
   `batch_size`. Adjust the lists to expand or narrow the grid.
2. Pick a simulator configuration and results directory. By default the script
   writes one folder per run under `WorkingFiles/Sweeps/a2c_slurm` populated with
   the run metadata and trained checkpoint.
3. Submit the sweep with the SLURM options you need. For example:

   ```bash
   python scripts/submit_slurm_a2c_sweep.py \
       --config WorkingFiles/Config/default.json \
       --num-envs 8 \
       --num-updates 400 \
       --partition compute \
       --gres gpu:1 \
       --dry-run
   ```

Aggregate the results with:

```bash
python scripts/collect_a2c_sweep_results.py
```

### Submitting all sweeps in one command

When you want to schedule PPO, DQN, and A2C sweeps back-to-back, use
`scripts/submit_all_sweeps.py`. The wrapper sequentially launches the individual
helpers and accepts both shared and algorithm-specific flags:

* `--common-args`: repeated flag that forwards arguments to every sweep.
* `--ppo-args`, `--dqn-args`, `--a2c-args`: repeated flags that inject options
  for the corresponding script only.
* `--skip-<algo>`: omit a particular sweep.
* `--dry-run`: append `--dry-run` to any sweep command that does not already
  include it.

Example command: train every agent for 1000 updates, request a 23-hour walltime
limit, and cap memory at 16G for each job:

```bash
python scripts/submit_all_sweeps.py \
    --common-args "--num-updates 1000" \
    --common-args "--time 23:00:00" \
    --common-args "--mem 16G"
```

You can add algorithm-specific overrides by repeating `--ppo-args`,
`--dqn-args`, or `--a2c-args` with quoted argument groups, or skip a sweep (for
example `--skip-dqn`) when only a subset of algorithms should run.

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
