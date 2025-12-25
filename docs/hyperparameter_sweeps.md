# Hyperparameter Sweep Utilities

The `scripts` directory contains helpers for launching large reinforcement-learning sweeps locally or on SLURM clusters. Each script expands a search grid into training jobs and collects the results in dedicated folders.

## PPO Sweeps

Launch multiple PPO runs with `scripts/submit_slurm_ppo_sweep.py`.

1. **Inspect the search space.** The script defines rollout settings in `paired_rollout_settings` and remaining PPO hyperparameters in `hyperparameter_space`. Edit those lists to expand, reduce, or selectively comment out entries. Use `--max-runs` to sample only the first *n* combinations.
2. **Choose the simulator configuration.** By default the script copies `WorkingFiles/Config/default.json`, injects a unique `results_dir` per run, and rewrites any agent checkpoint paths to point at the sweep output directory. Override with `--config` when needed.
3. **Select the results directory.** Runs are stored inside the folder passed to `--output-dir` (default `WorkingFiles/Sweeps/ppo_slurm`). Each run contains:
   - `hyperparameters.json`: the exact PPO settings for that job.
   - `config.json`: the generated simulator configuration.
   - `Agent.zip`: the trained Stable-Baselines3 checkpoint.
   - Additional simulator outputs produced during evaluation, if any.
4. **Launch the sweep.** Provide both training flags and SLURM options (these scripts now submit a single SLURM job array). For example:
   ```bash
   python scripts/submit_slurm_ppo_sweep.py \
       --config WorkingFiles/Config/default.json \
       --num-envs 8 \
       --num-updates 400 \
       --partition compute \
       --gres gpu:1 \
       --array-max-concurrent 50 \
       --max-runs 10
   ```
5. **Monitor submissions.** The script prints queued runs, writes a manifest under the output directory, and submits a single array job. Use `--dry-run` to generate the array script without submitting, or append scheduler settings such as `--time`, `--account`, or repeated `--extra-sbatch "#SBATCH ..."` arguments.
6. **Collect results** after the jobs finish:
   ```bash
   python scripts/collect_ppo_sweep_results.py
   ```

## DQN Sweeps

`scripts/submit_slurm_dqn_sweep.py` mirrors the PPO helper but iterates over `hyperparameter_space` and `training_schedules` to explore DQN-specific options.

1. Confirm the search grid aligns with your experiment. Edit `hyperparameter_space` to adjust optimizer and exploration settings, and tweak `training_schedules` for different `train_freq`, `gradient_steps`, or `target_update_interval` values.
2. Provide a simulator configuration and output directory (default `WorkingFiles/Sweeps/dqn_slurm`). Each run folder contains `hyperparameters.json`, the generated `config.json`, and the resulting `Agent.zip` checkpoint.
3. Launch the sweep with the same SLURM options you would pass to `submit_slurm_training_array.sh`:
   ```bash
   python scripts/submit_slurm_dqn_sweep.py \
       --config WorkingFiles/Config/default.json \
       --num-envs 4 \
       --num-updates 500 \
       --partition compute \
       --mem 32G \
       --array-max-concurrent 50 \
       --max-runs 8
   ```
4. Collect the metrics after all jobs complete:
   ```bash
   python scripts/collect_dqn_sweep_results.py
   ```

## A2C Sweeps

Use `scripts/submit_slurm_a2c_sweep.py` to manage A2C sweeps.

1. Audit `hyperparameter_space` (learning rate, GAE λ, entropy/vf coefficients, etc.) and the paired `rollout_settings` entries that specify `n_steps` and `batch_size`.
2. Pick a simulator configuration and results directory (default `WorkingFiles/Sweeps/a2c_slurm`). Each run folder contains run metadata and the trained checkpoint.
3. Submit the sweep with the SLURM options you need. For example:
   ```bash
   python scripts/submit_slurm_a2c_sweep.py \
       --config WorkingFiles/Config/default.json \
       --num-envs 8 \
       --num-updates 400 \
       --partition compute \
       --gres gpu:1 \
       --array-max-concurrent 50 \
       --dry-run
   ```
4. Aggregate the results:
   ```bash
   python scripts/collect_a2c_sweep_results.py
   ```

## Launching All Sweeps

`scripts/submit_all_sweeps.py` orchestrates PPO, DQN, and A2C sweeps in a single command.

* `--common-args`: repeatable flag forwarded to every sweep.
* `--ppo-args`, `--dqn-args`, `--a2c-args`: repeatable flags passed to the corresponding helper only.
* `--skip-<algo>`: omit a specific sweep.
* `--dry-run`: append `--dry-run` to any sweep command that does not already include it.

Example command requesting 1000 updates, a 23-hour walltime limit, and 16G of memory per job:

```bash
python scripts/submit_all_sweeps.py \
    --common-args "--num-updates 1000" \
    --common-args "--time 23:00:00" \
    --common-args "--mem 16G" \
    --common-args "--array-max-concurrent 50"
```

The collector writes a `summary.csv` inside the sweep directory and, by default, removes generated scripts in `WorkingFiles/SlurmJobs`. Pass `--skip-cleanup` to keep them for auditing. Forward additional environment customizations (such as `--venv`, `--build-dir`, or `--use-gpu`) exactly as you would when calling `submit_slurm_training_array.sh` directly.
