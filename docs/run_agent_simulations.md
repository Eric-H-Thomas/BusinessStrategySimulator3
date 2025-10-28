# Batch evaluation with `run_agent_simulations.py`

The `scripts/run_agent_simulations.py` helper evaluates a directory of trained agent scenarios by launching the compiled C++ simulator, preserving the outputs, and generating publication-ready plots. Use it when you already have trained policies and want to produce the summary figures without retraining.

## Prerequisites

* The simulator binary must be built (see the main README for build instructions). By default the script searches common build locations such as `cmake-build-release/BusinessStrategySimulator3`. Provide `--simulator-bin` to override the path if your executable lives elsewhere.
* Each scenario lives in its own subdirectory beneath a root directory. Inside every scenario directory you should have:
  * `config.json` – the simulator configuration that references `1000` simulation runs and any other scenario specifics.
  * `Agent.zip` – the trained agent checkpoint referenced by the configuration.

## Usage

Run the helper from the repository root:

```bash
python scripts/run_agent_simulations.py /path/to/scenario/root
```

Optional arguments:

* `--simulator-bin /custom/path/to/BusinessStrategySimulator3` – explicitly provide the simulator executable when it is not stored in the default build folders.

## Outputs

For each scenario directory the script creates an `outputs/` folder containing:

* `raw/MasterOutput.csv` and `raw/MarketOverlap.csv` – direct simulator exports.
* `simulation_outputs.zip` – an archive containing the two CSV files for easy distribution.
* `plots/` – a set of PNG figures (`avg_bankruptcy.png`, `cumulative_capital.png`, `performance_summary.png`). When the scenario folder name includes `VariousSophisticated`, the figures use the specialized "various sophisticated types" ordering.
* `summary.txt` – plain-text percent change summaries produced by the plotting utilities.

Existing `outputs/` directories are removed before running new simulations to ensure the files reflect the latest run.
