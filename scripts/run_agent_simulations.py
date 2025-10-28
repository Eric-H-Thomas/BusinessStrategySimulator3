#!/usr/bin/env python3
"""Batch simulation runner for evaluating trained agents."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Iterable, Optional

# Ensure Matplotlib uses a non-interactive backend before importing plotting helpers.
os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parent.parent

# Import plotting utilities after configuring the backend.
sys.path.insert(0, str(REPO_ROOT))
from Analysis import business_strategy_plots  # type: ignore  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the compiled simulator for every agent configuration in a directory "
            "and generate analysis plots."
        )
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Directory containing per-agent subdirectories with config.json and Agent.zip.",
    )
    parser.add_argument(
        "--simulator-bin",
        type=Path,
        default=None,
        help=(
            "Path to the compiled BusinessStrategySimulator3 executable. If omitted, "
            "common build directories will be searched."
        ),
    )
    return parser.parse_args()


def discover_simulator_binary(explicit_path: Optional[Path]) -> Path:
    """Locate the compiled simulator executable."""

    candidates: Iterable[Path]
    if explicit_path is not None:
        candidates = [explicit_path]
    else:
        candidates = [
            REPO_ROOT / "cmake-build-release" / "BusinessStrategySimulator3",
            REPO_ROOT / "cmake-build-debug" / "BusinessStrategySimulator3",
            REPO_ROOT / "build" / "BusinessStrategySimulator3",
            REPO_ROOT / "BusinessStrategySimulator3",
        ]

    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    message = (
        "Unable to locate the simulator executable. Please provide --simulator-bin "
        "or build the project (e.g. cmake --build cmake-build-release)."
    )
    raise FileNotFoundError(message)


def prepare_config(
    config_path: Path, agent_path: Path, results_dir: Path
) -> dict:
    """Load and adapt a configuration for evaluation."""

    with config_path.open("r", encoding="utf-8") as fp:
        config = json.load(fp)

    simulation_params = config.setdefault("simulation_parameters", {})
    simulation_params["results_dir"] = str(results_dir)

    for ai_agent in config.get("ai_agents", []):
        ai_agent["path_to_agent"] = str(agent_path)

    return config


def _build_simulator_environment(sim_bin: Path) -> dict:
    """Return a subprocess environment that mirrors the active Python setup."""

    env = os.environ.copy()

    pythonpath_parts = [str(REPO_ROOT), str(sim_bin.parent)]
    try:
        pythonpath_parts.extend(site.getsitepackages())
    except AttributeError:
        # ``getsitepackages`` is not available in virtualenv's "no site" mode.
        pass

    try:
        user_site = site.getusersitepackages()
    except AttributeError:
        user_site = None
    if user_site:
        pythonpath_parts.append(user_site)

    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)

    pythonpath_parts = [part for part in pythonpath_parts if part]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_parts))

    env.setdefault("PYTHONHOME", sys.prefix)

    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        ld_library_parts = [libdir]
        existing_ld = env.get("LD_LIBRARY_PATH")
        if existing_ld:
            ld_library_parts.append(existing_ld)
        ld_library_parts = [part for part in ld_library_parts if part]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(ld_library_parts))

    return env


def run_simulator(sim_bin: Path, config_data: dict) -> None:
    """Execute the C++ simulator with a temporary configuration file."""

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_config = Path(tmpdir) / "config.json"
        temp_config.write_text(json.dumps(config_data, indent=2), encoding="utf-8")

        env = _build_simulator_environment(sim_bin)

        subprocess.run(
            [str(sim_bin), str(temp_config)],
            check=True,
            cwd=sim_bin.parent,
            env=env,
        )


def zip_outputs(master_output: Path, market_overlap: Path, destination: Path) -> None:
    """Bundle CSV outputs into a ZIP archive."""

    import zipfile

    with zipfile.ZipFile(destination, "w") as archive:
        archive.write(master_output, arcname="MasterOutput.csv")
        archive.write(market_overlap, arcname="MarketOverlap.csv")


def generate_plots(
    zip_path: Path,
    plots_dir: Path,
    summary_path: Path,
    various_sophisticated_types: bool,
) -> None:
    """Run analysis utilities and persist figures and summary statistics."""

    plots_dir.mkdir(parents=True, exist_ok=True)

    df = business_strategy_plots.load_data(zip_path, "MasterOutput.csv")
    if various_sophisticated_types:
        df = business_strategy_plots.sort_by_agent_type_various_sophisticated_agent_types(df)
        bankruptcy_fig = business_strategy_plots.avg_bankruptcy_various_sophisticated_agent_types(
            df, clear_previous=True
        )
        capital_fig = business_strategy_plots.plot_cumulative_capital_various_sophisticated_agent_types(
            df, clear_previous=False
        )
    else:
        df = business_strategy_plots.sort_by_agent_type(df)
        bankruptcy_fig = business_strategy_plots.avg_bankruptcy(df, clear_previous=True)
        capital_fig = business_strategy_plots.plot_cumulative_capital(df, clear_previous=False)

    performance_fig = business_strategy_plots.performance_summary_std_error(
        df, clear_previous=False
    )

    bankruptcy_fig.savefig(plots_dir / "avg_bankruptcy.png", dpi=300, bbox_inches="tight")
    capital_fig.savefig(plots_dir / "cumulative_capital.png", dpi=300, bbox_inches="tight")
    performance_fig.savefig(plots_dir / "performance_summary.png", dpi=300, bbox_inches="tight")

    percent_changes = business_strategy_plots.calculate_percent_change(df)
    normalized_changes = business_strategy_plots.normalize_percent_change(df)

    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("Percent change in capital by agent type\n")
        for agent_type, percent_change in percent_changes.items():
            summary_file.write(f"  {agent_type}: {percent_change:.2f}%\n")
        summary_file.write("\nRelative growth (normalized percent change)\n")
        for agent_type, norm_change in normalized_changes.items():
            summary_file.write(f"  {agent_type}: {norm_change:.2f}\n")

    business_strategy_plots.plt.close("all")


def main() -> None:
    args = parse_args()

    root_dir = args.root.resolve()
    if not root_dir.is_dir():
        raise NotADirectoryError(f"{root_dir} is not a valid directory")

    simulator_bin = discover_simulator_binary(args.simulator_bin)

    for scenario_dir in sorted(d for d in root_dir.iterdir() if d.is_dir()):
        config_path = scenario_dir / "config.json"
        agent_path = scenario_dir / "Agent.zip"

        if not config_path.is_file() or not agent_path.is_file():
            continue

        print(f"Running simulations for {scenario_dir.name}...")

        outputs_root = scenario_dir / "outputs"
        raw_output_dir = outputs_root / "raw"
        plots_dir = outputs_root / "plots"
        summary_path = outputs_root / "summary.txt"
        zip_path = outputs_root / "simulation_outputs.zip"

        if outputs_root.exists():
            shutil.rmtree(outputs_root)
        raw_output_dir.mkdir(parents=True)

        config_data = prepare_config(config_path, agent_path.resolve(), raw_output_dir.resolve())
        run_simulator(simulator_bin, config_data)

        master_output = raw_output_dir / "MasterOutput.csv"
        market_overlap = raw_output_dir / "MarketOverlap.csv"
        if not master_output.is_file() or not market_overlap.is_file():
            raise FileNotFoundError(
                f"Simulator outputs missing for {scenario_dir}: expected MasterOutput.csv and MarketOverlap.csv"
            )

        zip_outputs(master_output, market_overlap, zip_path)
        various_sophisticated_types = bool(
            re.search(r"VariousSophisticated", scenario_dir.name, flags=re.IGNORECASE)
        )

        generate_plots(
            zip_path,
            plots_dir,
            summary_path,
            various_sophisticated_types,
        )

        print(f"Completed analysis for {scenario_dir.name}. Outputs saved to {outputs_root}.")


if __name__ == "__main__":
    main()
