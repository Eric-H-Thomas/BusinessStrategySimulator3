"""Plotting utilities for analyzing business strategy simulation results."""

import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns
import re
import zipfile
from pathlib import Path


def load_data(zip_path: Path, csv_name: str) -> pd.DataFrame:
    """Load a CSV file from within a ZIP archive.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP archive containing the CSV.
    csv_name : str
        Name of the CSV file within the archive.

    Returns
    -------
    pd.DataFrame
        Data loaded from the CSV file.
    """

    # ``zipfile.ZipFile`` allows reading a member file directly from the
    # archive without extracting it to disk.
    with zipfile.ZipFile(zip_path) as zip_ref:
        with zip_ref.open(csv_name) as csv_file:
            return pd.read_csv(csv_file)


SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def format_subscript(index: int) -> str:
    """Convert an integer into a unicode subscript string."""
    return str(index).translate(SUBSCRIPT_DIGITS)


def base_agent_label(label: str) -> str:
    """Return the base agent label with trailing subscripts stripped."""
    return re.sub(r"[₀-₉]+$", "", label)


def add_agent_type_subscripts(df: pd.DataFrame) -> pd.DataFrame:
    """Append per-type subscripts to agent labels so each agent is unique."""
    unique_pairs = df[["Agent Type", "Firm"]].drop_duplicates()
    unique_pairs["Agent Index"] = unique_pairs.groupby("Agent Type").cumcount()
    unique_pairs["Agent Label"] = unique_pairs["Agent Type"] + unique_pairs[
        "Agent Index"
    ].map(format_subscript)

    df = df.merge(unique_pairs, on=["Agent Type", "Firm"], how="left")
    df["Agent Type"] = df["Agent Label"]
    return df.drop(columns=["Agent Index", "Agent Label"])


def has_agent_subscripts(df: pd.DataFrame) -> bool:
    """Return True if any agent labels include unicode subscripts."""
    return df["Agent Type"].astype(str).str.contains(r"[₀-₉]+$").any()


def aggregate_capital_by_firm(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate capital across markets while preserving per-firm values."""
    return (
        df.groupby(["Sim", "Step", "Firm", "Agent Type"], as_index=False)["Capital"]
        .mean()
    )


def assert_single_simulation(df: pd.DataFrame, context: str) -> None:
    """Raise if data contains multiple simulations where a single one is required."""
    if "Sim" not in df.columns:
        return
    simulation_count = df["Sim"].nunique()
    if simulation_count > 1:
        raise ValueError(
            f"{context} expects a single simulation when agent subscripts are present, "
            f"but found {simulation_count} simulations."
        )


def sort_by_agent_type(df: pd.DataFrame) -> pd.DataFrame:
    """Simplify agent type labels to broad categories.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing an ``Agent Type`` column.

    Returns
    -------
    pd.DataFrame
        Dataframe with the ``Agent Type`` values replaced with
        ``AI``, ``Naive`` or ``Sophisticated``. Both the verbose names
        (e.g. ``HighestOverlap``) and abbreviated forms like ``1S`` are
        recognised.
    """
    agent_types = sorted(set(df["Agent Type"].tolist()))
    for agent in agent_types:
        # The agent type string encodes the strategy. Use pattern
        # matching to map both verbose and abbreviated names onto concise
        # labels used in plots.
        sophisticated = re.search(r"HighestOverlap", agent) or re.fullmatch(r"\d+S", agent)
        naive = re.search(r"All", agent) or re.fullmatch(r"\d+N", agent)
        ai = re.search(r"StableBaselines3", agent) or re.fullmatch(r"\d+AI", agent)

        if sophisticated:
            df.replace({"Agent Type": agent}, "Sophisticated", inplace=True)
        elif naive:
            df.replace({"Agent Type": agent}, "Naive", inplace=True)
        elif ai:
            df.replace({"Agent Type": agent}, "AI", inplace=True)

    return df


def sort_by_agent_type_various_sophisticated_agent_types(df: pd.DataFrame) -> pd.DataFrame:
    """Map verbose agent type names to simplified categories with suffixes.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing an ``Agent Type`` column.

    Returns
    -------
    pd.DataFrame
        Dataframe with agent types replaced by ``AI``, ``Naive`` and
        ``Sophisticated A-D``. Supports both verbose and abbreviated
        agent identifiers (e.g. ``HighestOverlap`` or ``1S``).
    """
    agent_types = sorted(set(df["Agent Type"].tolist()))
    sophisticated_suffixes = ["A", "B", "C", "D"]
    sophisticated_idx = 0
    abbreviated_mapping = {
        "0S": "Sophisticated A",
        "1S": "Sophisticated B",
        "2S": "Sophisticated C",
        "3S": "Sophisticated D",
        "4AI": "AI",
    }
    for agent in agent_types:
        # Similar pattern matching to ``sort_by_agent_type`` but we assign
        # sequential suffixes to distinguish the sophisticated variants.
        if agent in abbreviated_mapping:
            df.replace({"Agent Type": agent}, abbreviated_mapping[agent], inplace=True)
            continue

        sophisticated = re.search(r"HighestOverlap", agent) or re.fullmatch(r"\d+S", agent)
        naive = re.search(r"All", agent) or re.fullmatch(r"\d+N", agent)
        ai = re.search(r"StableBaselines3", agent) or re.fullmatch(r"\d+AI", agent)

        if sophisticated:
            df.replace(
                {"Agent Type": agent},
                "Sophisticated " + sophisticated_suffixes[sophisticated_idx],
                inplace=True,
            )
            sophisticated_idx += 1
        elif naive:
            df.replace({"Agent Type": agent}, "Naive", inplace=True)
        elif ai:
            df.replace({"Agent Type": agent}, "AI", inplace=True)

    return df


def bankruptcy_rate_by_agent_type(df: pd.DataFrame) -> pd.Series:
    """Compute bankruptcy rates using the final timestep for each simulation."""
    # Aggregate across markets to avoid counting duplicate capital values.
    capital_by_step = (
        df.groupby(["Sim", "Step", "Firm", "Agent Type"], as_index=False)["Capital"]
        .mean()
    )
    # Select the final timestep for each simulation/firm/agent type combo.
    final_rows = capital_by_step.loc[
        capital_by_step.groupby(["Sim", "Firm", "Agent Type"])["Step"].idxmax()
    ]
    bankruptcy_mask = final_rows["Capital"] == -1e-09
    return (
        final_rows.assign(Bankrupt=bankruptcy_mask)
        .groupby("Agent Type")["Bankrupt"]
        .mean()
    )


def avg_bankruptcy(df: pd.DataFrame, clear_previous: bool = True) -> plt.Figure:
    """Plot the percentage of simulations ending in bankruptcy for each agent type."""
    if clear_previous:
        plt.close("all")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel("Percentage of Simulations Ending in Bankruptcy")
    ax.set_ylim(0, 100)
    type_to_color = {
        "AI": "#79AEA3",
        "Naive": "#9E4770",
        "Sophisticated": "#1446A0",
    }

    agent_types = np.array(df["Agent Type"].unique())
    bankruptcy_rates = bankruptcy_rate_by_agent_type(df)
    avg_bankruptcies_per_agent_type = bankruptcy_rates.reindex(
        agent_types, fill_value=0
    ).to_numpy()

    colors = [
        type_to_color.get(base_agent_label(agent_type), "#999999")
        for agent_type in agent_types
    ]  # Default to gray if not found
    avg_bankruptcies_per_agent_type = np.array(avg_bankruptcies_per_agent_type)
    ax.bar(agent_types, avg_bankruptcies_per_agent_type * 100, color=colors)

    plt.tight_layout()
    return fig


def avg_bankruptcy_various_sophisticated_agent_types(
    df: pd.DataFrame, clear_previous: bool = True
) -> plt.Figure:
    """Plot bankruptcy frequency with additional Sophisticated subtypes."""
    if clear_previous:
        plt.close("all")

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel("Percentage of Simulations Ending in Bankruptcy")
    ax.set_ylim(0, 100)
    type_to_color = {
        "AI": "#79AEA3",
        "Naive": "#9E4770",
        "Sophisticated A": "#1446A0",
        "Sophisticated B": "#2166C4",
        "Sophisticated C": "#0094C8",
        "Sophisticated D": "#00B894",
    }

    agent_types = np.array(df["Agent Type"].unique())
    bankruptcy_rates = bankruptcy_rate_by_agent_type(df)
    avg_bankruptcies_per_agent_type = bankruptcy_rates.reindex(
        agent_types, fill_value=0
    ).to_numpy()

    colors = [
        type_to_color.get(base_agent_label(agent_type), "#999999")
        for agent_type in agent_types
    ]  # Default to gray if not found
    avg_bankruptcies_per_agent_type = np.array(avg_bankruptcies_per_agent_type)
    ax.bar(agent_types, avg_bankruptcies_per_agent_type * 100, color=colors)

    plt.tight_layout()
    return fig


def avg_bankruptcy_combined(dfs, labels, clear_previous=True):
    """Compare bankruptcy rates across multiple datasets on a single plot.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        List of dataframes to compare.
    labels : list[str]
        Labels for each dataframe.
    clear_previous : bool, optional
        If ``True`` (default), closes previously open figures.
    """
    if clear_previous:
        plt.close("all")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_ylim(0, 100)
    colors = ["#79AEA3", "#9E4770", "#1446A0", "#00FFFF", "r", "b", "g"]
    type_to_color = {"AI": colors[0], "Naive": colors[1], "Sophisticated": colors[2]}

    agent_types = set()
    for df in dfs:
        agent_types.update(df["Agent Type"].unique())

    agent_types = sorted(agent_types, reverse=True)
    total_clusters = len(dfs)
    x = np.arange(total_clusters)
    # Spread the bars for different agent types within each cluster so that
    # clusters (datasets) occupy 80% of the available horizontal space.
    width = 0.8 / len(agent_types)

    for i, (df, label) in enumerate(zip(dfs, labels)):
        bankruptcy_rates = bankruptcy_rate_by_agent_type(df)
        avg_bankruptcies_per_agent_type = bankruptcy_rates.reindex(
            agent_types, fill_value=0
        ).to_numpy()

        for j, agent_type in enumerate(agent_types):
            # Offset each agent type within the cluster corresponding to the
            # dataset ``i``.
            ax.bar(
                x[i] + (j - len(agent_types) / 2) * width + width / 2,
                avg_bankruptcies_per_agent_type[j] * 100,
                width=width,
                label=f"{agent_type}" if i == 0 else "",
                color=type_to_color.get(base_agent_label(agent_type), "#999999"),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.legend(title="Agent Types")

    plt.tight_layout()
    return fig


def aggregate_data_with_std(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate market data and compute mean and standard deviation.

    The raw simulation output contains one row per market. This function first
    sums across markets within each simulation and then calculates the mean and
    standard deviation across simulations for each step and agent type.
    """

    # Sum across markets within each simulation, step and firm
    market_sum = (
        df.groupby(["Sim", "Step", "Firm", "Agent Type"])
        .agg(
            {
                "Rev": "sum",
                "Fix Cost": "sum",
                "Var Cost": "sum",
                # Capital is duplicated across markets, so average instead of sum
                "Capital": "mean",
            }
        )
        .reset_index()
    )

    # Calculate mean, standard deviation and count across simulations for each
    # step and agent type. ``Rev`` is arbitrarily used for the ``count`` since
    # all metrics share the same number of observations.
    sim_stats = (
        market_sum.groupby(["Step", "Agent Type"])
        .agg(
            {
                "Rev": ["mean", "std", "count"],
                "Fix Cost": ["mean", "std"],
                "Var Cost": ["mean", "std"],
                "Capital": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten the MultiIndex columns produced by aggregation
    sim_stats.columns = ["_".join(col).strip("_") for col in sim_stats.columns]

    # ``Rev_count`` holds the number of simulations (or simulation-firm pairs)
    # contributing to each mean value. Rename for clarity and return.
    sim_stats.rename(columns={"Rev_count": "Count"}, inplace=True)

    return sim_stats

def performance_summary_std_error(
    df: pd.DataFrame, clear_previous: bool = True, show_std_error: bool = True
) -> plt.Figure:
    """Plot revenue, cost and profit with shaded standard error regions."""
    if clear_previous:
        plt.close("all")

    # Aggregate the raw data and include counts needed for the standard error.
    df = aggregate_data_with_std(df)

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    metrics = ["Profit", "Rev", "Total Cost", "Fix Cost", "Var Cost"]
    colors = ["#79AEA3", "#9E4770", "#1446A0", "#00FFFF", "r", "b", "g"]
    type_to_color = {"AI": colors[0], "Naive": colors[1], "Sophisticated": colors[2]}
    agent_types = sorted(df["Agent Type"].unique())

    max_step = df["Step"].max()
    all_macrostep_ticks = list(range(0, max_step + 10, 10))
    label_macrostep_ticks = list(range(0, max_step + 50, 50))

    # Proxy handles for legend entries describing shaded areas
    shaded_region_patches = [
        patches.Patch(
            color=type_to_color.get(base_agent_label(agent_type), "#999999"),
            alpha=0.2,
            label=f"{agent_type}",
        )
        for agent_type in agent_types
    ]

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        ax.clear()

        for agent_type in agent_types:
            agent_type_df = df[df["Agent Type"] == agent_type]

            if metric == "Profit":
                mean_values = agent_type_df["Rev_mean"] - (
                    agent_type_df["Fix Cost_mean"] + agent_type_df["Var Cost_mean"]
                )
                std_values = np.sqrt(
                    agent_type_df["Rev_std"] ** 2
                    + (
                        agent_type_df["Fix Cost_std"] ** 2
                        + agent_type_df["Var Cost_std"] ** 2
                    )
                )
            elif metric == "Total Cost":
                mean_values = (
                    agent_type_df["Fix Cost_mean"] + agent_type_df["Var Cost_mean"]
                )
                std_values = np.sqrt(
                    agent_type_df["Fix Cost_std"] ** 2
                    + agent_type_df["Var Cost_std"] ** 2
                )
            else:
                mean_values = agent_type_df[f"{metric}_mean"]
                std_values = agent_type_df[f"{metric}_std"]

            ax.plot(
                agent_type_df["Step"],
                mean_values,
                color=type_to_color.get(base_agent_label(agent_type), "#999999"),
                label=f"{agent_type}" if idx == 0 else "",
                linewidth=1,
            )

            # Standard error for each step is ``std / sqrt(n)`` where ``n`` is
            # the number of simulations contributing to that point. The
            # ``Count`` column returned by ``aggregate_data_with_std`` contains
            # this value for every step.
            if show_std_error:
                ax.fill_between(
                    agent_type_df["Step"],
                    np.clip(
                        mean_values - (std_values / np.sqrt(agent_type_df["Count"])),
                        0,
                        None,
                    ),
                    mean_values + (std_values / np.sqrt(agent_type_df["Count"])),
                    color=type_to_color.get(base_agent_label(agent_type), "#999999"),
                    alpha=0.2,
                )

        ax.set_ylabel(metric)
        ax.grid(True, linestyle="--", alpha=0.7)

        if idx == 0:
            ax.legend(loc="upper left", title="Agent Type", fontsize="x-small")

        ax.set_xticks(all_macrostep_ticks)
        labels = []
        for tick in all_macrostep_ticks:
            labels.append(str(int(tick / 10)) if tick in label_macrostep_ticks else "")
        ax.set_xticklabels(labels)
        ax.tick_params(axis="x", which="major", length=4, labelbottom=True)

    if show_std_error:
        fig.legend(
            handles=shaded_region_patches,
            loc="upper center",
            ncol=len(shaded_region_patches),
            bbox_to_anchor=(0.5, 0.95),
            frameon=False,
            title="Shaded Regions: Standard Error",
        )

    for ax in axes:
        ax.set_xlabel("Macrostep")

    plt.tight_layout(rect=[0, 0, 1, 0.93 if show_std_error else 1])
    return fig

"""# Cumulative Average Capital Plots (Nate)"""


def plot_cumulative_capital(
    df: pd.DataFrame, clear_previous: bool = True
) -> plt.Figure:
    """Plot cumulative average capital for each agent type."""
    if clear_previous:
        plt.close("all")

    if has_agent_subscripts(df):
        assert_single_simulation(df, "plot_cumulative_capital")
        df = aggregate_capital_by_firm(df)
    else:
        df = aggregate_data_with_std(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel("Average Capital")
    ax.set_xlabel("Micro Time Step")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    colors = ["#79AEA3", "#9E4770", "#1446A0", "#00FFFF", "r", "b", "g"]
    type_to_color = {"AI": colors[0], "Naive": colors[1], "Sophisticated": colors[2]}
    agent_types = sorted(df["Agent Type"].unique())
    for agent_type in agent_types:
        agent_type_df = df[df["Agent Type"] == agent_type]
        # Plot average capital trajectory for the current agent type.
        if "Capital_mean" in agent_type_df:
            capital_series = agent_type_df["Capital_mean"]
        else:
            capital_series = agent_type_df["Capital"]
        ax.plot(
            agent_type_df["Step"],
            capital_series,
            color=type_to_color.get(base_agent_label(agent_type), "#999999"),
            linewidth=2,
            label=f"{agent_type}",
        )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))

    max_step = df["Step"].max()
    all_ticks = list(range(0, max_step + 10, 10))
    label_ticks = list(range(0, max_step + 50, 50))

    ax.set_xticks(all_ticks)
    labels = [str(int(tick / 10)) if tick in label_ticks else "" for tick in all_ticks]
    ax.set_xticklabels(labels)

    ax.legend(fontsize="large")
    plt.tight_layout()
    return fig


def plot_cumulative_capital_various_sophisticated_agent_types(
    df: pd.DataFrame, clear_previous: bool = True
) -> plt.Figure:
    """Special edition of cumulative capital plot with more Sophisticated types."""
    if clear_previous:
        plt.close("all")

    if has_agent_subscripts(df):
        assert_single_simulation(df, "plot_cumulative_capital_various_sophisticated_agent_types")
        df = aggregate_capital_by_firm(df)
    else:
        df = aggregate_data_with_std(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel("Average Capital")
    ax.set_xlabel("Micro Time Step")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    type_to_color = {
        "AI": "#79AEA3",  # Teal
        "Naive": "#9E4770",  # Deep Magenta
        "Sophisticated A": "#1446A0",  # Deep Blue
        "Sophisticated B": "#2166C4",  # Vivid Royal Blue
        "Sophisticated C": "#0094C8",  # Bright Cyan-Blue
        "Sophisticated D": "#00B894",  # Vibrant Teal-Green
    }

    agent_types = sorted(df["Agent Type"].unique())
    for agent_type in agent_types:
        agent_type_df = df[df["Agent Type"] == agent_type]
        # Use a default gray color if an unexpected agent type is encountered.
        if "Capital_mean" in agent_type_df:
            capital_series = agent_type_df["Capital_mean"]
        else:
            capital_series = agent_type_df["Capital"]
        ax.plot(
            agent_type_df["Step"],
            capital_series,
            color=type_to_color.get(base_agent_label(agent_type), "#999999"),
            linewidth=2,
            label=f"{agent_type}",
        )

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x/1000)}K"))

    max_step = df["Step"].max()
    all_ticks = list(range(0, max_step + 10, 10))
    label_ticks = list(range(0, max_step + 50, 50))

    ax.set_xticks(all_ticks)
    labels = [str(int(tick / 10)) if tick in label_ticks else "" for tick in all_ticks]
    ax.set_xticklabels(labels)

    ax.legend(fontsize="large")
    plt.tight_layout()
    return fig


def calculate_percent_change(df: pd.DataFrame) -> dict:
    """Return percent change in capital from start to end for each agent type."""
    percent_changes = {}

    df = aggregate_data_with_std(df)
    # Ensure rows are ordered chronologically within each agent type so the
    # first and last entries correspond to the start and end of the horizon.
    df = df.sort_values(by=["Agent Type", "Step"])

    for agent_type in df["Agent Type"].unique():
        agent_df = df[df["Agent Type"] == agent_type]

        initial_capital = agent_df.iloc[0]["Capital_mean"]
        final_capital = agent_df.iloc[-1]["Capital_mean"]

        percent_change = ((final_capital - initial_capital) / initial_capital) * 100
        percent_changes[agent_type] = percent_change

    return percent_changes


def normalize_percent_change(df: pd.DataFrame) -> dict:
    """Return relative growth in capital for each agent type."""
    normalized_values = {}

    df = aggregate_data_with_std(df)
    # Ensure chronological ordering for accurate growth calculations.
    df = df.sort_values(by=["Agent Type", "Step"])

    for agent_type in df["Agent Type"].unique():
        agent_df = df[df["Agent Type"] == agent_type]

        initial_capital = agent_df.iloc[0]["Capital_mean"]
        final_capital = agent_df.iloc[-1]["Capital_mean"]

        normalized_cap = (final_capital - initial_capital) / initial_capital
        normalized_values[agent_type] = normalized_cap

    return normalized_values


def calculate_avg_bankruptcy_rate(df: pd.DataFrame) -> dict:
    """Return average bankruptcy rate for each agent type."""
    return bankruptcy_rate_by_agent_type(df).to_dict()


def build_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Build a summary statistics table suitable for CSV export."""
    rows = []

    percent_changes = calculate_percent_change(df)
    avg_bankruptcy_rates = calculate_avg_bankruptcy_rate(df)
    normalized_changes = normalize_percent_change(df)

    for agent_type, pct_change in percent_changes.items():
        rows.append(
            {
                "Statistic": f"Percent Change - {agent_type}",
                "Value": float(pct_change),
            }
        )
        rows.append(
            {
                "Statistic": f"Average Bankruptcy Rate - {agent_type}",
                "Value": float(avg_bankruptcy_rates.get(agent_type, 0.0)),
            }
        )

    for agent_type, norm_value in normalized_changes.items():
        rows.append(
            {
                "Statistic": f"Relative Growth - {agent_type}",
                "Value": float(norm_value),
            }
        )

    return pd.DataFrame(rows, columns=["Statistic", "Value"])

"""# Overlap/Firm-Market Entry Plots (Jake)"""

# The following heatmap utilities rely on additional market overlap data.
# They are provided for completeness but are not invoked by default.

def plot_agent_type_market_heatmap(data: pd.DataFrame, step_interval: int = 1, sim: str = 'All') -> None:
    """
    Plots a heatmap showing the frequency of agent types being in markets over time, averaged across simulations,
    with horizontal lines to separate each agent type.

    Parameters:
    - data: DataFrame containing the simulation data.
    - step_interval: Number of timesteps to average over (e.g., 1 for each step, 5 for blocks of 5 steps).
    - sim: Which simulation we are checking activity in.
    """
    if sim != 'All':
        data = data[data['Sim'] == sim].copy()  # Ensure a copy is created for safe modification

    # Group time steps into ``step_interval``-sized bins for averaging.
    data.loc[:, 'Step Interval'] = (data['Step'] // step_interval) * step_interval

    # Aggregate data by averaging over simulations and the specified step intervals
    aggregated_data = (
        data.groupby(['Step Interval', 'Agent Type', 'Market'])['In Market']
        .mean()
        .reset_index()
    )

    # Pivot to create a matrix where rows are (Agent Type, Market), columns are Step Intervals, and values are averaged 'In Market'
    heatmap_data = aggregated_data.pivot(index=['Agent Type', 'Market'], columns='Step Interval', values='In Market')

    # Plot the heatmap
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Avg Frequency In Market'})

    # Add horizontal lines to separate each agent type and agent type-market pair
    total_rows = heatmap_data.index.size
    for line_pos in range(1, total_rows):
        ax.axhline(line_pos, color='black', linewidth=0.5)

    # Draw thicker lines to separate agent types
    agent_types = heatmap_data.index.get_level_values('Agent Type')
    unique_agent_types = agent_types.drop_duplicates()

    # Find where each new agent type starts and draw thicker lines there
    agent_type_changes = [agent_types.tolist().index(a_type) for a_type in unique_agent_types[1:]]
    for line_pos in agent_type_changes:
        ax.axhline(line_pos, color='black', linewidth=1.5)

    # Add titles and labels
    if sim == 'All':
        title = f'Agent Type-Market Presence Over Time (Averaged Over {step_interval} Timesteps)'
    else:
        title = f'Agent Type-Market Presence Over Time (Averaged Over {step_interval} Timesteps) in Simulation {sim}'

    plt.title(title)
    plt.xlabel('Time Step Intervals')
    plt.ylabel('Agent Type, Market')
    plt.tight_layout()
    plt.show()


def plot_market_agent_type_heatmap(data: pd.DataFrame, step_interval: int = 1, sim: str = 'All') -> None:
    """
    Plots a heatmap showing the frequency of agent types being in markets over time (market-to-agent type perspective),
    averaged across simulations, with horizontal lines to separate each market.

    Parameters:
    - data: DataFrame containing the simulation data.
    - step_interval: Number of timesteps to average over (e.g., 1 for each step, 5 for blocks of 5 steps).
    - sim: Which simulation we are checking activity in.
    """
    if sim != 'All':
        data = data[data['Sim'] == sim].copy()  # Ensure a copy is created for safe modification

    # Group time steps into ``step_interval``-sized bins for averaging.
    data.loc[:, 'Step Interval'] = (data['Step'] // step_interval) * step_interval

    # Aggregate data by averaging over simulations and the specified step intervals
    aggregated_data = (
        data.groupby(['Step Interval', 'Market', 'Agent Type'])['In Market']
        .mean()
        .reset_index()
    )

    # Pivot to create a matrix where rows are (Market, Agent Type), columns are Step Intervals, and values are averaged 'In Market'
    heatmap_data = aggregated_data.pivot(index=['Market', 'Agent Type'], columns='Step Interval', values='In Market')

    # Plot the heatmap
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Avg Frequency In Market'})

    # Add horizontal lines to separate each agent type and agent type-market pair
    total_rows = heatmap_data.index.size
    for line_pos in range(1, total_rows):
        ax.axhline(line_pos, color='black', linewidth=0.5)

    # Add horizontal lines to separate each market
    markets = heatmap_data.index.get_level_values('Market')
    market_changes = markets.to_series().diff().ne(0).to_numpy().nonzero()[0]
    for line_pos in market_changes:
        ax.axhline(line_pos, color='black', linewidth=1.5)

    # Add titles and labels
    if sim == 'All':
        title = f'Market-Agent Type Presence Over Time (Averaged Over {step_interval} Timesteps)'
    else:
        title = f'Market-Agent Type Presence Over Time (Averaged Over {step_interval} Timesteps) in Simulation {sim}'

    plt.title(title)
    plt.xlabel('Time Step Intervals')
    plt.ylabel('Market, Agent Type')
    plt.tight_layout()
    plt.show()

"""## Original Overlap/Firm-Market Entry Plots (Not By Agent Type)"""

def plot_firm_market_heatmap(data: pd.DataFrame, step_interval: int = 1, sim: str = 'All') -> None:
    """
    Plots a heatmap showing the frequency of agent types being in markets over time,
    averaged across simulations.

    Parameters:
    - data: DataFrame containing the simulation data.
    - step_interval: Deprecated. Kept for backwards compatibility but ignored.
    - sim: Which simulation we are checking activity in.
    """
    if sim != 'All':
        data = data[data['Sim'] == sim].copy()  # Ensure a copy is created for safe modification

    # Average over broad agent types and preserve each raw timestep.
    data = sort_by_agent_type(data.copy())
    agent_type_order = ['Sophisticated', 'Naive', 'AI']
    data.loc[:, 'Agent Type'] = pd.Categorical(
        data['Agent Type'],
        categories=agent_type_order,
        ordered=True,
    )

    # Aggregate data by averaging over simulations and agent-type membership per market.
    aggregated_data = (
        data.groupby(['Step', 'Agent Type', 'Market'])['In Market']
        .mean()
        .reset_index()
    )
    aggregated_data = aggregated_data.sort_values(['Agent Type', 'Market', 'Step'])

    # Pivot to create a matrix where rows are (Agent Type, Market), columns are timesteps,
    # and values are averaged 'In Market'.
    heatmap_data = aggregated_data.pivot(index=['Agent Type', 'Market'], columns='Step', values='In Market')
    heatmap_data = heatmap_data.reindex(agent_type_order, level='Agent Type')

    # Plot the heatmap
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Average Frequency in Market'})

    # Add horizontal lines to separate each row.
    total_rows = heatmap_data.index.size
    for line_pos in range(1, total_rows):
        ax.axhline(line_pos, color='black', linewidth=0.5)

    # Add thicker separators between agent-type bands.
    type_values = heatmap_data.index.get_level_values('Agent Type')
    type_changes = type_values.to_series().ne(type_values.to_series().shift()).to_numpy().nonzero()[0]
    for line_pos in type_changes[1:]:
        ax.axhline(line_pos, color='black', linewidth=1.5)

    # Show only market labels in each row.
    market_labels = [str(market) for market in heatmap_data.index.get_level_values('Market')]
    ax.set_yticks(np.arange(total_rows) + 0.5)
    ax.set_yticklabels(market_labels, rotation=0)

    # Add a merged-cell style label column for each agent type on the left.
    # Use the same agent-color palette used for CI shading in batch_plot_generation.py.
    type_to_band_color = {
        'Sophisticated': '#1446A0',
        'Naive': '#9E4770',
        'AI': '#79AEA3',
    }

    # Keep a compact legend band, while leaving a visible gap to the y-axis ticks.
    legend_x = -0.18
    legend_width = 0.06

    start = 0
    for agent_type in agent_type_order:
        mask = type_values == agent_type
        count = int(mask.sum())
        if count == 0:
            continue
        center = start + count / 2
        band = patches.Rectangle(
            (legend_x, start),
            legend_width,
            count,
            transform=ax.get_yaxis_transform(),
            facecolor=type_to_band_color.get(agent_type, '#f0f0f0'),
            edgecolor='black',
            linewidth=2.5,
            alpha=0.2,
            clip_on=False,
        )
        ax.add_patch(band)
        ax.text(
            legend_x + (legend_width / 2),
            center,
            agent_type,
            transform=ax.get_yaxis_transform(),
            ha='center',
            va='center',
            rotation=90,
            fontsize=11,
            fontweight='bold',
            clip_on=False,
        )
        start += count

    # Add a thick outer border around the full legend area.
    legend_outline = patches.Rectangle(
        (legend_x, 0),
        legend_width,
        total_rows,
        transform=ax.get_yaxis_transform(),
        facecolor='none',
        edgecolor='black',
        linewidth=2.5,
        clip_on=False,
    )
    ax.add_patch(legend_outline)

    # Emphasize horizontal divider lines between agent-type legend sections.
    section_starts = [0]
    running = 0
    for agent_type in agent_type_order:
        running += int((type_values == agent_type).sum())
        section_starts.append(running)

    for y_pos in section_starts[1:-1]:
        ax.plot(
            [legend_x, legend_x + legend_width],
            [y_pos, y_pos],
            transform=ax.get_yaxis_transform(),
            color='black',
            linewidth=2.5,
            clip_on=False,
        )

    # Label x-axis ticks every 50 timesteps to reduce clutter.
    step_values = heatmap_data.columns.to_numpy()
    tick_positions = [idx + 0.5 for idx, step in enumerate(step_values) if step % 50 == 0]
    tick_labels = [str(step) for step in step_values if step % 50 == 0]
    if tick_positions:
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90)

    # Labels only (no title by request).
    plt.xlabel('Time Step')
    plt.ylabel('Market #', labelpad=34)
    plt.tight_layout()
    plt.subplots_adjust(left=0.18)
    plt.show()

def plot_market_firm_heatmap(data: pd.DataFrame, step_interval: int = 1, sim: str = 'All') -> None:
    """
    Plots a heatmap showing the frequency of firms being in markets over time (market-to-firm perspective),
    averaged across simulations, with horizontal lines to separate each market.

    Parameters:
    - data: DataFrame containing the simulation data.
    - step_interval: Number of timesteps to average over (e.g., 1 for each step, 5 for blocks of 5 steps).
    - sim: Which simulation we are checking activity in.
    """
    if sim != 'All':
        data = data[data['Sim'] == sim].copy()  # Ensure a copy is created for safe modification

    # Group time steps into ``step_interval``-sized bins for averaging.
    data.loc[:, 'Step Interval'] = (data['Step'] // step_interval) * step_interval

    # Aggregate data by averaging over simulations and the specified step intervals
    aggregated_data = (
        data.groupby(['Step Interval', 'Market', 'Firm'])['In Market']
        .mean()
        .reset_index()
    )

    # Pivot to create a matrix where rows are (Market, Firm), columns are Step Intervals, and values are averaged 'In Market'
    heatmap_data = aggregated_data.pivot(index=['Market', 'Firm'], columns='Step Interval', values='In Market')

    # Plot the heatmap
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Avg Frequency In Market'})

    # Add horizontal lines to separate each market
    markets = heatmap_data.index.get_level_values('Market')
    market_changes = markets.to_series().diff().ne(0).to_numpy().nonzero()[0]
    for line_pos in market_changes:
        ax.axhline(line_pos, color='black', linewidth=1.5)

    # Add horizontal lines to separate each market-firm pair
    total_rows = heatmap_data.index.size
    for line_pos in range(1, total_rows):
        ax.axhline(line_pos, color='black', linewidth=0.5)

    # Add titles and labels
    if sim == 'All':
        title = f'Market-Firm Presence Over Time (Averaged Over {step_interval} Timesteps)'
    else:
        title = f'Market-Firm Presence Over Time (Averaged Over {step_interval} Timesteps) in Simulation {sim}'

    plt.title(title)
    plt.xlabel('Time Step Intervals')
    plt.ylabel('Market, Firm')
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Run plotting utilities using command-line arguments."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Business strategy plotting utilities")
    parser.add_argument("--zip-path", type=Path, required=True, help="Path to the ZIP file containing the output data")
    parser.add_argument("--master-output-file-name", required=True, help="Name of master output CSV file inside the ZIP")
    parser.add_argument("--market-overlap-file-name", required=True, help="Name of market overlap CSV file inside the ZIP")
    parser.add_argument(
        "--various-sophisticated-agent-types",
        action="store_true",
        help="Use this flag if the data contain multiple sophisticated agent types."
    )
    parser.add_argument(
        "--plot-heatmaps",
        action="store_true",
        help="Use this flag if you want to plot heat maps in addition to the core plots."
    )
    parser.add_argument(
        "--single-simulation",
        type=int,
        help="If set, only plot data from the specified simulation number."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="If set, save generated plots in this directory instead of showing them interactively.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Skip plot generation and only write summary statistics to CSV.",
    )
    args = parser.parse_args()

    # Read in the master output CSV
    df = load_data(args.zip_path, args.master_output_file_name)
    if args.single_simulation is not None:
        available_simulations = df["Sim"].unique()
        if args.single_simulation not in available_simulations:
            parser.error(
                f"Simulation {args.single_simulation} not found. "
                f"Available simulations: {sorted(available_simulations)}"
            )
        df = df[df["Sim"] == args.single_simulation].copy()
    df = (
        sort_by_agent_type_various_sophisticated_agent_types(df)
        if args.various_sophisticated_agent_types
        else sort_by_agent_type(df)
    )
    if args.single_simulation is not None:
        df = add_agent_type_subscripts(df)

    summary_stats = build_summary_statistics(df)
    summary_output_path = args.zip_path.parent / f"{args.zip_path.stem}_summary_statistics.csv"
    summary_stats.to_csv(summary_output_path, index=False)

    if not args.stats_only:
        # Generate core plots
        if args.various_sophisticated_agent_types:
            _fig1 = avg_bankruptcy_various_sophisticated_agent_types(df, clear_previous=True)
            _fig2 = plot_cumulative_capital_various_sophisticated_agent_types(df, clear_previous=False)
        else:
            _fig1 = avg_bankruptcy(df, clear_previous=True)
            _fig2 = plot_cumulative_capital(df, clear_previous=False)

        _fig3 = performance_summary_std_error(
            df,
            clear_previous=False,
            show_std_error=args.single_simulation is None,
        )

        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            _fig1.savefig(args.output_dir / "avg_bankruptcy.png", dpi=300)
            _fig2.savefig(args.output_dir / "cumulative_capital.png", dpi=300)
            _fig3.savefig(args.output_dir / "performance_summary_std_error.png", dpi=300)
        else:
            # Show all open figures at once (prevents later plots from closing earlier ones)
            plt.show()

    # Heatmap utilities require additional overlap data and are not executed by default.
    if args.plot_heatmaps and not args.stats_only:
        zip_filename = Path(args.zip_path)
        output = load_data(zip_filename, args.master_output_file_name)
        if args.single_simulation is not None:
            output = output[output["Sim"] == args.single_simulation].copy()
        plot_market_agent_type_heatmap(output, step_interval=5)
        if args.output_dir is not None:
            plt.gcf().savefig(args.output_dir / "market_agent_type_heatmap.png", dpi=300)
        plot_firm_market_heatmap(output)
        if args.output_dir is not None:
            plt.gcf().savefig(args.output_dir / "firm_market_heatmap.png", dpi=300)
        plot_market_firm_heatmap(output, step_interval=5)
        if args.output_dir is not None:
            plt.gcf().savefig(args.output_dir / "market_firm_heatmap.png", dpi=300)

if __name__ == "__main__":
    main()
