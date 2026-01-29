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
    agent_types = set(df["Agent Type"].tolist())
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
    agent_types = set(df["Agent Type"].tolist())
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
        type_to_color.get(agent_type, "#999999") for agent_type in agent_types
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
        type_to_color.get(agent_type, "#999999") for agent_type in agent_types
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
                color=type_to_color[agent_type],
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
        patches.Patch(color=type_to_color[agent_type], alpha=0.2, label=f"{agent_type}")
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
                color=type_to_color[agent_type],
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
                    color=type_to_color[agent_type],
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

    df = aggregate_data_with_std(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel("Avg. Capital")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    colors = ["#79AEA3", "#9E4770", "#1446A0", "#00FFFF", "r", "b", "g"]
    type_to_color = {"AI": colors[0], "Naive": colors[1], "Sophisticated": colors[2]}
    agent_types = sorted(df["Agent Type"].unique())
    for agent_type in agent_types:
        agent_type_df = df[df["Agent Type"] == agent_type]
        # Plot average capital trajectory for the current agent type.
        ax.plot(
            agent_type_df["Step"],
            agent_type_df["Capital_mean"],
            color=type_to_color[agent_type],
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

    ax.legend()
    plt.tight_layout()
    return fig


def plot_cumulative_capital_various_sophisticated_agent_types(
    df: pd.DataFrame, clear_previous: bool = True
) -> plt.Figure:
    """Special edition of cumulative capital plot with more Sophisticated types."""
    if clear_previous:
        plt.close("all")

    df = aggregate_data_with_std(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel("Avg. Capital")
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
        ax.plot(
            agent_type_df["Step"],
            agent_type_df["Capital_mean"],
            color=type_to_color.get(agent_type, "#999999"),
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

    ax.legend()
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
    Plots a heatmap showing the frequency of firms being in markets over time, averaged across simulations,
    with horizontal lines to separate each firm.

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
        data.groupby(['Step Interval', 'Firm', 'Market'])['In Market']
        .mean()
        .reset_index()
    )

    # Pivot to create a matrix where rows are (Firm, Market), columns are Step Intervals, and values are averaged 'In Market'
    heatmap_data = aggregated_data.pivot(index=['Firm', 'Market'], columns='Step Interval', values='In Market')

    # Plot the heatmap
    plt.figure(figsize=(15, 12))
    ax = sns.heatmap(heatmap_data, cmap="YlGnBu", cbar_kws={'label': 'Avg Frequency In Market'})

    # Add horizontal lines to separate each firm
    firms = heatmap_data.index.get_level_values('Firm')
    firm_changes = firms.to_series().diff().ne(0).to_numpy().nonzero()[0]
    for line_pos in firm_changes:
        ax.axhline(line_pos, color='black', linewidth=1.5)

    # Add horizontal lines to separate each firm and firm-market pair
    total_rows = heatmap_data.index.size
    for line_pos in range(1, total_rows):
        ax.axhline(line_pos, color='black', linewidth=0.5)

    # Add titles and labels
    if sim == 'All':
        title = f'Firm-Market Presence Over Time (Averaged Over {step_interval} Timesteps)'
    else:
        title = f'Firm-Market Presence Over Time (Averaged Over {step_interval} Timesteps) in Simulation {sim}'

    plt.title(title)
    plt.xlabel('Time Step Intervals')
    plt.ylabel('Firm, Market')
    plt.tight_layout()
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
    df = sort_by_agent_type_various_sophisticated_agent_types(df) if args.various_sophisticated_agent_types else sort_by_agent_type(df)

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

    # Show all open figures at once (prevents later plots from closing earlier ones)
    plt.show()


    # Print summary statistics
    percent_changes = calculate_percent_change(df)
    avg_bankruptcy_rates = calculate_avg_bankruptcy_rate(df)
    for agent_type, pct_change in percent_changes.items():
        bankruptcy_rate = avg_bankruptcy_rates.get(agent_type, 0)
        print(
            f"{agent_type}: {pct_change:.2f}% "
            f"(avg bankruptcy rate: {bankruptcy_rate:.2%})"
        )

    normalized_changes = normalize_percent_change(df)
    for agent_type, norm_value in normalized_changes.items():
        print(f"{agent_type}: {norm_value:.2f} (relative growth)")

    # Heatmap utilities require additional overlap data and are not executed by default.
    if args.plot_heatmaps:
        zip_filename = Path(args.zip_path)
        output = load_data(zip_filename, args.master_output_file_name)
        if args.single_simulation is not None:
            output = output[output["Sim"] == args.single_simulation].copy()
        plot_market_agent_type_heatmap(output, step_interval=5)
        plot_firm_market_heatmap(output, step_interval=5)
        plot_market_firm_heatmap(output, step_interval=5)

if __name__ == "__main__":
    main()
