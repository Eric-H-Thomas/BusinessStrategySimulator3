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

    with zipfile.ZipFile(zip_path) as zip_ref:
        with zip_ref.open(csv_name) as csv_file:
            return pd.read_csv(csv_file)


def sort_by_agent_type(df):
  agent_types = set(df["Agent Type"].tolist())
  for agent in agent_types:
    sophisticated = re.search(r"HighestOverlap", agent)
    naive = re.search(r"All", agent)
    ai = re.search(r"StableBaselines3", agent)

    if sophisticated:
      df.replace({"Agent Type": agent}, "Sophisticated", inplace=True)
    elif naive:
      df.replace({"Agent Type": agent}, "Naive", inplace=True)
    elif ai:
      df.replace({"Agent Type": agent}, "AI", inplace=True)

  return df

def sort_by_agent_type_special_edition(df):
  agent_types = set(df["Agent Type"].tolist())
  sophisticated_suffixes = ['A', 'B', 'C', 'D']
  sophisticated_idx = 0
  for agent in agent_types:
    sophisticated = re.search(r"HighestOverlap", agent)
    naive = re.search(r"All", agent)
    ai = re.search(r"StableBaselines3", agent)

    if sophisticated:
      df.replace({"Agent Type": agent}, "Sophisticated" + " " + sophisticated_suffixes[sophisticated_idx], inplace=True)
      sophisticated_idx += 1
    elif naive:
      df.replace({"Agent Type": agent}, "Naive", inplace=True)
    elif ai:
      df.replace({"Agent Type": agent}, "AI", inplace=True)

  return df

def avg_bankruptcy(df, clear_previous=True):
    if clear_previous:
        plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.suptitle("Frequency of Bankruptcy by Agent Type", fontsize=16)

    ax.set_ylabel("Percentage of Simulations Ending in Bankruptcy")
    # ax.set_xlabel("Agent Type")
    ax.set_ylim(0,100)
    type_to_color = {
        "AI": "#79AEA3",
        "Naive": "#9E4770",
        "Sophisticated": "#1446A0"
    }

    agent_types = np.array(df["Agent Type"].unique())
    avg_bankruptcies_per_agent_type = []
    for agent_type in agent_types:
        agent_type_df = df[df["Agent Type"] == agent_type]

        bankruptcy_cnt = agent_type_df["Capital"].value_counts()[-1e-09]
        avg_bankruptcies_per_agent_type.append(bankruptcy_cnt / len(agent_type_df["Capital"]))

    colors = [type_to_color.get(agent_type, "#999999") for agent_type in agent_types]  # Default to gray if not found
    avg_bankruptcies_per_agent_type = np.array(avg_bankruptcies_per_agent_type)
    ax.bar(agent_types, avg_bankruptcies_per_agent_type * 100, label=agent_types, color=colors)

    # Adjust layout with more space for labels
    plt.tight_layout()
    return fig

def avg_bankruptcy_special_edition(df, clear_previous=True):
    if clear_previous:
        plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.suptitle("Frequency of Bankruptcy by Agent Type", fontsize=16)

    ax.set_ylabel("Percentage of Simulations Ending in Bankruptcy")
    # ax.set_xlabel("Agent Type")
    ax.set_ylim(0,100)
    type_to_color = {
        "AI": "#79AEA3",
        "Naive": "#9E4770",
        "Sophisticated A": "#1446A0",
        "Sophisticated B": "#2166C4",
        "Sophisticated C": "#0094C8",
        "Sophisticated D": "#00B894"
    }

    agent_types = np.array(df["Agent Type"].unique())
    avg_bankruptcies_per_agent_type = []
    for agent_type in agent_types:
        agent_type_df = df[df["Agent Type"] == agent_type]

        bankruptcy_cnt = agent_type_df["Capital"].value_counts()[-1e-09]
        avg_bankruptcies_per_agent_type.append(bankruptcy_cnt / len(agent_type_df["Capital"]))

    colors = [type_to_color.get(agent_type, "#999999") for agent_type in agent_types]  # Default to gray if not found
    avg_bankruptcies_per_agent_type = np.array(avg_bankruptcies_per_agent_type)
    ax.bar(agent_types, avg_bankruptcies_per_agent_type * 100, label=agent_types, color=colors)

    # Adjust layout with more space for labels
    plt.tight_layout()
    return fig

def avg_bankruptcy_combined(dfs, labels, clear_previous=True):
    if clear_previous:
        plt.close('all')

    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.suptitle("Frequency of Bankruptcy by Agent Type", fontsize=16)

    # ax.set_ylabel("Percentage of Simulations Ending in Bankruptcy")
    # ax.set_xlabel("Dataset")
    ax.set_ylim(0,100)
    colors = ['#79AEA3','#9E4770', '#1446A0', '#00FFFF', 'r', 'b', 'g']
    type_to_color = {
        "AI" : colors[0],
        "Naive": colors[1],
        "Sophisticated": colors[2]
    }

    agent_types = set()
    for df in dfs:
        agent_types.update(df["Agent Type"].unique())

    agent_types = sorted(agent_types, reverse=True)
    total_clusters = len(dfs)
    x = np.arange(total_clusters)  # X locations for each cluster
    width = 0.8 / len(agent_types)  # Adjust bar width based on number of agent types

    for i, (df, label) in enumerate(zip(dfs, labels)):
      avg_bankruptcies_per_agent_type = []
      for agent_type in agent_types:
          agent_type_df = df[df["Agent Type"] == agent_type]
          bankruptcy_cnt = agent_type_df["Capital"].value_counts().get(-1e-09, 0)
          avg_bankruptcies_per_agent_type.append(bankruptcy_cnt / len(agent_type_df["Capital"]))

      avg_bankruptcies_per_agent_type = np.array(avg_bankruptcies_per_agent_type)

      # Offset to separate clusters of the same agent type across dfs
      for j, agent_type in enumerate(agent_types):
            ax.bar(x[i] + (j - len(agent_types) / 2) * width + width / 2,  # Positioning bars within the cluster
                   avg_bankruptcies_per_agent_type[j] * 100,
                   width=width,
                   label=f"{agent_type}" if i == 0 else "",  # Label first occurrence only
                   color=type_to_color[agent_type])

    tick_positions = x
    tick_labels = labels
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=0, ha="center")

    ax.legend(title="Agent Types")

    # Adjust layout with more space for labels
    plt.tight_layout()
    return fig

avg_bankruptcy_special_edition(df).show()

# x_labels = ["Low Market Overlap", "High Market Overlap"]
# x_labels = ["Very Easy Economy", "Easy Economy", "Difficult Economy"]
# x_labels = ["1 Sophisticated", "2 Sophisticated", "3 Sophisticated", "4 Sophisticated", "5 Sophisticated", "6 Sophisticated", "7 Sophisticated"]
# avg_bankruptcy_combined(dfs, x_labels).show()

agent_types = np.array(df["Agent Type"].unique())
avg_bankruptcies_per_agent_type = []
for agent_type in agent_types:
    agent_type_df = df[df["Agent Type"] == agent_type]

    bankruptcy_cnt = agent_type_df["Capital"].value_counts()[-1e-09]
    avg_bankruptcies_per_agent_type.append(bankruptcy_cnt / len(agent_type_df["Capital"]))

print([f"{val*100:.2f}%" for val in avg_bankruptcies_per_agent_type])

def aggregate_data_with_std(df):
    """
    First sum across markets within each simulation, then calculate std and average across simulations.
    """

    # First sum across markets within each simulation, step, and firm
    market_sum = df.groupby(['Sim', 'Step', 'Firm', 'Agent Type']).agg({
        'Rev': 'sum',
        'Fix Cost': 'sum',
        'Var Cost': 'sum',
        'Capital': 'mean'  # Include 'Capital' in the aggregation but mean instead of sum because Capital is duplicated across markets
    }).reset_index()

    # Calculate mean and std across simulations for each step and firm
    sim_stats = market_sum.groupby(['Step', 'Agent Type']).agg({
        'Rev': ['mean', 'std'],
        'Fix Cost': ['mean', 'std'],
        'Var Cost': ['mean', 'std'],
        'Capital': ['mean', 'std']
    }).reset_index()

    # Flatten the MultiIndex columns
    sim_stats.columns = ['_'.join(col).strip('_') for col in sim_stats.columns]

    return sim_stats

aggregate_data_with_std(df).head(30)

def performance_summary_std_error(df, clear_previous=True):
    if clear_previous:
        plt.close('all')

    df = aggregate_data_with_std(df)
    n = len(df) # get n to calculate standard error

    fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
    # fig.suptitle('External Performance', fontsize=16)
    metrics = ['Profit', 'Rev', 'Total Cost', 'Fix Cost', 'Var Cost']
    colors = ['#79AEA3','#9E4770', '#1446A0', '#00FFFF', 'r', 'b', 'g']
    type_to_color = {
        "AI" : colors[0],
        "Naive": colors[1],
        "Sophisticated": colors[2]
    }
    agent_types = sorted(df['Agent Type'].unique())

    max_step = df['Step'].max()
    all_macrostep_ticks = list(range(0, max_step + 10, 10))
    label_macrostep_ticks = list(range(0, max_step + 50, 50))

    # Create a list of proxy handles for the global legend (shaded regions)
    shaded_region_patches = [
        patches.Patch(color=color, alpha=0.2, label=f'{agent_type}')
        for agent_type, color in zip(sorted(df['Agent Type'].unique()), colors)
    ]

    for idx, (ax, metric) in enumerate(zip(axes, metrics)):
        ax.clear()

        for j, agent_type in enumerate(agent_types):
            agent_type_df = df[df['Agent Type'] == agent_type]

            # Calculate the values for the metric
            if metric == 'Profit':
                mean_values = agent_type_df['Rev_mean'] - (
                    agent_type_df['Fix Cost_mean'] + agent_type_df['Var Cost_mean']
                )
                std_values = np.sqrt(
                    agent_type_df['Rev_std'] ** 2 +
                    (agent_type_df['Fix Cost_std'] ** 2 + agent_type_df['Var Cost_std'] ** 2)
                )
            elif metric == 'Total Cost':
                mean_values = agent_type_df['Fix Cost_mean'] + agent_type_df['Var Cost_mean']
                std_values = np.sqrt(agent_type_df['Fix Cost_std'] ** 2 + agent_type_df['Var Cost_std'] ** 2)
            else:
                mean_values = agent_type_df[f'{metric}_mean']
                std_values = agent_type_df[f'{metric}_std']

            # Plot the mean line
            ax.plot(agent_type_df['Step'], mean_values,
                    color=type_to_color[agent_type],
                    label=f'Agent Type {agent_type}',
                    linewidth=1)

            # Add shaded region for standard deviation
            ax.fill_between(
                agent_type_df['Step'],
                (mean_values - (std_values / np.sqrt(n)) ).clip(lower=0),  # Ensure no negative values
                mean_values + (std_values / np.sqrt(n)),
                color=type_to_color[agent_type],
                alpha=0.2
            )

        ax.set_ylabel(metric)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add line legends for the first plot only
        if idx == 0:
            ax.legend(loc='upper left', title="Firm Lines", fontsize='x-small')

        # Set all ticks
        ax.set_xticks(all_macrostep_ticks)

        # Create labels
        labels = []
        for tick in all_macrostep_ticks:
            if tick in label_macrostep_ticks:
                labels.append(str(int(tick / 10)))
            else:
                labels.append('')

        # Set the labels and make them visible for all subplots
        ax.set_xticklabels(labels)
        ax.tick_params(axis='x', which='major', length=4, labelbottom=True)

    # Add a global legend for the shaded regions at the top
    fig.legend(
        handles=shaded_region_patches,
        loc='upper center',
        ncol=len(shaded_region_patches),
        bbox_to_anchor=(0.5, 0.95),
        frameon=False,
        title="Shaded Regions: Standard Error"
    )

    # Set x-label for all subplots
    for ax in axes:
        ax.set_xlabel('Macrostep')

    # Adjust layout with more space for labels
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust the layout to fit the legend
    return fig

performance_summary_std_error(df).show()

"""# Cumulative Average Capital Plots (Nate)"""

def plot_cumulative_capital(df, clear_previous=True):
    if clear_previous:
        plt.close('all')

    df = aggregate_data_with_std(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    # ax.set_title('Cumulative Average Capital', pad=15, fontsize=14)
    ax.set_ylabel('Avg. Capital')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    colors = ['#79AEA3','#9E4770', '#1446A0', '#00FFFF', 'r', 'b', 'g']
    type_to_color = {
        "AI" : colors[0],
        "Naive": colors[1],
        "Sophisticated": colors[2]
    }
    agent_types = sorted(df['Agent Type'].unique())
    for i, agent_type in enumerate(agent_types):
        agent_type_df = df[df['Agent Type'] == agent_type]
        line = ax.plot(agent_type_df['Step'], agent_type_df['Capital_mean'],
                      color=type_to_color[agent_type],
                      linewidth=2,
                      label=f'{agent_type}')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))

    max_step = df['Step'].max()
    all_ticks = list(range(0, max_step + 10, 10))
    label_ticks = list(range(0, max_step + 50, 50))

    ax.set_xticks(all_ticks)
    labels = [str(int(tick/10)) if tick in label_ticks else '' for tick in all_ticks]
    ax.set_xticklabels(labels)

    ax.legend()
    plt.tight_layout()
    return fig

plot_cumulative_capital(df).show()

def plot_cumulative_capital_special_edition(df, clear_previous=True):
    if clear_previous:
        plt.close('all')

    df = aggregate_data_with_std(df)
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.set_ylabel('Avg. Capital')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    type_to_color = {
        "AI": "#79AEA3",  # Teal
        "Naive": "#9E4770",  # Deep Magenta
        "Sophisticated B": "#2166C4",  # Vivid Royal Blue
        "Sophisticated C": "#0094C8",  # Bright Cyan-Blue
        "Sophisticated D": "#00B894",  # Vibrant Teal-Green
        "Sophisticated E": "#F4A261"   # Warm Orange
    }

    agent_types = sorted(df['Agent Type'].unique())
    for agent_type in agent_types:
        agent_type_df = df[df['Agent Type'] == agent_type]
        ax.plot(agent_type_df['Step'], agent_type_df['Capital_mean'],
                color=type_to_color.get(agent_type, "#999999"),  # Default to gray if missing
                linewidth=2,
                label=f'{agent_type}')

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}K'))

    max_step = df['Step'].max()
    all_ticks = list(range(0, max_step + 10, 10))
    label_ticks = list(range(0, max_step + 50, 50))

    ax.set_xticks(all_ticks)
    labels = [str(int(tick/10)) if tick in label_ticks else '' for tick in all_ticks]
    ax.set_xticklabels(labels)

    ax.legend()
    plt.tight_layout()
    return fig

plot_cumulative_capital_special_edition(df).show()

def calculate_percent_change(df):
    percent_changes = {}

    # Ensure sorted steps
    df = aggregate_data_with_std(df)
    df = df.sort_values(by=['Agent Type', 'Step'])

    for agent_type in df['Agent Type'].unique():
        agent_df = df[df['Agent Type'] == agent_type]

        initial_capital = agent_df.iloc[0]['Capital_mean']  # First value
        final_capital = agent_df.iloc[-1]['Capital_mean']  # Last value

        percent_change = ((final_capital - initial_capital) / initial_capital) * 100
        percent_changes[agent_type] = percent_change

    return percent_changes

percent_changes = calculate_percent_change(df)

for agent_type, pct_change in percent_changes.items():
    print(f"{agent_type}: {pct_change:.2f}%")

def normalize_percent_change(df):
    normalized_values = {}

    # Ensure sorted steps
    df = aggregate_data_with_std(df)
    df = df.sort_values(by=['Agent Type', 'Step'])

    for agent_type in df['Agent Type'].unique():
        agent_df = df[df['Agent Type'] == agent_type]

        initial_capital = agent_df.iloc[0]['Capital_mean']  # First value
        final_capital = agent_df.iloc[-1]['Capital_mean']  # Last value

        normalized_cap = (final_capital - initial_capital) / initial_capital  # Scale by initial capital
        normalized_values[agent_type] = normalized_cap

    return normalized_values

normalized_changes = normalize_percent_change(df)

for agent_type, norm_value in normalized_changes.items():
    print(f"{agent_type}: {norm_value:.2f} (relative growth)")

"""# Overlap/Firm-Market Entry Plots (Jake)"""

zip_filename = Path("MarketOverlapVarious.zip")

# Read in data (change file names if needed)
overlap = load_data(zip_filename, 'MarketOverlap4Random.csv')
overlap = overlap.drop(columns=['Unnamed: 0', 'Unnamed: 6'])

output = load_data(zip_filename, 'MasterOutputVaryingAgentTypes4Random.csv')
output = output.drop(columns=['Unnamed: 0', 'Unnamed: 14'])
overlap

# overlap = sort_by_agent_type(overlap)
output = sort_by_agent_type(output)

# if sim != 'All':
#     data = data[data['Sim'] == sim].copy()  # Ensure a copy is created for safe modification

# Create a new column to group time steps into intervals so we don't get warnings and undefined behavior
output.loc[:, 'Step Interval'] = (output['Step'] // 5) * 5

# Aggregate data by averaging over simulations and the specified step intervals
aggregated_data = (
    output.groupby(['Step Interval', 'Firm', 'Market'])['In Market']
    .mean()
    .reset_index()
)

# Pivot to create a matrix where rows are (Firm, Market), columns are Step Intervals, and values are averaged 'In Market'
heatmap_data = aggregated_data.pivot(index=['Firm', 'Market'], columns='Step Interval', values='In Market')

output[output['Firm'] == 2]['In Market'].describe()

aggregated_data = (
    output.groupby(['Step Interval', 'Firm', 'Market'])['In Market']
    .mean()
    .reset_index()
)

aggregated_data[aggregated_data['Firm'] == 2]

# Step 1: Group by Market A and Market B to calculate the average number of common capabilities
avg_capabilities = overlap.groupby(['Market A', 'Market B'])['Num Common Capabilities'].mean().reset_index()

# Step 2: Pivot the data to create a matrix format
heatmap_data = avg_capabilities.pivot(index='Market A', columns='Market B', values='Num Common Capabilities')

# Step 3: Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Avg Common Capabilities'})
plt.title('Average Common Capabilities Between Market Pairs')
plt.xlabel('Market B')
plt.ylabel('Market A')
plt.show()

def plot_agent_type_market_heatmap(data, step_interval=1, sim='All'):
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

    # Create a new column to group time steps into intervals so we don't get warnings and undefined behavior
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
    agent_type_changes = [agent_types.tolist().index(atype) for atype in unique_agent_types[1:]]
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

plot_agent_type_market_heatmap(output, step_interval=5)

plot_agent_type_market_heatmap(output, step_interval=5, sim=1)

def plot_market_agent_type_heatmap(data, step_interval=1, sim='All'):
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

    # Create a new column to group time steps into intervals
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

def plot_firm_market_heatmap(data, step_interval=1, sim='All'):
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

    # Create a new column to group time steps into intervals so we don't get warnings and undefined behavior
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

def plot_market_firm_heatmap(data, step_interval=1, sim='All'):
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

    # Create a new column to group time steps into intervals
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
    parser = argparse.ArgumentParser(
        description="Business strategy plotting utilities"
    )
    parser.add_argument(
        "zip_path",
        type=Path,
        help="Path to the ZIP file containing simulation output",
    )
    parser.add_argument(
        "csv_name", help="Name of the CSV file within the ZIP archive"
    )
    args = parser.parse_args()

    df = load_data(args.zip_path, args.csv_name)
    df = sort_by_agent_type_special_edition(df)

    avg_bankruptcy_special_edition(df).show()

    agent_types = np.array(df["Agent Type"].unique())
    avg_bankruptcies_per_agent_type = []
    for agent_type in agent_types:
        agent_type_df = df[df["Agent Type"] == agent_type]
        bankruptcy_cnt = agent_type_df["Capital"].value_counts()[-1e-09]
        avg_bankruptcies_per_agent_type.append(
            bankruptcy_cnt / len(agent_type_df["Capital"])
        )
    print([f"{val*100:.2f}%" for val in avg_bankruptcies_per_agent_type])

    aggregate_data_with_std(df).head(30)
    performance_summary_std_error(df).show()
    plot_cumulative_capital(df).show()
    plot_cumulative_capital_special_edition(df).show()

    percent_changes = calculate_percent_change(df)
    for agent_type, pct_change in percent_changes.items():
        print(f"{agent_type}: {pct_change:.2f}%")

    normalized_changes = normalize_percent_change(df)
    for agent_type, norm_value in normalized_changes.items():
        print(f"{agent_type}: {norm_value:.2f} (relative growth)")

    zip_filename = Path("MarketOverlapVarious.zip")
    overlap = load_data(zip_filename, "MarketOverlap4Random.csv")
    overlap = overlap.drop(columns=["Unnamed: 0", "Unnamed: 6"])
    output = load_data(zip_filename, "MasterOutputVaryingAgentTypes4Random.csv")

    plot_market_agent_type_heatmap(output, step_interval=5)
    plot_firm_market_heatmap(output, step_interval=5)
    plot_market_firm_heatmap(output, step_interval=5)


if __name__ == "__main__":
    main()

