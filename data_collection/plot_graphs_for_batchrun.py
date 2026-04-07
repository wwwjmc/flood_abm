"""
Batch Run Results Visualization Script

This script generates line plots from simulation data stored in CSV files
produced by flood_batchrun.py. It supports per-scenario filtering, average
with std-band plots, and dam scenario comparison line/bar charts.

CSV files expected in 'data_collection/':
    results_S0.csv, results_S1.csv, results_S2.csv, results_S3.csv

Images are saved in the graphs directory.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# ========================= Load & Prepare Data ========================= #
# Root folder of this script
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# CSV files are in flood_abm/run/data_collection
csv_dir = os.path.join(PROJECT_DIR, "..", "run", "data_collection")

# Folder to save generated plots (can be data_collection or a separate folder)
output_dir = os.path.join(PROJECT_DIR, "data_collection")  # or "graphs" if you prefer
os.makedirs(output_dir, exist_ok=True)

print("Reading CSVs from:", csv_dir)
print("Saving graphs to:", output_dir)

scenarios = ["S0", "S1", "S2", "S3"]
all_dfs = []

for s in scenarios:
    path = os.path.join(csv_dir, f"results_{s}.csv")
    print("Loading:", path)
    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        continue
    df = pd.read_csv(path)
    df["dam_scenario_name"] = s
    all_dfs.append(df)

# Combine all scenario results
results_df = pd.concat(all_dfs, ignore_index=True)

# Convert steps to days (batchrun uses hourly steps, 24 steps = 1 day)
if results_df["Step"].max() > 100:
    results_df["Step"] = results_df["Step"] / 24

# ========================= Column Groups ========================= #

columns_to_include = [
    ["Step", "Evacuated", "Preflood_Non_Evacuation_Measure_Implemented",
     "Duringflood_Coping_Action_Implemented", "Postflood_Adaptation_Measures_Planned"],
    ["Step", "Stranded", "Injured", "Sheltered", "Hospitalized", "Death"],
    ["Step", "evacuated_SES_1_0_0.3", "evacuated_SES_1_0.7_1"],
    ["Step", "evacuated_SES_2_0_0.3", "evacuated_SES_2_0.7_1"],
    ["Step", "stranded_SES_1_0_0.3", "stranded_SES_1_0.7_1"],
    ["Step", "stranded_SES_2_0_0.3", "stranded_SES_2_0.7_1"],
    ["Step", "injured_SES_1_0_0.3", "injured_SES_1_0.7_1"],
    ["Step", "injured_SES_2_0_0.3", "injured_SES_2_0.7_1"],
    ["Step", "hospitalized_SES_1_0_0.3", "hospitalized_SES_1_0.7_1"],
    ["Step", "hospitalized_SES_2_0_0.3", "hospitalized_SES_2_0.7_1"],
    ["Step", "sheltered_SES_1_0_0.3", "sheltered_SES_1_0.7_1"],
    ["Step", "sheltered_SES_2_0_0.3", "sheltered_SES_2_0.7_1"],
    ["Step", "dead_SES_1_0_0.3", "dead_SES_1_0.7_1"],
    ["Step", "dead_SES_2_0_0.3", "dead_SES_2_0.7_1"],
    ["Step", "Houses_Flooded", "Businesses_Flooded", "Schools_Flooded"],
    ["Step", "Wealth_People", "Wealth_Businesses", "Wealth_Shelter",
     "Wealth_Healthcare", "Wealth_Government"],
    ["Step", "PMT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3",
     "TPB_preflood_non_evacuation_measure_implemented_SES_1_0_0.3",
     "SCT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3",
     "CRT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3"],
    ["Step", "PMT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1",
     "TPB_preflood_non_evacuation_measure_implemented_SES_1_0.7_1",
     "SCT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1",
     "CRT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1"],
    ["Step", "PMT_preflood_non_evacuation_measure_implemented_SES_2_0_0.3",
     "TPB_preflood_non_evacuation_measure_implemented_SES_2_0_0.3",
     "SCT_preflood_non_evacuation_measure_implemented_SES_2_0_0.3",
     "CRT_preflood_non_evacuation_measure_implemented_SES_2_0_0.3"],
    ["Step", "PMT_preflood_non_evacuation_measure_implemented_SES_2_0.7_1",
     "TPB_preflood_non_evacuation_measure_implemented_SES_2_0.7_1",
     "SCT_preflood_non_evacuation_measure_implemented_SES_2_0.7_1",
     "CRT_preflood_non_evacuation_measure_implemented_SES_2_0.7_1"],
    ["Step", "PMT_evacuation_SES_1_0_0.3", "TPB_evacuation_SES_1_0_0.3",
     "SCT_evacuation_SES_1_0_0.3", "CRT_evacuation_SES_1_0_0.3"],
    ["Step", "PMT_evacuation_SES_1_0.7_1", "TPB_evacuation_SES_1_0.7_1",
     "SCT_evacuation_SES_1_0.7_1", "CRT_evacuation_SES_1_0.7_1"],
    ["Step", "PMT_evacuation_SES_2_0_0.3", "TPB_evacuation_SES_2_0_0.3",
     "SCT_evacuation_SES_2_0_0.3", "CRT_evacuation_SES_2_0_0.3"],
    ["Step", "PMT_evacuation_SES_2_0.7_1", "TPB_evacuation_SES_2_0.7_1",
     "SCT_evacuation_SES_2_0.7_1", "CRT_evacuation_SES_2_0.7_1"],
    ["Step", "PMT_duringflood_coping_action_implemented_SES_1_0_0.3",
     "TPB_duringflood_coping_action_implemented_SES_1_0_0.3",
     "SCT_duringflood_coping_action_implemented_SES_1_0_0.3",
     "CRT_duringflood_coping_action_implemented_SES_1_0_0.3"],
    ["Step", "PMT_duringflood_coping_action_implemented_SES_1_0.7_1",
     "TPB_duringflood_coping_action_implemented_SES_1_0.7_1",
     "SCT_duringflood_coping_action_implemented_SES_1_0.7_1",
     "CRT_duringflood_coping_action_implemented_SES_1_0.7_1"],
    ["Step", "PMT_duringflood_coping_action_implemented_SES_2_0_0.3",
     "TPB_duringflood_coping_action_implemented_SES_2_0_0.3",
     "SCT_duringflood_coping_action_implemented_SES_2_0_0.3",
     "CRT_duringflood_coping_action_implemented_SES_2_0_0.3"],
    ["Step", "PMT_duringflood_coping_action_implemented_SES_2_0.7_1",
     "TPB_duringflood_coping_action_implemented_SES_2_0.7_1",
     "SCT_duringflood_coping_action_implemented_SES_2_0.7_1",
     "CRT_duringflood_coping_action_implemented_SES_2_0.7_1"],
    ["Step", "PMT_postflood_adaptation_measures_planned_SES_1_0_0.3",
     "TPB_postflood_adaptation_measures_planned_SES_1_0_0.3",
     "SCT_postflood_adaptation_measures_planned_SES_1_0_0.3",
     "CRT_postflood_adaptation_measures_planned_SES_1_0_0.3"],
    ["Step", "PMT_postflood_adaptation_measures_planned_SES_1_0.7_1",
     "TPB_postflood_adaptation_measures_planned_SES_1_0.7_1",
     "SCT_postflood_adaptation_measures_planned_SES_1_0.7_1",
     "CRT_postflood_adaptation_measures_planned_SES_1_0.7_1"],
    ["Step", "PMT_postflood_adaptation_measures_planned_SES_2_0_0.3",
     "TPB_postflood_adaptation_measures_planned_SES_2_0_0.3",
     "SCT_postflood_adaptation_measures_planned_SES_2_0_0.3",
     "CRT_postflood_adaptation_measures_planned_SES_2_0_0.3"],
    ["Step", "PMT_postflood_adaptation_measures_planned_SES_2_0.7_1",
     "TPB_postflood_adaptation_measures_planned_SES_2_0.7_1",
     "SCT_postflood_adaptation_measures_planned_SES_2_0.7_1",
     "CRT_postflood_adaptation_measures_planned_SES_2_0.7_1"],
]

# ========================= Plot Function ========================= #

def plot_graphs(df, output_dir, group_num, columns_to_include, x_label, y_label,
                legend_labels=None, legend_fontsize=None, line_thickness=1.5,
                x_range=None, y_range=None, x_interval=None, y_interval=None,
                axis_label_size=12, tick_label_size=None, colors=None,
                plot_average=False, include_std=False, plot_title=None,
                show_title=False, save_as=True, scenario_filter=None):
    """
    Plot simulation results from a combined multi-scenario DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame produced by flood_batchrun.py.
        Must contain a 'dam_scenario_name' column if scenario_filter is used.
    scenario_filter : str or None
        If given (e.g. "S0"), only rows matching that scenario are plotted.
        If None, all rows are used (averaged across scenarios).
    All other parameters are identical to the original plot_graphs signature.
    """

    # Filter by scenario if requested
    if scenario_filter is not None and "dam_scenario_name" in df.columns:
        plot_df = df[df["dam_scenario_name"] == scenario_filter].copy()
    else:
        plot_df = df.copy()

    if group_num < 0 or group_num >= len(columns_to_include):
        raise ValueError("Invalid group number.")

    selected_columns = columns_to_include[group_num]

    # Keep only columns that exist in the dataframe
    available_cols = [c for c in selected_columns if c in plot_df.columns]
    if len(available_cols) < 2:
        print(f"[WARNING] Group {group_num}: not enough columns found, skipping.")
        return

    numeric_data = plot_df[available_cols].copy()
    numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    grouped = numeric_data.groupby(available_cols[0])
    average_data = grouped.mean()
    std_data = grouped.std()

    value_cols = available_cols[1:]

    sns.set(style="white")
    plt.figure(figsize=(10, 6))

    if plot_average:
        palette = colors[:len(value_cols)] if colors is not None else None
        for idx, column in enumerate(value_cols):
            color = palette[idx] if palette is not None else None
            ax = sns.lineplot(
                data=average_data,
                x=average_data.index,
                y=column,
                label=column,
                linewidth=line_thickness,
                color=color,
            )
            if include_std and column in std_data.columns:
                plt.fill_between(
                    average_data.index,
                    average_data[column] - std_data[column],
                    average_data[column] + std_data[column],
                    alpha=0.2,
                    color=color,
                )

        if legend_labels:
            sns.move_legend(ax, "best", labels=legend_labels, title=None, frameon=True)
        if legend_fontsize:
            plb.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize)

    plt.xlabel(x_label, fontsize=axis_label_size)
    plt.ylabel(y_label, fontsize=axis_label_size)

    if show_title and plot_title:
        plt.title(plot_title, fontsize=axis_label_size + 2)

    if x_range:
        plt.xlim(x_range)
    if y_range:
        plt.ylim(y_range)

    if x_interval and x_range:
        plt.xticks(
            np.arange(x_range[0], x_range[1] + x_interval, x_interval),
            fontsize=tick_label_size,
        )
    if y_interval and y_range:
        plt.yticks(
            np.arange(y_range[0], y_range[1] + y_interval, y_interval),
            fontsize=tick_label_size,
        )

    plt.grid(False)
    plt.tight_layout()

    if save_as:
        os.makedirs(output_dir, exist_ok=True)
        suffix = f"_{scenario_filter}" if scenario_filter else ""
        filename = f"{save_as}{suffix}" if isinstance(save_as, str) else save_as
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path)
        print(f"Graph saved to: {save_path}")

    plt.show()


# ========================= Dam Scenario Comparison Plots ========================= #

SCENARIO_ORDER = ["S0", "S1", "S2", "S3"]
SCENARIO_PALETTE = {"S0": "#2ca02c", "S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#d62728"}

scenario_outcome_groups = [
    ["Stranded", "Injured", "Sheltered", "Hospitalized", "Death"],
    ["Houses_Flooded", "Businesses_Flooded", "Schools_Flooded"],
    ["Wealth_People", "Wealth_Businesses", "Wealth_Shelter",
     "Wealth_Healthcare", "Wealth_Government"],
    ["stranded_SES_1_0_0.3", "injured_SES_1_0_0.3", "dead_SES_1_0_0.3"],
    ["stranded_SES_1_0.7_1", "injured_SES_1_0.7_1", "dead_SES_1_0.7_1"],
    ["stranded_SES_2_0_0.3", "injured_SES_2_0_0.3", "dead_SES_2_0_0.3"],
    ["stranded_SES_2_0.7_1", "injured_SES_2_0.7_1", "dead_SES_2_0.7_1"],
    ["Evacuated", "evacuated_SES_1_0_0.3", "evacuated_SES_1_0.7_1"],
]

if "dam_scenario_name" in results_df.columns:
    print("\n=== Generating dam scenario comparison plots ===")

    max_day = results_df["Step"].max()
    xticks_scen = np.arange(0, max_day + 1, 7)

    for outcome_group in scenario_outcome_groups:
        outcomes_present = [c for c in outcome_group if c in results_df.columns]
        if not outcomes_present:
            continue

        needed_cols = ["Step", "dam_scenario_name"] + outcomes_present
        plot_df = results_df[needed_cols].copy()
        plot_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        grouped = (
            plot_df.groupby(["Step", "dam_scenario_name"])[outcomes_present]
            .mean()
            .reset_index()
        )

        fig, axes = plt.subplots(
            1, len(outcomes_present),
            figsize=(6 * len(outcomes_present), 5),
            sharey=False,
        )
        if len(outcomes_present) == 1:
            axes = [axes]

        for ax, outcome in zip(axes, outcomes_present):
            for scenario in SCENARIO_ORDER:
                subset = grouped[grouped["dam_scenario_name"] == scenario]
                if subset.empty:
                    continue
                ax.plot(
                    subset["Step"], subset[outcome],
                    label=scenario,
                    color=SCENARIO_PALETTE.get(scenario),
                    linewidth=2,
                )
            ax.set_xlabel("Days")
            ax.set_ylabel(outcome)
            ax.set_title(outcome)
            ax.set_xticks(xticks_scen)
            ax.legend(title="Dam Scenario")

        fig.tight_layout()
        group_tag = "_".join(outcomes_present[:2])
        save_path = os.path.join(output_dir, f"scenario_comparison_{group_tag}.png")
        fig.savefig(save_path)
        print(f"Scenario comparison graph saved to: {save_path}")
        plt.show()

    # Summary bar chart: peak value per scenario for key outcomes
    peak_outcomes = ["Stranded", "Injured", "Death", "Houses_Flooded"]
    peak_outcomes = [c for c in peak_outcomes if c in results_df.columns]

    if peak_outcomes:
        peak_df = (
            results_df.groupby("dam_scenario_name")[peak_outcomes]
            .max()
            .reindex(SCENARIO_ORDER)
            .reset_index()
        )

        fig, axes = plt.subplots(
            1, len(peak_outcomes), figsize=(5 * len(peak_outcomes), 5)
        )
        if len(peak_outcomes) == 1:
            axes = [axes]

        for ax, outcome in zip(axes, peak_outcomes):
            bar_colors = [
                SCENARIO_PALETTE.get(s, "grey") for s in peak_df["dam_scenario_name"]
            ]
            ax.bar(
                peak_df["dam_scenario_name"], peak_df[outcome],
                color=bar_colors, edgecolor="black",
            )
            ax.set_xlabel("Dam Scenario")
            ax.set_ylabel(f"Peak {outcome}")
            ax.set_title(f"Peak {outcome} by Scenario")

        fig.suptitle(
            "Peak Outcome Comparison Across Dam Scenarios", fontsize=13, y=1.02
        )
        fig.tight_layout()
        bar_save_path = os.path.join(output_dir, "peak_outcomes_bar.png")
        fig.savefig(bar_save_path)
        print(f"Peak outcomes bar chart saved to: {bar_save_path}")
        plt.show()

else:
    print(
        "\n[WARNING] 'dam_scenario_name' column not found. "
        "Skipping scenario comparison plots."
    )


# ====================== Individual Group Line Plots ====================== #
#
# Each call to plot_graphs below uses scenario_filter=None to average across
# all scenarios. Pass scenario_filter="S0" (or S1/S2/S3) to plot a single
# scenario instead.
#
# ======================================================================== #

group_num = 0
x_label = "Days"
y_label = "Percentage of population"
legend_labels = ["Evacuations", "Pre-flood non-evacuation measures",
                 "During-flood coping actions", "Post-flood adaptation measures"]
legend_fontsize = 18
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.65)
x_interval = 7
y_interval = 0.1
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = True
plot_title = "."
show_title = False
save_filename = "all_phases"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 1
x_label = "Days"
y_label = "Percentage of population"
legend_labels = ["Stranded", "Health-compromised", "Sheltered", "Hospitalized", "Deceased"]
legend_fontsize = 22
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.3)
x_interval = 7
y_interval = 0.1
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = True
plot_title = "."
show_title = False
save_filename = "persons_effects"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 14
x_label = "Days"
y_label = "Proportion of flooded structures"
legend_labels = ["Homes", "Businesses", "Schools"]
legend_fontsize = "22"
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.8)
x_interval = 7
y_interval = 0.2
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = True
plot_title = "."
show_title = False
save_filename = "entity_effects"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 15
x_label = "Days"
y_label = "Relative wealth growth"
legend_labels = ["Persons", "Businesses", "Shelter", "Healthcare", "Government"]
legend_fontsize = "17"
line_thickness = 4
x_range = (0, 38)
y_range = (-0.5, 0.5)
x_interval = 7
y_interval = 0.2
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = True
plot_title = "."
show_title = False
save_filename = "wealth_effects"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 2
x_label = "Days"
y_label = "Evacuated persons proportions"
legend_labels = ["High SES (low-vulnerability)", "Low SES (high-vulnerability)"]
legend_fontsize = "22"
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.5)
x_interval = 7
y_interval = 0.1
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "evacuation_vul"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 4
x_label = "Days"
y_label = "Stranded persons proportions"
legend_labels = ["High SES (low-vulnerability)", "Low SES (high-vulnerability)"]
legend_fontsize = "22"
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.3)
x_interval = 7
y_interval = 0.1
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "stranded_vul"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 6
x_label = "Days"
y_label = "Health-compromised persons proportions"
legend_labels = ["High SES (low-vulnerability)", "Low SES (high-vulnerability)"]
legend_fontsize = "22"
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.5)
x_interval = 7
y_interval = 0.1
axis_label_size = 20
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "injured_vul"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 8
x_label = "Days"
y_label = "Hospitalized persons proportions"
legend_labels = ["High SES (low-vulnerability)", "Low SES (high-vulnerability)"]
legend_fontsize = "22"
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.5)
x_interval = 7
y_interval = 0.1
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "hospitalized_vul"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 10
x_label = "Days"
y_label = "Sheltered persons proportions"
legend_labels = ["High SES (low-vulnerability)", "Low SES (high-vulnerability)"]
legend_fontsize = "22"
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.5)
x_interval = 7
y_interval = 0.1
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "sheltered_vul"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 12
x_label = "Days"
y_label = "Deceased persons proportions"
legend_labels = ["High SES (low-vulnerability)", "Low SES (high-vulnerability)"]
legend_fontsize = "22"
line_thickness = 4
x_range = (0, 38)
y_range = (0, 0.5)
x_interval = 7
y_interval = 0.1
axis_label_size = 22
tick_label_size = 18
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "deceased_vul"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


# ---------- Decision model plots ----------

group_num = 17
x_label = "Days"
y_label = "Proportion of high-vulnerability agents"
legend_labels = ["PMT", "TPB", "SCT", "CRT"]
legend_fontsize = "22"
line_thickness = 4
x_range = (7, 14.1)
y_range = (0, 0.2)
x_interval = 7
y_interval = 0.05
axis_label_size = 22
tick_label_size = 22
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "nonevac_high_dec"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 16
x_label = "Days"
y_label = "Proportion of low-vulnerability agents"
legend_labels = ["PMT", "TPB", "SCT", "CRT"]
legend_fontsize = "22"
line_thickness = 4
x_range = (7, 14.1)
y_range = (0, 0.04)
x_interval = 7
y_interval = 0.025
axis_label_size = 22
tick_label_size = 22
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "nonevac_low_dec"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 20
x_label = "Days"
y_label = "Proportion of low-vulnerability agents"
legend_labels = ["PMT", "TPB", "SCT", "CRT"]
legend_fontsize = "22"
line_thickness = 4
x_range = (7, 24.1)
y_range = (0, 0.21)
x_interval = 7
y_interval = 0.07
axis_label_size = 22
tick_label_size = 22
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "evac_low_dec"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 21
x_label = "Days"
y_label = "Proportion of high-vulnerability agents"
legend_labels = ["PMT", "TPB", "SCT", "CRT"]
legend_fontsize = "22"
line_thickness = 4
x_range = (7, 24.1)
y_range = (0, 0.21)
x_interval = 7
y_interval = 0.07
axis_label_size = 22
tick_label_size = 22
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "evac_high_dec"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)

group_num = 24
x_label = "Days"
y_label = "Proportion of low-vulnerability agents"
legend_labels = ["PMT", "TPB", "SCT", "CRT"]
legend_fontsize = "22"
line_thickness = 4
x_range = (14, 24.1)
y_range = (0, 0.04)
x_interval = 7
y_interval = 0.01
axis_label_size = 22
tick_label_size = 22
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "during_low_dec"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)


group_num = 25
x_label = "Days"
y_label = "Proportion of high-vulnerability agents"
legend_labels = ["PMT", "TPB", "SCT", "CRT"]
legend_fontsize = "22"
line_thickness = 4
x_range = (14, 24.1)
y_range = (0, 0.08)
x_interval = 7
y_interval = 0.02
axis_label_size = 22
tick_label_size = 22
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "during_High_dec"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)

group_num = 28
x_label = "Days"
y_label = "Proportion of low-vulnerability agents"
legend_labels = ["PMT", "TPB", "SCT", "CRT"]
legend_fontsize = "22"
line_thickness = 4
x_range = (24, 38.2)
y_range = (0, 0.05)
x_interval = 7
y_interval = 0.01
axis_label_size = 22
tick_label_size = 22
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "post_low_dec"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)

group_num = 29
x_label = "Days"
y_label = "Proportion of high-vulnerability agents"
legend_labels = ["PMT", "TPB", "SCT", "CRT"]
legend_fontsize = "22"
line_thickness = 4
x_range = (24, 38.2)
y_range = (0, 0.15)
x_interval = 7
y_interval = 0.05
axis_label_size = 22
tick_label_size = 22
colors = sns.color_palette("deep")
plt_std = False
plot_title = "."
show_title = False
save_filename = "post_High_dec"

plot_graphs(results_df, output_dir, group_num, columns_to_include, x_label, y_label,
            legend_labels, legend_fontsize, line_thickness, x_range, y_range,
            x_interval, y_interval, axis_label_size, tick_label_size, colors,
            plot_average=True, include_std=plt_std, plot_title=plot_title,
            show_title=show_title, save_as=save_filename)