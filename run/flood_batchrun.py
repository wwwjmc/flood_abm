"""
Batch Run Script

This script runs multiple simulations of the flood model with varying parameters 
to analyze the effects of different scenarios. It collects results and saves 
them to a CSV file for further analysis.

Output:
Data saved in 'data_collection/batchrun_results.csv'.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.flood_model import FloodModel

from mesa.batchrunner import batch_run
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import warnings
import time
import psutil

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")


# Function to get memory and CPU usage
def get_resource_usage():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    cpu_usage = process.cpu_percent(interval=None)  # CPU usage as a percentage
    return memory_usage, cpu_usage


# =============================== BatchRun ================================

# Ensure output directory exists
os.makedirs("data_collection", exist_ok=True)

# Start the timer
start_time = time.time()

# Measure resource usage before the run
mem_before, cpu_before = get_resource_usage()
print(f"Memory usage before batch run: {mem_before:.2f} MB")
print(f"CPU usage before batch run: {cpu_before:.2f}%")

batch_run_params = {
    "N_persons": [100],
    "dam_scenario_name": ["S0", "S1", "S2", "S3"],
    "shelter_cap_limit": [1],
    "healthcare_cap_limit": [5],
    "shelter_funding": [50000],
    "healthcare_funding": [100000],
    "pre_flood_days": [14],
    "flood_days": [10],
    "post_flood_days": [14],

    "houses_file": "../malolos_map_data/houses.zip",
    "businesses_file": "../malolos_map_data/business.zip",
    "schools_file": "../malolos_map_data/schools.zip",
    "shelter_file": "../malolos_map_data/evacuation_centers.zip",
    "healthcare_file": "../malolos_map_data/healthcare.zip",
    "government_file": "../malolos_map_data/government.zip",
    "flood_file_1": "../malolos_map_data/flood1.zip",
    "flood_file_2": "../malolos_map_data/flood2.zip",
    "flood_file_3": "../malolos_map_data/flood3.zip",
    "dam_scenarios_file": "../malolos_map_data/dam_scenarios_simple.csv",
    "merged_dams_file": "../malolos_map_data/merged_dams.zip",
    "malolos_hydrorivers_file": "../malolos_map_data/malolos_hydrorivers.zip",
    "malolos_channels_file": "../malolos_map_data/malolos_channels.zip",
    "model_crs": "EPSG:32651"
}

# Create and run the batch
all_results = []

for scenario in ["S0", "S1", "S2", "S3"]:
    print(f"Running scenario: {scenario}")

    params = batch_run_params.copy()
    params["dam_scenario_name"] = [scenario]

    results = batch_run(
        FloodModel,
        params,
        iterations=1,
        max_steps=24 * 38,
        number_processes=1,
        data_collection_period=1,
        display_progress=True
    )

    df = pd.DataFrame(results)
    df["dam_scenario_name"] = scenario

    # Save per scenario
    df.to_csv(f"data_collection/results_{scenario}.csv", index=False)

    all_results.append(df)

# Combine all scenario results
results_df = pd.concat(all_results, ignore_index=True)

# Convert steps to days
results_df["Step"] = results_df["Step"] / 24

# Specify the columns you want to include in the analysis
columns_to_include = [
    ["Step", "Evacuated", "Preflood_Non_Evacuation_Measure_Implemented", "Duringflood_Coping_Action_Implemented", "Postflood_Adaptation_Measures_Planned"],
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
    ["Step", "Wealth_People", "Wealth_Businesses", "Wealth_Shelter", "Wealth_Healthcare", "Wealth_Government"],
    ["Step", "PMT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3", "TPB_preflood_non_evacuation_measure_implemented_SES_1_0_0.3", "SCT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3", "CRT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3"],
    ["Step", "PMT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1", "TPB_preflood_non_evacuation_measure_implemented_SES_1_0.7_1", "SCT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1", "CRT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1"],
    ["Step", "PMT_preflood_non_evacuation_measure_implemented_SES_2_0_0.3", "TPB_preflood_non_evacuation_measure_implemented_SES_2_0_0.3", "SCT_preflood_non_evacuation_measure_implemented_SES_2_0_0.3", "CRT_preflood_non_evacuation_measure_implemented_SES_2_0_0.3"],
    ["Step", "PMT_preflood_non_evacuation_measure_implemented_SES_2_0.7_1", "TPB_preflood_non_evacuation_measure_implemented_SES_2_0.7_1", "SCT_preflood_non_evacuation_measure_implemented_SES_2_0.7_1", "CRT_preflood_non_evacuation_measure_implemented_SES_2_0.7_1"],
    ["Step", "PMT_evacuation_SES_1_0_0.3", "TPB_evacuation_SES_1_0_0.3", "SCT_evacuation_SES_1_0_0.3", "CRT_evacuation_SES_1_0_0.3"],
    ["Step", "PMT_evacuation_SES_1_0.7_1", "TPB_evacuation_SES_1_0.7_1", "SCT_evacuation_SES_1_0.7_1", "CRT_evacuation_SES_1_0.7_1"],
    ["Step", "PMT_evacuation_SES_2_0_0.3", "TPB_evacuation_SES_2_0_0.3", "SCT_evacuation_SES_2_0_0.3", "CRT_evacuation_SES_2_0_0.3"],
    ["Step", "PMT_evacuation_SES_2_0.7_1", "TPB_evacuation_SES_2_0.7_1", "SCT_evacuation_SES_2_0.7_1", "CRT_evacuation_SES_2_0.7_1"],
    ["Step", "PMT_duringflood_coping_action_implemented_SES_1_0_0.3", "TPB_duringflood_coping_action_implemented_SES_1_0_0.3", "SCT_duringflood_coping_action_implemented_SES_1_0_0.3", "CRT_duringflood_coping_action_implemented_SES_1_0_0.3"],
    ["Step", "PMT_duringflood_coping_action_implemented_SES_1_0.7_1", "TPB_duringflood_coping_action_implemented_SES_1_0.7_1", "SCT_duringflood_coping_action_implemented_SES_1_0.7_1", "CRT_duringflood_coping_action_implemented_SES_1_0.7_1"],
    ["Step", "PMT_duringflood_coping_action_implemented_SES_2_0_0.3", "TPB_duringflood_coping_action_implemented_SES_2_0_0.3", "SCT_duringflood_coping_action_implemented_SES_2_0_0.3", "CRT_duringflood_coping_action_implemented_SES_2_0_0.3"],
    ["Step", "PMT_duringflood_coping_action_implemented_SES_2_0.7_1", "TPB_duringflood_coping_action_implemented_SES_2_0.7_1", "SCT_duringflood_coping_action_implemented_SES_2_0.7_1", "CRT_duringflood_coping_action_implemented_SES_2_0.7_1"],
    ["Step", "PMT_postflood_adaptation_measures_planned_SES_1_0_0.3", "TPB_postflood_adaptation_measures_planned_SES_1_0_0.3", "SCT_postflood_adaptation_measures_planned_SES_1_0_0.3", "CRT_postflood_adaptation_measures_planned_SES_1_0_0.3"],
    ["Step", "PMT_postflood_adaptation_measures_planned_SES_1_0.7_1", "TPB_postflood_adaptation_measures_planned_SES_1_0.7_1", "SCT_postflood_adaptation_measures_planned_SES_1_0.7_1", "CRT_postflood_adaptation_measures_planned_SES_1_0.7_1"],
    ["Step", "PMT_postflood_adaptation_measures_planned_SES_2_0_0.3", "TPB_postflood_adaptation_measures_planned_SES_2_0_0.3", "SCT_postflood_adaptation_measures_planned_SES_2_0_0.3", "CRT_postflood_adaptation_measures_planned_SES_2_0_0.3"],
    ["Step", "PMT_postflood_adaptation_measures_planned_SES_2_0.7_1", "TPB_postflood_adaptation_measures_planned_SES_2_0.7_1", "SCT_postflood_adaptation_measures_planned_SES_2_0.7_1", "CRT_postflood_adaptation_measures_planned_SES_2_0.7_1"]
]

# ======================== Plot batch run results ========================= #
for i, group_columns in enumerate(columns_to_include):
    numeric_data = results_df[group_columns].select_dtypes(include='number')

    # Replace infinite values with NaN
    numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Group by the "Step" column
    grouped_data = numeric_data.groupby("Step")

    # Calculate the mean and standard deviation for each group
    average_data = grouped_data.mean()
    std_data = grouped_data.std()

    # Define the x-tick intervals (every 7 days)
    xticks = np.arange(0, max(average_data.index) + 1, 7)

    # Lineplot with error bands
    plt.figure(figsize=(10, 6))
    for column in group_columns[1:]:
        sns.lineplot(
            data=average_data,
            x="Step",
            y=column,
            label=column,
        )
        plt.fill_between(
            average_data.index,
            average_data[column] - std_data[column],
            average_data[column] + std_data[column],
            alpha=0.2
        )

    plt.xlabel("Days")
    plt.xticks(xticks)
    plt.legend()
    plt.show()

    # Lineplot without error bands
    plt.figure(figsize=(10, 6))
    for column in group_columns[1:]:
        sns.lineplot(
            data=average_data,
            x="Step",
            y=column,
            label=column,
        )

    plt.xlabel("Days")
    plt.xticks(xticks)
    plt.legend()
    plt.show()

# ======================== Dam Scenario Comparison Plots ========================= #

SCENARIO_ORDER = ["S0", "S1", "S2", "S3"]
SCENARIO_PALETTE = {"S0": "#2ca02c", "S1": "#1f77b4", "S2": "#ff7f0e", "S3": "#d62728"}

scenario_outcome_groups = [
    ["Stranded", "Injured", "Sheltered", "Hospitalized", "Death"],
    ["Houses_Flooded", "Businesses_Flooded", "Schools_Flooded"],
    ["Wealth_People", "Wealth_Businesses", "Wealth_Shelter", "Wealth_Healthcare", "Wealth_Government"],
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
            sharey=False
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
                    linewidth=2
                )
            ax.set_xlabel("Days")
            ax.set_ylabel(outcome)
            ax.set_title(outcome)
            ax.set_xticks(xticks_scen)
            ax.legend(title="Dam Scenario")

        fig.tight_layout()
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

        fig, axes = plt.subplots(1, len(peak_outcomes), figsize=(5 * len(peak_outcomes), 5))
        if len(peak_outcomes) == 1:
            axes = [axes]

        for ax, outcome in zip(axes, peak_outcomes):
            colors = [SCENARIO_PALETTE.get(s, "grey") for s in peak_df["dam_scenario_name"]]
            ax.bar(peak_df["dam_scenario_name"], peak_df[outcome], color=colors, edgecolor="black")
            ax.set_xlabel("Dam Scenario")
            ax.set_ylabel(f"Peak {outcome}")
            ax.set_title(f"Peak {outcome} by Scenario")

        fig.suptitle("Peak Outcome Comparison Across Dam Scenarios", fontsize=13, y=1.02)
        fig.tight_layout()
        plt.show()

else:
    print(
        "\n[WARNING] 'dam_scenario_name' column not found in results. "
        "Skipping scenario comparison plots. "
        "Make sure FloodModel accepts dam_scenario_name as a constructor argument."
    )

# Measure resource usage after the run
mem_after, cpu_after = get_resource_usage()
print(f"Memory usage after batch run: {mem_after:.2f} MB")
print(f"CPU usage after batch run: {cpu_after:.2f}%")

# Elapsed time calculation
end_time = time.time()
elapsed_time = (end_time - start_time) / 60
print(f"Batch run completed in {elapsed_time:.2f} minutes.")
print(f"Total memory used: {mem_after - mem_before:.2f} MB")