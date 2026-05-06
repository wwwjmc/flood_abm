"""
Batch Run Results Visualization Script

Purpose
-------
Generate batch-run output graphs that match the chart categories used in
flood_serverrun.py / the pasted serverrun chart setup, while keeping graphs
simple and readable.

Serverrun chart categories matched here:
    1. Decisions chart
    2. Persons chart
    3. Entities chart
    4. Economic chart
    5. Decision-theory charts by phase and SES range
    6. Grouped SES outcome charts for SES index 1 and SES index 2

Decision-theory colors:
    PMT = blue
    TPB = orange
    SCT = green
    CRT = red

Input CSVs expected:
    Preferred:
        ../run/data_collection/results_S0.csv
        ../run/data_collection/results_S1.csv
        ../run/data_collection/results_S2.csv
        ../run/data_collection/results_S3.csv

    Fallback:
        batchrun_results.csv in the same folder as this script

Output:
    PNG graphs saved in ./data_collection/serverrun_matched_graphs
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# ========================= Settings ========================= #

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(PROJECT_DIR, "..", "run", "data_collection")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data_collection", "serverrun_matched_graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use "S0", "S1", "S2", "S3", or None.
# Use "S3" to match your current serverrun default dam scenario.
TARGET_SCENARIO = "S3"

# Your model uses hourly steps; batch plots are easier to read in days.
CONVERT_STEP_TO_DAYS = True

DECISION_THEORY_COLORS = {
    "PMT": "blue",
    "TPB": "orange",
    "SCT": "green",
    "CRT": "red",
}

SERVER_STYLE_COLORS = {
    # General decision chart colors from serverrun
    "Preflood_Non_Evacuation_Measure_Implemented": "orange",
    "Evacuated": "green",
    "Duringflood_Coping_Action_Implemented": "red",
    "Postflood_Adaptation_Measures_Planned": "blue",

    # Persons chart colors from serverrun
    "Stranded": "red",
    "Injured": "orange",
    "Health-compromised": "orange",
    "Sheltered": "blue",
    "Hospitalized": "grey",
    "Death": "black",

    # Entities chart colors from serverrun
    "Houses_Flooded": "red",
    "Schools_Flooded": "orange",
    "Businesses_Flooded": "blue",

    # Economic chart colors from serverrun
    "Wealth_People": "blue",
    "Wealth_Businesses": "green",
    "Wealth_Shelter": "orange",
    "Wealth_Healthcare": "purple",
    "Wealth_Government": "red",

    # SES grouped chart colors from serverrun
    "SES_1_0_0.3": "green",
    "SES_1_0.7_1": "red",
    "SES_2_0_0.3": "blue",
    "SES_2_0.7_1": "magenta",
}

# ========================= Data Loading ========================= #

def load_results():
    """Load per-scenario CSVs if available; otherwise load batchrun_results.csv."""
    scenarios = ["S0", "S1", "S2", "S3"]
    all_dfs = []

    print("Reading scenario CSVs from:", CSV_DIR)
    for scenario in scenarios:
        path = os.path.join(CSV_DIR, f"results_{scenario}.csv")
        if os.path.exists(path):
            print("Loading:", path)
            df = pd.read_csv(path)
            df["dam_scenario_name"] = scenario
            all_dfs.append(df)
        else:
            print(f"[WARNING] File not found: {path}")

    if all_dfs:
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        fallback_path = os.path.join(PROJECT_DIR, "batchrun_results.csv")
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(
                "No scenario CSVs found and fallback batchrun_results.csv does not exist.\n"
                f"Checked folder: {CSV_DIR}\n"
                f"Checked fallback: {fallback_path}"
            )
        print("Loading fallback:", fallback_path)
        df = pd.read_csv(fallback_path)

    if CONVERT_STEP_TO_DAYS and "Step" in df.columns and df["Step"].max() > 100:
        df["Step"] = df["Step"] / 24

    return df

results_df = load_results()
print("Saving graphs to:", OUTPUT_DIR)

# ========================= Helpers ========================= #

def safe_filename(text):
    return (
        str(text)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace(".", "p")
        .replace(":", "")
    )


def filter_scenario(df, scenario_filter=TARGET_SCENARIO):
    if scenario_filter is not None and "dam_scenario_name" in df.columns:
        filtered = df[df["dam_scenario_name"] == scenario_filter].copy()
        if filtered.empty:
            print(f"[WARNING] No rows found for scenario_filter={scenario_filter}.")
        return filtered
    return df.copy()


def prepare_average_data(df, columns, scenario_filter=TARGET_SCENARIO):
    plot_df = filter_scenario(df, scenario_filter=scenario_filter)
    available = [c for c in columns if c in plot_df.columns]

    if "Step" not in available:
        print("[WARNING] Step column missing. Skipping plot.")
        return None, []

    value_cols = [c for c in available if c != "Step"]
    if not value_cols:
        print(f"[WARNING] No value columns available from: {columns}")
        return None, []

    numeric_data = plot_df[["Step"] + value_cols].copy()
    numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    average_data = numeric_data.groupby("Step").mean()
    return average_data, value_cols


def plot_line_group(
    df,
    columns,
    save_name,
    y_label,
    colors=None,
    legend_labels=None,
    x_range=(0, 38),
    y_range=None,
    x_interval=7,
    y_interval=None,
    scenario_filter=TARGET_SCENARIO,
    one_line_per_figure=False,
):
    """
    Plot a serverrun-equivalent chart category.

    If one_line_per_figure=True, each variable is saved as its own figure.
    If False, variables are grouped in one figure like serverrun ChartModule.
    """
    average_data, value_cols = prepare_average_data(df, columns, scenario_filter)
    if average_data is None:
        return

    if one_line_per_figure:
        for col in value_cols:
            plot_line_group(
                df=df,
                columns=["Step", col],
                save_name=f"{save_name}_{safe_filename(col)}",
                y_label=y_label,
                colors={col: colors.get(col) if isinstance(colors, dict) else None} if colors else None,
                legend_labels={col: legend_labels.get(col) if isinstance(legend_labels, dict) else col} if legend_labels else None,
                x_range=x_range,
                y_range=y_range,
                x_interval=x_interval,
                y_interval=y_interval,
                scenario_filter=scenario_filter,
                one_line_per_figure=False,
            )
        return

    plt.figure(figsize=(10, 6))
    sns.set(style="white")

    for col in value_cols:
        label = legend_labels.get(col, col) if isinstance(legend_labels, dict) else col
        color = colors.get(col, None) if isinstance(colors, dict) else None
        sns.lineplot(
            data=average_data,
            x=average_data.index,
            y=col,
            label=label,
            color=color,
            linewidth=4,
        )

    plt.xlabel("Days", fontsize=22)
    plt.ylabel(y_label, fontsize=22)

    if x_range:
        plt.xlim(x_range)
        if x_interval:
            plt.xticks(np.arange(x_range[0], x_range[1] + x_interval, x_interval), fontsize=18)
    else:
        plt.xticks(fontsize=18)

    if y_range:
        plt.ylim(y_range)
        if y_interval:
            plt.yticks(np.arange(y_range[0], y_range[1] + y_interval, y_interval), fontsize=18)
    else:
        plt.yticks(fontsize=18)

    plt.legend(fontsize=14)
    plt.grid(False)
    plt.tight_layout()

    suffix = f"_{scenario_filter}" if scenario_filter else ""
    save_path = os.path.join(OUTPUT_DIR, f"{save_name}{suffix}.png")
    plt.savefig(save_path)
    print("Graph saved to:", save_path)
    plt.show()
    plt.close()

# ========================= Serverrun-Matched Graph Inventory ========================= #

# Set this to True if you still want literally one variable per image.
# Set False to match serverrun ChartModule grouping more closely.
ONE_LINE_PER_FIGURE_FOR_GENERAL_CHARTS = False

# ---------------- General charts, matching serverrun ---------------- #

# Serverrun: decisions_chart
plot_line_group(
    df=results_df,
    columns=[
        "Step",
        "Preflood_Non_Evacuation_Measure_Implemented",
        "Evacuated",
        "Duringflood_Coping_Action_Implemented",
        "Postflood_Adaptation_Measures_Planned",
    ],
    save_name="server_decisions_chart",
    y_label="Percentage of population",
    colors=SERVER_STYLE_COLORS,
    legend_labels={
        "Preflood_Non_Evacuation_Measure_Implemented": "Pre-flood non-evacuation measures",
        "Evacuated": "Evacuations",
        "Duringflood_Coping_Action_Implemented": "During-flood coping actions",
        "Postflood_Adaptation_Measures_Planned": "Post-flood adaptation measures",
    },
    x_range=(0, 38),
    y_range=(0, 0.65),
    x_interval=7,
    y_interval=0.1,
    one_line_per_figure=ONE_LINE_PER_FIGURE_FOR_GENERAL_CHARTS,
)

# Serverrun: persons_chart
plot_line_group(
    df=results_df,
    columns=["Step", "Stranded", "Injured", "Sheltered", "Hospitalized", "Death"],
    save_name="server_persons_chart",
    y_label="Percentage of population",
    colors=SERVER_STYLE_COLORS,
    legend_labels={
        "Stranded": "Stranded",
        "Injured": "Health-compromised",
        "Sheltered": "Sheltered",
        "Hospitalized": "Hospitalized",
        "Death": "Deceased",
    },
    x_range=(0, 38),
    y_range=(0, 0.3),
    x_interval=7,
    y_interval=0.1,
    one_line_per_figure=ONE_LINE_PER_FIGURE_FOR_GENERAL_CHARTS,
)

# Serverrun: entities_chart
plot_line_group(
    df=results_df,
    columns=["Step", "Houses_Flooded", "Schools_Flooded", "Businesses_Flooded"],
    save_name="server_entities_chart",
    y_label="Proportion of flooded structures",
    colors=SERVER_STYLE_COLORS,
    legend_labels={
        "Houses_Flooded": "Homes",
        "Schools_Flooded": "Schools",
        "Businesses_Flooded": "Businesses",
    },
    x_range=(0, 38),
    y_range=(0, 0.8),
    x_interval=7,
    y_interval=0.2,
    one_line_per_figure=ONE_LINE_PER_FIGURE_FOR_GENERAL_CHARTS,
)

# Serverrun: economic_chart
plot_line_group(
    df=results_df,
    columns=[
        "Step",
        "Wealth_People",
        "Wealth_Businesses",
        "Wealth_Shelter",
        "Wealth_Healthcare",
        "Wealth_Government",
    ],
    save_name="server_economic_chart",
    y_label="Relative wealth growth",
    colors=SERVER_STYLE_COLORS,
    legend_labels={
        "Wealth_People": "Persons",
        "Wealth_Businesses": "Businesses",
        "Wealth_Shelter": "Shelter",
        "Wealth_Healthcare": "Healthcare",
        "Wealth_Government": "Government",
    },
    x_range=(0, 38),
    y_range=(-0.5, 0.5),
    x_interval=7,
    y_interval=0.2,
    one_line_per_figure=ONE_LINE_PER_FIGURE_FOR_GENERAL_CHARTS,
)

# ---------------- Decision-theory charts by SES range ---------------- #
# Serverrun create_ses_charts() creates one chart per decision phase + SES range.
# Here the same chart kinds are produced, with requested theory colors:
# PMT blue, TPB orange, SCT green, CRT red.

SES_RANGES = ["SES_1_0_0.3", "SES_1_0.7_1", "SES_2_0_0.3", "SES_2_0.7_1"]

DECISION_PHASES = [
    {
        "decision": "preflood_non_evacuation_measure_implemented",
        "save_prefix": "server_preflood_non_evacuation_decision",
        "y_label": "Proportion of agents",
        "x_range": (7, 14.1),
        "y_range": None,
        "x_interval": 7,
        "y_interval": None,
    },
    {
        "decision": "evacuation",
        "save_prefix": "server_evacuation_decision",
        "y_label": "Proportion of agents",
        "x_range": (7, 24.1),
        "y_range": None,
        "x_interval": 7,
        "y_interval": None,
    },
    {
        "decision": "duringflood_coping_action_implemented",
        "save_prefix": "server_duringflood_coping_decision",
        "y_label": "Proportion of agents",
        "x_range": (14, 24.1),
        "y_range": None,
        "x_interval": 7,
        "y_interval": None,
    },
    {
        "decision": "postflood_adaptation_measures_planned",
        "save_prefix": "server_postflood_adaptation_decision",
        "y_label": "Proportion of agents",
        "x_range": (24, 38.2),
        "y_range": None,
        "x_interval": 7,
        "y_interval": None,
    },
]

for phase in DECISION_PHASES:
    for ses_range in SES_RANGES:
        decision_cols = [
            "Step",
            f"PMT_{phase['decision']}_{ses_range}",
            f"TPB_{phase['decision']}_{ses_range}",
            f"SCT_{phase['decision']}_{ses_range}",
            f"CRT_{phase['decision']}_{ses_range}",
        ]
        plot_line_group(
            df=results_df,
            columns=decision_cols,
            save_name=f"{phase['save_prefix']}_{safe_filename(ses_range)}",
            y_label=phase["y_label"],
            colors={
                f"PMT_{phase['decision']}_{ses_range}": DECISION_THEORY_COLORS["PMT"],
                f"TPB_{phase['decision']}_{ses_range}": DECISION_THEORY_COLORS["TPB"],
                f"SCT_{phase['decision']}_{ses_range}": DECISION_THEORY_COLORS["SCT"],
                f"CRT_{phase['decision']}_{ses_range}": DECISION_THEORY_COLORS["CRT"],
            },
            legend_labels={
                f"PMT_{phase['decision']}_{ses_range}": "PMT",
                f"TPB_{phase['decision']}_{ses_range}": "TPB",
                f"SCT_{phase['decision']}_{ses_range}": "SCT",
                f"CRT_{phase['decision']}_{ses_range}": "CRT",
            },
            x_range=phase["x_range"],
            y_range=phase["y_range"],
            x_interval=phase["x_interval"],
            y_interval=phase["y_interval"],
            one_line_per_figure=False,
        )

# ---------------- Grouped SES outcome charts ---------------- #
# Serverrun creates two grouped charts per metric: one for SES index 1 and one for SES index 2.

SES_OUTCOME_METRICS = [
    {
        "metric": "evacuated",
        "save_prefix": "server_evacuated_ses",
        "y_label": "Evacuated persons proportions",
        "y_range": (0, 0.5),
    },
    {
        "metric": "stranded",
        "save_prefix": "server_stranded_ses",
        "y_label": "Stranded persons proportions",
        "y_range": (0, 0.3),
    },
    {
        "metric": "injured",
        "save_prefix": "server_injured_ses",
        "y_label": "Health-compromised persons proportions",
        "y_range": (0, 0.5),
    },
    {
        "metric": "sheltered",
        "save_prefix": "server_sheltered_ses",
        "y_label": "Sheltered persons proportions",
        "y_range": (0, 0.5),
    },
    {
        "metric": "hospitalized",
        "save_prefix": "server_hospitalized_ses",
        "y_label": "Hospitalized persons proportions",
        "y_range": (0, 0.5),
    },
    {
        "metric": "dead",
        "save_prefix": "server_dead_ses",
        "y_label": "Deceased persons proportions",
        "y_range": (0, 0.5),
    },
]

SES_GROUPS = {
    "SES_1": ["SES_1_0_0.3", "SES_1_0.7_1"],
    "SES_2": ["SES_2_0_0.3", "SES_2_0.7_1"],
}

for metric_cfg in SES_OUTCOME_METRICS:
    metric = metric_cfg["metric"]
    for ses_index, ses_ranges in SES_GROUPS.items():
        columns = ["Step"] + [f"{metric}_{ses_range}" for ses_range in ses_ranges]
        plot_line_group(
            df=results_df,
            columns=columns,
            save_name=f"{metric_cfg['save_prefix']}_{ses_index}",
            y_label=metric_cfg["y_label"],
            colors={
                f"{metric}_{ses_ranges[0]}": SERVER_STYLE_COLORS[ses_ranges[0]],
                f"{metric}_{ses_ranges[1]}": SERVER_STYLE_COLORS[ses_ranges[1]],
            },
            legend_labels={
                f"{metric}_{ses_ranges[0]}": f"{ses_ranges[0]}",
                f"{metric}_{ses_ranges[1]}": f"{ses_ranges[1]}",
            },
            x_range=(0, 38),
            y_range=metric_cfg["y_range"],
            x_interval=7,
            y_interval=0.1,
            one_line_per_figure=False,
        )

print("\nDone. Serverrun-matched batch graphs saved in:", OUTPUT_DIR)