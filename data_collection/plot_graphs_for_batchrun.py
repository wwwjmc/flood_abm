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
OUTPUT_DIR = os.path.join(PROJECT_DIR, "graphs_steps")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Use "S0", "S1", "S2", "S3", or None.
TARGET_SCENARIO = "S3"
CONVERT_STEPS_TO_DAYS = True

DECISION_THEORY_COLORS = {
    "PMT": "blue",
    "TPB": "orange",
    "SCT": "green",
    "CRT": "red",
}

# ========================= Load data ========================= #

def load_results():
    scenarios = ["S0", "S1", "S2", "S3"]
    all_dfs = []

    print(f"Target scenario: {TARGET_SCENARIO}")
    for s in scenarios:
        path = os.path.join(CSV_DIR, f"results_{s}.csv")
        print("Loading:", path)
        if not os.path.exists(path):
            print(f"[WARNING] File not found: {path}")
            continue
        df = pd.read_csv(path)
        df["dam_scenario_name"] = s
        all_dfs.append(df)

    if not all_dfs:
        fallback = os.path.join(PROJECT_DIR, "batchrun_results.csv")
        if not os.path.exists(fallback):
            raise FileNotFoundError(
                f"No scenario CSVs found in {CSV_DIR} and no fallback file at {fallback}"
            )
        print("Loading fallback:", fallback)
        df = pd.read_csv(fallback)
    else:
        df = pd.concat(all_dfs, ignore_index=True)

    if CONVERT_STEPS_TO_DAYS and "Step" in df.columns and df["Step"].max() > 100:
        df["Step"] = df["Step"] / 24

    return df


results_df = load_results()

# ========================= Helpers ========================= #

def filter_df(df, scenario_filter=TARGET_SCENARIO):
    if scenario_filter is not None and "dam_scenario_name" in df.columns:
        out = df[df["dam_scenario_name"] == scenario_filter].copy()
        if out.empty:
            print(f"[WARNING] No rows found for scenario {scenario_filter}")
        return out
    return df.copy()


def prepare_group_average(df, columns, scenario_filter=TARGET_SCENARIO):
    plot_df = filter_df(df, scenario_filter)
    available = [c for c in columns if c in plot_df.columns]

    if "Step" not in available:
        print("[WARNING] 'Step' not found. Skipping plot.")
        return None, []

    value_cols = [c for c in available if c != "Step"]
    if not value_cols:
        print(f"[WARNING] No valid value columns found in {columns}")
        return None, []

    numeric_data = plot_df[["Step"] + value_cols].copy()
    numeric_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    avg = numeric_data.groupby("Step").mean()
    return avg, value_cols


def plot_multi_line_graph(
    df,
    columns,
    save_filename,
    y_label,
    legend_labels=None,
    colors=None,
    x_range=None,
    y_range=None,
    x_interval=None,
    y_interval=None,
    axis_label_size=22,
    tick_label_size=18,
    line_thickness=4,
    legend_fontsize=18,
    scenario_filter=TARGET_SCENARIO,
    plot_title=None,
):
    avg, value_cols = prepare_group_average(df, columns, scenario_filter)
    if avg is None:
        return

    plt.figure(figsize=(10, 6))
    sns.set(style="white")

    for col in value_cols:
        label = legend_labels.get(col, col) if isinstance(legend_labels, dict) else col
        color = colors.get(col) if isinstance(colors, dict) else None
        sns.lineplot(
            data=avg,
            x=avg.index,
            y=col,
            label=label,
            linewidth=line_thickness,
            color=color,
        )

    plt.xlabel("Days", fontsize=axis_label_size)
    plt.ylabel(y_label, fontsize=axis_label_size)
    if plot_title:
        plt.title(plot_title, fontsize=axis_label_size + 1)

    if x_range is not None:
        plt.xlim(x_range)
        if x_interval is not None:
            plt.xticks(np.arange(x_range[0], x_range[1] + x_interval, x_interval), fontsize=tick_label_size)
    else:
        plt.xticks(fontsize=tick_label_size)

    if y_range is not None:
        plt.ylim(y_range)
        if y_interval is not None:
            plt.yticks(np.arange(y_range[0], y_range[1] + y_interval, y_interval), fontsize=tick_label_size)
    else:
        plt.yticks(fontsize=tick_label_size)

    plt.legend(fontsize=legend_fontsize)
    plt.grid(False)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")
    plt.show()
    plt.close()


def plot_single_theory_graph(
    df,
    column_name,
    model_name,
    save_filename,
    y_label,
    x_range=None,
    y_range=None,
    x_interval=None,
    y_interval=None,
    scenario_filter=TARGET_SCENARIO,
    legend_label=None,
    plot_title=None,
    legend_fontsize=18,
):
    avg, value_cols = prepare_group_average(df, ["Step", column_name], scenario_filter)
    if avg is None or not value_cols:
        return

    plt.figure(figsize=(10, 6))
    sns.set(style="white")

    sns.lineplot(
        data=avg,
        x=avg.index,
        y=column_name,
        label=legend_label if legend_label else model_name,
        linewidth=4,
        color=DECISION_THEORY_COLORS.get(model_name),
    )

    plt.xlabel("Days", fontsize=22)
    plt.ylabel(y_label, fontsize=22)
    if plot_title:
        plt.title(plot_title, fontsize=23)

    if x_range is not None:
        plt.xlim(x_range)
        if x_interval is not None:
            plt.xticks(np.arange(x_range[0], x_range[1] + x_interval, x_interval), fontsize=22)
    else:
        plt.xticks(fontsize=22)

    if y_range is not None:
        plt.ylim(y_range)
        if y_interval is not None:
            plt.yticks(np.arange(y_range[0], y_range[1] + y_interval, y_interval), fontsize=22)
    else:
        plt.yticks(fontsize=22)

    plt.legend(fontsize=legend_fontsize)
    plt.grid(False)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, save_filename)
    plt.savefig(save_path)
    print(f"Graph saved to: {save_path}")
    plt.show()
    plt.close()


def plot_combined_and_separate_decision_group(
    df,
    decision_columns,
    save_prefix,
    y_label,
    x_range,
    y_range,
    x_interval,
    y_interval,
    phase_label,
    group_label,
    scenario_filter=TARGET_SCENARIO,
):
    # combined graph
    plot_multi_line_graph(
        df=df,
        columns=["Step"] + list(decision_columns.values()),
        save_filename=f"{save_prefix}.png",
        y_label=y_label,
        legend_labels={v: k for k, v in decision_columns.items()},
        colors={v: DECISION_THEORY_COLORS[k] for k, v in decision_columns.items()},
        x_range=x_range,
        y_range=y_range,
        x_interval=x_interval,
        y_interval=y_interval,
        axis_label_size=22,
        tick_label_size=22,
        line_thickness=4,
        scenario_filter=scenario_filter,
        plot_title=f"{phase_label} {group_label}",
    )

    # separate PMT/TPB/SCT/CRT graphs
    for model_name, column_name in decision_columns.items():
        plot_single_theory_graph(
            df=df,
            column_name=column_name,
            model_name=model_name,
            save_filename=f"{save_prefix}_{model_name}.png",
            y_label=y_label,
            x_range=x_range,
            y_range=y_range,
            x_interval=x_interval,
            y_interval=y_interval,
            scenario_filter=scenario_filter,
            legend_label=f"{model_name} ({phase_label})",
            plot_title=f"{model_name} {phase_label} {group_label}",
        )

# ========================= Required outputs ========================= #

# 1) All Phases
plot_multi_line_graph(
    df=results_df,
    columns=[
        "Step",
        "Preflood_Non_Evacuation_Measure_Implemented",
        "Evacuated",
        "Duringflood_Coping_Action_Implemented",
        "Postflood_Adaptation_Measures_Planned",
    ],
    save_filename="all_phases.png",
    y_label="Percentage of population",
    legend_fontsize=16,
    legend_labels={
        "Preflood_Non_Evacuation_Measure_Implemented": "Preflood_Non_Evacuation_Measure_Implemented",
        "Evacuated": "Evacuated",
        "Duringflood_Coping_Action_Implemented": "Duringflood_Coping_Action_Implemented",
        "Postflood_Adaptation_Measures_Planned": "Postflood_Adaptation_Measures_Planned",
    },
    colors={
        "Preflood_Non_Evacuation_Measure_Implemented": "orange",
        "Evacuated": "green",
        "Duringflood_Coping_Action_Implemented": "red",
        "Postflood_Adaptation_Measures_Planned": "blue",
    },
    x_range=(0, 38),
    y_range=(0, 0.65),
    x_interval=7,
    y_interval=0.1
)

# 2) Person Effects
plot_multi_line_graph(
    df=results_df,
    columns=["Step", "Stranded", "Injured", "Sheltered", "Hospitalized", "Death"],
    save_filename="persons_effects.png",
    y_label="Percentage of population",
    legend_labels={
        "Stranded": "Stranded",
        "Injured": "Health-compromised",
        "Sheltered": "Sheltered",
        "Hospitalized": "Hospitalized",
        "Death": "Dead",
    },
    colors={
        "Stranded": "red",
        "Injured": "orange",
        "Sheltered": "blue",
        "Hospitalized": "grey",
        "Death": "black",
    },
    x_range=(0, 21),
    y_range=(0, 0.045),
    x_interval=7,
    y_interval=0.005
)

# 3) Entity Effects
plot_multi_line_graph(
    df=results_df,
    columns=["Step", "Houses_Flooded", "Businesses_Flooded", "Schools_Flooded"],
    save_filename="entity_effects.png",
    y_label="Proportion of flooded structures",
    legend_labels={
        "Houses_Flooded": "Houses_Flooded",
        "Businesses_Flooded": "Businesses_Flooded",
        "Schools_Flooded": "Schools_Flooded",
    },
    colors={
        "Houses_Flooded": "red",
        "Businesses_Flooded": "blue",
        "Schools_Flooded": "orange",
    },
    x_range=(0, 21),
    y_range=(0, 0.09),
    x_interval=7,
    y_interval=0.025
)

# 4) SES vulnerability charts
ses_group_specs = [
    {
        "columns": ["Step", "evacuated_SES_1_0_0.3", "evacuated_SES_1_0.7_1"],
        "save_filename": "evacuation_vul.png",
        "y_label": "Evacuated persons proportions",
        "legend_labels": {
            "evacuated_SES_1_0_0.3": "High SES (low-vulnerability)",
            "evacuated_SES_1_0.7_1": "Low SES (high-vulnerability)",
        },
        "y_range": (0, 0.5),
        "plot_title": "Evacuation by Vulnerability Group",
    },
    {
        "columns": ["Step", "stranded_SES_1_0_0.3", "stranded_SES_1_0.7_1"],
        "save_filename": "stranded_vul.png",
        "y_label": "Stranded persons proportions",
        "legend_labels": {
            "stranded_SES_1_0_0.3": "High SES (low-vulnerability)",
            "stranded_SES_1_0.7_1": "Low SES (high-vulnerability)",
        },
        "y_range": (0, 0.5),
        "plot_title": "Stranded Population by Vulnerability Group",
    },
    {
        "columns": ["Step", "injured_SES_1_0_0.3", "injured_SES_1_0.7_1"],
        "save_filename": "injured_vul.png",
        "y_label": "Health-compromised persons proportions",
        "legend_labels": {
            "injured_SES_1_0_0.3": "High SES (low-vulnerability)",
            "injured_SES_1_0.7_1": "Low SES (high-vulnerability)",
        },
        "y_range": (0, 0.5),
        "plot_title": "Health-compromised Population by Vulnerability Group",
    },
    {
        "columns": ["Step", "hospitalized_SES_1_0_0.3", "hospitalized_SES_1_0.7_1"],
        "save_filename": "hospitalized_vul.png",
        "y_label": "Hospitalized persons proportions",
        "legend_labels": {
            "hospitalized_SES_1_0_0.3": "High SES (low-vulnerability)",
            "hospitalized_SES_1_0.7_1": "Low SES (high-vulnerability)",
        },
        "y_range": (0, 0.5),
        "plot_title": "Hospitalized Population by Vulnerability Group",
    },
    {
        "columns": ["Step", "sheltered_SES_1_0_0.3", "sheltered_SES_1_0.7_1"],
        "save_filename": "sheltered_vul.png",
        "y_label": "Sheltered persons proportions",
        "legend_labels": {
            "sheltered_SES_1_0_0.3": "High SES (low-vulnerability)",
            "sheltered_SES_1_0.7_1": "Low SES (high-vulnerability)",
        },
        "y_range": (0, 0.5),
        "plot_title": "Sheltered Population by Vulnerability Group",
    },
    {
        "columns": ["Step", "dead_SES_1_0_0.3", "dead_SES_1_0.7_1"],
        "save_filename": "deceased_vul.png",
        "y_label": "Deceased persons proportions",
        "legend_labels": {
            "dead_SES_1_0_0.3": "High SES (low-vulnerability)",
            "dead_SES_1_0.7_1": "Low SES (high-vulnerability)",
        },
        "y_range": (0, 0.5),
        "plot_title": "Deceased Population by Vulnerability Group",
    },
]

for spec in ses_group_specs:
    plot_multi_line_graph(
        df=results_df,
        columns=spec["columns"],
        save_filename=spec["save_filename"],
        y_label=spec["y_label"],
        legend_labels=spec["legend_labels"],
        colors=None,
        x_range=(0, 38),
        y_range=spec["y_range"],
        x_interval=7,
        y_interval=0.1,
        plot_title=spec["plot_title"],
    )

# 6) Decision groups: one combined graph and separate graphs
decision_groups = [
    {
        "save_prefix": "nonevac_high_dec",
        "columns": {
            "PMT": "PMT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1",
            "TPB": "TPB_preflood_non_evacuation_measure_implemented_SES_1_0.7_1",
            "SCT": "SCT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1",
            "CRT": "CRT_preflood_non_evacuation_measure_implemented_SES_1_0.7_1",
        },
        "y_label": "Proportion of high-vulnerability agents",
        "x_range": (0, 8),
        "y_range": (0, 0.5),
        "x_interval": 7,
        "y_interval": 0.05,
        "phase_label": "Pre-flood Non-Evacuation",
        "group_label": "of High-Vulnerability Agents",
    },
    {
        "save_prefix": "nonevac_low_dec",
        "columns": {
            "PMT": "PMT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3",
            "TPB": "TPB_preflood_non_evacuation_measure_implemented_SES_1_0_0.3",
            "SCT": "SCT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3",
            "CRT": "CRT_preflood_non_evacuation_measure_implemented_SES_1_0_0.3",
        },
        "y_label": "Proportion of low-vulnerability agents",
        "x_range": (0, 8),
        "y_range": (0, 0.03),
        "x_interval": 7,
        "y_interval": 0.01,
        "phase_label": "Pre-flood Non-Evacuation",
        "group_label": "of Low-Vulnerability Agents",
    },
    
    {
        "save_prefix": "evac_low_dec",
        "columns": {
            "PMT": "PMT_evacuation_SES_1_0_0.3",
            "TPB": "TPB_evacuation_SES_1_0_0.3",
            "SCT": "SCT_evacuation_SES_1_0_0.3",
            "CRT": "CRT_evacuation_SES_1_0_0.3",
        },
        "y_label": "Proportion of low-vulnerability agents",
        "x_range": (7, 21.1),
        "y_range": (0, 0.03),
        "x_interval": 7,
        "y_interval": 0.025,
        "phase_label": "Evacuation",
        "group_label": "of Low-Vulnerability Agents",
    },
    # OK
    {
        "save_prefix": "evac_high_dec",
        "columns": {
            "PMT": "PMT_evacuation_SES_1_0.7_1",
            "TPB": "TPB_evacuation_SES_1_0.7_1",
            "SCT": "SCT_evacuation_SES_1_0.7_1",
            "CRT": "CRT_evacuation_SES_1_0.7_1",
        },
        "y_label": "Proportion of high-vulnerability agents",
        "x_range": (0, 21.1),
        "y_range": (0, 0.6),
        "x_interval": 7,
        "y_interval": 0.05,
        "phase_label": "Evacuation",
        "group_label": "of High-Vulnerability Agents",
    },
    # CHECKKK
    {
        "save_prefix": "during_low_dec",
        "columns": {
            "PMT": "PMT_duringflood_coping_action_implemented_SES_1_0_0.3",
            "TPB": "TPB_duringflood_coping_action_implemented_SES_1_0_0.3",
            "SCT": "SCT_duringflood_coping_action_implemented_SES_1_0_0.3",
            "CRT": "CRT_duringflood_coping_action_implemented_SES_1_0_0.3",
        },
        "y_label": "Proportion of low-vulnerability agents",
        "x_range": (7, 19.1),
        "y_range": (0, 0.03),
        "x_interval": 7,
        "y_interval": 0.006,
        "phase_label": "During-flood Coping",
        "group_label": "of Low-Vulnerability Agents",
    },
    # CHECK
    {
        "save_prefix": "during_high_dec",
        "columns": {
            "PMT": "PMT_duringflood_coping_action_implemented_SES_1_0.7_1",
            "TPB": "TPB_duringflood_coping_action_implemented_SES_1_0.7_1",
            "SCT": "SCT_duringflood_coping_action_implemented_SES_1_0.7_1",
            "CRT": "CRT_duringflood_coping_action_implemented_SES_1_0.7_1",
        },
        "y_label": "Proportion of high-vulnerability agents",
        "x_range": (7, 19.1),
        "y_range": (0, 0.2),
        "x_interval": 7,
        "y_interval": 0.05,
        "phase_label": "During-flood Coping",
        "group_label": "of High-Vulnerability Agents",
    },
        # OK
    {
        "save_prefix": "post_low_dec",
        "columns": {
            "PMT": "PMT_postflood_adaptation_measures_planned_SES_1_0_0.3",
            "TPB": "TPB_postflood_adaptation_measures_planned_SES_1_0_0.3",
            "SCT": "SCT_postflood_adaptation_measures_planned_SES_1_0_0.3",
            "CRT": "CRT_postflood_adaptation_measures_planned_SES_1_0_0.3",
        },
        "y_label": "Proportion of low-vulnerability agents",
        "x_range": (21, 38.2),
        "y_range": (0, 0.05),
        "x_interval": 7,
        "y_interval": 0.0125,
        "phase_label": "Post-flood Adaptation",
        "group_label": "of Low-Vulnerability Agents",
    },
    # OK
    {
        "save_prefix": "post_high_dec",
        "columns": {
            "PMT": "PMT_postflood_adaptation_measures_planned_SES_1_0.7_1",
            "TPB": "TPB_postflood_adaptation_measures_planned_SES_1_0.7_1",
            "SCT": "SCT_postflood_adaptation_measures_planned_SES_1_0.7_1",
            "CRT": "CRT_postflood_adaptation_measures_planned_SES_1_0.7_1",
        },
        "y_label": "Proportion of high-vulnerability agents",
        "x_range": (21, 38.2),
        "y_range": (0, 0.3),
        "x_interval": 7,
        "y_interval": 0.05,
        "phase_label": "Post-flood Adaptation",
        "group_label": "of High-Vulnerability Agents",
    },
]

for spec in decision_groups:
    plot_combined_and_separate_decision_group(
        df=results_df,
        decision_columns=spec["columns"],
        save_prefix=spec["save_prefix"],
        y_label=spec["y_label"],
        x_range=spec["x_range"],
        y_range=spec["y_range"],
        x_interval=spec["x_interval"],
        y_interval=spec["y_interval"],
        phase_label=spec["phase_label"],
        group_label=spec["group_label"],
        scenario_filter=TARGET_SCENARIO,
    )