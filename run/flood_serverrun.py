"""
Server Run Script

This script launches an interactive visualization of the flood model, 
allowing real-time observation of agent behaviors, decisions, and flood impacts 
through a web interface.

Output:
Interactive visualization accessible via a web browser and data saved in 
  data_collection/serverrun_results.csv
"""

import sys
import os

from matplotlib import legend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import agents.flood_agents as FA
from model.flood_model import FloodModel
from space.flood_space import FloodArea, MergedDamRoute, MalolosHydroRiver, MalolosChannel

import mesa_geo as mg
from mesa.visualization import Choice, ModularServer
from mesa.visualization.modules import ChartModule, TextElement
from mesa.visualization import Slider

try:
    from mesa.visualization import StaticText
except ImportError:
    try:
        from mesa.visualization.UserParam import UserSettableParameter

        def StaticText(name, value=""):
            return UserSettableParameter("static_text", name, value=value)
    except Exception:
        StaticText = None
    
import warnings
import psutil
import xyzservices.providers as xyz

# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="seaborn")

# Function to get memory and CPU usage
def get_resource_usage():
    process = psutil.Process()
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    cpu_usage = process.cpu_percent(interval=None)  # CPU usage as a percentage
    return memory_usage, cpu_usage

#========================== Launch to server observe ======================

#Polygon representation in map
def agent_portrayal(agent):
    portrayal = {}

    # Portrayal for person agents based on their status (stranded, injured, deceased)
    if isinstance(agent, FA.Person_Agent):
        portrayal["color"] = "Green"
        portrayal["radius"] = "3"
        portrayal["fillOpacity"] = 1
        
        if agent.stranded:
            portrayal["color"] = "Red"
        elif not agent.alive:
            portrayal["color"] = "Black"
        elif agent.injured:
            portrayal["color"] = "Orange"

    # Portayal for flood areas based on severity        
    elif isinstance(agent, FloodArea):
        print("Rendering FloodArea:", agent.var_value, agent.flood_file)
        if agent.var_value == 1:
            portrayal["color"] = "Yellow"
            portrayal["fillColor"] = "Yellow"
        elif agent.var_value == 2:
            portrayal["color"] = "Orange"
            portrayal["fillColor"] = "Orange"
        elif agent.var_value == 3:
            portrayal["color"] = "Red"
            portrayal["fillColor"] = "Red"
        else:  # var_value == 0 or unknown
            portrayal["color"] = "#95253400"
            portrayal["fillColor"] = "#95253400"
            portrayal["weight"] = 0
            portrayal["fillOpacity"] = 0   # fully invisible
            portrayal["opacity"] = 0       # hide border too

        portrayal["weight"] = 1
        portrayal["fillOpacity"] = 0.35
        portrayal["opacity"] = 0.6

    # Portrayal for other entities    
    elif isinstance(agent, FA.Business_Agent):
        portrayal["color"] = "Purple"

    elif isinstance(agent, FA.House_Agent):
        portrayal["color"] = "#952534"
        portrayal["weight"] = 0.3
        portrayal["fillOpacity"] = 0.6
    
    elif isinstance(agent, FA.School_Agent):
        portrayal["color"] = "Yellow"
    
    elif isinstance(agent, FA.Shelter_Agent):
        portrayal["color"] = "Blue"
    
    elif isinstance(agent, FA.Healthcare_Agent):
        portrayal["color"] = "Orange"
    
    elif isinstance(agent, FA.Government_Agent):
        portrayal["color"] = "Magenta"   
    
    # -----------------------------------------------------------------------
    # Network layer portrayals
    # Lines shown only when the reach is active (carrying flow this step).
    # Color encodes severity: blue=low, orange=moderate, red=high.
    # Inactive reaches are rendered very faintly so the map stays readable.
    # -----------------------------------------------------------------------
    elif isinstance(agent, MergedDamRoute):
        if getattr(agent, "active", False):
            sev = getattr(agent, "current_sev", 0)
            portrayal["color"] = "#1a6faf" if sev <= 1 else ("#e07b00" if sev == 2 else "#cc0000")
            portrayal["weight"] = 2
            portrayal["opacity"] = 0.7
        else:
            portrayal["color"] = "#aaaaaa"
            portrayal["weight"] = 0.5
            portrayal["opacity"] = 0.15
 
    elif isinstance(agent, MalolosHydroRiver):
        if getattr(agent, "active", False):
            sev = getattr(agent, "current_sev", 0)
            portrayal["color"] = "#1a6faf" if sev <= 1 else ("#e07b00" if sev == 2 else "#cc0000")
            portrayal["weight"] = 2.5
            portrayal["opacity"] = 0.8
        else:
            portrayal["color"] = "#5599cc"
            portrayal["weight"] = 1
            portrayal["opacity"] = 0.2
 
    elif isinstance(agent, MalolosChannel):
        if getattr(agent, "active", False):
            sev = getattr(agent, "current_sev", 0)
            portrayal["color"] = "#33aadd" if sev <= 1 else ("#e07b00" if sev == 2 else "#cc0000")
            portrayal["weight"] = 1.5
            portrayal["opacity"] = 0.75
        else:
            portrayal["color"] = "#88ccee"
            portrayal["weight"] = 0.8
            portrayal["opacity"] = 0.15
 
    return portrayal


class colorLegend(TextElement):
    def __init__(self):
        pass

    def render(self, model):

        # legend = "<div style='padding:10px;font-size:14px;display:grid;"
        # legend += "grid-template-columns:repeat(4,auto);gap:6px 18px;align-items:center;'>"

        # legend += "<strong>Legend</strong><span></span><span></span>"

        # # Persons (circles)
        # legend += "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:green;margin-right:6px;'></span>Person</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:grey;border:1px solid #555;margin-right:6px;'></span>House</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:orange;border:1px solid #555;margin-right:6px;'></span>Healthcare</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:yellow;opacity:0.35;border:1px solid #555;margin-right:6px;'></span>Flood Low</span>"

        # legend += "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:red;margin-right:6px;'></span>Stranded</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:purple;border:1px solid #555;margin-right:6px;'></span>Business</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:yellow;border:1px solid #555;margin-right:6px;'></span>School</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:orange;opacity:0.35;border:1px solid #555;margin-right:6px;'></span>Flood Medium</span>"

        # legend += "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:orange;margin-right:6px;'></span>Injured</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:magenta;border:1px solid #555;margin-right:6px;'></span>Government</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:blue;border:1px solid #555;margin-right:6px;'></span>Shelter</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:red;opacity:0.35;border:1px solid #555;margin-right:6px;'></span>Flood High</span>"

        # legend += "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:black;margin-right:6px;'></span>Deceased</span>"
        # legend += "<span><span style='display:inline-block;width:12px;height:12px;background:orange;border:1px solid #555;margin-right:6px;'></span>Healthcare</span>"
        # legend += "</div>"

        legend = "<div style='padding:10px;font-size:14px;display:grid;"
        legend += "grid-template-columns:repeat(4,auto);gap:6px 18px;align-items:center;'>"
 
        legend += "<strong>Legend</strong><span></span><span></span>"
 
        # Persons (circles)
        legend += "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:green;margin-right:6px;'></span>Person</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:grey;border:1px solid #555;margin-right:6px;'></span>House</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:orange;border:1px solid #555;margin-right:6px;'></span>Healthcare</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:yellow;opacity:0.35;border:1px solid #555;margin-right:6px;'></span>Flood Low</span>"
 
        legend += "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:red;margin-right:6px;'></span>Stranded</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:purple;border:1px solid #555;margin-right:6px;'></span>Business</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:yellow;border:1px solid #555;margin-right:6px;'></span>School</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:orange;opacity:0.35;border:1px solid #555;margin-right:6px;'></span>Flood Medium</span>"
 
        legend += "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:orange;margin-right:6px;'></span>Injured</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:magenta;border:1px solid #555;margin-right:6px;'></span>Government</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:blue;border:1px solid #555;margin-right:6px;'></span>Shelter</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:red;opacity:0.35;border:1px solid #555;margin-right:6px;'></span>Flood High</span>"
 
        legend += "<span><span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:black;margin-right:6px;'></span>Deceased</span>"
        legend += "<span><span style='display:inline-block;width:12px;height:12px;background:orange;border:1px solid #555;margin-right:6px;'></span>Healthcare</span>"
 
        legend += "<strong>Network (active)</strong><span></span><span></span><span></span>"
        legend += "<span><span style='display:inline-block;width:24px;height:3px;background:#1a6faf;margin-right:6px;vertical-align:middle;'></span>Low severity</span>"
        legend += "<span><span style='display:inline-block;width:24px;height:3px;background:#e07b00;margin-right:6px;vertical-align:middle;'></span>Moderate severity</span>"
        legend += "<span><span style='display:inline-block;width:24px;height:3px;background:#cc0000;margin-right:6px;vertical-align:middle;'></span>High severity</span>"
        legend += "<span><span style='display:inline-block;width:24px;height:3px;background:#aaaaaa;margin-right:6px;vertical-align:middle;opacity:0.4;'></span>Inactive reach</span>"
 
        legend += "</div>"
        return legend

model_params = {
    "N_persons": Slider("Number of persons", 300, 10, 1500, 10),
    "shelter_cap_limit": Slider("Shelter Capacity(% of pop.)", 1, 0, 10, 0.3),
    "healthcare_cap_limit": Slider("Healthcare Capacity(% of pop.)", 5, 0, 10, 1),
    "shelter_funding": Slider("Shelter funds $", 50000, 5000, 200000, 5000),
    "healthcare_funding": Slider("Healthcare funds $", 100000, 50000, 500000, 10000),
    "pre_flood_days": Slider("Pre Flood Days", 8, 0, 90, 1),
    "flood_days": Slider("Flood Days", 10, 3, 30, 1),
    "post_flood_days": Slider("Post Flood Days", 14, 0, 90, 1),

    "dam_scenario_name": Choice(
        "Dam scenario",
        value="S2",
        choices=["S0", "S1", "S2", "S3"],
    ),
}

if StaticText is not None:
    _dam_scenario_legend_html = (
        "<div style='font-size:12px; line-height:1.45; padding:6px 8px; "
        "margin:2px 0 8px 0; background:#f7f7f7; border:1px solid #d9d9d9; border-radius:6px;'>"
        "<strong>Scenario guide</strong><br>"
        "<strong>S0</strong> - Baseline: no dam effect added.<br>"
        "<strong>S1</strong> - Normal operations: low or typical release.<br>"
        "<strong>S2</strong> - Controlled release: moderate pre-emptive release before spill.<br>"
        "<strong>S3</strong> - Emergency spill: high-release stress test for downstream flooding."
        "</div>"
    )

    try:
        model_params["dam_scenario_legend"] = StaticText(value=_dam_scenario_legend_html)
    except TypeError:
        try:
            model_params["dam_scenario_legend"] = StaticText(_dam_scenario_legend_html)
        except TypeError:
            try:
                model_params["dam_scenario_legend"] = StaticText("", _dam_scenario_legend_html)
            except TypeError:
                model_params["dam_scenario_legend"] = StaticText("", value=_dam_scenario_legend_html)

model_params.update({
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
})

# Create a canvas grid with given portrayal function and agent dimensions
map_element = mg.visualization.MapModule(
    agent_portrayal,
    map_height=500,
    map_width=860
    # tiles=xyz.Esri.WorldImagery for satellite basemap
)

# Add the legend to the visualization
legend = colorLegend()

#---------------------- SES index data visuals-----------------------------
# Define a function to create SES-specific chart modules for decision-making phases
def create_ses_charts(decision, ses_ranges):
    charts = []
    for ses_range in ses_ranges:
        charts.append(
            ChartModule([
                {"Label": f"PMT_{decision}_{ses_range}", "Color": "blue"},
                {"Label": f"TPB_{decision}_{ses_range}", "Color": "green"},
                {"Label": f"SCT_{decision}_{ses_range}", "Color": "orange"},
                {"Label": f"CRT_{decision}_{ses_range}", "Color": "red"}
            ])
        )
    return charts

# Define SES ranges and their corresponding colors for each index
ses_index_1_ranges = ['SES_1_0_0.3', 'SES_1_0.7_1']
ses_index_2_ranges = ['SES_2_0_0.3', 'SES_2_0.7_1']
all_ses_ranges = ses_index_1_ranges + ses_index_2_ranges
ses_index_1_colors = ['green', 'red']
ses_index_2_colors = ['blue', 'magenta']

# Create SES-specific charts for decision phases
preflood_non_evacuation_charts = create_ses_charts("preflood_non_evacuation_measure_implemented", all_ses_ranges)
evacuation_charts = create_ses_charts("evacuation", all_ses_ranges)
duringflood_coping_charts = create_ses_charts("duringflood_coping_action_implemented", all_ses_ranges)
postflood_recovery_charts = create_ses_charts("postflood_adaptation_measures_planned", all_ses_ranges)

# Define a function to create grouped SES charts for a given metric (e.g., Evacuated, Stranded)
def create_grouped_ses_charts(metric, ses_ranges, colors):
    return ChartModule([
        {"Label": f"{metric}_{ses_range}", "Color": color} for ses_range, color in zip(ses_ranges, colors)
    ])

# Create grouped charts for selected SES-based metrics for index 1 and index 2 separately
evacuated_chart_ses_1 = create_grouped_ses_charts("evacuated", ses_index_1_ranges, ses_index_1_colors)
evacuated_chart_ses_2 = create_grouped_ses_charts("evacuated", ses_index_2_ranges, ses_index_2_colors)

stranded_chart_ses_1 = create_grouped_ses_charts("stranded", ses_index_1_ranges, ses_index_1_colors)
stranded_chart_ses_2 = create_grouped_ses_charts("stranded", ses_index_2_ranges, ses_index_2_colors)

injured_chart_ses_1 = create_grouped_ses_charts("injured", ses_index_1_ranges, ses_index_1_colors)
injured_chart_ses_2 = create_grouped_ses_charts("injured", ses_index_2_ranges, ses_index_2_colors)

sheltered_chart_ses_1 = create_grouped_ses_charts("sheltered", ses_index_1_ranges, ses_index_1_colors)
sheltered_chart_ses_2 = create_grouped_ses_charts("sheltered", ses_index_2_ranges, ses_index_2_colors)

hospitalized_chart_ses_1 = create_grouped_ses_charts("hospitalized", ses_index_1_ranges, ses_index_1_colors)
hospitalized_chart_ses_2 = create_grouped_ses_charts("hospitalized", ses_index_2_ranges, ses_index_2_colors)

dead_chart_ses_1 = create_grouped_ses_charts("dead", ses_index_1_ranges, ses_index_1_colors)
dead_chart_ses_2 = create_grouped_ses_charts("dead", ses_index_2_ranges, ses_index_2_colors)

# General charts (not SES-specific)
persons_chart = ChartModule([
    {"Label": "Stranded", "Color": "red"},
    {"Label": "Health-compromised", "Color": "orange"},
    {"Label": "Sheltered", "Color": "blue"},
    {"Label": "Hospitalized", "Color": "grey"},
    {"Label": "Death", "Color": "black"}
])

decisions_chart = ChartModule([
    {"Label": "Preflood_Non_Evacuation_Measure_Implemented", "Color": "orange"},
    {"Label": "Evacuated", "Color": "green"},
    {"Label": "Duringflood_Coping_Action_Implemented", "Color": "red"},
    {"Label": "Postflood_Adaptation_Measures_Planned", "Color": "blue"}
])

entities_chart = ChartModule([
    {"Label": "Houses_Flooded", "Color": "red"},
    {"Label": "Schools_Flooded", "Color": "orange"},
    {"Label": "Businesses_Flooded", "Color": "blue"}
])

economic_chart = ChartModule([
    {"Label": "Wealth_People", "Color": "blue"},
    {"Label": "Wealth_Businesses", "Color": "green"},
    {"Label": "Wealth_Shelter", "Color": "orange"},
    {"Label": "Wealth_Healthcare", "Color": "purple"},
    {"Label": "Wealth_Government", "Color": "red"}
])

# Combine all the chart modules into a single list
all_charts = (
    [decisions_chart] + [persons_chart] + [entities_chart] + [economic_chart] + 
    preflood_non_evacuation_charts + evacuation_charts + duringflood_coping_charts + 
    postflood_recovery_charts + [evacuated_chart_ses_1, evacuated_chart_ses_2,
                                 stranded_chart_ses_1, stranded_chart_ses_2,
                                 injured_chart_ses_1, injured_chart_ses_2,
                                 sheltered_chart_ses_1, sheltered_chart_ses_2,
                                 hospitalized_chart_ses_1, hospitalized_chart_ses_2,
                                 dead_chart_ses_1, dead_chart_ses_2]
)

# Now you can pass all_charts to the server
server = ModularServer(
    FloodModel,
    [map_element, legend] + all_charts,
    "Flood Model - Vulnerabilities and Decision Making",
    model_params,
)
            
# Run the server
server.port = 8521  # The default port number
server.launch()  
    
# Measure resource usage after the run
mem_after, cpu_after = get_resource_usage()
print(f"Memory usage after batch run: {mem_after:.2f} MB")
print(f"CPU usage after batch run: {cpu_after:.2f}%")