"""
Flood Model Simulation
 
This model simulates a flood scenario where agents (persons, homes, businesses, shelter system, healthcare, and government) 
interact within a grid-based environment. Each agent has specific attributes and behaviors, such as risk perception, 
movement, distress reactions, and financial transactions. The simulation tracks the effects of flooding on agent behavior, 
economic activities, rescue operations, and shelter systems.
 
Network layers integrated:
- MergedDamRoute    : upstream/shared dam-route backbone (Angat, Ipo, Bustos dams)
- MalolosHydroRiver : HydroRIVERS subset that receives dam-route handoff signal and routes it through the study area
- MalolosChannel    : local rivers, creeks, and esteros that inherit hazard from the HydroRIVERS backbone
 
Signal flow each flood step:
  update_merged_dam_routes()
      → handoff_to_malolos_hydrorivers()
          → update_malolos_hydrorivers()
              → activate_malolos_channels()
 
The combined spatial severity (base hazard + backbone bonus + local channel bonus) is then available
via space.get_total_flood_var_at_position(), which is re-evaluated each step for every person agent.
"""
 
import sys
import os

from tomlkit import value

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import pandas as pd
from shapely.geometry import Point
from space.flood_space import StudyArea
from mesa import Model
from agents import person_agent_assign as psn_agnt
from mesa.time import RandomActivation
from data_collection import data_collect
from collections import defaultdict
import agents.flood_agents as FA


# FloodModel inherits from Mesa’s Model class. So it becomes a Mesa-compatible simulation model.
class FloodModel(Model):
    """
    The Model Class defines the environment and interaction rules for the flood simulation. 
    It manages the grid-based environment, schedules agent actions, tracks time progression, 
    and monitors the effects of flooding on agents, their behaviors, and economic activities.
    """

    # Constructor that runs as the model is created. It initializes the environment, creates agents, 
    # and sets up the schedule and data collection.
    def __init__(self, N_persons,
                    shelter_cap_limit, healthcare_cap_limit, shelter_funding, healthcare_funding, pre_flood_days, flood_days, post_flood_days, 
                    houses_file, businesses_file, schools_file, shelter_file, healthcare_file, government_file, flood_file_1, flood_file_2, flood_file_3,
                    dam_scenario_name="S1", dam_scenarios_file=None, merged_dams_file=None, malolos_hydrorivers_file=None, malolos_channels_file=None, model_crs="EPSG:32651"):
            
            super().__init__()
            random.seed(42)
            self.random.seed(42)
            
            self.crs = model_crs
            self.debug_network = True

            self.dam_scenarios_file = dam_scenarios_file
            self.dam_scenarios = pd.read_csv(dam_scenarios_file) if dam_scenarios_file else pd.DataFrame()

            # Dam scenario selected from serverrun / batchrun UI
            self.dam_scenario_name = str(dam_scenario_name).strip() if dam_scenario_name else "S0"

            # River-dam network propagation controls
            self.hydro_attenuation = 0.85
            self.hydro_max_downstream_steps = 10

            # Local channel-to-channel propagation controls
            self.channel_attenuation = 0.90
            self.channel_max_steps = 5

            self.dam_scenario_lookup = {}
            if not self.dam_scenarios.empty:
                scenario_df = self.dam_scenarios.copy()

                # Optional scenario filter
                if "scenario_name" in scenario_df.columns:
                    scenario_df = scenario_df[
                        scenario_df["scenario_name"].astype(str).str.strip() == self.dam_scenario_name
                    ]

                for _, row in scenario_df.iterrows():
                    key = str(
                        row.get("dam_name", row.get("dam_names", row.get("source_dam", "")))
                    ).strip()
                    if key:
                        self.dam_scenario_lookup[key] = row.to_dict()
                
                print("\nDam Scenario Lookup Table Loaded:")
                print("Selected dam scenario:", self.dam_scenario_name)
                print("Dam scenario lookup:")
                for dam, values in self.dam_scenario_lookup.items():
                    print(dam, values)
                        
            # Creation of Spatial Environment
            self.space = StudyArea(
                self,
                houses_file,
                businesses_file,
                schools_file,
                shelter_file,
                healthcare_file,
                government_file,
                flood_file_1,
                flood_file_2,
                flood_file_3,
                merged_dams_file,
                malolos_hydrorivers_file,
                malolos_channels_file,
                model_crs,
            )

            handoff_ids = {
                self._clean_network_id(getattr(r, "handoff_id", None))
                for r in self.space.merged_dams
            }
            handoff_ids = {x for x in handoff_ids if x}

            hydro_ids = {
                self._clean_network_id(getattr(r, "reach_id", None))
                for r in self.space.malolos_hydrorivers
            }
            hydro_ids = {x for x in hydro_ids if x}

            missing_handoff_ids = handoff_ids - hydro_ids

            print("\nHandoff ID Validation:")
            print("Total handoff IDs from merged dams:", len(handoff_ids))
            print("Total HydroRIVERS reach IDs:", len(hydro_ids))
            print("Missing handoff IDs:", missing_handoff_ids)

            print("\n================ RAW ID SAMPLE CHECK ================")
            print("\nMerged dam handoff IDs:")
            for r in self.space.merged_dams:
                print(
                    "reach_id=", repr(getattr(r, "reach_id", None)),
                    "handoff_id=", repr(getattr(r, "handoff_id", None)),
                    "dam_name=", repr(getattr(r, "dam_name", None)),
                    "source_dam=", repr(getattr(r, "source_dam", None)),
                    "seg_role=", repr(getattr(r, "seg_role", None)),
                )

            print("\nHydroRIVERS reach IDs:")
            for r in self.space.malolos_hydrorivers:
                print(
                    "reach_id=", repr(getattr(r, "reach_id", None)),
                    "down_id=", repr(getattr(r, "down_id", None)),
                )

            print("\nChannel connection IDs:")
            for ch in self.space.malolos_channels:
                print(
                    "channel_id=", repr(getattr(ch, "reach_id", None)),
                    "name=", repr(getattr(ch, "reach_name", None)),
                    "con_reach_=", repr(getattr(ch, "con_reach_", None)),
                )

            print("=====================================================\n")

            channel_con_ids = {
                self._clean_network_id(getattr(ch, "con_reach_", None))
                for ch in self.space.malolos_channels
            }
            channel_con_ids = {x for x in channel_con_ids if x}

            hydro_ids = {
                self._clean_network_id(getattr(r, "reach_id", None))
                for r in self.space.malolos_hydrorivers
            }
            hydro_ids = {x for x in hydro_ids if x}

            missing_channel_connections = channel_con_ids - hydro_ids

            print("\nChannel Connection Validation:")
            print("Total channel con_reach_ IDs:", len(channel_con_ids))
            print("Total HydroRIVERS reach IDs:", len(hydro_ids))
            print("Missing channel connection IDs:", missing_channel_connections)

            self.num_persons = N_persons

            # Barangays
            self.barangay_populations = {
                "Anilao": 3019,
                "Atlag": 4778,
                "Babatnin": 1002,
                "Bagna": 4944,
                "Bagong Bayan": 3206,
                "Balayong": 4618,
                "Balite": 3556,
                "Bangkal": 12935,
                "Barihan": 5869,
                "Bulihan": 16224,
                "Bungahan": 3354,
                "Dakila": 7215,
                "Guinhawa": 4335,
                "Caingin": 7375,
                "Calero": 1347,
                "Caliligawan": 530,
                "Canalate": 3710,
                "Caniogan": 5297,
                "Catmon": 2357,
                "Ligas": 6684,
                "Liang": 1403,
                "Longos": 17863,
                "Look 1st": 9937,
                "Look 2nd": 3364,
                "Lugam": 4871,
                "Mabolo": 6309,
                "Mambog": 3101,
                "Masile": 788,
                "Matimbo": 6699,
                "Mojon": 16706,
                "Namayan": 664,
                "Niugan": 781,
                "Pamarawan": 2741,
                "Panasahan": 9664,
                "Pinagbakahan": 7947,
                "San Agustin": 2072,
                "San Gabriel": 2177,
                "San Juan": 4618,
                "San Pablo": 5106,
                "San Vicente (Pob.)": 2475,
                "Santiago": 1786,
                "Santisima Trinidad": 6797,
                "Santo Cristo": 2044,
                "Santo Niño (Pob.)": 661,
                "Santo Rosario (Pob.)": 6509,
                "Santor": 8745,
                "Sumapang Bata": 2577,
                "Sumapang Matanda": 9166,
                "Taal": 1799,
                "Tikay": 13359,
                "Cofradia": 4725,
            }
            
            # PWDs
            self.barangay_pwd_raw = {
                "Anilao": 46, 
                "Atlag": 106, 
                "Babatnin": 11, 
                "Bagna": 66, 
                "Bagong Bayan": 81, 
                "Balayong": 76, 
                "Balite": 67, 
                "Bangkal": 162, 
                "Barihan": 116, 
                "Bulihan": 108, 
                "Bungahan": 66, 
                "Dakila": 109, 
                "Guinhawa": 30, 
                "Caingin": 168, 
                "Calero": 17, 
                "Caliligawan": 12, 
                "Canalate": 78, 
                "Caniogan": 102, 
                "Catmon": 46, 
                "Ligas": 38, 
                "Liang": 32, 
                "Longos": 291, 
                "Look 1st": 170, 
                "Look 2nd": 29, 
                "Lugam": 71, 
                "Mabolo": 130, 
                "Mambog": 44, 
                "Masile": 14, 
                "Matimbo": 101, 
                "Mojon": 333, 
                "Namayan": 8, 
                "Niugan": 17, 
                "Pamarawan": 66, 
                "Panasahan": 163, 
                "Pinagbakahan": 79, 
                "San Agustin": 57, 
                "San Gabriel": 58, 
                "San Juan": 61, 
                "San Pablo": 92, 
                "San Vicente (Pob.)": 52, 
                "Santiago": 32, 
                "Santisima Trinidad": 119, 
                "Santo Cristo": 59, 
                "Santo Niño (Pob.)": 17, 
                "Santo Rosario (Pob.)": 132, 
                "Santor": 131, 
                "Sumapang Bata": 49, 
                "Sumapang Matanda": 134, 
                "Taal": 50, 
                "Tikay": 117, 
                "Cofradia": 82,
            }

            self.original_barangay_populations = self.barangay_populations.copy()
            total_brgy_pop = sum(self.barangay_populations.values())
            scale = self.num_persons/total_brgy_pop
            self.barangay_populations = {
                k: int(v * scale)
                for k, v in self.barangay_populations.items()
            }
            self.barangay_pwd_ratio = {
                k: self.barangay_pwd_raw[k]/self.original_barangay_populations[k]
                for k in self.barangay_pwd_raw
                if k in self.original_barangay_populations
            }

            # -----------------------------------------------------------------------
            # Simple dam scenario selection
            # Set this manually for testing: S0, S1, S2, S3
            # -----------------------------------------------------------------------
            
            # Print loaded agent counts for verification
            print("\nFlood Model - Initialization with the following parameters:")
            print("houses loaded:", len(self.space.houses))
            print("businesses loaded:", len(self.space.businesses))
            print("schools loaded:", len(self.space.schools))
            print("shelters loaded:", len(self.space.shelter))
            print("healthcare facilities loaded:", len(self.space.healthcare))
            print("government loaded:", len(self.space.government))
            print("flood areas loaded initially:", len(self.space.flood_areas))
            
            # Simulation time settings and flood file reference
            self.total_days = pre_flood_days + flood_days + post_flood_days
            self.pre_flood_days = pre_flood_days
            self.flood_days = flood_days
            self.flood_file_1 = flood_file_1
            self.flood_file_2 = flood_file_2
            self.flood_file_3 = flood_file_3
            
            self.disaster_period = "baseline"                           # Will later store values like 'pre_flood_evac_period', 'during_flood', 'post_flood'
            
            # Evacuation and rescue timing settings
            self.evacuation_time = (self.pre_flood_days - 7) * 24       # Evacuation starts 7 days before flood, converted to hours
            self.last_evacuation_time = self.pre_flood_days * 24        # Last evacuation time is at the end of the pre-flood period, converted to hours
            self.hours_before_rescue = 2                                # Time threshold for stranded agents to be rescued by shelter system, in hours
            self.hours_before_healthcare = 0                            # Time threshold for injured agents to receive healthcare, in hours
            
            # Track whether flood layers are currently active in the space
            self.flood_layers_active = False
            self.loaded_flood_files = set()

            self.perc_education_people = 0.89                           # Percentage of people who attend school, used to determine how many person agents will be assigned to schools
            self.schedule = RandomActivation(self)                      # Scheduler that activates agents in random order each step, ensuring a more realistic simulation of interactions and behaviors
            
            # Agent and Infrastructure Count
            self.num_houses = len(self.space.houses)
            self.num_businesses = len(self.space.businesses)
            self.num_schools = len(self.space.schools)
            
            # Calculate shelter and healthcare capacity limits based on the total population and specified percentage limits
            # Percentage capacity to actual capacity
            self.shelter_cap_limit = shelter_cap_limit/100 * self.num_persons
            self.healthcare_cap_limit = healthcare_cap_limit/100 * self.num_persons
            
            # TO BE MODIFIED - GDP CALCULATIONS BASED ON ACTUAL DATA !!!!
            self.business_gdp = (self.num_persons * 22000)/365 * 14 #* self.total_days
            self.school_gdp = (self.num_persons * 1200)/365 * 14 #* self.total_days
            self.shelter_gdp = (self.num_persons * 4200)/365 * 14 #* self.total_days
            self.healthcare_gdp = (self.num_persons * 4200)/365 * 14 #* self.total_days
            self.government_gdp = (self.num_persons * 7000)/365 * 14 #* self.total_days
            self.persons_gdp = 0
            self.total_gdp = self.business_gdp + self.school_gdp  + self.shelter_gdp + self.healthcare_gdp + self.government_gdp      
            
        #------------Initialize government agent-----------------------------------
            self.governments = self.space.government 
            for gov in self.governments:
                gov.wealth = self.government_gdp / len(self.governments)     
                gov.tax_revenue = 0
                self.schedule.add(gov)
        
        # TO MODIFY - ASSIGN DIFFERENT VALUES BASED ON LOCATION, SIZE, OR OTHER FACTORS.
        #------------Initialize shelter system agent------------------------------
            self.shelters = self.space.shelter
            for shelter in self.shelters:
                shelter.wealth = self.shelter_gdp / len(self.shelters)
                shelter.capacity_limit = self.shelter_cap_limit / len(self.shelters)
                shelter.sheltered_agents = []
                self.schedule.add(shelter)
            # Assumes all shelters are equally capable, which is simple but not always realistic.   
            # Assign different capacities and resources to each shelter based on their size, location, or funding.

        # TO MODIFY - ASSIGN DIFFERENT VALUES BASED ON LOCATION, SIZE, OR OTHER FACTORS.              
        #------------Initialize healthcare system agent------------------------------
            self.healthcare_facilities = self.space.healthcare
            for health in self.healthcare_facilities:
                health.wealth = self.healthcare_gdp / len(self.healthcare_facilities)
                health.capacity_limit = self.healthcare_cap_limit / len(self.healthcare_facilities)
                health.hospitalized_agents = []
                self.schedule.add(health)
            # Assumes all healthcare facilities are equally capable, which is simple but not always realistic.
            # Assign different capacities and resources to each healthcare facility based on their size, location, or funding.
        
        #---------Initialize businesses, schools and houses--------------------
            self._initialize_businesses()
            self._initialize_schools()
            self._initialize_houses() 
            self.houses_by_barangay = {}

            for house in self.space.houses:
                brgy = house.barangay
                self.houses_by_barangay.setdefault(brgy, []).append(house)
            
            print("\nBarangays from houses:", list(self.houses_by_barangay.keys())[::])
            print("\nBarangays from population:", list(self.barangay_populations.keys())[::])
            
        #-----------------Initialize person agents--------------------------------
            # Create person agents in the model with demographics, wealth classes, and other attributes based on the number of 
            # persons specified and assign them to schools, workplaces, and homes.
            # From person_agent_assign
            psn_agnt.create_person_agents(self)
            psn_agnt.assign_pwd_by_brgy(self)

            from collections import Counter
            from agents import flood_agents as FA
            assigned_counts = Counter(
                p.barangay for p in self.schedule.agents
                if isinstance(p, FA.Person_Agent)
            )
            print("\nBarangay Distribution:")
            for brgy, count in assigned_counts.items():
                print(f"{brgy}: {count}")
            print("Total persons:", sum(assigned_counts.values()))

            # Collect initial data on the model state before the simulation starts, such as the number of agents, their attributes, and the initial flood conditions.
            # From data_collect.py
            data_collect.data_collection(self)
            print("Persons created:", self.num_persons)
            print("Total scheduled agents:", len(self.schedule.agents)) 
               
    #-------------------------------Step function------------------------------
    # The step function defines the actions that occur at each time step of the simulation. 
    # It updates the disaster period based on the current time, 
    # handles the addition and removal of flood maps to simulate changing flood conditions, 
    # activates all agents according to the schedule, collects data, and saves the results to a CSV file for analysis.
    
    def step(self):
        # Update disaster period based on current time and defined thresholds for evacuation and flood duration. 
        # This allows the model to simulate different phases of the flood event,
        
        current_time = self.schedule.time
        flood_start_time = self.last_evacuation_time + 1
        flood_end_time = (self.pre_flood_days + self.flood_days) * 24

        # ------------------------------------------------------------------
        # 1. Determine disaster period
        # ------------------------------------------------------------------
        # Determine the current disaster period based on the defined time thresholds for evacuation and flood duration.
        if self.evacuation_time <= current_time <= self.last_evacuation_time:
            self.disaster_period = "pre_flood_evac_period"
        elif flood_start_time <= current_time < flood_end_time:
            self.disaster_period = "during_flood"
        elif current_time >= flood_end_time:
            self.disaster_period = "post_flood"
        else:
            self.disaster_period = "baseline"

        # ------------------------------------------------------------------
        # 2. Add / remove static flood hazard maps
        # ------------------------------------------------------------------
        # Handle flood map addition and removal based on the current disaster period.
        if self.disaster_period == "during_flood" and not self.flood_layers_active:
            self.add_flood_maps(self.flood_file_1)
            self.add_flood_maps(self.flood_file_2)
            self.add_flood_maps(self.flood_file_3)
            self.flood_layers_active = True

        if self.disaster_period == "post_flood" and self.flood_layers_active:
            self.remove_flood_maps(self.flood_file_1)
            self.remove_flood_maps(self.flood_file_2)
            self.remove_flood_maps(self.flood_file_3)
            self.flood_layers_active = False

        # ------------------------------------------------------------------
        # 3. Network update: dam route → HydroRIVERS → local channels
        # ------------------------------------------------------------------
        if self.disaster_period == "during_flood":
            if self.schedule.time == flood_start_time:
                self._debug_channel_propagation_printed = False

            # Update dam route → HydroRIVERS → local channels
            self.update_dynamic_network_hazard()

            # Update person/house flood exposure using total flood severity
            # total severity = static flood map + river bonus + channel bonus
            self._update_agent_flood_exposure()

            if self.debug_network and self.schedule.time == flood_start_time:
                self._debug_channel_propagation_printed = True
                self.debug_network_state("first flood step")
                self.debug_channel_connections("first flood step")
                self.debug_person_exposure_state("first flood step")
                
        # ------------------------------------------------------------------
        # 5. Activate all agents
        # ------------------------------------------------------------------
        # Activate all agents in the schedule, allowing them to perform their actions based on their defined behaviors and the current state of the environment.
        self.schedule.step()
        self.datacollector.collect(self)

        self.barangay_stats = self.compute_barangay_stats()

        # ------------------------------------------------------------------
        # 6. Save collected data
        # ------------------------------------------------------------------
        data_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data_collection")
        )
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, "serverrun_results.csv")
        data = self.datacollector.get_model_vars_dataframe()
        data.to_csv(file_path, index=False)

# ==========================================================================
# FLOOD MAP MANAGEMENT 
# ==========================================================================   
    
        
    def add_flood_maps(self, flood_file):
        """Reveal the actual hazard Var for all FloodArea agents from this file."""
        for agent in self.space.flood_areas:
            if agent.flood_file == flood_file:
                agent.var_value = agent._actual_var_value  # reveal actual severity

    def remove_flood_maps(self, flood_file):
        """Mask flood hazard by zeroing var_value — agents treat location as unaffected."""
        for agent in self.space.flood_areas:
            if agent.flood_file == flood_file:
                agent.var_value = 0  # hide again

    def save_results(self, filename="serverrun_results.csv"):
        data_folder = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data_collection")
        )
        os.makedirs(data_folder, exist_ok=True)
        file_path = os.path.join(data_folder, filename)
        data = self.datacollector.get_model_vars_dataframe()
        data.to_csv(file_path, index=False)
        print(f"Results saved to: {file_path}")

# ==========================================================================
# AGENT INITIALIZATION HELPERS
# ==========================================================================

    def _initialize_businesses(self):
        avg_business_wealth = self.business_gdp / self.num_businesses
        for business in self.space.businesses:
            business.wealth = avg_business_wealth
            business.physical_resilience = self._get_physical_resilience_from_hazard(
                business.geometry, agent_type="business"
            )
            self.schedule.add(business)
 
    def _initialize_schools(self):
        avg_school_wealth = self.school_gdp / self.num_schools
        for school in self.space.schools:
            school.wealth = avg_school_wealth
            school.physical_resilience = self._get_physical_resilience_from_hazard(
                school.geometry, agent_type="school"
            )
            self.schedule.add(school)
 
    def _initialize_houses(self):
        for house in self.space.houses:
            house.physical_resilience = self._get_physical_resilience_from_hazard(
                house.geometry, agent_type="house"
            )
            self.schedule.add(house)
        
    
# ==========================================================================
# SHELTER AND HEALTHCARE
# ==========================================================================
                       
    def notify_shelter(self, agent):
        "Send stranded person to nearest shelter"
        
        # Only notify if the agent is stranded and has been stranded for at least the specified hours before rescue. 
        # This prevents immediate evacuation of agents who just became stranded, allowing for a more realistic response time from the shelter system.
        if not agent.stranded or agent.time_stranded < self.hours_before_rescue:
            return

        # Find available shelters that have not reached their capacity limit. 
        # This ensures that only shelters with available space are considered for evacuation.
        available_shelters = [
            s for s in self.shelters
            if len(s.sheltered_agents) < s.capacity_limit
        ]

        # If no shelters are available, the agent cannot be evacuated and will remain stranded until a shelter becomes available or the situation changes.
        if not available_shelters:
            return

        #TO MODIFY: Straight-line distance is used here for simplicity, 
        # but in a real-world scenario, you would want to consider the actual road network and accessibility, 
        # especially during a flood when certain routes may be impassable.
        # Find the nearest available shelter to the stranded agent using spatial distance calculations.
        shelter = min(
            available_shelters,
            key=lambda s: s.geometry.distance(agent.geometry)
        )

        # Evacuate the agent to the shelter by adding them to the shelter's list of sheltered agents 
        # and moving their position to a random point within the shelter's geometry.
        shelter.sheltered_agents.append(agent)
        shelter_position = self.get_random_point_in_polygon(shelter.geometry)
        self.space.move_agent(agent, shelter_position)

        # After evacuation, reset the agent's stranded status and time stranded to reflect that they have been rescued and are no longer in immediate danger.
        agent.stranded = False
        agent.time_stranded = 0                              
    
    # Repeatedly generates random points inside the polygon’s bounding box until one falls inside the polygon.
    def get_random_point_in_polygon(self, polygon):
        """Generate a random point within a given polygon."""
        min_x, min_y, max_x, max_y = polygon.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), 
                                 random.uniform(min_y, max_y))
            if polygon.contains(random_point):
                return random_point
    
    def receive_healthcare(self, patient):
        # Only provide healthcare if the patient is injured and has been injured for at least the specified hours before healthcare.
        available_hospitals = [
            h for h in self.healthcare_facilities
            if len(h.hospitalized_agents) < h.capacity_limit and patient.injured and patient.time_injured >= self.hours_before_healthcare
        ]

        if not available_hospitals:
            return

        # Find the nearest available healthcare facility to the injured agent using spatial distance calculations.
        healthcare = min(
            available_hospitals,
            key=lambda h: h.geometry.distance(patient.geometry)
        )

        # Provide healthcare to the patient by adding them to the healthcare facility's list of hospitalized agents 
        # and moving their position to a random point within the healthcare facility's geometry.
        if patient not in healthcare.hospitalized_agents:
            healthcare.hospitalized_agents.append(patient)

            healthcare_position = self.get_random_point_in_polygon(healthcare.geometry)
            self.space.move_agent(patient, healthcare_position)
    
    # ==========================================================================
    # PHYSICAL RESILIENCE
    # ==========================================================================

    def _get_physical_resilience_from_hazard(self, geometry, agent_type="generic"):
        """..."""
        hazard_var = self.space.get_total_flood_var_at_position(geometry)
        base_map = {
            0: 2.5,
            1: 1.5,
            2: 0.9,
            3: 0.3,
        }
        base_value = base_map.get(hazard_var, 1.0)
        type_adjustment = {
            "person":     0.0,
            "house":      0.2,
            "business":   0.3,
            "school":     0.4,
            "shelter":    0.8,
            "healthcare": 0.8,
            "government": 0.8,
        }
        return max(0.1, base_value + type_adjustment.get(agent_type, 0.0))
    
# ==========================================================================
# PER-STEP AGENT FLOOD EXPOSURE UPDATE
# ==========================================================================
    def _update_agent_flood_exposure(self):
        """..."""
        from agents import flood_agents as FA
        depth_map = {0: 0.0, 1: 0.25, 2: 1.00, 3: 2.00}
        for agent in self.schedule.agents:
            if isinstance(agent, FA.Person_Agent):
                flood_var = self.space.get_total_flood_var_at_position(agent.geometry)
                agent.current_flood_var   = flood_var
                agent.current_flood_depth = depth_map.get(flood_var, 0.0)
            elif isinstance(agent, FA.House_Agent):
                agent.physical_resilience = self._get_physical_resilience_from_hazard(
                    agent.geometry, agent_type="house"
                )

    def debug_person_exposure_state(self, label=""):
        """
        Print counts of person agents by current_flood_var.
        Use this to compare S0, S1, S2, S3.
        """

        counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for agent in self.schedule.agents:
            if isinstance(agent, FA.Person_Agent):
                flood_var = getattr(agent, "current_flood_var", 0)
                try:
                    flood_var = int(flood_var)
                except (TypeError, ValueError):
                    flood_var = 0

                flood_var = max(0, min(3, flood_var))
                counts[flood_var] += 1

        print("\n================ PERSON EXPOSURE CHECK ================")
        print("Label:", label)
        print("Time:", self.schedule.time)
        print("Scenario:", self.dam_scenario_name)
        print("Persons with flood_var 0:", counts[0])
        print("Persons with flood_var 1:", counts[1])
        print("Persons with flood_var 2:", counts[2])
        print("Persons with flood_var 3:", counts[3])
        print("=======================================================\n")

# ==========================================================================
# RIVER-DAM NETWORK UPDATE METHODS
# ==========================================================================

    def _reset_network_states(self):
        """
        Reset runtime flood states on all river-dam network agents.

        This is called at the start of every dynamic network update so that
        each simulation step recomputes active dam-route, HydroRIVERS, and
        channel hazard based on the selected dam scenario.
        """

        # Reset upstream / merged dam route reaches
        for reach in getattr(self.space, "merged_dams", []):
            reach.active = False
            reach.current_q = 0.0
            reach.current_stage = 0.0
            reach.current_sev = 0

        # Reset Malolos HydroRIVERS backbone reaches
        for reach in getattr(self.space, "malolos_hydrorivers", []):
            reach.active = False
            reach.current_q = 0.0
            reach.current_stage = 0.0
            reach.current_sev = 0

        # Reset local Malolos channels / creeks
        for ch in getattr(self.space, "malolos_channels", []):
            ch.active = False
            ch.current_stage = 0.0
            ch.current_sev = 0

    def update_merged_dam_routes(self):
        """
        Activate merged dam-route reaches based on the selected dam scenario.

        This version works with the current merged_dams shapefile, where
        dam_name/source_dam are missing but seg_role contains values such as:
        - angat_dam_entry
        - ipo_dam_entry
        - bustos_entry
        - handoff_to_malolos
        """

        self._reset_network_states()

        for reach in getattr(self.space, "merged_dams", []):
            seg_role = str(getattr(reach, "seg_role", "")).strip().lower()

            # Keep only dam-entry reaches and the final handoff reach.
            is_dam_entry = (
                "dam_entry" in seg_role
                or "angat" in seg_role
                or "ipo" in seg_role
                or "bustos" in seg_role
            )
            is_handoff = "handoff" in seg_role

            if not is_dam_entry and not is_handoff:
                continue

            # Infer dam identity from seg_role.
            dam_key = ""

            if "angat" in seg_role:
                dam_key = "Angat"
            elif "ipo" in seg_role:
                dam_key = "Ipo"
            elif "bustos" in seg_role:
                dam_key = "Bustos"

            # Special case: shared handoff reach.
            # It should carry the strongest signal from all active dams.
            if is_handoff and not dam_key:
                q_release = 0.0
                sev = 0

                for key in ["Angat", "Ipo", "Bustos"]:
                    scenario = self.dam_scenario_lookup.get(key, {})

                    try:
                        q = float(scenario.get("q_release", 0.0))
                    except (TypeError, ValueError):
                        q = 0.0

                    try:
                        s = int(float(scenario.get("severity", 0)))
                    except (TypeError, ValueError):
                        s = 0

                    q_release = max(q_release, q)
                    sev = max(sev, s)

                sev = max(0, min(3, sev))

                if q_release <= 0 and sev <= 0:
                    continue

                reach.active = True
                reach.current_q = q_release
                reach.current_stage = q_release
                reach.current_sev = sev
                continue

            if not dam_key:
                continue

            scenario = self.dam_scenario_lookup.get(dam_key, {})

            try:
                q_release = float(scenario.get("q_release", 0.0))
            except (TypeError, ValueError):
                q_release = 0.0

            try:
                sev = int(float(scenario.get("severity", 0)))
            except (TypeError, ValueError):
                sev = 0

            sev = max(0, min(3, sev))

            if q_release <= 0 and sev <= 0:
                continue

            reach.active = True
            reach.current_q = q_release
            reach.current_stage = q_release
            reach.current_sev = sev

    def handoff_to_malolos_hydrorivers(self):
        """
        Transfer active merged dam-route signal to matching Malolos HydroRIVERS reach.

        Logic:
        - Find active merged dam-route reaches.
        - Read each route's handoff_id.
        - Match handoff_id to HydroRIVERS reach_id.
        - Transfer q, stage, and severity.
        """

        # Create an index of Malolos HydroRIVERS reaches by their reach_id for quick lookup during handoff.
        hydro_index = {
            self._clean_network_id(getattr(r, "reach_id", None)): r
            for r in getattr(self.space, "malolos_hydrorivers", [])
            if self._clean_network_id(getattr(r, "reach_id", None))
        }

        # 
        for route in getattr(self.space, "merged_dams", []):
            if not getattr(route, "active", False):
                continue

            handoff_id = self._clean_network_id(getattr(route, "handoff_id", None))

            if not handoff_id:
                continue

            if handoff_id not in hydro_index:
                print(f"Warning: handoff_id {handoff_id} not found in Malolos HydroRIVERS.")
                continue

            hr = hydro_index[handoff_id]

            hr.active = True
            hr.current_q = max(
                getattr(hr, "current_q", 0.0),
                getattr(route, "current_q", 0.0)
            )
            hr.current_stage = max(
                getattr(hr, "current_stage", 0.0),
                getattr(route, "current_stage", 0.0)
            )
            hr.current_sev = max(
                getattr(hr, "current_sev", 0),
                getattr(route, "current_sev", 0)
            )

    def update_malolos_hydrorivers(self):
        """
        Propagate active hazard downstream through the Malolos HydroRIVERS backbone.

        This method starts from HydroRIVERS reaches that were activated by
        handoff_to_malolos_hydrorivers(), then follows reach_id -> down_id
        connectivity to activate downstream HydroRIVERS reaches.

        Required attributes:
        - MalolosHydroRiver.reach_id
        - MalolosHydroRiver.down_id
        - MalolosHydroRiver.active
        - MalolosHydroRiver.current_q
        - MalolosHydroRiver.current_stage
        - MalolosHydroRiver.current_sev
        """

        # Build lookup: reach_id -> HydroRIVER agent
        hydro_index = {
            self._clean_network_id(getattr(r, "reach_id", None)): r
            for r in getattr(self.space, "malolos_hydrorivers", [])
            if self._clean_network_id(getattr(r, "reach_id", None))
        }

        # Starting reaches are those activated by the merged dam handoff
        starting_reaches = [
            r for r in getattr(self.space, "malolos_hydrorivers", [])
            if getattr(r, "active", False)
        ]

        # Propagation controls
        attenuation = getattr(self, "hydro_attenuation", 0.85)
        max_steps = getattr(self, "hydro_max_downstream_steps", 10)

        try:
            attenuation = float(attenuation)
        except (TypeError, ValueError):
            attenuation = 0.85

        try:
            max_steps = int(max_steps)
        except (TypeError, ValueError):
            max_steps = 10

        attenuation = max(0.0, min(1.0, attenuation))
        max_steps = max(1, max_steps)

        # Propagate from each active handoff reach downstream
        for start_reach in starting_reaches:
            current = start_reach

            current_q = getattr(start_reach, "current_q", 0.0)
            current_stage = getattr(start_reach, "current_stage", 0.0)
            current_sev = getattr(start_reach, "current_sev", 0)

            visited = set()

            for _ in range(max_steps):
                current_id = self._clean_network_id(getattr(current, "reach_id", None))

                if not current_id:
                    break

                if current_id in visited:
                    print(f"Warning: loop detected in Malolos HydroRIVERS at reach_id {current_id}.")
                    break

                visited.add(current_id)

                down_id = self._clean_network_id(getattr(current, "down_id", None))

                if not down_id:
                    break

                if down_id not in hydro_index:
                    break

                downstream = hydro_index[down_id]

                # Attenuate the signal before assigning it downstream
                current_q *= attenuation
                current_stage *= attenuation

                if current_sev > 0:
                    propagated_sev = max(1, round(current_sev * attenuation))
                else:
                    propagated_sev = 0

                # Activate downstream reach
                downstream.active = True
                downstream.current_q = max(
                    getattr(downstream, "current_q", 0.0),
                    current_q
                )
                downstream.current_stage = max(
                    getattr(downstream, "current_stage", 0.0),
                    current_stage
                )
                downstream.current_sev = max(
                    getattr(downstream, "current_sev", 0),
                    propagated_sev
                )

                # Continue downstream
                current = downstream
                current_sev = propagated_sev

    def activate_malolos_channels(self):
        """
        Activate Malolos local channels in two stages.

        Stage 1:
        HydroRIVERS activates directly connected channels using:
            MalolosChannel.con_reach_ == MalolosHydroRiver.reach_id

        Stage 2:
        Active local channels activate connected local channels using:
            MalolosChannel.connection = space-separated local channel reach_id values

        Example:
            Atlag River reach_id = 1
            Atlag River.connection = "112 12 113 111"

        If Atlag River becomes active, channels 112, 12, 113, and 111
        can also become active.
        """

        # ------------------------------------------------------------
        # 1. Activate channels directly connected to active HydroRIVERS
        # ------------------------------------------------------------
        active_hydro = {
            self._clean_network_id(getattr(r, "reach_id", None)): r
            for r in getattr(self.space, "malolos_hydrorivers", [])
            if getattr(r, "active", False)
            and self._clean_network_id(getattr(r, "reach_id", None))
        }

        for ch in getattr(self.space, "malolos_channels", []):
            inherits_hazard = getattr(ch, "inherits_hazard", True)

            if str(inherits_hazard).strip().lower() in ("false", "0", "no", "n"):
                continue

            con_id = self._clean_network_id(getattr(ch, "con_reach_", None))

            if not con_id:
                continue

            if con_id not in active_hydro:
                continue

            source = active_hydro[con_id]

            try:
                transfer_factor = float(getattr(ch, "transfer_factor", 1.0))
            except (TypeError, ValueError):
                transfer_factor = 1.0

            transfer_factor = max(0.0, min(1.0, transfer_factor))

            source_stage = getattr(source, "current_stage", 0.0)
            source_sev = getattr(source, "current_sev", 0)

            inherited_stage = source_stage * transfer_factor
            inherited_sev = round(source_sev * transfer_factor)

            if source_sev > 0:
                inherited_sev = max(1, inherited_sev)

            inherited_sev = max(0, min(3, inherited_sev))

            ch.active = True
            ch.current_stage = max(
                getattr(ch, "current_stage", 0.0),
                inherited_stage
            )
            ch.current_sev = max(
                getattr(ch, "current_sev", 0),
                inherited_sev
            )

        # ------------------------------------------------------------
        # 2. Propagate from active local channels to connected channels
        # ------------------------------------------------------------
        max_channel_steps = getattr(self, "channel_max_steps", 5)
        channel_attenuation = getattr(self, "channel_attenuation", 0.90)

        try:
            max_channel_steps = int(max_channel_steps)
        except (TypeError, ValueError):
            max_channel_steps = 5

        try:
            channel_attenuation = float(channel_attenuation)
        except (TypeError, ValueError):
            channel_attenuation = 0.90

        max_channel_steps = max(1, max_channel_steps)
        channel_attenuation = max(0.0, min(1.0, channel_attenuation))

        # Local channel lookup by reach_id
        channel_index = {
            self._clean_network_id(getattr(ch, "reach_id", None)): ch
            for ch in getattr(self.space, "malolos_channels", [])
            if self._clean_network_id(getattr(ch, "reach_id", None))
        }

        for _ in range(max_channel_steps):
            changed = False

            active_channels = [
                ch for ch in getattr(self.space, "malolos_channels", [])
                if getattr(ch, "active", False)
            ]

            for source in active_channels:
                source_stage = getattr(source, "current_stage", 0.0)
                source_sev = getattr(source, "current_sev", 0)

                if source_sev <= 0:
                    continue

                connected_ids = self._parse_connection_ids(
                    getattr(source, "connection", None)
                )

                for target_id in connected_ids:
                    if target_id not in channel_index:
                        print(
                            f"Warning: local channel connection ID {target_id} "
                            f"not found in malolos_channels."
                        )
                        continue

                    target = channel_index[target_id]

                    if getattr(target, "active", False):
                        continue

                    inherits_hazard = getattr(target, "inherits_hazard", True)

                    if str(inherits_hazard).strip().lower() in ("false", "0", "no", "n"):
                        continue

                    inherited_stage = source_stage * channel_attenuation
                    inherited_sev = round(source_sev * channel_attenuation)

                    if source_sev > 0:
                        inherited_sev = max(1, inherited_sev)

                    inherited_sev = max(0, min(3, inherited_sev))

                    target.active = True
                    target.current_stage = inherited_stage
                    target.current_sev = inherited_sev

                if getattr(self, "debug_network", False):
                    print(
                        f"[step={self.schedule.time}, scenario={self.dam_scenario_name}] "
                        f"Channel propagation: "
                        f"{getattr(source, 'reach_id', None)} ({getattr(source, 'reach_name', None)}) "
                        f"→ {getattr(target, 'reach_id', None)} ({getattr(target, 'reach_name', None)}), "
                        f"sev={target.current_sev}"
                    )                  
                    changed = True

            if not changed:
                break

    def update_dynamic_network_hazard(self):
        """
        Full river-dam network update sequence.

        This updates runtime active/current_sev/current_stage values on:
        - merged dam routes
        - Malolos HydroRIVERS
        - Malolos local channels

        The actual location-based hazard query is handled by flood_space.py
        through self.space.get_total_flood_var_at_position().
        """

        self.update_merged_dam_routes()
        self.handoff_to_malolos_hydrorivers()
        self.update_malolos_hydrorivers()
        self.activate_malolos_channels()

    # Debug method to print the current state of the river-dam network agents
    def debug_network_state(self, label=""):
        """
        Print summary counts of active river-dam network agents.
        Use this to check whether each dam scenario changes the network state.
        """

        active_dam_routes = [
            r for r in getattr(self.space, "merged_dams", [])
            if getattr(r, "active", False)
        ]

        active_hydrorivers = [
            r for r in getattr(self.space, "malolos_hydrorivers", [])
            if getattr(r, "active", False)
        ]

        active_channels = [
            ch for ch in getattr(self.space, "malolos_channels", [])
            if getattr(ch, "active", False)
        ]

        max_dam_sev = max(
            [getattr(r, "current_sev", 0) for r in active_dam_routes],
            default=0
        )

        max_hydro_sev = max(
            [getattr(r, "current_sev", 0) for r in active_hydrorivers],
            default=0
        )

        max_channel_sev = max(
            [getattr(ch, "current_sev", 0) for ch in active_channels],
            default=0
        )

        print("\n================ NETWORK STATE CHECK ================")
        print("Label:", label)
        print("Time:", self.schedule.time)
        print("Disaster period:", self.disaster_period)
        print("Scenario:", self.dam_scenario_name)
        print("Active merged dam routes:", len(active_dam_routes))
        print("Active HydroRIVERS reaches:", len(active_hydrorivers))
        print("Active local channels:", len(active_channels))
        print("Max dam route severity:", max_dam_sev)
        print("Max HydroRIVERS severity:", max_hydro_sev)
        print("Max local channel severity:", max_channel_sev)

        print("\nSample active dam routes:")
        for r in active_dam_routes[:5]:
            print(
                "reach_id=", getattr(r, "reach_id", None),
                "dam_name=", getattr(r, "dam_name", None),
                "source_dam=", getattr(r, "source_dam", None),
                "is_shared=", getattr(r, "is_shared", None),
                "handoff_id=", getattr(r, "handoff_id", None),
                "q=", getattr(r, "current_q", None),
                "sev=", getattr(r, "current_sev", None),
            )

        print("\nSample active HydroRIVERS:")
        for r in active_hydrorivers[:5]:
            print(
                "reach_id=", getattr(r, "reach_id", None),
                "down_id=", getattr(r, "down_id", None),
                "q=", getattr(r, "current_q", None),
                "sev=", getattr(r, "current_sev", None),
            )

        print("\nSample active local channels:")
        for ch in active_channels[:5]:
            print(
                "reach_id=", getattr(ch, "reach_id", None),
                "name=", getattr(ch, "reach_name", None),
                "con_reach_=", getattr(ch, "con_reach_", None),
                "sev=", getattr(ch, "current_sev", None),
            )

        print("=====================================================\n")

    def compute_barangay_stats(self):
        stats = defaultdict(lambda: {
            "population": 0,
            "evacuated": 0,
            "stranded": 0,
            "injured": 0,
            "dead": 0
        })
        for agent in self.schedule.agents:
            if isinstance(agent, FA.Person_Agent):
                brgy = agent.barangay
                stats[brgy]["population"] += 1
                if getattr(agent, "evacuated", False):
                    stats[brgy]["evacuated"] += 1
                if getattr(agent, "stranded", False):
                    stats[brgy]["stranded"] += 1
                if getattr(agent, "injured", False):
                    stats[brgy]["injured"] += 1
                if not getattr(agent, "alive", True):
                    stats[brgy]["dead"] += 1
        return dict(stats)
    
    def _clean_network_id(self, value):
        """
        Convert network IDs into comparable string IDs.

        Examples:
        - np.int64(50012742)      -> "50012742"
        - np.float64(50012742.0)  -> "50012742"
        - "50012742.0"            -> "50012742"
        - nan                     -> ""
        """

        if value is None:
            return ""

        text = str(value).strip()

        if text.lower() in ("", "nan", "none", "null"):
            return ""

        try:
            number = float(text)
            if number.is_integer():
                return str(int(number))
        except (TypeError, ValueError):
            pass

        return text
    
    def _parse_connection_ids(self, value):
        """
        Parse a local-channel connection field.

        Example:
        "112 12 113 111" -> ["112", "12", "113", "111"]

        Supports:
        - space-separated IDs
        - comma-separated IDs
        - semicolon-separated IDs
        """

        if value is None:
            return []

        text = str(value).strip()

        if text.lower() in ("", "nan", "none", "null"):
            return []

        # Normalize separators
        text = text.replace(",", " ")
        text = text.replace(";", " ")

        ids = []
        for part in text.split():
            cleaned = self._clean_network_id(part)
            if cleaned:
                ids.append(cleaned)

        return ids

    def debug_channel_connections(self, label=""):
        """
        Print active local channels and their connection targets.
        Use this to verify HydroRIVERS → channel → connected channels propagation.
        """

        print("\n================ CHANNEL PROPAGATION CHECK ================")
        print("Label:", label)
        print("Time:", self.schedule.time)
        print("Scenario:", self.dam_scenario_name)

        active_channels = [
            ch for ch in getattr(self.space, "malolos_channels", [])
            if getattr(ch, "active", False)
        ]

        print("Active local channels:", len(active_channels))

        for ch in active_channels:
            reach_id = self._clean_network_id(getattr(ch, "reach_id", None))
            reach_name = getattr(ch, "reach_name", None)
            con_reach = self._clean_network_id(getattr(ch, "con_reach_", None))
            connection = getattr(ch, "connection", None)
            sev = getattr(ch, "current_sev", 0)
            stage = getattr(ch, "current_stage", 0.0)

            print(
                "channel_id=", reach_id,
                "name=", reach_name,
                "con_reach_=", con_reach,
                "connection=", repr(connection),
                "sev=", sev,
                "stage=", stage,
            )

        print("===========================================================\n")