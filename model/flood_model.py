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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random
import pandas as pd
from shapely.geometry import Point
from space.flood_space import StudyArea
from mesa import Model
from agents import person_agent_assign as psn_agnt
from mesa.time import RandomActivation
from data_collection import data_collect


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
                    dam_scenario_name="S2", dam_scenarios_file=None, merged_dams_file=None, malolos_hydrorivers_file=None, malolos_channels_file=None, model_crs="EPSG:32651"):
            
            super().__init__()
            self.crs = model_crs

            self.dam_scenarios_file = dam_scenarios_file
            self.dam_scenarios = pd.read_csv(dam_scenarios_file) if dam_scenarios_file else pd.DataFrame()

            # Dam scenario selected from serverrun / batchrun UI
            self.dam_scenario_name = str(dam_scenario_name).strip() if dam_scenario_name else "S2"

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
                "Santos": 8745,
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
                "Santos": 131, 
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
            
            print("Barangays from houses:", list(self.houses_by_barangay.keys())[:10])
            print("Barangays from population:", list(self.barangay_populations.keys())[:10])
            
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
            
        
# ==========================================================================
# STEP FUNCTION
# ==========================================================================
    
    """
        Actions per simulation step (1 step = 1 hour).
        Order of operations:
          1. Determine disaster period from current time.
          2. Add / remove static flood hazard maps on period transitions.
          3. Run the four-stage network update (dam → hydro → channel) during flood.
          4. Re-evaluate each person agent's current flood exposure from the combined
             spatial signal (static hazard + network bonuses).
          5. Activate all agents via the scheduler.
          6. Collect and save data.
    """
    
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
            self.update_merged_dam_routes()
            self.handoff_to_malolos_hydrorivers()
            self.update_malolos_hydrorivers()
            self.activate_malolos_channels()
            self.update_community_hazard()
        
        # ------------------------------------------------------------------
        # 4. Re-evaluate per-agent flood exposure from combined spatial signal.
        #    This updates current_flood_var on every Person_Agent each step so
        #    decision_making_module can read the live, network-augmented severity.
        # ------------------------------------------------------------------  
        if self.disaster_period in ("during_flood", "post_flood"):
            self._update_agent_flood_exposure()
        
        # ------------------------------------------------------------------
        # 5. Activate all agents
        # ------------------------------------------------------------------
        # Activate all agents in the schedule, allowing them to perform their actions based on their defined behaviors and the current state of the environment.
        self.schedule.step()
        self.datacollector.collect(self)

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

# ==========================================================================
# SIMPLE NETWORK UPDATE METHODS
# ==========================================================================

    def _reset_network_states(self):
        """
        Zero out runtime state on every network agent before each flood step.
        """
        for reach in self.space.merged_dams:
            reach.active = False
            reach.current_q = 0.0
            reach.current_stage = 0.0
            reach.current_sev = 0

        for reach in self.space.malolos_hydrorivers:
            reach.active = False
            reach.current_q = 0.0
            reach.current_stage = 0.0
            reach.current_sev = 0

        for ch in self.space.malolos_channels:
            ch.active = False
            ch.current_stage = 0.0
            ch.current_sev = 0

    def update_merged_dam_routes(self):
        """
        Simple version:
        - do NOT propagate through down_id
        - only activate dam entry / handoff reaches
        - read one severity per dam from the CSV
        - allow S0 = no effect
        """
        self._reset_network_states()

        for reach in self.space.merged_dams:
            seg_role = str(getattr(reach, "seg_role", "")).strip().lower()

            if "dam_entry" not in seg_role and seg_role != "handoff":
                continue

            # Prefer explicit dam_name/source_dam, fall back to is_shared
            dam_key = str(
                getattr(reach, "dam_name", None)
                or getattr(reach, "source_dam", None)
                or getattr(reach, "is_shared", "")
            ).strip()

            scenario = self.dam_scenario_lookup.get(dam_key, {})

            q_release = float(scenario.get("q_release", 0.0))
            sev = int(scenario.get("severity", 0))

            sev = max(0, min(3, sev))

            # True baseline / no-effect scenario
            if q_release <= 0 and sev <= 0:
                continue

            reach.active = True
            reach.current_q = q_release
            reach.current_stage = q_release
            reach.current_sev = sev

    def handoff_to_malolos_hydrorivers(self):
        """
        Simple handoff:
        - if a merged dam route is active
        - and it has a handoff_id
        - activate the matching HydroRIVERS reach
        """
        hydro_index = {
            str(getattr(r, "reach_id", "")).strip(): r
            for r in self.space.malolos_hydrorivers
        }

        for route in self.space.merged_dams:
            if not getattr(route, "active", False):
                continue

            handoff_id = str(getattr(route, "handoff_id", "")).strip()
            if not handoff_id:
                continue

            if handoff_id not in hydro_index:
                continue
            
            hr = hydro_index[handoff_id]
            hr.active = True
            hr.current_q = max(getattr(hr, "current_q", 0.0), getattr(route, "current_q", 0.0))
            hr.current_stage = max(getattr(hr, "current_stage", 0.0), getattr(route, "current_stage", 0.0))
            hr.current_sev = max(getattr(hr, "current_sev", 0), getattr(route, "current_sev", 0))

    def update_malolos_hydrorivers(self):
        """
        Simple version:
        - no downstream propagation through down_id yet
        - keep only handoff-received active HydroRIVERS reaches
        """
        pass

    def activate_malolos_channels(self):
        """
        Simple channel activation:
        - if a channel's con_reach_ matches an active HydroRIVERS reach
        - channel inherits that reach's severity
        - optional transfer_factor can reduce inherited stage/severity
        """
        active_hydro = {
            str(getattr(r, "reach_id", "")).strip(): r
            for r in self.space.malolos_hydrorivers
            if getattr(r, "active", False)
        }

        for ch in self.space.malolos_channels:
            inherits_hazard = getattr(ch, "inherits_hazard", True)
            if str(inherits_hazard).lower() in ("false", "0", "no", "n"):
                continue

            con_id = str(getattr(ch, "con_reach_", "")).strip()
            if not con_id or con_id not in active_hydro:
                continue

            source = active_hydro[con_id]

            try:
                transfer_factor = float(getattr(ch, "transfer_factor", 1.0))
            except (TypeError, ValueError):
                transfer_factor = 1.0

            transfer_factor = max(0.0, min(1.0, transfer_factor))

            ch.active = True
            ch.current_stage = getattr(source, "current_stage", 0.0) * transfer_factor

            inherited_sev = round(getattr(source, "current_sev", 0) * transfer_factor)
            if getattr(source, "current_sev", 0) > 0:
                inherited_sev = max(1, inherited_sev)

            ch.current_sev = max(getattr(ch, "current_sev", 0), inherited_sev)

    def update_community_hazard(self):
        """
        Optional helper:
        If you later add community/barangay polygon agents to self.space.communities,
        this computes a simple centroid-based dynamic hazard for each polygon.
        """
        if not hasattr(self.space, "communities"):
            return

        for comm in self.space.communities:
            try:
                comm.current_flood_var = self.space.get_total_flood_var_at_position(comm.geometry.centroid)
            except Exception:
                comm.current_flood_var = 0