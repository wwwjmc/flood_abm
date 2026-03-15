"""
Flood Model Simulation

This model simulates a flood scenario where agents (persons, homes, businesses, shelter system, healthcare, and government) 
interact within a grid-based environment. Each agent has specific attributes and behaviors, such as risk perception, 
movement, distress reactions, and financial transactions. The simulation tracks the effects of flooding on agent behavior, 
economic activities, rescue operations, and shelter systems.

"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import random

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
                 houses_file, businesses_file, schools_file, shelter_file, healthcare_file, government_file,
                 flood_file_1, flood_file_2, flood_file_3, model_crs):
        super().__init__()
        
        self.crs = model_crs

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
            model_crs
        )

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
        self.num_persons = N_persons
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
        
    #-----------------Initialize person agents--------------------------------
        # Create person agents in the model with demographics, wealth classes, and other attributes based on the number of 
        # persons specified and assign them to schools, workplaces, and homes.
        # From person_agent_assign
        psn_agnt.create_person_agents(self)

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

        # Determine the current disaster period based on the defined time thresholds for evacuation and flood duration.
        if self.evacuation_time <= current_time <= self.last_evacuation_time:
            self.disaster_period = "pre_flood_evac_period"
        elif flood_start_time <= current_time < flood_end_time:
            self.disaster_period = "during_flood"
        elif current_time >= flood_end_time:
            self.disaster_period = "post_flood"
        else:
            self.disaster_period = "baseline"

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
        
        # Activate all agents in the schedule, allowing them to perform their actions based on their defined behaviors and the current state of the environment.
        self.schedule.step()
        self.datacollector.collect(self)
    

        # TO MODIFY!!!
        # Save the collected data to a CSV file after each step, allowing for analysis of the simulation results over time. 
        # The data is saved in a 'data_collection' folder
        # This allows for tracking changes in agent attributes, behaviors, and flood conditions throughout the simulation.
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data_collection"))
        os.makedirs(data_folder, exist_ok=True)  # Ensure the folder exists
        
        # Define the full path to the output file
        file_path = os.path.join(data_folder, "serverrun_results.csv")
        
        # Save the data to 'data_collection/model_data.csv'
        data = self.datacollector.get_model_vars_dataframe()
        data.to_csv(file_path, index=False)
        
    def add_flood_maps(self, flood_file):
        for agent in self.space.flood_areas:
            if agent.flood_file == flood_file:
                agent.var_value = agent._actual_var_value  # reveal actual severity

    def remove_flood_maps(self, flood_file):
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

    #-------------------------Initialize businesses----------------------------
    
    def _initialize_businesses(self):
        avg_business_wealth = self.business_gdp / self.num_businesses    
        
        for business in self.space.businesses:
            business.wealth = avg_business_wealth
        
            self.schedule.add(business)

    #---------------------------Initialize schools-----------------------------
    
    def _initialize_schools(self):
        # School GDP share
        avg_school_wealth = self.school_gdp / self.num_schools
        
        # Assign all types to each school and distribute GDP
        for school in self.space.schools:
            school.wealth = avg_school_wealth
            
            # Add school agent to the schedule
            self.schedule.add(school)

    def _initialize_houses(self):
        # Assign random resilience to houses
        for house in self.space.houses:
            self.schedule.add(house)    
    
    #-------------Notify Shelter and get healthcare----------------------------
                       
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
