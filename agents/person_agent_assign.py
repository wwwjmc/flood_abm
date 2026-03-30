import uuid
from . import flood_agents as FA

from scipy.stats import beta
import random
import numpy as np


def create_person_agents(model):
    """
    Create person agents in the model with demographics, wealth classes, and other attributes.
    
    Steps:
    1. Define proportions of different wealth classes and ethnicities.
    2. Shuffle and assign wealth classes and ethnicity groups to agents.
    3. Initialize agents with attributes: age, wealth, education, gender, mobility, etc.
    4. Add agents to the simulation's space and schedule.
    5. Assign agents to houses, businesses, and schools.
    
    Parameters:
    - model: The simulation model containing spatial agents and parameters.
    """
    model.num_upper_class_persons = int(0.015 * model.num_persons)  # VALUE UPDATED 1.5% #
    model.num_upper_middle_class_persons = int(0.031 * model.num_persons)  # VALUE UPDATED 3.1% #
    model.num_middle_class_persons = int(0.403 * model.num_persons)  # VALUE UPDATED 40.3% #
    model.num_lower_class_persons = model.num_persons - model.num_upper_class_persons - model.num_upper_middle_class_persons - model.num_middle_class_persons # VALUE UPDATED #

    model.wealth_classes = (
        ["Upper_Class"] * model.num_upper_class_persons +
        ["Upper_Middle_Class"] * model.num_upper_middle_class_persons +
        ["Middle_Class"] * model.num_middle_class_persons +
        ["Lower_Class"] * model.num_lower_class_persons
    )
    random.shuffle(model.wealth_classes)

    model.num_working_class_persons = 0
    model.num_age_0_14_persons = 0
    model.num_age_15_64_persons = 0
    model.num_age_65_100_persons = 0
    model.num_male_persons = 0
    model.num_female_persons = 0

    model.persons_by_wealth_class = {
        "Upper_Class": [],
        "Upper_Middle_Class": [],
        "Middle_Class": [],
        "Lower_Class": []
    }

    for i in range(model.num_persons):
        unique_id = uuid.uuid4().int
        person = FA.Person_Agent(unique_id, model, geometry=None, crs = model.crs)

        person.physical_resilience = 0.0   
        person.employed = False            
        person.student = False
        person.schoolplace = None
        person.workplace = None
        person.household = None
        person.homeless = True
        person.working_class = False
        
        assign_age(model, i, person)
        assign_wealth(model, i, person)
        assign_mobility(model, person)
        assign_education(model, person)
        assign_gender(model, person)
        
        model.space.add_agents(person)
        model.schedule.add(person)

        assign_working_class(model, person)
        assign_SES_index(model, person)

        model.persons_by_wealth_class[person.wealth_class].append(person)

    assign_persons_to_barangays(model)
    assign_persons_to_businesses(model)
    assign_persons_to_schools(model)
    assign_positions_to_homeless(model)

    for p in model.schedule.agents:
        if isinstance(p, FA.Person_Agent) and p.geometry is None:
            fallback = random.choice(model.space.houses)
            point = model.get_random_point_in_polygon(fallback.geometry)
            model.space.move_agent(p, point)    

def assign_persons_to_barangays(model):
    from . import flood_agents as FA

    for house in model.space.houses:
        house.residents = []

    persons = [p for p in model.schedule.agents if isinstance(p, FA.Person_Agent)]
    random.shuffle(persons)

    index = 0

    for brgy_name, population in model.barangay_populations.items():
        brgy_houses = model.houses_by_barangay.get(brgy_name, [])
        if not brgy_houses:
            print(f"No houses found for: {brgy_name}")
            continue

        brgy_persons = []
        for _ in range(population):
            if index >= len(persons):
                break
            brgy_persons.append(persons[index])
            index += 1

        adults = [p for p in brgy_persons if p.age >= 18]
        if not adults and brgy_persons:
            adults = brgy_persons[:]
        
        for house in brgy_houses:
            if not brgy_persons:
                break

            if adults:
                person = adults.pop(0)
                brgy_persons.remove(person)
            else:
                person = brgy_persons.pop(0)

            house.residents.append(person)
            person.household = house
            person.homeless = False
            person.barangay = brgy_name

            person.physical_resilience = model._get_physical_resilience_from_hazard(
                house.geometry,
                agent_type="person"
            )

            point = model.get_random_point_in_polygon(house.geometry)
            model.space.move_agent(person, point)
        
        while brgy_persons:
            person = brgy_persons.pop(0)
            house = random.choice(brgy_houses)
            house.residents.append(person)
            person.household = house
            person.homeless = False
            person.barangay = brgy_name
            person.physical_resilience = model._get_physical_resilience_from_hazard(
                house.geometry,
                agent_type="person"
            )

            point = model.get_random_point_in_polygon(house.geometry)
            model.space.move_agent(person, point)

    print("Total houses:", len(model.space.houses))
    print("Residents assigned:", sum(len(h.residents) for h in model.space.houses))

def assign_pwd_by_brgy(model):
    from . import flood_agents as FA
    import random

    persons_by_brgy = {}

    for agent in model.schedule.agents:
        if isinstance(agent, FA.Person_Agent):
            brgy = getattr(agent, "barangay", None)
            if brgy:
                persons_by_brgy.setdefault(brgy, []).append(agent)
        
    for brgy, persons in persons_by_brgy.items():
        ratio = model.barangay_pwd_ratio.get(brgy, 0)
        pwd_count = int(len(persons) * ratio)
        if pwd_count <= 0:
            continue
        pwd_count = min(pwd_count, len(persons))
        selected = random.sample(persons, pwd_count)
        for person in selected:
            person.pwd = True
    
    for agent in model.schedule.agents:
        if isinstance(agent, FA.Person_Agent):
            assign_mobility(model, agent)

def assign_positions_to_homeless(model):
    houses_by_brgy = {}
    businesses_by_brgy = {}

    for house in model.space.houses:
        brgy = getattr(house, "barangay", None)
        if brgy:
            houses_by_brgy.setdefault(brgy, []).append(house)

    for business in model.space.businesses:
        brgy = getattr(business, "barangay", None)
        if brgy:
            businesses_by_brgy.setdefault(brgy, []).append(business)
    
    for person in model.schedule.agents:
        if not isinstance(person, FA.Person_Agent):
            continue

        if person.homeless:
            brgy = getattr(person, "barangay", None)

            candidate_locations = []

            if brgy:
                candidate_locations += houses_by_brgy.get(brgy, [])
                candidate_locations += businesses_by_brgy.get(brgy, [])

            if not candidate_locations:
                continue

            location = random.choice(candidate_locations)
            point = model.get_random_point_in_polygon(location.geometry)

            model.space.move_agent(person, point)

def assign_working_class(model, agent):
    if 18 <= agent.age <= 64:
        agent.working_class = True
        model.num_working_class_persons += 1           


def assign_education(model, person):
    if person.age >= 18:          
        if random.uniform(0,1) <= model.perc_education_people:
            person.education = 0.9  # Most educated with full understanding of flood risks
        else:
            person.education = random.uniform(0.4,0.8)   # Partial education
    else:
        person.education = 0.3/18*person.age     # Education scales with age for individuals under 18   # VALUE UPDATED #


def assign_gender(model, person):           
    perc_male = 0.501
    if random.uniform(0,1) <= perc_male:            
        model.num_male_persons += 1            
        person.gender = "Male"
    else:
        model.num_female_persons += 1
        person.gender = "Female"
        
     
def assign_age(model, i, person):
    if i < int(0.26 * model.num_persons):	# Assign age for [0-14] age group   # VALUE UPDATED 26% #
        person.age = random.randint(0, 14)
        model.num_age_0_14_persons += 1
    elif i < int(0.93 * model.num_persons):	# [15-64] age group                 # VALUE UPDATED 93% = 26 + 67 #
        person.age = random.randint(15, 64)              
        model.num_age_15_64_persons += 1
    else:  				                # Assign age for [65-100] age group     # VALUE UPDATED 7% #
        person.age = random.randint(65, 100)            
        model.num_age_65_100_persons += 1        


def assign_wealth(model, i, person):
    person.wealth_class = model.wealth_classes[i]  # Assign economy class based on shuffled list

    if person.wealth_class == "Upper_Class":
        person.income = random.uniform(145000, 250000)/365 * 14 #Two weeks saved income
    elif person.wealth_class == "Upper_Middle_Class":
        person.income = random.uniform(85000, 144000)/365 * 14
    elif person.wealth_class == "Middle_Class":
       person.income = random.uniform(25000, 84000)/365 * 14 
    elif person.wealth_class == "Lower_Class":
        person.income = random.uniform(0, 24000)/365 * 14 
    
    model.persons_gdp += person.income
    model.total_gdp += person.income


def assign_SES_index(model, agent):   # High value represents high vulnerability
    # Age vulnerability
    if 0 <= agent.age <= 14:
        age_vul = 0.9
    elif 15 <= agent.age <= 64:
        age_vul = 0.3
    else:
        age_vul = 1.0
        
    # Education vulnerability 
    edu_vul = (1-0.8*agent.education)
    
    # Gender vulnerability
    if agent.gender == "Male":
        gen_vul = 0.3
    else:
        gen_vul = 1
    
    # Wealth status vulnerability
    if agent.wealth_class == "Upper_Class":
        wth_vul = 0.1
    elif agent.wealth_class == "Upper_Middle_Class":
        wth_vul = 0.2
    elif agent.wealth_class == "Middle_Class":
        wth_vul = 0.85
    else: 
        wth_vul = 1

    agent.SES_1 = (age_vul + edu_vul + gen_vul + wth_vul )/4
    agent.SES_2 = (age_vul * edu_vul * gen_vul * wth_vul )**(1/4) 
    agent.vulnerability = (agent.SES_1 + agent.SES_2) / 2
    

def assign_mobility(model, person):
    # Define parameters for the beta function of age
    a_age, b_age = 4, 5
    
    # Define parameters for the logistic function of wealth
    k_wealth, x0_wealth = 0.5, 55
    
    # Normalize age between 0 and 1
    x_age = person.age / 100
    
    # Compute mobility based on age using the beta function
    age_mobility = beta.pdf(x_age, a_age, b_age)
    
    # Compute mobility based on wealth using the logistic function
    wealth_mobility = 1 / (1 + np.exp(-k_wealth * (person.income - x0_wealth)))
    
    # Combine mobility from age and wealth
    person.mobility = age_mobility + wealth_mobility

    if person.pwd:
        person.mobility *= 0.3


def assign_persons_to_businesses(model):
    # Ensure there are businesses available
    if not model.space.businesses:
        print("No businesses available for assignment.")
        return
    
    for economy, persons in model.persons_by_wealth_class.items():
        for person in persons:
            # Assign only if the person is working class and not already employed
            if person.working_class and not person.employed:
                business = random.choice(model.space.businesses)
                business.employees.append(person)
                person.employed = True
                person.workplace = business


def assign_persons_to_schools(model):
    # Ensure there are schools available
    if not model.space.schools:
        print("No schools available for assignment.")
        return
    
    for economy, persons in model.persons_by_wealth_class.items():
        for person in persons:
            # Check if the person is within the school age range and not already assigned to a school
            if 5 <= person.age < 18 and not person.schoolplace:
                school = random.choice(model.space.schools)
                school.students.append(person)
                person.student = True
                person.schoolplace = school