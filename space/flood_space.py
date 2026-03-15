"""
The StudyArea class represents a geographical space for simulating flood events and their impacts. 
It extends Mesa-Geo's GeoSpace to integrate geospatial data with agent-based modeling.

Key features:
- Loads spatial data and converts it into agents (e.g., houses, businesses, schools, etc.).
- Manages flood areas and calculates flood impacts.
- Supports agent movement and spatial queries.

This class is designed for flexible and reusable flood simulation models with geospatial interactions.
"""

import uuid
import agents.flood_agents as FA
import random
import mesa_geo as mg
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pyproj

class StudyArea(mg.GeoSpace):
    # Initialize the StudyArea with geospatial data and create agents
    # This is where to add agents
    def __init__(
        self,
        model,
        houses_file,
        businesses_file,
        schools_file,
        shelter_file,
        healthcare_file,
        government_file,
        flood_file_1,
        flood_file_2,
        flood_file_3,
        crs,
    ) -> None: 
        super().__init__(crs=crs) 

        # Create empty lists for each layer
        # Force CRS to exist for older/incompatible mesa-geo builds
        self._crs = pyproj.CRS.from_user_input(crs)

        # Each one will later store the agents created from GIS files.
        attributes = ["houses", "businesses", "schools", "healthcare", "shelter", "government", "flood_areas"]
        for attr in attributes:
            setattr(self, attr, [])

        self.data_crs = "EPSG:4326"    
        
        # Initialize attributes for each layer to store agents
        attributes = ["houses", "businesses", "schools", "healthcare", "shelter", "government", "flood_areas"]
        for attr in attributes:
            setattr(self, attr, [])
        
        self.data_crs = "EPSG:4326" # Input CRS used in QGIS
        
        # Load all spatial files agents from GIS files and convert to model CRS
        self._load_entity_agents_from_file(model, houses_file, FA.House_Agent, "houses", crs)
        self._load_entity_agents_from_file(model, businesses_file, FA.Business_Agent, "businesses", crs)
        self._load_entity_agents_from_file(model, schools_file, FA.School_Agent, "schools", crs)
        self._load_entity_agents_from_file(model, shelter_file, FA.Shelter_Agent, "shelter", crs)
        self._load_entity_agents_from_file(model, healthcare_file, FA.Healthcare_Agent, "healthcare", crs)
        self._load_entity_agents_from_file(model, government_file, FA.Government_Agent, "government", crs)
        self._load_flood_maps_from_file(model, flood_file_1, crs)
        self._load_flood_maps_from_file(model, flood_file_2, crs)
        self._load_flood_maps_from_file(model, flood_file_3, crs)


    # Helper method to load agents from GIS files, clean data, and convert to model CRS
    def _load_entity_agents_from_file(self, model, file_path, agent_class, attr_name, crs) -> None:
        # loads the shapefile/GeoJSON into a GeoDataFrame.
        df = gpd.read_file(file_path)

        # Check if the GeoDataFrame has a CRS, and if not, set it to the expected data CRS
        if df.crs is None:
            df = df.set_crs(self.data_crs)

        df = df.to_crs(crs)                 # All layers are converted into the same coordinate system used by the model
        df = df[df.geometry.notnull()]      # Remove rows with null geometries
        df = df[df.is_valid]                # Remove rows with invalid geometries
        print(f"{attr_name}: rows after cleaning = {len(df)}")

        # Create centroid and unique IDs for each feature to use as agent attributes
        df["centroid"] = list(zip(df.geometry.centroid.x, df.geometry.centroid.y))  
        df["unique_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Create agents from the GeoDataFrame using Mesa-Geo's AgentCreator
        # Convert GIS rows into agents (House Polygons become House_Agents, etc.) and store them in the corresponding attribute list
        agent_creator = mg.AgentCreator(agent_class, model=model, crs=crs)
        agents = agent_creator.from_GeoDataFrame(df, unique_id="unique_id")
        print(f"{attr_name}: agents created = {len(agents)}")

        # Store the created agents in the corresponding attribute list (e.g., self.houses, self.businesses, etc.)
        # Using extend to add agents to the existing list, ensuring that if this method is called multiple times, it will accumulate agents rather than overwrite them.
        getattr(self, attr_name).extend(agents)
        print(f"{attr_name}: stored agents = {len(getattr(self, attr_name))}")
        self.add_agents(agents)


    # Helper method to load flood maps, clean data, convert to model CRS, and create FloodArea agents
    def _load_flood_maps_from_file(self, model, flood_file, crs) -> None:
        # Load the GeoDataFrame with the appropriate CRS
        flood_df = gpd.read_file(flood_file).to_crs(crs)
    
        # Remove invalid, empty, or null geometries
        flood_df = flood_df[flood_df.is_valid]
        
        # Merge all flood polygons into a single geometry to create one FloodArea agent per file
        merged_polygon = flood_df.union_all()
        flood_df = gpd.GeoDataFrame(geometry=[merged_polygon], crs=crs)
    
        # Assign unique IDs to each feature
        # flood_df['id'] = range(1, len(flood_df) + 1)
    
        # Create FloodArea agents from the GeoDataFrame
        flood_creator = mg.AgentCreator(FloodArea, model=model, crs=crs)
        flood_area = flood_creator.from_GeoDataFrame(flood_df)
    
        # Assign flood_file to each agent after creation
        for agent in flood_area:
            agent.flood_file = flood_file
    
        # Add the agents to the space and extend the flood_areas list
        self.add_agents(flood_area)
        self.flood_areas.extend(flood_area)
        print(f"Number of flood areas added to the model: {len(flood_area)}")


    def remove_flood_maps(self, flood_file: str) -> None:
        # Find agents that match the flood_file
        agents_to_remove = [agent for agent in self.flood_areas if agent.flood_file == flood_file]
        
        # Remove each matching agent
        for agent in agents_to_remove:
            self.remove_agent(agent)
            self.flood_areas.remove(agent)

    # Iterates through all flood areas and checks if the given position is contained within any of them. 
    # If it is, it returns a random flood height between 10 and 55. If the position is not within any flood area, it returns 0.
    def get_flood_height_at_position(self, position):
        """
        Check if the given position is within any flood area and return a random flood height if it is.
        
        Parameters:
        position (shapely.geometry.Point): The position to check.
        
        Returns:
        float: The flood height at the position, or 0 if the position is not flooded.
        """
        for flood_area in self.flood_areas:
            if flood_area.geometry.contains(position):
                return random.uniform(10, 55)
        return 0.0
    # This is a simplified representation of flood depth. It does not read actual flood depth values from the map.
    # flooded = random height
    # not flooded = zero
    
    # Allows agents to move to a new position by updating their geometry attribute. 
    # It checks if the new position is a Point or Polygon and uses the centroid for movement. 
    # If the new position is an object with a geometry attribute, it also uses the centroid of that geometry. 
    def move_agent(self, agent, new_position):
        """
        Move the agent to a new position.
        
        Parameters:
        agent: The agent to move.
        new_position: A geometry object or an object with a geometry attribute.
        """
        if isinstance(new_position, (Point, Polygon)):  # Check if new_position is a Point or Polygon
            new_pos = new_position.centroid             # Use the centroid for movement
        elif hasattr(new_position, 'geometry'):         # Check if new_position has a geometry attribute
            new_pos = new_position.geometry.centroid    
        else:                                           # If new_position is neither a Point/Polygon nor has a geometry attribute, raise an error
            raise ValueError("new_position must be a Point, Polygon, or an object with a geometry attribute.")
        agent.geometry = Point(new_pos.x, new_pos.y)    # Update the agent's geometry to the new position (as a Point)
        

class FloodArea(mg.GeoAgent):
    def __init__(self, unique_id, model, geometry, crs, flood_file=None):
        super().__init__(unique_id, model, geometry, crs)
        self.flood_file = flood_file  # Assign the flood_file attribute
        # Initialize other necessary attributes