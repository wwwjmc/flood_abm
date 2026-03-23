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
from shapely.geometry import Point, Polygon, MultiPolygon
import agents.flood_agents as FA
import mesa_geo as mg
import geopandas as gpd
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
        
        # Initialize attributes for each layer to store agents
        attributes = ["houses", "businesses", "schools", "healthcare", "shelter", "government", "flood_areas"]
        for attr in attributes:
            setattr(self, attr, [])
        
        self.data_crs = "EPSG:32651" # Input CRS used in QGIS

        # Define mappings for Project NOAH / hazard class values
        # 1 = Low hazard (0-0.5 meters)
        # 2 = Medium hazard (>0.5-1.5 meters)
        # 3 = High hazard (>1.5 meters)
        self.var_to_severity_class = {
            1: "low",
            2: "moderate",
            3: "high",
        }

        # Representative flood depth values in meters for each hazard class
        # These are proxy values, not exact raster depths
        self.var_to_flood_depth = {
            1: 0.25,
            2: 1.00,
            3: 2.00,
        }

        # Numeric ranking for comparing overlapping flood polygons
        self.var_to_severity_score = {
            1: 1,
            2: 2,
            3: 3,
        }
        
        # Load all spatial files agents from GIS files and convert to model CRS
        self._load_entity_agents_from_file(model, houses_file, FA.House_Agent, "houses", crs)
        self._load_entity_agents_from_file(model, businesses_file, FA.Business_Agent, "businesses", crs)
        self._load_entity_agents_from_file(model, schools_file, FA.School_Agent, "schools", crs)
        self._load_entity_agents_from_file(model, shelter_file, FA.Shelter_Agent, "shelter", crs)
        self._load_entity_agents_from_file(model, healthcare_file, FA.Healthcare_Agent, "healthcare", crs)
        self._load_entity_agents_from_file(model, government_file, FA.Government_Agent, "government", crs)
        self.flood_file_1 = flood_file_1
        self.flood_file_2 = flood_file_2
        self.flood_file_3 = flood_file_3
        self._load_flood_maps_from_file(model, flood_file_1, crs)
        self._load_flood_maps_from_file(model, flood_file_2, crs)
        self._load_flood_maps_from_file(model, flood_file_3, crs)


    # Helper method to prepare geospatial data before creating agents
    # This avoids repeating the same cleaning logic across entity layers and flood layers
    def _read_and_prepare_geodata(self, file_path, crs):
        # loads the shapefile/GeoJSON into a GeoDataFrame.
        df = gpd.read_file(file_path)

        # Check if the GeoDataFrame has a CRS, and if not, set it to the expected data CRS
        if df.crs is None:
            df = df.set_crs(self.data_crs)

        # Remove rows with null, empty, or invalid geometries before projection
        df = df[df.geometry.notnull()]
        df = df[~df.geometry.is_empty]
        df = df[df.is_valid]

        # Raise a clear error if the layer becomes empty after cleaning
        if df.empty:
            raise ValueError(f"No valid geometries found in file: {file_path}")

        # All layers are converted into the same coordinate system used by the model
        df = df.to_crs(crs)
        return df

    def _safe_var_to_int(self, value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    def _map_var_to_severity_class(self, var_value):
        var_int = self._safe_var_to_int(var_value)
        return self.var_to_severity_class.get(var_int, "none")

    def _map_var_to_flood_depth(self, var_value):
        var_int = self._safe_var_to_int(var_value)
        return self.var_to_flood_depth.get(var_int, 0.0)

    def _map_var_to_severity_score(self, var_value):
        var_int = self._safe_var_to_int(var_value)
        return self.var_to_severity_score.get(var_int, 0)

    # Helper method to load agents from GIS files, clean data, and convert to model CRS
    def _load_entity_agents_from_file(self, model, file_path, agent_class, attr_name, crs) -> None:
        df = self._read_and_prepare_geodata(file_path, crs)
        print(f"{attr_name}: rows after cleaning = {len(df)}")

        # Create centroid and unique IDs for each feature to use as agent attributes
        df["centroid"] = list(zip(df.geometry.centroid.x, df.geometry.centroid.y))
        df["unique_id"] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Create agents from the GeoDataFrame using Mesa-Geo's AgentCreator
        # Convert GIS rows into agents (House Polygons become House_Agents, etc.) and store them in the corresponding attribute list
        agent_creator = mg.AgentCreator(agent_class, model=model, crs=crs)
        agents = agent_creator.from_GeoDataFrame(df, unique_id="unique_id")
        print(f"{attr_name}: agents created = {len(agents)}")

        for i, agent in enumerate(agents):
            row = df.iloc[i]

            if "ADM4_EN" in df.columns:
                agent.barangay = row["ADM4_EN"]
            if "fid" in df.columns:
                agent.barangay_id = row["fid"]
            if "type_id" in df.columns:
                agent.type_id = row["type_id"]

        # Store the created agents in the corresponding attribute list (e.g., self.houses, self.businesses, etc.)
        # Using extend to add agents to the existing list, ensuring that if this method is called multiple times, it will accumulate agents rather than overwrite them.
        getattr(self, attr_name).extend(agents)
        print(f"{attr_name}: stored agents = {len(getattr(self, attr_name))}")
        self.add_agents(agents)


    # Helper method to load flood maps, clean data, convert to model CRS, and create FloodArea agents
    def _load_flood_maps_from_file(self, model, flood_file, crs) -> None:
        flood_df = self._read_and_prepare_geodata(flood_file, crs)

        # Find the hazard classification field in the flood layer
        # Project NOAH data commonly uses 'Var', but this also checks lowercase variants
        possible_var_fields = ["Var", "var"]
        var_field = next((field for field in possible_var_fields if field in flood_df.columns), None)

        # Raise an error if the flood hazard layer does not contain the expected Var field
        if var_field is None:
            raise ValueError(f"Flood file '{flood_file}' does not contain a 'Var' field for hazard classification.")

        # Keep flood polygons as separate features instead of merging them into one geometry
        # This preserves the Var value of each feature so severity classes can be assigned correctly
        flood_df["unique_id"] = [str(uuid.uuid4()) for _ in range(len(flood_df))]

        # Convert Var to flood severity information
        flood_df["severity_class"] = flood_df[var_field].apply(self._map_var_to_severity_class)
        flood_df["flood_depth"] = flood_df[var_field].apply(self._map_var_to_flood_depth)
        flood_df["severity_score"] = flood_df[var_field].apply(self._map_var_to_severity_score)

        # Create FloodArea agents from the GeoDataFrame
        flood_creator = mg.AgentCreator(FloodArea, model=model, crs=crs)
        flood_areas = flood_creator.from_GeoDataFrame(flood_df, unique_id="unique_id")

        # Assign flood_file and derived flood severity attributes to each agent after creation
        for i, agent in enumerate(flood_areas):
            agent.flood_file = flood_file
            agent._actual_var_value = self._safe_var_to_int(flood_df.iloc[i][var_field])
            agent.var_value = 0 # Default to 0 for non-flooded areas, will be updated to actual Var value if flooded
            agent.severity_class = flood_df.iloc[i]["severity_class"]
            agent.flood_depth = flood_df.iloc[i]["flood_depth"]
            agent.severity_score = flood_df.iloc[i]["severity_score"]

        # Add the agents to the space and extend the flood_areas list
        self.add_agents(flood_areas)
        self.flood_areas.extend(flood_areas)
        print(f"Number of flood areas added to the model from {flood_file}: {len(flood_areas)}")


    def remove_flood_maps(self, flood_file: str) -> None:
        # Find agents that match the flood_file
        agents_to_remove = [agent for agent in self.flood_areas if agent.flood_file == flood_file]
        
        # Remove each matching agent
        for agent in agents_to_remove:
            self.remove_agent(agent)
            self.flood_areas.remove(agent)
        
        # Rebuild the flood_areas list without the removed agents
        self.flood_areas = [agent for agent in self.flood_areas if agent.flood_file != flood_file]
    
    def _normalize_to_point(self, position):
        """
        Convert a supported geometry input into a Point for spatial flood checks.

        Supported inputs:
        - shapely Point
        - shapely Polygon
        - shapely MultiPolygon
        - object with a geometry attribute
        """
        if isinstance(position, Point):
            return position
        elif isinstance(position, (Polygon, MultiPolygon)):
            return position.centroid
        elif hasattr(position, "geometry") and position.geometry is not None:
            geom = position.geometry
            if isinstance(geom, Point):
                return geom
            elif isinstance(geom, (Polygon, MultiPolygon)):
                return geom.centroid

        raise TypeError(
            "position must be a shapely.geometry.Point, Polygon, MultiPolygon, "
            "or an object with a geometry attribute."
        )


    # Iterates through all flood areas and checks if the given position is contained within any of them.
    # If it is, it returns the representative flood depth based on the highest Var severity value found at that location.
    # If the position is not within any flood area, it returns 0.
    def get_flood_height_at_position(self, position):
        """
        Check if the given position is within any flood area and return a representative flood depth if it is.

        Parameters:
        position: A shapely Point, Polygon, MultiPolygon, or an object with a geometry attribute.

        Returns:
        float: The flood height at the position, or 0 if the position is not flooded.
        """
        position_point = self._normalize_to_point(position)

        matched_flood_areas = []

        for flood_area in self.flood_areas:
            # covers() is used instead of contains() so points on polygon boundaries are also treated as flooded
            if flood_area.geometry.covers(position_point):
                matched_flood_areas.append(flood_area)

        # If no flood area covers the position, return 0
        if not matched_flood_areas:
            return 0.0

        # If multiple flood polygons overlap, return the flood depth from the highest severity score
        highest_flood_area = max(matched_flood_areas, key=lambda fa: fa.severity_score)
        return highest_flood_area.flood_depth

    
    # Returns the flood severity class at a given position using the highest overlapping severity class
    def get_flood_severity_at_position(self, position):
        """
        Check if the given position is within any flood area and return the flood severity class.

        Parameters:
        position: A shapely Point, Polygon, MultiPolygon, or an object with a geometry attribute.

        Returns:
        str: The flood severity class at the position, or 'none' if the position is not flooded.
        """
        position_point = self._normalize_to_point(position)

        matched_flood_areas = []

        for flood_area in self.flood_areas:
            if flood_area.geometry.covers(position_point):
                matched_flood_areas.append(flood_area)

        if not matched_flood_areas:
            return "none"

        highest_flood_area = max(matched_flood_areas, key=lambda fa: fa.severity_score)
        return highest_flood_area.severity_class
    
    # Returns the numeric hazard Var value at a given position using the highest overlapping severity
    def get_flood_var_at_position(self, position):
        """
        Check if the given position is within any flood area and return the hazard Var value.

        Parameters:
        position: A shapely Point, Polygon, MultiPolygon, or an object with a geometry attribute.

        Returns:
        int: The hazard Var value at the position, or 0 if the position is not flooded.
        """
        position_point = self._normalize_to_point(position)

        matched_flood_areas = []

        for flood_area in self.flood_areas:
            if flood_area.geometry.covers(position_point):
                matched_flood_areas.append(flood_area)

        if not matched_flood_areas:
            return 0

        highest_flood_area = max(matched_flood_areas, key=lambda fa: fa.severity_score)
        return highest_flood_area.var_value if highest_flood_area.var_value is not None else 0
        # This now uses actual flood severity values derived from the Var field of the hazard layer.
        # flooded = representative depth based on Var
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
        if isinstance(new_position, Point):  # Check if new_position is a Point
            new_pos = new_position
        elif isinstance(new_position, (Polygon, MultiPolygon)):  # Check if new_position is a Polygon or MultiPolygon
            new_pos = new_position.centroid  # Use the centroid for movement
        elif hasattr(new_position, 'geometry') and new_position.geometry is not None:  # Check if new_position has a geometry attribute
            if isinstance(new_position.geometry, Point):
                new_pos = new_position.geometry
            else:
                new_pos = new_position.geometry.centroid
        else:  # If new_position is neither a supported geometry nor has a geometry attribute, raise an error
            raise ValueError("new_position must be a Point, Polygon, MultiPolygon, or an object with a geometry attribute.")

        agent.geometry = Point(new_pos.x, new_pos.y)  # Update the agent's geometry to the new position (as a Point)
    
    
# FloodArea agent class to represent flood polygons with severity attributes derived from the Var field of the hazard layer.
class FloodArea(mg.GeoAgent):
    def __init__(
        self,
        unique_id,
        model,
        geometry,
        crs,
        flood_file=None,
        var_value=None,
        severity_class="none",
        flood_depth=0.0,
        severity_score=0,
    ):
        super().__init__(unique_id, model, geometry, crs)
        self.flood_file = flood_file            # Assign the flood_file attribute
        self.var_value = var_value              # Store the original hazard Var value
        self._actual_var_value = var_value  # Store the original Var value for reference, even if var_value is later updated to 0 for non-flooded areas
        self.severity_class = severity_class    # Store the interpreted flood severity class
        self.flood_depth = flood_depth          # Store the representative flood depth in meters
        self.severity_score = severity_score    # Store numeric severity rank for comparisons