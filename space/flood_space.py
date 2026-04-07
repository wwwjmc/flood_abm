"""
The StudyArea class represents a geographical space for simulating flood events and their impacts.
It extends Mesa-Geo's GeoSpace to integrate geospatial data with agent-based modeling.
 
Key features:
- Loads spatial data and converts it into agents (houses, businesses, schools, etc.).
- Loads three network layers: MergedDamRoute, MalolosHydroRiver, MalolosChannel.
- Builds reach-indexed lookup dicts and downstream-graph dicts for fast network traversal.
- Manages flood areas and calculates flood impacts via a three-tier spatial query:
    get_total_flood_var_at_position() = base hazard + backbone bonus + local channel bonus
- Supports agent movement and spatial queries.
 
Network layer classes (defined at the bottom of this file):
  FloodArea        — polygon representing a Project NOAH / hazard zone
  MergedDamRoute   — upstream/shared dam-route line feature
  MalolosHydroRiver — HydroRIVERS subset backbone within the study area
  MalolosChannel   — local creek / estero / drainage channel
 
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
        merged_dams_file,
        malolos_hydrorivers_file,
        malolos_channels_file,
        crs,
    ) -> None: 
        super().__init__(crs=crs) 

        # Create empty lists for each layer
        # Force CRS to exist for older/incompatible mesa-geo builds
        self._crs = pyproj.CRS.from_user_input(crs)
        
        # Initialize attributes for each layer to store agents
        attributes = ["houses", "businesses", "schools", "healthcare", "shelter", "government", "flood_areas",
                      "merged_dams", "malolos_hydrorivers", "malolos_channels"]
        for attr in attributes:
            setattr(self, attr, [])
        
        self.data_crs = "EPSG:32651" # Input CRS used in QGIS

        # -----------------------------------------------------------------------
        # Hazard classification mappings (Project NOAH / Var field)
        # Var = 1 → low hazard   (0 – 0.5 m)
        # Var = 2 → moderate     (> 0.5 – 1.5 m)
        # Var = 3 → high         (> 1.5 m)
        # -----------------------------------------------------------------------
        self.var_to_severity_class = {1: "low", 2: "moderate", 3: "high"}
        self.var_to_flood_depth    = {1: 0.25,  2: 1.00,       3: 2.00}
        self.var_to_severity_score = {1: 1,     2: 2,          3: 3}
 
        # -----------------------------------------------------------------------
        # Load entity layers
        # -----------------------------------------------------------------------

        # Load all spatial files agents from GIS files and convert to model CRS
        self._load_entity_agents_from_file(model, houses_file, FA.House_Agent, "houses", crs)
        self._load_entity_agents_from_file(model, businesses_file, FA.Business_Agent, "businesses", crs)
        self._load_entity_agents_from_file(model, schools_file, FA.School_Agent, "schools", crs)
        self._load_entity_agents_from_file(model, shelter_file, FA.Shelter_Agent, "shelter", crs)
        self._load_entity_agents_from_file(model, healthcare_file, FA.Healthcare_Agent, "healthcare", crs)
        self._load_entity_agents_from_file(model, government_file, FA.Government_Agent, "government", crs)

        # -----------------------------------------------------------------------
        # Load flood hazard maps (three return-period / scenario layers)
        # Flood areas are loaded with var_value set to their actual Var field value.
        # The model masks/reveals them each step via add_flood_maps / remove_flood_maps.
        # -------------------------------------------------------------------------
        self.flood_file_1 = flood_file_1
        self.flood_file_2 = flood_file_2
        self.flood_file_3 = flood_file_3
        self._load_flood_maps_from_file(model, flood_file_1, crs)
        self._load_flood_maps_from_file(model, flood_file_2, crs)
        self._load_flood_maps_from_file(model, flood_file_3, crs)

        # Load the new river/dam connectivity layers
        self._load_network_agents_from_file(model, merged_dams_file, MergedDamRoute, "merged_dams", crs)
        self._load_network_agents_from_file(model, malolos_hydrorivers_file, MalolosHydroRiver, "malolos_hydrorivers", crs)
        self._load_network_agents_from_file(model, malolos_channels_file, MalolosChannel, "malolos_channels", crs)

        # -----------------------------------------------------------------------
        # Build O(1) lookup dicts keyed by reach_id
        # -----------------------------------------------------------------------
        self.merged_dam_index = {
            agent.reach_id: agent
            for agent in self.merged_dams
            if getattr(agent, "reach_id", None) is not None
        }
        self.malolos_hydroriver_index = {
            agent.reach_id: agent
            for agent in self.malolos_hydrorivers
            if getattr(agent, "reach_id", None) is not None
        }
        self.malolos_channel_index = {
            agent.reach_id: agent
            for agent in self.malolos_channels
            if getattr(agent, "reach_id", None) is not None
        }

        # -----------------------------------------------------------------------
        # Build downstream graphs for iterative propagation
        # Maps reach_id → down_id (the immediately downstream reach)
        # -----------------------------------------------------------------------       

        self.merged_dam_graph = {
            agent.reach_id: agent.down_id
            for agent in self.merged_dams
            if getattr(agent, "reach_id", None) is not None
        }
        self.malolos_hydroriver_graph = {
            agent.reach_id: agent.down_id
            for agent in self.malolos_hydrorivers
            if getattr(agent, "reach_id", None) is not None
        }

        print(f"Merged dam route reaches loaded: {len(self.merged_dams)}")
        print(f"Malolos HydroRIVERS reaches loaded: {len(self.malolos_hydrorivers)}")
        print(f"Malolos local channels loaded: {len(self.malolos_channels)}")

 
# ==========================================================================
# INTERNAL DATA LOADING HELPERS
# ==========================================================================

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

# ------------------------------------------------------------------
# Var field conversion helpers
# ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Entity layer loader (houses, businesses, schools, etc.)
    # ------------------------------------------------------------------

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

# ------------------------------------------------------------------
# DAM ROUTE, HYDRORIVER, CHANNELS LOADER 
# ------------------------------------------------------------------

    def _load_network_agents_from_file(self, model, file_path, agent_class, attr_name, crs) -> None:
        """
        Load a line-feature network layer, copy all QGIS attributes into each
        agent, initialise runtime state fields (active, current_q, current_stage,
        current_sev), and store agents in both the GeoSpace and the named list.
 
        Gracefully skips if file_path is None or the layer is empty.
        """
        if not file_path:
            print(f"No file provided for {attr_name}.")
            return
 
        df = self._read_and_prepare_geodata(file_path, crs)
        if df.empty:
            print(f"No features found in {file_path}")
            return
 
        df["unique_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
 
        creator = mg.AgentCreator(agent_class, model=model, crs=crs)
        agents  = creator.from_GeoDataFrame(df, unique_id="unique_id")
 
        for i, agent in enumerate(agents):
            row = df.iloc[i]
 
            # Copy every column from the GeoDataFrame to agent attributes
            for col in df.columns:
                if col == "geometry":
                    continue
                setattr(agent, col, row[col])
 
            # Ensure runtime state fields are initialised correctly regardless
            # of whether the shapefile happens to contain columns with those names.
            if isinstance(agent, MergedDamRoute):
                agent.active        = False
                agent.current_q     = 0.0
                agent.current_stage = 0.0
                agent.current_sev   = 0
 
            elif isinstance(agent, MalolosHydroRiver):
                agent.active        = False
                agent.current_q     = 0.0
                agent.current_stage = 0.0
                agent.current_sev   = 0
 
            elif isinstance(agent, MalolosChannel):
                agent.active        = False
                agent.current_stage = 0.0
                agent.current_sev   = 0
 
        self.add_agents(agents)
        getattr(self, attr_name).extend(agents)
        print(f"Loaded {len(agents)} {attr_name} from {file_path}")
 
    # ------------------------------------------------------------------
    # Flood hazard map loader
    # ------------------------------------------------------------------
 
    def _load_flood_maps_from_file(self, model, flood_file, crs) -> None:
        """
        Load a Project NOAH / hazard polygon layer, classify each polygon by its
        Var field, and create FloodArea agents.
 
        All agents are loaded with their actual var_value set immediately.
        The model controls visibility via add_flood_maps / remove_flood_maps
        (masking approach — zeroes var_value rather than removing agents from
        the GeoSpace, which is faster and avoids index rebuild overhead).
        """
        flood_df = self._read_and_prepare_geodata(flood_file, crs)
 
        possible_var_fields = ["Var", "var"]
        var_field = next((f for f in possible_var_fields if f in flood_df.columns), None)
        if var_field is None:
            raise ValueError(
                f"Flood file '{flood_file}' does not contain a 'Var' field for hazard classification."
            )
 
        flood_df["unique_id"]      = [str(uuid.uuid4()) for _ in range(len(flood_df))]
        flood_df["severity_class"] = flood_df[var_field].apply(self._map_var_to_severity_class)
        flood_df["flood_depth"]    = flood_df[var_field].apply(self._map_var_to_flood_depth)
        flood_df["severity_score"] = flood_df[var_field].apply(self._map_var_to_severity_score)
 
        flood_creator = mg.AgentCreator(FloodArea, model=model, crs=crs)
        flood_areas   = flood_creator.from_GeoDataFrame(flood_df, unique_id="unique_id")
 
        for i, agent in enumerate(flood_areas):
            agent.flood_file        = flood_file
            agent._actual_var_value = self._safe_var_to_int(flood_df.iloc[i][var_field])
            agent.var_value         = agent._actual_var_value
            agent.severity_class    = flood_df.iloc[i]["severity_class"]
            agent.flood_depth       = flood_df.iloc[i]["flood_depth"]
            agent.severity_score    = flood_df.iloc[i]["severity_score"]
 
        self.add_agents(flood_areas)
        self.flood_areas.extend(flood_areas)
        print(f"Flood areas added from {flood_file}: {len(flood_areas)}")
 
 
    # ==========================================================================
    # FLOOD MAP MANAGEMENT
    # ==========================================================================
 
    def remove_flood_maps(self, flood_file: str) -> None:
        """
        Mask flood hazard by zeroing var_value for all FloodArea agents from this
        file.  Agents remain in the GeoSpace — this is the masking approach used
        throughout the model.  Call add_flood_maps() to restore them.
        """
        for agent in self.flood_areas:
            if agent.flood_file == flood_file:
                agent.var_value = 0
 
 
    # ==========================================================================
    # GEOMETRY NORMALISATION
    # ==========================================================================
 
    def _normalize_to_point(self, position):
        """
        Convert any supported geometry input into a shapely Point for spatial checks.
 
        Accepted inputs (in order of preference):
          - shapely Point
          - Any geometry with a .centroid property (Polygon, MultiPolygon, LineString, etc.)
          - Any object with .x and .y attributes
          - (x, y) tuple or list
        """
        if isinstance(position, Point):
            return position
 
        if hasattr(position, "centroid"):
            try:
                return position.centroid
            except Exception:
                pass
 
        if hasattr(position, "x") and hasattr(position, "y"):
            return Point(position.x, position.y)
 
        if isinstance(position, (tuple, list)) and len(position) == 2:
            return Point(position[0], position[1])
 
        raise ValueError(f"Cannot normalize position to Point: {position}")
 
 
    # ==========================================================================
    # STATIC HAZARD LAYER QUERIES
    # ==========================================================================
 
    def get_flood_height_at_position(self, position):
        """
        Return the representative flood depth (metres) from the static hazard
        layer at the given position.  Returns the depth from the highest-severity
        polygon when multiple polygons overlap.  Returns 0.0 if not flooded.
        """
        position_point  = self._normalize_to_point(position)
        matched         = [fa for fa in self.flood_areas
                           if fa.geometry.covers(position_point) and fa.var_value]
 
        if not matched:
            return 0.0
 
        highest = max(matched, key=lambda fa: fa.severity_score)
        return highest.flood_depth
 
    def get_flood_severity_at_position(self, position):
        """
        Return the flood severity class string ('none', 'low', 'moderate', 'high')
        from the static hazard layer at the given position.
        """
        position_point = self._normalize_to_point(position)
        matched        = [fa for fa in self.flood_areas
                          if fa.geometry.covers(position_point) and fa.var_value]
 
        if not matched:
            return "none"
 
        highest = max(matched, key=lambda fa: fa.severity_score)
        return highest.severity_class
 
    def get_flood_var_at_position(self, position):
        """
        Return the numeric hazard Var value (0–3) from the static hazard layer at
        the given position.  Returns 0 if the position is not within any active
        flood polygon.  This is the base term used by get_total_flood_var_at_position().
        """
        position_point = self._normalize_to_point(position)
        matched        = [fa for fa in self.flood_areas
                          if fa.geometry.covers(position_point) and fa.var_value]
 
        if not matched:
            return 0
 
        highest = max(matched, key=lambda fa: fa.severity_score)
        return highest.var_value if highest.var_value is not None else 0
 
 
     # ==========================================================================
    # NETWORK BONUS QUERIES - SIMPLE VERSION
    # ==========================================================================

    def get_malolos_backbone_bonus_at_position(self, position):
        """
        Simple HydroRIVERS bonus:
        - full bonus if within 50 m
        - lower bonus if within 100 m
        - no bonus beyond 100 m
        """
        position_point = self._normalize_to_point(position)
        bonus = 0

        for reach in self.malolos_hydrorivers:
            if not getattr(reach, "active", False):
                continue

            try:
                dist = reach.geometry.distance(position_point)
                sev = int(getattr(reach, "current_sev", 0))

                if dist <= 50:
                    bonus = max(bonus, sev)
                elif dist <= 100 and sev > 1:
                    bonus = max(bonus, sev - 1)
            except Exception:
                continue

        return max(0, min(3, bonus))

    def get_local_channel_bonus_at_position(self, position):
        """
        Simple local channel bonus:
        - full bonus if within 25 m
        - lower bonus if within 75 m
        - no bonus beyond 75 m
        """
        position_point = self._normalize_to_point(position)
        bonus = 0

        for channel in self.malolos_channels:
            if not getattr(channel, "active", False):
                continue

            try:
                dist = channel.geometry.distance(position_point)
                sev = int(getattr(channel, "current_sev", 0))

                if dist <= 25:
                    bonus = max(bonus, sev)
                elif dist <= 75 and sev > 1:
                    bonus = max(bonus, sev - 1)
            except Exception:
                continue

        return max(0, min(3, bonus))

    # ==========================================================================
    # COMBINED FLOOD SEVERITY QUERY - SIMPLE VERSION
    # ==========================================================================

    def get_total_flood_var_at_position(self, position):
        """
        Combined flood severity:

            total = min(3, base_var + river_increment + channel_increment)

        The river and channel layers act as amplifiers, not full replacements
        of the static flood map.
        """
        base_var = self.get_flood_var_at_position(position)
        backbone_bonus = self.get_malolos_backbone_bonus_at_position(position)
        local_bonus = self.get_local_channel_bonus_at_position(position)

        # Convert network bonus into small increments so the flood map remains primary
        river_increment = 1 if backbone_bonus >= 2 else 0
        channel_increment = 1 if local_bonus >= 1 else 0

        return min(3, base_var + river_increment + channel_increment)
 
    # ==========================================================================
    # AGENT MOVEMENT
    # ==========================================================================
 
    def move_agent(self, agent, new_position):
        """
        Move the agent to a new position by updating its geometry attribute.
 
        Parameters
        ----------
        agent        : any Mesa-Geo agent with a geometry attribute
        new_position : shapely Point, Polygon, MultiPolygon, or an object with
                       a geometry attribute (e.g. another agent)
        """
        if isinstance(new_position, Point):
            new_pos = new_position
        elif isinstance(new_position, (Polygon, MultiPolygon)):
            new_pos = new_position.centroid
        elif hasattr(new_position, "geometry") and new_position.geometry is not None:
            geom = new_position.geometry
            new_pos = geom if isinstance(geom, Point) else geom.centroid
        else:
            raise ValueError(
                "new_position must be a Point, Polygon, MultiPolygon, "
                "or an object with a geometry attribute."
            )
 
        agent.geometry = Point(new_pos.x, new_pos.y)
 
 
# ==============================================================================
# GEOSPATIAL AGENT CLASSES
# ==============================================================================
 
class FloodArea(mg.GeoAgent):
    """
    Polygon agent representing a Project NOAH / flood hazard zone.
 
    Attributes
    ----------
    flood_file       : path to the source file (used for masking by file)
    var_value        : currently active Var value (0 when masked, actual value when flood is on)
    _actual_var_value: original Var value from the shapefile, never overwritten
    severity_class   : 'none' | 'low' | 'moderate' | 'high'
    flood_depth      : representative depth in metres
    severity_score   : integer rank (0–3) used for overlap resolution
    """
 
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
        self.flood_file        = flood_file
        self.var_value         = var_value
        self._actual_var_value = var_value
        self.severity_class    = severity_class
        self.flood_depth       = flood_depth
        self.severity_score    = severity_score
 
 
class MergedDamRoute(mg.GeoAgent):
    """
    Line agent representing a reach in the upstream/shared dam-route network.
 
    Derived from HydroRIVERS; used to carry the discharge signal from the
    Angat, Ipo, and Bustos dam systems toward the Malolos study area boundary.
 
    Static attributes (from shapefile)
    -----------------------------------
    reach_id  : unique reach identifier
    down_id   : reach_id of the immediately downstream reach
    seg_role  : 'dam_entry' | 'connector' | 'shared_corridor' | 'handoff' | 'outlet'
    is_shared : dam system tag (e.g. 'yes_ai', 'yes_aib') — matched against dam_scenario_lookup
    handoff_id: reach_id of the MalolosHydroRiver reach that receives the handoff signal
 
    Runtime state (reset each flood step by _reset_network_states)
    ---------------------------------------------------------------
    active        : bool — reach is carrying flow this step
    current_q     : normalised discharge (0.0–1.0)
    current_stage : normalised water-surface elevation (0.0–1.0)
    current_sev   : integer hazard severity (0–3)
    """
 
    def __init__(
        self,
        unique_id,
        model,
        geometry,
        crs,
        reach_id=None,
        down_id=None,
        seg_role="connector",
        is_shared="",
        handoff_id=None,
    ):
        super().__init__(unique_id, model, geometry, crs)
        self.reach_id   = reach_id
        self.down_id    = down_id
        self.seg_role   = seg_role
        self.is_shared  = is_shared
        self.handoff_id = handoff_id
 
        # Runtime state
        self.active        = False
        self.current_q     = 0.0
        self.current_stage = 0.0
        self.current_sev   = 0
 
 
class MalolosHydroRiver(mg.GeoAgent):
    """
    Line agent for the retained HydroRIVERS subset within the Malolos study area.
 
    Receives the handoff signal from MergedDamRoute and routes it downstream
    through the local backbone network.  Active reaches amplify the flood
    severity of nearby agents via get_malolos_backbone_bonus_at_position().
 
    Static attributes (from shapefile)
    -----------------------------------
    reach_id   : unique reach identifier
    down_id    : reach_id of the immediately downstream reach
    rec_handof : bool — this reach is the named recipient of a dam-route handoff
    handoff_id : reach_id it receives handoff from (mirrors MergedDamRoute.handoff_id)
    has_local_ : bool — (has_local_link) reach has a connected MalolosChannel
    outlet_To_ : bool — (outlet_To_bay) reach drains to Manila Bay
 
    Runtime state (reset each flood step)
    ---------------------------------------------------------------
    active, current_q, current_stage, current_sev  (same semantics as MergedDamRoute)
    """
 
    def __init__(
        self,
        unique_id,
        model,
        geometry,
        crs,
        reach_id=None,
        down_id=None,
        rec_handof=False,
        handoff_id=None,
        has_local_=False,
        outlet_To_=False,
    ):
        super().__init__(unique_id, model, geometry, crs)
        self.reach_id   = reach_id
        self.down_id    = down_id
        self.rec_handof = rec_handof
        self.handoff_id = handoff_id
        self.has_local_ = has_local_
        self.outlet_To_ = outlet_To_
 
        # Runtime state
        self.active        = False
        self.current_q     = 0.0
        self.current_stage = 0.0
        self.current_sev   = 0
 
 
class MalolosChannel(mg.GeoAgent):
    """
    Line agent for local rivers, creeks, and esteros (barangay-level drainage).
 
    Inherits hazard from the MalolosHydroRiver backbone reach it is connected to
    (con_reach_).  Active channels amplify the flood severity of nearby agents
    via get_local_channel_bonus_at_position().
 
    Static attributes (from shapefile)
    -----------------------------------
    reach_id       : unique reach identifier
    reach_name     : human-readable name (e.g. 'Estero de Bangkal')
    reach_type     : 'creek' | 'estero' | 'drainage' | 'river'
    priority       : 'high' | 'medium' | 'low' — used for future triage / visualization
    con_reach_     : (connected_reach) reach_id of the MalolosHydroRiver this channel links to
    inherits_hazard: bool — if False, channel is not activated by the backbone signal
    outlet_to_     : bool — (outlet_to_bay) channel drains directly to Manila Bay
    receives_h     : bool — (receives_handoff) channel receives direct handoff (future use)
 
    Runtime state (reset each flood step)
    ---------------------------------------------------------------
    active        : bool
    current_stage : normalised water-surface elevation (0.0–1.0)
    current_sev   : integer hazard severity (0–3)
    Note: MalolosChannel does not track current_q (local drainage volume is not modelled).
    """
 
    def __init__(
        self,
        unique_id,
        model,
        geometry,
        crs,
        reach_id=None,
        reach_name="",
        reach_type="creek",
        priority="medium",
        con_reach_=None,
        inherits_hazard=True,
        outlet_to_=False,
        receives_h=False,
    ):
        super().__init__(unique_id, model, geometry, crs)
        self.reach_id       = reach_id
        self.reach_name     = reach_name
        self.reach_type     = reach_type
        self.priority       = priority
        self.con_reach_     = con_reach_
        self.inherits_hazard = inherits_hazard
        self.outlet_to_     = outlet_to_
        self.receives_h     = receives_h
 
        # Runtime state
        self.active        = False
        self.current_stage = 0.0
        self.current_sev   = 0