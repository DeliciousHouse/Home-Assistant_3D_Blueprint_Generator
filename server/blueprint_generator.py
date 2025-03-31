import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import math

from .bluetooth_processor import BluetoothProcessor
from .ai_processor import AIProcessor
from .db import save_blueprint_to_sqlite, get_latest_blueprint_from_sqlite, execute_sqlite_query
from .ha_client import HomeAssistantClient

logger = logging.getLogger(__name__)

class BlueprintGenerator:
    """Generate 3D blueprints from room detection data."""

    # Class variable to hold the single instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(BlueprintGenerator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the blueprint generator."""
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Use standardized config loader
        from .config_loader import load_config

        # Load configuration
        self.config = load_config(config_path)

        # Initialize components with config path for consistency
        self.bluetooth_processor = BluetoothProcessor(config_path)
        self.ai_processor = AIProcessor(config_path)

        # Get validation settings
        self.validation = self.config.get('blueprint_validation', {
            'min_room_area': 4,
            'max_room_area': 100,
            'min_room_dimension': 1.5,
            'max_room_dimension': 15,
            'min_wall_thickness': 0.1,
            'max_wall_thickness': 0.5,
            'min_ceiling_height': 2.2,
            'max_ceiling_height': 4.0
        })

        self.status = {"state": "idle", "progress": 0}
        self.latest_job_id = None
        self.latest_generated_blueprint = None

        self._initialized = True

    def generate_blueprint(self):
        """Generate a 3D blueprint using Area predictions and AI/heuristics."""
        try:
            logger.info("Starting AI-driven blueprint generation process...")
            self.status = {"state": "processing", "progress": 0.1}

            # 1. Get current device -> area predictions from the processor
            logger.info("Processing sensor data for current area predictions...")
            # This now logs observations to DB and returns current state
            device_area_predictions = self.bluetooth_processor.process_sensor_data()
            self.status["progress"] = 0.3

            if not device_area_predictions:
                logger.warning("No device area predictions available. Cannot generate blueprint.")
                self.status = {"state": "idle", "progress": 0, "error": "No device data"}
                return None

            # 2. Get HA Area definitions
            ha_client = HomeAssistantClient()
            all_ha_areas_list = ha_client.get_areas()
            ha_areas = {area['area_id']: area['name'] for area in all_ha_areas_list if area.get('area_id')}
            if not ha_areas:
                 logger.error("Could not fetch Area definitions from Home Assistant.")
                 self.status = {"state": "idle", "progress": 0, "error": "No HA Areas found"}
                 return None
            logger.info(f"Using {len(ha_areas)} HA Areas for blueprint structure.")
            self.status["progress"] = 0.4

            # 3. Determine Area Adjacency (using AI Processor)
            logger.info("Calculating area adjacency based on transitions...")
            adjacency = self.ai_processor.calculate_area_adjacency()
            self.status["progress"] = 0.5

            # 4. Generate Heuristic Layout (using AI Processor)
            logger.info("Generating heuristic layout for areas...")
            # Use only areas present in adjacency keys or values, or predicted areas
            active_area_ids = set(adjacency.keys())
            for neighbors in adjacency.values():
                active_area_ids.update(neighbors)
            active_area_ids.update(a for a in device_area_predictions.values() if a)
            relevant_area_ids = [aid for aid in active_area_ids if aid in ha_areas] # Ensure they exist in HA

            layout_centers = self.ai_processor.generate_heuristic_layout(relevant_area_ids, adjacency)
            self.status["progress"] = 0.6

            # 5. Create Room Definitions
            rooms = []
            devices_per_area = {}
            for device, area_id in device_area_predictions.items():
                 if area_id:
                     devices_per_area.setdefault(area_id, []).append(device)

            for area_id in relevant_area_ids: # Iterate through relevant areas
                 area_name = ha_areas.get(area_id, f"Unknown Area {area_id}")
                 center = layout_centers.get(area_id, {'x': 0.0, 'y': 0.0}) # Default if layout failed
                 center_x = center['x']
                 center_y = center['y']
                 center_z = 1.5 # Default mid-floor height (can be refined later)

                 # Estimate dimensions
                 dims = self.ai_processor.estimate_room_dimensions(area_id, devices_per_area.get(area_id, []))

                 # Calculate bounds from center and dimensions
                 min_x = center_x - dims['width'] / 2
                 max_x = center_x + dims['width'] / 2
                 min_y = center_y - dims['length'] / 2
                 max_y = center_y + dims['length'] / 2
                 min_z = center_z - dims['height'] / 2
                 max_z = center_z + dims['height'] / 2

                 room = {
                     'id': f"room_{area_id}", # Use area_id for room ID
                     'name': area_name,
                     'center': {'x': round(center_x, 2), 'y': round(center_y, 2), 'z': round(center_z, 2)},
                     'dimensions': dims,
                     'bounds': {
                         'min': {'x': round(min_x, 2), 'y': round(min_y, 2), 'z': round(min_z, 2)},
                         'max': {'x': round(max_x, 2), 'y': round(max_y, 2), 'z': round(max_z, 2)}
                     },
                     'devices': devices_per_area.get(area_id, []),
                     'area_id': area_id
                 }
                 rooms.append(room)

            if not rooms:
                 logger.warning("No rooms could be generated based on area predictions.")
                 self.status = {"state": "idle", "progress": 0, "error": "No rooms generated"}
                 return None
            self.status["progress"] = 0.7

            # 6. Generate Walls (using AI Processor - placeholder for now)
            logger.info("Generating walls between adjacent areas...")
            walls = self._generate_walls_ai(rooms, adjacency) # Pass generated rooms and adjacency
            logger.info(f"Generated {len(walls)} walls.")
            self.status["progress"] = 0.8

            # 7. Assemble Final Blueprint
            blueprint = {
                'version': '1.1-AI-Phase1', # Indicate version/method
                'generated_at': datetime.now().isoformat(),
                'rooms': rooms,
                'walls': walls,
                'floors': self._group_rooms_into_floors(rooms),
                'metadata': {
                    'device_count': len(device_area_predictions),
                    'room_count': len(rooms),
                    'wall_count': len(walls),
                    'source': 'heuristic_area_generator'
                }
            }

            # 8. Validation (Keep this)
            logger.info("Validating final blueprint...")
            if not self._validate_blueprint(blueprint):
                logger.warning("Generated blueprint failed validation. Attempting minimal.")
                blueprint = self._create_minimal_valid_blueprint(rooms)
                # No need to re-validate minimal blueprint usually

            self.status["progress"] = 0.9

            # 9. Save and Cache
            logger.info(f"Saving final blueprint...")
            saved = self._save_blueprint(blueprint)
            if saved:
                self.latest_generated_blueprint = blueprint
            else:
                logger.error("Failed to save the generated blueprint.")

            self.status = {"state": "idle", "progress": 1.0, "last_run": datetime.now().isoformat()}
            return blueprint

        except Exception as e:
            logger.error(f"Critical error during blueprint generation: {e}", exc_info=True)
            self.status = {"state": "error", "progress": 0, "error": str(e)}
            return None

    def _generate_walls_ai(self, rooms: List[Dict], adjacency: Dict[str, List[str]]):
        """Generates wall segments based on adjacency and AI prediction (placeholder)."""
        walls = []
        processed_pairs = set()
        # Get threshold from config, default to 0 (no walls in phase 1)
        wall_prob_threshold = self.config.get('ai_settings', {}).get('wall_probability_threshold', 0.0)

        logger.info(f"Generating walls with probability threshold: {wall_prob_threshold}")

        for area1_id, neighbors in adjacency.items():
            room1 = next((r for r in rooms if r.get('area_id') == area1_id), None)
            if not room1: continue

            for area2_id in neighbors:
                pair = tuple(sorted((area1_id, area2_id)))
                if pair in processed_pairs: continue

                room2 = next((r for r in rooms if r.get('area_id') == area2_id), None)
                if not room2: continue

                # In Phase 1, we don't have real transition data easily accessible here
                # In Phase 2, query DB for transitions between area1_id and area2_id
                transition_data = {} # Placeholder

                # Predict wall probability (uses placeholder in AI processor)
                wall_prob = self.ai_processor.predict_walls_between_areas(area1_id, area2_id, transition_data)

                if wall_prob >= wall_prob_threshold:
                    # Find overlapping segment using the geometric helper logic
                    segment = self._find_overlapping_segment_ai(room1, room2)
                    if segment:
                        # Create wall object
                        thickness = self.validation.get('min_wall_thickness', 0.1)
                        height = min(room1['dimensions']['height'], room2['dimensions']['height'])
                        height = max(self.validation.get('min_ceiling_height', 2.2), min(self.validation.get('max_ceiling_height', 4.0), height))

                        dx = segment['end']['x'] - segment['start']['x']
                        dy = segment['end']['y'] - segment['start']['y']
                        angle = math.atan2(dy, dx)
                        length = math.sqrt(dx**2 + dy**2)

                        if length > 0.1: # Avoid zero-length walls
                             walls.append({
                                 'start': segment['start'],
                                 'end': segment['end'],
                                 'thickness': thickness,
                                 'height': height,
                                 'angle': angle,
                                 'probability': wall_prob # Store probability
                             })

                processed_pairs.add(pair)

        # TODO: Add logic to merge collinear walls if needed (can adapt from _generate_walls_geometric)
        logger.info(f"Generated {len(walls)} placeholder walls.")
        return walls

    def _find_overlapping_segment_ai(self, room1: Dict, room2: Dict) -> Optional[Dict]:
        """Finds the likely boundary segment between two potentially adjacent rooms (using bounds)."""
        bounds1 = room1['bounds']
        bounds2 = room2['bounds']
        adjacency_threshold = 0.5 # How close counts as touching

        # Calculate separation/overlap in X and Y
        x_sep = max(0, bounds1['min']['x'] - bounds2['max']['x'], bounds2['min']['x'] - bounds1['max']['x'])
        y_sep = max(0, bounds1['min']['y'] - bounds2['max']['y'], bounds2['min']['y'] - bounds1['max']['y'])

        # Determine primary axis of adjacency (closer edge)
        if x_sep <= y_sep + adjacency_threshold and x_sep < adjacency_threshold + 0.1: # Allow slight tolerance, prefer vertical
            # Vertical wall candidate (shared X edge)
            if abs(bounds1['max']['x'] - bounds2['min']['x']) < adjacency_threshold:
                wall_x = (bounds1['max']['x'] + bounds2['min']['x']) / 2
            elif abs(bounds2['max']['x'] - bounds1['min']['x']) < adjacency_threshold:
                wall_x = (bounds2['max']['x'] + bounds1['min']['x']) / 2
            else: return None # Not close enough on X

            # Find Y overlap
            start_y = max(bounds1['min']['y'], bounds2['min']['y'])
            end_y = min(bounds1['max']['y'], bounds2['max']['y'])

            if end_y > start_y: # Must have overlap
                 return {'start': {'x': wall_x, 'y': start_y}, 'end': {'x': wall_x, 'y': end_y}}

        elif y_sep <= x_sep + adjacency_threshold and y_sep < adjacency_threshold + 0.1:
            # Horizontal wall candidate (shared Y edge)
            if abs(bounds1['max']['y'] - bounds2['min']['y']) < adjacency_threshold:
                wall_y = (bounds1['max']['y'] + bounds2['min']['y']) / 2
            elif abs(bounds2['max']['y'] - bounds1['min']['y']) < adjacency_threshold:
                wall_y = (bounds2['max']['y'] + bounds1['min']['y']) / 2
            else: return None # Not close enough on Y

            # Find X overlap
            start_x = max(bounds1['min']['x'], bounds2['min']['x'])
            end_x = min(bounds1['max']['x'], bounds2['max']['x'])

            if end_x > start_x: # Must have overlap
                 return {'start': {'x': start_x, 'y': wall_y}, 'end': {'x': end_x, 'y': wall_y}}

        return None # No clear adjacent edge found

    def _create_minimal_valid_blueprint(self, rooms):
        """Create a minimal valid blueprint when validation fails."""
        # Basic structure with just rooms
        return {
            'version': '1.0',
            'generated_at': datetime.now().isoformat(),
            'rooms': rooms,
            'walls': [],
            'floors': [{'level': 0, 'rooms': [r['id'] for r in rooms]}],
            'metadata': {
                'room_count': len(rooms),
                'is_minimal': True
            }
        }

    def _validate_blueprint(self, blueprint: Dict) -> bool:
        """Validate generated blueprint."""
        try:
            # Log the validation criteria
            logger.debug(f"Validation criteria: {self.validation}")

            # Check if blueprint has the required keys
            if not all(key in blueprint for key in ['rooms', 'walls']):
                raise ValueError("Blueprint is missing required keys: rooms or walls")

            # Validate rooms
            for room in blueprint['rooms']:
                # Log room data for debugging
                logger.debug(f"Validating room: {room['id']}")
                logger.debug(f"Room dimensions: {room['dimensions']}")

                # Check dimensions
                dims = room['dimensions']
                if dims['width'] < self.validation['min_room_dimension'] or \
                   dims['width'] > self.validation['max_room_dimension'] or \
                   dims['length'] < self.validation['min_room_dimension'] or \
                   dims['length'] > self.validation['max_room_dimension'] or \
                   dims['height'] < self.validation['min_ceiling_height'] or \
                   dims['height'] > self.validation['max_ceiling_height']:
                    logger.error(f"Room {room['id']} dimensions out of valid range: width={dims['width']}, length={dims['length']}, height={dims['height']}")
                    raise ValueError(f"Room {room['id']} dimensions out of valid range")

                # Check area
                area = dims['width'] * dims['length']
                if area < self.validation['min_room_area'] or \
                   area > self.validation['max_room_area']:
                    logger.error(f"Room {room['id']} area out of valid range: {area}")
                    return False

            # Validate walls
            for idx, wall in enumerate(blueprint['walls']):
                # Log wall data
                logger.debug(f"Validating wall {idx}: {wall}")

                # Check thickness
                if wall['thickness'] < self.validation['min_wall_thickness'] or \
                   wall['thickness'] > self.validation['max_wall_thickness']:
                    logger.error(f"Wall {idx} thickness out of valid range: {wall['thickness']}")
                    raise ValueError(f"Wall {idx} thickness out of valid range")

                # Check height
                if wall['height'] < self.validation['min_ceiling_height'] or \
                   wall['height'] > self.validation['max_ceiling_height']:
                    logger.error(f"Wall {idx} height out of valid range: {wall['height']}")
                    raise ValueError(f"Wall {idx} height out of valid range")

            logger.info("Blueprint validation passed successfully")
            return True

        except Exception as e:
            logger.error(f"Blueprint validation failed with exception: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def get_latest_blueprint(self):
        """Get the latest blueprint from the database."""
        try:
            # First check in-memory cache
            if hasattr(self, 'latest_generated_blueprint') and self.latest_generated_blueprint:
                return self.latest_generated_blueprint

            # Get from SQLite using helper function
            blueprint = get_latest_blueprint_from_sqlite()

            if blueprint:
                # Store in memory for later access
                self.latest_generated_blueprint = blueprint
                logger.info(f"Retrieved latest blueprint with {len(blueprint.get('rooms', []))} rooms")
                return blueprint

            logger.warning("No blueprint found in database")
            return None

        except Exception as e:
            logger.error(f"Error retrieving latest blueprint: {e}")
            return None

    def get_status(self):
        """Get the current status of blueprint generation."""
        return self.status

    def _refine_blueprint(self, blueprint: Dict) -> Dict:
        """Refine the blueprint using AI techniques."""
        try:
            logger.debug("Applying AI-based blueprint refinement")
            refined_blueprint = self.ai_processor.refine_blueprint(blueprint) if self.ai_processor else blueprint

            # Validate the refined blueprint
            if self._validate_blueprint(refined_blueprint):
                logger.debug("AI refinement successful")
                return refined_blueprint
            else:
                logger.warning("AI-refined blueprint failed validation, using original")
                return blueprint

        except Exception as e:
            logger.warning(f"Blueprint refinement failed: {str(e)}")
            return blueprint

    def _save_blueprint(self, blueprint):
        """Save blueprint to database."""
        try:
            if not blueprint:
                logger.warning("Blueprint is empty, not saving to database.")
                return False

            # Use helper function from db
            success = save_blueprint_to_sqlite(blueprint)

            if success:
                # Update in-memory cache
                self.latest_generated_blueprint = blueprint
                logger.info(f"Successfully saved blueprint with {len(blueprint.get('rooms', []))} rooms")

            return success

        except Exception as e:
            logger.error(f"Failed to save blueprint: {e}")
            return False

    def _group_rooms_into_floors(self, rooms: List[Dict]) -> List[Dict]:
        """Group rooms into floors based on their z-coordinate."""
        if not rooms:
            return []

        # Group rooms by their z-coordinate (floor level)
        floors_dict = {}
        for room in rooms:
            # Use the minimum z-coordinate as the floor level
            floor_level = int(room['bounds']['min']['z'])

            if floor_level not in floors_dict:
                floors_dict[floor_level] = []

            floors_dict[floor_level].append(room['id'])

        # Convert to list of floor objects
        floors = []
        for level, room_ids in sorted(floors_dict.items()):
            floors.append({
                'level': level,
                'rooms': room_ids,
                'height': 3.0  # Default floor height
            })

        logger.info(f"Grouped {len(rooms)} rooms into {len(floors)} floors")
        return floors

def ensure_reference_positions():
    """Make sure we have at least some reference positions in the SQLite database"""
    from .db import get_sqlite_connection

    conn = get_sqlite_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM device_positions")
    count = cursor.fetchone()[0]
    conn.close()

    if count == 0:
        logger.info("No device positions in database, creating initial reference points")
        # Create at least 3 reference points for the system to work with
        default_positions = {
            "reference_point_1": {"x": 0, "y": 0, "z": 0},
            "reference_point_2": {"x": 5, "y": 0, "z": 0},
            "reference_point_3": {"x": 0, "y": 5, "z": 0}
        }
        # Use SQLite functions directly
        from .db import save_device_position_to_sqlite
        for device_id, position in default_positions.items():
            position['source'] = 'initial_setup'
            position['accuracy'] = 1.0
            save_device_position_to_sqlite(device_id, position)
        return default_positions
    return None
