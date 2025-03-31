import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import math

from .db import (get_recent_distances, get_recent_area_predictions,
                save_blueprint_to_sqlite, get_latest_blueprint_from_sqlite,
                execute_sqlite_query)
from .data_logger import DataLogger  # Assuming renamed processor
from .ai_processor import AIProcessor
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
        self.data_logger = DataLogger(config_path)
        self.ai_processor = AIProcessor(config_path)
        self.ha_client = HomeAssistantClient()

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

        # Get generation settings
        self.generation_config = self.config.get('generation_settings', {
            'distance_window_minutes': 15,
            'area_window_minutes': 10,
            'mds_dimensions': 2,
            'use_adjacency': True,
            'min_points_per_room': 3
        })

        self.status = {"state": "idle", "progress": 0}
        self.latest_job_id = None
        self.latest_generated_blueprint = None

        self._initialized = True

    def generate_blueprint(self):
        """Generate blueprint using relative positioning and area anchoring."""
        logger.info("Starting blueprint generation (Relative Positioning Method)...")
        self.status = {"state": "processing", "progress": 0.0}
        job_start_time = datetime.now()

        try:
            # --- Parameters ---
            distance_window_min = self.generation_config.get('distance_window_minutes', 15)
            area_window_min = self.generation_config.get('area_window_minutes', 10)
            mds_dimensions = self.generation_config.get('mds_dimensions', 2)
            min_points_per_room = self.generation_config.get('min_points_per_room', 3)

            # --- Stage 1: Data Fetching ---
            logger.info("Fetching recent distance and area data...")
            self.status["progress"] = 0.1
            distance_data = get_recent_distances(distance_window_min)
            device_area_map = get_recent_area_predictions(area_window_min)

            if not distance_data:
                logger.warning("No recent distance data found. Cannot generate blueprint.")
                self.status = {"state": "error", "progress": 0, "error": "No distance data"}
                return None

            if not device_area_map:
                logger.warning("No recent area prediction data found. Proceeding without area info for anchoring.")
                # Anchoring will likely fail or be very basic

            # --- Stage 2: Relative Positioning ---
            logger.info(f"Running {mds_dimensions}D relative positioning with {len(distance_data)} distance records...")
            self.status["progress"] = 0.25
            relative_coords = self.ai_processor.run_relative_positioning(distance_data, n_dimensions=mds_dimensions)
            if not relative_coords:
                logger.error("Relative positioning failed.")
                self.status = {"state": "error", "progress": 0, "error": "Relative positioning failed"}
                return None

            # --- Stage 3: Anchoring Setup ---
            logger.info("Setting up for anchoring...")
            self.status["progress"] = 0.4
            # Get HA Areas and create target layout
            ha_areas_list = self.ha_client.get_areas()
            ha_areas_map = {a['area_id']: a for a in ha_areas_list if 'area_id' in a}

            # Get unique active area IDs from device predictions
            active_area_ids = list(set(area_id for area_id in device_area_map.values() if area_id))
            logger.info(f"Found {len(active_area_ids)} active areas with devices.")

            # Get adjacency for better layout if configured
            adjacency = {}
            if self.generation_config.get('use_adjacency', True):
                adjacency = self.ai_processor.calculate_area_adjacency()
                logger.info(f"Calculated adjacency for {len(adjacency)} areas.")

            # Generate target layout with adjacency if available
            target_layout = self.ai_processor.generate_heuristic_layout(active_area_ids, adjacency)
            logger.info(f"Generated target layout with {len(target_layout)} positioned areas.")

            # --- Stage 4: Calculate Anchoring Transform ---
            logger.info("Calculating anchoring transformation...")
            self.status["progress"] = 0.55
            transform_params = self.ai_processor.calculate_anchoring_transform(
                relative_coords, device_area_map, target_layout
            )

            if not transform_params:
                logger.warning("Anchoring transformation calculation failed. Blueprint will use relative coordinates.")
                # Proceed with relative coords, or fail? For now, proceed.
                anchored_coords = relative_coords  # Use relative if anchor fails
            else:
                # --- Stage 5: Apply Transformation ---
                logger.info("Applying anchoring transformation...")
                self.status["progress"] = 0.7
                anchored_coords = self.ai_processor.apply_transform(relative_coords, transform_params)
                logger.info(f"Applied transform to {len(anchored_coords)} entities.")

            # --- Stage 6: Room Generation ---
            logger.info("Generating room geometry...")
            self.status["progress"] = 0.8
            # Group anchored device points by area
            anchored_device_coords_by_area = {}
            for dev_id, area_id in device_area_map.items():
                if area_id and dev_id in anchored_coords:
                    # Filter out scanner coordinates if they exist in anchored_coords
                    # Assuming device IDs don't look like typical scanner IDs
                    scanner_keywords = ['scanner', 'esp', 'beacon_fix', 'gateway']
                    if not any(kw in dev_id.lower() for kw in scanner_keywords):
                        point = anchored_coords[dev_id]
                        anchored_device_coords_by_area.setdefault(area_id, []).append(point)

            # Log areas with points for debugging
            for area_id, points in anchored_device_coords_by_area.items():
                logger.debug(f"Area {area_id}: {len(points)} device points")
                if len(points) < min_points_per_room:
                    logger.warning(f"Area {area_id} has only {len(points)} points, may not generate proper room.")

            rooms = self.ai_processor.generate_rooms_from_points(anchored_device_coords_by_area)
            logger.info(f"Generated {len(rooms)} rooms from device positions.")

            # Add area names back to rooms
            for room in rooms:
                area_id = room.get('area_id')
                if area_id and area_id in ha_areas_map:
                    room['name'] = ha_areas_map[area_id].get('name', room['name'])

            # If rooms were generated but some areas had too few points, add estimated rooms
            if rooms and len(rooms) < len(active_area_ids):
                missing_areas = set(active_area_ids) - set(room.get('area_id') for room in rooms)
                logger.info(f"Missing rooms for {len(missing_areas)} areas, will estimate dimensions.")

                for area_id in missing_areas:
                    if area_id in target_layout:
                        # Get devices in this area
                        devices_in_area = [d for d, a in device_area_map.items() if a == area_id]
                        # Estimate dimensions
                        dims = self.ai_processor.estimate_room_dimensions(area_id, devices_in_area)
                        center = target_layout[area_id]
                        center_z = 1.5  # Default mid-floor height

                        # Calculate bounds from center and dimensions
                        min_x = center['x'] - dims['width'] / 2
                        max_x = center['x'] + dims['width'] / 2
                        min_y = center['y'] - dims['length'] / 2
                        max_y = center['y'] + dims['length'] / 2
                        min_z = center_z - dims['height'] / 2
                        max_z = center_z + dims['height'] / 2

                        room = {
                            'id': f"room_{area_id}",
                            'name': ha_areas_map.get(area_id, {}).get('name', f"Area {area_id}"),
                            'area_id': area_id,
                            'center': {'x': round(center['x'], 2), 'y': round(center['y'], 2), 'z': round(center_z, 2)},
                            'dimensions': dims,
                            'bounds': {
                                'min': {'x': round(min_x, 2), 'y': round(min_y, 2), 'z': round(min_z, 2)},
                                'max': {'x': round(max_x, 2), 'y': round(max_y, 2), 'z': round(max_z, 2)}
                            },
                            'estimated': True  # Flag that this room was estimated, not generated from points
                        }
                        rooms.append(room)
                        logger.info(f"Added estimated room for area {area_id}")

            # --- Stage 7: Wall Generation ---
            logger.info("Generating walls...")
            self.status["progress"] = 0.85
            walls = self.ai_processor.generate_walls_between_rooms(rooms)
            logger.info(f"Generated {len(walls)} walls between rooms.")

            # --- Stage 8: Assemble and Save ---
            logger.info("Assembling and saving blueprint...")
            self.status["progress"] = 0.9
            blueprint = {
                'version': '2.0-MDS-Anchor',
                'generated_at': job_start_time.isoformat(),
                'rooms': rooms,
                'walls': walls,
                'floors': self._group_rooms_into_floors(rooms),  # Adapt based on Z
                'metadata': {
                    'generation_method': 'relative_positioning_mds',
                    'anchoring_success': transform_params is not None,
                    'anchoring_disparity': transform_params.get('disparity') if transform_params else None,
                    'source_distance_records': len(distance_data),
                    'source_area_records': len(device_area_map),
                    'entities_positioned': len(relative_coords),
                    'rooms_generated': len(rooms),
                    'walls_generated': len(walls),
                }
            }

            # Validate the blueprint
            if not self._validate_blueprint(blueprint):
                logger.warning("Generated blueprint failed validation, creating minimal valid blueprint.")
                blueprint = self._create_minimal_valid_blueprint(rooms)

            # Save the blueprint
            saved = self._save_blueprint(blueprint)
            if saved:
                self.latest_generated_blueprint = blueprint
                logger.info("Blueprint saved successfully.")
            else:
                logger.error("Failed to save the generated blueprint.")

            self.status = {"state": "complete", "progress": 1.0, "last_run": datetime.now().isoformat()}
            logger.info("Blueprint generation complete.")
            return blueprint

        except Exception as e:
            logger.error(f"Blueprint generation failed: {e}", exc_info=True)
            self.status = {"state": "error", "progress": 0, "error": str(e)}
            return None

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
