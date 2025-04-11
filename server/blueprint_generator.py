import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import math

import numpy as np
from scipy.spatial import Delaunay

from .bluetooth_processor import BluetoothProcessor
from .ai_processor import AIProcessor
from .db import (
    get_recent_distances,          # ESSENTIAL: To get data for relative positioning
    get_recent_area_predictions,   # ESSENTIAL: To get data for anchoring
    save_blueprint_to_sqlite,      # ESSENTIAL: To save the final result
    get_latest_blueprint_from_sqlite, # ESSENTIAL: To retrieve the last blueprint for API/UI
    execute_sqlite_query # Only if needed for other direct queries
)

# --- Imports from other project files ---
from .bluetooth_processor import BluetoothProcessor # Still needed to instantiate it for the scheduler
from .ai_processor import AIProcessor
from .ha_client import HomeAssistantClient
from .config_loader import load_config
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
        self.bluetooth_processor = BluetoothProcessor()
        self.ai_processor = AIProcessor(config_path)

        # Initialize Home Assistant client
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

        # Get generation configuration
        self.generation_config = self.config.get('generation', {
            'distance_window_minutes': 15,
            'area_window_minutes': 10,
            'mds_dimensions': 2,
            'min_points_per_room': 3,
            'use_adjacency': True
        })

        self.status = {"state": "idle", "progress": 0}
        self.latest_job_id = None
        self.latest_generated_blueprint = None

        self._initialized = True

    def init_ai_processor(self) -> None:
        """Initialize or reinitialize the AI processor instance."""
        logger.info("Initializing AI processor...")
        self.ai_processor = AIProcessor()
        logger.info("AI processor initialized successfully")

    def generate_blueprint(self):
        """Generate blueprint using relative positioning and area anchoring."""
        logger.info("Starting blueprint generation (Relative Positioning Method)...")
        self.status = {"state": "processing", "progress": 0.0}
        job_start_time = datetime.now()

        try:
            # Step 1: Get recent distance readings
            distance_window = self.generation_config.get('distance_window_minutes', 15)
            distance_data = get_recent_distances(distance_window)
            logger.info(f"Retrieved {len(distance_data)} distance readings for processing")

            # Step 2: Get area predictions for anchoring
            area_window = self.generation_config.get('area_window_minutes', 10)
            area_predictions = get_recent_area_predictions(area_window)
            logger.info(f"Retrieved {len(area_predictions)} area predictions for anchoring")

            # Step 3: Calculate relative positions using MDS
            dimensions = self.generation_config.get('mds_dimensions', 2)
            relative_positions = self.ai_processor.run_relative_positioning(distance_data, dimensions)
            logger.info(f"Generated relative positions for {len(relative_positions)} devices")

            # Step 4: Group devices by area/room
            device_coords_by_area = {}
            for device_id, coords in relative_positions.items():
                area_id = area_predictions.get(device_id)
                if area_id:
                    if area_id not in device_coords_by_area:
                        device_coords_by_area[area_id] = []
                    device_coords_by_area[area_id].append({
                        'device_id': device_id,
                        'x': coords.get('x', 0),
                        'y': coords.get('y', 0),
                        'z': coords.get('z', 0)
                    })

            # Step 5: Generate rooms from points
            rooms = self.ai_processor.generate_rooms_from_points(device_coords_by_area)

            # Step 6: Generate walls between rooms
            walls = self._generate_walls(rooms)

            # Step 7: Group rooms into floors
            floors = self._group_rooms_into_floors(rooms)

            # Step 8: Create the final blueprint
            blueprint = {
                'version': '1.0',
                'generated_at': datetime.now().isoformat(),
                'rooms': rooms,
                'walls': walls,
                'floors': floors,
                'metadata': {
                    'room_count': len(rooms),
                    'device_count': len(relative_positions),
                    'generation_time_seconds': (datetime.now() - job_start_time).total_seconds()
                }
            }

            # Save the blueprint
            if self._save_blueprint(blueprint):
                self.latest_generated_blueprint = blueprint
                self.status = {"state": "completed", "progress": 1.0}
                logger.info("Blueprint generation completed successfully")
                return True
            else:
                logger.error("Failed to save blueprint")
                self.status = {"state": "failed", "progress": 0.0}
                return False

        except Exception as e:
            logger.error(f"Error generating blueprint: {str(e)}", exc_info=True)
            self.status = {"state": "failed", "progress": 0.0}
            return False

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

    def _generate_walls(self, rooms: List[Dict], positions: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict]:
        """Generate walls between rooms using AI prediction when available."""
        if not rooms:
            return []

        # Check if ML wall prediction is enabled
        use_ml = self.config.get('ai_settings', {}).get('use_ml_wall_prediction', True)

        if use_ml and positions:
            try:
                # Try to use the ML-based wall prediction
                walls = self.ai_processor.predict_walls(positions, rooms) if self.ai_processor else []
                if walls:
                    logger.debug(f"Using ML-based wall prediction: generated {len(walls)} walls")
                    return walls
            except Exception as e:
                logger.warning(f"ML-based wall prediction failed: {str(e)}")

        # Fall back to Delaunay triangulation
        logger.debug("Using Delaunay triangulation for wall generation")

        walls = []

        # Extract room vertices
        vertices = []
        for room in rooms:
            bounds = room['bounds']
            vertices.extend([
                [bounds['min']['x'], bounds['min']['y']],
                [bounds['min']['x'], bounds['max']['y']],
                [bounds['max']['x'], bounds['min']['y']],
                [bounds['max']['x'], bounds['max']['y']]
            ])

        if len(vertices) < 3 or len(rooms) < 2:
            return walls

        # Create Delaunay triangulation
        vertices = np.array(vertices)
        tri = Delaunay(vertices)

        # Extract edges
        edges = set()
        for simplex in tri.simplices:
            for i in range(3):
                edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                edges.add(edge)

        # Convert edges to walls
        for edge in edges:
            p1 = vertices[edge[0]]
            p2 = vertices[edge[1]]

            # Calculate wall properties
            length = np.linalg.norm(p2 - p1)
            angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

            # Skip walls that are too short or too long
            if length < self.validation['min_room_dimension'] or \
               length > self.validation['max_room_dimension']:
                continue

            walls.append({
                'start': {'x': float(p1[0]), 'y': float(p1[1])},
                'end': {'x': float(p2[0]), 'y': float(p2[1])},
                'thickness': self.validation['min_wall_thickness'],
                'height': self.validation['min_ceiling_height'],
                'angle': float(angle)
            })

        return walls

    def _generate_walls_between_rooms(self, rooms):
        """Generate walls between rooms using more realistic algorithms (fallback)."""
        walls = []

        try:
            # First identify rooms on the same floor
            rooms_by_floor = {}
            for room in rooms:
                floor = int(room['center']['z'] // 3)  # Assuming 3m floor height
                if floor not in rooms_by_floor:
                    rooms_by_floor[floor] = []
                rooms_by_floor[floor].append(room)

            # For each floor, generate walls between adjacent rooms
            for floor, floor_rooms in rooms_by_floor.items():
                logger.info(f"Generating walls for floor {floor} with {len(floor_rooms)} rooms")  # Add this line
                if len(floor_rooms) <= 1:
                    continue

                # Use Delaunay triangulation to find potential adjacent rooms
                from scipy.spatial import Delaunay
                import numpy as np

                # Extract room centers
                centers = []
                for room in floor_rooms:
                    centers.append([room['center']['x'], room['center']['y']])

                # Handle case with too few rooms
                if len(centers) < 3:
                    # Just connect them with a wall
                    if len(centers) == 2:
                        r1, r2 = floor_rooms[0], floor_rooms[1]
                        wall_height = min(r1['dimensions']['height'], r2['dimensions']['height'])
                        walls.append({
                            'start': {'x': r1['center']['x'], 'y': r1['center']['y']},
                            'end': {'x': r2['center']['x'], 'y': r2['center']['y']},
                            'height': wall_height,
                            'thickness': 0.2
                        })
                    continue

                # Create Delaunay triangulation
                tri = Delaunay(np.array(centers))

                # Generate walls from triangulation edges
                edges = set()
                for simplex in tri.simplices:
                    for i in range(3):
                        edge = tuple(sorted([simplex[i], simplex[(i+1)%3]]))
                        edges.add(edge)

                # Create walls from edges
                for i, j in edges:
                    r1, r2 = floor_rooms[i], floor_rooms[j]

                    # Check if rooms are too far apart
                    dx = r1['center']['x'] - r2['center']['x']
                    dy = r1['center']['y'] - r2['center']['y']
                    distance = (dx**2 + dy**2)**0.5

                    # Skip if too far apart
                    if distance > 10:  # Adjust threshold as needed
                        continue

                    # Create wall
                    wall_height = min(r1['dimensions']['height'], r2['dimensions']['height'])
                    walls.append({
                        'start': {'x': r1['center']['x'], 'y': r1['center']['y']},
                        'end': {'x': r2['center']['x'], 'y': r2['center']['y']},
                        'height': wall_height,
                        'thickness': 0.2
                    })
        except Exception as e:
            logger.error(f"Wall generation failed: {str(e)}")

        logger.info(f"Generated {len(walls)} walls between rooms")
        return walls

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

    def get_device_positions_from_db(self):
        """Get the latest device positions from the SQLite database."""
        try:
            # Use the helper function from db module
            device_positions = get_device_positions_from_sqlite()
            logger.info(f"Loaded {len(device_positions)} device positions from SQLite database")
            return device_positions
        except Exception as e:
            logger.error(f"Error getting device positions from database: {e}")
            return {}

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
    from .db import get_reference_positions_from_sqlite, save_reference_position

    # Check if we already have reference positions
    existing_refs = get_reference_positions_from_sqlite()
    if len(existing_refs) >= 3:
        logger.info(f"Found {len(existing_refs)} existing reference positions, no need to create more")
        return existing_refs

    logger.info("Insufficient reference positions found, creating initial reference points")

    # Create at least 3 reference points for the system to work with
    default_positions = {
        "reference_point_1": {"x": 0, "y": 0, "z": 0, "area_id": "lounge"},
        "reference_point_2": {"x": 5, "y": 0, "z": 0, "area_id": "kitchen"},
        "reference_point_3": {"x": 0, "y": 5, "z": 0, "area_id": "master_bedroom"},
        "reference_point_4": {"x": 5, "y": 5, "z": 0, "area_id": "office"}
    }

    # Save only the additional reference points we need
    for device_id, position in default_positions.items():
        if device_id not in existing_refs:
            save_reference_position(
                device_id=device_id,
                x=position['x'],
                y=position['y'],
                z=position['z'],
                area_id=position.get('area_id')
            )
            logger.info(f"Created reference position: {device_id} at ({position['x']}, {position['y']}, {position['z']})")

    # Return the updated reference positions
    return get_reference_positions_from_sqlite()
