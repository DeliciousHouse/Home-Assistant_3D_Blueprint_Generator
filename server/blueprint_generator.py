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
    execute_sqlite_query,          # Only if needed for other direct queries
    get_device_positions_from_sqlite, # Added to retrieve device positions from SQLite
    get_reference_positions_from_sqlite, # Added to retrieve reference positions
    save_reference_position         # Added to save reference positions
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

    def generate_blueprint(self) -> bool:
        """Generate a 3D blueprint of the environment."""
        try:
            logger.info("Starting blueprint generation process...")

            # Retrieve MDS positions from AI processor
            device_positions, anchor_positions = self.ai_processor.get_relative_positions()
            if not device_positions:
                logger.error("Failed to generate device positions")
                return False

            logger.info(f"Generated relative positions for {len(device_positions)} devices")

            # Process the positions and prepare for room generation
            device_coords_by_area = {}

            # Skip specific entities to exclude from the tracked devices list
            skip_entities = set()
            for entity_id, pos_data in anchor_positions.items():
                if pos_data.get('is_reference', False):
                    skip_entities.add(entity_id)

            # Process all device positions
            tracked_device_count = 0
            excluded_count = len(skip_entities)
            no_area_count = 0
            device_count_by_area = {}

            # Get recent area predictions for these devices
            from server.db import get_recent_area_predictions
            area_predictions = get_recent_area_predictions()
            logger.debug(f"Found {len(area_predictions)} recent area predictions")

            # Process device positions and assign them to areas
            for device_id, coords in device_positions.items():
                tracked_device_count += 1

                # Skip if this is in our exclude list
                if device_id in skip_entities:
                    continue

                # Get the area prediction for this device
                area_id = area_predictions.get(device_id)

                # If we have an area prediction, add the device to the area
                if area_id:
                    if area_id not in device_coords_by_area:
                        device_coords_by_area[area_id] = []
                        device_count_by_area[area_id] = 0

                    # Add the device coordinates to this area
                    device_data = {
                        'x': coords.get('x', 0),
                        'y': coords.get('y', 0),
                        'z': coords.get('z', 0),
                        'device_id': device_id
                    }
                    device_coords_by_area[area_id].append(device_data)
                    device_count_by_area[area_id] += 1
                else:
                    no_area_count += 1

            logger.info(f"Processed {tracked_device_count + excluded_count} total entities from MDS.")
            logger.info(f"  Skipped {excluded_count} scanners/reference points.")
            logger.info(f"  Considered {tracked_device_count} as potential tracked devices.")
            logger.info(f"  {no_area_count} tracked devices had no area prediction.")

            # Calculate devices with area predictions
            devices_with_area = tracked_device_count - no_area_count
            logger.info(f"  {devices_with_area} tracked devices were added to {len(device_coords_by_area)} areas.")

            # Fetch all Home Assistant areas to ensure we include even areas without devices
            try:
                # Use the already initialized HA client instead of creating a new one
                all_areas = self.ha_client.get_areas()
                logger.info(f"Retrieved {len(all_areas)} areas from Home Assistant")
            except Exception as e:
                logger.error(f"Error fetching Home Assistant areas: {str(e)}")
                all_areas = []  # Use empty list instead of None so the empty areas still get processed

            # Generate rooms from the grouped points
            rooms = self.ai_processor.generate_rooms_from_points(device_coords_by_area, all_areas)
            if not rooms:
                logger.error("Failed to generate rooms from points")
                return False

            # Generate walls between rooms
            walls = self.ai_processor.generate_walls(rooms)
            logger.info(f"Generated {len(walls)} walls")

            # Group rooms into floors
            floors = self._group_rooms_into_floors(rooms)
            logger.info(f"Grouped {len(rooms)} rooms into {len(floors)} floors")

            # Predict objects (furniture, fixtures, etc.) for each room
            objects = self.ai_processor.predict_objects(rooms)
            logger.info(f"Predicted {len(objects)} objects")

            # Build the complete blueprint
            blueprint = {
                'generated_at': datetime.now().isoformat(),
                'rooms': rooms,
                'walls': walls,
                'floors': floors,
                'objects': objects
            }

            # Save to database
            from server.db import save_blueprint_to_sqlite
            success = save_blueprint_to_sqlite(blueprint)
            if success:
                logger.info(f"Successfully saved blueprint with {len(rooms)} rooms")
            else:
                logger.error("Failed to save blueprint to database")

            logger.info("Blueprint generation completed successfully")
            return True

        except Exception as e:
            logger.error(f"Blueprint generation failed: {str(e)}", exc_info=True)
            return False

    def _generate_basic_walls(self, rooms: List[Dict]) -> List[Dict]:
        """Generate basic walls from room bounds when AI wall generation fails."""
        walls = []
        wall_id = 0

        for room in rooms:
            if not room.get('bounds'):
                continue

            bounds = room['bounds']
            min_x, min_y = bounds['min']['x'], bounds['min']['y']
            max_x, max_y = bounds['max']['x'], bounds['max']['y']

            wall_height = room.get('dimensions', {}).get('height', 2.4)
            wall_thickness = 0.15

            wall_id += 1
            walls.append({
                'id': f"wall_{wall_id}",
                'start': {'x': min_x, 'y': min_y},
                'end': {'x': min_x, 'y': max_y},
                'thickness': wall_thickness,
                'height': wall_height
            })

            wall_id += 1
            walls.append({
                'id': f"wall_{wall_id}",
                'start': {'x': min_x, 'y': max_y},
                'end': {'x': max_x, 'y': max_y},
                'thickness': wall_thickness,
                'height': wall_height
            })

            wall_id += 1
            walls.append({
                'id': f"wall_{wall_id}",
                'start': {'x': max_x, 'y': max_y},
                'end': {'x': max_x, 'y': min_y},
                'thickness': wall_thickness,
                'height': wall_height
            })

            wall_id += 1
            walls.append({
                'id': f"wall_{wall_id}",
                'start': {'x': max_x, 'y': min_y},
                'end': {'x': min_x, 'y': min_y},
                'thickness': wall_thickness,
                'height': wall_height
            })

        return walls

    def _group_rooms_into_floors(self, rooms: List[Dict]) -> List[Dict]:
        """Group rooms into floors based on their z-coordinate and area metadata."""
        if not rooms:
            return []

        floor_heights = {
            0: 0.0,
            1: 3.0,
            2: 6.0
        }

        area_floor_map = {
            "lounge": 0,
            "kitchen": 0,
            "dining_room": 0,
            "front_porch": 0,
            "laundry_room": 0,
            "master_bedroom": 0,
            "master_bathroom": 0,
            "office": 1,
            "dressing_room": 1,
            "sky_floor": 1,
            "balcony": 1,
            "garage": 0,
            "nova_room": 0,
            "christian_room": 0
        }

        floors_dict = {}
        for room in rooms:
            try:
                area_id = room.get('area_id')
                if area_id and area_id in area_floor_map:
                    floor_level = area_floor_map[area_id]
                else:
                    if 'center' in room and 'z' in room['center']:
                        center_z = room['center']['z']
                        if center_z < 1.5:
                            floor_level = 0
                        elif center_z < 4.5:
                            floor_level = 1
                        else:
                            floor_level = 2
                    else:
                        logger.warning(f"Room {room.get('id', 'unknown')} is missing 'center' data, defaulting to ground floor")
                        floor_level = 0

                if floor_level not in floors_dict:
                    floors_dict[floor_level] = []

                floors_dict[floor_level].append(room['id'])
                room['floor'] = floor_level

            except Exception as e:
                logger.error(f"Error processing room for floor grouping: {str(e)}")
                if 'id' in room:
                    if 0 not in floors_dict:
                        floors_dict[0] = []
                    floors_dict[0].append(room['id'])
                    room['floor'] = 0

        floors = []
        for level, room_ids in sorted(floors_dict.items()):
            floors.append({
                'level': level,
                'name': f"Floor {level}",
                'rooms': room_ids,
                'height': 3.0,
                'elevation': floor_heights.get(level, level * 3.0)
            })

        logger.info(f"Grouped {len(rooms)} rooms into {len(floors)} floors")
        return floors

    def get_latest_blueprint(self):
        """Get the latest blueprint from the database."""
        try:
            if hasattr(self, 'latest_generated_blueprint') and self.latest_generated_blueprint:
                return self.latest_generated_blueprint

            blueprint = get_latest_blueprint_from_sqlite()

            if blueprint:
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

            success = save_blueprint_to_sqlite(blueprint)

            if success:
                self.latest_generated_blueprint = blueprint
                logger.info(f"Successfully saved blueprint with {len(blueprint.get('rooms', []))} rooms")

            return success

        except Exception as e:
            logger.error(f"Failed to save blueprint: {e}")
            return False

    def get_device_positions_from_db(self):
        """Get the latest device positions from the SQLite database."""
        try:
            device_positions = get_device_positions_from_sqlite()
            logger.info(f"Loaded {len(device_positions)} device positions from SQLite database")
            return device_positions
        except Exception as e:
            logger.error(f"Error getting device positions from database: {e}")
            return {}

def ensure_reference_positions():
    """Make sure we have at least some reference positions in the SQLite database"""
    from .db import get_reference_positions_from_sqlite, save_reference_position

    existing_refs = get_reference_positions_from_sqlite()
    if len(existing_refs) >= 3:
        logger.info(f"Found {len(existing_refs)} existing reference positions, no need to create more")
        return existing_refs

    logger.info("Insufficient reference positions found, creating initial reference points")

    # Updated positions - master_bedroom now correctly positioned to the LEFT of kitchen
    default_positions = {
        "reference_point_1": {"x": 0, "y": 0, "z": 0, "area_id": "lounge"},
        "reference_point_2": {"x": 5, "y": 0, "z": 0, "area_id": "kitchen"},
        "reference_point_3": {"x": -5, "y": 0, "z": 0, "area_id": "master_bedroom"},  # Changed position to negative X
        "reference_point_4": {"x": 0, "y": 5, "z": 0, "area_id": "office"}
    }

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

    return get_reference_positions_from_sqlite()
