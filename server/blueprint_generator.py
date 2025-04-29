import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import math
import random

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
# Import HAClient with HomeAssistantClient alias and the get_ha_client function
from .ha_client import HAClient as HomeAssistantClient, get_ha_client
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
        """Generate a full 3D blueprint."""
        try:
            from .db import get_recent_distances, get_recent_area_predictions
            from .ai_processor import AIProcessor

            logger.info("Starting blueprint generation process...")

            # Initialize AI processor
            ai_processor = AIProcessor()

            # Fetch distance data
            distance_window = self.config.get('generation_settings', {}).get('distance_window_minutes', 15)
            distance_data = get_recent_distances(time_window_minutes=distance_window)

            if not distance_data:
                logger.error("No distance data available for blueprint generation")
                return False

            logger.info(f"Retrieved {len(distance_data)} distance records for blueprint generation")

            # Get area predictions
            area_window = self.config.get('generation_settings', {}).get('area_window_minutes', 10)
            area_predictions = get_recent_area_predictions(time_window_minutes=area_window)

            logger.info(f"Retrieved area predictions for {len(area_predictions)} devices")

            # Step 1: Calculate relative positions using MDS
            # Use the improved get_relative_positions method instead of run_relative_positioning
            device_positions, scanner_positions = ai_processor.get_relative_positions()

            if not device_positions and not scanner_positions:
                logger.error("Failed to calculate relative positions")
                return False

            logger.info(f"Calculated relative positions for {len(device_positions)} devices and {len(scanner_positions)} scanners")

            # Step 2: Get area definitions
            try:
                from .ha_client import get_ha_client
                ha_client = get_ha_client()
                areas = ha_client.get_areas()
                logger.info(f"Retrieved {len(areas)} areas from Home Assistant")
            except Exception as e:
                logger.error(f"Error retrieving areas: {e}")
                # Create some default areas
                areas = [
                    {"area_id": "living_room", "name": "Living Room"},
                    {"area_id": "kitchen", "name": "Kitchen"},
                    {"area_id": "bedroom", "name": "Bedroom"},
                    {"area_id": "bathroom", "name": "Bathroom"},
                    {"area_id": "office", "name": "Office"}
                ]
                logger.warning(f"Using {len(areas)} default areas")

            # Step 3: Generate target layout
            target_layout = self._generate_target_layout(areas)
            logger.info("Generated target layout for areas")

            # Step 4: Group devices by area
            device_area_groups = self._group_devices_by_area(device_positions, area_predictions, areas)
            logger.info(f"Grouped devices into {len(device_area_groups)} areas")

            # Step 5: Calculate centroids for each area group
            area_centroids = self._calculate_area_centroids(device_area_groups)
            logger.info(f"Calculated centroids for {len(area_centroids)} areas")

            # Step 6: Calculate transformation using Procrustes analysis
            transform_params = self._calculate_transformation(area_centroids, target_layout)
            if not transform_params:
                logger.error("Failed to calculate transformation")
                return False

            logger.info("Calculated spatial transformation parameters")

            # Step 7: Transform all positions (devices and scanners)
            transformed_positions = self._apply_transformation(
                {**device_positions, **scanner_positions},
                transform_params
            )
            logger.info(f"Transformed {len(transformed_positions)} positions")

            # Step 8: Generate room geometries
            rooms = self._generate_rooms(transformed_positions, device_area_groups, areas)
            logger.info(f"Generated {len(rooms)} room geometries")

            # Step 9: Infer walls between rooms
            walls = self._infer_walls(rooms)
            logger.info(f"Inferred {len(walls)} walls")

            # Step 10: Predict objects for each room
            objects = ai_processor.predict_objects(rooms)
            logger.info(f"Predicted {len(objects)} objects")

            # Step 11: Assemble and save the blueprint
            blueprint = {
                'generated_at': datetime.now().isoformat(),
                'rooms': rooms,
                'walls': walls,
                'objects': objects,
                'positions': transformed_positions,
                'floors': self._determine_floors(rooms)
            }

            # Save blueprint to database
            from .db import save_blueprint_to_sqlite
            success = save_blueprint_to_sqlite(blueprint)

            if success:
                logger.info("Blueprint successfully generated and saved")
                return True
            else:
                logger.error("Failed to save blueprint")
                return False

        except Exception as e:
            logger.error(f"Error generating blueprint: {str(e)}", exc_info=True)
            return False

    def _determine_rooms(self, areas: List[Dict], device_coords_by_area: Dict) -> List[Dict]:
        """Determine the rooms and their shapes based on Area and sensor data."""
        rooms = []
        for area_id, devices in device_coords_by_area.items():
            area_name = next((area['name'] for area in areas if area['area_id'] == area_id), f"Room {area_id}")
            avg_x = sum(device['x'] for device in devices) / len(devices)
            avg_y = sum(device['y'] for device in devices) / len(devices)
            avg_z = sum(device['z'] for device in devices) / len(devices)
            room = {
                'id': f"room_{area_id}",
                'name': area_name,
                'center': {'x': avg_x, 'y': avg_y, 'z': avg_z},
                'devices': devices
            }
            rooms.append(room)
        return rooms

    def _generate_target_layout(self, areas: List[Dict]) -> Dict:
        """Generate a target layout for area centroids.

        This creates an idealized arrangement of rooms based on common floor plan designs.
        """
        logger.info(f"Generating target layout for {len(areas)} areas")

        # Create a map of area_id -> common room positions
        # Values are normalized coordinates (x, y) to create a sensible default layout
        common_room_positions = {
            "living_room": (0, 0),       # Center
            "lounge": (0, 0),            # Same as living room
            "kitchen": (5, 0),           # To the right of living room
            "dining_room": (2.5, 0),     # Between living room and kitchen
            "master_bedroom": (-5, 0),   # To the left of living room
            "bedroom": (-5, 3),          # Up and left from living room
            "bathroom": (-3, 4),         # Up and slightly left from living room
            "office": (0, 5),            # Above living room
            "hallway": (0, 2),           # Connecting living room to other rooms
            "entrance": (0, -4),         # Below living room
            "garage": (-4, -4),          # Down and left from living room
            "laundry": (4, 3),           # Up and right from living room
            "basement": (0, -6),         # Far below living room
            "attic": (0, 7),             # Far above living room
            "guest_room": (4, -3),       # Down and right from living room
            "garden": (7, 0),            # Far right from living room
            "balcony": (7, 3),           # Up and far right
            "terrace": (0, -7)           # Far below living room
        }

        target_layout = {}

        # Assign positions based on area_id or name matching, with fallback to random positions
        for i, area in enumerate(areas):
            area_id = area.get('area_id', '').lower()
            area_name = area.get('name', '').lower().replace(' ', '_')

            # Try to match by area_id first, then by name
            if area_id in common_room_positions:
                target_layout[area_id] = common_room_positions[area_id]
            elif area_name in common_room_positions:
                target_layout[area_id] = common_room_positions[area_name]
            else:
                # For unrecognized areas, place in a circle around the center
                angle = (i * 2 * math.pi) / len(areas)
                radius = 4.0  # Distance from center
                x = radius * math.cos(angle)
                y = radius * math.sin(angle)
                target_layout[area_id] = (x, y)
                logger.info(f"Generated position for unmapped area '{area_id}': ({x:.2f}, {y:.2f})")

        return target_layout

    def _group_devices_by_area(self, device_positions: Dict, area_predictions: Dict, areas: List[Dict]) -> Dict:
        """Group device coordinates by their predicted area."""
        device_area_groups = {}

        for device_id, position in device_positions.items():
            # Get predicted area for this device
            area_id = area_predictions.get(device_id)

            # If no area prediction, use a default
            if not area_id:
                area_id = "unknown"

            # Initialize the area group if it doesn't exist
            if area_id not in device_area_groups:
                device_area_groups[area_id] = []

            # Add device with its position to the area group
            device_area_groups[area_id].append({
                'device_id': device_id,
                'x': position[0],  # X coordinate from position tuple
                'y': position[1],  # Y coordinate from position tuple
                'z': 0 if len(position) < 3 else position[2]  # Z coordinate if available
            })

        return device_area_groups

    def _calculate_area_centroids(self, device_area_groups: Dict) -> Dict:
        """Calculate the centroid of each device group by area."""
        area_centroids = {}

        for area_id, devices in device_area_groups.items():
            if not devices:
                continue

            # Calculate average position
            x_coords = [d['x'] for d in devices]
            y_coords = [d['y'] for d in devices]

            # Centroid is the average position
            avg_x = sum(x_coords) / len(x_coords)
            avg_y = sum(y_coords) / len(y_coords)

            area_centroids[area_id] = (avg_x, avg_y)

        return area_centroids

    def _calculate_transformation(self, source_centroids: Dict, target_layout: Dict) -> Dict:
        """Calculate transformation from source to target using Procrustes analysis."""
        # Filter for areas that exist in both source and target
        common_areas = set(source_centroids.keys()) & set(target_layout.keys())

        if len(common_areas) < 2:
            logger.error(f"Not enough common areas for transformation: {len(common_areas)}")
            return None

        # Extract matching points
        source_points = np.array([source_centroids[area] for area in common_areas])
        target_points = np.array([target_layout[area] for area in common_areas])

        # Calculate means
        source_mean = source_points.mean(axis=0)
        target_mean = target_points.mean(axis=0)

        # Center the points
        source_centered = source_points - source_mean
        target_centered = target_points - target_mean

        # Calculate optimal rotation
        covariance = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(covariance)
        rotation = U @ Vt

        # Check for reflection
        if np.linalg.det(rotation) < 0:
            # Handle reflection if needed
            V = Vt.T
            V[:, -1] = -V[:, -1]
            rotation = U @ V.T

        # Calculate scaling
        source_var = np.sum(source_centered ** 2)
        if source_var == 0:
            scale = 1.0
        else:
            scale = np.sqrt(np.sum(target_centered ** 2) / source_var)

        # Calculate translation
        translation = target_mean - scale * source_mean @ rotation

        return {
            'rotation': rotation,
            'scale': scale,
            'translation': translation,
            'source_mean': source_mean,
            'target_mean': target_mean
        }

    def _apply_transformation(self, positions: Dict, transform_params: Dict) -> Dict:
        """Apply the calculated transformation to all positions."""
        if not transform_params:
            return positions

        rotation = transform_params['rotation']
        scale = transform_params['scale']
        translation = transform_params['translation']

        transformed_positions = {}

        for entity_id, position in positions.items():
            # Convert position to numpy array
            if isinstance(position, tuple):
                pos_array = np.array(position)
            else:
                pos_array = np.array([position[0], position[1]])

            # Apply transformation: scale, rotate, translate
            transformed = scale * pos_array @ rotation + translation

            # Convert back to the original format
            if isinstance(position, tuple):
                if len(position) > 2:  # If we had z-coordinate
                    transformed_positions[entity_id] = (float(transformed[0]), float(transformed[1]), position[2])
                else:
                    transformed_positions[entity_id] = (float(transformed[0]), float(transformed[1]))
            else:
                if len(position) > 2:  # If we had z-coordinate
                    transformed_positions[entity_id] = [float(transformed[0]), float(transformed[1]), position[2]]
                else:
                    transformed_positions[entity_id] = [float(transformed[0]), float(transformed[1])]

        return transformed_positions

    def _generate_rooms(self, transformed_positions: Dict, device_area_groups: Dict, areas: List[Dict]) -> List[Dict]:
        """Generate room geometries from transformed positions."""
        rooms = []

        # Create a mapping of area_id to area name
        area_names = {area['area_id']: area.get('name', area['area_id']) for area in areas}

        # Process each area group
        for area_id, devices in device_area_groups.items():
            if not devices or area_id == "unknown":
                continue

            # Extract device positions for this area
            area_device_ids = [device['device_id'] for device in devices]
            area_positions = [transformed_positions.get(device_id) for device_id in area_device_ids if device_id in transformed_positions]

            if not area_positions:
                continue

            # Convert to numpy array for easier processing
            points = np.array(area_positions)

            # Calculate room bounds
            if len(points) >= 3:
                # Use convex hull or alpha shape for more complex shapes
                try:
                    hull = Delaunay(points[:,:2]).convex_hull
                    # Extract unique vertices from convex hull
                    vertices = np.unique(hull.flatten())
                    # Extract coordinates of hull vertices
                    hull_points = points[vertices]

                    # Calculate bounds from hull points
                    min_x = float(np.min(hull_points[:,0]))
                    max_x = float(np.max(hull_points[:,0]))
                    min_y = float(np.min(hull_points[:,1]))
                    max_y = float(np.max(hull_points[:,1]))
                except:
                    # Fallback to simple min/max if convex hull fails
                    min_x = float(np.min(points[:,0]))
                    max_x = float(np.max(points[:,0]))
                    min_y = float(np.min(points[:,1]))
                    max_y = float(np.max(points[:,1]))
            else:
                # Simple bounds for few points
                min_x = float(np.min(points[:,0]))
                max_x = float(np.max(points[:,0]))
                min_y = float(np.min(points[:,1]))
                max_y = float(np.max(points[:,1]))

            # Add some padding around the bounds
            padding = 0.5  # 0.5m padding
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding

            # Calculate room dimensions
            width = max_x - min_x
            length = max_y - min_y
            height = 2.5  # Default ceiling height

            # Calculate room center
            center_x = (min_x + max_x) / 2
            center_y = (min_y + max_y) / 2
            center_z = 0  # Default floor level

            # Create room entry
            room = {
                'id': f"room_{area_id}",
                'name': area_names.get(area_id, area_id.replace('_', ' ').title()),
                'type': area_id,
                'bounds': {
                    'min': {'x': min_x, 'y': min_y, 'z': 0},
                    'max': {'x': max_x, 'y': max_y, 'z': height}
                },
                'center': {'x': center_x, 'y': center_y, 'z': center_z},
                'dimensions': {
                    'width': width,
                    'length': length,
                    'height': height,
                    'area': width * length
                },
                'devices': [{"id": device_id} for device_id in area_device_ids],
                'floor': 0  # Default to ground floor
            }

            rooms.append(room)

        return rooms

    def _infer_walls(self, rooms: List[Dict]) -> List[Dict]:
        """Infer walls between rooms based on proximity."""
        walls = []
        wall_id = 1

        # First, create walls for room perimeters
        for room in rooms:
            bounds = room['bounds']
            height = room['dimensions']['height']
            floor = room.get('floor', 0)

            # Create four walls for rectangular room
            walls.extend([
                {
                    'id': f"wall_{wall_id}",
                    'start': {'x': bounds['min']['x'], 'y': bounds['min']['y']},
                    'end': {'x': bounds['max']['x'], 'y': bounds['min']['y']},
                    'height': height,
                    'thickness': 0.15,
                    'floor': floor
                },
                {
                    'id': f"wall_{wall_id + 1}",
                    'start': {'x': bounds['max']['x'], 'y': bounds['min']['y']},
                    'end': {'x': bounds['max']['x'], 'y': bounds['max']['y']},
                    'height': height,
                    'thickness': 0.15,
                    'floor': floor
                },
                {
                    'id': f"wall_{wall_id + 2}",
                    'start': {'x': bounds['max']['x'], 'y': bounds['max']['y']},
                    'end': {'x': bounds['min']['x'], 'y': bounds['max']['y']},
                    'height': height,
                    'thickness': 0.15,
                    'floor': floor
                },
                {
                    'id': f"wall_{wall_id + 3}",
                    'start': {'x': bounds['min']['x'], 'y': bounds['max']['y']},
                    'end': {'x': bounds['min']['x'], 'y': bounds['min']['y']},
                    'height': height,
                    'thickness': 0.15,
                    'floor': floor
                }
            ])
            wall_id += 4

        # TODO: Optimize walls by removing overlapping ones and creating doorways

        return walls

    def _determine_floors(self, rooms: List[Dict]) -> List[Dict]:
        """Determine floor levels from room positions."""
        # Group rooms by floor levels
        floors = {}
        for room in rooms:
            floor_level = room.get('floor', 0)
            if floor_level not in floors:
                floors[floor_level] = []
            floors[floor_level].append(room['id'])

        # Create floor objects
        floor_objects = []
        for level, room_ids in sorted(floors.items()):
            floor_name = "Ground Floor" if level == 0 else f"{self._get_ordinal_suffix(level)} Floor"
            floor_objects.append({
                'level': level,
                'name': floor_name,
                'room_ids': room_ids
            })

        return floor_objects

    def _calculate_room_dimensions(self, rooms: List[Dict]) -> List[Dict]:
        """Calculate reasonable room dimensions based on their positions."""
        for room in rooms:
            # Default dimensions if no better calculation is available
            width = 5.0
            length = 5.0
            height = 2.7

            # Better approach if we have devices
            if 'devices' in room and len(room['devices']) >= 2:
                # Get x and y coordinates of devices in this room
                x_coords = [device['x'] for device in room['devices']]
                y_coords = [device['y'] for device in room['devices']]
                z_coords = [device.get('z', 0) for device in room['devices']]

                # Calculate width and length based on device spread, add margin
                margin = 1.0  # Add 1m margin around detected points
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                min_z, max_z = min(z_coords), max(z_coords)

                width = max(2.0, max_x - min_x + margin * 2)  # At least 2 meters wide
                length = max(2.0, max_y - min_y + margin * 2)  # At least 2 meters long

                # Determine floor number based on z-coordinate
                floor_level = int(sum(z_coords) / len(z_coords) // 3)  # 3m per floor

                # Ensure room has floor info
                room['floor'] = floor_level

                # Calculate bounds for the room
                room['bounds'] = {
                    'min': {'x': min_x - margin, 'y': min_y - margin, 'z': min_z},
                    'max': {'x': max_x + margin, 'y': max_y + margin, 'z': max_z}
                }

                # Make sure room has center coordinates
                room['center'] = {
                    'x': (min_x + max_x) / 2,
                    'y': (min_y + max_y) / 2,
                    'z': (min_z + max_z) / 2
                }
            else:
                # For rooms without enough devices, create default bounds
                center = room.get('center', {'x': 0, 'y': 0, 'z': 0})
                floor_level = int(center['z'] // 3)  # 3m per floor
                room['floor'] = floor_level

                room['bounds'] = {
                    'min': {'x': center['x'] - width/2, 'y': center['y'] - length/2, 'z': 0},
                    'max': {'x': center['x'] + width/2, 'y': center['y'] + length/2, 'z': height}
                }

            room['dimensions'] = {
                'width': width,
                'length': length,
                'height': height,
                'area': width * length
            }

            # Ensure every room has a name
            if not room.get('name'):
                room['name'] = room.get('id', 'Unknown Room').replace('_', ' ').title()

        return rooms

    def _generate_walls(self, rooms: List[Dict]) -> List[Dict]:
        """Generate walls between rooms based on room positions and dimensions."""
        walls = []
        for room in rooms:
            center = room['center']
            dimensions = room['dimensions']
            floor_level = room.get('floor', 0)  # Get floor level from room

            # Generate 4 walls for room (basic rectangle)
            walls.append({
                'id': f"wall_{room['id']}_1",
                'start': {'x': center['x'] - dimensions['width'] / 2, 'y': center['y'] - dimensions['length'] / 2},
                'end': {'x': center['x'] + dimensions['width'] / 2, 'y': center['y'] - dimensions['length'] / 2},
                'height': dimensions['height'],
                'thickness': 0.15,
                'floor': floor_level  # Add floor level to wall
            })
            walls.append({
                'id': f"wall_{room['id']}_2",
                'start': {'x': center['x'] + dimensions['width'] / 2, 'y': center['y'] - dimensions['length'] / 2},
                'end': {'x': center['x'] + dimensions['width'] / 2, 'y': center['y'] + dimensions['length'] / 2},
                'height': dimensions['height'],
                'thickness': 0.15,
                'floor': floor_level
            })
            walls.append({
                'id': f"wall_{room['id']}_3",
                'start': {'x': center['x'] + dimensions['width'] / 2, 'y': center['y'] + dimensions['length'] / 2},
                'end': {'x': center['x'] - dimensions['width'] / 2, 'y': center['y'] + dimensions['length'] / 2},
                'height': dimensions['height'],
                'thickness': 0.15,
                'floor': floor_level
            })
            walls.append({
                'id': f"wall_{room['id']}_4",
                'start': {'x': center['x'] - dimensions['width'] / 2, 'y': center['y'] + dimensions['length'] / 2},
                'end': {'x': center['x'] - dimensions['width'] / 2, 'y': center['y'] - dimensions['length'] / 2},
                'height': dimensions['height'],
                'thickness': 0.15,
                'floor': floor_level
            })
        return walls

    def _organize_floors(self, rooms: List[Dict]) -> List[Dict]:
        """Organize rooms into floor levels."""
        floors = {}
        for room in rooms:
            # Get the floor level from the room data
            floor_level = room.get('floor', 0)

            if floor_level not in floors:
                floors[floor_level] = []

            floors[floor_level].append(room['id'])  # Just store room ID references

            # Make sure room has its floor information
            room['floor'] = floor_level

        # Create floor objects with names and room references
        floor_objects = []
        for level, room_ids in sorted(floors.items()):
            floor_name = "Ground Floor" if level == 0 else f"{level}{self._get_ordinal_suffix(level)} Floor"

            floor_objects.append({
                'level': level,
                'name': floor_name,
                'room_ids': room_ids
            })

        logger.info(f"Organized rooms into {len(floor_objects)} floors: {', '.join(f['name'] for f in floor_objects)}")
        return floor_objects

    def _get_ordinal_suffix(self, n):
        """Return the ordinal suffix for a number (1st, 2nd, 3rd, etc.)"""
        if 11 <= (n % 100) <= 13:
            return 'th'
        else:
            return {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')

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
