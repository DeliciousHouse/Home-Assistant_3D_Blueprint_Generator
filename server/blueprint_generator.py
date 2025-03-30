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
from .db import save_blueprint_to_sqlite, execute_sqlite_query, execute_write_query

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
        """Generate a 3D blueprint by processing sensor data and detecting rooms/walls."""
        try:
            logger.info("Starting blueprint generation process...")

            # 1. Process sensors to get positions and detect rooms
            #    This implicitly saves positions to DB via the processor
            logger.info("Processing Bluetooth sensors for latest data...")
            processing_result = self.bluetooth_processor.process_bluetooth_sensors()

            if not processing_result or "error" in processing_result:
                logger.error(f"Bluetooth processing failed: {processing_result.get('error', 'Unknown error')}")
                return None # Cannot proceed

            # Extract the results needed
            # Use .get() with defaults to avoid KeyErrors if processing partially fails
            device_positions = processing_result.get("device_positions", {})
            rooms = processing_result.get("rooms", [])
            logger.info(f"Processing complete. Found {len(device_positions)} positions, detected {len(rooms)} rooms.")

            # Check if we have enough data
            if not rooms:
                logger.warning("No rooms were detected. Cannot generate a blueprint.")
                return None # Stop if still no rooms


            # 2. Generate Walls
            logger.info(f"Generating walls for {len(rooms)} rooms...")
            walls = self._generate_walls_geometric(rooms) # Use the geometric method
            logger.info(f"Generated {len(walls)} wall segments.")

            # 3. Assemble Blueprint
            blueprint = {
                'version': '1.0',
                'generated_at': datetime.now().isoformat(),
                'rooms': rooms,
                'walls': walls,
                'floors': self._group_rooms_into_floors(rooms), # Ensure this uses the final 'rooms' list
                'metadata': {
                    'device_count': len(device_positions), # Count from processed data
                    'room_count': len(rooms),
                    'wall_count': len(walls),
                    'source': 'auto_generator_v2' # Indicate source
                }
            }

            # 4. (Optional) AI Refinement
            if self.config.get('ai_settings', {}).get('use_ml_blueprint_refinement', False): # Check config
                try:
                    logger.info("Applying AI blueprint refinement...")
                    refined_blueprint = self.ai_processor.refine_blueprint(blueprint)
                    if refined_blueprint:
                        # Important: Validate the *refined* blueprint
                        if self._validate_blueprint(refined_blueprint):
                            logger.info("AI refinement successful and validated.")
                            blueprint = refined_blueprint
                        else:
                            logger.warning("AI refined blueprint failed validation. Using original.")
                    else:
                         logger.warning("AI refinement returned None. Using original.")
                except Exception as e:
                    logger.warning(f"AI refinement failed with exception: {e}")

            # 5. Validate Final Blueprint
            logger.info("Validating final blueprint...")
            if not self._validate_blueprint(blueprint):
                logger.warning("Generated blueprint failed validation. Attempting minimal.")
                # Use the original 'rooms' list before potential refinement issues
                blueprint = self._create_minimal_valid_blueprint(rooms)
                if not self._validate_blueprint(blueprint): # Validate minimal too
                     logger.error("Minimal blueprint also failed validation. Cannot proceed.")
                     return None


            # 6. Save and Cache
            logger.info(f"Saving final blueprint with {len(blueprint.get('rooms',[]))} rooms and {len(blueprint.get('walls',[]))} walls.")
            saved = self._save_blueprint(blueprint) # Uses SQLite helper
            if saved:
                self.latest_generated_blueprint = blueprint # Update cache
            else:
                logger.error("Failed to save the generated blueprint to the database.")

            return blueprint

        except Exception as e:
            logger.error(f"Critical error during blueprint generation: {e}", exc_info=True)
            return None

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

    def _generate_walls_geometric(self, rooms: List[Dict]) -> List[Dict]:
        """Generate walls between rooms using geometric analysis."""
        if not rooms:
            return []

        walls = []
        logger.info(f"Generating walls for {len(rooms)} rooms using geometric method")

        # Threshold for considering rooms adjacent (in meters)
        adjacency_threshold = 0.5

        # Function to check if two rooms are adjacent
        def are_adjacent(room1, room2):
            # Get bounding boxes
            bounds1 = room1['bounds']
            bounds2 = room2['bounds']

            # Check if bounds overlap or are very close
            x_overlap = (
                bounds1['min']['x'] <= bounds2['max']['x'] + adjacency_threshold and
                bounds2['min']['x'] <= bounds1['max']['x'] + adjacency_threshold
            )

            y_overlap = (
                bounds1['min']['y'] <= bounds2['max']['y'] + adjacency_threshold and
                bounds2['min']['y'] <= bounds1['max']['y'] + adjacency_threshold
            )

            z_overlap = (
                bounds1['min']['z'] <= bounds2['max']['z'] + adjacency_threshold and
                bounds2['min']['z'] <= bounds1['max']['z'] + adjacency_threshold
            )

            # Rooms must overlap in at least 2 dimensions to be adjacent
            overlap_count = sum([x_overlap, y_overlap, z_overlap])
            return overlap_count >= 2

        # Function to find the overlapping segment between two rooms
        def find_overlapping_segment(room1, room2):
            bounds1 = room1['bounds']
            bounds2 = room2['bounds']

            # Determine which dimension has the smallest overlap or is closest
            x_distance = min(
                abs(bounds1['min']['x'] - bounds2['max']['x']),
                abs(bounds2['min']['x'] - bounds1['max']['x'])
            )

            y_distance = min(
                abs(bounds1['min']['y'] - bounds2['max']['y']),
                abs(bounds2['min']['y'] - bounds1['max']['y'])
            )

            # Default to vertical wall (along x-axis)
            is_vertical = x_distance <= y_distance

            # Find the overlapping segment
            if is_vertical:
                # Wall runs north-south
                if bounds1['min']['x'] <= bounds2['min']['x']:
                    # Room1 is west of Room2
                    wall_x = (bounds1['max']['x'] + bounds2['min']['x']) / 2
                else:
                    # Room1 is east of Room2
                    wall_x = (bounds1['min']['x'] + bounds2['max']['x']) / 2

                # Find y range of overlap
                start_y = max(bounds1['min']['y'], bounds2['min']['y'])
                end_y = min(bounds1['max']['y'], bounds2['max']['y'])

                # Ensure proper ordering
                if start_y > end_y:
                    start_y, end_y = end_y, start_y

                return {
                    'start': {'x': wall_x, 'y': start_y},
                    'end': {'x': wall_x, 'y': end_y},
                    'is_vertical': True
                }
            else:
                # Wall runs east-west
                if bounds1['min']['y'] <= bounds2['min']['y']:
                    # Room1 is south of Room2
                    wall_y = (bounds1['max']['y'] + bounds2['min']['y']) / 2
                else:
                    # Room1 is north of Room2
                    wall_y = (bounds1['min']['y'] + bounds2['max']['y']) / 2

                # Find x range of overlap
                start_x = max(bounds1['min']['x'], bounds2['min']['x'])
                end_x = min(bounds1['max']['x'], bounds2['max']['x'])

                # Ensure proper ordering
                if start_x > end_x:
                    start_x, end_x = end_x, start_x

                return {
                    'start': {'x': start_x, 'y': wall_y},
                    'end': {'x': end_x, 'y': wall_y},
                    'is_vertical': False
                }

        # Function to snap point to grid
        def snap_to_grid(x, grid_size=0.1):
            return round(x / grid_size) * grid_size

        # Function to check if two wall segments are collinear and can be merged
        def can_merge_walls(wall1, wall2):
            # Must have same orientation
            if wall1.get('is_vertical') != wall2.get('is_vertical'):
                return False

            if wall1.get('is_vertical'):
                # Vertical walls - must have same x coordinate and overlapping y range
                if abs(wall1['start']['x'] - wall2['start']['x']) > 0.1:
                    return False

                # Check if y ranges overlap
                y_min1, y_max1 = min(wall1['start']['y'], wall1['end']['y']), max(wall1['start']['y'], wall1['end']['y'])
                y_min2, y_max2 = min(wall2['start']['y'], wall2['end']['y']), max(wall2['start']['y'], wall2['end']['y'])

                return (y_min1 <= y_max2 + 0.1) and (y_min2 <= y_max1 + 0.1)
            else:
                # Horizontal walls - must have same y coordinate and overlapping x range
                if abs(wall1['start']['y'] - wall2['start']['y']) > 0.1:
                    return False

                # Check if x ranges overlap
                x_min1, x_max1 = min(wall1['start']['x'], wall1['end']['x']), max(wall1['start']['x'], wall1['end']['x'])
                x_min2, x_max2 = min(wall2['start']['x'], wall2['end']['x']), max(wall2['start']['x'], wall2['end']['x'])

                return (x_min1 <= x_max2 + 0.1) and (x_min2 <= x_max1 + 0.1)

        # Function to merge two collinear wall segments
        def merge_walls(wall1, wall2):
            if wall1.get('is_vertical'):
                # Merge vertical walls
                x = (wall1['start']['x'] + wall2['start']['x']) / 2  # Average x value
                y_min = min(wall1['start']['y'], wall1['end']['y'], wall2['start']['y'], wall2['end']['y'])
                y_max = max(wall1['start']['y'], wall1['end']['y'], wall2['start']['y'], wall2['end']['y'])

                merged = {
                    'start': {'x': x, 'y': y_min},
                    'end': {'x': x, 'y': y_max},
                    'is_vertical': True,
                    'thickness': max(wall1.get('thickness', 0.2), wall2.get('thickness', 0.2)),
                    'height': max(wall1.get('height', 2.5), wall2.get('height', 2.5))
                }
            else:
                # Merge horizontal walls
                y = (wall1['start']['y'] + wall2['start']['y']) / 2  # Average y value
                x_min = min(wall1['start']['x'], wall1['end']['x'], wall2['start']['x'], wall2['end']['x'])
                x_max = max(wall1['start']['x'], wall1['end']['x'], wall2['start']['x'], wall2['end']['x'])

                merged = {
                    'start': {'x': x_min, 'y': y},
                    'end': {'x': x_max, 'y': y},
                    'is_vertical': False,
                    'thickness': max(wall1.get('thickness', 0.2), wall2.get('thickness', 0.2)),
                    'height': max(wall1.get('height', 2.5), wall2.get('height', 2.5))
                }

            return merged

        # Create a list of adjacent room pairs
        adjacent_pairs = []
        for i in range(len(rooms)):
            for j in range(i+1, len(rooms)):
                if are_adjacent(rooms[i], rooms[j]):
                    adjacent_pairs.append((rooms[i], rooms[j]))

        logger.info(f"Found {len(adjacent_pairs)} adjacent room pairs")

        # Generate raw wall segments for each adjacent pair
        raw_walls = []
        for room1, room2 in adjacent_pairs:
            segment = find_overlapping_segment(room1, room2)
            if segment:
                # Snap to grid
                segment['start']['x'] = snap_to_grid(segment['start']['x'])
                segment['start']['y'] = snap_to_grid(segment['start']['y'])
                segment['end']['x'] = snap_to_grid(segment['end']['x'])
                segment['end']['y'] = snap_to_grid(segment['end']['y'])

                # Add wall properties
                thickness = self.validation['min_wall_thickness']

                # Get height based on rooms
                height = min(
                    room1['dimensions']['height'],
                    room2['dimensions']['height'],
                    self.validation['max_ceiling_height']
                )

                # Create wall from segment
                wall = {
                    'start': segment['start'],
                    'end': segment['end'],
                    'is_vertical': segment['is_vertical'],
                    'thickness': thickness,
                    'height': height
                }

                # Calculate angle
                dx = wall['end']['x'] - wall['start']['x']
                dy = wall['end']['y'] - wall['start']['y']
                angle = 0 if dx == 0 else math.atan2(dy, dx)
                wall['angle'] = angle

                # Only add walls with non-zero length
                length = math.sqrt(dx**2 + dy**2)
                if length > 0.1:
                    raw_walls.append(wall)

        # Merge collinear wall segments
        processed_walls = []
        raw_walls_copy = raw_walls.copy()

        while raw_walls_copy:
            current_wall = raw_walls_copy.pop(0)
            merged = False

            for i, other_wall in enumerate(raw_walls_copy):
                if can_merge_walls(current_wall, other_wall):
                    merged_wall = merge_walls(current_wall, other_wall)
                    raw_walls_copy[i] = merged_wall
                    merged = True
                    break

            if not merged:
                # Convert to the final wall format expected by the system
                final_wall = {
                    'start': current_wall['start'],
                    'end': current_wall['end'],
                    'thickness': current_wall.get('thickness', self.validation['min_wall_thickness']),
                    'height': current_wall.get('height', self.validation['min_ceiling_height']),
                    'angle': current_wall.get('angle', 0)
                }
                processed_walls.append(final_wall)

        logger.info(f"Generated {len(processed_walls)} walls after merging")
        return processed_walls

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
            from .db import get_latest_blueprint_from_sqlite
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
            from .db import save_blueprint_to_sqlite
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
            from .db import get_sqlite_connection

            # Get connection to SQLite
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Query for latest device positions
            query = """
            SELECT device_id, position_data, source, timestamp
            FROM device_positions
            ORDER BY timestamp DESC
            LIMIT 100
            """

            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()

            # Process results
            positions = {}
            seen_devices = set()

            for row in results:
                device_id = row[0]
                position_data = row[1]
                source = row[2]

                # Skip if we already have a position for this device
                if device_id in seen_devices:
                    continue

                seen_devices.add(device_id)

                # Parse position data
                try:
                    if position_data:
                        if isinstance(position_data, str):
                            position = json.loads(position_data)
                        else:
                            position = position_data

                        # Ensure required fields exist
                        if all(k in position for k in ['x', 'y', 'z']):
                            positions[device_id] = {
                                'x': float(position['x']),
                                'y': float(position['y']),
                                'z': float(position['z']),
                                'accuracy': float(position.get('accuracy', 1.0)),
                                'source': source or position.get('source', 'unknown'),
                                'area_id': position.get('area_id')
                            }
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing position data for {device_id}: {e}")

            logger.info(f"Loaded {len(positions)} device positions from SQLite database")
            return positions

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
