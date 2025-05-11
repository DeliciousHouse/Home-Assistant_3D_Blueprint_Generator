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
# Import new AI Image Generation modules
from .room_description_generator import description_generator
from .ai_image_generator import image_generator
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
# Import for static device detection
from .static_device_detector import detector as static_detector
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
        """Generate a full 3D blueprint with improved error handling and status updates."""
        try:
            from .db import get_recent_distances, get_recent_area_predictions
            from .ai_processor import AIProcessor

            logger.info("Starting blueprint generation process...")
            self.status = {"state": "generating", "progress": 0}

            # Initialize AI processor
            ai_processor = AIProcessor()

            # STEP 1: Fetch distance data with validation
            distance_window = self.config.get('generation_settings', {}).get('distance_window_minutes', 15)
            distance_data = get_recent_distances(time_window_minutes=distance_window)

            if not distance_data:
                logger.error("No distance data available for blueprint generation")
                self.status = {"state": "failed", "reason": "no_distance_data", "progress": 0}
                return False

            # Validate minimum data quality
            if len(distance_data) < 10:  # Require at least 10 distance readings
                logger.warning(f"Insufficient distance data: only {len(distance_data)} records (minimum 10 recommended)")
                # Continue but update status to warn user
                self.status["warning"] = "limited_data"

            logger.info(f"Retrieved {len(distance_data)} distance records for blueprint generation")
            self.status["progress"] = 10

            # STEP 2: Get area predictions with proper error handling
            area_window = self.config.get('generation_settings', {}).get('area_window_minutes', 10)
            try:
                area_predictions = get_recent_area_predictions(time_window_minutes=area_window)
                logger.info(f"Retrieved area predictions for {len(area_predictions)} devices")
            except Exception as e:
                logger.error(f"Error retrieving area predictions: {e}")
                area_predictions = {}  # Use empty dict as fallback
                self.status["warning"] = "area_prediction_error"

            self.status["progress"] = 20

            # STEP 3: Calculate relative positions with robust validation
            try:
                device_positions, scanner_positions = ai_processor.get_relative_positions()

                # Validate position results
                if not device_positions and not scanner_positions:
                    logger.error("Both device and scanner positions are empty - using fallback positions")
                    # Try to use fallback positioning
                    device_positions, scanner_positions = ai_processor._generate_fallback_positions()
                    if not device_positions:
                        logger.error("Fallback positioning also failed")
                        self.status = {"state": "failed", "reason": "position_calculation_failed", "progress": 30}
                        return False

                # Validate minimum entity requirement
                total_positioned_entities = len(device_positions) + len(scanner_positions)
                if total_positioned_entities < 3:
                    logger.error(f"Insufficient positioned entities ({len(device_positions)} devices, {len(scanner_positions)} scanners)")
                    self.status = {"state": "failed", "reason": "insufficient_entities", "progress": 30}
                    return False

                logger.info(f"Calculated relative positions for {len(device_positions)} devices and {len(scanner_positions)} scanners")
            except Exception as e:
                logger.error(f"Error calculating positions: {str(e)}", exc_info=True)
                self.status = {"state": "failed", "reason": "position_calculation_failed", "message": str(e), "progress": 30}
                return False

            self.status["progress"] = 40

            # STEP 4: Get area definitions with error handling
            try:
                from .ha_client import get_ha_client
                ha_client = get_ha_client()
                areas = ha_client.get_areas() or []
                logger.info(f"Retrieved {len(areas)} areas from Home Assistant")

                # Check for sufficient areas
                if not areas:
                    logger.warning("No areas retrieved from Home Assistant, using defaults")
                    # Create default areas
                    areas = [
                        {"area_id": "living_room", "name": "Living Room"},
                        {"area_id": "kitchen", "name": "Kitchen"},
                        {"area_id": "bedroom", "name": "Bedroom"},
                        {"area_id": "bathroom", "name": "Bathroom"},
                        {"area_id": "office", "name": "Office"}
                    ]
                    self.status["warning"] = "using_default_areas"
            except Exception as e:
                logger.error(f"Error retrieving areas: {e}", exc_info=True)
                # Create default areas as fallback
                areas = [
                    {"area_id": "living_room", "name": "Living Room"},
                    {"area_id": "kitchen", "name": "Kitchen"},
                    {"area_id": "bedroom", "name": "Bedroom"},
                    {"area_id": "bathroom", "name": "Bathroom"},
                    {"area_id": "office", "name": "Office"}
                ]
                self.status["warning"] = "using_default_areas"

            self.status["progress"] = 50

            # STEP 5-7: Generate target layout and transform positions
            try:
                # Generate target layout
                target_layout = self._generate_target_layout(areas)
                logger.info("Generated target layout for areas")

                # Get RSSI data
                rssi_data = ai_processor.get_rssi_data()

                # Group devices by area
                device_area_groups = self._extract_device_area_mappings(device_positions)
                logger.info(f"Grouped devices into {len(device_area_groups)} areas")

                # Calculate area centroids
                area_centroids = self._calculate_area_centroids(device_area_groups)
                logger.info(f"Calculated centroids for {len(area_centroids)} areas")

                # Calculate transformation
                transform_params = self._calculate_transformation(area_centroids, target_layout)
                if not transform_params:
                    logger.error("Failed to calculate transformation")
                    self.status = {"state": "failed", "reason": "transformation_failed", "progress": 60}
                    return False

                # Apply transformation
                transformed_positions = self._apply_transformation(
                    {**device_positions, **scanner_positions},
                    transform_params
                )
                logger.info(f"Transformed {len(transformed_positions)} positions")
            except Exception as e:
                logger.error(f"Error in layout generation or transformation: {str(e)}", exc_info=True)
                self.status = {"state": "failed", "reason": "layout_transformation_failed", "message": str(e), "progress": 60}
                return False

            self.status["progress"] = 70

            # STEP 8-9: Generate room geometries and walls
            try:
                # Generate rooms
                rooms = self._generate_rooms(device_area_groups, transformed_positions)

                # Check if rooms were generated successfully
                if not rooms:
                    logger.error("Failed to generate room geometries - no rooms created")
                    self.status = {"state": "failed", "reason": "no_rooms_created", "progress": 80}
                    return False

                logger.info(f"Generated {len(rooms)} room geometries")

                # Infer walls
                walls = self._infer_walls(rooms)
                logger.info(f"Inferred {len(walls)} walls")
            except Exception as e:
                logger.error(f"Error generating rooms or walls: {str(e)}", exc_info=True)
                self.status = {"state": "failed", "reason": "room_generation_failed", "message": str(e), "progress": 80}
                return False

            self.status["progress"] = 85

            # STEP 10-11: Predict objects and assemble blueprint
            try:
                # Predict objects
                objects = ai_processor.predict_objects(rooms)
                logger.info(f"Predicted {len(objects)} objects")

                # Determine floors
                floors = self._determine_floors(rooms)

                # Assemble blueprint
                blueprint = {
                    'generated_at': datetime.now().isoformat(),
                    'rooms': rooms,
                    'walls': walls,
                    'objects': objects,
                    'positions': transformed_positions,
                    'floors': floors
                }

                self.status["progress"] = 90

                # STEP 12: Generate AI room images if enabled
                ai_image_config = self.config.get('ai_image_generation', {})
                if ai_image_config.get('enabled', False):
                    try:
                        logger.info("AI image generation is enabled, generating images...")
                        self.status["progress"] = 91
                        self.status["message"] = "Generating AI room images..."

                        # Create an organized structure for floors
                        floor_data_by_number = {}
                        for room in rooms:
                            floor_num = room.get('floor', 0)
                            if floor_num not in floor_data_by_number:
                                floor_data_by_number[floor_num] = {
                                    'floor_number': floor_num,
                                    'name': f"Floor {floor_num}" if floor_num > 0 else ("Ground Floor" if floor_num == 0 else f"Basement {abs(floor_num)}"),
                                    'rooms': []
                                }
                            floor_data_by_number[floor_num]['rooms'].append(room)

                        # Check if AI image generation is enabled
                        ai_image_config = self.config.get('ai_image_generation', {})
                        ai_images_enabled = ai_image_config.get('enabled', False)

                        if ai_images_enabled:
                            # Generate images for each room
                            room_count = len(rooms)
                            for i, room in enumerate(rooms):
                                room_name = room.get('name', f"Room {i+1}")
                                if i % 5 == 0:  # Update progress periodically
                                    self.status["progress"] = 91 + int((i / room_count) * 4)
                                    self.status["message"] = f"Generating room image {i+1}/{room_count}..."

                                try:
                                    # Get style preset from config
                                    style = self.config.get('room_description', {}).get('default_style', 'modern')
                                    room_image_path = image_generator.generate_room_image(room, style)
                                    if room_image_path:
                                        room['room_image'] = room_image_path
                                        logger.info(f"Added AI-generated image for {room_name}")
                                except Exception as e:
                                    logger.error(f"Failed to generate image for room {room_name}: {str(e)}")

                            logger.info("Completed room image generation")
                        else:
                            logger.info("AI image generation is disabled, skipping room image generation")

                        # Generate floor plan images if AI image generation is enabled
                        self.status["progress"] = 95
                        self.status["message"] = "Generating floor plan images..."

                        if ai_images_enabled:
                            for floor_num, floor_data in floor_data_by_number.items():
                                try:
                                    # Get style preset from config
                                    style = self.config.get('room_description', {}).get('default_style', 'modern')
                                    floor_image_path = image_generator.generate_floor_plan(floor_data, style)
                                    if floor_image_path:
                                        # Find the corresponding floor in the blueprint
                                        for floor in floors:
                                            if floor.get('floor_number') == floor_num:
                                                floor['floor_plan_image'] = floor_image_path
                                                break
                                        logger.info(f"Added AI-generated floor plan for floor {floor_num}")
                                except Exception as e:
                                    logger.error(f"Failed to generate floor plan for floor {floor_num}: {str(e)}")

                            # Generate home exterior image
                            self.status["progress"] = 97
                            self.status["message"] = "Generating exterior view..."
                            try:
                                # Get style preset from config
                                style = self.config.get('room_description', {}).get('default_style', 'modern')
                                exterior_image_path = image_generator.generate_home_exterior(blueprint, style)
                                if exterior_image_path:
                                    blueprint['exterior_image'] = exterior_image_path
                                    logger.info("Added AI-generated exterior view image")
                            except Exception as e:
                                logger.error(f"Failed to generate exterior view image: {str(e)}")

                        # Complete the blueprint generation
                        self.status["progress"] = 98
                        self.status["message"] = "Finalizing blueprint..."

                    except Exception as e:
                        logger.error(f"Error during AI image generation: {str(e)}")
                        # Continue with saving blueprint even if image generation fails

                self.status["progress"] = 98
                self.status["message"] = "Saving blueprint..."

                # Save blueprint to database
                from .db import save_blueprint_to_sqlite
                success = save_blueprint_to_sqlite(blueprint)

                if not success:
                    logger.error("Failed to save blueprint to database")
                    self.status = {"state": "failed", "reason": "database_save_failed", "progress": 98}
                    return False

                # Store the blueprint in memory for quick access
                self.latest_generated_blueprint = blueprint
            except Exception as e:
                logger.error(f"Error with object prediction or blueprint saving: {str(e)}", exc_info=True)
                self.status = {"state": "failed", "reason": "object_prediction_or_save_failed", "message": str(e), "progress": 95}
                return False

            # Final success update
            logger.info("Blueprint successfully generated and saved")
            self.status = {
                "state": "completed",
                "progress": 100,
                "room_count": len(rooms),
                "wall_count": len(walls),
                "object_count": len(objects)
            }
            return True

        except Exception as e:
            logger.error(f"Unhandled error generating blueprint: {str(e)}", exc_info=True)
            self.status = {"state": "failed", "reason": "exception", "message": str(e), "progress": 0}
            return False

    def _extract_device_area_mappings(self, transformed_positions: Dict) -> Dict[str, List[Dict]]:
        """
        Extract mappings of areas to device positions, creating a mapping of area_id to list of devices.
        Will fetch area information directly from Home Assistant if available.
        """
        logger.info("Extracting device area mappings...")
        # Get device->area mappings from database
        recent_area_predictions = get_recent_area_predictions(self.config.get('generation_settings', {}).get('area_window_minutes', 10))

        # Fetch area information from Home Assistant - this provides official room assignments
        try:
            logger.info("Fetching areas from Home Assistant for direct room assignments...")
            from .ha_client import HAClient
            ha_client = HAClient()
            ha_areas = ha_client.get_areas() or []

            # Log area information for debugging
            logger.info(f"Retrieved {len(ha_areas)} areas from Home Assistant")
            for area in ha_areas:
                entity_count = len(area.get('entities', []))
                logger.info(f"Area: {area.get('name')} (ID: {area.get('area_id')}) - {entity_count} entities")

            # Create a mapping of device_id to area_id from the device trackers
            device_trackers = ha_client.get_device_trackers() or []
            tracker_area_mappings = {}

            for tracker in device_trackers:
                device_id = tracker.get('entity_id', '').replace('device_tracker.', '')
                area_id = tracker.get('attributes', {}).get('area_id')
                if device_id and area_id:
                    tracker_area_mappings[device_id] = area_id
                    logger.debug(f"Found device tracker area mapping: {device_id} -> {area_id}")

            # Get all entities to check for area assignments
            all_entities = ha_client.get_states() or []
            entity_area_mappings = {}

            # Get light entities - these are good for determining room placement
            lights = ha_client.get_all_lights() or []
            logger.info(f"Found {len(lights)} light entities for area mapping")

            # Collect entities with area assignments from sensors, lights, and binary sensors
            for entity in all_entities:
                entity_id = entity.get('entity_id', '')
                entity_type = entity_id.split('.')[0] if '.' in entity_id else ''

                # Focus on certain entity types that are typically fixed in specific rooms
                if entity_type not in ['sensor', 'light', 'binary_sensor', 'switch', 'media_player']:
                    continue

                device_id = entity_id.replace(f"{entity_type}.", '')
                area_id = entity.get('attributes', {}).get('area_id')

                if device_id and area_id:
                    entity_area_mappings[device_id] = area_id
                    logger.debug(f"Found entity area mapping: {device_id} -> {area_id}")

                # Special handling for BLE proxy entities with location names in them
                if 'ble' in device_id.lower() and (entity_type == 'sensor' or entity_type == 'binary_sensor'):
                    # Handle specific naming patterns from ESP proxies (e.g., to_bathroom_ble, to_office_ble)
                    if device_id.startswith('to_') and '_ble' in device_id:
                        room_part = device_id.replace('to_', '').replace('_ble', '')

                        # Try to match this device to an area by name
                        matched_area = None
                        for area in ha_areas:
                            area_name = area.get('name', '').lower().replace(' ', '_')
                            area_id = area.get('area_id', '')

                            # Check if the room part is in the area name or area ID
                            if room_part in area_name or room_part in area_id.lower():
                                matched_area = area_id
                                logger.info(f"Matched BLE proxy {device_id} to area {area_id} by name")
                                break

                        if matched_area:
                            entity_area_mappings[device_id] = matched_area
                            # Also map the actual BLE proxy entity to this area
                            if entity_id:
                                entity_area_mappings[entity_id] = matched_area
                                logger.info(f"Also mapped BLE entity {entity_id} to area {matched_area}")
                        else:
                            # If no direct match, create a reasonable area name from the device ID
                            # This ensures we have area information even without explicit matching
                            inferred_area = room_part
                            logger.info(f"Inferred area name '{inferred_area}' for BLE proxy {device_id}")
                            entity_area_mappings[device_id] = inferred_area
                            if entity_id:
                                entity_area_mappings[entity_id] = inferred_area

            logger.info(f"Found {len(tracker_area_mappings)} device tracker area mappings and {len(entity_area_mappings)} entity area mappings")
        except Exception as e:
            logger.error(f"Error fetching areas from Home Assistant: {str(e)}")
            ha_areas = []
            tracker_area_mappings = {}
            entity_area_mappings = {}

        # Map each device to its area, prioritizing direct HA assignments over database predictions
        device_areas = {}
        assigned_devices = 0

        # Process each device in transformed positions
        for device_id, position in transformed_positions.items():
            # 1. First priority: Check if device has a direct assignment in device trackers
            if device_id in tracker_area_mappings:
                area_id = tracker_area_mappings[device_id]
                device_areas[device_id] = area_id
                assigned_devices += 1
                logger.debug(f"Assigned device {device_id} to area {area_id} from tracker")

            # 2. Second priority: Check entities for area assignments (e.g., ESP proxies)
            elif device_id in entity_area_mappings:
                area_id = entity_area_mappings[device_id]
                device_areas[device_id] = area_id
                assigned_devices += 1
                logger.debug(f"Assigned device {device_id} to area {area_id} from entity attributes")

            # 3. Third priority: Use recent predictions from database
            elif device_id in recent_area_predictions:
                area_id = recent_area_predictions[device_id]
                if area_id:  # Skip None values
                    device_areas[device_id] = area_id
                    assigned_devices += 1
                    logger.debug(f"Assigned device {device_id} to area {area_id} from recent predictions")

            # 4. For ESPresense/BLE proxies with location in name, use that
            elif any(prefix in device_id for prefix in ['to_', 'ble_', 'esp32_']):
                # Extract location from name (to_kitchen_ble -> kitchen)
                if device_id.startswith('to_') and '_ble' in device_id:
                    location = device_id.replace('to_', '').replace('_ble', '')
                    device_areas[device_id] = location
                    assigned_devices += 1
                    logger.debug(f"Assigned device {device_id} to area {location} from naming pattern")

        # Group devices by area
        area_groups = {}
        for device_id, area_id in device_areas.items():
            if area_id not in area_groups:
                area_groups[area_id] = []

            # Only include devices that exist in transformed positions
            if device_id in transformed_positions:
                device_data = {
                    "device_id": device_id,
                    "position": transformed_positions[device_id],
                }
                area_groups[area_id].append(device_data)

        # Make sure "unknown" exists in the groups (we won't skip it anymore)
        if "unknown" not in area_groups:
            area_groups["unknown"] = []

        # Add devices without area assignments to "unknown"
        for device_id, position in transformed_positions.items():
            if device_id not in device_areas:
                # This device has no area assignment
                device_data = {
                    "device_id": device_id,
                    "position": position,
                }
                area_groups["unknown"].append(device_data)

        # Try to map scanner entities to areas based on name patterns if they weren't already assigned
        self._map_scanners_to_areas_by_name(transformed_positions, area_groups, device_areas, ha_areas)

        num_areas = len(area_groups)
        total_devices = sum(len(devices) for devices in area_groups.values())
        logger.info(f"Grouped {total_devices} devices into {num_areas} area groups")

        # Log the centroids of each area group
        centroids = {}
        for area_id, devices in area_groups.items():
            if devices:
                x_coords = [d['position'].get('x', 0) for d in devices]
                y_coords = [d['position'].get('y', 0) for d in devices]
                if x_coords and y_coords:
                    centroids[area_id] = (
                        sum(x_coords) / len(x_coords),
                        sum(y_coords) / len(y_coords)
                    )
        logger.info(f"Source centroids: {centroids}")

        return area_groups

    def _map_scanners_to_areas_by_name(self, transformed_positions: Dict, area_groups: Dict, device_areas: Dict, ha_areas: List[Dict]):
        """
        Maps scanner devices to areas based on naming patterns if they weren't already assigned.

        Args:
            transformed_positions: The positions of all devices
            area_groups: Current grouping of devices by area
            device_areas: Mapping from device_id to area_id
            ha_areas: List of areas from Home Assistant
        """
        # Look for scanner devices that aren't assigned to specific areas
        for device_id, position in transformed_positions.items():
            # Skip devices that already have area assignments
            if device_id in device_areas:
                continue

            # Check if this looks like a scanner/proxy based on name patterns
            is_likely_scanner = any(pattern in device_id.lower() for pattern in
                                  ['ble_', 'bt_', 'proxy', 'scanner', 'esp', 'beacon'])

            if is_likely_scanner:
                # First, check for ESP32 BLE proxy naming pattern (to_room_ble)
                if device_id.startswith('to_') and '_ble' in device_id.lower():
                    room_part = device_id.replace('to_', '').replace('_ble', '')
                    matched_area = None

                    # Try to match with area names from Home Assistant
                    for area in ha_areas:
                        area_id = area.get('area_id', '')
                        area_name = area.get('name', '').lower().replace(' ', '_')

                        # Check if room part is in area name or area ID with fuzzy matching
                        if (room_part in area_name or room_part in area_id.lower() or
                            area_name in room_part or area_id.lower() in room_part):
                            matched_area = area_id
                            logger.info(f"Matched ESP32 BLE proxy {device_id} to area {area_id} based on name")
                            break

                    if matched_area:
                        # Add scanner to the matched area
                        if matched_area not in area_groups:
                            area_groups[matched_area] = []

                        device_data = {
                            "device_id": device_id,
                            "position": position,
                        }
                        area_groups[matched_area].append(device_data)
                        device_areas[device_id] = matched_area

                        # Remove from unknown if it was there
                        if "unknown" in area_groups:
                            area_groups["unknown"] = [d for d in area_groups["unknown"] if d["device_id"] != device_id]
                            logger.debug(f"Moved ESP32 BLE proxy {device_id} from 'unknown' to '{matched_area}'")
                    else:
                        # If not matched to existing area, create its own area with its name
                        inferred_area = room_part
                        if inferred_area not in area_groups:
                            area_groups[inferred_area] = []

                        device_data = {
                            "device_id": device_id,
                            "position": position,
                        }
                        area_groups[inferred_area].append(device_data)
                        device_areas[device_id] = inferred_area

                        # Remove from unknown
                        if "unknown" in area_groups:
                            area_groups["unknown"] = [d for d in area_groups["unknown"] if d["device_id"] != device_id]
                            logger.info(f"Created new area '{inferred_area}' for ESP32 BLE proxy {device_id}")

                    continue  # Skip to next device since we've handled this one

                # Try to match with area name for other scanner types
                matched_area = None
                for area in ha_areas:
                    area_id = area.get('area_id', '')
                    area_name = area.get('name', '').lower().replace(' ', '_')

                    if area_name and (area_name in device_id.lower() or area_id.lower() in device_id.lower()):
                        matched_area = area_id
                        logger.debug(f"Matched scanner {device_id} to area {area_id} based on name")
                        break

                if matched_area:
                    # Add scanner to the matched area
                    if matched_area not in area_groups:
                        area_groups[matched_area] = []

                    device_data = {
                        "device_id": device_id,
                        "position": position,
                    }
                    area_groups[matched_area].append(device_data)
                    device_areas[device_id] = matched_area

                    # Remove from unknown if it was there
                    if "unknown" in area_groups:
                        area_groups["unknown"] = [d for d in area_groups["unknown"] if d["device_id"] != device_id]
                        logger.debug(f"Moved scanner {device_id} from 'unknown' to '{matched_area}'")

    def _generate_rooms(self, area_groups: Dict[str, List[Dict]], transformed_positions: Dict) -> List[Dict]:
        """
        Generate room geometries from area groups.

        Args:
            area_groups: Dictionary mapping area IDs to lists of device data
            transformed_positions: Dictionary of all transformed positions

        Returns:
            List of room objects with geometries
        """
        logger.info(f"Generating rooms from {len(area_groups)} area groups")
        rooms = []
        room_id = 1

        # Get min points per room from config
        min_points_per_room = self.config.get('generation_settings', {}).get('min_points_per_room', 0)

        for area_id, devices in area_groups.items():
            # No longer skip 'unknown' - we'll include it if it has enough devices
            # Instead, require a minimum number of points
            if len(devices) < min_points_per_room:
                logger.warning(f"Skipping area '{area_id}' with only {len(devices)} devices (minimum required: {min_points_per_room})")
                continue

            # Generate room for this area
            try:
                # Calculate centroid based on device positions
                if len(devices) == 0:
                    logger.warning(f"No devices found in area '{area_id}', skipping")
                    continue

                x_coords = []
                y_coords = []
                z_coords = []

                for device in devices:
                    pos = device.get('position', {})
                    if 'x' in pos and 'y' in pos:
                        x_coords.append(pos['x'])
                        y_coords.append(pos['y'])
                        z_coords.append(pos.get('z', 0))

                if not x_coords or not y_coords:
                    logger.warning(f"No valid positions in area '{area_id}', skipping")
                    continue

                # Calculate centroid
                center_x = sum(x_coords) / len(x_coords)
                center_y = sum(y_coords) / len(y_coords)
                center_z = sum(z_coords) / len(z_coords) if z_coords else 0

                # Generate room ID
                room_name = area_id.replace('_', ' ').title()

                # Calculate bounds and dimensions
                min_x = min(x_coords)
                max_x = max(x_coords)
                min_y = min(y_coords)
                max_y = max(y_coords)

                # Expand bounds to ensure minimum room size
                width = max_x - min_x
                length = max_y - min_y
                min_room_size = 2.0  # Minimum room dimension in meters

                if width < min_room_size:
                    padding = (min_room_size - width) / 2
                    min_x -= padding
                    max_x += padding
                    width = min_room_size

                if length < min_room_size:
                    padding = (min_room_size - length) / 2
                    min_y -= padding
                    max_y += padding
                    length = min_room_size

                # Add padding for aesthetics
                padding = 0.5  # Padding in meters
                min_x -= padding
                max_x += padding
                min_y -= padding
                max_y += padding
                width = max_x - min_x
                length = max_y - min_y

                # Define standard room height
                height = 2.5  # Standard room height in meters

                # Create room object
                room = {
                    'id': f"room_{room_id}",
                    'name': room_name,
                    'area_id': area_id,
                    'type': self._determine_room_type(room_name, area_id),
                    'center': {
                        'x': center_x,
                        'y': center_y,
                        'z': center_z
                    },
                    'bounds': {
                        'min': {'x': min_x, 'y': min_y, 'z': 0},
                        'max': {'x': max_x, 'y': max_y, 'z': height}
                    },
                    'dimensions': {
                        'width': width,
                        'length': length,
                        'height': height
                    },
                    'devices': devices,
                    'floor': 0  # Default to ground floor
                }

                rooms.append(room)
                room_id += 1

            except Exception as e:
                logger.error(f"Error generating room for area '{area_id}': {str(e)}")

        # Log the results
        logger.info(f"Generated {len(rooms)} room geometries")
        return rooms

    def _determine_room_type(self, room_name: str, area_id: str) -> str:
        """
        Determine the type of room based on its name and area ID.

        Args:
            room_name: The name of the room
            area_id: The ID of the area

        Returns:
            The determined room type
        """
        # Convert to lowercase for case-insensitive matching
        name_lower = room_name.lower()
        area_lower = area_id.lower()

        # Define room type patterns
        type_patterns = {
            'living_room': ['living', 'lounge', 'family'],
            'kitchen': ['kitchen', 'cooking'],
            'bathroom': ['bath', 'shower', 'toilet', 'wc'],
            'bedroom': ['bed', 'master', 'guest room'],
            'office': ['office', 'study', 'work'],
            'dining_room': ['dining', 'breakfast'],
            'hallway': ['hall', 'corridor', 'entrance'],
            'laundry_room': ['laundry', 'utility'],
            'garage': ['garage', 'carport'],
            'balcony': ['balcony', 'terrace', 'patio']
        }

        # Check for matches in name or area_id
        for room_type, patterns in type_patterns.items():
            for pattern in patterns:
                if pattern in name_lower or pattern in area_lower:
                    return room_type

        # Default to standard room
        return 'room'

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

    def _calculate_area_centroids(self, device_area_groups: Dict) -> Dict:
        """Calculate the centroid of each device group by area."""
        area_centroids = {}

        for area_id, devices in device_area_groups.items():
            if not devices:
                continue

            # Calculate average position
            x_coords = [d['position']['x'] for d in devices]
            y_coords = [d['position']['y'] for d in devices]

            # Centroid is the average position
            avg_x = sum(x_coords) / len(x_coords)
            avg_y = sum(y_coords) / len(y_coords)

            area_centroids[area_id] = (avg_x, avg_y)

        return area_centroids

    def _calculate_transformation(self, source_centroids: Dict, target_layout: Dict) -> Dict:
        """Calculate transformation from source to target using Procrustes analysis."""
        # Detailed logging of input data
        logger.info(f"Source centroids: {source_centroids}")
        logger.info(f"Target layout: {target_layout}")
        logger.info(f"Source area IDs: {list(source_centroids.keys())}")
        logger.info(f"Target area IDs: {list(target_layout.keys())}")

        # Check if either source_centroids or target_layout is empty
        if not source_centroids or not target_layout:
            logger.error(f"Cannot calculate transformation: source_centroids empty: {not source_centroids}, target_layout empty: {not target_layout}")

            # Return identity transformation as emergency fallback
            return {
                'rotation': np.eye(2),
                'scale': 1.0,
                'translation': np.array([0.0, 0.0]),
                'source_mean': np.array([0.0, 0.0]),
                'target_mean': np.array([0.0, 0.0]),
                'is_fallback': True
            }

        # Filter for areas that exist in both source and target
        common_areas = set(source_centroids.keys()) & set(target_layout.keys())
        logger.info(f"Common areas for transformation: {list(common_areas)}")

        if len(common_areas) < 2:
            logger.warning(f"Not enough common areas for transformation: {len(common_areas)}")

            # IMPROVED FALLBACK MECHANISM
            if len(source_centroids) > 0 and len(target_layout) > 0:
                logger.info("Using improved fallback transformation mechanism")

                # Get first areas from each set for simple translation
                source_area = list(source_centroids.keys())[0]
                target_area = list(target_layout.keys())[0]

                # Get centroid coordinates
                source_coords = np.array(source_centroids[source_area])
                target_coords = np.array(target_layout[target_area])

                logger.info(f"Fallback using source area '{source_area}' at {source_coords}")
                logger.info(f"Fallback using target area '{target_area}' at {target_coords}")

                # Create simple identity transformation with translation only
                return {
                    'rotation': np.eye(2),  # Identity rotation matrix
                    'scale': 1.0,            # No scaling
                    'translation': target_coords - source_coords,
                    'source_mean': source_coords,
                    'target_mean': target_coords,
                    'is_fallback': True      # Flag this as a fallback transformation
                }
            else:
                logger.error("No source centroids or target layout available for transformation")
                return {
                    'rotation': np.eye(2),
                    'scale': 1.0,
                    'translation': np.array([0.0, 0.0]),
                    'source_mean': np.array([0.0, 0.0]) if not source_centroids else np.mean(np.array(list(source_centroids.values())), axis=0),
                    'target_mean': np.array([0.0, 0.0]) if not target_layout else np.mean(np.array(list(target_layout.values())), axis=0),
                    'is_fallback': True
                }

        # Extract matching points as numpy arrays for Procrustes analysis
        source_points = np.array([source_centroids[area] for area in common_areas])
        target_points = np.array([target_layout[area] for area in common_areas])

        logger.info(f"Using {len(common_areas)} common areas for transformation")
        logger.debug(f"Source points: {source_points}")
        logger.debug(f"Target points: {target_points}")

        # Calculate means
        source_mean = source_points.mean(axis=0)
        target_mean = target_points.mean(axis=0)
        logger.debug(f"Source mean: {source_mean}, Target mean: {target_mean}")

        # Center the points
        source_centered = source_points - source_mean
        target_centered = target_points - target_mean

        try:
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
                logger.warning("Source variance is zero, using scale=1.0")
            else:
                scale = np.sqrt(np.sum(target_centered ** 2) / source_var)

            # Calculate translation
            translation = target_mean - scale * source_mean @ rotation

            logger.info(f"Transformation calculated successfully: scale={scale:.2f}, det(rotation)={np.linalg.det(rotation):.2f}")
            logger.debug(f"Rotation matrix: {rotation}")
            logger.debug(f"Translation vector: {translation}")

            return {
                'rotation': rotation,
                'scale': scale,
                'translation': translation,
                'source_mean': source_mean,
                'target_mean': target_mean,
                'is_fallback': False  # This is a proper transformation
            }
        except Exception as e:
            logger.error(f"Error during transformation calculation: {e}", exc_info=True)

            # Emergency fallback
            logger.warning("Using emergency fallback for transformation")
            return {
                'rotation': np.eye(2),
                'scale': 1.0,
                'translation': np.array([0.0, 0.0]),
                'source_mean': source_mean,
                'target_mean': target_mean,
                'is_fallback': True
            }

    def _apply_transformation(self, positions: Dict, transform_params: Dict) -> Dict:
        """Apply the calculated transformation to all positions."""
        if not transform_params:
            logger.error("No transformation parameters provided, returning original positions")
            return positions

        # Log input parameters
        logger.info(f"Applying transformation to {len(positions)} positions")
        logger.debug(f"Transformation params: {transform_params}")

        rotation = transform_params.get('rotation')
        scale = transform_params.get('scale', 1.0)
        translation = transform_params.get('translation')
        is_fallback = transform_params.get('is_fallback', False)

        if rotation is None or translation is None:
            logger.error("Missing required transformation parameters")
            return positions

        if is_fallback:
            logger.info("Using fallback transformation (minimal adjustment)")

        transformed_positions = {}

        # Sample some positions to log
        sample_keys = list(positions.keys())[:2]
        logger.debug(f"Sample position keys: {sample_keys}")
        for key in sample_keys:
            logger.debug(f"Original position for {key}: {positions[key]}")

        for entity_id, position in positions.items():
            try:
                # Convert position to numpy array based on its format
                if isinstance(position, dict):
                    # Handle dictionary format with 'x', 'y', 'z' keys
                    pos_array = np.array([float(position.get('x', 0.0)), float(position.get('y', 0.0))])
                    has_z = 'z' in position
                    z_value = float(position.get('z', 0.0))
                elif isinstance(position, (list, tuple)):
                    # Handle tuple/list format with indices
                    if len(position) >= 2:
                        pos_array = np.array([float(position[0]), float(position[1])])
                        has_z = len(position) > 2
                        z_value = float(position[2]) if has_z else 0.0
                    else:
                        logger.warning(f"Position for {entity_id} doesn't have enough elements: {position}")
                        pos_array = np.array([0.0, 0.0])
                        has_z = False
                        z_value = 0.0
                else:
                    logger.warning(f"Unexpected position format for {entity_id}: {type(position)} - {position}")
                    continue

                # Apply transformation: scale, rotate, translate
                # Use matrix multiplication for numpy arrays
                if isinstance(rotation, np.ndarray) and rotation.shape == (2, 2):
                    # Matrix rotation
                    transformed = scale * (rotation @ pos_array) + translation
                else:
                    # Scalar rotation (angle in radians)
                    rot_matrix = np.array([
                        [np.cos(rotation), -np.sin(rotation)],
                        [np.sin(rotation), np.cos(rotation)]
                    ])
                    transformed = scale * (rot_matrix @ pos_array) + translation

                # Convert back to the original format
                if isinstance(position, dict):
                    # Return in same dictionary format
                    transformed_positions[entity_id] = {
                        'x': float(transformed[0]),
                        'y': float(transformed[1]),
                        'z': z_value  # Keep original z coordinate
                    }
                elif isinstance(position, list):
                    # Return as list
                    if has_z:
                        transformed_positions[entity_id] = [float(transformed[0]), float(transformed[1]), z_value]
                    else:
                        transformed_positions[entity_id] = [float(transformed[0]), float(transformed[1])]
                else:
                    # Return as tuple
                    if has_z:
                        transformed_positions[entity_id] = (float(transformed[0]), float(transformed[1]), z_value)
                    else:
                        transformed_positions[entity_id] = (float(transformed[0]), float(transformed[1]))

            except Exception as e:
                logger.error(f"Error transforming position for {entity_id}: {e}", exc_info=True)
                # Keep original position in case of error
                transformed_positions[entity_id] = position

        # Log a sample of transformed positions
        for key in sample_keys:
            if key in transformed_positions:
                logger.debug(f"Transformed position for {key}: {transformed_positions[key]}")

        logger.info(f"Successfully transformed {len(transformed_positions)} positions")
        return transformed_positions

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

    # Get dynamic anchors from the static device detector
    try:
        dynamic_anchors = static_detector.get_dynamic_anchors()
        if dynamic_anchors:
            logger.info(f"Found {len(dynamic_anchors)} dynamic anchors from static device detection")

            # Process dynamic anchors and add them as reference positions
            for device_id, anchor_data in dynamic_anchors.items():
                # Only add if not already in existing references
                if device_id not in existing_refs:
                    save_reference_position(
                        device_id=device_id,
                        x=anchor_data.get('x', 0),
                        y=anchor_data.get('y', 0),
                        z=anchor_data.get('z', 0),
                        area_id=anchor_data.get('area_id'),
                        confidence=anchor_data.get('confidence', 0.8),  # Use confidence from static device detection
                        is_dynamic=True  # Mark as dynamically identified
                    )
                    logger.info(f"Added dynamic anchor as reference position: {device_id} at "
                               f"({anchor_data.get('x', 0)}, {anchor_data.get('y', 0)}, {anchor_data.get('z', 0)}) "
                               f"with confidence {anchor_data.get('confidence', 0.8)}")
        else:
            logger.info("No dynamic anchors found from static device detection")
    except Exception as e:
        logger.warning(f"Failed to get dynamic anchors: {e}")

    # Refresh our list of reference positions after adding dynamic anchors
    existing_refs = get_reference_positions_from_sqlite()

    if len(existing_refs) >= 3:
        logger.info(f"Found {len(existing_refs)} reference positions including dynamic anchors")
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
