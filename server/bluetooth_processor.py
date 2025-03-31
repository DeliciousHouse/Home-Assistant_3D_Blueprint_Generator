import json
import logging
import math
import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import DBSCAN
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from .ai_processor import AIProcessor
from .db import get_sqlite_connection, get_reference_positions_from_sqlite, save_device_position_to_sqlite

logger = logging.getLogger(__name__)

class BluetoothProcessor:
    """Process Bluetooth signals for position estimation."""

    def __init__(self, config_path=None):
        """Initialize the Bluetooth processor.

        Args:
            config_path: Optional path to configuration file. If None, will load default config.
        """
        # Import and load configuration
        from .config_loader import load_config
        self.config = load_config(config_path)

        # Extract what we need from config
        self.fixed_sensors = self.config.get('fixed_sensors', {})
        self.rssi_threshold = self.config.get('processing_params', {}).get('rssi_threshold', -85)

        # Set processing parameters
        self.reference_power = self.config.get('processing_params', {}).get('distance_calculation', {}).get('reference_power', -65)
        self.path_loss_exponent = self.config.get('processing_params', {}).get('distance_calculation', {}).get('path_loss_exponent', 2.8)
        self.minimum_sensors = self.config.get('processing_params', {}).get('minimum_sensors', 3)
        self.accuracy_threshold = self.config.get('processing_params', {}).get('accuracy_threshold', 2.0)

        # Load fixed scanner locations from config
        self.scanner_locations = self.config.get('fixed_sensors', {})
        logger.info(f"Loaded {len(self.scanner_locations)} fixed scanner locations from config")

        # Initialize AI processor for ML-based distance estimation
        self.ai_processor = AIProcessor(config_path)
        self.use_ml_distance = self.config.get('processing_params', {}).get('use_ml_distance', True)

        # Load additional reference positions from database
        self.load_reference_positions()

    def load_reference_positions(self):
        """Load fixed reference positions from the database."""
        try:
            logger.info("Loading fixed reference positions from database")

            # Use the function from db to get reference positions
            db_scanner_positions = get_reference_positions_from_sqlite()

            # Update the scanner_locations dictionary
            if db_scanner_positions:
                # Add or update positions from database
                for scanner_id, position in db_scanner_positions.items():
                    if scanner_id not in self.scanner_locations:
                        self.scanner_locations[scanner_id] = position
                logger.info(f"Loaded {len(db_scanner_positions)} reference positions from database")
                return True
            else:
                logger.warning("No reference positions found in database")
                return False

        except Exception as e:
            logger.error(f"Error loading reference positions: {e}")
            return False

    def process_bluetooth_sensors(self) -> Dict:
        """Process Bluetooth signals from Home Assistant to determine device positions."""
        from .ha_client import HomeAssistantClient

        try:
            logger.info("Starting Bluetooth sensor processing from Home Assistant")

            # Get all sensor entities from HA
            ha_client = HomeAssistantClient()
            all_sensors = ha_client.get_sensor_entities()

            logger.info(f"Fetched {len(all_sensors)} total sensors")

            # Initialize tracking structures
            distance_readings = {}     # {device_id: [{'scanner_id': scanner_name, 'distance': value}, ...]}
            device_positions = {}      # Final positions after processing
            device_groups = {}         # Group other BLE readings by device
            device_area_ids = {}       # Store area_id by device_id

            # First, collect all direct position readings and area_ids
            for entity in all_sensors:
                entity_id = entity.get('entity_id', '')
                state = entity.get('state', '')
                attrs = entity.get('attributes', {})
                area_id = entity.get('area_id')

                # Extract device ID
                device_id = entity.get('device_id')
                if not device_id:
                    # Skip if no device ID (needed for tracking)
                    continue

                # Store area_id if available
                if device_id and area_id:
                    device_area_ids[device_id] = area_id

                # Skip entities with unavailable/unknown states
                if state in ('unavailable', 'unknown', None, '') or not isinstance(state, (str, int, float)):
                    continue

                # Process direct position data first (priority)
                if device_id and any(k in attrs for k in ['coordinates', 'x', 'y']):
                    # Device already has position data
                    try:
                        if 'coordinates' in attrs:
                            coords = attrs['coordinates']
                            if isinstance(coords, dict) and all(k in coords for k in ['x', 'y']):
                                device_positions[device_id] = {
                                    'x': float(coords['x']),
                                    'y': float(coords['y']),
                                    'z': float(coords.get('z', 0)),
                                    'accuracy': float(attrs.get('accuracy', 1.0)),
                                    'source': entity_id,
                                    'area_id': area_id
                                }
                        elif all(k in attrs for k in ['x', 'y']):
                            device_positions[device_id] = {
                                'x': float(attrs['x']),
                                'y': float(attrs['y']),
                                'z': float(attrs.get('z', 0)),
                                'accuracy': float(attrs.get('accuracy', 1.0)),
                                'source': entity_id,
                                'area_id': area_id
                            }
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing position for {device_id}: {e}")

            # Process Bermuda distance readings
            bermuda_count = 0
            for entity in all_sensors:
                entity_id = entity.get('entity_id', '')
                state = entity.get('state', '')
                attrs = entity.get('attributes', {})

                # Skip if already processed or invalid state
                if state in ('unavailable', 'unknown', None, '') or not isinstance(state, (str, int, float)):
                    continue

                # Try to parse as a Bermuda distance reading
                is_bermuda = False
                device_id = None
                scanner_id = None

                # Pattern 1: sensor.bermuda_<mac>_distance
                if 'bermuda' in entity_id and '_distance' in entity_id:
                    is_bermuda = True
                    parts = entity_id.split('_')
                    if len(parts) >= 3:
                        device_id = parts[1]
                        scanner_id = attrs.get('scanner_id')
                elif '_distance_' in entity_id:
                    # Pattern 2: sensor.<mac>_distance_<scanner>
                    is_bermuda = True
                    parts = entity_id.split('_distance_')
                    if len(parts) == 2:
                        device_part = parts[0].replace('sensor.', '')
                        device_id = device_part
                        scanner_id = parts[1]

                # Process Bermuda distance reading
                if is_bermuda and device_id and scanner_id:
                    bermuda_count += 1
                    try:
                        distance = float(state)
                        if distance > 0:
                            if device_id not in distance_readings:
                                distance_readings[device_id] = []

                            # Add the distance reading
                            distance_readings[device_id].append({
                                'scanner_id': scanner_id,
                                'distance': distance,
                                'source': entity_id,
                                'area_id': device_area_ids.get(device_id)
                            })
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Error processing Bermuda distance for {device_id}: {e}")

            # Collect RSSI readings as fallback
            for entity in all_sensors:
                entity_id = entity.get('entity_id', '')
                attrs = entity.get('attributes', {})
                device_id = entity.get('device_id')

                # Skip if we already have position or Bermuda readings for this device
                if device_id and (device_id in device_positions or device_id in distance_readings):
                    continue

                # Process RSSI readings
                rssi = attrs.get('rssi')
                if rssi is not None and device_id:
                    try:
                        # Skip weak signals
                        if float(rssi) < self.rssi_threshold:
                            continue

                        # Group by device
                        if device_id not in device_groups:
                            device_groups[device_id] = {}

                        # Remember RSSI reading associated with each entity
                        device_groups[device_id][entity_id] = {
                            'rssi': float(rssi),
                            'attrs': attrs,
                            'area_id': device_area_ids.get(device_id)
                        }
                    except (ValueError, TypeError):
                        continue

            logger.info(f"Found {bermuda_count} Bermuda distance readings for {len(distance_readings)} devices")
            logger.info(f"Already have {len(device_positions)} devices with direct position data")

            # Process Bermuda distance readings (primary method)
            for device_id, readings in distance_readings.items():
                # Skip if already have position
                if device_id in device_positions:
                    continue

                # Need at least 3 readings for trilateration
                if len(readings) >= 3:
                    # Extract scanner positions and distances
                    scanner_positions = []
                    scanner_distances = []

                    for reading in readings:
                        scanner_id = reading['scanner_id']
                        distance = reading['distance']

                        # Get scanner position from collected locations
                        if scanner_id in self.scanner_locations:
                            scanner_pos = self.scanner_locations[scanner_id]
                            scanner_positions.append((scanner_pos['x'], scanner_pos['y'], scanner_pos.get('z', 0)))
                            scanner_distances.append(distance)

                    # Need at least 3 valid readings with known scanner positions
                    if len(scanner_positions) >= 3:
                        # Perform trilateration
                        position = self._trilaterate_from_distances(scanner_positions, scanner_distances)

                        if position:
                            device_positions[device_id] = {
                                'x': position[0],
                                'y': position[1],
                                'z': position[2],
                                'accuracy': 1.0,  # High confidence
                                'source': 'bermuda_trilateration',
                                'area_id': device_area_ids.get(device_id)
                            }

            # Fall back to RSSI-based positioning for remaining devices
            for device_id, entities in device_groups.items():
                # Skip if already have position
                if device_id in device_positions:
                    continue

                # Convert RSSI readings to distance readings for each scanner
                rssi_readings = {}
                for entity_id, data in entities.items():
                    rssi = data['rssi']

                    # Get the scanner ID - try to find which scanner this entity belongs to
                    scanner_id = entity_id.replace('sensor.', '')

                    # Only use readings from known scanners
                    if scanner_id in self.scanner_locations:
                        rssi_readings[scanner_id] = rssi

                # Calculate position from RSSI readings
                if len(rssi_readings) >= 3:
                    position = self.calculate_device_position(device_id, rssi_readings)
                    if position:
                        device_positions[device_id] = position

            # Save all collected device positions to the database
            if device_positions:
                saved = self._save_device_positions_to_db(device_positions)
                logger.info(f"Saved {len(device_positions)} device positions to database: {saved}")
            else:
                logger.warning("No device positions found to save")

            # Detect rooms based on device positions
            rooms = []
            if len(device_positions) >= 3:
                rooms = self.detect_rooms(device_positions)
                logger.info(f"Detected {len(rooms)} rooms")

            return {
                "processed": len(all_sensors),
                "positions_found": len(device_positions),
                "rooms_detected": len(rooms),
                "bermuda_readings": bermuda_count,
                "device_positions": device_positions,
                "rooms": rooms
            }

        except Exception as e:
            logger.error(f"Error processing Bluetooth sensors: {e}", exc_info=True)
            return {"error": str(e)}

    def detect_rooms(self, positions: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Detect rooms based on device positions using Home Assistant areas."""
        if not positions:
            return []

        # Get Home Assistant areas
        from .ha_client import HomeAssistantClient
        ha_client = HomeAssistantClient()
        ha_areas = ha_client.get_areas()

        logger.info(f"Detecting rooms using {len(ha_areas)} Home Assistant areas")

        # Group device positions by area_id
        positions_by_area = {}
        for device_id, position in positions.items():
            area_id = position.get('area_id')
            if area_id:
                if area_id not in positions_by_area:
                    positions_by_area[area_id] = []
                positions_by_area[area_id].append((device_id, position))

        # Generate rooms from areas
        rooms = []

        # Process areas with devices
        for area in ha_areas:
            area_id = area.get('area_id')
            if area_id in positions_by_area and len(positions_by_area[area_id]) > 0:
                # Calculate room properties based on device positions
                devices_in_area = positions_by_area[area_id]

                # Extract coordinates
                x_coords = [pos['x'] for _, pos in devices_in_area]
                y_coords = [pos['y'] for _, pos in devices_in_area]
                z_coords = [pos['z'] for _, pos in devices_in_area]

                # Calculate bounds
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                min_z, max_z = min(z_coords), max(z_coords)

                # Add some padding
                padding = 0.5  # 0.5m padding
                min_x -= padding
                min_y -= padding
                max_x += padding
                max_y += padding

                # Calculate center
                center_x = (min_x + max_x) / 2
                center_y = (min_y + max_y) / 2
                center_z = (min_z + max_z) / 2

                # Create room object
                room = {
                    'id': f"room_{area_id}",
                    'name': area.get('name', f"Room {area_id}"),
                    'center': {'x': center_x, 'y': center_y, 'z': center_z},
                    'dimensions': {
                        'width': max(max_x - min_x, 2.0),  # Minimum 2m width
                        'length': max(max_y - min_y, 2.0),  # Minimum 2m length
                        'height': max(max_z - min_z, 2.4)   # Minimum 2.4m height
                    },
                    'bounds': {
                        'min': {'x': min_x, 'y': min_y, 'z': min_z},
                        'max': {'x': max_x, 'y': max_y, 'z': max_z}
                    },
                    'devices': [device_id for device_id, _ in devices_in_area],
                    'area_id': area_id
                }

                rooms.append(room)

        logger.info(f"Generated {len(rooms)} rooms from Home Assistant areas")

        if not rooms:
            logger.warning("No rooms detected from Home Assistant areas")

        return rooms

    def calculate_device_position(self, device_id: str, rssi_readings: Dict[str, float]) -> Optional[Dict]:
        """
        Calculate device position from RSSI readings using trilateration.

        Args:
            device_id: The device identifier
            rssi_readings: Dict mapping scanner_id to RSSI value

        Returns:
            Position dictionary with x, y, z coordinates or None if calculation fails
        """
        try:
            scanner_positions = []
            scanner_distances = []

            # Convert RSSI to distance for each scanner
            for scanner_id, rssi in rssi_readings.items():
                # Look up scanner location
                if scanner_id in self.scanner_locations:
                    scanner_pos = self.scanner_locations[scanner_id]

                    # Estimate distance from RSSI using AI processor or fallback method
                    distance = self.ai_processor.estimate_distance(
                        rssi,
                        environment_type=scanner_pos.get('environment_type', 'indoor')
                    )

                    # Add to our lists for trilateration
                    if scanner_pos and 'x' in scanner_pos and 'y' in scanner_pos:
                        scanner_positions.append((
                            scanner_pos['x'],
                            scanner_pos['y'],
                            scanner_pos.get('z', 0)
                        ))
                        scanner_distances.append(distance)

            # Perform trilateration if we have enough readings
            if len(scanner_positions) >= 3:
                position = self._trilaterate_from_distances(scanner_positions, scanner_distances)

                if position:
                    return {
                        'x': position[0],
                        'y': position[1],
                        'z': position[2],
                        'accuracy': 2.0,  # Lower confidence than direct measurements
                        'source': 'rssi_trilateration'
                    }

            return None

        except Exception as e:
            logger.error(f"Error calculating position for {device_id}: {str(e)}")
            return None

    def _trilaterate_from_distances(self, scanner_positions, distances) -> Optional[Tuple[float, float, float]]:
        """Perform trilateration using scanner positions and distances."""
        try:
            if len(scanner_positions) < 3 or len(distances) < 3:
                logger.warning("Not enough reference points for trilateration")
                return None

            # Convert to numpy arrays for calculation
            positions = np.array(scanner_positions)
            distances = np.array(distances)

            # Initial guess: center of all scanners
            initial_position = np.mean(positions, axis=0)

            # Define optimization function
            def error_function(point):
                # Calculate distance from point to each scanner
                calculated_distances = np.sqrt(np.sum((positions - point)**2, axis=1))
                # Calculate mean squared error between calculated and measured distances
                mse = np.mean((calculated_distances - distances)**2)
                return mse

            # Perform optimization to find best position
            result = minimize(
                error_function,
                initial_position,
                method='L-BFGS-B',
                options={'ftol': 1e-5, 'maxiter': 100}
            )

            if result.success:
                # Return optimized position
                return (
                    float(result.x[0]),
                    float(result.x[1]),
                    float(result.x[2]) if len(result.x) > 2 else 0.0
                )
            else:
                logger.warning(f"Trilateration optimization failed: {result.message}")
                return None

        except Exception as e:
            logger.error(f"Trilateration error: {str(e)}")
            return None

    def _save_device_positions_to_db(self, device_positions):
        """Save device positions to database for blueprint generation."""
        try:
            count = 0
            for device_id, position in device_positions.items():
                source = position.get('source', 'calculated')
                accuracy = position.get('accuracy', 1.0)
                area_id = position.get('area_id')

                # Call db helper directly with all parameters
                if save_device_position_to_sqlite(
                    device_id,
                    position,
                    source=source,
                    accuracy=accuracy,
                    area_id=area_id
                ):
                    count += 1

            logger.info(f"Saved {count}/{len(device_positions)} device positions to database")
            return count > 0
        except Exception as e:
            logger.error(f"Error saving device positions: {e}")
            return False
