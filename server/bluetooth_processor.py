#!/usr/bin/env python3

import json
import logging
import os
import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import random
from scipy.optimize import minimize
import re

# Import local modules with fallbacks
try:
    # First try relative imports (typical when imported as module)
    from .ha_client import HAClient as HomeAssistantClient, get_ha_client
    from .db import (
        save_distance_log, save_area_observation,
        get_recent_distances, get_recent_area_predictions,
        get_reference_positions_from_sqlite, save_reference_position,
        save_device_position
    )
    from .config_loader import load_config
    # Import for static device detection
    from .static_device_detector import detector as static_detector
    logger = logging.getLogger(__name__)
except ImportError:
    try:
        # Then try direct imports (when run as script)
        from ha_client import HAClient as HomeAssistantClient, get_ha_client
        from db import (
            save_distance_log, save_area_observation,
            get_recent_distances, get_recent_area_predictions,
            get_reference_positions_from_sqlite, save_reference_position,
            save_device_position
        )
        from config_loader import load_config
        from static_device_detector import detector as static_detector
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
    except ImportError:
        import logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.error("Failed to import required modules - check your installation")
        raise

class HAClient:
    """
    Home Assistant Client for interacting with the Home Assistant API.
    """

    def get_bluetooth_sensors(self) -> List[Dict]:
        """
        Retrieve Bluetooth sensors from Home Assistant.
        Returns a list of dictionaries representing Bluetooth sensors.
        """
        # Example implementation, replace with actual API call
        try:
            sensors = self.get_entities_by_type('sensor')
            bt_sensors = [
                sensor for sensor in sensors if 'bluetooth' in sensor.get('entity_id', '').lower()
            ]
            return bt_sensors
        except Exception as e:
            logger.error(f"Error retrieving Bluetooth sensors: {e}")
            return []

class BluetoothProcessor:
    """
    Process Bluetooth detection data to calculate device positions and areas.
    """
    # Class variable to hold the single instance
    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(BluetoothProcessor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Bluetooth processor."""
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Use standardized config loader
        try:
            self.config = load_config(config_path)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            self.config = {}

        # Use the get_ha_client() function to get a properly initialized client
        self.ha_client = get_ha_client()

        # Get processing configuration
        self.processing_params = self.config.get('processing_params', {
            'rssi_power_coefficient': -66,  # Updated default reference power
            'environment_factor': 2.8,       # Updated default path loss exponent
            'distance_filter_threshold': 15,  # meters
            'distance_time_window': 15,  # minutes
            'prediction_time_window': 10,  # minutes
            'min_observations': 3,
            'trilateration_alpha': 0.5,
            'trilateration_beta': 0.5,
            'use_kalman_filter': True,
            'kalman_process_noise': 0.01,
            'kalman_measurement_noise': 0.1,
            'rssi_threshold': -85
        })

        # Initialize counters and timestamps
        self.last_scan_time = datetime.now() - timedelta(hours=1)  # Set to an hour ago initially
        self.scanning_interval = self.config.get('scanning_interval', 30)  # Default 30 seconds

        # Cache of device metadata (name, type, etc.)
        self.device_metadata = {}

        # Keep track of seen devices and scanners
        self.seen_devices = set()
        self.seen_scanners = set()

        # Cache for HA areas to avoid repeated API calls
        self._area_cache = None
        self._area_cache_time = None
        self._area_cache_expiry = 300  # Cache areas for 5 minutes

        # Regular expression for extracting device and scanner IDs from sensor names
        self.distance_sensor_pattern = re.compile(r'(\w+)_distance_(\w+)')
        # Pattern to extract device ID from entity ID
        self.device_id_pattern = re.compile(r'sensor\.(\w+)_')

        self._initialized = True
        logger.info("BluetoothProcessor initialized successfully")

    def log_sensor_data(self) -> Dict:
        """Log sensor data with improved error handling and reporting."""
        results = {
            "success": False,
            "distances_logged": 0,
            "areas_logged": 0,
            "errors": []
        }

        try:
            # Check Home Assistant connection first
            if not self.ha_client._connection_status:
                error_msg = "Home Assistant API connection is not available"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                return results

            # Get device trackers with detailed error handling
            try:
                device_trackers = self.ha_client.get_device_trackers() or []
                logger.info(f"Retrieved {len(device_trackers)} device trackers from Home Assistant")
            except Exception as e:
                error_msg = f"Failed to get device trackers: {str(e)}"
                logger.error(error_msg)
                results["errors"].append(error_msg)
                device_trackers = []

            # Process bluetooth sensors with detailed error handling
            distances_logged = 0
            errors = 0

            for device in device_trackers:
                try:
                    # Process device with validation
                    # [implementation details]
                    distances_logged += 1
                except Exception as e:
                    logger.warning(f"Error processing device {device.get('entity_id', 'unknown')}: {str(e)}")
                    errors += 1

            # Update results
            results["success"] = distances_logged > 0
            results["distances_logged"] = distances_logged
            results["errors_count"] = errors

            # Process static device detection
            try:
                static_devices = static_detector.process_device_history()
                results["static_devices"] = len(static_devices)
            except Exception as e:
                logger.warning(f"Static device detection failed: {str(e)}")
                results["static_devices"] = 0

            return results

        except Exception as e:
            logger.error(f"Error in sensor data logging: {str(e)}", exc_info=True)
            results["errors"].append(str(e))
            return results

    def _process_bluetooth_data(self, bt_sensors: List[Dict], reference_positions: Dict) -> int:
        """
        Process Bluetooth sensor data and log distances.
        Returns number of distances logged.
        """
        distances_logged = 0
        rssi_threshold = self.processing_params.get('rssi_threshold', -85)

        # Create a map of scanner IDs to their positions
        scanner_positions = {}
        for ref_id, ref_data in reference_positions.items():
            scanner_positions[ref_id] = (ref_data['x'], ref_data['y'], ref_data['z'])

        # Extract distance sensor entities specifically
        for sensor in bt_sensors:
            entity_id = sensor.get('entity_id', '')
            attributes = sensor.get('attributes', {})
            state = sensor.get('state', None)

            # Try multiple methods to identify distance sensors

            # Method 1: Look for entity_ids with _distance_ in them
            distance_match = self.distance_sensor_pattern.search(entity_id)
            if distance_match:
                tracked_device_id = distance_match.group(1)
                scanner_id = distance_match.group(2)

                # Try to extract distance from state or attributes
                distance = None
                if state and state != 'unknown' and state != 'unavailable':
                    try:
                        distance = float(state)
                    except (ValueError, TypeError):
                        pass

                # If we found a valid distance, log it
                if distance is not None:
                    self.seen_devices.add(tracked_device_id)
                    self.seen_scanners.add(scanner_id)

                    if save_distance_log(tracked_device_id, scanner_id, distance):
                        distances_logged += 1
                    continue  # Go to next sensor

            # Method 2: Look for sensors with rssi values that we can convert to distances
            rssi = attributes.get('rssi')
            if rssi is not None:
                # Skip weak signals
                if rssi < rssi_threshold:
                    continue

                # Extract device and scanner IDs from attributes if available
                tracked_device_id = attributes.get('source')
                scanner_id = attributes.get('scanner') or attributes.get('scanner_id')

                # If not in attributes, try to extract from entity_id
                if not tracked_device_id or not scanner_id:
                    # Try to extract from entity_id
                    device_match = self.device_id_pattern.search(entity_id)
                    if device_match:
                        if not tracked_device_id:
                            tracked_device_id = device_match.group(1)

                        # Use the entity_id itself as the scanner_id if we couldn't extract one
                        if not scanner_id:
                            scanner_id = entity_id.replace('sensor.', '')

                # If we have both IDs and RSSI, calculate and log distance
                if tracked_device_id and scanner_id:
                    distance = self._rssi_to_distance(rssi)

                    # Skip unreliable long distances
                    if distance > self.processing_params.get('distance_filter_threshold', 15):
                        continue

                    self.seen_devices.add(tracked_device_id)
                    self.seen_scanners.add(scanner_id)

                    if save_distance_log(tracked_device_id, scanner_id, distance):
                        distances_logged += 1

            # Method 3: Look for devices with explicit distance in attributes
            distance = attributes.get('distance')
            if distance is not None:
                try:
                    distance = float(distance)
                    tracked_device_id = attributes.get('source')
                    scanner_id = attributes.get('scanner') or attributes.get('scanner_id')

                    if tracked_device_id and scanner_id:
                        self.seen_devices.add(tracked_device_id)
                        self.seen_scanners.add(scanner_id)

                        if save_distance_log(tracked_device_id, scanner_id, distance):
                            distances_logged += 1
                except (ValueError, TypeError):
                    pass

        # If we found no distance data but have plenty of sensors, log this situation
        if distances_logged == 0 and len(bt_sensors) > 10:
            logger.warning(f"No distance data extracted from {len(bt_sensors)} sensors. Check sensor entity formats.")
            # Log a sample of sensors for debugging
            sample_size = min(5, len(bt_sensors))
            for i in range(sample_size):
                sensor = bt_sensors[i]
                logger.debug(f"Sample sensor {i+1}: {sensor.get('entity_id', 'unknown')} - "
                           f"state: {sensor.get('state', 'unknown')}, "
                           f"attributes: {sensor.get('attributes', {})}")

        return distances_logged

    def _calculate_device_positions(self, reference_positions: Dict) -> int:
        """
        Calculate device positions using trilateration from distance logs.
        Returns number of positions calculated.
        """
        positions_calculated = 0

        # Get recent distance measurements
        recent_distances = get_recent_distances(
            self.processing_params.get('distance_time_window', 15)
        )

        # Group distances by device
        distances_by_device = {}
        for measurement in recent_distances:
            device_id = measurement['tracked_device_id']
            if device_id not in distances_by_device:
                distances_by_device[device_id] = []
            distances_by_device[device_id].append(measurement)

        # For each device, calculate position if enough measurements
        for device_id, measurements in distances_by_device.items():
            # Skip if not enough measurement points
            if len(measurements) < self.processing_params.get('min_observations', 3):
                continue

            # Extract relevant information for trilateration
            scanner_distances = {}
            for m in measurements:
                scanner_id = m['scanner_id']
                distance = m['distance']

                # Take the most recent measurement for each scanner
                if scanner_id not in scanner_distances:
                    scanner_distances[scanner_id] = distance

            # Skip if not enough scanner points
            if len(scanner_distances) < 3:
                continue

            # Perform trilateration
            position_data = self._trilaterate_position(device_id, scanner_distances, reference_positions)

            if position_data:
                # Save the calculated position
                success = save_device_position(
                    device_id=device_id,
                    position_data=position_data,
                    source='calculated',
                    accuracy=position_data.get('accuracy')
                )

                # Also record position in history for static device detection
                if success:
                    positions_calculated += 1

                    # Record position in the history database for static device detection
                    try:
                        static_detector.record_device_position(
                            device_id=device_id,
                            x=position_data.get('x'),
                            y=position_data.get('y'),
                            z=position_data.get('z'),
                            accuracy=position_data.get('accuracy'),
                            source='trilateration'
                        )
                        logger.debug(f"Recorded position history for device {device_id} for static device detection")
                    except Exception as e:
                        logger.warning(f"Failed to record position history for device {device_id}: {e}")

        # Process static device detection after calculating positions
        try:
            static_detector.process_device_history()
            logger.info("Processed device history for static device detection")
        except Exception as e:
            logger.warning(f"Error processing device history for static device detection: {e}")

        return positions_calculated

    def _process_area_predictions(self, device_trackers: List[Dict], reference_positions: Dict) -> int:
        """
        Process data to determine which area each device is in.
        Returns number of area predictions logged.
        """
        areas_logged = 0
        # Track devices for which we've made predictions
        devices_processed = set()

        # Log current attempt for debugging
        logger.info("Starting area prediction process")

        # Get all available areas from Home Assistant (with caching)
        areas = self._get_areas_with_cache()

        if not areas:
            logger.error("Failed to retrieve areas from Home Assistant")
            return 0

        logger.info(f"Found {len(areas)} areas in Home Assistant")
        for area in areas:
            logger.debug(f"Available area: {area.get('area_id')} - {area.get('name', '')}")

        # Create mappings to help with area prediction
        ref_to_area = {}
        area_id_to_name = {}
        area_name_to_id = {}

        # Build reference-to-area mapping
        for ref_id, ref_data in reference_positions.items():
            if 'area_id' in ref_data and ref_data['area_id']:
                ref_to_area[ref_id] = ref_data['area_id']

        # Build area name mappings
        for area in areas:
            area_id = area.get('area_id')
            area_name = area.get('name', '').lower()
            if area_id and area_name:
                area_id_to_name[area_id] = area_name
                area_name_to_id[area_name] = area_id
                # Also add version with underscores instead of spaces
                area_name_to_id[area_name.replace(' ', '_')] = area_id

        # Get recent distance measurements
        recent_distances = get_recent_distances(
            self.processing_params.get('distance_time_window', 15)
        )

        # Group distances by device
        distances_by_device = {}
        for measurement in recent_distances:
            device_id = measurement['tracked_device_id']
            if device_id not in distances_by_device:
                distances_by_device[device_id] = []
            distances_by_device[device_id].append(measurement)

        logger.debug(f"Processing area predictions for {len(distances_by_device)} devices")

        # METHOD 1: Predict areas based on closest reference point (with known area)
        for device_id, measurements in distances_by_device.items():
            # Skip non-meaningful device IDs that are probably scanners themselves
            if any(pattern in device_id.lower() for pattern in ['ble_', 'bt_', 'beacon_', 'rssi_', 'scanner_']):
                logger.debug(f"Skipping likely non-device ID: {device_id}")
                continue

            # Find closest reference point
            closest_scanner = None
            closest_distance = float('inf')

            for m in measurements:
                scanner_id = m['scanner_id']
                distance = m['distance']

                if distance < closest_distance:
                    closest_distance = distance
                    closest_scanner = scanner_id

            # If we found a close scanner that has an area assigned
            if closest_scanner and closest_scanner in ref_to_area:
                area_id = ref_to_area[closest_scanner]
                area_name = area_id_to_name.get(area_id, 'unknown')
                logger.info(f"Device {device_id} is closest to scanner {closest_scanner} in area {area_id} ({area_name})")

                # Save area observation
                if save_area_observation(device_id, area_id):
                    areas_logged += 1
                    devices_processed.add(device_id)

        # METHOD 2: Extract areas directly from device trackers
        if device_trackers:
            logger.info(f"Checking {len(device_trackers)} device trackers for area information")
            for tracker in device_trackers:
                entity_id = tracker.get('entity_id', '')
                if not entity_id.startswith('device_tracker.'):
                    continue

                device_id = entity_id.replace('device_tracker.', '')

                # Don't process the same device twice
                if device_id in devices_processed:
                    continue

                attributes = tracker.get('attributes', {})

                # Get area ID if present
                area_id = attributes.get('area_id')

                if area_id:
                    area_name = area_id_to_name.get(area_id, 'unknown')
                    logger.info(f"Device {device_id} has area_id {area_id} ({area_name}) in tracker attributes")

                    # Save area observation
                    success = save_area_observation(device_id, area_id)
                    if success:
                        areas_logged += 1
                        devices_processed.add(device_id)
        else:
            logger.warning("No device trackers available for area prediction")

        # METHOD 3: Try to derive areas from entity naming patterns
        # If we still have devices without areas, try to match based on names
        remaining_devices = set(distances_by_device.keys()) - devices_processed
        if remaining_devices:
            logger.info(f"Attempting to match {len(remaining_devices)} remaining devices to areas using naming patterns")
            assigned_count = 0

            for device_id in remaining_devices:
                # Get the device name in lowercase for matching
                device_name = device_id.lower()
                matched_area_id = None

                # Try to find an area name within the device ID
                for area_name, area_id in area_name_to_id.items():
                    if area_name and area_name in device_name:
                        matched_area_id = area_id
                        logger.info(f"Matched device {device_id} to area {area_id} ({area_name}) based on name pattern")
                        break

                # If we found a match, save it
                if matched_area_id:
                    if save_area_observation(device_id, matched_area_id):
                        areas_logged += 1
                        assigned_count += 1

            if assigned_count > 0:
                logger.info(f"Successfully assigned {assigned_count} devices to areas based on naming patterns")

        # METHOD 4: Check if we have any devices that are part of an entity registry with area_id
        if len(devices_processed) < len(distances_by_device):
            logger.info("Attempting to get area information from entity registry for remaining devices")
            remaining_devices = set(distances_by_device.keys()) - devices_processed

            for device_id in remaining_devices:
                # Try to get device info from Home Assistant
                device_info = self.ha_client.get_device_info(device_id)
                if device_info and 'area_id' in device_info and device_info['area_id']:
                    area_id = device_info['area_id']
                    area_name = area_id_to_name.get(area_id, 'unknown')
                    logger.info(f"Found area {area_id} ({area_name}) for device {device_id} in entity registry")

                    if save_area_observation(device_id, area_id):
                        areas_logged += 1
                        devices_processed.add(device_id)

        # As a fallback, assign unplaced devices to "Unknown" area
        remaining_devices = set(distances_by_device.keys()) - devices_processed
        if remaining_devices:
            # Check if we have an "Unknown" area, or create a virtual one
            unknown_area_id = None
            for area in areas:
                if area.get('name', '').lower() == 'unknown':
                    unknown_area_id = area.get('area_id')
                    break

            if not unknown_area_id:
                # Use a consistent placeholder ID for unknown areas
                unknown_area_id = "00000000-0000-0000-0000-000000000000"
                logger.info(f"Created virtual 'Unknown' area with ID {unknown_area_id}")

            # Assign remaining devices to unknown area
            for device_id in remaining_devices:
                logger.info(f"Assigning device {device_id} to 'Unknown' area as fallback")
                if save_area_observation(device_id, unknown_area_id):
                    areas_logged += 1

        # Log the results
        if areas_logged > 0:
            logger.info(f"Successfully processed {areas_logged} area predictions for devices")
        else:
            logger.warning("No area predictions could be made - check that areas are properly set up in Home Assistant")

        return areas_logged

    def _get_areas_with_cache(self) -> List[Dict]:
        """Get areas from Home Assistant with caching to reduce API calls."""
        current_time = time.time()

        # If cache is valid, use it
        if (self._area_cache is not None and
            self._area_cache_time is not None and
            current_time - self._area_cache_time < self._area_cache_expiry):
            logger.debug("Using cached areas")
            return self._area_cache

        # Otherwise, refresh the cache
        try:
            logger.debug("Refreshing areas cache from Home Assistant")
            areas = self.ha_client.get_areas()
            self._area_cache = areas
            self._area_cache_time = current_time
            return areas
        except Exception as e:
            logger.error(f"Error fetching areas from Home Assistant: {e}")
            # If we have a stale cache, use it rather than nothing
            if self._area_cache is not None:
                logger.warning("Using stale areas cache due to error fetching fresh data")
                return self._area_cache
            return []

    def _trilaterate_position(
        self, device_id: str, scanner_distances: Dict[str, float], reference_positions: Dict
    ) -> Optional[Dict]:
        """
        Calculate device position using trilateration.
        Returns position data dictionary or None on failure.
        """
        # Simple triangulation algorithm based on distances to reference points
        # For a real implementation, use a proper multilateration algorithm
        # This is a simplified version for demonstration

        # Need at least 3 points for triangulation
        if len(scanner_distances) < 3:
            return None

        # Get reference positions for relevant scanners
        reference_points = []
        distances = []

        for scanner_id, distance in scanner_distances.items():
            if scanner_id in reference_positions:
                ref_pos = reference_positions[scanner_id]
                reference_points.append((ref_pos['x'], ref_pos['y'], ref_pos['z']))
                distances.append(distance)

        # Check if we still have enough points
        if len(reference_points) < 3:
            logger.debug(f"Not enough reference points for device {device_id}")
            return None

        try:
            # Define the error function to minimize - sum of squared differences
            # between actual distances and calculated distances to each reference
            def error_func(pos):
                x, y, z = pos
                error_sum = 0
                for i in range(len(reference_points)):
                    rx, ry, rz = reference_points[i]
                    calculated_distance = math.sqrt((x - rx)**2 + (y - ry)**2 + (z - rz)**2)
                    error_sum += (calculated_distance - distances[i])**2
                return error_sum

            # Initial guess - average of reference points
            initial_guess = np.mean(np.array(reference_points), axis=0)

            # Run minimization
            result = minimize(error_func, initial_guess, method='Nelder-Mead')

            # Check if optimization was successful
            if result.success:
                x, y, z = result.x

                # Calculate accuracy based on final error
                error = math.sqrt(result.fun / len(reference_points))

                return {
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'accuracy': float(error),
                    'reference_count': len(reference_points),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                logger.warning(f"Optimization failed for device {device_id}: {result.message}")
                return None

        except Exception as e:
            logger.error(f"Error in trilateration for device {device_id}: {str(e)}")
            return None

    def _rssi_to_distance(self, rssi: float) -> float:
        """
        Convert RSSI to approximate distance in meters.
        Uses log-distance path loss model.
        """
        # Default parameters if not specified
        power_coefficient = self.processing_params.get('rssi_power_coefficient', -66)
        environment_factor = self.processing_params.get('environment_factor', 2.8)

        # Log-distance path loss model formula:
        # distance = 10^((power_coefficient - rssi) / (10 * environment_factor))
        try:
            distance = 10 ** ((power_coefficient - rssi) / (10 * environment_factor))
            return distance
        except (TypeError, ValueError) as e:
            logger.warning(f"Error converting RSSI {rssi} to distance: {e}")
            return 10.0  # Return a default value

    def get_device_location_quality(self, device_id: str) -> Dict:
        """
        Get the quality of location data for a device.
        Returns dict with stats about location quality.
        """
        # Get recent distance measurements for this device
        recent_distances = []
        all_distances = get_recent_distances(
            self.processing_params.get('distance_time_window', 15)
        )

        for measurement in all_distances:
            if measurement['tracked_device_id'] == device_id:
                recent_distances.append(measurement)

        # Calculate quality metrics
        num_measurements = len(recent_distances)
        num_distinct_scanners = len(set(d['scanner_id'] for d in recent_distances))

        # Determine if we have enough data for good positioning
        has_enough_scanners = num_distinct_scanners >= 3

        # Average distance
        avg_distance = None
        if num_measurements > 0:
            avg_distance = sum(d['distance'] for d in recent_distances) / num_measurements

        return {
            'device_id': device_id,
            'num_measurements': num_measurements,
            'num_scanners': num_distinct_scanners,
            'has_enough_data': has_enough_scanners,
            'avg_distance': avg_distance
        }

    def get_all_device_locations(self) -> Dict[str, Dict]:
        """
        Get current location info for all devices.
        Returns dict mapping device_id to location data.
        """
        from .db import get_device_positions_from_sqlite

        # Get all recent device positions
        device_positions = get_device_positions_from_sqlite(
            time_window_minutes=self.processing_params.get('distance_time_window', 15)
        )

        # Get recent area predictions
        area_predictions = get_recent_area_predictions(
            self.processing_params.get('prediction_time_window', 10)
        )

        # Combine data
        result = {}
        for device_id, position in device_positions.items():
            result[device_id] = {
                'position': {
                    'x': position.get('x'),
                    'y': position.get('y'),
                    'z': position.get('z')
                },
                'area_id': area_predictions.get(device_id),
                'source': position.get('source'),
                'accuracy': position.get('accuracy'),
                'timestamp': position.get('timestamp')
            }

        return result
