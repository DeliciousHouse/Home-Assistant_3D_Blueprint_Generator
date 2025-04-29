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

# Import local modules with fallbacks
try:
    # First try relative imports (typical when imported as module)
    from .ha_client import HAClient as HomeAssistantClient
    from .db import (
        save_distance_log, save_area_observation,
        get_recent_distances, get_recent_area_predictions,
        get_reference_positions_from_sqlite, save_reference_position,
        save_device_position
    )
    from .config_loader import load_config
    logger = logging.getLogger(__name__)
except ImportError:
    try:
        # Then try direct imports (when run as script)
        from ha_client import HAClient as HomeAssistantClient
        from db import (
            save_distance_log, save_area_observation,
            get_recent_distances, get_recent_area_predictions,
            get_reference_positions_from_sqlite, save_reference_position,
            save_device_position
        )
        from config_loader import load_config
        logger = logging.getLogger("bluetooth_processor")
    except ImportError as e:
        # If both fail, handle gracefully but log error
        import sys
        logging.error(f"Failed to import required modules for BluetoothProcessor: {e}")
        sys.exit(1)  # Exit if critical modules can't be loaded

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

        # Get Home Assistant client
        self.ha_client = HomeAssistantClient()

        # Get processing configuration
        self.processing_params = self.config.get('processing_params', {
            'rssi_power_coefficient': -20,
            'environment_factor': 2.0,
            'distance_filter_threshold': 15,  # meters
            'distance_time_window': 15,  # minutes
            'prediction_time_window': 10,  # minutes
            'min_observations': 3,
            'trilateration_alpha': 0.5,
            'trilateration_beta': 0.5,
            'use_kalman_filter': True,
            'kalman_process_noise': 0.01,
            'kalman_measurement_noise': 0.1
        })

        # Initialize counters and timestamps
        self.last_scan_time = datetime.now() - timedelta(hours=1)  # Set to an hour ago initially
        self.scanning_interval = self.config.get('scanning_interval', 30)  # Default 30 seconds

        # Cache of device metadata (name, type, etc.)
        self.device_metadata = {}

        self._initialized = True
        logger.info("BluetoothProcessor initialized successfully")

    def log_sensor_data(self) -> Dict:
        """
        Scan for Bluetooth devices and log data.
        Returns dictionary with log statistics.
        """
        start_time = datetime.now()
        distances_logged = 0
        areas_logged = 0

        try:
            # Check if enough time has passed since last scan
            time_since_last_scan = (start_time - self.last_scan_time).total_seconds()
            if time_since_last_scan < self.scanning_interval:
                logger.debug(f"Skipping scan, only {time_since_last_scan:.1f}s since last scan "
                           f"(interval: {self.scanning_interval}s)")
                return {
                    'status': 'skipped',
                    'reason': 'interval_not_reached',
                    'distances_logged': 0,
                    'areas_logged': 0
                }

            self.last_scan_time = start_time

            # Get Bluetooth sensors from Home Assistant
            bt_sensors = self.ha_client.get_bluetooth_sensors()
            logger.info(f"Found {len(bt_sensors)} Bluetooth sensors")

            # Get device trackers from Home Assistant
            device_trackers = self.ha_client.get_device_trackers()
            logger.info(f"Found {len(device_trackers)} device trackers")

            # Get reference positions
            reference_positions = get_reference_positions_from_sqlite()
            logger.info(f"Found {len(reference_positions)} reference positions")

            # Process Bluetooth data
            distances_logged = self._process_bluetooth_data(bt_sensors, reference_positions)

            # Process device positions
            positions_calculated = self._calculate_device_positions(reference_positions)

            # Process area predictions
            areas_logged = self._process_area_predictions(device_trackers, reference_positions)

            # Log statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Processing complete in {processing_time:.2f}s. "
                       f"Logged {distances_logged} distances and {areas_logged} area predictions. "
                       f"Calculated {positions_calculated} positions.")

            return {
                'status': 'success',
                'processing_time': processing_time,
                'distances_logged': distances_logged,
                'areas_logged': areas_logged,
                'positions_calculated': positions_calculated
            }

        except Exception as e:
            logger.error(f"Error processing Bluetooth data: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error_message': str(e),
                'distances_logged': distances_logged,
                'areas_logged': areas_logged
            }

    def _process_bluetooth_data(self, bt_sensors: List[Dict], reference_positions: Dict) -> int:
        """
        Process Bluetooth sensor data and log distances.
        Returns number of distances logged.
        """
        distances_logged = 0

        # Create a map of scanner IDs to their positions
        scanner_positions = {}
        for ref_id, ref_data in reference_positions.items():
            scanner_positions[ref_id] = (ref_data['x'], ref_data['y'], ref_data['z'])

        for sensor in bt_sensors:
            sensor_id = sensor.get('entity_id', '').replace('sensor.', '')
            tracked_device_id = sensor.get('attributes', {}).get('source', 'unknown')

            # Skip if no scanner position available
            if sensor_id not in scanner_positions:
                continue

            # Get RSSI value
            rssi = sensor.get('attributes', {}).get('rssi')
            if rssi is None:
                continue

            # Convert RSSI to distance
            distance = self._rssi_to_distance(rssi)

            # Skip unreliable long distances
            if distance > self.processing_params.get('distance_filter_threshold', 15):
                continue

            # Log the distance
            if save_distance_log(tracked_device_id, sensor_id, distance):
                distances_logged += 1

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

                if success:
                    positions_calculated += 1

        return positions_calculated

    def _process_area_predictions(self, device_trackers: List[Dict], reference_positions: Dict) -> int:
        """
        Process data to determine which area each device is in.
        Returns number of area predictions logged.
        """
        areas_logged = 0

        # Create a mapping of reference_id to area_id
        ref_to_area = {}
        for ref_id, ref_data in reference_positions.items():
            if 'area_id' in ref_data:
                ref_to_area[ref_id] = ref_data['area_id']

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

        # For each device, predict area
        for device_id, measurements in distances_by_device.items():
            # Find closest reference point
            closest_scanner = None
            closest_distance = float('inf')

            for m in measurements:
                scanner_id = m['scanner_id']
                distance = m['distance']

                if distance < closest_distance:
                    closest_distance = distance
                    closest_scanner = scanner_id

            if closest_scanner and closest_scanner in ref_to_area:
                area_id = ref_to_area[closest_scanner]

                # Save area observation
                success = save_area_observation(device_id, area_id)
                if success:
                    areas_logged += 1

        return areas_logged

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
        power_coefficient = self.processing_params.get('rssi_power_coefficient', -20)
        environment_factor = self.processing_params.get('environment_factor', 2.0)

        # Log-distance path loss model formula:
        # distance = 10^((power_coefficient - rssi) / (10 * environment_factor))
        distance = 10 ** ((power_coefficient - rssi) / (10 * environment_factor))

        return distance

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
