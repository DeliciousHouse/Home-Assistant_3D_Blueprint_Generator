import json
import logging
import math
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import uuid

# Import database functions
from .db import (
    save_distance_log,
    get_recent_distances,
    save_area_observation,
    get_device_positions_from_sqlite,
    get_reference_positions_from_sqlite
)

# Import Home Assistant client
try:
    from .ha_client import HAClient
except ImportError:
    # For standalone testing
    class HAClient:
        def __init__(self):
            pass
        def get_distances(self):
            return {}
        def get_area_predictions(self):
            return {}
        def get_areas(self):
            return []

# Import config loader
try:
    from .config_loader import load_config
except ImportError:
    def load_config(): return {}

logger = logging.getLogger(__name__)
config = load_config()

class BluetoothProcessor:
    """Processes Bluetooth data for the blueprint generator."""

    def __init__(self):
        """Initialize the Bluetooth processor."""
        self.ha_client = HAClient()
        self.device_predictions = {}  # Cache of most recent area predictions
        self.scanner_positions = {}   # Cache of scanner positions
        self.last_update_time = datetime.now() - timedelta(minutes=10)  # Force initial update

        # Get settings from config
        self.settings = config.get('bluetooth_processor', {})
        self.update_interval = self.settings.get('update_interval', 10)  # seconds
        self.distance_max = self.settings.get('distance_max', 15.0)  # meters
        self.rssi_cutoff = self.settings.get('rssi_cutoff', -90)  # dBm

    def log_sensor_data(self) -> Dict[str, int]:
        """
        Log the latest sensor data from Home Assistant.
        Returns a dictionary with counts of logged distances and areas.
        """
        result = {"distances_logged": 0, "areas_logged": 0}

        try:
            # Check if we need to update based on interval
            now = datetime.now()
            if (now - self.last_update_time).total_seconds() < self.update_interval:
                logger.debug("Skipping update - too soon since last update")
                return result

            self.last_update_time = now

            # Get distance data from Home Assistant
            distances = self.ha_client.get_distances()
            if not distances:
                logger.warning("No distance data available from HA")
            else:
                # Log distance data to the database
                for device_id, distance_data in distances.items():
                    for scanner, distance_info in distance_data.items():
                        distance = distance_info.get('distance')
                        if distance is not None and distance <= self.distance_max:
                            if save_distance_log(device_id, scanner, distance):
                                result["distances_logged"] += 1

            # Get area predictions from Home Assistant
            area_predictions = self.ha_client.get_area_predictions()
            if not area_predictions:
                logger.warning("No area predictions available from HA")
            else:
                # Log area predictions to the database
                for device_id, area_id in area_predictions.items():
                    if save_area_observation(device_id, area_id):
                        result["areas_logged"] += 1
                        # Update the cache
                        self.device_predictions[device_id] = area_id

            logger.info(f"Logged {result['distances_logged']} distance readings and {result['areas_logged']} area observations.")
            return result

        except Exception as e:
            logger.error(f"Error logging sensor data: {str(e)}", exc_info=True)
            return result

    def process_distances(self, time_window_minutes: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Process recent distance measurements to create a device-to-device distance matrix.

        Args:
            time_window_minutes: Time window in minutes for recent distances

        Returns:
            Dictionary mapping device pairs to distances
        """
        try:
            # Get recent distance measurements from the database
            recent_distances = get_recent_distances(time_window_minutes)
            if not recent_distances:
                logger.warning(f"No distance data found in the last {time_window_minutes} minutes")
                return {}

            # Create a matrix of distances between devices
            device_distances = {}

            # Process each distance measurement
            for record in recent_distances:
                device_id = record.get('tracked_device_id')
                scanner_id = record.get('scanner_id')
                distance = record.get('distance')

                if not device_id or not scanner_id or distance is None:
                    continue

                # Create unique key for the device pair (sorted to ensure consistency)
                pair_key = tuple(sorted([device_id, scanner_id]))

                # Store the minimum distance observed (more reliable than averaging)
                if pair_key not in device_distances or distance < device_distances[pair_key]:
                    device_distances[pair_key] = distance

            logger.debug(f"Processed {len(recent_distances)} distance records into {len(device_distances)} unique device pairs")
            return device_distances

        except Exception as e:
            logger.error(f"Error processing distances: {str(e)}", exc_info=True)
            return {}

    def get_area_predictions_for_devices(self, device_ids: List[str], time_window_minutes: int = 10) -> Dict[str, str]:
        """
        Get the most recent area predictions for the specified devices.

        Args:
            device_ids: List of device IDs to get predictions for
            time_window_minutes: Time window in minutes for recent predictions

        Returns:
            Dictionary mapping device_id to area_id
        """
        try:
            # First, check our cached predictions
            predictions = {d: self.device_predictions.get(d) for d in device_ids if d in self.device_predictions}

            # For devices without cached predictions, get fresh data from HA
            missing_devices = [d for d in device_ids if d not in predictions]
            if missing_devices:
                fresh_predictions = self.ha_client.get_area_predictions()
                for device_id in missing_devices:
                    area_id = fresh_predictions.get(device_id)
                    if area_id:
                        predictions[device_id] = area_id
                        # Update our cache
                        self.device_predictions[device_id] = area_id

            return predictions

        except Exception as e:
            logger.error(f"Error getting area predictions: {str(e)}", exc_info=True)
            return {}

    def get_all_ha_areas(self) -> List[Dict[str, Any]]:
        """
        Get all areas defined in Home Assistant.

        Returns:
            List of area dictionaries with 'area_id' and 'name' keys
        """
        try:
            return self.ha_client.get_areas() or []
        except Exception as e:
            logger.error(f"Error getting Home Assistant areas: {str(e)}", exc_info=True)
            return []

    def get_reference_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get reference positions for devices (scanners or fixed beacons).

        Returns:
            Dictionary mapping device_id to position data
        """
        try:
            # Get reference positions from the database
            reference_positions = get_reference_positions_from_sqlite()

            if not reference_positions:
                logger.warning("No reference positions found in the database")

            return reference_positions

        except Exception as e:
            logger.error(f"Error getting reference positions: {str(e)}", exc_info=True)
            return {}

    def create_distance_matrix(self, device_distances: Dict[Tuple[str, str], float]) -> Tuple[np.ndarray, List[str]]:
        """
        Create a symmetric distance matrix from device-to-device distances.

        Args:
            device_distances: Dictionary mapping device pairs to distances

        Returns:
            Tuple of (distance matrix, list of device IDs)
        """
        try:
            # Get all unique device IDs from the distance data
            unique_devices = set()
            for device1, device2 in device_distances.keys():
                unique_devices.add(device1)
                unique_devices.add(device2)

            # Convert to sorted list for consistent indexing
            device_list = sorted(unique_devices)
            n_devices = len(device_list)

            if n_devices < 2:
                logger.warning("Not enough devices for distance matrix")
                return np.array([]), []

            # Create a mapping from device ID to matrix index
            device_to_idx = {device: idx for idx, device in enumerate(device_list)}

            # Initialize distance matrix with large values
            # Use a reasonable maximum distance (e.g., 30 meters)
            max_distance = 30.0
            dist_matrix = np.ones((n_devices, n_devices)) * max_distance

            # Set diagonal to zero (distance to self)
            np.fill_diagonal(dist_matrix, 0)

            # Fill in known distances
            for (device1, device2), distance in device_distances.items():
                idx1 = device_to_idx[device1]
                idx2 = device_to_idx[device2]
                # Ensure symmetric matrix
                dist_matrix[idx1, idx2] = distance
                dist_matrix[idx2, idx1] = distance

            logger.info(f"Created distance matrix for {n_devices} devices")
            return dist_matrix, device_list

        except Exception as e:
            logger.error(f"Error creating distance matrix: {str(e)}", exc_info=True)
            return np.array([]), []

    def calibrate_rssi_model(self, rssi_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calibrate an RSSI-to-distance model.

        Args:
            rssi_samples: List of dictionaries with 'rssi' and 'distance' values

        Returns:
            Dictionary with model parameters
        """
        try:
            if len(rssi_samples) < 5:
                logger.warning("Not enough RSSI samples for calibration")
                return {}

            # Extract RSSI and distance values
            rssi_values = np.array([sample['rssi'] for sample in rssi_samples])
            distance_values = np.array([sample['distance'] for sample in rssi_samples])

            # Log-distance path loss model: RSSI = A - 10*n*log10(d)
            # where A is the RSSI at 1m, n is the path loss exponent

            # Convert to log scale for linear regression
            log_distances = np.log10(distance_values)

            # Perform linear regression
            A = np.vstack([np.ones_like(log_distances), log_distances]).T
            # RSSI = a - b*log10(d)
            a, b = np.linalg.lstsq(A, rssi_values, rcond=None)[0]

            # Convert parameters to standard form
            rssi_at_1m = a
            path_loss_exponent = b / 10.0

            # Calculate R-squared to evaluate fit
            y_pred = a - b * log_distances
            ss_total = np.sum((rssi_values - np.mean(rssi_values)) ** 2)
            ss_residual = np.sum((rssi_values - y_pred) ** 2)
            r_squared = 1 - (ss_residual / ss_total)

            logger.info(f"Calibrated RSSI model: RSSI_1m={rssi_at_1m:.2f}, n={path_loss_exponent:.2f}, R²={r_squared:.3f}")

            return {
                'rssi_at_1m': float(rssi_at_1m),
                'path_loss_exponent': float(path_loss_exponent),
                'r_squared': float(r_squared)
            }

        except Exception as e:
            logger.error(f"Error calibrating RSSI model: {str(e)}", exc_info=True)
            return {}

    def rssi_to_distance(self, rssi: float, rssi_at_1m: float = -59, path_loss_exponent: float = 2.0) -> float:
        """
        Convert RSSI to distance using the log-distance path loss model.

        Args:
            rssi: Measured RSSI value in dBm
            rssi_at_1m: RSSI value at 1 meter distance
            path_loss_exponent: Path loss exponent (typically 2.0 to 4.0)

        Returns:
            Estimated distance in meters
        """
        try:
            # Log-distance path loss model: RSSI = RSSI_1m - 10*n*log10(d)
            # Solving for d: d = 10^((RSSI_1m - RSSI)/(10*n))
            if path_loss_exponent <= 0:
                path_loss_exponent = 2.0  # Default to free space

            distance = 10 ** ((rssi_at_1m - rssi) / (10 * path_loss_exponent))

            # Apply reasonable bounds
            min_distance = 0.1  # 10 cm minimum
            max_distance = 30.0  # 30 meters maximum
            distance = min(max(distance, min_distance), max_distance)

            return distance

        except Exception as e:
            logger.error(f"Error converting RSSI to distance: {str(e)}")
            return 5.0  # Return a reasonable default
