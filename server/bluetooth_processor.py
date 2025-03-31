import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from .db import save_distance_log, save_area_observation
from .ha_client import HomeAssistantClient

logger = logging.getLogger(__name__)

class BluetoothProcessor:
    """
    Logs raw Bluetooth sensor data (distances, area predictions) from Home Assistant
    to the database for later processing by the BlueprintGenerator.
    (Formerly handled positioning logic).
    """

    def __init__(self, config_path=None):
        """Initialize the Bluetooth processor.

        Args:
            config_path: Optional path to configuration file. If None, will load default config.
        """
        # Import and load configuration (keep for potential future use)
        from .config_loader import load_config
        self.config = load_config(config_path)

        # Initialize HA client
        self.ha_client = HomeAssistantClient()

        logger.info("BluetoothProcessor initialized (Mode: Data Logging).")

    def log_sensor_data(self) -> Dict[str, int]:
        """Fetches distance and area data from HA and logs it to the database."""
        logged_distances = 0
        logged_areas = 0
        errors = []

        try:
            # 1. Log Distance Data
            distance_sensors = self.ha_client.get_distance_sensors()
            if distance_sensors is None:
                errors.append("Failed to fetch distance sensors from HA.")
            else:
                for sensor in distance_sensors:
                    try:
                        # Ensure required keys exist before saving
                        dev_id = sensor.get('tracked_device_id')
                        scan_id = sensor.get('scanner_id')
                        dist = sensor.get('distance')
                        if dev_id and scan_id and dist is not None:
                            if save_distance_log(dev_id, scan_id, dist):
                                logged_distances += 1
                            else:
                                errors.append(f"Failed DB write for distance: {sensor.get('entity_id', 'N/A')}")
                        else:
                            errors.append(f"Incomplete distance data: {sensor.get('entity_id', 'N/A')}")
                    except Exception as e:
                        errors.append(f"Error logging distance {sensor.get('entity_id', 'N/A')}: {e}")

            # 2. Log Area Predictions
            area_predictions = self.ha_client.get_device_area_predictions()
            if area_predictions is None:
                errors.append("Failed to fetch area predictions from HA.")
            else:
                for device_id, area_id in area_predictions.items():
                    try:
                        if save_area_observation(device_id, area_id):  # area_id can be None
                            logged_areas += 1
                        else:
                            errors.append(f"Failed DB write for area: {device_id}")
                    except Exception as e:
                        errors.append(f"Error logging area {device_id}: {e}")

            if errors:
                # Log only a summary of errors if there are many
                error_summary = '; '.join(errors[:3]) + ('...' if len(errors) > 3 else '')
                logger.warning(f"Completed data logging with {len(errors)} errors: {error_summary}")

            logger.info(f"Logged {logged_distances} distance readings and {logged_areas} area observations.")
            return {"distances_logged": logged_distances, "areas_logged": logged_areas, "errors": len(errors)}

        except Exception as e:
            logger.error(f"Critical error during sensor data logging: {e}", exc_info=True)
            # Return counts accumulated so far, plus error
            return {"distances_logged": logged_distances, "areas_logged": logged_areas, "error": str(e), "errors": len(errors)+1}

    def process_sensor_data(self):
        """Legacy compatibility method - now logs data and returns area predictions.

        Returns:
            Dict of device IDs to area IDs for compatibility with existing blueprint generator code.
        """
        try:
            # Log all sensor data first
            self.log_sensor_data()

            # Return just the area predictions for backward compatibility
            area_predictions = self.ha_client.get_device_area_predictions() or {}
            logger.info(f"Processed sensor data and retrieved {len(area_predictions)} area predictions")
            return area_predictions

        except Exception as e:
            logger.error(f"Error in process_sensor_data: {e}", exc_info=True)
            return {}

    def process_bluetooth_sensors(self):
        """Process Bluetooth sensor data from Home Assistant."""
        try:
            logger.debug("Processing Bluetooth sensors data")
            # Use the existing method that already handles everything
            result = self.log_sensor_data()
            # Check if there was a critical error
            if "error" in result and result["error"]:
                return False
            return True
        except Exception as e:
            logger.error(f"Error processing Bluetooth sensors: {e}", exc_info=True)
            return False
