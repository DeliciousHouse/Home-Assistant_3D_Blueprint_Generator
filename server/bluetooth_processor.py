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
        processed_distance_entities = 0
        processed_area_entities = 0

        try:
            # 1. Log Distance Data
            logger.debug("Fetching distance sensors from HA...")
            distance_sensors = self.ha_client.get_distance_sensors() # Assumes this HA method exists
            if distance_sensors is None:
                errors.append("Failed to fetch distance sensors from HA.")
                logger.error("get_distance_sensors returned None")
            else:
                processed_distance_entities = len(distance_sensors)
                logger.info(f"Processing {processed_distance_entities} potential distance entities.")
                for sensor in distance_sensors:
                    try:
                        # Ensure required keys exist before saving
                        dev_id = sensor.get('tracked_device_id')
                        scan_id = sensor.get('scanner_id')
                        dist = sensor.get('distance')
                        entity_id = sensor.get('entity_id', 'N/A') # Get entity_id for logging

                        # Log the data received from HA client
                        logger.debug(f"Processing distance entity: ID='{entity_id}', Device='{dev_id}', Scanner='{scan_id}', Distance='{dist}'")

                        if dev_id and scan_id and dist is not None:
                            if not save_distance_log(dev_id, scan_id, dist):
                                # Error is logged within save_distance_log now
                                errors.append(f"Save failed for distance: {entity_id}")
                            else:
                                logged_distances += 1
                        else:
                             log_msg = f"Skipping incomplete distance data: Entity='{entity_id}', Device='{dev_id}', Scanner='{scan_id}', Distance='{dist}'"
                             logger.warning(log_msg)
                             errors.append(log_msg)
                    except Exception as e:
                         err_msg = f"Error processing distance entity {sensor.get('entity_id', 'N/A')}: {type(e).__name__} - {e}"
                         logger.error(err_msg, exc_info=True)
                         errors.append(err_msg)


            # 2. Log Area Predictions
            logger.debug("Fetching area predictions from HA...")
            area_predictions = self.ha_client.get_device_area_predictions() # Assumes this HA method exists
            if area_predictions is None:
                 errors.append("Failed to fetch area predictions from HA.")
                 logger.error("get_device_area_predictions returned None")
            else:
                processed_area_entities = len(area_predictions)
                logger.info(f"Processing {processed_area_entities} area predictions.")
                for device_id, area_id in area_predictions.items():
                    try:
                        # Log the data received from HA client
                        logger.debug(f"Processing area prediction: Device='{device_id}', Area='{area_id}'")
                        if not save_area_observation(device_id, area_id): # area_id can be None
                             # Error is logged within save_area_observation now
                             errors.append(f"Save failed for area: {device_id}")
                        else:
                             logged_areas += 1
                    except Exception as e:
                         err_msg = f"Error processing area prediction for {device_id}: {type(e).__name__} - {e}"
                         logger.error(err_msg, exc_info=True)
                         errors.append(err_msg)


            if errors:
                error_summary = '; '.join(errors[:3]) + ('...' if len(errors) > 3 else '')
                logger.warning(f"Completed data logging. Processed {processed_distance_entities} distances ({logged_distances} saved), {processed_area_entities} areas ({logged_areas} saved). Errors: {len(errors)}. Summary: {error_summary}")
            else:
                 logger.info(f"Completed data logging. Processed {processed_distance_entities} distances ({logged_distances} saved), {processed_area_entities} areas ({logged_areas} saved). No errors.")

            return {"distances_logged": logged_distances, "areas_logged": logged_areas, "errors": len(errors)}

        except Exception as e:
            logger.error(f"CRITICAL error during sensor data logging: {e}", exc_info=True)
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
