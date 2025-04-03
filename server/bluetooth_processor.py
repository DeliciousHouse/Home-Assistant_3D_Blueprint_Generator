import logging
import math
from typing import Dict
from .ha_client import HomeAssistantClient
from .db import save_distance_log, save_area_observation

logger = logging.getLogger(__name__)

class BluetoothProcessor:
    """Process Bluetooth signals for position estimation."""

    def __init__(self):
        """Initialize the processor, primarily setting up the HA client."""
        self.ha_client = HomeAssistantClient()
        logger.info("BluetoothProcessor initialized (Mode: Data Logging).")

    def log_sensor_data(self) -> Dict[str, int]:
        """Fetches distance and area data from HA and logs it to the database."""
        logged_distances = 0
        logged_areas = 0
        errors = []

        try:
            # 1. Log Distance Data using ha_client.get_distance_sensors()
            logger.debug("Fetching distance sensors...")
            distance_sensors = self.ha_client.get_distance_sensors()
            if distance_sensors is None:
                errors.append("Failed to fetch distance sensors from HA.")
            else:
                logger.debug(f"Processing {len(distance_sensors)} distance sensors.")
                for sensor in distance_sensors:
                    try:
                        dev_id = sensor.get('tracked_device_id')
                        scan_id = sensor.get('scanner_id')
                        dist = sensor.get('distance')

                        # Validate data before saving
                        if dev_id and scan_id and isinstance(dist, (int, float)) and not math.isnan(dist) and dist >= 0:
                            if save_distance_log(dev_id, scan_id, dist):
                                logged_distances += 1
                            else:
                                errors.append(f"Failed DB write for distance: {sensor.get('entity_id', 'N/A')}")
                        else:
                             # Log only if data is actually invalid, not just missing keys sometimes
                             if not (dev_id and scan_id and dist is not None):
                                 errors.append(f"Incomplete/Invalid distance data: {sensor.get('entity_id', 'N/A')} - Data: {sensor}")

                    except Exception as e:
                         errors.append(f"Error logging distance {sensor.get('entity_id', 'N/A')}: {e}")

            # 2. Log Area Predictions using ha_client.get_device_area_predictions()
            logger.debug("Fetching area predictions...")
            area_predictions = self.ha_client.get_device_area_predictions()
            if area_predictions is None:
                 errors.append("Failed to fetch area predictions from HA.")
            else:
                logger.debug(f"Processing {len(area_predictions)} area predictions.")
                for device_id, area_id in area_predictions.items():
                    try:
                        if save_area_observation(device_id, area_id):
                            logged_areas += 1
                        else:
                             errors.append(f"Failed DB write for area: {device_id}")
                    except Exception as e:
                         errors.append(f"Error logging area {device_id}: {e}")

            if errors:
                error_summary = '; '.join(errors[:min(len(errors), 5)]) + ('...' if len(errors) > 5 else '')
                logger.warning(f"Completed data logging with {len(errors)} errors: {error_summary}")

            logger.info(f"Logged {logged_distances} distance readings and {logged_areas} area observations.")
            return {"distances_logged": logged_distances, "areas_logged": logged_areas, "errors": len(errors)}

        except Exception as e:
            logger.error(f"Critical error during sensor data logging: {e}", exc_info=True)
            return {"distances_logged": logged_distances, "areas_logged": logged_areas, "error": str(e), "errors": len(errors)+1}
