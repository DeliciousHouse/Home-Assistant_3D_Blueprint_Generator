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

    def log_sensor_data(self, ha_client, entity_data=None):
        """Log sensor data to the database for analysis.

        Args:
            ha_client: HomeAssistantClient instance
            entity_data: Optional entity data, if None it will be fetched from HA

        Returns:
            Dict with summary of processed data
        """
        try:
            logger.info("Logging sensor data from Home Assistant")

            # Get data if not provided
            if entity_data is None:
                entity_data = ha_client.get_sensor_entities()

            logger.info(f"Processing {len(entity_data)} entities")

            # Connect to database
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Process entities
            timestamp = datetime.now().isoformat()
            bluetooth_count = 0
            rssi_count = 0
            bermuda_count = 0

            # Process each entity
            for entity in entity_data:
                entity_id = entity.get('entity_id', '')
                if not entity_id:
                    continue

                state = entity.get('state')
                attributes = entity.get('attributes', {})
                device_id = entity.get('device_id')
                area_id = entity.get('area_id')

                # Skip unavailable or unknown states
                if not state or state in ('unavailable', 'unknown'):
                    continue

                # Extract device name
                device_name = attributes.get('friendly_name', device_id)

                # Check for RSSI attribute (common in BLE sensors)
                rssi = attributes.get('rssi')
                if rssi is not None:
                    rssi_count += 1
                    cursor.execute(
                        """INSERT INTO sensor_logs
                        (timestamp, entity_id, device_id, sensor_type, value, attributes, area_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (timestamp, entity_id, device_id, 'rssi', rssi,
                         json.dumps(attributes), area_id)
                    )

                # Check for Bermuda distance readings
                is_bermuda = False
                if 'bermuda' in entity_id and '_distance' in entity_id:
                    is_bermuda = True
                    bermuda_count += 1
                elif '_distance_' in entity_id:
                    is_bermuda = True
                    bermuda_count += 1

                if is_bermuda:
                    try:
                        distance = float(state)
                        scanner_id = attributes.get('scanner_id')
                        if not scanner_id and '_distance_' in entity_id:
                            parts = entity_id.split('_distance_')
                            if len(parts) == 2:
                                scanner_id = parts[1]

                        cursor.execute(
                            """INSERT INTO sensor_logs
                            (timestamp, entity_id, device_id, sensor_type, value, attributes,
                             scanner_id, area_id)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                            (timestamp, entity_id, device_id, 'bermuda_distance', distance,
                             json.dumps(attributes), scanner_id, area_id)
                        )
                    except (ValueError, TypeError):
                        pass

                # Generic sensor logging for all BLE sensors
                if device_id:  # Only log if we have a device_id
                    bluetooth_count += 1
                    cursor.execute(
                        """INSERT INTO sensor_logs
                        (timestamp, entity_id, device_id, sensor_type, value, attributes, area_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (timestamp, entity_id, device_id, 'bluetooth_state', state,
                         json.dumps(attributes), area_id)
                    )

            conn.commit()
            conn.close()

            logger.info(f"Logged {bluetooth_count} Bluetooth entities, {rssi_count} RSSI readings, {bermuda_count} Bermuda distances")

            return {
                "timestamp": timestamp,
                "bluetooth_count": bluetooth_count,
                "rssi_count": rssi_count,
                "bermuda_count": bermuda_count,
                "total_processed": len(entity_data)
            }

        except Exception as e:
            logger.error(f"Error logging sensor data: {e}", exc_info=True)
            return {"error": str(e)}
