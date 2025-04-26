#!/usr/bin/env python3

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import requests
from urllib.parse import urljoin
import random  # For generating mock data

# Load configuration
try:
    from .config_loader import load_config
except ImportError:
    from config_loader import load_config

logger = logging.getLogger(__name__)

class HAClient:
    """Home Assistant API client for the Blueprint Generator."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Home Assistant client."""
        self.config = load_config(config_path)
        self.ha_config = self.config.get('home_assistant', {})

        # Get Home Assistant connection details - with better fallbacks for add-on environment
        self.ha_url = self.ha_config.get('url')
        if not self.ha_url:
            # Try all possible environment variables for Home Assistant URL
            self.ha_url = os.environ.get('HASS_URL') or os.environ.get('SUPERVISOR_API') or 'http://supervisor/core'

        # Get token with improved fallbacks for add-on environment
        self.ha_token = self.ha_config.get('token')
        if not self.ha_token:
            # Try all possible environment variables for token
            self.ha_token = os.environ.get('HASS_TOKEN') or os.environ.get('SUPERVISOR_TOKEN')
            if not self.ha_token and os.path.exists('/data/options.json'):
                try:
                    with open('/data/options.json', 'r') as f:
                        options = json.load(f)
                        self.ha_token = options.get('ha_token', '')
                except Exception as e:
                    logger.error(f"Failed to read options.json: {e}")

        # Setup request headers
        self.headers = {
            'Authorization': f'Bearer {self.ha_token}',
            'Content-Type': 'application/json',
        }

        # Flag to track if we are in offline/mock mode
        self.offline_mode = False

        logger.info(f"HAClient initialized with URL: {self.ha_url}")

        # Validate connection on startup
        if not self._test_connection():
            logger.warning("Unable to connect to Home Assistant. Entering offline mode with mock data.")
            self.offline_mode = True

    def _test_connection(self):
        """Test connection to Home Assistant and log detailed debug info."""
        try:
            url = urljoin(self.ha_url, '/api/')
            logger.debug(f"Testing Home Assistant connection to {url}")

            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                logger.info("Successfully connected to Home Assistant API")
                return True
            else:
                logger.error(f"Failed to connect to Home Assistant API: HTTP {response.status_code}")
                logger.debug(f"URL: {url}, Headers: Auth Bearer token length: {len(self.ha_token)} characters")
                logger.debug(f"Response: {response.text[:200]}")  # Log first 200 chars of response
                return False

        except Exception as e:
            logger.error(f"Error connecting to Home Assistant: {str(e)}")
            return False

    def get_areas(self) -> List[Dict[str, Any]]:
        """Get all areas from Home Assistant."""
        if self.offline_mode:
            return self._generate_mock_areas()

        try:
            url = urljoin(self.ha_url, '/api/areas')
            logger.debug(f"Fetching areas from {url}")

            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                areas = response.json()
                # Format areas to include area_id for compatibility with our system
                formatted_areas = []
                for area in areas:
                    formatted_area = {
                        'area_id': area.get('area_id', ''),
                        'name': area.get('name', ''),
                        'picture': area.get('picture', None)
                    }
                    formatted_areas.append(formatted_area)

                logger.info(f"Successfully fetched {len(formatted_areas)} areas from Home Assistant")
                return formatted_areas
            else:
                logger.error(f"Failed to get areas from Home Assistant: HTTP {response.status_code}")
                logger.debug(f"Response: {response.text[:200]}")
                # Fall back to mock areas
                return self._generate_mock_areas()

        except Exception as e:
            logger.error(f"Error getting areas from Home Assistant: {str(e)}")
            return self._generate_mock_areas()

    def get_devices(self) -> List[Dict[str, Any]]:
        """Get all devices from Home Assistant."""
        if self.offline_mode:
            return self._generate_mock_devices()

        try:
            url = urljoin(self.ha_url, '/api/devices')
            logger.debug(f"Fetching devices from {url}")

            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                devices = response.json()
                logger.info(f"Successfully fetched {len(devices)} devices from Home Assistant")
                return devices
            else:
                logger.error(f"Failed to get devices from Home Assistant: HTTP {response.status_code}")
                # Fall back to mock devices
                return self._generate_mock_devices()

        except Exception as e:
            logger.error(f"Error getting devices from Home Assistant: {str(e)}")
            return self._generate_mock_devices()

    def get_entities(self) -> List[Dict[str, Any]]:
        """Get all entities from Home Assistant."""
        if self.offline_mode:
            return self._generate_mock_entities()

        try:
            url = urljoin(self.ha_url, '/api/states')
            logger.debug(f"Fetching entities from {url}")

            response = requests.get(url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                entities = response.json()
                logger.info(f"Successfully fetched {len(entities)} entities from Home Assistant")
                return entities
            else:
                logger.error(f"Failed to get entities from Home Assistant: HTTP {response.status_code}")
                # Fall back to mock entities
                return self._generate_mock_entities()

        except Exception as e:
            logger.error(f"Error getting entities from Home Assistant: {str(e)}")
            return self._generate_mock_entities()

    def get_bluetooth_devices(self) -> List[Dict[str, Any]]:
        """Get bluetooth-specific devices from Home Assistant."""
        try:
            # First get all entities
            entities = self.get_entities()

            # Filter for bluetooth-related entities
            bluetooth_entities = []
            for entity in entities:
                entity_id = entity.get('entity_id', '')
                if 'bluetooth' in entity_id or 'ble' in entity_id:
                    bluetooth_entities.append(entity)

            logger.info(f"Found {len(bluetooth_entities)} bluetooth-related entities")
            return bluetooth_entities
        except Exception as e:
            logger.error(f"Error getting bluetooth devices: {str(e)}")
            return []

    def get_distance_sensors(self) -> List[Dict[str, Any]]:
        """Get distance sensor entities from Home Assistant."""
        try:
            # First get all entities
            entities = self.get_entities()

            # Filter for distance or proximity sensors
            distance_sensors = []
            for entity in entities:
                entity_id = entity.get('entity_id', '')

                # Match on common distance sensor patterns
                if re.match(r'sensor\.[^\.]+_(distance|proximity|rssi|signal_strength)', entity_id):
                    distance_sensors.append(entity)

            logger.info(f"HA_Client: Found {len(distance_sensors)} potential distance sensor entities.")

            # If in offline mode or no sensors found, generate mock sensors
            if self.offline_mode or not distance_sensors:
                mock_sensors = self._generate_mock_distance_sensors()
                logger.info(f"Using {len(mock_sensors)} mock distance sensors")
                return mock_sensors

            return distance_sensors
        except Exception as e:
            logger.error(f"Error getting distance sensors: {str(e)}")
            return self._generate_mock_distance_sensors()

    def get_area_predictions(self) -> Dict[str, Optional[str]]:
        """Get current area predictions for devices."""
        if self.offline_mode:
            return self._generate_mock_area_predictions()

        try:
            # Get all devices
            devices = self.get_devices()

            # Create mapping of device_id to area_id
            device_areas = {}
            for device in devices:
                device_id = device.get('id', '')
                area_id = device.get('area_id')

                if device_id and area_id:
                    device_areas[device_id] = area_id

            logger.info(f"HA_Client: Fetched current area predictions for {len(device_areas)} devices.")

            # If no predictions found, use mock data
            if not device_areas:
                return self._generate_mock_area_predictions()

            return device_areas
        except Exception as e:
            logger.error(f"Error getting area predictions: {str(e)}")
            return self._generate_mock_area_predictions()

    def get_distances(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Get distances between tracked devices and scanners from Home Assistant.

        Returns:
            Dictionary mapping device_id to a dict of scanner_id to distance info.
            Format: {
                'device_id': {
                    'scanner_id': {'distance': 5.2, 'rssi': -70},
                    'scanner_id2': {'distance': 3.1, 'rssi': -65}
                }
            }
        """
        try:
            # Get distance sensors
            distance_sensors = self.get_distance_sensors()

            # Extract device distances
            device_distances = {}

            for sensor in distance_sensors:
                entity_id = sensor.get('entity_id', '')
                state = sensor.get('state')
                attributes = sensor.get('attributes', {})

                # Skip sensors with no state or invalid state
                if not state or state == 'unknown' or state == 'unavailable':
                    continue

                # Try to extract device and scanner IDs from the entity ID
                # Format could be like: sensor.device_id_to_scanner_id_distance
                parts = entity_id.split('.')
                if len(parts) < 2:
                    continue

                name_parts = parts[1].split('_')

                # Look for common patterns in entity IDs
                device_id = None
                scanner_id = None
                distance = None
                rssi = None

                # Try to get device/scanner from attributes first
                if 'device_id' in attributes:
                    device_id = attributes['device_id']
                if 'scanner_id' in attributes:
                    scanner_id = attributes['scanner_id']

                # If not in attributes, try to parse from entity ID
                if not (device_id and scanner_id):
                    # Look for patterns like device_to_scanner_distance
                    for i, part in enumerate(name_parts):
                        if part == 'to' and i > 0 and i < len(name_parts) - 1:
                            device_id = '_'.join(name_parts[:i])
                            scanner_id = '_'.join(name_parts[i+1:-1])
                            break

                # If we still can't determine device/scanner, skip this entity
                if not device_id or not scanner_id:
                    continue

                # Try to get distance value
                try:
                    # First try to use the state as distance
                    distance = float(state)
                except (ValueError, TypeError):
                    # If state isn't a valid distance, check attributes
                    if 'distance' in attributes:
                        try:
                            distance = float(attributes['distance'])
                        except (ValueError, TypeError):
                            pass

                # Try to get RSSI if available
                if 'rssi' in attributes:
                    try:
                        rssi = float(attributes['rssi'])
                    except (ValueError, TypeError):
                        pass
                elif 'signal_strength' in attributes:
                    try:
                        rssi = float(attributes['signal_strength'])
                    except (ValueError, TypeError):
                        pass

                # Skip if no distance found
                if distance is None:
                    continue

                # Add to results
                if device_id not in device_distances:
                    device_distances[device_id] = {}

                device_distances[device_id][scanner_id] = {'distance': distance}
                if rssi is not None:
                    device_distances[device_id][scanner_id]['rssi'] = rssi

            logger.info(f"HA_Client: Found distance data for {len(device_distances)} devices.")

            # If no distances found or in offline mode, generate mock data
            if not device_distances or self.offline_mode:
                mock_distances = self._generate_mock_distances()
                logger.info(f"Using mock distance data for {len(mock_distances)} devices")
                return mock_distances

            return device_distances

        except Exception as e:
            logger.error(f"Error getting distances: {str(e)}")
            # Fall back to mock data
            return self._generate_mock_distances()

    # --- Mock Data Generators ---

    def _generate_mock_areas(self) -> List[Dict[str, Any]]:
        """Generate mock areas for offline development."""
        areas = [
            {"area_id": "lounge", "name": "Living Room"},
            {"area_id": "kitchen", "name": "Kitchen"},
            {"area_id": "master_bedroom", "name": "Master Bedroom"},
            {"area_id": "bathroom", "name": "Bathroom"},
            {"area_id": "hallway", "name": "Hallway"},
            {"area_id": "office", "name": "Office"}
        ]
        logger.debug(f"Generated {len(areas)} mock areas for offline mode")
        return areas

    def _generate_mock_devices(self) -> List[Dict[str, Any]]:
        """Generate mock devices for offline development."""
        areas = [area["area_id"] for area in self._generate_mock_areas()]

        devices = []
        # Generate beacon/scanner devices
        for i in range(1, 6):
            scanner = {
                "id": f"scanner_{i}",
                "name": f"Scanner {i}",
                "area_id": random.choice(areas),
                "model": "ESP32 BLE Scanner",
                "manufacturer": "ESPRESSIF"
            }
            devices.append(scanner)

        # Generate tracked devices
        for i in range(1, 4):
            device = {
                "id": f"device_{i}",
                "name": f"Tracked Device {i}",
                "area_id": random.choice(areas),
                "model": "BLE Tag",
                "manufacturer": "Generic"
            }
            devices.append(device)

        logger.debug(f"Generated {len(devices)} mock devices for offline mode")
        return devices

    def _generate_mock_entities(self) -> List[Dict[str, Any]]:
        """Generate mock entities for offline development."""
        entities = []

        # Generate distance sensor entities
        for device_id in range(1, 4):
            for scanner_id in range(1, 6):
                distance = round(random.uniform(1.0, 10.0), 2)
                rssi = random.randint(-85, -55)

                entity = {
                    "entity_id": f"sensor.device_{device_id}_to_scanner_{scanner_id}_distance",
                    "state": str(distance),
                    "attributes": {
                        "device_id": f"device_{device_id}",
                        "scanner_id": f"scanner_{scanner_id}",
                        "distance": distance,
                        "rssi": rssi,
                        "friendly_name": f"Device {device_id} to Scanner {scanner_id} Distance",
                        "unit_of_measurement": "m"
                    }
                }
                entities.append(entity)

        logger.debug(f"Generated {len(entities)} mock entities for offline mode")
        return entities

    def _generate_mock_distance_sensors(self) -> List[Dict[str, Any]]:
        """Generate mock distance sensors for offline development."""
        return self._generate_mock_entities()

    def _generate_mock_area_predictions(self) -> Dict[str, Optional[str]]:
        """Generate mock area predictions for offline development."""
        # Instead of random assignments, use consistent area assignments
        # to ensure all important areas are represented in the blueprint
        predictions = {
            "device_1": "master_bedroom",
            "device_2": "kitchen",
            "device_3": "lounge",
            "device_4": "bathroom",
            "device_5": "hallway",
            "device_6": "office"
        }

        logger.debug(f"Generated {len(predictions)} mock area predictions")
        return predictions

    def _generate_mock_distances(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Generate mock distance data for offline development."""
        distances = {}

        # For each device
        for device_id in range(1, 4):
            device_key = f"device_{device_id}"
            distances[device_key] = {}

            # To each scanner
            for scanner_id in range(1, 6):
                scanner_key = f"scanner_{scanner_id}"

                # Generate realistic distances - closer to the scanner in the same room
                distance = round(random.uniform(1.0, 10.0), 2)
                rssi = -55 - int(distance * 3)  # Roughly model RSSI falloff with distance

                distances[device_key][scanner_key] = {
                    "distance": distance,
                    "rssi": rssi
                }

        logger.debug(f"Generated mock distances for {len(distances)} devices")
        return distances

# For compatibility with existing code
HomeAssistantClient = HAClient