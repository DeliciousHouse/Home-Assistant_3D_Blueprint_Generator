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
import uuid  # For generating unique identifiers

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

        # The Supervisor provides these environment variables to add-ons
        # Try different environment variables that might contain the supervisor URL
        supervisor_url_vars = ['SUPERVISOR_URL', 'HASSIO_URL', 'HOME_ASSISTANT_URL']
        self.ha_url = None

        for var in supervisor_url_vars:
            if os.environ.get(var):
                self.ha_url = os.environ.get(var)
                logger.debug(f"Found Supervisor URL in {var}: {self.ha_url}")
                break

        # If no environment variable found, use the config or default
        if not self.ha_url:
            # Default URLs to try
            possible_urls = [
                self.ha_config.get('url'),  # From config file
                'http://supervisor/core/api',  # Standard addon path
                'http://supervisor/api',  # Alternative path
                'http://supervisor/core',   # Another common path
                'http://supervisor'         # Base path
            ]

            for url in possible_urls:
                if url:
                    self.ha_url = url
                    logger.debug(f"Using URL from config: {self.ha_url}")
                    break

        # Try to get the token from various environment variables
        token_vars = ['SUPERVISOR_TOKEN', 'HASSIO_TOKEN', 'HOME_ASSISTANT_TOKEN']
        self.ha_token = None

        for var in token_vars:
            if os.environ.get(var):
                self.ha_token = os.environ.get(var)
                logger.debug(f"Found auth token in {var}")
                break

        # If no token found in environment, try config
        if not self.ha_token:
            self.ha_token = self.ha_config.get('token', '')

        # Log connection details (but not the token itself)
        logger.info(f"Initializing Home Assistant client with URL: {self.ha_url}")
        logger.info(f"Authentication token available: {bool(self.ha_token)}")
        if not self.ha_token:
            logger.warning("No authentication token found! API calls will fail.")

        # Debug: Log all environment variables (useful for troubleshooting)
        if logger.isEnabledFor(logging.DEBUG):
            env_vars = {k: '***REDACTED***' if 'token' in k.lower() else v for k, v in os.environ.items()}
            logger.debug(f"Environment variables available: {env_vars}")

        # Setting offline mode to False initially and letting _test_connection determine if needed
        self.offline_mode = False

        # Setup request headers - Use X-HASSIO-KEY header for Supervisor API access
        # This is the key change needed to fix the 403 Forbidden error
        self.headers = {
            'X-HASSIO-KEY': f"{self.ha_token}",  # Special header for Supervisor API
            'Authorization': f'Bearer {self.ha_token}',  # Keep Bearer token for HA API
            'Content-Type': 'application/json',
        }

        # Validate connection on startup
        if not self._test_connection():
            logger.warning("Unable to connect to Home Assistant API. Entering offline mode with mock data.")
            self.offline_mode = True
        else:
            logger.info("Successfully connected to Home Assistant API")

    def _test_connection(self):
        """Test connection to Home Assistant and log detailed debug info."""
        try:
            # First try the Supervisor API endpoint
            base_url = self.ha_url

            # Try Supervisor API endpoints first
            supervisor_paths = [
                '/supervisor/info',  # Supervisor API
                '/core/api',         # HA API through Supervisor
                '/api',              # Direct API
            ]

            # Standard API paths as fallback
            api_paths = [
                '/api/',             # Standard API path
                '/',                 # Try directly
                '/api',              # Without trailing slash
            ]

            # Try supervisor paths first with X-HASSIO-KEY header
            for path in supervisor_paths:
                url = urljoin(base_url, path)
                try:
                    logger.debug(f"Testing Supervisor API connection to {url}")

                    # Create headers focused on Supervisor auth
                    supervisor_headers = {
                        'X-HASSIO-KEY': f"{self.ha_token}",
                        'Content-Type': 'application/json',
                    }

                    response = requests.get(url, headers=supervisor_headers, timeout=10)

                    if response.status_code == 200:
                        logger.info(f"Successfully connected to Supervisor API at {url}")
                        self.ha_url = base_url  # Save the working base URL
                        return True
                    else:
                        logger.warning(f"Supervisor API connection attempt to {url} failed: HTTP {response.status_code}")
                        logger.debug(f"Response: {response.text[:200]}")
                except Exception as e:
                    logger.warning(f"Supervisor API connection attempt to {url} failed: {str(e)}")
                    continue

            # Now try standard API paths with regular Bearer token auth
            for api_path in api_paths:
                url = urljoin(base_url, api_path)
                logger.debug(f"Testing Home Assistant API connection to {url}")

                try:
                    response = requests.get(url, headers=self.headers, timeout=10)

                    # Check response status
                    if response.status_code == 200:
                        logger.info(f"Successfully connected to Home Assistant API at {url}")
                        self.ha_url = base_url  # Save the working base URL
                        return True
                    else:
                        logger.warning(f"Connection attempt to {url} failed: HTTP {response.status_code}")
                        logger.debug(f"Response: {response.text[:200]}")
                except Exception as e:
                    logger.warning(f"Connection attempt to {url} failed: {str(e)}")
                    continue

            # Try alternative base URLs if all attempts with current URL failed
            alternative_bases = [
                'http://supervisor/core',
                'http://supervisor',
                'http://hassio/core',
                'http://hassio',
                'http://localhost:8123'
            ]

            for alt_base in alternative_bases:
                if alt_base == base_url:
                    continue  # Skip if we already tried this base URL

                logger.debug(f"Trying alternative base URL: {alt_base}")

                # First try supervisor endpoint
                sup_url = urljoin(alt_base, '/supervisor/info')
                try:
                    supervisor_headers = {
                        'X-HASSIO-KEY': f"{self.ha_token}",
                        'Content-Type': 'application/json',
                    }
                    response = requests.get(sup_url, headers=supervisor_headers, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Successfully connected using alternative Supervisor URL: {alt_base}")
                        self.ha_url = alt_base  # Update to working URL
                        return True
                except Exception:
                    pass  # Try the API endpoint

                # Then try API endpoint
                api_url = urljoin(alt_base, '/api/')
                try:
                    response = requests.get(api_url, headers=self.headers, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"Successfully connected using alternative API URL: {alt_base}")
                        self.ha_url = alt_base  # Update to working URL
                        return True
                except Exception:
                    continue  # Try next URL on exception

            # If we get here, all connection attempts failed
            logger.error("All connection attempts to Home Assistant API failed.")
            return False

        except Exception as e:
            logger.error(f"Error connecting to Home Assistant: {str(e)}")
            return False

    def _api_call(self, endpoint, method='GET', data=None, timeout=10, is_supervisor_api=False):
        """Make an API call to Home Assistant."""
        if self.offline_mode:
            logger.warning(f"Offline mode active, returning mock data for {endpoint}")
            return self._generate_mock_data(endpoint)

        # Determine which base URL to use
        if endpoint.startswith('/'):
            endpoint = endpoint[1:]  # Remove leading slash if present

        # Choose appropriate URL based on whether this is a supervisor API call
        api_url = self.ha_url
        if not is_supervisor_api and '/api/' not in self.ha_url:
            # For regular HA API calls, make sure we're targeting the correct endpoint
            if '/core' in api_url and not '/core/api' in api_url:
                api_url = urljoin(api_url, '/core/api/')
            else:
                api_url = urljoin(api_url, '/api/')

        url = urljoin(api_url, endpoint)
        logger.debug(f"Making {method} request to {url}")

        try:
            # Choose appropriate headers
            headers = self.headers
            if is_supervisor_api:
                # Use only the Supervisor header for Supervisor API calls
                headers = {
                    'X-HASSIO-KEY': f"{self.ha_token}",
                    'Content-Type': 'application/json',
                }

            response = None
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=timeout)
            elif method == 'POST':
                response = requests.post(url, headers=headers, json=data, timeout=timeout)
            elif method == 'PUT':
                response = requests.put(url, headers=headers, json=data, timeout=timeout)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=timeout)
            else:
                logger.error(f"Unsupported method: {method}")
                return None

            if response.status_code in [200, 201]:
                try:
                    return response.json()
                except ValueError:
                    # Not JSON content
                    return response.text
            else:
                logger.error(f"API call to {url} failed with status {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error making API call to {url}: {str(e)}")
            return None

    def _generate_mock_data(self, endpoint):
        """Generate mock data for offline mode."""
        logger.debug(f"Generating mock data for {endpoint}")

        # Mock data for different endpoints
        if 'states' in endpoint:
            return self._generate_mock_states()
        elif 'bluetooth' in endpoint:
            return self._generate_mock_bluetooth()
        elif 'device_tracker' in endpoint:
            return self._generate_mock_device_trackers()
        elif 'areas' in endpoint or 'config/area_registry' in endpoint:
            return self._generate_mock_areas()
        else:
            # Generic response for other endpoints
            return {"result": "success", "data": {}, "mock": True}

    def _generate_mock_states(self):
        """Generate mock states data."""
        # Generate basic entity states
        return [
            {
                "entity_id": "sensor.living_room_temperature",
                "state": "21.5",
                "attributes": {"unit_of_measurement": "°C", "friendly_name": "Living Room Temperature"}
            },
            {
                "entity_id": "sensor.living_room_humidity",
                "state": "45",
                "attributes": {"unit_of_measurement": "%", "friendly_name": "Living Room Humidity"}
            },
            {
                "entity_id": "sensor.test_device_distance_test_scanner",
                "state": "2.5",
                "attributes": {
                    "unit_of_measurement": "m",
                    "friendly_name": "Test Device Distance",
                    "source": "test_device",
                    "scanner": "test_scanner",
                    "rssi": -65
                }
            }
        ]

    def _generate_mock_bluetooth(self):
        """Generate mock Bluetooth sensor data."""
        return [
            {
                "entity_id": "sensor.bedroom_scanner_1",
                "state": "on",
                "attributes": {
                    "friendly_name": "Bedroom BT Scanner",
                    "rssi": -65,
                    "source": "smartphone_1"
                }
            },
            {
                "entity_id": "sensor.living_room_scanner_1",
                "state": "on",
                "attributes": {
                    "friendly_name": "Living Room BT Scanner",
                    "rssi": -70,
                    "source": "smartphone_1"
                }
            },
            {
                "entity_id": "sensor.kitchen_scanner_1",
                "state": "on",
                "attributes": {
                    "friendly_name": "Kitchen BT Scanner",
                    "rssi": -85,
                    "source": "smartphone_2"
                }
            }
        ]

    def _generate_mock_device_trackers(self):
        """Generate mock device tracker data."""
        return [
            {
                "entity_id": "device_tracker.smartphone_1",
                "state": "home",
                "attributes": {
                    "friendly_name": "Smartphone 1",
                    "source_type": "bluetooth",
                    "area_id": "living_room"
                }
            },
            {
                "entity_id": "device_tracker.smartphone_2",
                "state": "home",
                "attributes": {
                    "friendly_name": "Smartphone 2",
                    "source_type": "bluetooth",
                    "area_id": "kitchen"
                }
            }
        ]

    def _generate_mock_areas(self):
        """Generate mock areas data."""
        return [
            {
                "area_id": "living_room",
                "name": "Living Room",
                "picture": None
            },
            {
                "area_id": "kitchen",
                "name": "Kitchen",
                "picture": None
            },
            {
                "area_id": "master_bedroom",
                "name": "Master Bedroom",
                "picture": None
            },
            {
                "area_id": "office",
                "name": "Office",
                "picture": None
            }
        ]

    def get_states(self):
        """Get all entity states from Home Assistant."""
        return self._api_call('states')

    def get_bluetooth_sensors(self):
        """Get all Bluetooth sensors from Home Assistant."""
        logger.debug("Fetching Bluetooth sensors from Home Assistant")

        # In a production environment, we'd filter for entities that are Bluetooth sensors
        # For now, let's get all states and filter for Bluetooth sensors
        states = self._api_call('states')

        if not states and self.offline_mode:
            return self._generate_mock_bluetooth()

        bt_sensors = []
        # Filter for Bluetooth sensors based on common patterns
        for entity in states or []:
            entity_id = entity.get('entity_id', '')

            # Check if this is likely to be a Bluetooth sensor
            # Patterns to check: 'bluetooth' in entity_id, has rssi attribute, etc.
            attributes = entity.get('attributes', {})

            # Main condition: entity has RSSI attribute (common for BT sensors)
            has_rssi = 'rssi' in attributes

            # Secondary conditions: entity ID suggests Bluetooth
            is_bt_entity = ('bluetooth' in entity_id.lower() or
                           'bt_' in entity_id.lower() or
                           'ble_' in entity_id.lower() or
                           'rssi' in entity_id.lower() or
                           'proximity' in entity_id.lower())

            # If any of the Bluetooth conditions are met, include this entity
            if has_rssi or is_bt_entity:
                bt_sensors.append(entity)

        if not bt_sensors:
            logger.warning(f"No Bluetooth sensors found among {len(states) if states else 0} entities, returning mock data")
            return self._generate_mock_bluetooth()

        logger.info(f"Found {len(bt_sensors)} Bluetooth sensors")
        return bt_sensors

    def get_device_trackers(self):
        """Get all device trackers from Home Assistant."""
        logger.debug("Fetching device trackers from Home Assistant")

        # Get all states and filter for device trackers
        states = self._api_call('states')

        if not states and self.offline_mode:
            return self._generate_mock_device_trackers()

        device_trackers = []

        # Filter for device trackers
        for entity in states or []:
            entity_id = entity.get('entity_id', '')

            # Check if this is a device tracker
            if entity_id.startswith('device_tracker.'):
                device_trackers.append(entity)

        if not device_trackers:
            logger.warning("No device trackers found, returning mock data")
            return self._generate_mock_device_trackers()

        logger.info(f"Found {len(device_trackers)} device trackers")
        return device_trackers

    def get_areas(self):
        """Get all areas from Home Assistant."""
        try:
            # Try multiple area endpoint variations in order of likelihood
            area_endpoints = [
                'api/config/area_registry',   # Most current HA API endpoint
                'api/areas',                  # Alternative API endpoint
                'api/config/areas',           # Third alternative
                'config/area_registry',       # Slightly older endpoint
                'api/area_registry',          # Another possibility
                'core/api/config/area_registry', # Try with core prefix
                'api/states/area',            # Another possibility
                'areas',                      # Legacy endpoint
                'supervisor/core/api/config/area_registry',  # Try supervisor path
                'core/api/areas',             # Another core variation
                'supervisor/core/api/config/areas',  # Try supervisor path with areas
            ]

            for endpoint in area_endpoints:
                logger.debug(f"Trying area endpoint: {endpoint}")
                areas = self._api_call(endpoint)

                if areas and isinstance(areas, list):
                    logger.info(f"Retrieved {len(areas)} areas from Home Assistant API ({endpoint} endpoint)")
                    return areas
                elif areas:
                    logger.debug(f"Area endpoint {endpoint} returned non-list result: {type(areas)}")

            # If direct API calls fail, try extracting areas from entity attributes
            logger.debug("Trying to extract areas from entity attributes")
            area_ids = {}

            # First try device trackers
            device_trackers = self.get_device_trackers()
            for tracker in device_trackers:
                area_id = tracker.get('attributes', {}).get('area_id')
                area_name = tracker.get('attributes', {}).get('area_name') or area_id
                if area_id:
                    try:
                        area_name = area_name.replace('_', ' ').title() if area_name else area_id.replace('_', ' ').title()
                        area_ids[area_id] = {"area_id": area_id, "name": area_name, "derived": True}
                    except Exception as e:
                        logger.warning(f"Error formatting area name: {e}")
                        # Handle case where area_id is None
                        if area_id:
                            area_ids[area_id] = {"area_id": area_id, "name": "Unknown Area", "derived": True}

            # Then try all entities for more area references
            states = self.get_states()
            if states:
                for entity in states:
                    attrs = entity.get('attributes', {})
                    if 'area_id' in attrs and attrs['area_id'] and attrs['area_id'] not in area_ids:
                        area_id = attrs['area_id']
                        area_name = attrs.get('area_name', '')
                        if not area_name and area_id:
                            try:
                                area_name = area_id.replace('_', ' ').title()
                            except Exception:
                                area_name = "Unknown Area"
                        area_ids[area_id] = {"area_id": area_id, "name": area_name, "derived": True}

            if area_ids:
                logger.info(f"Extracted {len(area_ids)} areas from entity attributes")
                return list(area_ids.values())

            # As a last resort, create some standard areas for testing
            logger.warning("Failed to get areas from API or entity attributes, using standard areas")
            standard_areas = [
                {"area_id": "living_room", "name": "Living Room"},
                {"area_id": "kitchen", "name": "Kitchen"},
                {"area_id": "bedroom", "name": "Bedroom"},
                {"area_id": "office", "name": "Office"},
                {"area_id": "bathroom", "name": "Bathroom"},
                {"area_id": "entry", "name": "Entry"}
            ]
            return standard_areas
        except Exception as e:
            logger.error(f"Error getting areas: {str(e)}")
            # Don't return None, return mock areas if there's an error
            return self._generate_mock_areas()

    def get_devices(self):
        """Get all devices from Home Assistant."""
        try:
            # Try device registry endpoint
            devices = self._api_call('config/device_registry')

            if devices:
                logger.info(f"Retrieved {len(devices)} devices from Home Assistant API")
                return devices

            logger.warning("Failed to get devices from API, using mock data")
            return []
        except Exception as e:
            logger.error(f"Error getting devices: {str(e)}")
            return []

# For compatibility with existing code
HomeAssistantClient = HAClient