#!/usr/bin/env python3

import os
import logging
import json
import requests
import time
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class HAClient:
    """Home Assistant API Client for the Blueprint Generator add-on."""

    def __init__(self, config_path: Optional[str] = None, token: Optional[str] = None):
        """Initialize HA client with appropriate authentication token and base URL."""
        # Use config loader if available
        try:
            from .config_loader import load_config
            self.config = load_config(config_path)
        except (ImportError, ModuleNotFoundError):
            self.config = {}

        # HA API URL - supervisor path for add-on mode, can be overridden for dev
        self.base_url = self.config.get('home_assistant', {}).get('url', 'http://supervisor/core')

        # Authentication token - try from config first, then from constructor param
        self.token = token or self.config.get('home_assistant', {}).get('token')

        # If no token provided yet, try environment variable (preferred for add-on)
        if not self.token:
            self.token = os.environ.get('SUPERVISOR_TOKEN')

        logger.info(f"Initializing Home Assistant client with URL: {self.base_url}")
        logger.info(f"Authentication token available: {bool(self.token)}")

        # Test the connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test the connection to Home Assistant."""
        # First check if we can access the supervisor API
        try:
            supervisor_url = "http://supervisor/supervisor/info"
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json"
            }
            response = requests.get(supervisor_url, headers=headers)
            response.raise_for_status()
            logger.info(f"Successfully connected to Supervisor API at {supervisor_url}")
        except Exception as e:
            logger.warning(f"Could not connect to Supervisor API: {str(e)}")

        # Then check if we can access the HA API
        try:
            # Make a simple API call to check authentication
            states = self.get_states(entity_id="sensor.time")
            logger.info("Successfully connected to Home Assistant API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Home Assistant API: {str(e)}")
            return False

    def _authenticated_request(self, endpoint: str, method: str = 'GET',
                              data: Optional[Dict] = None) -> requests.Response:
        """Make an authenticated request to the HA API."""
        url = f"{self.base_url}/api/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_states(self, entity_id: Optional[str] = None) -> List[Dict]:
        """Get entity states from Home Assistant."""
        try:
            if entity_id:
                response = self._authenticated_request(f"states/{entity_id}")
                return [response.json()]
            else:
                response = self._authenticated_request("states")
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get states: {str(e)}")
            return []

    def get_areas(self) -> List[Dict]:
        """Get areas from Home Assistant."""
        try:
            # HA API doesn't have a direct endpoint for areas, so we need to extract from entities
            areas = []
            areas_seen = set()

            # Get all entities to look for area_id attributes
            all_entities = self.get_states()

            # Extract areas from entity attributes
            for entity in all_entities:
                area_id = entity.get('attributes', {}).get('area_id')
                area_name = entity.get('attributes', {}).get('friendly_name')

                if area_id and area_id not in areas_seen:
                    areas_seen.add(area_id)
                    areas.append({
                        'area_id': area_id,
                        'name': self._get_area_name_for_id(area_id, all_entities) or area_id,
                        'entities': []
                    })

            # If we found areas, add entities to them
            if areas:
                for entity in all_entities:
                    area_id = entity.get('attributes', {}).get('area_id')
                    if area_id:
                        for area in areas:
                            if area['area_id'] == area_id:
                                area['entities'].append(entity.get('entity_id'))
                                break

            logger.info(f"Extracted {len(areas)} areas from entity attributes")
            return areas

        except Exception as e:
            logger.error(f"Failed to get areas: {str(e)}")
            return []

    def _get_area_name_for_id(self, area_id: str, all_entities: List[Dict]) -> Optional[str]:
        """Try to find a friendly name for an area ID from entity attributes."""
        # First look for entities with matching area_id and have a 'friendly_name'
        for entity in all_entities:
            if (entity.get('attributes', {}).get('area_id') == area_id and
                'friendly_name' in entity.get('attributes', {})):
                # Extract area name from friendly_name
                friendly_name = entity.get('attributes', {}).get('friendly_name', '')

                # Use matching sensor.area_* entity if available
                if entity.get('entity_id', '').startswith('sensor.area_'):
                    return friendly_name

                # Use the first part of friendly name if it contains 'area' or 'room'
                name_parts = friendly_name.split()
                if any(area_term in friendly_name.lower() for area_term in ['area', 'room', 'office', 'bedroom', 'kitchen', 'bathroom', 'living']):
                    for term in ['area', 'room']:
                        if term in friendly_name.lower():
                            return friendly_name

        # If no good name found, try to derive from area_id
        if area_id:
            # Convert area_id to a readable name
            name = area_id.replace('_', ' ').title()
            return name

        return None

    def get_device_trackers(self) -> List[Dict]:
        """Get device tracker entities from Home Assistant."""
        try:
            all_states = self.get_states()
            trackers = [entity for entity in all_states if entity.get('entity_id', '').startswith('device_tracker.')]
            logger.info(f"Found {len(trackers)} device trackers")
            return trackers
        except Exception as e:
            logger.error(f"Failed to get device trackers: {str(e)}")
            return []

    def get_distance_sensors(self) -> List[Dict]:
        """Get distance sensor entities from Home Assistant."""
        try:
            all_states = self.get_states()

            # Look for entities that have distance in their name or attributes
            distance_sensors = []
            for entity in all_states:
                entity_id = entity.get('entity_id', '')

                # Check if it's a sensor
                if not entity_id.startswith('sensor.'):
                    continue

                # Check if it's a distance sensor by name
                if any(term in entity_id.lower() for term in ['distance', 'range', 'proximity']):
                    distance_sensors.append(entity)
                    continue

                # Check if it has a unit of measurement related to distance
                unit = entity.get('attributes', {}).get('unit_of_measurement', '')
                if unit in ['m', 'cm', 'in', 'ft', 'mm']:
                    distance_sensors.append(entity)
                    continue

                # Check if it's a BLE-related sensor
                if 'ble' in entity_id.lower() and not any(term in entity_id.lower() for term in
                                                        ['battery', 'humidity', 'temperature', 'pressure']):
                    distance_sensors.append(entity)

            logger.info(f"Found {len(distance_sensors)} distance sensors")
            return distance_sensors
        except Exception as e:
            logger.error(f"Failed to get distance sensors: {str(e)}")
            return []

    def get_entities_by_type(self, domain: str) -> List[Dict]:
        """Get entities of a specific domain from Home Assistant."""
        try:
            all_states = self.get_states()
            entities = [entity for entity in all_states if entity.get('entity_id', '').startswith(f"{domain}.")]
            return entities
        except Exception as e:
            logger.error(f"Failed to get {domain} entities: {str(e)}")
            return []

    def get_all_lights(self) -> List[Dict]:
        """Get all light entities from Home Assistant with their area assignments."""
        return self.get_entities_by_type('light')

    def get_all_binary_sensors(self) -> List[Dict]:
        """Get all binary sensor entities from Home Assistant."""
        return self.get_entities_by_type('binary_sensor')

    def get_beacon_devices(self) -> List[Dict]:
        """Get beacon/tag/tracker device information."""
        try:
            # Check for ESPresense or BLE monitor sensors
            all_states = self.get_states()
            beacon_devices = []

            for entity in all_states:
                entity_id = entity.get('entity_id', '')

                # Look for ESPresense sensors
                if entity_id.startswith('sensor.') and any(term in entity_id.lower() for term in
                                                         ['espresense', 'ble_tracker', 'beacon', 'tag']):
                    beacon_devices.append(entity)

            logger.info(f"Found {len(beacon_devices)} beacon devices")
            return beacon_devices
        except Exception as e:
            logger.error(f"Failed to get beacon devices: {str(e)}")
            return []

    def get_entities_by_area(self, area_id: str) -> List[Dict]:
        """Get all entities assigned to a specific area."""
        try:
            all_states = self.get_states()
            area_entities = []

            for entity in all_states:
                if entity.get('attributes', {}).get('area_id') == area_id:
                    area_entities.append(entity)

            return area_entities
        except Exception as e:
            logger.error(f"Failed to get entities for area {area_id}: {str(e)}")
            return []

# Singleton instance of HAClient
_ha_client_instance = None

def get_ha_client() -> HAClient:
    """
    Get or create a singleton instance of HAClient.
    This ensures we're using the same client throughout the application.
    """
    global _ha_client_instance
    if _ha_client_instance is None:
        _ha_client_instance = HAClient()
    return _ha_client_instance