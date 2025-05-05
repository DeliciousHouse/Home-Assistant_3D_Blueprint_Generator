#!/usr/bin/env python3
"""
Home Assistant Client Module for 3D Blueprint Generator.

This module handles communication with the Home Assistant API to retrieve
devices, areas, and other information needed for blueprint generation.
"""

import logging
import os
import json
import time
import requests
from typing import Dict, List, Optional, Union, Any
from urllib.parse import urljoin
import re

# Use the standardized config loader
try:
    from .config_loader import load_config
except ImportError:
    try:
        from config_loader import load_config
    except ImportError:
        def load_config(path=None):
            logger = logging.getLogger(__name__)
            logger.warning("Could not import config_loader. Using empty config.")
            return {}

logger = logging.getLogger(__name__)

class HAClient:
    """Client for interacting with Home Assistant API."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(HAClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Home Assistant client."""
        # Skip if already initialized
        if hasattr(self, '_initialized') and self._initialized:
            return

        # Load configuration
        self.config = load_config(config_path)
        ha_config = self.config.get('home_assistant', {})

        # Base URL and credentials
        self.base_url = ha_config.get('url', 'http://supervisor/core')
        self.token = ha_config.get('token', os.environ.get('HA_TOKEN', ''))

        # API endpoints
        self.api_url = f"{self.base_url}/api"

        # Cache for API responses
        self._cache = {}
        self._cache_times = {}
        self._cache_expiry = {
            'areas': 300,  # 5 minutes
            'devices': 60,  # 1 minute
            'entities': 60,  # 1 minute
            'states': 10    # 10 seconds
        }

        # Common headers for API requests
        self._headers = {}
        if self.token:
            self._headers['Authorization'] = f"Bearer {self.token}"
        self._headers['Content-Type'] = 'application/json'

        # Test API connection on initialization
        self._connection_status = self._test_connection()

        self._initialized = True
        logger.info(f"Home Assistant client initialized. API connection: {'OK' if self._connection_status else 'FAIL'}")

    def _test_connection(self) -> bool:
        """Test the connection to Home Assistant API."""
        try:
            response = requests.get(f"{self.api_url}/", headers=self._headers, timeout=5)
            if response.status_code == 200:
                logger.debug("Successfully connected to Home Assistant API")
                return True
            else:
                logger.warning(f"Failed to connect to Home Assistant API: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Home Assistant API: {e}")
            return False

    def _get_cached_or_api(self, endpoint: str, cache_key: str, params: Optional[Dict] = None) -> Any:
        """Get data from cache or API."""
        current_time = time.time()
        expiry = self._cache_expiry.get(cache_key, 60)  # Default 60 seconds

        # Check if we have a valid cached response
        if (cache_key in self._cache and
            cache_key in self._cache_times and
            current_time - self._cache_times[cache_key] < expiry):
            return self._cache[cache_key]

        # Otherwise, make API request
        try:
            url = f"{self.api_url}/{endpoint}"
            response = requests.get(url, headers=self._headers, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                # Cache the response
                self._cache[cache_key] = data
                self._cache_times[cache_key] = current_time
                return data
            else:
                logger.error(f"API request to {endpoint} failed: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error in API request to {endpoint}: {e}")
            return None

    def get_areas(self) -> List[Dict]:
        """Retrieve areas from Home Assistant API."""
        areas = self._get_cached_or_api('areas', 'areas')
        if areas:
            logger.info(f"Retrieved {len(areas)} areas from Home Assistant")
            return areas
        else:
            logger.warning("Failed to retrieve areas from Home Assistant")
            return []

    def get_devices(self) -> List[Dict]:
        """Retrieve devices from Home Assistant API."""
        devices = self._get_cached_or_api('devices', 'devices')
        if devices:
            logger.info(f"Retrieved {len(devices)} devices from Home Assistant")
            return devices
        else:
            logger.warning("Failed to retrieve devices from Home Assistant")
            return []

    def get_entities(self, entity_filter: Optional[str] = None) -> List[Dict]:
        """
        Retrieve entities from Home Assistant API.

        Args:
            entity_filter: Optional filter to get specific entity types (e.g., 'device_tracker')
        """
        cache_key = f'entities_{entity_filter}' if entity_filter else 'entities'
        entities = self._get_cached_or_api('states', cache_key)

        if entities:
            if entity_filter:
                filtered = [e for e in entities if e.get('entity_id', '').startswith(f"{entity_filter}.")]
                logger.info(f"Retrieved {len(filtered)} {entity_filter} entities from Home Assistant")
                return filtered
            else:
                logger.info(f"Retrieved {len(entities)} entities from Home Assistant")
                return entities
        else:
            logger.warning(f"Failed to retrieve {'filtered ' if entity_filter else ''}entities from Home Assistant")
            return []

    def get_entity_registry(self) -> List[Dict]:
        """Retrieve entity registry from Home Assistant API."""
        registry = self._get_cached_or_api('config/entity_registry', 'entity_registry')
        if registry:
            logger.info(f"Retrieved entity registry with {len(registry)} entries")
            return registry
        else:
            logger.warning("Failed to retrieve entity registry")
            return []

    def get_device_info(self, device_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific device.

        Args:
            device_id: The device ID to look up

        Returns:
            Device information dictionary or None if not found
        """
        # Try to find device in entity registry first
        entity_registry = self.get_entity_registry()
        for entity in entity_registry:
            if entity.get('entity_id') == f'device_tracker.{device_id}' or \
               entity.get('unique_id') == device_id:
                device_info = {
                    'id': entity.get('id'),
                    'entity_id': entity.get('entity_id'),
                    'name': entity.get('name'),
                    'area_id': entity.get('area_id'),
                    'device_id': entity.get('device_id')
                }
                return device_info

        # If not found, try device registry
        devices = self.get_devices()
        for device in devices:
            if device.get('id') == device_id or \
               device.get('name', '').lower() == device_id.lower():
                return device

        logger.debug(f"No device info found for {device_id}")
        return None

    def get_device_trackers(self) -> List[Dict]:
        """Get all device trackers from Home Assistant."""
        return self.get_entities('device_tracker')

    def get_bluetooth_devices(self) -> List[Dict]:
        """Get all Bluetooth-related entities from Home Assistant."""
        all_entities = self.get_entities()
        bluetooth_entities = []

        # Pattern to match Bluetooth-related entities
        ble_patterns = [
            r'_ble_',
            r'bluetooth_',
            r'ble_',
            r'_bt_',
            r'_rssi$',
            r'_distance_',
            r'proximity_'
        ]

        for entity in all_entities:
            entity_id = entity.get('entity_id', '')
            for pattern in ble_patterns:
                if re.search(pattern, entity_id, re.IGNORECASE):
                    bluetooth_entities.append(entity)
                    break

        logger.info(f"Found {len(bluetooth_entities)} Bluetooth-related entities")
        return bluetooth_entities

    def get_distance_sensors(self) -> List[Dict]:
        """Get all distance-related sensors from Home Assistant."""
        all_entities = self.get_entities()
        distance_sensors = []

        # Pattern to match distance-related entities
        distance_patterns = [
            r'_distance_',
            r'_dist_',
            r'_range_',
            r'_proximity_',
            r'_meters$'
        ]

        for entity in all_entities:
            entity_id = entity.get('entity_id', '')
            if entity_id.startswith('sensor.'):
                for pattern in distance_patterns:
                    if re.search(pattern, entity_id, re.IGNORECASE):
                        distance_sensors.append(entity)
                        break

        logger.info(f"Found {len(distance_sensors)} distance-related sensors")
        return distance_sensors

    def get_rssi_sensors(self) -> List[Dict]:
        """Get all RSSI-related sensors from Home Assistant."""
        all_entities = self.get_entities()
        rssi_sensors = []

        # Pattern to match RSSI-related entities
        rssi_patterns = [
            r'_rssi',
            r'_signal_strength',
            r'_signal$'
        ]

        for entity in all_entities:
            entity_id = entity.get('entity_id', '')
            if entity_id.startswith('sensor.'):
                for pattern in rssi_patterns:
                    if re.search(pattern, entity_id, re.IGNORECASE):
                        rssi_sensors.append(entity)
                        break

        logger.info(f"Found {len(rssi_sensors)} RSSI-related sensors")
        return rssi_sensors

    def get_area_entities(self, area_id: str) -> List[Dict]:
        """
        Get all entities associated with a specific area.

        Args:
            area_id: The area ID to filter by

        Returns:
            List of entities in the specified area
        """
        entity_registry = self.get_entity_registry()
        area_entities = []

        for entity in entity_registry:
            if entity.get('area_id') == area_id:
                # Get the current state of this entity if available
                entities = self.get_entities()
                for e in entities:
                    if e.get('entity_id') == entity.get('entity_id'):
                        area_entities.append(e)
                        break

        logger.info(f"Found {len(area_entities)} entities in area {area_id}")
        return area_entities

# Helper function to get a properly initialized HAClient
def get_ha_client() -> HAClient:
    """Get an initialized Home Assistant client instance."""
    return HAClient()