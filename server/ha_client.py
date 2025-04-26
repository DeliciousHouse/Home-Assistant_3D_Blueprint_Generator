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

        # Get Home Assistant connection details
        self.ha_url = self.ha_config.get('url', os.environ.get('HASS_URL', 'http://supervisor/core'))
        self.ha_token = self.ha_config.get('token', os.environ.get('HASS_TOKEN', ''))

        # Legacy/local development fallback
        if not self.ha_token:
            self.ha_token = os.environ.get('SUPERVISOR_TOKEN', '')

        # Setup request headers
        self.headers = {
            'Authorization': f'Bearer {self.ha_token}',
            'Content-Type': 'application/json',
        }

        logger.info(f"HAClient initialized with URL: {self.ha_url}")

    def get_areas(self) -> List[Dict[str, Any]]:
        """Get all areas from Home Assistant."""
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
                return []

        except Exception as e:
            logger.error(f"Error getting areas from Home Assistant: {str(e)}")
            return []

    def get_devices(self) -> List[Dict[str, Any]]:
        """Get all devices from Home Assistant."""
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
                return []

        except Exception as e:
            logger.error(f"Error getting devices from Home Assistant: {str(e)}")
            return []

    def get_entities(self) -> List[Dict[str, Any]]:
        """Get all entities from Home Assistant."""
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
                return []

        except Exception as e:
            logger.error(f"Error getting entities from Home Assistant: {str(e)}")
            return []

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
            return distance_sensors
        except Exception as e:
            logger.error(f"Error getting distance sensors: {str(e)}")
            return []

    def get_area_predictions(self) -> Dict[str, Optional[str]]:
        """Get current area predictions for devices."""
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
            return device_areas
        except Exception as e:
            logger.error(f"Error getting area predictions: {str(e)}")
            return {}

# For compatibility with existing code
HomeAssistantClient = HAClient