import threading
import requests
import json
import websocket
import logging
import os
from typing import Dict, List, Optional
from threading import Event
import time

logger = logging.getLogger(__name__)

def get_area_registry(base_url, token):
    """Get the area registry from Home Assistant using a WebSocket connection."""
    # Convert HTTP URL to WebSocket URL
    ws_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://')
    ws_url = f"{ws_url}/api/websocket"

    areas = []
    response_received = Event()
    message_id = 1

    def on_message(ws, message):
        nonlocal areas
        try:
            # Clean up any potential non-JSON content
            if message.startswith('\n'):
                message = message.strip()

            data = json.loads(message)
            logger.debug(f"WebSocket message: {data.get('type')}")

            if data.get('type') == 'auth_required':
                # Send authentication
                ws.send(json.dumps({
                    "type": "auth",
                    "access_token": token
                }))

            elif data.get('type') == 'auth_ok':
                # Now authenticated, request area registry
                ws.send(json.dumps({
                    "id": message_id,
                    "type": "config/area_registry/list"
                }))

            elif data.get('type') == 'result' and data.get('success'):
                # Got area registry results
                if 'result' in data and isinstance(data['result'], list):
                    areas = data['result']
                    logger.info(f"Received {len(areas)} areas via WebSocket")
                    response_received.set()
                    ws.close()
        except json.JSONDecodeError as e:
            logger.error(f"WebSocket message not valid JSON: {e}")
            logger.debug(f"Message content (first 100 chars): {message[:100]}")
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")

    def on_error(ws, error):
        logger.error(f"WebSocket error: {error}")
        response_received.set()

    def on_close(ws, close_status_code, close_msg):
        logger.debug(f"WebSocket closed: {close_msg}")
        response_received.set()

    def on_open(ws):
        logger.debug("WebSocket connection established")

    # Create WebSocket connection
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    # Start connection in a separate thread
    import threading
    ws_thread = threading.Thread(target=ws.run_forever)
    ws_thread.daemon = True
    ws_thread.start()

    # Wait for response or timeout
    response_received.wait(timeout=10)

    return areas

class HomeAssistantClient:
    """Client for interacting with Home Assistant API."""

    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None):
        """Initialize with Home Assistant connection details."""
        # Use provided values or get from environment (for add-on)
        self.base_url = base_url or self._get_base_url()
        self.token = token or self._get_token()
        self.headers = {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }
        logger.debug(f"Initialized HomeAssistantClient with base_url: {self.base_url}")

    def _get_base_url(self) -> str:
        """Get base URL from environment or use default."""
        # For Home Assistant add-on
        if os.environ.get('SUPERVISOR_TOKEN'):
            return "http://supervisor/core"

        # Check for options file
        options_path = '/data/options.json'
        if os.path.exists(options_path):
            try:
                with open(options_path) as f:
                    options = json.load(f)
                    if 'ha_url' in options:
                        return options['ha_url']
            except Exception as e:
                logger.error(f"Error reading options.json: {e}")

        # Default for development
        return "http://localhost:8123"

    def _get_token(self) -> str:
        """Get authentication token from environment."""
        # Try to get token from Home Assistant add-on environment
        supervisor_token = os.environ.get('SUPERVISOR_TOKEN')
        if supervisor_token:
            return supervisor_token

        # Check for options file
        options_path = '/data/options.json'
        if os.path.exists(options_path):
            try:
                with open(options_path) as f:
                    options = json.load(f)
                    if 'ha_token' in options:
                        return options['ha_token']
            except Exception as e:
                logger.error(f"Error reading options.json: {e}")

        # Return empty string if no token found (will fail authentication)
        return ""

    def test_connection(self) -> bool:
        """Test connection to Home Assistant API."""
        try:
            url = f"{self.base_url}/api/" # Test base API endpoint
            response = requests.get(url, headers=self.headers, timeout=5)
            # Check for successful status code (e.g., 200 OK)
            # Also check response content if necessary (e.g., {"message": "API running."})
            if response.status_code == 200:
                 # Simple check if response looks like expected HA API response
                 try:
                     data = response.json()
                     if isinstance(data, dict) and 'message' in data:
                         logger.info("Home Assistant API connection successful.")
                         return True
                 except json.JSONDecodeError:
                      logger.warning("HA API connection test: Unexpected response format, but status 200.")
                      return True # Assume connected if status is OK
            logger.error(f"Home Assistant API connection test failed: Status {response.status_code}")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Home Assistant API connection test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during HA connection test: {e}")
            return False

    def find_entities_by_pattern(self, patterns: List[str], domains: List[str] = None) -> List[Dict]:
        """Find entities matching any of the patterns in their entity_id or attributes."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            matching_entities = []

            for state in all_states:
                entity_id = state.get('entity_id', '')

                # Check domain filter if provided
                if domains and not any(entity_id.startswith(f"{domain}.") for domain in domains):
                    continue

                # Check for pattern matches
                if any(pattern.lower() in entity_id.lower() for pattern in patterns):
                    matching_entities.append({
                        'entity_id': entity_id,
                        'state': state.get('state'),
                        'attributes': state.get('attributes', {})
                    })

            return matching_entities

        except Exception as e:
            logger.error(f"Failed to find entities by pattern: {str(e)}")
            return []

    def get_bluetooth_devices(self) -> List[Dict]:
        """Get all bluetooth devices from Home Assistant."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            bluetooth_devices = []

            for state in all_states:
                entity_id = state.get('entity_id', '')
                if entity_id.startswith('sensor.', 'binary_sensor.'):
                    bluetooth_devices.append({
                        'entity_id': entity_id,
                        'attributes': state.get('attributes', {}),
                        'state': state.get('state')
                    })

            return bluetooth_devices

        except Exception as e:
            logger.error(f"Failed to get bluetooth devices from Home Assistant: {str(e)}")
            return []

    def get_private_ble_devices(self) -> List[Dict]:
        """Get all BLE and distance-sensing devices with improved flexibility."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            devices = []

            # Keywords that indicate relevant entities
            device_keywords = ['ble', 'bluetooth', 'watch', 'iphone', 'phone', 'mmwave']
            measurement_keywords = ['distance', 'rssi', 'signal', 'detection']

            for state in all_states:
                entity_id = state.get('entity_id', '').lower()
                attributes = state.get('attributes', {})
                friendly_name = attributes.get('friendly_name', entity_id)

                # Don't limit to just sensors - include device trackers too
                if entity_id.startswith(('sensor.', 'device_tracker.', 'binary_sensor.')):

                    # Check if entity contains both device and measurement keywords
                    has_device_keyword = any(keyword in entity_id for keyword in device_keywords)
                    has_measurement = any(keyword in entity_id for keyword in measurement_keywords)

                    # Either it needs both types of keywords, or it has 'ble' and is related to devices
                    if (has_device_keyword and has_measurement) or ('ble' in entity_id):
                        # Get the value - could be distance or RSSI
                        value = state.get('state')
                        try:
                            value = float(value)
                        except (ValueError, TypeError):
                            if value == 'unavailable' or value == 'unknown':
                                value = -100  # Default for unavailable devices
                            else:
                                # Skip non-numeric values that aren't standard states
                                continue

                        # Try to extract a device identifier
                        parts = entity_id.split('_')
                        if len(parts) >= 2:
                            # Use part after domain as device identifier
                            device_type = parts[1]
                        else:
                            device_type = 'unknown'

                        # Take a better guess at whether this is distance or RSSI
                        is_distance = any(k in entity_id for k in ['distance', 'meters', 'range'])
                        is_rssi = any(k in entity_id for k in ['rssi', 'signal', 'strength'])

                        # Convert between distance and RSSI if needed
                        if is_distance:
                            distance = value
                            rssi = -59 + (value * -2)  # Simple conversion
                        elif is_rssi:
                            rssi = value
                            distance = max(0, ((-1 * (rssi + 59)) / 2))  # Reverse of above formula
                        else:
                            # If unclear, assume it's RSSI if negative, distance if positive
                            if value < 0:
                                rssi = value
                                distance = max(0, ((-1 * (rssi + 59)) / 2))
                            else:
                                distance = value
                                rssi = -59 + (value * -2)

                        devices.append({
                            'mac': device_type,  # Use extracted identifier
                            'rssi': rssi,
                            'entity_id': state.get('entity_id'),  # Use original case
                            'friendly_name': friendly_name,
                            'distance': distance
                        })

                        # Debug logging to see what we found
                        logger.debug(f"Found relevant device: {state.get('entity_id')} with value {value}")

            logger.info(f"Found {len(devices)} BLE/distance devices")
            return devices

        except Exception as e:
            logger.error(f"Failed to get BLE devices from Home Assistant: {str(e)}", exc_info=True)
            return []

    def get_bermuda_positions(self) -> List[Dict]:
        """Get device positions with more flexible matching."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            positions = []

            # Keywords that indicate position entities
            position_keywords = ['position', 'location', 'tracker', 'bermuda', 'mmwave', 'coordinates']

            for state in all_states:
                entity_id = state.get('entity_id', '').lower()
                attributes = state.get('attributes', {})

                # More flexible matching - look for position indicators in entity ID or attributes
                has_position_keyword = any(keyword in entity_id for keyword in position_keywords)
                has_position_attrs = ('position' in attributes or
                                      ('x' in attributes and 'y' in attributes) or
                                      ('latitude' in attributes and 'longitude' in attributes))

                if has_position_keyword or has_position_attrs:
                    # Try different formats of position data
                    position_data = {}

                    # Check for nested position attribute
                    if 'position' in attributes and isinstance(attributes['position'], dict):
                        position_data = attributes['position']

                    # Check for top-level x,y,z coordinates
                    elif 'x' in attributes and 'y' in attributes:
                        position_data = {
                            'x': attributes.get('x', 0),
                            'y': attributes.get('y', 0),
                            'z': attributes.get('z', 0)
                        }

                    # Check for lat/long coordinates and convert to x/y (simplified)
                    elif 'latitude' in attributes and 'longitude' in attributes:
                        # Very simple conversion - just for demonstration
                        position_data = {
                            'x': (attributes.get('longitude', 0) * 100),
                            'y': (attributes.get('latitude', 0) * 100),
                            'z': 0
                        }

                    # Skip if no position data found
                    if not position_data:
                        continue

                    # Extract device ID from entity
                    device_id = state.get('entity_id').replace('device_tracker.', '').replace('sensor.', '')

                    # Try to get a numeric position
                    try:
                        position = {
                            'x': float(position_data.get('x', 0)),
                            'y': float(position_data.get('y', 0)),
                            'z': float(position_data.get('z', 0))
                        }
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert position data to float for {state.get('entity_id')}")
                        continue

                    positions.append({
                        'device_id': device_id,
                        'position': position,
                        'entity_id': state.get('entity_id')  # Keep original entity ID for reference
                    })

                    # Debug logging
                    logger.debug(f"Found position entity: {state.get('entity_id')} at {position}")

            logger.info(f"Found {len(positions)} position entities")
            return positions

        except Exception as e:
            logger.error(f"Error in get_bermuda_positions: {e}")
            return []

    def process_bluetooth_data(self, data: Dict) -> Dict:
        """Process and transform bluetooth data."""
        # This is a placeholder for data transformation logic
        # Implement based on specific requirements
        return data

    def get_sensors(self, domain: str = None, device_class: str = None) -> List[Dict]:
        """Get sensors matching domain and/or device_class."""
        try:
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()

            all_states = response.json()
            matching_sensors = []

            for state in all_states:
                entity_id = state.get('entity_id', '')
                attributes = state.get('attributes', {})

                # Filter by domain
                if domain and not entity_id.startswith(f"{domain}."):
                    continue

                # Filter by device class
                if device_class and attributes.get('device_class') != device_class:
                    continue

                matching_sensors.append({
                    'entity_id': entity_id,
                    'state': state.get('state'),
                    'attributes': attributes
                })

            return matching_sensors

        except Exception as e:
            logger.error(f"Failed to get sensors from Home Assistant: {str(e)}")
            return []

    def get_areas(self):
        """Get all areas from Home Assistant."""
        try:
            # First try WebSocket approach
            areas = get_area_registry(self.base_url, self.token)
            if areas:
                # Convert to expected format
                return [{"area_id": area.get("area_id", area.get("id")),
                        "name": area.get("name")}
                    for area in areas]

            # Fallback to HTTP API
            logger.warning("WebSocket area fetch failed, trying HTTP API")
            response = requests.get(f"{self.base_url}/api/config/area_registry",
                                    headers=self.headers)
            if response.status_code == 200:
                areas = response.json()
                return [{"area_id": area.get("area_id", area.get("id")),
                        "name": area.get("name")}
                    for area in areas]

        except Exception as e:
            logger.error(f"Failed to get areas: {e}")

        # Return default areas if all methods fail
        logger.warning("Using default areas")
        return [
            {"area_id": "lounge", "name": "Lounge"},
            {"area_id": "kitchen", "name": "Kitchen"},
            {"area_id": "master_bedroom", "name": "Master Bedroom"},
            {"area_id": "master_bathroom", "name": "Master Bathroom"},
            {"area_id": "office", "name": "Office"},
            {"area_id": "dining_room", "name": "Dining Room"},
            {"area_id": "sky_floor", "name": "Sky Floor"},
            {"area_id": "front_porch", "name": "Front Porch"},
            {"area_id": "laundry_room", "name": "Laundry Room"},
            {"area_id": "balcony", "name": "Balcony"},
            {"area_id": "backyard", "name": "Backyard"},
            {"area_id": "garage", "name": "Garage"},
            {"area_id": "dressing_room", "name": "Dressing Room"}
        ]

    def get_entity_registry_websocket(self):
        """Get entity registry from Home Assistant using WebSocket."""
        # Convert HTTP URL to WebSocket URL
        ws_url = self.base_url.replace('http://', 'ws://').replace('https://', 'wss://')
        ws_url = f"{ws_url}/api/websocket"

        entities = []
        response_received = Event()

        def on_message(ws, message):
            nonlocal entities
            try:
                data = json.loads(message)
                logger.debug(f"WebSocket message: {data.get('type')}")

                if data.get('type') == 'auth_required':
                    # Send authentication
                    ws.send(json.dumps({
                        "type": "auth",
                        "access_token": self.token
                    }))

                elif data.get('type') == 'auth_ok':
                    # Request entity registry
                    ws.send(json.dumps({
                        "id": 2,
                        "type": "config/entity_registry/list"
                    }))

                elif data.get('type') == 'result' and data.get('id') == 2:
                    # Got entity registry results
                    if 'result' in data and isinstance(data['result'], list):
                        entities = data['result']
                        logger.info(f"Received {len(entities)} entity registry entries via WebSocket")
                    response_received.set()
                    ws.close()

            except json.JSONDecodeError as e:
                logger.error(f"WebSocket message not valid JSON: {e}")
                logger.debug(f"Message content (first 100 chars): {message[:100]}")
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")

        def on_error(ws, error):
            logger.error(f"WebSocket error: {error}")
            response_received.set()

        def on_close(ws, close_status_code, close_msg):
            logger.debug("WebSocket closed")
            response_received.set()

        # Create WebSocket connection
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )

        # Start connection in a separate thread
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()

        # Wait for response or timeout
        response_received.wait(timeout=10)

        return entities

    def get_entity_registry(self):
        """Get entity registry with multiple fallback methods."""
        try:
            # Try WebSocket first (already working)
            registry = self.get_entity_registry_websocket()
            if registry:
                logger.info(f"Successfully retrieved {len(registry)} entities from WebSocket")
                return registry

            # Check if we're running in Supervisor environment
            is_supervisor = os.environ.get('SUPERVISOR_TOKEN') is not None

            if is_supervisor:
                # Try Supervisor-specific APIs
                logger.info("Running in Supervisor environment, trying Supervisor API")

                # Try direct Supervisor API endpoint
                try:
                    registry = self.safe_json_request("http://supervisor/core/api/states", self.headers)
                    if registry:
                        # Convert to registry format
                        result = []
                        for state in registry:
                            if 'entity_id' in state:
                                entity_id = state['entity_id']
                                domain, entity = entity_id.split('.', 1)
                                result.append({
                                    'entity_id': entity_id,
                                    'name': state.get('attributes', {}).get('friendly_name', entity),
                                    'domain': domain,
                                    'area_id': state.get('attributes', {}).get('area', None)
                                })
                        logger.info(f"Created {len(result)} registry entries from Supervisor states")
                        return result
                except Exception as e:
                    logger.warning(f"Supervisor states API failed: {e}")

            # Continue with existing fallbacks...
            # Try GET API
            logger.info("Trying GET API for entity registry")
            registry = self.safe_json_request(f"{self.base_url}/api/config/entity_registry", self.headers)
            if registry:
                return registry

            # Try POST API - some versions use this
            logger.info("Trying POST API for entity registry")
            response = requests.post(
                f"{self.base_url}/api/config/entity_registry/list",
                headers=self.headers
            )
            if response.status_code == 200:
                return response.json()

            # Final fallback - extract from states
            logger.info("Falling back to state-based entity registry")
            states = self.safe_json_request(f"{self.base_url}/api/states", self.headers)

            # Convert states to registry-like format
            registry = []
            for state in states:
                if 'entity_id' in state:
                    # Extract basic info
                    entity_id = state['entity_id']
                    domain, entity = entity_id.split('.', 1)

                    # Create registry entry
                    registry.append({
                        'entity_id': entity_id,
                        'name': state.get('attributes', {}).get('friendly_name', entity),
                        'domain': domain,
                        'area_id': state.get('attributes', {}).get('area', None)
                    })

            logger.info(f"Created {len(registry)} registry entries from states")
            return registry

        except Exception as e:
            logger.error(f"All entity registry methods failed: {e}")
            return []

    def get_entities(self, domain: Optional[str] = None, name_contains: Optional[str] = None) -> List[Dict]:
        """
        Get entities from Home Assistant, optionally filtered by domain and name.

        Args:
            domain: Optional domain to filter entities (e.g., 'sensor', 'light')
            name_contains: Optional string to filter entities by name or ID

        Returns:
            List of entity dictionaries
        """
        try:
            # Get all entities
            response = requests.get(f"{self.base_url}/api/states", headers=self.headers)
            response.raise_for_status()
            entities = response.json()

            # Apply domain filter if specified
            if (domain):
                entities = [e for e in entities if e.get('entity_id', '').startswith(f"{domain}.")]

            # Apply name filter if specified
            if (name_contains):
                name_contains = name_contains.lower()
                entities = [e for e in entities if (
                    name_contains in e.get('entity_id', '').lower() or
                    name_contains in e.get('attributes', {}).get('friendly_name', '').lower()
                )]

            logger.debug(f"Found {len(entities)} entities matching filters: domain={domain}, name_contains={name_contains}")
            return entities

        except requests.exceptions.RequestException as e:
            logger.error(f"Error getting entities from Home Assistant: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_entities: {e}")
            return []

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        """Get a specific entity from Home Assistant."""
        try:
            response = requests.get(f"{self.base_url}/api/states/{entity_id}", headers=self.headers)
            if response.status_code == 200:
                return response.json()
            logger.warning(f"Entity {entity_id} not found, status code: {response.status_code}")
            return None
        except Exception as e:
            logger.error(f"Error getting entity {entity_id}: {e}")
            return None

    def get_area_coordinates(self, area_id: str) -> Optional[Dict[str, float]]:
        """
        Get coordinates for an area if available.
        This usually requires a custom component or blueprint that stores area coordinates.
        """
        try:
            # Try to get coordinates from area attributes
            # This is implementation dependent - you may need to adjust based on your setup

            # First check if there's a dedicated area position entity
            area_pos_entities = self.get_entities("sensor", f"area_position_{area_id}")
            if area_pos_entities:
                entity = area_pos_entities[0]
                attrs = entity.get('attributes', {})
                if 'x' in attrs and 'y' in attrs:
                    return {
                        'x': float(attrs['x']),
                        'y': float(attrs['y']),
                        'z': float(attrs.get('z', 0))
                    }

            # Alternatively, look for entities in this area that might have position
            area_entities = []

            # Get entities that might have area_id in their attributes
            all_entities = self.get_entities()
            for entity in all_entities:
                entity_area = entity.get('attributes', {}).get('area_id')
                if entity_area == area_id:
                    area_entities.append(entity)

            # Check if any entity has position information
            for entity in area_entities:
                attrs = entity.get('attributes', {})
                if all(k in attrs for k in ['x', 'y']):
                    return {
                        'x': float(attrs['x']),
                        'y': float(attrs['y']),
                        'z': float(attrs.get('z', 0))
                    }

            # If no position found, generate a placeholder position based on area index
            areas = self.get_areas()
            if areas:
                for i, area in enumerate(areas):
                    if area.get('area_id') == area_id:
                        # Generate a grid layout
                        grid_size = 5  # 5 areas per row
                        row = i // grid_size
                        col = i % grid_size
                        return {
                            'x': col * 5.0,  # 5 meter spacing
                            'y': row * 5.0,
                            'z': 0.0
                        }

            logger.warning(f"No coordinates found for area {area_id}")
            return None

        except Exception as e:
            logger.error(f"Error getting area coordinates for {area_id}: {e}")
            return None

    def get_sensor_entities(self) -> List[Dict]:
        """
        Get all sensor entities with position, distance, or RSSI data.
        More comprehensive than get_private_ble_devices, with area_id integration.
        """
        try:
            # Get all states for relevant domains
            url = f"{self.base_url}/api/states"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            all_states = response.json()

            # Get entity registry for area_id lookup
            entity_registry = {}
            try:
                registry_entries = self.get_entity_registry()
                for entry in registry_entries:
                    if isinstance(entry, dict) and 'entity_id' in entry:
                        entity_registry[entry['entity_id']] = entry
                logger.info(f"Loaded {len(entity_registry)} entries from entity registry")
            except Exception as e:
                logger.warning(f"Error loading entity registry: {e}")

            sensor_entities = []
            relevant_domains = ['sensor', 'device_tracker', 'binary_sensor']

            # Keywords that might indicate relevant entities
            position_keywords = ['position', 'location', 'coordinates', 'xy', 'xyz']
            distance_keywords = ['distance', 'range', 'proximity', 'meters']
            rssi_keywords = ['rssi', 'signal', 'strength', 'ble', 'bluetooth']

            for state in all_states:
                entity_id = state.get('entity_id', '')
                domain = entity_id.split('.')[0] if '.' in entity_id else ''

                # Skip if not in relevant domains
                if domain not in relevant_domains:
                    continue

                attributes = state.get('attributes', {})
                entity_state = state.get('state')

                # Initialize entity data
                entity_data = {
                    'entity_id': entity_id,
                    'state': entity_state,
                    'attributes': attributes.copy(),  # Copy to avoid modifying original
                    'domain': domain
                }

                # Add area_id from registry if available
                registry_entry = entity_registry.get(entity_id)
                if registry_entry and 'area_id' in registry_entry:
                    entity_data['area_id'] = registry_entry['area_id']
                elif 'area_id' in attributes:
                    entity_data['area_id'] = attributes['area_id']

                # Determine entity type and extract relevant data

                # 1. Check for direct position data
                has_position = False
                if 'position' in attributes and isinstance(attributes['position'], dict):
                    position = attributes['position']
                    if 'x' in position and 'y' in position:
                        entity_data['position_data'] = {
                            'x': float(position['x']),
                            'y': float(position['y']),
                            'z': float(position.get('z', 0))
                        }
                        has_position = True

                # Also check for top-level position attributes
                elif all(k in attributes for k in ['x', 'y']):
                    entity_data['position_data'] = {
                        'x': float(attributes['x']),
                        'y': float(attributes['y']),
                        'z': float(attributes.get('z', 0))
                    }
                    has_position = True

                # Or lat/long coordinates
                elif all(k in attributes for k in ['latitude', 'longitude']):
                    # Convert to x/y coordinates (simplified)
                    # In a real implementation, use proper geo to cartesian conversion
                    entity_data['position_data'] = {
                        'x': float(attributes['longitude']) * 100,
                        'y': float(attributes['latitude']) * 100,
                        'z': float(attributes.get('elevation', 0))
                    }
                    has_position = True

                # 2. Check for distance data
                has_distance = False
                if 'distance' in attributes:
                    entity_data['distance'] = float(attributes['distance'])
                    has_distance = True
                elif any(k in entity_id for k in distance_keywords):
                    try:
                        # If entity_id suggests this is distance and state is numeric
                        entity_data['distance'] = float(entity_state)
                        has_distance = True
                    except (ValueError, TypeError):
                        pass

                # Also check for source_id for distance readings
                if has_distance and 'source_id' in attributes:
                    entity_data['source_id'] = attributes['source_id']
                elif has_distance and ('scanner_id' in attributes):
                    entity_data['source_id'] = attributes['scanner_id']

                # For Bermuda format: sensor.<device>_distance_<scanner>
                if '_distance_' in entity_id and has_distance:
                    parts = entity_id.split('_distance_')
                    if len(parts) == 2:
                        device_part = parts[0].split('sensor.')[1] if parts[0].startswith('sensor.') else parts[0]
                        scanner_id = parts[1]
                        entity_data['device_id'] = device_part
                        entity_data['source_id'] = scanner_id

                # 3. Check for RSSI data
                has_rssi = False
                if 'rssi' in attributes:
                    try:
                        entity_data['rssi'] = float(attributes['rssi'])
                        has_rssi = True
                    except (ValueError, TypeError):
                        if attributes['rssi'] not in (None, 'unavailable', 'unknown'):
                            logger.debug(f"Non-numeric RSSI value in {entity_id}: {attributes['rssi']}")

                # 4. Check for transmission power (tx_power)
                if 'tx_power' in attributes:
                    try:
                        entity_data['tx_power'] = float(attributes['tx_power'])
                    except (ValueError, TypeError):
                        pass

                # 5. Extract MAC address or device identifier
                for id_field in ['mac', 'address', 'uuid', 'bluetooth_address']:
                    if id_field in attributes:
                        entity_data['device_id'] = attributes[id_field]
                        break

                # If we don't have a device_id yet, try to extract from entity_id
                if 'device_id' not in entity_data and '_' in entity_id:
                    parts = entity_id.split('_')
                    # Try to extract a meaningful identifier - often the last part
                    if len(parts) >= 3:
                        entity_data['device_id'] = parts[-1]

                # Determine if this entity is relevant for our purposes
                is_relevant = (
                    has_position or
                    has_distance or
                    has_rssi or
                    any(keyword in entity_id.lower() for keyword in position_keywords + distance_keywords + rssi_keywords)
                )

                if is_relevant:
                    sensor_entities.append(entity_data)

            logger.info(f"Found {len(sensor_entities)} relevant sensor entities")

            # Log some examples of what we found
            if sensor_entities and logger.isEnabledFor(logging.DEBUG):
                examples = sensor_entities[:min(3, len(sensor_entities))]
                for example in examples:
                    logger.debug(f"Example entity: {example['entity_id']}")
                    if 'position_data' in example:
                        logger.debug(f"  Position: {example['position_data']}")
                    if 'distance' in example:
                        logger.debug(f"  Distance: {example['distance']}")
                    if 'rssi' in example:
                        logger.debug(f"  RSSI: {example['rssi']}")

            return sensor_entities

        except Exception as e:
            logger.error(f"Error fetching sensor entities: {e}", exc_info=True)
            return []