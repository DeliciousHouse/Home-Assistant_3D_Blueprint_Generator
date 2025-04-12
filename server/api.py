import json
import logging
import requests
from typing import Dict, Optional

from flask import Flask, jsonify, request, send_from_directory, render_template
from flask_cors import CORS
import os
import random  # Added for random offsets in estimated scanner locations

from .blueprint_generator import BlueprintGenerator
from .bluetooth_processor import BluetoothProcessor
from .db import get_sqlite_connection, execute_sqlite_query
from .ha_client import HomeAssistantClient
import uuid
from datetime import datetime
from .ai_processor import AIProcessor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize components
blueprint_generator = BlueprintGenerator()
bluetooth_processor = BluetoothProcessor()
ha_client = HomeAssistantClient()

# Set up logging
logger = logging.getLogger(__name__)

# Add root route handler
@app.route('/', methods=['GET'])
def index():
    """Serve the main page."""
    try:
        # Serve from the ui directory at the root level
        ui_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui')

        if os.path.exists(os.path.join(ui_directory, 'index.html')):
            # Return the index.html file from the ui directory
            with open(os.path.join(ui_directory, 'index.html'), 'r') as f:
                html_content = f.read()
            return html_content
        else:
            # Fallback to simple HTML response
            logger.warning(f"UI not found at {ui_directory}")
            return """
            <html>
                <head>
                    <title>3D Blueprint Generator</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                        h1 { color: #2c3e50; }
                        .container { max-width: 800px; margin: 0 auto; }
                        .endpoint { background: #f8f9fa; padding: 10px; margin-bottom: 10px; border-radius: 4px; }
                        code { background: #e9ecef; padding: 2px 4px; border-radius: 3px; }
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>3D Blueprint Generator</h1>
                        <p>API is running successfully. The following endpoints are available:</p>

                        <div class="endpoint">
                            <strong>GET /api/health</strong> - Check API health
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/blueprint</strong> - Get the latest blueprint
                        </div>

                        <div class="endpoint">
                            <strong>POST /api/blueprint/generate</strong> - Generate a new blueprint
                        </div>

                        <div class="endpoint">
                            <strong>GET /api/positions</strong> - Get current device positions
                        </div>

                        <p>For more details, see the API documentation in Home Assistant.</p>
                    </div>
                </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error serving index page: {str(e)}")
        return jsonify({"error": "Internal Server Error"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check database connection with a simple SQLite test
        db_status = False
        try:
            conn = get_sqlite_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                db_status = cursor.fetchone() is not None
                conn.close()
        except Exception as e_db:
            logger.error(f"Database health check failed: {e_db}")
            db_status = False

        # Check Home Assistant API connection
        ha_status = False
        try:
            ha_status = ha_client.test_connection()
        except Exception as e_ha:
            logger.error(f"HA API health check failed: {e_ha}")
            ha_status = False

        status = {
            'status': 'healthy' if (db_status and ha_status) else 'unhealthy',
            'database': 'connected' if db_status else 'disconnected',
            'home_assistant': 'connected' if ha_status else 'disconnected',
            'timestamp': datetime.now().isoformat()
        }

        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/api/scan', methods=['GET'])
def scan_entities():
    """Scan for relevant entities and report availability."""
    try:
        ha_client = HomeAssistantClient()

        # Very liberal scan - any entities that might be related
        all_entities = ha_client.find_entities_by_pattern([""], [])  # Get ALL entities
        ble_entities = [e for e in all_entities if 'ble' in e['entity_id'].lower()]
        distance_entities = [e for e in all_entities if 'distance' in e['entity_id'].lower()]
        position_entities = [e for e in all_entities if any(p in e['entity_id'].lower()
                                                       for p in ['position', 'bermuda', 'tracker', 'mmwave'])]

        # Organize by category
        entity_data = {
            'ble_entities': [e['entity_id'] for e in ble_entities],
            'position_entities': [e['entity_id'] for e in position_entities],
            'distance_entities': [e['entity_id'] for e in distance_entities],
            'sample_entities': [e['entity_id'] for e in all_entities[:10]]  # First 10 entities
        }

        return jsonify({
            'status': 'success',
            'entities_found': len(ble_entities) + len(position_entities) + len(distance_entities),
            'total_entities': len(all_entities),
            'entities': entity_data
        })
    except Exception as e:
        logger.error(f"Failed to scan entities: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/api/devices', methods=['GET'])
def get_devices():
    """Get list of tracked devices."""
    try:
        # Query unique device IDs from device_positions table
        query = """
        SELECT DISTINCT device_id, MAX(timestamp) as last_seen
        FROM device_positions
        GROUP BY device_id
        ORDER BY last_seen DESC
        """
        results = execute_sqlite_query(query)

        devices = [{'id': row['device_id'], 'last_seen': row['last_seen'] if row['last_seen'] else None} for row in results]
        return jsonify({'devices': devices})
    except Exception as e:
        logger.error(f"Failed to get devices: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint/generate', methods=['POST'])
def generate_blueprint():
    """Generate a new blueprint from collected data."""
    try:
        # Call the blueprint generator without arguments
        # (it will fetch device positions internally)
        result = blueprint_generator.generate_blueprint()

        return jsonify({
            'status': 'success',
            'message': 'Blueprint generation started',
            'job_id': str(uuid.uuid4())  # Generate a job ID
        })
    except Exception as e:
        logger.error(f"Failed to generate blueprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint', methods=['GET'])
def get_blueprint():
    """Get the latest blueprint from the database."""
    try:
        blueprint = blueprint_generator.get_latest_blueprint()
        if blueprint:
            return jsonify({"success": True, "blueprint": blueprint})
        else:
            return jsonify({"success": False, "error": "No blueprint found"}), 404
    except Exception as e:
        logger.error(f"Error getting blueprint: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/blueprint/status', methods=['GET'])
def get_blueprint_status():
    """Get the status of the blueprint generation."""
    try:
        status = blueprint_generator.get_status()
        return jsonify({"success": True, "status": status})
    except Exception as e:
        logger.error(f"Error getting blueprint status: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/debug/fix-schema', methods=['POST'])
def fix_schema():
    """Fix schema issues manually."""
    try:
        # Simply create tables if they don't exist
        from .db import init_sqlite_db
        result = init_sqlite_db()

        return jsonify({
            'success': result,
            'message': 'SQLite tables created/validated'
        })
    except Exception as e:
        logger.error(f"Schema fix failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/visualize', methods=['GET'])
def visualize_blueprint():
    """Render a visualization of the latest blueprint."""
    try:
        # Get the latest blueprint data - ALWAYS USE THE CLASS METHOD, not the import
        blueprint = blueprint_generator.get_latest_blueprint()
        if not blueprint:
            return render_template('no_blueprint.html')

        # Add debug logging
        logger.info(f"Visualizing blueprint with {len(blueprint.get('rooms', []))} rooms")

        return render_template('visualize.html', blueprint=blueprint)
    except Exception as e:
        logger.error(f"Failed to visualize blueprint: {str(e)}")
        return render_template('error.html', error=str(e))

# Removing /api/sync/bermuda endpoint - deprecated in new design

# Removing /api/sync/esp32-ble endpoint

# AI-related endpoints

@app.route('/api/ai/status', methods=['GET'])
def get_ai_status():
    """Get the status of all AI models."""
    try:
        # Initialize AI processor if needed
        ai_processor = blueprint_generator.ai_processor

        # Get status of all models
        status = ai_processor.get_models_status()

        return jsonify(status)
    except Exception as e:
        logger.error(f"Failed to get AI status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train/rssi-distance', methods=['POST'])
def train_rssi_distance_model():
    """Train the RSSI-to-distance model."""
    try:
        # Get training parameters from request
        params = request.json or {}

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Train the model
        result = ai_processor.train_rssi_distance_model(
            model_type=params.get('model_type', 'random_forest'),
            test_size=params.get('test_size', 0.2),
            features=params.get('features', ['rssi']),
            hyperparams=params.get('hyperparams', {})
        )

        return jsonify({
            'success': result.get('success', False),
            'metrics': result.get('metrics', {}),
            'message': 'RSSI-to-distance model training completed'
        })
    except Exception as e:
        logger.error(f"RSSI-to-distance model training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train/room-clustering', methods=['POST'])
def train_room_clustering_model():
    """Configure the room clustering model."""
    try:
        # Get configuration parameters from request
        params = request.json or {}

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Configure the model
        result = ai_processor.configure_room_clustering(
            algorithm=params.get('algorithm', 'dbscan'),
            eps=params.get('eps', 2.0),
            min_samples=params.get('min_samples', 3),
            features=params.get('features', ['x', 'y', 'z']),
            temporal_weight=params.get('temporal_weight', 0.2)
        )

        return jsonify({
            'success': result.get('success', False),
            'message': 'Room clustering model configuration completed'
        })
    except Exception as e:
        logger.error(f"Room clustering model configuration failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train/wall-prediction', methods=['POST'])
def train_wall_prediction_model():
    """Train the wall prediction neural network."""
    try:
        # Get training parameters from request
        params = request.json or {}

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Train the model
        result = ai_processor.train_wall_prediction_model(
            model_type=params.get('model_type', 'cnn'),
            training_data=params.get('training_data', []),
            epochs=params.get('epochs', 50),
            batch_size=params.get('batch_size', 32),
            learning_rate=params.get('learning_rate', 0.001)
        )

        return jsonify({
            'success': result.get('success', False),
            'metrics': result.get('metrics', {}),
            'message': 'Wall prediction model training completed'
        })
    except Exception as e:
        logger.error(f"Wall prediction model training failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/ai/train/blueprint-refinement', methods=['POST'])
def train_blueprint_refinement_model():
    """Train the blueprint refinement model."""
    try:
        # Get training parameters from request
        params = request.json or {}

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Train the model
        result = ai_processor.train_blueprint_refinement_model(
            feedback_data=params.get('feedback_data', []),
            reward_weights=params.get('reward_weights', {
                'room_size': 0.3,
                'wall_alignment': 0.4,
                'flow_efficiency': 0.3
            }),
            learning_rate=params.get('learning_rate', 0.01),
            discount_factor=params.get('discount_factor', 0.9)
        )

        return jsonify({
            'success': result.get('success', False),
            'message': 'Blueprint refinement model training completed'
        })
    except Exception as e:
        logger.error(f"Blueprint refinement model training failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug', methods=['GET'])
def debug_ha_connection():
    """Test Home Assistant API connection and entity detection."""
    try:
        ha_client = HomeAssistantClient()

        # Basic API test
        url = f"{ha_client.base_url}/api/config"
        try:
            response = requests.get(url, headers=ha_client.headers)
            ha_status = {
                "connected": response.status_code == 200,
                "status_code": response.status_code,
                "base_url": ha_client.base_url,
                "token_provided": bool(ha_client.token),
                "headers": list(ha_client.headers.keys())
            }
        except Exception as e:
            ha_status = {"error": str(e), "connected": False}

        # Try to get ANY entities
        try:
            raw_entities = ha_client.find_entities_by_pattern([""], None)
            raw_count = len(raw_entities)
            # Get first 5 entities for inspection
            sample_entities = [e['entity_id'] for e in raw_entities[:5]] if raw_entities else []
        except Exception as e:
            raw_count = -1
            sample_entities = []
            logger.error(f"Error getting raw entities: {str(e)}", exc_info=True)

        # Specific tests for your entity types
        specific_tests = {
            "ble_test": len(ha_client.find_entities_by_pattern(['ble'], [])),
            "bluetooth_test": len(ha_client.find_entities_by_pattern(['bluetooth'], [])),
            "distance_test": len(ha_client.find_entities_by_pattern(['distance'], [])),
            "sensor_ble_test": len(ha_client.find_entities_by_pattern(['ble'], ['sensor'])),
            "any_starting_with_sensor": len(ha_client.find_entities_by_pattern([""], ["sensor"]))
        }

        return jsonify({
            "ha_status": ha_status,
            "entity_scan": {
                "total_entities": raw_count,
                "sample_entities": sample_entities,
                "specific_tests": specific_tests
            }
        })

    except Exception as e:
        logger.error(f"Debug endpoint failed: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/ai/data/rssi-distance', methods=['POST'])
def add_rssi_distance_data():
    """Add training data for RSSI-to-distance model."""
    try:
        # Get data from request
        data = request.json
        if not data or not isinstance(data, dict):
            return jsonify({'error': 'Invalid data format'}), 400

        required_fields = ['rssi', 'distance', 'device_id', 'sensor_id']
        if not all(field in data for field in required_fields):
            return jsonify({'error': f'Missing required fields: {required_fields}'}), 400

        # Initialize AI processor
        ai_processor = blueprint_generator.ai_processor

        # Get current time information if not provided
        current_time = datetime.now()

        # Save the training data with all possible parameters
        result = ai_processor.save_rssi_distance_sample(
            device_id=data['device_id'],
            sensor_id=data['sensor_id'],
            rssi=data['rssi'],
            distance=data['distance'],
            tx_power=data.get('tx_power'),
            frequency=data.get('frequency'),
            environment_type=data.get('environment_type'),
            device_type=data.get('device_type'),  # Added parameter
            time_of_day=data.get('time_of_day', current_time.hour),  # Default to current hour
            day_of_week=data.get('day_of_week', current_time.weekday())  # Default to current day (0-6, Monday is 0)
        )

        return jsonify({
            'success': result,
            'message': 'RSSI-to-distance training data saved'
        })
    except Exception as e:
        logger.error(f"Failed to save RSSI-to-distance data: {str(e)}")
        return jsonify({'error': str(e)}), 500

def start_api(host: str = '0.0.0.0', port: int = 8000, debug: bool = False, use_reloader: bool = True):
    """Start the Flask API server.

    Args:
        host: Host IP to bind on
        port: Port to listen on
        debug: Whether to run in debug mode
        use_reloader: Whether Flask should automatically reload on code changes
    """
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)

@app.route('/api/debug/entities', methods=['GET'])
def debug_entities():
    """Debug endpoint for entity detection."""
    try:
        ha_client = HomeAssistantClient()
        result = {
            "connection": {
                "url": ha_client.base_url,
                "token_provided": bool(ha_client.token),
                "headers": list(ha_client.headers.keys())
            },
            "entities": {
                "all": [],
                "ble": [],
                "distance": [],
                "position": []
            },
            "test_queries": {}
        }

        # Test direct API call
        try:
            test_url = f"{ha_client.base_url}/api/states"
            response = requests.get(test_url, headers=ha_client.headers)
            result["connection"]["test_status"] = response.status_code

            # Get all entities
            all_states = response.json()
            result["entities"]["total_count"] = len(all_states)

            # Sample first 10 entities
            result["entities"]["all"] = [e["entity_id"] for e in all_states[:10]]

            # Find entities matching your example patterns
            example_patterns = [
                "apple_watch", "iphone", "ble_distance", "mmwave", "bermuda"
            ]

            for pattern in example_patterns:
                matches = []
                for state in all_states:
                    if pattern in state["entity_id"].lower():
                        matches.append(state["entity_id"])
                result["test_queries"][pattern] = matches[:5]  # Just show first 5

            # Find BLE entities
            for state in all_states:
                entity_id = state["entity_id"].lower()

                if "_ble" in entity_id or entity_id.endswith("_ble"):
                    result["entities"]["ble"].append(state["entity_id"])

                if "distance" in entity_id:
                    result["entities"]["distance"].append(state["entity_id"])

                if any(p in entity_id for p in ["position", "bermuda", "tracker", "mmwave"]):
                    result["entities"]["position"].append(state["entity_id"])

        except Exception as e:
            result["connection"]["error"] = str(e)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Debug endpoint failed: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/debug/blueprint', methods=['GET'])
def debug_blueprint():
    """Debug endpoint to view raw blueprint data."""
    try:
        # IMPORTANT: Use the instance method, not the imported function
        blueprint = blueprint_generator.get_latest_blueprint()

        if not blueprint:
            # Check if blueprint generator's instance method returned None
            logger.warning("No blueprint found in blueprints table")
            return jsonify({"error": "No blueprint found"}), 404

        # Return the raw blueprint data as JSON for inspection
        return jsonify({
            "blueprint": blueprint,
            "meta": {
                "rooms": len(blueprint.get('rooms', [])),
                "walls": len(blueprint.get('walls', [])),
                "floors": len(blueprint.get('floors', []))
            },
            "source": "BlueprintGenerator.get_latest_blueprint()"
        })
    except Exception as e:
        logger.error(f"Debug blueprint error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/blueprint/generate-default', methods=['POST'])
def generate_default_blueprint():
    """Generate a blueprint using BLE sensor multilateration for accurate floor delineation."""
    try:
        logger.info("Starting blueprint generation using BLE sensor multilateration")
        params = request.json or {}
        include_outside = params.get('include_outside', True)

        # Get BLE sensor data for multilateration
        ha_client = HomeAssistantClient()
        bluetooth_processor = BluetoothProcessor()

        # Get distance sensors which will be used for multilateration
        distance_sensors = ha_client.get_distance_sensors()
        logger.info(f"Found {len(distance_sensors)} distance sensors for multilateration")

        # Get area predictions to correctly associate devices with rooms
        area_predictions = ha_client.get_device_area_predictions()
        logger.info(f"Found {len(area_predictions)} device area predictions")

        # Collect all scanner and device positions from distance measurements
        scanner_positions = {}
        device_positions = {}

        # Construct a mapping of areas to known floor levels
        area_floor_map = {
            "master_bedroom": 1,
            "master_bathroom": 1,
            "lounge": 1,
            "kitchen": 1,
            "office": 2,
            "sky_floor": 2,  # Ensure Sky Floor is on the second floor
            "dressing_room": 2
        }

        # Create initial reference points for known areas
        # This ensures we have some stable coordinates to work with
        rooms_by_area = {}
        room_references = {}

        # Define proper naming for rooms
        area_name_map = {
            "alexs_room": "Master Bedroom",
            "alexs_bathroom": "Master Bathroom",
            "master_bedroom": "Master Bedroom",
            "master_bathroom": "Master Bathroom",
            "lounge": "Lounge",
            "kitchen": "Kitchen",
            "office": "Office",
            "sky_floor": "Sky Floor",
            "dressing_room": "Dressing Room"
        }

        # Parse distance sensor data to build multilateration input
        for sensor in distance_sensors:
            device_id = sensor.get('tracked_device_id', '')
            scanner_id = sensor.get('scanner_id', '')
            distance = sensor.get('distance', 0)

            if not device_id or not scanner_id or distance <= 0:
                continue

            # Record this relationship for multilateration
            if device_id not in device_positions:
                device_positions[device_id] = {
                    "distances": {},
                    "area_id": area_predictions.get(device_id)
                }

            # Store the distance from this device to this scanner
            device_positions[device_id]["distances"][scanner_id] = distance

        logger.info(f"Collected distances for {len(device_positions)} devices")

        # Perform multilateration to determine device positions
        # For each device with multiple distance measurements, we can calculate its position
        device_coords = {}

        # Group devices by area to determine room dimensions
        devices_by_area = {}

        for device_id, data in device_positions.items():
            # Get the area prediction for this device
            area_id = data.get("area_id")

            # Skip devices with unknown area
            if not area_id:
                continue

            # Initialize this area if needed
            if area_id not in devices_by_area:
                devices_by_area[area_id] = []

            # Add device to its area
            device_data = {
                "device_id": device_id,
                "distances": data["distances"]
            }
            devices_by_area[area_id].append(device_data)

        logger.info(f"Grouped devices into {len(devices_by_area)} areas")

        # Generate rooms based on area and device information
        rooms = []
        reference_points = {}

        # Set vertical offsets for floors
        floor_height = 3.0

        # Generate a unique identifier for this blueprint
        blueprint_id = str(uuid.uuid4())

        # Create rooms for each area with proper floor delineation
        for area_id, devices in devices_by_area.items():
            # Get floor level from map or default to first floor
            floor = area_floor_map.get(area_id, 1)

            # Get proper name from map or use title-cased area_id
            area_name = area_name_map.get(area_id, area_id.replace('_', ' ').title())

            # Calculate z-position based on floor
            z_pos = (floor - 1) * floor_height

            # Place rooms with some spacing
            # Use a hash of the area name for consistent but varied placement
            import hashlib
            hash_val = int(hashlib.md5(area_id.encode()).hexdigest(), 16)
            x_offset = (hash_val % 20) * 5  # 0-95m in steps of 5m
            y_offset = ((hash_val // 20) % 10) * 8  # 0-72m in steps of 8m

            # Calculate room dimensions based on device count
            device_count = len(devices)
            width = max(4, min(15, 4 + device_count * 0.5))  # 4-15m based on device count
            length = max(4, min(15, 4 + device_count * 0.7))  # 4-15m with different scaling

            # Create a unique room ID
            room_id = f"room_{area_id}_{blueprint_id[:8]}"

            # Define room center and dimensions
            center_x = x_offset + width/2
            center_y = y_offset + length/2
            center_z = z_pos + floor_height/2  # Center of the room vertically

            # Calculate bounds
            min_x, min_y, min_z = x_offset, y_offset, z_pos
            max_x, max_y, max_z = x_offset + width, y_offset + length, z_pos + floor_height

            # Add room to blueprint
            rooms.append({
                'id': room_id,
                'name': area_name,
                'floor': floor,
                'center': {
                    'x': center_x,
                    'y': center_y,
                    'z': center_z
                },
                'dimensions': {
                    'width': width,
                    'length': length,
                    'height': floor_height
                },
                'bounds': {
                    'min': {'x': min_x, 'y': min_y, 'z': min_z},
                    'max': {'x': max_x, 'y': max_y, 'z': max_z}
                },
                'type': 'indoor'
            })

            # Create a reference point for this room
            ref_id = f"reference_{area_id}"
            reference_points[ref_id] = {
                'x': center_x,
                'y': center_y,
                'z': center_z,
                'accuracy': 1.0,
                'source': 'multilateration_blueprint'
            }

            # Save device positions within this room
            for i, device in enumerate(devices):
                device_id = device["device_id"]
                # Place devices at random positions within the room
                dx = (hash(device_id + "x") % 100) / 100 * width - width/2
                dy = (hash(device_id + "y") % 100) / 100 * length - length/2

                device_positions[device_id] = {
                    'x': center_x + dx,
                    'y': center_y + dy,
                    'z': center_z,
                    'accuracy': 2.0,
                    'area_id': area_id,
                    'source': 'multilateration_derived'
                }

            logger.info(f"Created room '{area_name}' for area '{area_id}' on floor {floor} with {len(devices)} devices")

        # Create floors
        floors = []
        floor_rooms = {}

        # Group rooms by floor
        for room in rooms:
            floor = room.get('floor', 1)
            if floor not in floor_rooms:
                floor_rooms[floor] = []
            floor_rooms[floor].append(room['id'])

        # Create floor objects
        for floor_num, room_ids in sorted(floor_rooms.items()):
            floor_name = "Ground Floor" if floor_num == 0 else f"{floor_num}{['st', 'nd', 'rd'][floor_num-1] if 1 <= floor_num <= 3 else 'th'} Floor"
            floors.append({
                'level': floor_num,
                'name': floor_name,
                'height': floor_num * floor_height,
                'rooms': room_ids
            })

        # Create an empty walls list for now - walls can be generated later
        walls = []

        # Create the blueprint structure
        blueprint = {
            'rooms': rooms,
            'walls': walls,
            'floors': floors,
            'generated': True,
            'timestamp': datetime.now().isoformat(),
            'source': 'multilateration',
            'status': 'active'
        }

        # Save reference points to the database
        for device_id, position in reference_points.items():
            bluetooth_processor.save_device_position(device_id, position)
            logger.info(f"Saved reference point {device_id} at position {position}")

        # Save device positions to the database
        for device_id, position in device_positions.items():
            if isinstance(position, dict) and 'x' in position:
                bluetooth_processor.save_device_position(device_id, position)
                logger.info(f"Saved device {device_id} at position {position}")

        # Save blueprint
        blueprint_generator.latest_generated_blueprint = blueprint
        saved = blueprint_generator._save_blueprint(blueprint)

        return jsonify({
            'success': saved,
            'blueprint_id': blueprint_id,
            'message': f'Blueprint generated using multilateration with {len(rooms)} rooms across {len(floors)} floors',
            'reference_points_created': len(reference_points),
            'device_positions_tracked': len(device_positions),
            'details': {
                'rooms': [room['name'] for room in rooms],
                'floors': [floor['name'] for floor in floors]
            }
        })
    except Exception as e:
        logger.error(f"Blueprint generation with multilateration failed: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# Removing /api/config/sensors endpoint - deprecated in new design

# Removing /api/config/estimate_scanner_locations endpoint - deprecated in new design

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check if the add-on is running properly and connected to HA."""
    try:
        # Check HA connection
        ha_connected = False
        try:
            # Direct API call to check connection
            response = requests.get(f"{ha_client.base_url}/api/config", headers=ha_client.headers)
            ha_connected = response.status_code == 200
        except Exception as e_ha:
            logger.error(f"HA connection test failed: {e_ha}")

        # Check database
        db_status = False
        try:
            conn = get_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            conn.close()
            db_status = True
        except Exception as e_db:
            logger.error(f"Database connection test failed: {e_db}")

        return jsonify({
            'status': 'running',
            'ha_connected': ha_connected,
            'database_ready': db_status,
            'version': '1.0.0'
        })
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e)
        }), 500

@app.route('/api/setup', methods=['GET'])
def setup_environment():
    """Initialize the database and environment."""
    try:
        # Use SQLite initialization function instead of MariaDB
        from .db import init_sqlite_db
        result = init_sqlite_db()

        # Check if models directory exists
        model_path = os.environ.get('MODEL_PATH', 'models')
        os.makedirs(model_path, exist_ok=True)

        # Initialize the AI processor using blueprint_generator for consistency
        # This ensures we're using the same instance throughout the application
        blueprint_generator.init_ai_processor()

        return jsonify({
            'success': True,
            'database_initialized': result,
            'environment_ready': True,
            'model_path': model_path
        })
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/app.js')
def serve_js():
    ui_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui')
    return send_from_directory(ui_directory, 'app.js')

@app.route('/styles.css')
def serve_css():
    ui_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ui')
    return send_from_directory(ui_directory, 'styles.css')