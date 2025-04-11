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
        try:
            conn = get_sqlite_connection()
            conn.execute("SELECT 1")  # Simple query
            conn.close()
            db_status = True
        except Exception as e_db:
            logger.error(f"SQLite connection test failed: {e_db}")
            db_status = False

        status = {
            'status': 'healthy' if db_status else 'unhealthy',
            'database': 'connected' if db_status else 'disconnected',
        }

        return jsonify(status), 200 if status['status'] == 'healthy' else 503

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 503

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
        # Use the existing instance instead of creating a new one
        blueprint = blueprint_generator.get_latest_blueprint()

        if not blueprint:
            logger.warning("No blueprint found in database")
            return jsonify({'error': 'No blueprint found'}), 404

        # Add debug logging
        logger.info(f"Returning blueprint with {len(blueprint.get('rooms', []))} rooms")
        return jsonify(blueprint)
    except Exception as e:
        logger.error(f"Error retrieving blueprint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint/status', methods=['GET'])
def get_blueprint_status():
    """Get the status of the blueprint generation."""
    try:
        # Get the status of the generation process
        status = blueprint_generator.get_status()

        return jsonify(status)
    except Exception as e:
        logger.error(f"Failed to get blueprint status: {str(e)}")
        return jsonify({'error': str(e)}), 500

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
        logger.error(f"Wall prediction model training failed: {str(e)}")
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
    """Generate a default blueprint based on the actual home areas."""
    try:
        # Get parameters from request
        params = request.json or {}
        include_outside = params.get('include_outside', True)

        # Your actual home areas with device counts to estimate room sizes
        areas = {
            # Outside areas
            'outside': [
                {'name': 'Backyard', 'devices': 10, 'floor': 0},
                {'name': 'Balcony', 'devices': 8, 'floor': 0},
                {'name': 'Driveway', 'devices': 4, 'floor': 0},
                {'name': 'Front Porch', 'devices': 7, 'floor': 0},
                {'name': 'Garage', 'devices': 6, 'floor': 0}
            ],
            # First floor areas
            'first_floor': [
                {'name': 'Bathroom', 'devices': 5, 'floor': 1},
                {'name': 'Christian Room', 'devices': 3, 'floor': 1},
                {'name': 'Dining Room', 'devices': 14, 'floor': 1},
                {'name': 'Kitchen', 'devices': 14, 'floor': 1},
                {'name': 'Laundry Room', 'devices': 5, 'floor': 1},
                {'name': 'Lounge', 'devices': 4, 'floor': 1},
                {'name': 'Master Bathroom', 'devices': 9, 'floor': 1},
                {'name': 'Master Bedroom', 'devices': 12, 'floor': 1},
                {'name': 'Nova Room', 'devices': 5, 'floor': 1}
            ],
            # Second floor areas
            'second_floor': [
                {'name': 'Dressing Room', 'devices': 2, 'floor': 2},
                {'name': 'Office', 'devices': 17, 'floor': 2},
                {'name': 'Sky Floor', 'devices': 1, 'floor': 2}
            ]
        }

        # Select areas to include
        included_areas = []
        if include_outside:
            included_areas.extend(areas['outside'])
        included_areas.extend(areas['first_floor'])
        included_areas.extend(areas['second_floor'])

        # Calculate room sizes based on device count
        # More devices = larger room
        total_devices = sum(area['devices'] for area in included_areas)
        base_area = 100  # Total floor space to distribute

        for area in included_areas:
            # Scale size based on device count
            size_factor = (area['devices'] / total_devices) * 3 + 0.5  # Ensure even small rooms get some size

            # Calculate width and length (sized proportionally to devices)
            area['width'] = max(3, round(size_factor * 2, 1))  # Minimum 3m width
            area['length'] = max(3, round(size_factor * 2.5, 1))  # Minimum 3m length

        # Create a layout for the rooms
        default_rooms = []
        device_positions = {}

        # Position generation
        x_pos = 0
        y_pos = 0
        max_width = 0

        # Create room layouts by floor
        for floor in [0, 1, 2]:
            floor_areas = [a for a in included_areas if a['floor'] == floor]
            if not floor_areas:
                continue

            # Reset x position for new floor
            x_pos = 0
            # If ground floor, we'll place some areas (outside) below y=0
            if floor == 0:
                y_pos = -20  # Outside areas
            elif floor == 1:
                y_pos = 0   # First floor
            else:
                y_pos = 25  # Second floor

            # Arrange rooms in rows (simple layout)
            current_row_y = y_pos
            current_row_height = 0

            for i, area in enumerate(floor_areas):
                width = area['width']
                length = area['length']

                # Try to arrange in rows
                if x_pos + width > 25:  # Start a new row if we exceed 25m width
                    x_pos = 0
                    current_row_y += current_row_height + 1  # 1m corridor
                    current_row_height = 0

                # Create unique room ID
                room_id = f"room_{area['name'].lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"

                # Define room center and dimensions
                center_x = x_pos + width/2
                center_y = current_row_y + length/2
                center_z = floor * 3 + 1.5  # 3m per floor, centers at 1.5m above floor

                # Calculate bounds
                min_x, min_y, min_z = x_pos, current_row_y, floor * 3
                max_x, max_y, max_z = x_pos + width, current_row_y + length, floor * 3 + 3

                # Add room to blueprint
                default_rooms.append({
                    'id': room_id,
                    'name': area['name'],
                    'floor': floor,
                    'center': {
                        'x': center_x,
                        'y': center_y,
                        'z': center_z
                    },
                    'dimensions': {
                        'width': width,
                        'length': length,
                        'height': 3  # Standard height
                    },
                    'bounds': {
                        'min': {'x': min_x, 'y': min_y, 'z': min_z},
                        'max': {'x': max_x, 'y': max_y, 'z': max_z}
                    },
                    'type': 'outdoor' if floor == 0 else 'indoor'
                })

                # Add reference points for each room
                device_id = f"reference_{area['name'].lower().replace(' ', '_')}"
                position = {
                    'x': center_x,
                    'y': center_y,
                    'z': center_z,
                    'accuracy': 1.0,
                    'source': 'default_blueprint'
                }

                # Store for batch saving
                device_positions[device_id] = position

                # Update position for next room and track row height
                x_pos += width + 1  # 1m gap between rooms
                current_row_height = max(current_row_height, length)
                max_width = max(max_width, x_pos)

        # Create a basic wall layout
        walls = []

        # Create a blueprint structure
        blueprint = {
            'rooms': default_rooms,
            'walls': walls,
            'floors': [
                {'level': 0, 'name': 'Outside', 'height': 0, 'rooms': [r['id'] for r in default_rooms if r['floor'] == 0]},
                {'level': 1, 'name': 'First Floor', 'height': 3, 'rooms': [r['id'] for r in default_rooms if r['floor'] == 1]},
                {'level': 2, 'name': 'Second Floor', 'height': 6, 'rooms': [r['id'] for r in default_rooms if r['floor'] == 2]}
            ],
            'generated': True,
            'timestamp': datetime.now().isoformat(),
            'source': 'default_generator',
            'status': 'active' # Required by your existing _save_blueprint method
        }

        # Save device positions to database
        for device_id, position in device_positions.items():
            # Use the initialized bluetooth_processor instance instead of importing the function
            bluetooth_processor.save_device_position(device_id, position)
            logger.info(f"Created reference point {device_id} at position {position}")

        # Use blueprint generator to save the blueprint
        # This uses your existing _save_blueprint method
        blueprint_generator.latest_generated_blueprint = blueprint
        saved = blueprint_generator._save_blueprint(blueprint)

        # Get a blueprint ID (can be made up since we don't have a return ID)
        blueprint_id = str(uuid.uuid4())

        return jsonify({
            'success': saved,
            'blueprint_id': blueprint_id,
            'message': f'Default blueprint generated with {len(default_rooms)} rooms based on your Home Assistant areas',
            'reference_points_created': len(device_positions),
            'details': {
                'rooms': [room['name'] for room in default_rooms],
                'floors': [floor['name'] for floor in blueprint['floors']]
            }
        })

    except Exception as e:
        logger.error(f"Default blueprint generation failed: {str(e)}")
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