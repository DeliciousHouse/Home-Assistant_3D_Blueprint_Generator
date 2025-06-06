import json
import logging
import requests
from typing import Dict, Optional
from flask import Flask, jsonify, request, send_from_directory, render_template, redirect, send_file
from flask_cors import CORS
import os
import random  # Added for random offsets in estimated scanner locations
import time
import sys
from pathlib import Path
import traceback
import threading

from .blueprint_generator import BlueprintGenerator
from .bluetooth_processor import BluetoothProcessor
from .db import get_sqlite_connection, execute_sqlite_query, get_latest_blueprint_from_sqlite
# Fix the import to use HAClient as HomeAssistantClient
from .ha_client import HAClient as HomeAssistantClient
from .config_loader import load_config
import uuid
from datetime import datetime
from .ai_processor import AIProcessor

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logger = logging.getLogger(__name__)

# Enable CORS for API endpoints
config = load_config()
cors_origins = config.get('api', {}).get('cors_origins', ['*'])
CORS(app, resources={r"/api/*": {"origins": cors_origins}})

# Initialize components
blueprint_generator = BlueprintGenerator()
bluetooth_processor = BluetoothProcessor()
ha_client = HomeAssistantClient()

@app.route('/')
def index():
    """Serve the main UI page or redirect to it."""
    # Check if we're accessed via ingress or direct URL
    # Ingress requests typically have a different path structure with multi-level paths
    if request.path == '/':
        return send_from_directory(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui'), 'index.html')
    else:
        # If this is a more complex path, redirect to the root
        return redirect('/')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from the UI directory."""
    ui_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ui')
    # Handle the case where index.html is explicitly requested
    if filename == 'index.html':
        return send_from_directory(ui_path, filename)
    return send_from_directory(ui_path, filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'version': '3.21'  # Updated to match version in repository.json and build.yaml
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    """Return the current configuration."""
    config = load_config()
    # Filter out any sensitive information
    safe_config = {
        'processing_params': config.get('processing_params', {}),
        'log_level': config.get('log_level', 'info'),
        'generation_settings': config.get('generation_settings', {}),
        'blueprint_validation': config.get('blueprint_validation', {})
    }
    return jsonify(safe_config)

@app.route('/api/blueprint/latest', methods=['GET'])
def get_latest_blueprint():
    """Get the latest generated blueprint."""
    try:
        blueprint = get_latest_blueprint_from_sqlite()
        if blueprint:
            return jsonify(blueprint)
        else:
            return jsonify({'error': 'No blueprints available'}), 404
    except Exception as e:
        logger.error(f"Error retrieving latest blueprint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint/generate', methods=['POST'])
def generate_blueprint():
    """Generate a new blueprint."""
    try:
        # Extract parameters from request
        params = request.json or {}

        # Start the generation in a background thread to not block the API
        def generate_async():
            try:
                success = blueprint_generator.generate_blueprint()
                logger.info(f"Blueprint generation {'succeeded' if success else 'failed'}")
            except Exception as e:
                logger.error(f"Error in async blueprint generation: {e}")

        thread = threading.Thread(target=generate_async)
        thread.daemon = True
        thread.start()

        return jsonify({
            'status': 'generation_started',
            'message': 'Blueprint generation has been started in the background'
        })
    except Exception as e:
        logger.error(f"Error starting blueprint generation: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/blueprint/visualize', methods=['GET'])
def visualize_blueprint():
    """Visualize the latest blueprint."""
    try:
        blueprint_id = request.args.get('id')
        if blueprint_id:
            # Future improvement: retrieve specific blueprint by ID
            pass

        # Get latest blueprint
        blueprint = get_latest_blueprint_from_sqlite()

        if not blueprint:
            return render_template('no_blueprint.html')

        # Prepare data for the template
        template_data = {
            'blueprint': json.dumps(blueprint),
            'rooms': blueprint.get('rooms', []),
            'walls': blueprint.get('walls', []),
            'objects': blueprint.get('objects', []),
            'generated_at': blueprint.get('generated_at', 'unknown')
        }

        return render_template('visualize.html', **template_data)
    except Exception as e:
        logger.error(f"Error visualizing blueprint: {e}")
        tb = traceback.format_exc()
        return render_template('error.html', error=str(e), traceback=tb)

@app.route('/api/blueprint/status', methods=['GET'])
def get_blueprint_status():
    """Get current blueprint generation status for the UI."""
    try:
        from .blueprint_generator import BlueprintGenerator
        generator = BlueprintGenerator()
        status = generator.get_status()

        # Add helpful messages based on status
        if status.get('state') == 'failed':
            reason = status.get('reason', 'unknown')
            user_message = {
                'no_distance_data': 'No Bluetooth distance data available. Check your BLE scanners.',
                'insufficient_entities': 'Not enough devices detected for positioning.',
                'position_calculation_failed': 'Position calculation failed. Check your device placement.',
                'no_rooms_created': 'Room layout generation failed. Try adding more reference points.',
                'database_save_failed': 'Failed to save blueprint to database.',
            }.get(reason, 'Blueprint generation failed. Check logs for details.')

            status['user_message'] = user_message

        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting blueprint status: {e}")
        return jsonify({
            'state': 'error',
            'message': str(e)
        }), 500

@app.route('/api/data/log', methods=['POST'])
def log_sensor_data():
    """Manual endpoint to trigger data logging."""
    try:
        # Log sensor data using the BluetoothProcessor
        result = bluetooth_processor.log_sensor_data()
        return jsonify({
            'status': 'success',
            'data': result
        })
    except Exception as e:
        logger.error(f"Error logging sensor data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/debug', methods=['GET'])
def show_debug():
    """Show debug information."""
    try:
        # Collect debug information
        debug_data = {
            'timestamp': time.time(),
            'config': load_config(),
            'platform': {
                'python_version': sys.version,
                'platform': sys.platform
            }
        }

        # Render debug template
        return render_template('debug.html', data=debug_data)
    except Exception as e:
        logger.error(f"Error showing debug info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/images/<path:filename>', methods=['GET'])
def serve_images(filename):
    """Serve AI-generated image files."""
    try:
        # Get the current configuration
        config = load_config()
        output_dir = config.get('ai_image_generation', {}).get('output_dir', 'data/generated_images')

        # Ensure path is absolute
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), output_dir)

        # Check if file exists
        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            logger.warning(f"Requested image file not found: {file_path}")
            return jsonify({'error': 'Image not found'}), 404

        # Determine content type based on file extension
        content_type = None
        if filename.lower().endswith('.png'):
            content_type = 'image/png'
        elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            content_type = 'image/jpeg'
        elif filename.lower().endswith('.webp'):
            content_type = 'image/webp'
        else:
            content_type = 'application/octet-stream'

        logger.debug(f"Serving image file: {filename} with content-type: {content_type}")
        return send_file(file_path, mimetype=content_type)
    except Exception as e:
        logger.error(f"Error serving image file {filename}: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500
    except Exception as e:
        logger.error(f"Error serving image: {e}")
        return jsonify({"error": str(e)}), 500

def start_api(host='0.0.0.0', port=8001, debug=False, use_reloader=False):
    """Start the Flask API server."""
    logger.info(f"Starting API server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)