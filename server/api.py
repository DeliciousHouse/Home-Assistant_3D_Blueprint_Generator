import json
import logging
import requests
from typing import Dict, Optional
from flask import Flask, jsonify, request, send_from_directory, render_template, redirect
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

def start_api(host='0.0.0.0', port=8001, debug=False, use_reloader=False):
    """Start the Flask API server."""
    logger.info(f"Starting API server on {host}:{port} (debug={debug})")
    app.run(host=host, port=port, debug=debug, use_reloader=use_reloader)