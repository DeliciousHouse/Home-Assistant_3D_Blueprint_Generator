#!/usr/bin/env python3

import logging
import os
import sys
import threading
import time
from pathlib import Path
import json

# Set up basic logging instead of using logging.config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("blueprint_generator")
logger.info("Starting External Blueprint Generator")

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the working directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load or create external config
def load_or_create_external_config():
    """Load external configuration or create if it doesn't exist."""
    config_dir = Path('/opt/blueprint_generator/config')
    config_path = config_dir / 'config.json'

    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
            logger.info(f"Loaded existing config from {config_path}")
    else:
        logger.info("Creating new external configuration file")
        config = {
            "processing_params": {
                "update_interval": 300,
                "rssi_threshold": -85,
                "minimum_sensors": 2,
                "accuracy_threshold": 10.0,
                "use_ml_distance": True,
                "distance_calculation": {
                    "reference_power": -66,
                    "path_loss_exponent": 2.8
                },
                "max_distance": 1000
            },
            "blueprint_validation": {
                "min_room_area": 4,
                "max_room_area": 100,
                "min_room_dimension": 1.5,
                "max_room_dimension": 15,
                "min_wall_thickness": 0.1,
                "max_wall_thickness": 0.5,
                "min_ceiling_height": 2.2,
                "max_ceiling_height": 4.0
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8001,
                "debug": False,
                "cors_origins": ["*"]
            },
            "generation_settings": {
                "distance_window_minutes": 15,
                "area_window_minutes": 10,
                "mds_dimensions": 2,
                "use_adjacency": True,
                "min_points_per_room": 1
            },
            "ai_settings": {
                "enable_refinement": True
            },
            "room_detection": {
                "use_areas": True
            },
            "home_assistant": {
                "api_url": os.environ.get("HA_URL", "http://localhost:8123"),
                "token": os.environ.get("HA_TOKEN", "")
            },
            "log_level": os.environ.get("LOG_LEVEL", "info")
        }

        # Ensure config directory exists
        config_dir.mkdir(parents=True, exist_ok=True)

        # Save new config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            logger.info(f"Created new config at {config_path}")

    # Always override Home Assistant connection settings from environment
    if "home_assistant" not in config:
        config["home_assistant"] = {}

    # Always use environment variables for Home Assistant URL and token if provided
    ha_url = os.environ.get("HA_URL")
    if ha_url:
        config["home_assistant"]["api_url"] = ha_url

    ha_token = os.environ.get("HA_TOKEN")
    if ha_token:
        config["home_assistant"]["token"] = ha_token

    # Override log level from environment if provided
    log_level = os.environ.get("LOG_LEVEL")
    if log_level:
        config["log_level"] = log_level

    # Update API port from environment if provided
    port_env = os.environ.get("PORT")
    if port_env:
        try:
            port = int(port_env)
            config["api"]["port"] = port
        except ValueError:
            logger.warning(f"Invalid PORT environment variable: {port_env}")

    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return config

# Load or create external config
config = load_or_create_external_config()

# Set log level based on config
log_level = config.get('log_level', 'info').upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

logger.info("External configuration loaded and processed successfully")

# Import required modules after config is loaded
from server.api import app, start_api
from server.db import init_sqlite_db
from server.bluetooth_processor import BluetoothProcessor
from server.blueprint_generator import BlueprintGenerator, ensure_reference_positions

def initialize_databases():
    """Initialize database schema."""
    try:
        # Initialize SQLite database
        logger.info("Initializing SQLite database...")
        if not init_sqlite_db():
            logger.error("Failed to initialize SQLite database")
            return False
        logger.info("SQLite database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

def start_processing_scheduler(config):
    """Start a background thread that periodically processes Bluetooth data."""
    # Initialize components
    bluetooth_processor = BluetoothProcessor()
    blueprint_generator = BlueprintGenerator()

    def process_loop():
        counter = 0
        # First log data for a while
        logger.info("Starting initial data collection phase...")

        # Ensure we have reference positions before collecting data
        reference_positions = ensure_reference_positions()
        logger.info(f"Reference positions confirmed: {len(reference_positions)} positions available")

        while counter < 5:  # Change this to adjust how many test data points to collect
            try:
                # Log Bluetooth data
                result = bluetooth_processor.log_sensor_data()
                logger.info(f"Data collection cycle {counter}: Logged {result.get('distances_logged', 0)} distances and {result.get('areas_logged', 0)} area observations")
                counter += 1
            except Exception as e:
                logger.error(f"Error collecting data: {str(e)}", exc_info=True)

            # Short sleep for data collection
            time.sleep(5)  # 5 seconds between data collection cycles

        # After collecting data, generate the blueprint
        logger.info("Initial data collection complete, generating blueprint...")
        try:
            # Double check reference positions before blueprint generation
            if len(ensure_reference_positions()) < 3:
                logger.warning("Still insufficient reference positions, adding more...")
                reference_positions = ensure_reference_positions()

            success = blueprint_generator.generate_blueprint()
            if success:
                logger.info("Blueprint generation SUCCESSFUL!")
            else:
                logger.error("Blueprint generation FAILED.")
        except Exception as e:
            logger.error(f"Error generating blueprint: {str(e)}", exc_info=True)

        # Continue with normal processing cycle after initial generation
        while True:
            try:
                # Log new data
                bluetooth_processor.log_sensor_data()

                # Generate updated blueprint every hour
                if counter % 360 == 0:  # Assuming 10s between cycles, this is ~1 hour
                    logger.info("Generating updated blueprint...")
                    # Ensure reference positions are still available
                    ensure_reference_positions()
                    blueprint_generator.generate_blueprint()

                counter += 1
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}", exc_info=True)

            # Normal processing interval
            time.sleep(10)  # 10 seconds between cycles

    # Start the processing thread
    processing_thread = threading.Thread(target=process_loop, daemon=True)
    processing_thread.start()
    logger.info("Background processing scheduler started")

def configure_multilateration_and_room_naming():
    """Configure multilateration for floor detection and fix room naming."""
    try:
        from server.ha_client import HomeAssistantClient
        from server.db import save_reference_position

        logger.info("Configuring system to use multilateration for room positioning and floor detection")

        # Initialize Home Assistant client
        ha_client = HomeAssistantClient()

        # Define the areas we know should exist with proper naming
        correct_areas = {
            "master_bedroom": {"name": "Master Bedroom", "floor": 1},
            "master_bathroom": {"name": "Master Bathroom", "floor": 1},
            "sky_floor": {"name": "Sky Floor", "floor": 2},  # Make sure Sky Floor is on second floor
            "lounge": {"name": "Lounge", "floor": 1},
            "kitchen": {"name": "Kitchen", "floor": 1},
            "office": {"name": "Office", "floor": 2},
        }

        # Update area registry via Home Assistant to ensure correct naming
        areas = ha_client.get_areas()
        area_map = {area["area_id"]: area for area in areas}

        # Log all available areas for debugging
        logger.info(f"Available Home Assistant areas: {[area['name'] for area in areas]}")

        # Create reference positions for each area to help with multilateration
        # These will serve as anchors for the multilateration algorithm
        reference_points = {}

        # Create a reference point for each area we want to ensure exists
        floor_z_offset = 3.0  # 3 meters per floor
        for area_id, area_info in correct_areas.items():
            # Calculate z-coordinate based on floor
            z_coord = (area_info['floor'] - 1) * floor_z_offset

            # Create a unique reference point ID for this area
            ref_id = f"reference_{area_id}"

            # Ensure each reference point has distinct coordinates
            # Use area-specific offsets to help the multilateration algorithm
            x_offset = hash(area_id) % 5  # Simple hash-based offset between 0-4
            y_offset = (hash(area_id) // 5) % 5  # Different offset for y

            # Create reference position
            reference_points[ref_id] = {
                "x": 10 + x_offset,
                "y": 10 + y_offset,
                "z": z_coord,
                "area_id": area_id
            }

            # Save the reference position to the database
            save_reference_position(
                device_id=ref_id,
                x=reference_points[ref_id]['x'],
                y=reference_points[ref_id]['y'],
                z=reference_points[ref_id]['z'],
                area_id=area_id
            )

            logger.info(f"Created reference position for {area_id} at floor {area_info['floor']}")

        # Update configuration to enable multilateration
        config_updates = {
            "generation_settings": {
                "use_multilateration": True,
                "floor_height": floor_z_offset,
                "min_points_per_room": 1
            },
            "room_detection": {
                "use_areas": True,
                "rename_map": {
                    "alexs_room": "master_bedroom",
                    "alexs_bathroom": "master_bathroom"
                }
            }
        }

        # Save the updated configuration
        config_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(config_dir, '..', 'config', 'config.json')

        # Load existing config
        try:
            with open(config_path, 'r') as f:
                existing_config = json.load(f)

            # Update with our multilateration settings
            for section, settings in config_updates.items():
                if section not in existing_config:
                    existing_config[section] = {}
                existing_config[section].update(settings)

            # Save back to disk
            with open(config_path, 'w') as f:
                json.dump(existing_config, f, indent=2)

            logger.info("Updated configuration to use multilateration for floor detection")
        except Exception as e:
            logger.warning(f"Could not update config file, but will continue with reference points: {e}")

        return True
    except Exception as e:
        logger.error(f"Error configuring multilateration: {e}", exc_info=True)
        return False

def main():
    """Main entry point."""
    try:
        # Initialize databases
        if not initialize_databases():
            logger.error("Database initialization failed. Exiting application.")
            return

        # Start background data collection and processing
        start_processing_scheduler(config)
        logger.info("Background processing started")

        # Start API server
        host = config.get('api', {}).get('host', '0.0.0.0')
        port = config.get('api', {}).get('port', 8001)
        debug = config.get('api', {}).get('debug', False)

        logger.info(f"Starting API server on {host}:{port}")

        # Start the API
        try:
            # Before starting the API, ensure room naming is correct and floor detection is enabled
            configure_multilateration_and_room_naming()

            start_api(host=host, port=port, debug=debug, use_reloader=False)
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"Port {port} already in use. Please choose a different port.")
                # Try again with port+1
                logger.info(f"Attempting to start with port {port+1}")
                start_api(host=host, port=port+1, debug=debug, use_reloader=False)
            else:
                raise

    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}", exc_info=True)

if __name__ == '__main__':
    main()