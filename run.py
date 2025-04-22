#!/usr/bin/env python3

import logging
import os
import sys
import threading
import time
from pathlib import Path
from flask import Flask
import random
from server.db import save_distance_log

from server.config_loader import load_config

# Set up basic logging instead of using logging.config
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("blueprint_generator")
logger.info("Starting Blueprint Generator")

# Add the parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set the working directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load config first
config = load_config()

# Set log level based on config
log_level = config.get('log_level', 'info').upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

logger.info("Configuration loaded and processed successfully")

from server.api import app, start_api
from server.db import init_sqlite_db, SQLITE_DB_PATH

def initialize_databases():
    """Initialize database schema."""
    try:
        # Check if we have access to the data directory
        data_dir = os.path.dirname(SQLITE_DB_PATH)

        # Try to create directories if they don't exist
        try:
            os.makedirs(data_dir, exist_ok=True)
            logger.info(f"Ensured data directory exists at: {data_dir}")
        except PermissionError:
            logger.error(f"Permission error creating data directory: {data_dir}")
            logger.info("Will attempt to use the database path anyway")
        except Exception as dir_e:
            logger.warning(f"Issue with data directory {data_dir}: {dir_e}")

        # Check if we can write to the data directory
        if os.path.exists(data_dir):
            if os.access(data_dir, os.W_OK):
                logger.info(f"Data directory {data_dir} is writable")
            else:
                logger.warning(f"Data directory {data_dir} exists but is not writable!")

        # Initialize SQLite database
        logger.info(f"Initializing SQLite database at {SQLITE_DB_PATH}...")
        if not init_sqlite_db():
            logger.error("Failed to initialize SQLite database")
            return False
        logger.info("SQLite database initialized successfully")

        # Test direct DB write after schema initialization
        try:
            logger.info("Attempting direct DB write test...")
            from server.db import get_sqlite_connection
            from datetime import datetime
            conn_test = get_sqlite_connection()
            if not conn_test:
                logger.error("Failed to connect to database for test write")
                return False

            cursor_test = conn_test.cursor()
            cursor_test.execute("INSERT INTO distance_log (timestamp, tracked_device_id, scanner_id, distance) VALUES (?, ?, ?, ?)",
                               (datetime.now().isoformat(), 'test_device', 'test_scanner', 1.23))
            conn_test.commit()
            logger.info("Direct DB write test SUCCESSFUL.")
            # Optionally query it back to be sure
            cursor_test.execute("SELECT COUNT(*) FROM distance_log WHERE tracked_device_id='test_device'")
            count = cursor_test.fetchone()[0]
            logger.info(f"Direct DB read test: Found {count} test records.")
            conn_test.close()
        except Exception as test_e:
            logger.error(f"Direct DB write test FAILED: {test_e}", exc_info=True)
            # Continue anyway, as this might be a transient issue

        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
        return False

def start_processing_scheduler(config):
    """Start a background thread that periodically processes Bluetooth data."""
    from server.bluetooth_processor import BluetoothProcessor
    from server.blueprint_generator import BlueprintGenerator, ensure_reference_positions
    import random

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

def main():
    """Main entry point."""
    try:
        # We already loaded config at the module level
        global config

        # Initialize databases
        if not initialize_databases():
            logger.error("Database initialization failed. Exiting application.")
            return

        # Pass config to scheduler
        start_processing_scheduler(config)
        logger.info("Background processing started")

        # Start API server
        host = config.get('api', {}).get('host', '0.0.0.0')
        port = config.get('api', {}).get('port', 8001)
        debug = config.get('api', {}).get('debug', False)

        logger.info(f"Starting API server on {host}:{port}")
        # Ensure your start_api function or the components it uses can access the loaded config
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                start_api(host=host, port=port, debug=debug, use_reloader=False)
                break
            except OSError as e:
                if "Address already in use" in str(e) and attempt < max_attempts - 1:
                    logger.warning(f"Port {port} already in use, trying port {port+1}")
                    port += 1
                else:
                    raise

    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}", exc_info=True) # Add exc_info for traceback

if __name__ == '__main__':
    main()
