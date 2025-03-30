#!/usr/bin/env python3

import logging
import os
import sys
import threading
import time
from pathlib import Path
from flask import Flask

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

# Set the working directory to the script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load config first
config = load_config()

logger.info("Configuration loaded and processed successfully")

from server.api import app, start_api
from server.db import init_sqlite_db

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
    from server.bluetooth_processor import BluetoothProcessor
    from server.blueprint_generator import BlueprintGenerator

    # Components can load config themselves using load_config(), or you can pass parts
    # Passing relevant parts can be cleaner:
    # bluetooth_processor = BluetoothProcessor(fixed_sensors=config.get('fixed_sensors', {}))
    # Or let them load the full config internally:
    bluetooth_processor = BluetoothProcessor() # Assumes it calls load_config() internally
    blueprint_generator = BlueprintGenerator() # Assumes it calls load_config() internally

    def process_loop():
        while True:
            try:
                logger.info("Running scheduled Bluetooth data processing")
                # The processor instance should now have access to the correct config
                result = bluetooth_processor.process_bluetooth_sensors()

                if result:
                    logger.info("Attempting to generate blueprint...")
                    # Simplified call to generate_blueprint() without arguments
                    blueprint = blueprint_generator.generate_blueprint()

                    if blueprint and blueprint.get('rooms'):
                        logger.info(f"Blueprint generated with {len(blueprint.get('rooms', []))} rooms.")
                    else:
                        logger.warning("Blueprint generation resulted in an empty or invalid blueprint.")

            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}", exc_info=True) # Add exc_info

            # Use update interval from config if available, else default
            update_interval_sec = config.get('processing_params', {}).get('update_interval', 300)
            logger.debug(f"Sleeping for {update_interval_sec} seconds.")
            time.sleep(update_interval_sec)

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
        port = config.get('api', {}).get('port', 5000)
        debug = config.get('api', {}).get('debug', False)

        logger.info(f"Starting API server on {host}:{port}")
        # Ensure your start_api function or the components it uses can access the loaded config
        start_api(config=config, host=host, port=port, debug=debug) # Pass config if needed by API directly

    except Exception as e:
        logger.error(f"Application startup failed: {str(e)}", exc_info=True) # Add exc_info for traceback

if __name__ == '__main__':
    main()
