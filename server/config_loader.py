#!/usr/bin/env python3

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional
# Removed yaml import as we only load JSON now

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from defaults and HA options."""
    config = {}

    # Define paths relative to this file's location
    config_dir = Path(__file__).parent.parent / 'config'
    default_config_path = config_dir / 'config.json'
    ha_options_path = Path('/data/options.json')

    # 1. Load defaults from config/config.json
    if (default_config_path.exists()):
        try:
            with open(default_config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded default config from {default_config_path}")
        except Exception as e:
            logger.error(f"Failed to load default config: {str(e)}")
            config = {}  # Start with empty config if default fails

    # 2. Load user-provided options from /data/options.json
    if (ha_options_path.exists()):
        try:
            with open(ha_options_path, 'r') as f:
                ha_options = json.load(f)
            logger.info(f"Loaded HA options from {ha_options_path}")

            # --- Merge and Map HA Options ---

            # Log level
            config['log_level'] = ha_options.get('log_level', config.get('log_level', 'info'))

            # Processing parameters
            if 'processing_params' not in config: config['processing_params'] = {}
            config['processing_params']['update_interval'] = ha_options.get(
                'processing_interval',
                config.get('processing_params', {}).get('update_interval', 300)
            )

            # AI settings
            if 'ai_settings' not in config: config['ai_settings'] = {}
            config['ai_settings']['enable_refinement'] = ha_options.get(
                'enable_ai_refinement',
                config.get('ai_settings', {}).get('enable_refinement', False)  # Default refinement to False unless enabled
            )

            # Room detection settings
            if 'room_detection' not in config: config['room_detection'] = {}
            config['room_detection']['use_areas'] = ha_options.get(
                'use_room_areas',
                config.get('room_detection', {}).get('use_areas', True)  # Default to using areas
            )

            # Generation settings specific to MDS/Anchoring
            if 'generation_settings' not in config: config['generation_settings'] = {}
            config['generation_settings']['distance_window_minutes'] = ha_options.get(
                'distance_window_minutes',  # Add this option to schema/options if user-configurable
                config.get('generation_settings', {}).get('distance_window_minutes', 15)
            )

        except Exception as e:
            logger.error(f"Failed to process HA options: {str(e)}")
    else:
        # This is expected if user hasn't configured options
        logger.info(f"{ha_options_path} not found. Using default configuration only.")

    # Debug output
    safe_config = {k: v for k, v in config.items()}  # Shallow copy
    logger.debug(f"Final config loaded: {json.dumps(safe_config, default=str)}")  # Log final structure

    return config

# Helper function for API configuration
def get_api_config():
    config = load_config()
    return {
        'host': config.get('api', {}).get('host', '0.0.0.0'),
        'port': config.get('api', {}).get('port', 8001),  # Default to 8001
        'debug': config.get('api', {}).get('debug', False)
    }
