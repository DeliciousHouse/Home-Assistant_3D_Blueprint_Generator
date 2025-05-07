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

            # --- Static Device Detection Settings ---
            # Ensure section exists in config with proper initialization
            if 'static_device_detection' not in config:
                config['static_device_detection'] = {}

            # Handle primary option name with fallback to alternative name for backward compatibility
            # For enable_dynamic_anchors (primary) or static_device_detection_enabled (alternative)
            config['static_device_detection']['enable_dynamic_anchors'] = ha_options.get(
                'enable_dynamic_anchors',  # Primary option name
                ha_options.get(  # Fallback to alternative option name
                    'static_device_detection_enabled',
                    config.get('static_device_detection', {}).get('enable_dynamic_anchors', True)
                )
            )

            # Movement threshold - how much movement is allowed before a device is no longer considered static
            config['static_device_detection']['movement_threshold_meters'] = ha_options.get(
                'movement_threshold_meters',
                config.get('static_device_detection', {}).get('movement_threshold_meters', 0.5)
            )

            # Time window for analyzing device movements
            config['static_device_detection']['time_window_seconds'] = ha_options.get(
                'time_window_seconds',
                config.get('static_device_detection', {}).get('time_window_seconds', 300)
            )

            # Minimum observations required to classify a device as static
            config['static_device_detection']['min_observations_for_static'] = ha_options.get(
                'min_observations_for_static',  # Primary option name
                ha_options.get(  # Fallback to alternative option name
                    'static_device_min_observations',
                    config.get('static_device_detection', {}).get('min_observations_for_static', 5)
                )
            )

            # How quickly confidence in a static anchor decays over time
            config['static_device_detection']['static_anchor_confidence_decay_hours'] = ha_options.get(
                'static_anchor_confidence_decay_hours',
                config.get('static_device_detection', {}).get('static_anchor_confidence_decay_hours', 1.0)
            )

            # Maximum number of dynamic anchors to use
            config['static_device_detection']['max_dynamic_anchors'] = ha_options.get(
                'max_dynamic_anchors',
                config.get('static_device_detection', {}).get('max_dynamic_anchors', 10)
            )

            # Generation settings specific to MDS/Anchoring
            if 'generation_settings' not in config: config['generation_settings'] = {}
            config['generation_settings']['distance_window_minutes'] = ha_options.get(
                'distance_window_minutes',  # Add this option to schema/options if user-configurable
                config.get('generation_settings', {}).get('distance_window_minutes', 15)
            )

            # Home Assistant connection settings
            if 'home_assistant' not in config: config['home_assistant'] = {}
            # Get the token from options
            if 'ha_token' in ha_options and ha_options['ha_token']:
                config['home_assistant']['token'] = ha_options['ha_token']
                logger.info("Found Home Assistant token in options")
            # Default URL for supervisor add-on
            config['home_assistant']['url'] = ha_options.get(
                'ha_url',
                config.get('home_assistant', {}).get('url', 'http://supervisor/core')
            )

        except Exception as e:
            logger.error(f"Failed to process HA options: {str(e)}")
    else:
        # This is expected if user hasn't configured options
        logger.info(f"{ha_options_path} not found. Using default configuration only.")

    # Debug output with sensitive information redacted
    safe_config = {k: v for k, v in config.items()}  # Shallow copy
    if 'home_assistant' in safe_config and 'token' in safe_config['home_assistant']:
        safe_config['home_assistant']['token'] = '***REDACTED***'
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
