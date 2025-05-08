#!/usr/bin/env python3

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Any

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
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse default config - invalid JSON: {str(e)}")
            config = {}  # Start with empty config if default fails
        except Exception as e:
            logger.error(f"Failed to load default config: {str(e)}")
            config = {}  # Start with empty config if default fails

    # 2. Check for custom config path (mainly used for testing)
    if config_path:
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            config.update(custom_config)  # Merge custom config over defaults
            logger.info(f"Loaded custom config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load custom config from {config_path}: {str(e)}")

    # 3. Load user-provided options from /data/options.json
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
            update_static_device_settings(config, ha_options)

            # Generation settings specific to MDS/Anchoring
            if 'generation_settings' not in config: config['generation_settings'] = {}
            config['generation_settings']['distance_window_minutes'] = ha_options.get(
                'distance_window_minutes',  # Add this option to schema/options if user-configurable
                config.get('generation_settings', {}).get('distance_window_minutes', 15)
            )

            # Min points per room setting
            config['generation_settings']['min_points_per_room'] = ha_options.get(
                'min_points_per_room',
                config.get('generation_settings', {}).get('min_points_per_room', 3)
            )

            # Home Assistant connection settings
            if 'home_assistant' not in config: config['home_assistant'] = {}
            # Set the URL (only from options or default)
            config['home_assistant']['url'] = ha_options.get(
                'ha_url',
                config.get('home_assistant', {}).get('url', 'http://supervisor/core')
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse HA options - invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to process HA options: {str(e)}")
    else:
        # This is expected if user hasn't configured options
        logger.info(f"{ha_options_path} not found. Using default configuration only.")

    # 4. Set Home Assistant token from SUPERVISOR_TOKEN environment variable
    # Home Assistant Add-ons should ALWAYS use SUPERVISOR_TOKEN for API access
    if 'home_assistant' not in config:
        config['home_assistant'] = {}

    supervisor_token = os.environ.get('SUPERVISOR_TOKEN')
    if supervisor_token:
        config['home_assistant']['token'] = supervisor_token
        logger.info("Using SUPERVISOR_TOKEN from environment for Home Assistant API authentication")
    else:
        logger.warning("SUPERVOR_TOKEN environment variable not found. Home Assistant API calls may fail.")
        # Do not use any token from config.json or options.json - only use SUPERVISOR_TOKEN

    # Validate and fill required sections with defaults
    validate_and_ensure_defaults(config)

    # Debug output with sensitive information redacted
    safe_config = {k: v for k, v in config.items()}  # Shallow copy
    if 'home_assistant' in safe_config and 'token' in safe_config['home_assistant']:
        safe_config['home_assistant']['token'] = '***REDACTED***'
    logger.debug(f"Final config loaded: {json.dumps(safe_config, default=str)}")  # Log final structure

    return config

def update_static_device_settings(config: Dict[str, Any], ha_options: Dict[str, Any]) -> None:
    """
    Update static device detection settings from HA options.

    This function ensures that static device detection settings from config.json
    are properly merged with any overrides from Home Assistant options.json.
    """
    # Ensure section exists in config with proper initialization
    if 'static_device_detection' not in config:
        config['static_device_detection'] = {}

    # Settings mapping from Home Assistant option keys to internal config keys
    # Format: (config_key, ha_option_key, default_value, value_type)
    settings_map = [
        ('enable_dynamic_anchors', 'static_device_detection_enabled', True, bool),
        ('movement_threshold_meters', 'movement_threshold_meters', 0.5, float),
        ('time_window_seconds', 'time_window_seconds', 300, int),
        ('min_observations_for_static', 'static_device_min_observations', 5, int),
        ('static_anchor_confidence_decay_hours', 'static_anchor_confidence_decay_hours', 1.0, float),
        ('max_dynamic_anchors', 'max_dynamic_anchors', 10, int)
    ]

    for config_key, ha_option_key, default_value, value_type in settings_map:
        # First check for the exact key in HA options
        value = ha_options.get(ha_option_key)

        # If not found, use existing config value or default
        if value is None:
            value = config['static_device_detection'].get(config_key, default_value)

        try:
            # Type conversion - important for numeric values from JSON
            if value_type is bool and isinstance(value, str):
                value = value.lower() in ('true', 'yes', '1', 'on')
            elif value is not None:
                value = value_type(value)
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid value for {config_key}: '{value}'. Using default {default_value}. Error: {str(e)}")
            value = default_value

        # Store in config
        config['static_device_detection'][config_key] = value

    # Log the final static device configuration
    logger.debug(f"Static device detection config: {config['static_device_detection']}")

    # Validate movement threshold to ensure it's positive
    if config['static_device_detection']['movement_threshold_meters'] <= 0:
        logger.warning("Movement threshold must be positive, setting to default 0.5")
        config['static_device_detection']['movement_threshold_meters'] = 0.5

def validate_and_ensure_defaults(config: Dict[str, Any]) -> None:
    """Validate config and ensure all required sections exist with defaults."""
    required_sections = {
        'processing_params': {
            'update_interval': 300,
            'rssi_threshold': -85,
            'minimum_sensors': 2
        },
        'blueprint_validation': {
            'min_room_area': 4,
            'max_room_area': 20,
            'min_room_dimension': 1.5,
            'max_room_dimension': 15
        },
        'generation_settings': {
            'distance_window_minutes': 15,
            'area_window_minutes': 10,
            'mds_dimensions': 2,
            'min_points_per_room': 3
        },
        'static_device_detection': {
            'enable_dynamic_anchors': True,
            'movement_threshold_meters': 0.5,
            'time_window_seconds': 300,
            'min_observations_for_static': 5,
            'static_anchor_confidence_decay_hours': 1.0,
            'max_dynamic_anchors': 10
        },
        'home_assistant': {
            'url': 'http://supervisor/core'
        },
        'api': {
            'host': '0.0.0.0',
            'port': 8001,
            'debug': False
        }
    }

    # Ensure each required section exists with defaults for missing values
    for section, defaults in required_sections.items():
        if section not in config:
            config[section] = {}

        # Fill in missing values with defaults
        for key, default_value in defaults.items():
            if key not in config[section]:
                config[section][key] = default_value

# Helper function for API configuration
def get_api_config():
    config = load_config()
    return {
        'host': config.get('api', {}).get('host', '0.0.0.0'),
        'port': config.get('api', {}).get('port', 8001),  # Default to 8001
        'debug': config.get('api', {}).get('debug', False)
    }
