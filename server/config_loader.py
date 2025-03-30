#!/usr/bin/env python3

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional
import yaml  # Add yaml import for reading .yaml files

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from the specified path or default locations.

    Args:
        config_path: Optional path to a specific config file

    Returns:
        Dict containing the merged configuration
    """
    config = {}

    # 1. First try the specified config path if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")

    # 2. Try to load the config.yaml file from one directory level up (not two)
    # The config.yaml is now in the blueprint_generator directory
    root_config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'config.yaml'))
    if root_config_path.exists():
        try:
            with open(root_config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # Update config with yaml settings
                    _merge_configs(config, yaml_config)
            logger.info(f"Loaded YAML config from {root_config_path}")
        except Exception as e:
            logger.error(f"Failed to load YAML config from {root_config_path}: {str(e)}")

    # 3. Try app-specific config
    app_config_path = Path(__file__).parent.parent / 'config' / 'config.json'
    if app_config_path.exists():
        try:
            with open(app_config_path, 'r') as f:
                app_config = json.load(f)
                # Update config with app-specific settings
                _merge_configs(config, app_config)
            logger.info(f"Loaded app config from {app_config_path}")
        except Exception as e:
            logger.error(f"Failed to load app config: {str(e)}")

    # 4. Try Home Assistant options (these override app config)
    ha_options_path = Path('/data/options.json')
    if ha_options_path.exists():
        try:
            with open(ha_options_path, 'r') as f:
                ha_options = json.load(f)
            logger.info(f"Loaded HA options from {ha_options_path}")

            # Initialize config sections if they don't exist
            if 'processing_params' not in config:
                config['processing_params'] = {}

            if 'blueprint_validation' not in config:
                config['blueprint_validation'] = {}

            if 'ai_settings' not in config:
                config['ai_settings'] = {}

            # Map the new options structure

            # Processing parameters
            config['processing_params']['update_interval'] = ha_options.get('processing_interval',
                config['processing_params'].get('update_interval', 300))

            # AI settings
            config['ai_settings']['enable_refinement'] = ha_options.get('enable_ai_refinement',
                config['ai_settings'].get('enable_refinement', True))

            # Room detection settings
            config['room_detection'] = {
                'use_areas': ha_options.get('use_room_areas', True)
            }

            # Fixed sensors (reference positions)
            if 'fixed_sensors' in ha_options:
                # It's now definitely a string, so parse it as JSON
                try:
                    if isinstance(ha_options['fixed_sensors'], str) and ha_options['fixed_sensors'].strip():
                        config['fixed_sensors'] = json.loads(ha_options['fixed_sensors'])
                    else:
                        config['fixed_sensors'] = {}
                except json.JSONDecodeError:
                    logger.error("Failed to parse fixed_sensors as JSON, using empty dict")
                    config['fixed_sensors'] = {}
            else:
                config['fixed_sensors'] = {}

            # Log level
            config['log_level'] = ha_options.get('log_level', 'info')

        except Exception as e:
            logger.error(f"Failed to load HA options: {str(e)}")

    # Debug output - REMOVE database masking
    # Just log the entire config structure without any special handling for DB
    logger.debug(f"Final config structure: {json.dumps(config, default=str)}")

    return config

def _merge_configs(target, source):
    """Recursively merge source config into target config."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            # Recursively merge nested dictionaries
            _merge_configs(target[key], value)
        else:
            # Override or add key-value pairs
            target[key] = value
