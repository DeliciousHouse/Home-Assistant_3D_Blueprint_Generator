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

    # 2. Try to load the config.yaml file from one directory level up
    root_config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', '..', 'config.yaml'))
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
    app_config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '..', 'config', 'config.json'))
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

            # Map HA options to internal config structure
            if 'db' not in config:
                config['db'] = {}

            # Map database options
            config['db']['host'] = ha_options.get('db_host', config['db'].get('host'))
            config['db']['port'] = ha_options.get('db_port', config['db'].get('port'))
            config['db']['database'] = ha_options.get('db_database', config['db'].get('database'))
            config['db']['user'] = ha_options.get('db_user', config['db'].get('user'))
            config['db']['password'] = ha_options.get('db_password', config['db'].get('password', ''))

            # Add fixed_sensors directly (keep this structure)
            config['fixed_sensors'] = ha_options.get('fixed_sensors', {})

            # Add processing parameters
            if 'processing_params' not in config:
                config['processing_params'] = {}

            config['processing_params']['rssi_threshold'] = ha_options.get('min_rssi',
                config['processing_params'].get('rssi_threshold', -85))
            config['processing_params']['update_interval'] = ha_options.get('update_interval',
                config['processing_params'].get('update_interval', 300))

        except Exception as e:
            logger.error(f"Failed to load HA options: {str(e)}")

    # Debug output
    logger.debug(f"Final config: {json.dumps(config, default=str)}")
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
