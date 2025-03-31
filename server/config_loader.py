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
    if default_config_path.exists():
        try:
            with open(default_config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded default config from {default_config_path}")
        except Exception as e:
            logger.error(f"Failed to load default config: {str(e)}")

    # 2. Load user-provided options from /data/options.json
    if ha_options_path.exists():
        try:
            with open(ha_options_path, 'r') as f:
                ha_options = json.load(f)
            logger.info(f"Loaded HA options from {ha_options_path}")

            # --- Merge and Map HA Options ---

            # Log level
            config['log_level'] = ha_options.get('log_level', config.get('log_level', 'info'))

            # Processing parameters
            if 'processing_params' not in config: config['processing_params'] = {}
            config['processing_params']['update_interval'] = ha_options.get('processing_interval', config.get('processing_params', {}).get('update_interval', 300))
            # Add other processing params here if needed in options.json

            # AI settings
            if 'ai_settings' not in config: config['ai_settings'] = {}
            config['ai_settings']['enable_refinement'] = ha_options.get('enable_ai_refinement', config.get('ai_settings', {}).get('enable_refinement', True))
            # Add other AI params here

            # Room detection settings
            config['room_detection'] = {
                'use_areas': ha_options.get('use_room_areas', config.get('room_detection', {}).get('use_areas', True))
            }

        except Exception as e:
            logger.error(f"Failed to process HA options: {str(e)}")
    else:
        logger.warning(f"{ha_options_path} not found. Using default configuration only.")

    # Debug output
    safe_config = {k: v for k, v in config.items()} # Shallow copy
    logger.debug(f"Final config loaded: {json.dumps(safe_config, default=str)}") # Log final structure

    return config

# Removed _merge_configs as direct update/get is used now
