#!/usr/bin/env python3

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import requests
from urllib.parse import urljoin
import random  # For generating mock data
import uuid  # For generating unique identifiers

# Load configuration
try:
    from .config_loader import load_config
except ImportError:
    from config_loader import load_config

logger = logging.getLogger(__name__)

class HAClient:
    """Home Assistant API client for the Blueprint Generator."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Home Assistant client."""
        self.config = load_config(config_path)
        self.ha_config = self.config.get('home_assistant', {})

        # The Supervisor provides these environment variables to add-ons
        # Try different environment variables that might contain the supervisor URL
        supervisor_url_vars = ['SUPERVISOR_URL', 'HASSIO_URL', 'HOME_ASSISTANT_URL']
        self.ha_url = None

        for var in supervisor_url_vars:
            if os.environ.get(var):
                self.ha_url = os.environ.get(var)
                logger.debug(f"Found Supervisor URL in {var}: {self.ha_url}")
                break

        # If no environment variable found, use the config or default
        if not self.ha_url:
            # Default URLs to try
            possible_urls = [
                self.ha_config.get('url'),  # From config file
                'http://supervisor/core/api',  # Standard addon path
                'http://supervisor/api',  # Alternative path
                'http://supervisor/core',   # Another common path
                'http://supervisor'         # Base path
            ]

            for url in possible_urls:
                if url:
                    self.ha_url = url
                    logger.debug(f"Using URL from config: {self.ha_url}")
                    break

        # Try to get the token from various environment variables
        token_vars = ['SUPERVISOR_TOKEN', 'HASSIO_TOKEN', 'HOME_ASSISTANT_TOKEN']
        self.ha_token = None

        for var in token_vars:
            if os.environ.get(var):
                self.ha_token = os.environ.get(var)
                logger.debug(f"Found auth token in {var}")
                break

        # If no token found in environment, try config
        if not self.ha_token:
            self.ha_token = self.ha_config.get('token', '')

        # Log connection details (but not the token itself)
        logger.info(f"Initializing Home Assistant client with URL: {self.ha_url}")
        logger.info(f"Authentication token available: {bool(self.ha_token)}")
        if not self.ha_token:
            logger.warning("No authentication token found! API calls will fail.")

        # Debug: Log all environment variables (useful for troubleshooting)
        if logger.isEnabledFor(logging.DEBUG):
            env_vars = {k: '***REDACTED***' if 'token' in k.lower() else v for k, v in os.environ.items()}
            logger.debug(f"Environment variables available: {env_vars}")

        # Setting offline mode to False initially and letting _test_connection determine if needed
        self.offline_mode = False

        # Setup request headers
        self.headers = {
            'Authorization': f'Bearer {self.ha_token}',
            'Content-Type': 'application/json',
        }

        # Validate connection on startup
        if not self._test_connection():
            logger.warning("Unable to connect to Home Assistant API. Entering offline mode with mock data.")
            self.offline_mode = True
        else:
            logger.info("Successfully connected to Home Assistant API")

    def _test_connection(self):
        """Test connection to Home Assistant and log detailed debug info."""
        try:
            # First try the classic API endpoint
            base_url = self.ha_url
            api_paths = [
                '/api/',  # Standard API path
                '/',      # Try directly
                '/api',   # Without trailing slash
            ]

            for api_path in api_paths:
                url = urljoin(base_url, api_path)
                logger.debug(f"Testing Home Assistant connection to {url}")
                logger.debug(f"Using headers: {{'Authorization': 'Bearer ***REDACTED***', 'Content-Type': '{self.headers.get('Content-Type')}'}}")

                try:
                    response = requests.get(url, headers=self.headers, timeout=10)

                    # Check response status
                    if response.status_code == 200:
                        logger.info(f"Successfully connected to Home Assistant API at {url}")
                        self.ha_url = base_url  # Save the working base URL
                        return True
                    else:
                        logger.warning(f"Connection attempt to {url} failed: HTTP {response.status_code}")
                        logger.debug(f"Response: {response.text[:200]}")  # Log first 200 chars
                except Exception as e:
                    logger.warning(f"Connection attempt to {url} failed: {str(e)}")
                    continue

            # Try alternative base URLs if all attempts with current URL failed
            alternative_bases = [
                'http://supervisor/core',
                'http://supervisor',
                'http://hassio/core',
                'http://hassio',
                'http://localhost:8123'
            ]

            for alt_base in alternative_bases:
                if alt_base == base_url:
                    continue  # Skip if we already tried this base URL

                logger.debug(f"Trying alternative base URL: {alt_base}")
                url = urljoin(alt_base, '/api/')

                try:
                    response = requests.get(url, headers=self.headers, timeout=5)  # Shorter timeout for alternatives
                    if response.status_code == 200:
                        logger.info(f"Successfully connected using alternative URL: {alt_base}")
                        self.ha_url = alt_base  # Update to working URL
                        return True
                except Exception:
                    continue  # Try next URL on exception

            # If we get here, all connection attempts failed
            logger.error("All connection attempts to Home Assistant API failed.")
            return False

        except Exception as e:
            logger.error(f"Error connecting to Home Assistant: {str(e)}")
            return False

    # The rest of the class remains unchanged
    # ...

# For compatibility with existing code
HomeAssistantClient = HAClient