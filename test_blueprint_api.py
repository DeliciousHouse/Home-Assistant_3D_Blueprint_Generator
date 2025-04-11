#!/usr/bin/env python3

import requests
import json
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("blueprint_tester")

def test_blueprint_api(base_url="http://localhost:8001"):
    """Test the blueprint API endpoint and print the response."""
    try:
        # Call the API
        logger.info(f"Testing Blueprint API at {base_url}/api/blueprint")
        response = requests.get(f"{base_url}/api/blueprint")

        # Check response status
        if response.status_code != 200:
            logger.error(f"API returned error status: {response.status_code}")
            logger.error(f"Response text: {response.text}")
            return False

        # Parse and validate response
        try:
            data = response.json()
            logger.info(f"API Response Structure:\n{json.dumps(data, indent=2)[:500]}...")

            # Check basic structure
            if 'success' not in data:
                logger.warning("API response is missing 'success' field")

            if 'blueprint' not in data:
                logger.error("API response is missing 'blueprint' field")
                return False

            blueprint = data['blueprint']

            # Check blueprint contents
            if not isinstance(blueprint, dict):
                logger.error(f"Blueprint is not a dictionary, got: {type(blueprint)}")
                return False

            # Check required fields
            required_fields = ['rooms', 'floors']
            missing_fields = [field for field in required_fields if field not in blueprint]

            if missing_fields:
                logger.error(f"Blueprint is missing required fields: {missing_fields}")
                logger.info(f"Blueprint keys: {blueprint.keys()}")
                return False

            # Check rooms
            if not isinstance(blueprint['rooms'], list):
                logger.error(f"Blueprint 'rooms' is not a list: {type(blueprint['rooms'])}")
                return False

            logger.info(f"Blueprint contains {len(blueprint['rooms'])} rooms and {len(blueprint['floors'])} floors")

            # Check first room
            if blueprint['rooms']:
                first_room = blueprint['rooms'][0]
                logger.info(f"First room: {first_room}")

                # Required room fields
                room_fields = ['id', 'name', 'bounds', 'floor']
                missing_room_fields = [field for field in room_fields if field not in first_room]

                if missing_room_fields:
                    logger.warning(f"Room is missing fields: {missing_room_fields}")

            return True

        except json.JSONDecodeError:
            logger.error("Failed to parse API response as JSON")
            logger.error(f"Response text: {response.text[:500]}")
            return False

    except requests.RequestException as e:
        logger.error(f"Request failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    # Use the first command line argument as the base URL, if provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
    success = test_blueprint_api(base_url)

    if success:
        logger.info("Blueprint API test successful!")
        sys.exit(0)
    else:
        logger.error("Blueprint API test failed")
        sys.exit(1)