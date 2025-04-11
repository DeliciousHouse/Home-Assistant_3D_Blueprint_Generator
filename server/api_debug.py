# Add these endpoints to your Flask API to help with debugging
# This file extends your existing API with diagnostic endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Check health of various system components."""
    import time

    # Check database connection
    db_status = "healthy"
    try:
        import server.db as db
        # Try a simple query
        db.test_db_connection()
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = f"error: {str(e)}"

    # Check Home Assistant connection
    ha_status = "healthy"
    try:
        from server.ha_client import HomeAssistantClient
        client = HomeAssistantClient()
        connected = client.test_connection()
        if not connected:
            ha_status = "disconnected"
    except Exception as e:
        logger.error(f"Home Assistant health check failed: {e}")
        ha_status = f"error: {str(e)}"

    return jsonify({
        "status": "healthy",
        "database": db_status,
        "home_assistant": ha_status,
        "timestamp": time.time()
    })

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Return detailed debug information about the system."""
    from server.ha_client import HomeAssistantClient

    # Home Assistant connection info
    ha_client = HomeAssistantClient()
    ha_connected = ha_client.test_connection()

    ha_status = {
        "connected": ha_connected,
        "base_url": ha_client.base_url
    }

    if not ha_connected:
        ha_status["error"] = "Failed to connect to Home Assistant"

    # Entity scan
    entity_scan = {
        "total_entities": 0,
        "sample_entities": [],
        "specific_tests": {}
    }

    if ha_connected:
        try:
            # Get a sample of entities
            all_entities = ha_client.get_entities()
            entity_scan["total_entities"] = len(all_entities)

            # Get a few sample entity IDs
            sample_size = min(5, len(all_entities))
            entity_scan["sample_entities"] = [e.get("entity_id", "unknown") for e in all_entities[:sample_size]]

            # Check for specific entity types
            distance_sensors = ha_client.get_distance_sensors()
            entity_scan["specific_tests"]["distance_sensors"] = len(distance_sensors)

            # Area predictions if available
            area_predictions = ha_client.get_device_area_predictions()
            entity_scan["specific_tests"]["area_predictions"] = len(area_predictions)

        except Exception as e:
            logger.error(f"Entity scan failed: {e}")
            entity_scan["error"] = str(e)

    # Blueprint data
    blueprint_data = {
        "available": False
    }

    try:
        from server.blueprint_generator import BlueprintGenerator
        from server.db import get_latest_blueprint

        # Check if a blueprint exists
        blueprint = get_latest_blueprint()
        if blueprint:
            blueprint_data["available"] = True
            blueprint_data["timestamp"] = blueprint.get("timestamp", "unknown")
            blueprint_data["room_count"] = len(blueprint.get("rooms", []))
            blueprint_data["floor_count"] = len(blueprint.get("floors", []))
    except Exception as e:
        logger.error(f"Blueprint debug info failed: {e}")
        blueprint_data["error"] = str(e)

    # Return combined debug info
    return jsonify({
        "ha_status": ha_status,
        "entity_scan": entity_scan,
        "blueprint_data": blueprint_data
    })