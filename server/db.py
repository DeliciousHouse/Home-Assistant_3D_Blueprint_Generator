import json
import logging
import sqlite3
import os
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from typing import Union

# Use the standardized config loader
try:
    from .config_loader import load_config
except ImportError:
    try:
        from config_loader import load_config
    except ImportError:
        def load_config(path=None): return {}
        logger = logging.getLogger(__name__)
        logger.warning("Could not import config_loader. Using empty config.")

logger = logging.getLogger(__name__)
app_config = load_config()

SQLITE_DB_PATH = '/data/blueprint_data.db'

# --- SQLite Connection ---

def get_sqlite_connection():
    """Get connection to the add-on's local SQLite database."""
    try:
        # Ensure the /data directory exists
        os.makedirs('/data', exist_ok=True)
        conn = sqlite3.connect(SQLITE_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row # Use Row factory for dict-like access
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;") # Enforce foreign keys if used
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite at {SQLITE_DB_PATH}: {str(e)}", exc_info=True)
        raise # Re-raise critical error

# --- Schema Initialization ---

def init_sqlite_db():
    """Initialize SQLite database schema for the add-on."""
    conn = None
    try:
        logger.info(f"Initializing/Verifying SQLite database schema at {SQLITE_DB_PATH}")
        conn = get_sqlite_connection()
        cursor = conn.cursor()

        # --- Blueprints Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blueprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL -- Store as ISO format string
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_blueprints_created ON blueprints (created_at DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_blueprints_status ON blueprints (status);')

        # --- Area Observations Table (UPDATED) ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS area_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Store as ISO format string
                tracked_device_id TEXT NOT NULL,
                predicted_area_id TEXT -- Can be NULL if prediction is unavailable
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_obs_time ON area_observations (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_obs_device_area ON area_observations (tracked_device_id, predicted_area_id);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_area_obs_device_time ON area_observations (tracked_device_id, timestamp DESC);')

        # --- RSSI Distance Samples Table (for Training) ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rssi_distance_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                sensor_id TEXT NOT NULL,
                rssi REAL NOT NULL,
                distance REAL,
                tx_power REAL,
                frequency REAL,
                environment_type TEXT,
                device_type TEXT,
                time_of_day INTEGER,
                day_of_week INTEGER,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP -- Store as ISO format string
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rssi_samples_time ON rssi_distance_samples (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rssi_samples_device_sensor ON rssi_distance_samples (device_id, sensor_id);')

        # --- AI Models Metadata Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL, -- e.g., 'rssi_distance', 'wall_prediction'
                model_type TEXT,               -- e.g., 'random_forest', 'cnn'
                model_path TEXT,               -- Path within /data/models
                metrics TEXT,                  -- JSON string of training metrics
                version INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, -- Store as ISO format string
                last_trained_at TEXT           -- Store as ISO format string
            )
        ''')
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_models_name ON ai_models (model_name);')


        # --- AI Blueprint Feedback Table (Optional, for RL) ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_blueprint_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blueprint_id TEXT NOT NULL,    -- Identifier for the original blueprint
                original_blueprint TEXT,       -- JSON of blueprint before refinement
                modified_blueprint TEXT,       -- JSON of blueprint after refinement
                feedback_data TEXT,            -- JSON containing score or other feedback
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP -- Store as ISO format string
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON ai_blueprint_feedback (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_blueprint_id ON ai_blueprint_feedback (blueprint_id);')

        # --- Device Positions Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                position_data TEXT NOT NULL, -- JSON blob with x, y, z
                source TEXT NOT NULL,        -- 'fixed_reference', 'calculated', 'manual', etc.
                accuracy REAL,               -- Estimated accuracy in meters
                area_id TEXT,                -- Optional area/room identifier
                timestamp TEXT NOT NULL      -- Store as ISO format string
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pos_device ON device_positions (device_id);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pos_timestamp ON device_positions (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pos_source ON device_positions (source);')

        # --- Distance Log Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS distance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                tracked_device_id TEXT NOT NULL,
                scanner_id TEXT NOT NULL,
                distance REAL NOT NULL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_distance_log_timestamp ON distance_log (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_distance_log_device ON distance_log (tracked_device_id);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_distance_log_scanner ON distance_log (scanner_id);')

        conn.commit()
        logger.info("SQLite database schema initialized/verified successfully.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to initialize SQLite schema: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()


# --- Internal Execution Helpers ---

def _execute_sqlite_write(query: str, params: Optional[Tuple] = None, fetch_last_id: bool = False) -> Optional[int]:
    """Helper function for SQLite writes."""
    conn = None
    last_id = None
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        if fetch_last_id:
            last_id = cursor.lastrowid
        conn.commit()
        logger.debug(f"SQLite write successful: {query[:60]}...")
        return last_id
    except sqlite3.Error as e:
        logger.error(f"SQLite write query failed: {query[:60]}... Error: {str(e)}", exc_info=True)
        if conn:
            conn.rollback()
        return None # Indicate failure explicitly
    finally:
        if conn:
            conn.close()

def _execute_sqlite_read(query: str, params: Optional[Tuple] = None, fetch_one: bool = False) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
    """Helper function for SQLite reads. Returns list of dicts or single dict."""
    conn = None
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        cursor.execute(query, params or ())
        if fetch_one:
            row = cursor.fetchone()
            return dict(row) if row else None
        else:
            # Convert all rows to dictionaries
            return [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as e:
        logger.error(f"SQLite read query failed: {query[:60]}... Error: {str(e)}", exc_info=True)
        return None # Indicate failure
    finally:
        if conn:
            conn.close()


# --- Public Helper Functions ---

def save_blueprint_to_sqlite(blueprint_data: Dict) -> bool:
    """Save a blueprint to the SQLite database."""
    if not blueprint_data or 'rooms' not in blueprint_data:
        logger.error("Attempted to save invalid blueprint data (missing 'rooms')")
        return False

    query = """
    INSERT INTO blueprints (data, status, created_at)
    VALUES (?, ?, ?)
    """
    created_at = blueprint_data.get('generated_at', datetime.now().isoformat())
    data_json = json.dumps(blueprint_data)
    status = blueprint_data.get('status', 'active')
    params = (data_json, status, created_at)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.info(f"Successfully saved blueprint created at {created_at} to SQLite.")
        return True
    else:
        logger.error(f"Failed to save blueprint created at {created_at} to SQLite.")
        return False


def save_device_position_to_sqlite(
    device_id: str,
    position_data: Dict,
    source: str = 'calculated',
    accuracy: Optional[float] = None,
    area_id: Optional[str] = None
) -> bool:
    """Save a device position to the SQLite database.

    Args:
        device_id: Unique identifier for the device
        position_data: Dictionary containing x, y, z coordinates
        source: Source of the position data ('fixed_reference', 'calculated', 'manual')
        accuracy: Estimated accuracy in meters (optional)
        area_id: Optional area/room identifier

    Returns:
        bool: True if saved successfully, False otherwise
    """
    # Ensure we have the minimum position data
    if not all(k in position_data for k in ['x', 'y', 'z']):
        logger.error(f"Invalid position data for device {device_id}: missing x/y/z coordinate")
        return False

    query = """
    INSERT INTO device_positions (device_id, position_data, source, accuracy, area_id, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """

    position_json = json.dumps(position_data)
    timestamp = datetime.now().isoformat()
    params = (device_id, position_json, source, accuracy, area_id, timestamp)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.info(f"Saved position for device {device_id} (source: {source})")
        return True
    else:
        logger.error(f"Failed to save position for device {device_id}")
        return False

def get_reference_positions_from_sqlite() -> Dict[str, Dict]:
    """Get the latest positions of all fixed reference devices.

    Returns:
        Dict mapping device_id to position {x, y, z} dict
    """
    query = """
    SELECT d1.device_id, d1.position_data
    FROM device_positions d1
    JOIN (
        SELECT device_id, MAX(timestamp) as max_time
        FROM device_positions
        WHERE source = 'fixed_reference'
        GROUP BY device_id
    ) d2 ON d1.device_id = d2.device_id AND d1.timestamp = d2.max_time
    WHERE d1.source = 'fixed_reference'
    """

    results = _execute_sqlite_read(query)
    reference_positions = {}

    if results:
        for row in results:
            try:
                device_id = row['device_id']
                position = json.loads(row['position_data'])
                reference_positions[device_id] = position
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing reference position for {row.get('device_id', 'unknown')}: {e}")

    logger.debug(f"Retrieved {len(reference_positions)} reference positions from database")
    return reference_positions

def get_device_positions_from_sqlite() -> Dict[str, Dict]:
    """Get the latest positions of all tracked devices.

    Returns:
        Dict mapping device_id to full position dict including source, accuracy, etc.
    """
    query = """
    SELECT d1.device_id, d1.position_data, d1.source, d1.accuracy, d1.area_id, d1.timestamp
    FROM device_positions d1
    JOIN (
        SELECT device_id, MAX(timestamp) as max_time
        FROM device_positions
        GROUP BY device_id
    ) d2 ON d1.device_id = d2.device_id AND d1.timestamp = d2.max_time
    """

    results = _execute_sqlite_read(query)
    device_positions = {}

    if results:
        for row in results:
            try:
                device_id = row['device_id']
                position_data = json.loads(row['position_data'])

                # Create a complete position record
                position_record = {
                    **position_data,  # Unpack the x, y, z
                    'source': row['source'],
                    'timestamp': row['timestamp'],
                }

                # Add optional fields if present
                if row['accuracy'] is not None:
                    position_record['accuracy'] = float(row['accuracy'])
                if row['area_id']:
                    position_record['area_id'] = row['area_id']

                device_positions[device_id] = position_record
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing device position for {row.get('device_id', 'unknown')}: {e}")

    logger.debug(f"Retrieved {len(device_positions)} device positions from database")
    return device_positions

def save_ai_feedback_to_sqlite(blueprint_id: str, feedback_data: Dict, original_blueprint: Optional[Dict] = None, modified_blueprint: Optional[Dict] = None) -> bool:
    """Save AI blueprint feedback to the SQLite database."""
    query = """
    INSERT INTO ai_blueprint_feedback (blueprint_id, original_blueprint, modified_blueprint, feedback_data, timestamp)
    VALUES (?, ?, ?, ?, ?)
    """
    feedback_json = json.dumps(feedback_data)
    original_json = json.dumps(original_blueprint) if original_blueprint else None
    modified_json = json.dumps(modified_blueprint) if modified_blueprint else None
    timestamp = datetime.now().isoformat()
    params = (blueprint_id, original_json, modified_json, feedback_json, timestamp)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.info(f"Successfully saved AI feedback for blueprint {blueprint_id}")
        return True
    else:
        logger.error(f"Error saving AI feedback to SQLite for blueprint {blueprint_id}")
        return False

def save_rssi_sample_to_sqlite(
    device_id: str,
    sensor_id: str,
    rssi: float,
    distance: Optional[float] = None,
    tx_power: Optional[float] = None,
    frequency: Optional[float] = None,
    environment_type: Optional[str] = None,
    device_type: Optional[str] = None,
    time_of_day: Optional[int] = None,
    day_of_week: Optional[int] = None
) -> bool:
    """Save an RSSI-to-distance training sample to SQLite database."""
    # Get current time information if not provided
    current_time = datetime.now()
    if time_of_day is None:
        time_of_day = current_time.hour
    if day_of_week is None:
        day_of_week = current_time.weekday() # 0-6, Monday is 0

    query = '''
    INSERT INTO rssi_distance_samples (
        device_id, sensor_id, rssi, distance, tx_power, frequency,
        environment_type, device_type, time_of_day, day_of_week, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    params = (
        device_id, sensor_id, rssi, distance, tx_power, frequency,
        environment_type, device_type, time_of_day, day_of_week,
        current_time.isoformat()
    )

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.debug(f"Saved RSSI sample: {device_id} via {sensor_id} RSSI:{rssi} -> Dist:{distance}")
        return True
    else:
        logger.error("Failed to save RSSI sample to SQLite.")
        return False


def get_latest_blueprint_from_sqlite() -> Optional[Dict[str, Any]]:
    """Get the latest active blueprint from the SQLite database."""
    query = """
    SELECT data, created_at FROM blueprints
    WHERE status = 'active'
    ORDER BY created_at DESC, id DESC LIMIT 1
    """
    result = _execute_sqlite_read(query, fetch_one=True)
    if result and result.get('data'):
        try:
            blueprint = json.loads(result['data'])
            logger.info(f"Retrieved latest active blueprint from {result.get('created_at')}")
            return blueprint
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse latest blueprint from SQLite: {e}")
            return None
    logger.warning("No active blueprints found in SQLite database")
    return None


def save_ai_model_sqlite(model_name: str, model_type: str, model_path: str, metrics: Dict) -> bool:
    """Save or update AI model metadata in SQLite."""
    if not model_name or not model_path:
        logger.error("Cannot save AI model: missing required fields (model_name or model_path)")
        return False

    try:
        # Ensure model_path is a string
        model_path_str = str(model_path)
        metrics_json = json.dumps(metrics)
        timestamp = datetime.now().isoformat()

        query = """
        INSERT INTO ai_models (model_name, model_type, model_path, metrics, last_trained_at, version)
        VALUES (?, ?, ?, ?, ?, 1)
        ON CONFLICT(model_name) DO UPDATE SET
            model_type=excluded.model_type,
            model_path=excluded.model_path,
            metrics=excluded.metrics,
            last_trained_at=excluded.last_trained_at,
            version=version + 1
        """
        params = (model_name, model_type, model_path_str, metrics_json, timestamp)

        result = _execute_sqlite_write(query, params)
        if result is not None:
            logger.info(f"Saved/Updated AI model info for '{model_name}'")
            return True
        else:
            logger.error(f"Failed to save AI model info for '{model_name}'")
            return False
    except (TypeError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to save AI model '{model_name}': Invalid data format: {e}")
        return False

# --- NEW Helper for Area Observations ---
def save_area_observation(tracked_device_id: str, predicted_area_id: Optional[str]) -> bool:
    """Saves a snapshot of a device's predicted area."""
    query = """
    INSERT INTO area_observations (timestamp, tracked_device_id, predicted_area_id)
    VALUES (?, ?, ?)
    """
    timestamp = datetime.now().isoformat()
    params = (timestamp, tracked_device_id, predicted_area_id)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.debug(f"Saved area observation for {tracked_device_id} -> {predicted_area_id}")
        return True
    else:
        logger.error(f"Failed to save area observation for {tracked_device_id}")
        return False

def get_area_observations(
    limit: int = 10000, # Increased default limit for adjacency calculation
    device_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves area observation records from the database."""
    # NOTE: Adjusted SELECT based on your save_area_observation function
    # (removed rssi_vector, is_transition)
    query = """
    SELECT timestamp, tracked_device_id, predicted_area_id
    FROM area_observations
    """
    conditions = []
    params = []

    if device_id:
        conditions.append("tracked_device_id = ?")
        params.append(device_id)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY timestamp DESC LIMIT ?"
    params.append(limit)

    # Use the internal helper function
    results = _execute_sqlite_read(query, tuple(params))
    # _execute_sqlite_read already converts rows to dicts
    return results if results is not None else []

def get_recent_distances(time_window_minutes: int = 15) -> List[Dict[str, Any]]:
    """Fetches recent distance logs within specified time window."""
    query = """
    SELECT timestamp, tracked_device_id, scanner_id, distance
    FROM distance_log
    WHERE timestamp >= ?
    ORDER BY timestamp DESC
    """
    # Calculate cutoff time based on current time
    cutoff_dt = datetime.now() - timedelta(minutes=time_window_minutes)
    cutoff_time_iso = cutoff_dt.isoformat()

    params = (cutoff_time_iso,)
    logger.debug(f"Querying distances since {cutoff_time_iso}")
    results = _execute_sqlite_read(query, params) # Use the enhanced reader
    return results if results is not None else []

def get_recent_area_predictions(time_window_minutes: int = 10) -> Dict[str, Optional[str]]:
    """Gets the most recent area prediction for each device within the window."""
    query = """
    SELECT tracked_device_id, predicted_area_id
    FROM area_observations
    WHERE timestamp >= ? AND id IN (
        SELECT MAX(id)
        FROM area_observations
        WHERE timestamp >= ?
        GROUP BY tracked_device_id
    )
    """
    # Calculate cutoff time based on current time
    cutoff_dt = datetime.now() - timedelta(minutes=time_window_minutes)
    cutoff_time_iso = cutoff_dt.isoformat()

    params = (cutoff_time_iso, cutoff_time_iso)
    logger.debug(f"Querying most recent area predictions since {cutoff_time_iso}")
    results = _execute_sqlite_read(query, params) # Use the enhanced reader
    if results is None:
        return {}
    # Create a dictionary mapping device_id to the predicted_area_id
    predictions = {row['tracked_device_id']: row['predicted_area_id'] for row in results}
    logger.debug(f"Found {len(predictions)} recent area predictions.")
    return predictions

# Add this function
def save_distance_log(tracked_device_id: str, scanner_id: str, distance: float) -> bool:
    """Save a distance measurement to the log."""
    query = """
    INSERT INTO distance_log (timestamp, tracked_device_id, scanner_id, distance)
    VALUES (?, ?, ?, ?)
    """
    timestamp = datetime.now().isoformat()
    params = (timestamp, tracked_device_id, scanner_id, distance)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.debug(f"Saved distance log: {tracked_device_id} to {scanner_id} = {distance}m")
        return True
    else:
        logger.error(f"Failed to save distance log for {tracked_device_id}")
        return False

# For compatibility with existing code
execute_sqlite_query = _execute_sqlite_read
execute_query = _execute_sqlite_read
execute_write_query = _execute_sqlite_write