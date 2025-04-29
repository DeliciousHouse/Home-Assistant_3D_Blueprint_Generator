import json
import logging
import sqlite3
import os
import threading
import time  # Added for retry delays
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from typing import Union
import uuid
import traceback
import numpy as np
import websocket

# Use the standardized config loader
try:
    from .config_loader import load_config
except ImportError:
    try:
        from config_loader import load_config
    except ImportError:
        def load_config(path=None):
            logger = logging.getLogger(__name__)
            logger.warning("Could not import config_loader. Using empty config.")
            return {}

logger = logging.getLogger(__name__)
app_config = load_config()

# Update the path to use a local project directory instead of system directories
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SQLITE_DB_PATH = os.path.join(PROJECT_DIR, 'data', 'blueprint_generator_db')
os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)

logger.info(f"Using SQLite database path: {SQLITE_DB_PATH}")

# --- SQLite Connection ---

def get_sqlite_connection():
    """Get connection to the add-on's local SQLite database."""
    try:
        # Ensure data directory exists
        data_dir = os.path.dirname(SQLITE_DB_PATH)
        os.makedirs(data_dir, exist_ok=True)

        # Connect with row factory set to Row
        conn = sqlite3.connect(f"{SQLITE_DB_PATH}.sqlite")
        conn.row_factory = sqlite3.Row  # This is critical for dict conversion later
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to SQLite database: {str(e)}")
        return None

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

        # --- Reference Positions Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reference_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL UNIQUE,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL,
                area_id TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ref_pos_device ON reference_positions (device_id);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ref_pos_area ON reference_positions (area_id);')

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
    """Helper function for SQLite writes with retry logic for database locks."""
    conn = None
    last_id = None
    max_retries = 3  # Number of times to retry if locked
    retry_delay = 0.1  # Initial delay between retries (seconds)

    for attempt in range(max_retries):
        try:
            conn = get_sqlite_connection()
            if not conn:
                logger.error(f"Failed to get database connection on attempt {attempt+1}/{max_retries}")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                continue

            cursor = conn.cursor()
            cursor.execute(query, params or ())
            conn.commit()

            if fetch_last_id:
                last_id = cursor.lastrowid

            cursor.close()
            conn.close()

            return last_id if fetch_last_id else 1  # Return ID or success indicator

        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                logger.warning(f"Database locked, retrying in {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"SQLite operational error on write: {e}")
                break

        except Exception as e:
            logger.error(f"Error executing SQLite write: {e}")
            break

        finally:
            if conn:
                try:
                    conn.close()
                except:
                    pass

    return None  # Indicate failure after all retries

def _execute_sqlite_read(query: str, params: Optional[Tuple] = None, fetch_one: bool = False) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
    """Helper function for SQLite reads with improved error handling."""
    conn = None
    max_retries = 2  # Fewer retries for reads
    retry_delay = 0.05  # Shorter delay for reads

    for attempt in range(max_retries):
        try:
            conn = get_sqlite_connection()
            if not conn:
                logger.error("Failed to get SQLite connection")
                return None

            # Ensure row_factory is set to Row
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            cursor.execute(query, params or ())

            if fetch_one:
                row = cursor.fetchone()
                result = dict(row) if row else None
            else:
                # Convert sqlite3.Row objects to dictionaries properly
                result = []
                for row in cursor.fetchall():
                    try:
                        # Create a dictionary with column names as keys
                        row_dict = {}
                        for idx, col in enumerate(cursor.description):
                            row_dict[col[0]] = row[idx]
                        result.append(row_dict)
                    except Exception as row_e:
                        logger.error(f"Error converting row to dict: {row_e}")

            return result
        except sqlite3.Error as e:
            logger.error(f"SQLite read attempt {attempt+1} failed: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
            else:
                logger.error(f"SQLite read failed after {max_retries} attempts: {query[:100]}...")
        except Exception as e:
            logger.error(f"Unexpected error in _execute_sqlite_read: {str(e)}", exc_info=True)
        finally:
            if conn:
                conn.close()

    return None  # Indicate failure after all retries

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
        logger.info(f"Blueprint saved to SQLite database with {len(blueprint_data.get('rooms', []))} rooms")
        return True
    else:
        logger.error("Failed to save blueprint to SQLite database")
        return False

def get_reference_positions_from_sqlite() -> Dict[str, Dict[str, Any]]:
    """Retrieve reference positions from SQLite database.
    Returns a dictionary mapping device_id to position data.
    """
    query = """
    SELECT device_id, x, y, z, area_id
    FROM reference_positions
    """
    results = _execute_sqlite_read(query) # Use the internal helper

    if results is None: # Handle potential read error
        logger.error("Failed to read reference positions from database.")
        return {}
    if not results:
        logger.info("No reference positions found in database.")
        return {}

    reference_positions = {}
    for row in results:
        device_id = row['device_id']
        reference_positions[device_id] = {
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'area_id': row['area_id']
            # Add other relevant fields if needed, e.g., created_at
        }
    logger.info(f"Retrieved {len(reference_positions)} reference positions.")
    return reference_positions

def save_reference_position(device_id: str, x: float, y: float, z: float, area_id: Optional[str] = None) -> bool:
    """Save or update a reference position in the database."""
    query = """
    INSERT INTO reference_positions (device_id, x, y, z, area_id, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(device_id) DO UPDATE SET
        x = excluded.x,
        y = excluded.y,
        z = excluded.z,
        area_id = excluded.area_id,
        updated_at = excluded.updated_at
    """
    timestamp = datetime.now().isoformat()
    params = (device_id, x, y, z, area_id, timestamp, timestamp)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.info(f"Saved reference position for device {device_id} at ({x}, {y}, {z})")
        return True
    else:
        logger.error(f"Failed to save reference position for device {device_id}")
        return False

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
            # Check if the data string starts with a comment and remove it
            data_str = result['data']
            if data_str.startswith('//'):
                # Find the end of the comment line
                newline_pos = data_str.find('\n')
                if newline_pos > 0:
                    data_str = data_str[newline_pos + 1:].strip()

            # Parse the cleaned JSON data
            blueprint_data = json.loads(data_str)
            logger.info(f"Retrieved blueprint from {result.get('created_at', 'unknown date')}")
            return blueprint_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse blueprint JSON: {e}")
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

    # Add extensive logging to diagnose the issue
    logger.debug(f"Attempting to save area observation: device='{tracked_device_id}', area='{predicted_area_id}'")

    try:
        conn = get_sqlite_connection()
        if not conn:
            logger.error(f"Failed to get database connection for area observation: {tracked_device_id}")
            return False

        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()

        if cursor.rowcount > 0:
            logger.debug(f"Successfully saved area observation for {tracked_device_id} -> {predicted_area_id}")
            cursor.close()
            conn.close()
            return True
        else:
            logger.warning(f"No rows affected when saving area observation for {tracked_device_id}")
            cursor.close()
            conn.close()
            return False

    except sqlite3.Error as e:
        logger.error(f"SQLite error saving area observation for {tracked_device_id}: {e}")
        if 'no such table' in str(e).lower():
            logger.error("The area_observations table doesn't exist! Check database initialization.")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving area observation for {tracked_device_id}: {e}", exc_info=True)
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
    """Get recent distance logs within the specified time window."""
    query = """
    SELECT timestamp, tracked_device_id, scanner_id, distance
    FROM distance_log
    WHERE timestamp >= ?
    ORDER BY timestamp DESC
    """
    cutoff_time = (datetime.now() - timedelta(minutes=time_window_minutes)).isoformat()
    params = (cutoff_time,)

    results = _execute_sqlite_read(query, params)

    # Add debug logging HERE to inspect the 'results' structure
    if results is not None:
        logger.debug(f"get_recent_distances: Retrieved {len(results)} records.")
        if len(results) > 0:
            logger.debug(f"First record structure: {results[0]}") # Log the first record to see its format
            # Check if the first record looks like the header row
            if isinstance(results[0], dict) and results[0].get('tracked_device_id') == 'tracked_device_id':
                 logger.error("!!! Database query returned header row as data !!!")
                 # Potentially skip the first row if it's consistently the header
                 # return results[1:] if len(results) > 1 else []
    else:
        logger.warning("get_recent_distances: Query returned None.")

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

def get_device_positions_from_sqlite(limit: int = 1000, time_window_minutes: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Get the latest device positions from the SQLite database.

    Args:
        limit: Maximum number of positions to retrieve per device
        time_window_minutes: Optional time window to filter positions by

    Returns:
        Dict mapping device_id to position data
    """
    try:
        # Build query with optional time filter
        query = """
        SELECT dp.device_id, dp.position_data, dp.source, dp.accuracy, dp.area_id, dp.timestamp
        FROM device_positions dp
        INNER JOIN (
            SELECT device_id, MAX(timestamp) as latest_timestamp
            FROM device_positions
        """

        params = []

        # Add time window condition if specified
        if time_window_minutes is not None:
            cutoff_time = (datetime.now() - timedelta(minutes=time_window_minutes)).isoformat()
            query += " WHERE timestamp >= ?"
            params.append(cutoff_time)

        query += """
            GROUP BY device_id
        ) latest ON dp.device_id = latest.device_id AND dp.timestamp = latest.latest_timestamp
        LIMIT ?
        """
        params.append(limit)

        # Execute query
        results = _execute_sqlite_read(query, tuple(params))

        if not results:
            logger.info("No device positions found in database")
            return {}

        # Process results
        device_positions = {}
        for row in results:
            device_id = row['device_id']

            try:
                # Parse the position data JSON
                position_data = json.loads(row['position_data']) if isinstance(row['position_data'], str) else row['position_data']

                # Ensure position data has x, y, z coordinates
                if not all(key in position_data for key in ['x', 'y', 'z']):
                    logger.warning(f"Invalid position data for device {device_id}: missing coordinates")
                    continue

                # Add additional metadata
                position_data.update({
                    'source': row['source'],
                    'accuracy': row['accuracy'],
                    'area_id': row['area_id'],
                    'timestamp': row['timestamp']
                })

                device_positions[device_id] = position_data

            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error parsing position data for device {device_id}: {e}")
                continue

        logger.info(f"Retrieved {len(device_positions)} device positions from database")
        return device_positions

    except Exception as e:
        logger.error(f"Error getting device positions from database: {e}", exc_info=True)
        return {}

def save_device_position(device_id: str, position_data: Dict[str, Any], source: str = 'calculated', accuracy: Optional[float] = None, area_id: Optional[str] = None) -> bool:
    """
    Save a device position to the SQLite database.

    Args:
        device_id: Unique identifier for the device
        position_data: Dict with position data (must contain x, y, z)
        source: Source of the position data (e.g., 'calculated', 'manual')
        accuracy: Optional estimated accuracy in meters
        area_id: Optional area/room identifier

    Returns:
        True if successful, False otherwise
    """
    try:
        # Validate position data
        if not isinstance(position_data, dict) or not all(key in position_data for key in ['x', 'y', 'z']):
            logger.error(f"Invalid position data for device {device_id}: {position_data}")
            return False

        # Convert position data to JSON
        position_json = json.dumps(position_data)

        # Prepare query and parameters
        query = """
        INSERT INTO device_positions (device_id, position_data, source, accuracy, area_id, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """

        timestamp = datetime.now().isoformat()
        params = (device_id, position_json, source, accuracy, area_id, timestamp)

        # Execute query
        result = _execute_sqlite_write(query, params)

        if result is not None:
            logger.debug(f"Saved position for device {device_id} at ({position_data.get('x')}, {position_data.get('y')}, {position_data.get('z')})")
            return True
        else:
            logger.error(f"Failed to save position for device {device_id}")
            return False

    except Exception as e:
        logger.error(f"Error saving device position: {e}", exc_info=True)
        return False

# For compatibility with existing code
execute_sqlite_query = _execute_sqlite_read
execute_query = _execute_sqlite_read
execute_write_query = _execute_sqlite_write