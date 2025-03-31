# server/db.py
import json
import logging
import sqlite3
import os
from typing import Any, Dict, List, Optional, Tuple, Union # Added Union
from datetime import datetime

# Use the standardized config loader
try:
    # Assuming config_loader.py is in the same directory (server/)
    from .config_loader import load_config
except ImportError:
    # Fallback if structure is different (e.g., config_loader is one level up)
    try:
        from config_loader import load_config
    except ImportError:
        # Last resort: simple default
        def load_config(path=None): return {}
        # Setup basic logger if config load fails early
        logging.basicConfig(level=logging.WARNING)
        logger = logging.getLogger(__name__)
        logger.warning("Could not import config_loader. Using empty config.")

logger = logging.getLogger(__name__)
app_config = load_config() # Load config once when module is loaded

SQLITE_DB_PATH = '/data/blueprint_data.db'

# --- SQLite Connection ---

def get_sqlite_connection():
    """Get connection to the add-on's local SQLite database."""
    try:
        db_dir = os.path.dirname(SQLITE_DB_PATH)
        os.makedirs(db_dir, exist_ok=True) # Ensure directory exists
        logger.debug(f"Attempting to connect to SQLite DB at {SQLITE_DB_PATH}")
        conn = sqlite3.connect(SQLITE_DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row # Use Row factory for dict-like access
        # Enable WAL mode for better concurrency
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys = ON;") # Enforce foreign keys if used
        logger.debug("SQLite connection successful.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"CRITICAL: Failed to connect to SQLite at {SQLITE_DB_PATH}: {type(e).__name__} - {str(e)}", exc_info=True)
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

        # --- Distance Log Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS distance_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, -- ISO format string
                tracked_device_id TEXT NOT NULL,
                scanner_id TEXT NOT NULL,
                distance REAL NOT NULL
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dist_log_time ON distance_log (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_dist_log_device_scanner_time ON distance_log (tracked_device_id, scanner_id, timestamp DESC);')

        # --- Area Observations Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS area_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL, -- ISO format string
                tracked_device_id TEXT NOT NULL,
                predicted_area_id TEXT -- Can be NULL if prediction is unavailable
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_area_obs_time ON area_observations (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_area_obs_device_time ON area_observations (tracked_device_id, timestamp DESC);')

        # # --- RSSI Distance Samples Table (Optional: For separate distance model training) ---
        # # Keep if you might manually calibrate distance later, otherwise remove
        # cursor.execute('''
        #     CREATE TABLE IF NOT EXISTS rssi_distance_samples (
        #         id INTEGER PRIMARY KEY AUTOINCREMENT,
        #         device_id TEXT NOT NULL,
        #         sensor_id TEXT NOT NULL,
        #         rssi REAL NOT NULL,
        #         distance REAL,
        #         tx_power REAL,
        #         frequency REAL,
        #         environment_type TEXT,
        #         device_type TEXT,
        #         time_of_day INTEGER,
        #         day_of_week INTEGER,
        #         timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP -- Store as ISO format string
        #     )
        # ''')
        # cursor.execute('CREATE INDEX IF NOT EXISTS idx_rssi_samples_time ON rssi_distance_samples (timestamp DESC);')
        # cursor.execute('CREATE INDEX IF NOT EXISTS idx_rssi_samples_device_sensor ON rssi_distance_samples (device_id, sensor_id);')

        # --- AI Models Metadata Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL,
                model_type TEXT,
                model_path TEXT,
                metrics TEXT,
                version INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_trained_at TEXT
            )
        ''')
        cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_models_name ON ai_models (model_name);')


        # --- AI Blueprint Feedback Table (Optional, for RL/Refinement) ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ai_blueprint_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                blueprint_id TEXT NOT NULL,
                original_blueprint TEXT,
                modified_blueprint TEXT,
                feedback_data TEXT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON ai_blueprint_feedback (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_blueprint_id ON ai_blueprint_feedback (blueprint_id);')

        # --- REMOVED device_positions table ---
        # cursor.execute('DROP TABLE IF EXISTS device_positions;') # Optional cleanup

        # --- Reference Positions Table ---
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS reference_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_id TEXT NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL,
            area_id TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_ref_positions_device_id ON reference_positions (device_id);')

        # --- Test Write/Read ---
        cursor.execute("INSERT INTO blueprints (data, status, created_at) VALUES (?, ?, ?)", ('{"test": true}', 'test', datetime.now().isoformat()))
        conn.commit()
        cursor.execute("SELECT id FROM blueprints WHERE status = 'test' LIMIT 1")
        test_result = cursor.fetchone()
        if test_result:
            cursor.execute("DELETE FROM blueprints WHERE status = 'test'")
            conn.commit()
            logger.info("SQLite basic write/read test successful.")
        else:
            logger.error("SQLite basic write/read test FAILED.")
            return False # Indicate failure if test fails
        # --- End Test ---

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
    """Helper function for SQLite writes with enhanced logging."""
    conn = None
    last_id = None
    # Limit query logging for brevity
    log_query = query.strip()[:150] + ('...' if len(query.strip()) > 150 else '')
    # Mask sensitive data in params if necessary (not needed here)
    log_params = params or ()

    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        logger.debug(f"Executing SQLite write: Query='{log_query}' PARAMS={log_params}")
        cursor.execute(query, params or ())
        if fetch_last_id:
            last_id = cursor.lastrowid
        conn.commit()
        logger.debug(f"SQLite write successful for Query='{log_query}'")
        return last_id
    except sqlite3.Error as e:
        logger.error(f"SQLite write FAILED! Query='{log_query}' PARAMS={log_params} Error: {type(e).__name__} - {str(e)}", exc_info=True)
        if conn:
            try:
                conn.rollback()
                logger.info("SQLite transaction rolled back.")
            except sqlite3.Error as rb_e:
                logger.error(f"Failed to rollback transaction: {rb_e}")
        return None # Indicate failure explicitly
    finally:
        if conn:
            conn.close()

def _execute_sqlite_read(query: str, params: Optional[Tuple] = None, fetch_one: bool = False) -> Optional[Union[List[Dict[str, Any]], Dict[str, Any]]]:
    """Helper function for SQLite reads with enhanced logging."""
    conn = None
    log_query = query.strip()[:150] + ('...' if len(query.strip()) > 150 else '')
    log_params = params or ()
    try:
        conn = get_sqlite_connection()
        cursor = conn.cursor()
        logger.debug(f"Executing SQLite read: Query='{log_query}' PARAMS={log_params}")
        cursor.execute(query, params or ())
        if fetch_one:
            row = cursor.fetchone()
            logger.debug(f"SQLite read fetch_one successful for Query='{log_query}'. Found: {row is not None}")
            return dict(row) if row else None
        else:
            rows = cursor.fetchall()
            logger.debug(f"SQLite read fetch_all successful for Query='{log_query}'. Rows fetched: {len(rows)}")
            return [dict(row) for row in rows] # Convert all rows
    except sqlite3.Error as e:
        logger.error(f"SQLite read FAILED! Query='{log_query}' PARAMS={log_params} Error: {type(e).__name__} - {str(e)}", exc_info=True)
        return None # Indicate failure
    finally:
        if conn:
            conn.close()

def execute_sqlite_query(query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """Alias for _execute_sqlite_read for backward compatibility."""
    return _execute_sqlite_read(query, params)

def execute_query(query: str, params: Optional[Tuple] = None) -> List[Dict[str, Any]]:
    """Alias for _execute_sqlite_read for backward compatibility."""
    return _execute_sqlite_read(query, params)

def execute_write_query(query: str, params: Optional[Tuple] = None, fetch_last_id: bool = False) -> Optional[int]:
    """Alias for _execute_sqlite_write for backward compatibility."""
    return _execute_sqlite_write(query, params, fetch_last_id)

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
    if time_of_day is None: time_of_day = current_time.hour
    if day_of_week is None: day_of_week = current_time.weekday() # 0-6, Monday is 0

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
    model_path_str = str(model_path)
    metrics_json = json.dumps(metrics)
    timestamp = datetime.now().isoformat()
    params = (model_name, model_type, model_path_str, metrics_json, timestamp)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.info(f"Saved/Updated AI model info for '{model_name}'")
        return True
    else:
        logger.error(f"Failed to save AI model info for '{model_name}'")
        return False

def save_distance_log(tracked_device_id: str, scanner_id: str, distance: float) -> bool:
    """Save a distance reading to the database."""
    query = """
    INSERT INTO distance_log (timestamp, tracked_device_id, scanner_id, distance)
    VALUES (?, ?, ?, ?)
    """
    timestamp = datetime.now().isoformat()
    params = (timestamp, tracked_device_id, scanner_id, distance)
    logger.debug(f"Attempting to save distance log with params: {params}") # Log before write attempt
    result = _execute_sqlite_write(query, params)
    if result is None:
        logger.error(f"Failed DB write for distance log: device={tracked_device_id}, scanner={scanner_id}, dist={distance}")
        return False
    else:
        return True

def save_area_observation(tracked_device_id: str, predicted_area_id: Optional[str]) -> bool:
    """Save an area prediction observation to the database."""
    query = """
    INSERT INTO area_observations (timestamp, tracked_device_id, predicted_area_id)
    VALUES (?, ?, ?)
    """
    timestamp = datetime.now().isoformat()
    params = (timestamp, tracked_device_id, predicted_area_id)
    logger.debug(f"Attempting to save area observation with params: {params}") # Log before write attempt
    result = _execute_sqlite_write(query, params)
    if result is None:
        logger.error(f"Failed DB write for area observation: device={tracked_device_id}, area={predicted_area_id}")
        return False
    else:
        return True

def get_recent_distances(time_window_minutes: int = 10) -> List[Dict[str, Any]]:
    """Fetches recent distance logs within specified time window."""
    query = """
    SELECT timestamp, tracked_device_id, scanner_id, distance
    FROM distance_log
    WHERE timestamp >= ?
    ORDER BY timestamp DESC
    """
    cutoff_time = datetime.fromtimestamp(datetime.now().timestamp() - time_window_minutes * 60).isoformat()
    params = (cutoff_time,)
    results = _execute_sqlite_read(query, params) # Use the enhanced reader
    return results if results is not None else []

def get_recent_area_predictions(time_window_minutes: int = 5) -> Dict[str, Optional[str]]:
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
    cutoff_time = datetime.fromtimestamp(datetime.now().timestamp() - time_window_minutes * 60).isoformat()
    params = (cutoff_time, cutoff_time)
    results = _execute_sqlite_read(query, params) # Use the enhanced reader
    if results is None:
        return {}
    return {row['tracked_device_id']: row['predicted_area_id'] for row in results}

def get_area_observations(
    limit: int = 5000,
    device_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Retrieves area observation records from the database."""
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

    results = _execute_sqlite_read(query, tuple(params))
    return results if results is not None else []

def get_reference_positions_from_sqlite() -> Dict[str, Dict[str, Any]]:
    """Retrieve reference positions from SQLite database.
    Returns a dictionary mapping device_id to position data.
    """
    query = """
    SELECT device_id, x, y, z, area_id
    FROM reference_positions
    """
    results = _execute_sqlite_read(query)

    if not results:
        return {}

    reference_positions = {}
    for row in results:
        device_id = row['device_id']
        reference_positions[device_id] = {
            'x': row['x'],
            'y': row['y'],
            'z': row['z'],
            'area_id': row['area_id']
        }

    return reference_positions

def save_device_position_to_sqlite(device_id: str, x: float, y: float, z: float = 0.0, area_id: Optional[str] = None) -> bool:
    """Save a device's reference position to the SQLite database.
    This replaces the old device_positions functionality with reference_positions.
    """
    query = """
    INSERT INTO reference_positions (device_id, x, y, z, area_id, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(device_id) DO UPDATE SET
        x=excluded.x,
        y=excluded.y,
        z=excluded.z,
        area_id=excluded.area_id,
        created_at=excluded.created_at
    """
    timestamp = datetime.now().isoformat()
    params = (device_id, x, y, z, area_id, timestamp)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.debug(f"Saved reference position for {device_id}: ({x}, {y}, {z}) in area {area_id}")
        return True
    else:
        logger.error(f"Failed to save reference position for {device_id}")
        return False