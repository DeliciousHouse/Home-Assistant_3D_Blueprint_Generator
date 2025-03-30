import json
import logging
import sqlite3
import os
from typing import Any, Dict, List, Optional, Tuple
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
        logger = logging.getLogger(__name__) # Need logger if fallback used
        logger.warning("Could not import config_loader. Using empty config.")


logger = logging.getLogger(__name__)
app_config = load_config() # Load config once when module is loaded

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

        # --- Device Positions Table ---
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id TEXT NOT NULL,
                position_data TEXT NOT NULL, -- JSON: {'x':float, 'y':float, 'z':float, 'accuracy':float}
                source TEXT NOT NULL,
                accuracy REAL, -- Store separately for easier querying if needed
                area_id TEXT,
                timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP -- Store as ISO format string
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_positions_timestamp ON device_positions (timestamp DESC);')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_positions_device ON device_positions (device_id, timestamp DESC);');
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_device_positions_source ON device_positions (source);')

        # --- RSSI Distance Samples Table (for Training) ---
        # Schema matches parameters in save_rssi_sample_to_sqlite
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

def _execute_sqlite_read(query: str, params: Optional[Tuple] = None, fetch_one: bool = False) -> Optional[List[Dict[str, Any]] | Dict[str, Any]]:
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


def save_device_position_to_sqlite(device_id: str, position_data: Dict) -> bool:
    """Save a device's position in SQLite."""
    if not all(k in position_data for k in ['x', 'y', 'z']):
        logger.error(f"Attempted to save invalid position data for {device_id} (missing x, y, or z)")
        return False

    query = """
    INSERT INTO device_positions (device_id, position_data, source, accuracy, area_id, timestamp)
    VALUES (?, ?, ?, ?, ?, ?)
    """
    pos_json = json.dumps({ # Store essential coords + accuracy in JSON blob
        'x': position_data.get('x'),
        'y': position_data.get('y'),
        'z': position_data.get('z'),
        'accuracy': position_data.get('accuracy', 1.0)
    })
    source = position_data.get('source', 'unknown')
    accuracy = position_data.get('accuracy', 1.0)
    area_id = position_data.get('area_id')
    timestamp = datetime.now().isoformat()
    params = (device_id, pos_json, source, accuracy, area_id, timestamp)

    result = _execute_sqlite_write(query, params)
    if result is not None:
        logger.debug(f"Saved position for device {device_id} from source {source}")
        return True
    else:
        logger.error(f"Failed to save position for device {device_id}")
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
            blueprint = json.loads(result['data'])
            logger.info(f"Retrieved latest active blueprint from {result.get('created_at')}")
            return blueprint
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse latest blueprint from SQLite: {e}")
            return None
    logger.warning("No active blueprints found in SQLite database")
    return None


def get_reference_positions_from_sqlite() -> Dict[str, Dict]:
    """Get all unique, latest 'fixed_reference' positions from SQLite database."""
    query = """
    SELECT device_id, position_data
    FROM (
        SELECT
            device_id,
            position_data,
            ROW_NUMBER() OVER(PARTITION BY device_id ORDER BY timestamp DESC) as rn
        FROM device_positions
        WHERE source = 'fixed_reference'
    )
    WHERE rn = 1
    """
    results = _execute_sqlite_read(query)
    positions = {}
    if results:
        for row in results:
            device_id = row.get('device_id')
            pos_data_str = row.get('position_data')
            if device_id and pos_data_str:
                try:
                    pos_data = json.loads(pos_data_str)
                    # Ensure basic structure
                    if all(k in pos_data for k in ['x','y','z']):
                        positions[device_id] = pos_data
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse position_data for fixed reference {device_id}")

    logger.info(f"Loaded {len(positions)} unique fixed reference positions from SQLite")
    return positions

def get_device_positions_from_sqlite() -> Dict[str, Dict]:
    """Get the latest position for each device from the SQLite database."""
    # This query gets the most recent entry for each device_id
    query = """
    SELECT device_id, position_data, source, accuracy, area_id, timestamp
    FROM (
        SELECT
            *,
            ROW_NUMBER() OVER(PARTITION BY device_id ORDER BY timestamp DESC) as rn
        FROM device_positions
    )
    WHERE rn = 1
    ORDER BY timestamp DESC
    LIMIT 200 -- Limit number of devices returned if needed
    """
    results = _execute_sqlite_read(query)
    positions = {}
    if results:
        for row in results:
            device_id = row.get('device_id')
            pos_data_str = row.get('position_data')
            if device_id and pos_data_str:
                try:
                    pos_data = json.loads(pos_data_str)
                    # Combine stored data with other columns from the row
                    positions[device_id] = {
                        'x': float(pos_data['x']),
                        'y': float(pos_data['y']),
                        'z': float(pos_data['z']),
                        'accuracy': float(row.get('accuracy', pos_data.get('accuracy', 1.0))),
                        'source': row.get('source', 'unknown'),
                        'area_id': row.get('area_id'),
                        'timestamp': row.get('timestamp')
                    }
                except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                    logger.warning(f"Error parsing position data for device {device_id}: {e}")

    logger.info(f"Loaded latest positions for {len(positions)} devices from SQLite")
    return positions

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
    # Ensure model_path is a string
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

# For compatibility with existing code
execute_sqlite_query = _execute_sqlite_read
execute_query = _execute_sqlite_read
execute_write_query = _execute_sqlite_write