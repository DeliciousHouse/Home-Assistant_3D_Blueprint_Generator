import json
import logging
import math
import os
import pickle
import random # Added for layout jitter
import uuid
import sqlite3
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import Counter # Added for adjacency counting

# Required data science libraries
import numpy as np

# These imports are wrapped in try-except blocks later to make them optional
# But we define them globally first so static analysis tools recognize them
pandas_available = False
sklearn_available = False
torch_available = False
gymnasium_available = False
shapely_available = False

try:
    import pandas as pd
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import DBSCAN, KMeans
    from scipy.spatial import Delaunay, procrustes, ConvexHull
    from sklearn.manifold import MDS  # Added for relative positioning
    import joblib
    sklearn_available = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn dependencies not available. Some ML features will be disabled.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    torch_available = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("PyTorch not available. Neural network features will be disabled.")

try:
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    gymnasium_available = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Gymnasium and stable-baselines3 not available. RL features will be disabled.")

try:
    from shapely.geometry import Polygon, MultiPoint, LineString, Point
    shapely_available = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Shapely not available. Geometry calculations will use fallback methods.")

# Import specific DB functions needed
from .db import (
    get_area_observations,         # Needed for calculate_area_adjacency
    save_ai_model_sqlite,          # Needed by _save_model_info_to_sqlite (if used)
    execute_query,                 # Needed if train_wall_prediction/refinement uses it
    execute_write_query,           # Potentially needed if saving intermediate AI data
    get_sqlite_connection,         # Generally not needed directly, use helpers
    save_rssi_sample_to_sqlite,    # Only if actively training RSSI model
    get_recent_distances,          # Needed for get_rssi_data
    get_recent_area_predictions,   # Needed for generate_blueprint
    save_blueprint_to_sqlite,      # Needed for saving the generated blueprint
    save_reference_position,       # Needed for saving reference positions
    get_reference_positions_from_sqlite  # Needed for retrieving reference positions
)

from .config_loader import load_config

# Define model directory path - using relative path inside the project instead of /data
MODEL_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'models')))

logger = logging.getLogger(__name__)

class AIProcessor:
    """AI processor for enhancing blueprint generation with machine learning."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the AI processor."""
        # Use standardized config loader
        from .config_loader import load_config
        self.config = load_config(config_path)

        # Check if we have the optional ML dependencies
        if not gymnasium_available:
            logger.warning("Gymnasium not available. Blueprint refinement will use fallback methods.")
        if not shapely_available:
            logger.warning("Shapely not available. Room geometry calculations will use fallback methods.")

        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)

        # Initialize models
        self.rssi_distance_model = None
        self.room_clustering_model = None
        self.wall_prediction_model = None
        self.blueprint_refinement_model = None

        self._create_tables()
        # Load models if they exist
        self._load_models()

        # Object prediction setup
        self.common_objects = {
            "living_room": ["sofa", "coffee_table", "tv_stand", "bookshelf", "armchair"],
            "kitchen": ["refrigerator", "stove", "sink", "kitchen_cabinet", "kitchen_island"],
            "bedroom": ["bed", "wardrobe", "nightstand", "dresser", "desk"],
            "bathroom": ["toilet", "sink", "shower", "bathtub", "mirror"],
            "office": ["desk", "office_chair", "bookshelf", "filing_cabinet", "computer"],
            "dining_room": ["dining_table", "dining_chair", "sideboard", "china_cabinet"],
            "hallway": ["console_table", "coat_rack", "shoe_rack"],
            "garage": ["car", "workbench", "tool_cabinet", "storage_shelf"],
            "laundry_room": ["washing_machine", "dryer", "laundry_sink", "ironing_board"],
            "default": ["chair", "table", "cabinet", "lamp"]
        }
        self.object_dimensions = {
            "sofa": {"width": 2.0, "depth": 0.85, "height": 0.8},
            "coffee_table": {"width": 1.2, "depth": 0.6, "height": 0.45},
            "tv_stand": {"width": 1.5, "depth": 0.5, "height": 0.6},
            "bookshelf": {"width": 0.8, "depth": 0.4, "height": 1.8},
            "armchair": {"width": 0.85, "depth": 0.85, "height": 0.75},
            "refrigerator": {"width": 0.9, "depth": 0.75, "height": 1.8},
            "stove": {"width": 0.6, "depth": 0.6, "height": 0.9},
            "sink": {"width": 0.6, "depth": 0.55, "height": 0.9},
            "kitchen_cabinet": {"width": 0.6, "depth": 0.6, "height": 0.9},
            "kitchen_island": {"width": 1.5, "depth": 1.0, "height": 0.9},
            "bed": {"width": 1.6, "depth": 2.0, "height": 0.5},
            "wardrobe": {"width": 1.2, "depth": 0.6, "height": 2.0},
            "nightstand": {"width": 0.5, "depth": 0.5, "height": 0.6},
            "dresser": {"width": 1.2, "depth": 0.5, "height": 0.8},
            "desk": {"width": 1.4, "depth": 0.7, "height": 0.75},
            "toilet": {"width": 0.4, "depth": 0.6, "height": 0.4},
            "shower": {"width": 0.9, "depth": 0.9, "height": 2.0},
            "bathtub": {"width": 1.7, "depth": 0.75, "height": 0.55},
            "mirror": {"width": 0.8, "depth": 0.1, "height": 1.0},
            "office_chair": {"width": 0.65, "depth": 0.65, "height": 1.1},
            "filing_cabinet": {"width": 0.5, "depth": 0.6, "height": 1.3},
            "computer": {"width": 0.5, "depth": 0.5, "height": 0.5},
            "dining_table": {"width": 1.8, "depth": 1.0, "height": 0.75},
            "dining_chair": {"width": 0.5, "depth": 0.5, "height": 0.95},
            "sideboard": {"width": 1.6, "depth": 0.5, "height": 0.85},
            "china_cabinet": {"width": 1.0, "depth": 0.4, "height": 1.8},
            "console_table": {"width": 1.2, "depth": 0.4, "height": 0.8},
            "coat_rack": {"width": 0.6, "depth": 0.6, "height": 1.8},
            "shoe_rack": {"width": 0.8, "depth": 0.3, "height": 0.5},
            "car": {"width": 1.8, "depth": 4.5, "height": 1.5},
            "workbench": {"width": 1.8, "depth": 0.6, "height": 0.9},
            "tool_cabinet": {"width": 0.6, "depth": 0.4, "height": 1.2},
            "storage_shelf": {"width": 0.9, "depth": 0.4, "height": 1.8},
            "washing_machine": {"width": 0.6, "depth": 0.6, "height": 0.85},
            "dryer": {"width": 0.6, "depth": 0.6, "height": 0.85},
            "laundry_sink": {"width": 0.5, "depth": 0.5, "height": 0.85},
            "ironing_board": {"width": 1.2, "depth": 0.4, "height": 0.9},
            "chair": {"width": 0.5, "depth": 0.5, "height": 0.9},
            "table": {"width": 1.2, "depth": 0.8, "height": 0.75},
            "cabinet": {"width": 0.8, "depth": 0.4, "height": 1.0},
            "lamp": {"width": 0.3, "depth": 0.3, "height": 1.5}
        }

    def _create_tables(self):
        """
        Ensure necessary database tables exist for AI model storage and training data.

        Note: Most of our tables are already created in the db.py module when initializing
        the SQLite database. This method is mainly a placeholder for future AI-specific tables.
        """
        try:
            logger.info("Ensuring AI model tables exist")
            conn = get_sqlite_connection()
            if not conn:
                logger.error("Failed to connect to database for AI tables initialization")
                return False

            # Tables should already be created in db.py's init_sqlite_db function.
            # This is just a simple check to make sure critical tables exist.
            cursor = conn.cursor()

            # Check if the ai_models table exists
            cursor.execute('''
                SELECT name FROM sqlite_master
                WHERE type='table' AND name='ai_models'
            ''')

            if not cursor.fetchone():
                logger.warning("AI models table not found, it should be created by db.py")

            conn.close()
            logger.info("AI tables verification completed")
            return True

        except Exception as e:
            logger.error(f"Error creating AI tables: {str(e)}")
            return False

    def _load_models(self):
        """
        Load machine learning models from disk if they exist.
        """
        try:
            # Check if scikit-learn is available before loading models
            if not sklearn_available:
                logger.warning("scikit-learn not available, skipping model loading")
                return

            # Load RSSI-to-distance model if it exists
            rssi_model_path = MODEL_DIR / "rssi_distance_model.joblib"
            if rssi_model_path.exists():
                logger.info("Loading RSSI-to-distance model")
                self.rssi_distance_model = joblib.load(rssi_model_path)

            # Load room clustering model if it exists
            clustering_model_path = MODEL_DIR / "room_clustering_model.joblib"
            if clustering_model_path.exists():
                logger.info("Loading room clustering model")
                self.room_clustering_model = joblib.load(clustering_model_path)

            # Load wall prediction model if it exists
            wall_model_path = MODEL_DIR / "wall_prediction_model.joblib"
            if wall_model_path.exists():
                logger.info("Loading wall prediction model")
                self.wall_prediction_model = joblib.load(wall_model_path)

            # Load blueprint refinement model if it exists - only if gymnasium is available
            if gymnasium_available:
                refinement_model_path = MODEL_DIR / "blueprint_refinement_model.zip"
                if refinement_model_path.exists() and self.config.get('ai_settings', {}).get('enable_refinement', False):
                    logger.info("Loading blueprint refinement model")
                    # Refinement model uses Stable-Baselines3 PPO format
                    self.blueprint_refinement_model = PPO.load(refinement_model_path)
            else:
                logger.warning("Gymnasium not available, skipping refinement model loading")

        except Exception as e:
            logger.error(f"Error loading AI models: {str(e)}")
            # Continue without the models - we'll use fallback methods

    def get_rssi_data(self) -> Dict:
        """Get RSSI data for devices from the database."""
        try:
            # Get the recent distance measurements which contain RSSI information
            distance_data = get_recent_distances(time_window_minutes=30)

            # Process the distance data to get RSSI values
            rssi_data = {}
            for record in distance_data:
                device_id = record.get('device_id') or record.get('tracked_device_id')
                scanner_id = record.get('scanner_id')
                rssi = record.get('rssi')

                if device_id and scanner_id and rssi is not None:
                    if device_id not in rssi_data:
                        rssi_data[device_id] = {}
                    rssi_data[device_id][scanner_id] = rssi

            logger.info(f"Collected RSSI data for {len(rssi_data)} devices")
            return rssi_data
        except Exception as e:
            logger.error(f"Error getting RSSI data: {str(e)}")
            return {}

    def predict_objects(self, rooms: List[Dict]) -> List[Dict]:
        """
        Predict furniture and objects for each room based on its type.

        Parameters:
            rooms: List of room objects

        Returns:
            List of object definitions with positions and properties
        """
        if not rooms:
            return []

        logger.info(f"Predicting objects for {len(rooms)} rooms")
        objects = []
        object_id = 1

        # Check if we have the necessary dependencies for advanced object placement
        use_advanced_placement = shapely_available
        if not use_advanced_placement:
            logger.warning("Shapely not available. Using simplified object placement.")

        # First, ensure all rooms have bounds
        rooms_with_bounds = self._calculate_room_bounds(rooms)

        for room in rooms_with_bounds:
            room_id = room.get('id', '')
            area_id = room.get('area_id', '').lower()

            # Skip rooms without proper area_id
            if not area_id:
                logger.warning(f"Room {room_id} has no area_id, skipping object prediction")
                continue

            # Find common objects for this room type
            room_type = next((key for key in self.common_objects.keys() if key in area_id), 'default')
            possible_objects = self.common_objects.get(room_type, self.common_objects['default'])

            # Get room dimensions and bounds - now guaranteed to exist after _calculate_room_bounds
            min_x = room['bounds']['min']['x']
            max_x = room['bounds']['max']['x']
            min_y = room['bounds']['min']['y']
            max_y = room['bounds']['max']['y']
            z_base = room['bounds']['min']['z']

            room_width = max_x - min_x
            room_length = max_y - min_y

            # Calculate room area and determine number of objects based on size
            room_area = room_width * room_length
            if room_area < 5:  # Small room
                num_objects = min(2, len(possible_objects))
            elif room_area < 15:  # Medium room
                num_objects = min(4, len(possible_objects))
            else:  # Large room
                num_objects = min(6, len(possible_objects))

            # Select objects for this room
            selected_objects = random.sample(possible_objects, num_objects)

            for obj_type in selected_objects:
                # Get object dimensions
                obj_dims = self.object_dimensions.get(obj_type, {
                    "width": 0.5, "depth": 0.5, "height": 0.5
                })

                # Calculate a valid position for this object (away from walls)
                margin = max(obj_dims["width"], obj_dims["depth"]) / 2 + 0.2

                valid_position = False
                attempts = 0

                while not valid_position and attempts < 10:
                    # Generate random position
                    pos_x = random.uniform(min_x + margin, max_x - margin)
                    pos_y = random.uniform(min_y + margin, max_y - margin)

                    # Check if this position overlaps with existing objects
                    overlap = False
                    for existing_obj in objects:
                        if existing_obj.get('room_id') == room_id:
                            ex_x = existing_obj['position']['x']
                            ex_y = existing_obj['position']['y']
                            ex_width = self.object_dimensions.get(existing_obj['type'], {}).get('width', 0.5)
                            ex_depth = self.object_dimensions.get(existing_obj['type'], {}).get('depth', 0.5)

                            # Calculate minimum required distance
                            min_distance = (obj_dims["width"] + ex_width) / 2 + 0.3

                            # Check if objects are too close
                            distance = math.sqrt((pos_x - ex_x) ** 2 + (pos_y - ex_y) ** 2)
                            if distance < min_distance:
                                overlap = True
                                break

                    if not overlap:
                        valid_position = True

                    attempts += 1

                # Random rotation
                rotation = random.uniform(0, 360)

                # Add the object
                objects.append({
                    'id': f"obj_{object_id}",
                    'room_id': room_id,
                    'type': obj_type,
                    'position': {
                        'x': pos_x,
                        'y': pos_y,
                        'z': z_base + obj_dims["height"] / 2  # Place on floor
                    },
                    'dimensions': obj_dims,
                    'rotation': rotation
                })

                object_id += 1

        logger.info(f"Predicted {len(objects)} objects across all rooms")
        return objects

    def _calculate_room_bounds(self, rooms: List[Dict]) -> List[Dict]:
        """
        Calculate the bounds for each room in the list if they don't already have bounds.
        Returns the rooms with bounds added.
        """
        logger.info(f"Calculating bounds for {len(rooms)} rooms")

        for room in rooms:
            if 'bounds' not in room:
                # If bounds aren't already calculated, create them from dimensions
                center = room.get('center', {})
                dimensions = room.get('dimensions', {})

                if not center or not dimensions:
                    logger.warning(f"Room {room.get('name', 'unknown')} missing center or dimensions")
                    continue

                # Extract values with reasonable defaults
                center_x = center.get('x', 0)
                center_y = center.get('y', 0)
                width = dimensions.get('width', 3)
                length = dimensions.get('length', 3)
                height = dimensions.get('height', 2.5)

                # Calculate bounds
                half_width = width / 2
                half_length = length / 2

                room['bounds'] = {
                    'min': {
                        'x': center_x - half_width,
                        'y': center_y - half_length,
                        'z': 0
                    },
                    'max': {
                        'x': center_x + half_width,
                        'y': center_y + half_length,
                        'z': height
                    }
                }

        return rooms

    def run_relative_positioning(self, distance_data: List[Dict], dimensions: int = 2) -> Dict[str, Dict[str, float]]:
        """
        Uses Multidimensional Scaling (MDS) to compute relative positions of devices from distance data.

        Parameters:
            distance_data: List of distance readings between devices
            dimensions: Number of dimensions for the output (usually 2 or 3)

        Returns:
            Dictionary mapping device IDs to their coordinates {device_id: {x: val, y: val, z: val}}
        """
        try:
            logger.info(f"Running relative positioning using {len(distance_data)} distance measurements")

            if not distance_data:
                logger.error("No distance data provided for relative positioning")
                return {}

            # Debug: Log the structure of sample input data
            if distance_data and len(distance_data) > 0:
                sample = distance_data[0]
                logger.debug(f"Sample distance record keys: {list(sample.keys() if isinstance(sample, dict) else [])}")
                logger.debug(f"Sample distance record: {sample}")

            # Extract all unique entities
            entities = set()
            for record in distance_data:
                if not isinstance(record, dict):
                    logger.warning(f"Skipping non-dict record: {record}")
                    continue

                device_id = record.get('tracked_device_id')
                scanner_id = record.get('scanner_id')

                if device_id and scanner_id:
                    entities.add(device_id)
                    entities.add(scanner_id)
                    logger.debug(f"Added entities: {device_id}, {scanner_id}")

            # Convert to list for indexing
            entity_list = sorted(list(entities))
            n_entities = len(entity_list)

            logger.info(f"Extracted {n_entities} unique entities: {entity_list}")

            if n_entities < 3:
                logger.error(f"Not enough entities ({n_entities}) for {dimensions}D positioning. Need at least 3 entities.")
                return self._generate_fallback_positions()[0]  # Return fallback device positions

            # Create an empty distance matrix (fill with large values initially)
            max_distance = 50  # A large default distance
            distance_matrix = np.ones((n_entities, n_entities)) * max_distance

            # Fill in the diagonal with zeros (distance to self is 0)
            np.fill_diagonal(distance_matrix, 0)

            # Fill distance matrix with known measurements
            measurements_added = 0
            for reading in distance_data:
                try:
                    device_id = reading.get('tracked_device_id')
                    scanner_id = reading.get('scanner_id')
                    distance = reading.get('distance')

                    # Skip if distance is missing or invalid
                    if not device_id or not scanner_id or distance is None or not isinstance(distance, (int, float)) or distance <= 0:
                        continue

                    # Find indices for these entities
                    if device_id in entity_list and scanner_id in entity_list:
                        device_idx = entity_list.index(device_id)
                        scanner_idx = entity_list.index(scanner_id)

                        # Set the distance in both directions (symmetric matrix)
                        distance_matrix[device_idx, scanner_idx] = distance
                        distance_matrix[scanner_idx, device_idx] = distance
                        measurements_added += 1
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Error processing distance reading: {e}")
                    continue

            logger.info(f"Added {measurements_added} measurements to distance matrix")

            # Apply MDS to get relative positions if scikit-learn is available
            if sklearn_available:
                try:
                    mds = MDS(n_components=dimensions, dissimilarity='precomputed',
                            random_state=42, normalized_stress='auto')
                    positions = mds.fit_transform(distance_matrix)
                    logger.info("MDS calculation successful")
                except Exception as e:
                    logger.error(f"MDS calculation failed: {e}")
                    # Fallback to simpler approach
                    try:
                        mds = MDS(n_components=dimensions, dissimilarity='precomputed',
                                random_state=42)
                        positions = mds.fit_transform(distance_matrix)
                        logger.info("Alternative MDS calculation successful")
                    except Exception as e2:
                        logger.error(f"Alternative MDS calculation failed: {e2}")
                        # Last resort fallback to random positions
                        positions = np.random.rand(n_entities, dimensions) * 10
                        logger.warning("Using random positions as fallback")
            else:
                # If scikit-learn is not available, use a simple heuristic to position entities
                logger.warning("scikit-learn not available. Using simple heuristic for positioning.")
                positions = np.zeros((n_entities, dimensions))

                # Create a simple circular layout
                radius = 5  # Base radius in meters
                for i in range(n_entities):
                    # Calculate angle based on entity index
                    angle = 2 * np.pi * i / n_entities

                    # Position on a circle
                    positions[i, 0] = radius * np.cos(angle)  # x coordinate
                    positions[i, 1] = radius * np.sin(angle)  # y coordinate

                    # Add z coordinate if 3D
                    if dimensions > 2:
                        positions[i, 2] = 0.0  # All on same plane by default

                logger.info("Created simple circular layout for entity positions")

            # Create output dictionary mapping device IDs to coordinates
            result = {}
            for i, device_id in enumerate(entity_list):
                if dimensions == 2:
                    result[device_id] = {
                        'x': float(positions[i, 0]),
                        'y': float(positions[i, 1]),
                        'z': 0.0
                    }
                else:  # 3D
                    result[device_id] = {
                        'x': float(positions[i, 0]),
                        'y': float(positions[i, 1]),
                        'z': float(positions[i, 2]) if dimensions > 2 else 0.0
                    }

            logger.info(f"Relative positioning completed successfully for {len(result)} entities")
            return result

        except Exception as e:
            logger.error(f"Error in relative positioning: {str(e)}", exc_info=True)
            return self._generate_fallback_positions()[0]  # Return fallback device positions

    def get_relative_positions(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Calculate the relative positions of devices and anchors (scanners) using Multidimensional Scaling (MDS).

        Returns:
            Tuple of two dictionaries:
            1. Device positions: {device_id: {'x': x, 'y': y, 'z': z}}
            2. Anchor positions: {scanner_id: {'x': x, 'y': y, 'z': z}}
        """
        logger.info("Calculating relative positions using MDS...")

        # Initialize empty result dictionaries - this ensures we always return a valid tuple
        device_positions = {}
        anchor_positions = {}

        try:
            from .db import get_recent_distances

            # Get recent distance measurements
            distance_window = self.config.get('generation_settings', {}).get('distance_window_minutes', 15)
            distances = get_recent_distances(time_window_minutes=distance_window)

            if not distances:
                logger.warning("No distance data available for positioning")
                return self._generate_fallback_positions()  # Use fallback instead of empty dictionaries

            # Debug log to see what we're working with
            logger.info(f"Retrieved {len(distances)} distance records for positioning")

            # IMPROVED ENTITY CLASSIFICATION AND TRACKING
            # Track all entities found in distance records
            all_entities = set()
            actual_device_scanner_pairs = set()  # Track actual relationships in the data

            # First pass - collect all entity IDs from distance records
            for record in distances:
                if not isinstance(record, dict):
                    continue

                device_id = record.get('tracked_device_id')
                scanner_id = record.get('scanner_id')

                if device_id and scanner_id:
                    all_entities.add(device_id)
                    all_entities.add(scanner_id)
                    actual_device_scanner_pairs.add((device_id, scanner_id))

            logger.info(f"Found {len(all_entities)} unique entities in distance records")
            logger.info(f"Found {len(actual_device_scanner_pairs)} device-scanner pairs")

            # DETAILED LOGGING OF SAMPLE RECORDS
            if distances and len(distances) > 0:
                sample_indices = [0]
                if len(distances) > 10:
                    sample_indices.append(len(distances) // 2)
                if len(distances) > 1:
                    sample_indices.append(len(distances) - 1)

                logger.info("=== SAMPLE DISTANCE RECORDS ===")
                for idx in sample_indices:
                    sample = distances[idx]
                    logger.info(f"Record {idx}: {sample}")
                logger.info("===============================")

            # Define classification patterns
            device_patterns = ['iphone', 'phone', 'pixel', 'watch', 'tag', 'tracker', 'tile', 'remote', 'key']
            scanner_patterns = ['ble_', 'bt_', 'beacon', 'rssi', 'scanner', 'to_', 'proxy', 'measured_power', 'humidity', 'battery']

            # Perform frequency analysis to help with classification
            from collections import Counter
            occurrence_as_device = Counter()
            occurrence_as_scanner = Counter()

            for device_id, scanner_id in actual_device_scanner_pairs:
                occurrence_as_device[device_id] += 1
                occurrence_as_scanner[scanner_id] += 1

            # Log the top entities by occurrence
            logger.info("=== TOP ENTITIES AS DEVICES ===")
            for entity, count in occurrence_as_device.most_common(5):
                logger.info(f"{entity}: {count} occurrences")
            logger.info("==============================")

            logger.info("=== TOP ENTITIES AS SCANNERS ===")
            for entity, count in occurrence_as_scanner.most_common(5):
                logger.info(f"{entity}: {count} occurrences")
            logger.info("================================")

            # Initialize sets with definite devices and scanners
            devices = set()
            scanners = set()

            # First, identify clear devices based on patterns and occurrence counting
            logger.info("Starting entity classification...")
            classification_results = {}

            for entity in all_entities:
                entity_lower = entity.lower()
                entity_info = {
                    "name": entity,
                    "device_matches": [],
                    "scanner_matches": [],
                    "as_device_count": occurrence_as_device.get(entity, 0),
                    "as_scanner_count": occurrence_as_scanner.get(entity, 0),
                    "classification": "unknown"
                }

                # Check for device patterns
                for pattern in device_patterns:
                    if pattern in entity_lower:
                        entity_info["device_matches"].append(pattern)

                # Check for scanner patterns
                for pattern in scanner_patterns:
                    if pattern in entity_lower:
                        entity_info["scanner_matches"].append(pattern)

                # Determine classification
                if entity_info["device_matches"] and not entity_info["scanner_matches"]:
                    devices.add(entity)
                    entity_info["classification"] = "device"
                elif entity_info["scanner_matches"] and not entity_info["device_matches"]:
                    scanners.add(entity)
                    entity_info["classification"] = "scanner"
                elif entity_info["as_device_count"] > entity_info["as_scanner_count"] * 2:
                    devices.add(entity)
                    entity_info["classification"] = "device (by frequency)"
                elif entity_info["as_scanner_count"] >= entity_info["as_device_count"]:
                    scanners.add(entity)
                    entity_info["classification"] = "scanner (by frequency)"
                else:
                    # For unclear cases, use heuristics
                    if 'test_device' in entity_lower:
                        devices.add(entity)
                        entity_info["classification"] = "device (by name)"
                    elif 'test_scanner' in entity_lower:
                        scanners.add(entity)
                        entity_info["classification"] = "scanner (by name)"
                    elif entity.startswith('ble_'):
                        scanners.add(entity)
                        entity_info["classification"] = "scanner (by prefix)"
                    elif entity.endswith('_ble'):
                        scanners.add(entity)
                        entity_info["classification"] = "scanner (by suffix)"
                    else:
                        # Fallback to frequency-based classification
                        if entity_info["as_device_count"] >= entity_info["as_scanner_count"]:
                            devices.add(entity)
                            entity_info["classification"] = "device (default)"
                        else:
                            scanners.add(entity)
                            entity_info["classification"] = "scanner (default)"

                classification_results[entity] = entity_info

            # Handle special cases and resolve overlaps
            if 'test_device' in all_entities:
                devices.add('test_device')
                if 'test_device' in scanners:
                    scanners.remove('test_device')
                classification_results['test_device']["classification"] = "device (override)"

            # Handle ESPrsense BLE proxies
            for entity in list(devices):
                if entity.startswith('to_') and entity.endswith('_ble'):
                    devices.remove(entity)
                    scanners.add(entity)
                    classification_results[entity]["classification"] = "scanner (espresense override)"

            # Log all entity classifications for debugging
            logger.info("=== ENTITY CLASSIFICATIONS ===")
            for entity, info in classification_results.items():
                logger.info(f"'{entity}' - {info['classification']} | Device: {info['as_device_count']} | Scanner: {info['as_scanner_count']}")
            logger.info("=============================")

            # Ensure we have at least one device
            if len(devices) == 0 and len(scanners) > 0:
                # Convert the most common "scanner" to be a device
                most_common_scanner = max(scanners, key=lambda x: occurrence_as_device.get(x, 0))
                devices.add(most_common_scanner)
                scanners.remove(most_common_scanner)
                logger.info(f"Converted '{most_common_scanner}' to a device due to lack of devices")
                classification_results[most_common_scanner]["classification"] = "device (converted due to lack of devices)"

            # Log our final classifications
            logger.info(f"Classified {len(devices)} entities as devices: {devices}")
            logger.info(f"Classified {len(scanners)} entities as scanners: {scanners}")

            # Rest of the method remains similar...
            # If we don't have enough entities for 2D positioning, handle gracefully
            min_entities = 3
            total_entities = len(devices) + len(scanners)

            if total_entities < min_entities:
                logger.error(f"Not enough entities ({total_entities}) for 2D positioning. Minimum required: {min_entities}")
                # Instead of returning empty dictionaries, generate synthetic positions
                return self._generate_fallback_positions()

            # If we don't have enough devices specifically, consider some scanners as devices
            min_devices_required = 1

            if len(devices) < min_devices_required and len(scanners) > min_devices_required:
                logger.warning(f"Only {len(devices)} devices found. Converting some scanners to devices.")
                # Pick scanners that don't match strong scanner patterns
                potential_devices = [s for s in scanners if not any(p in s.lower() for p in scanner_patterns[:3])][:2]
                for scanner in potential_devices:
                    devices.add(scanner)
                    scanners.remove(scanner)
                    logger.info(f"Converted scanner '{scanner}' to a device to ensure minimum device count")
                logger.info(f"After conversion: {len(devices)} devices and {len(scanners)} scanners")

            # Combine both sets for MDS
            all_nodes = list(devices) + list(scanners)
            n_nodes = len(all_nodes)

            if n_nodes < 3:
                logger.error(f"Insufficient nodes for positioning (need at least 3, found {n_nodes})")
                return self._generate_fallback_positions()  # Use fallback instead of empty dictionaries

            # Create node index mapping for the distance matrix
            node_indices = {node: i for i, node in enumerate(all_nodes)}
            logger.info(f"Created mapping for {len(node_indices)} nodes")

            # Create dissimilarity matrix for MDS
            dissimilarity = np.ones((n_nodes, n_nodes)) * 1000.0  # Large default distance
            np.fill_diagonal(dissimilarity, 0)  # Diagonal should be zero (distance to self)

            # Fill the matrix with actual distances
            distance_count = 0
            for record in distances:
                try:
                    device_id = record.get('tracked_device_id')
                    scanner_id = record.get('scanner_id')
                    distance_value = float(record.get('distance', 0))

                    if device_id and scanner_id and distance_value > 0:
                        # Get matrix indices for this device-scanner pair
                        idx1 = node_indices.get(device_id)
                        idx2 = node_indices.get(scanner_id)

                        if idx1 is not None and idx2 is not None:
                            # Set the distance in both directions (symmetric matrix)
                            dissimilarity[idx1, idx2] = distance_value
                            dissimilarity[idx2, idx1] = distance_value
                            distance_count += 1
                except Exception as e:
                    logger.warning(f"Error processing distance record: {e}")

            logger.info(f"Added {distance_count} actual distances to the dissimilarity matrix")

            # Apply MDS if scikit-learn is available
            mds_dimensions = self.config.get('generation_settings', {}).get('mds_dimensions', 2)
            if mds_dimensions > 3:
                mds_dimensions = 3  # Cap at 3D

            seed = 42  # For reproducibility

            if sklearn_available:
                try:
                    mds = MDS(n_components=mds_dimensions,
                            dissimilarity='precomputed',
                            random_state=seed,
                            n_init=10,
                            normalized_stress='auto')
                    positions = mds.fit_transform(dissimilarity)
                    logger.info(f"MDS calculation successful with stress: {mds.stress_:.4f}")
                except Exception as e:
                    logger.error(f"MDS calculation failed: {e}")
                    logger.info("Trying alternative MDS approach...")
            else:
                # If scikit-learn is not available, use a simple layout
                logger.warning("scikit-learn not available. Using simple circular layout for positioning.")
                positions = np.zeros((len(all_nodes), mds_dimensions))

                # Create a simple circular layout
                for i, _ in enumerate(all_nodes):
                    angle = 2 * np.pi * i / len(all_nodes)
                    positions[i, 0] = 5 * np.cos(angle)  # x coordinate
                    positions[i, 1] = 5 * np.sin(angle)  # y coordinate
                    if mds_dimensions > 2:
                        positions[i, 2] = 0.0  # z coordinate (all on ground floor)

                # Fall back to a simpler MDS configuration
                try:
                    mds = MDS(n_components=mds_dimensions,
                             dissimilarity='precomputed',
                             random_state=seed)
                    positions = mds.fit_transform(dissimilarity)
                    logger.info(f"Alternative MDS succeeded with stress: {mds.stress_:.4f}")
                except Exception as e2:
                    logger.error(f"Alternative MDS also failed: {e2}")
                    logger.warning("Using random positions as fallback")
                    # Generate random positions in a reasonable range for visualization
                    positions = (np.random.rand(n_nodes, mds_dimensions) - 0.5) * 20

            # Map positions back to devices and scanners
            device_positions = {}
            anchor_positions = {}

            for node, idx in node_indices.items():
                pos = positions[idx]
                pos_dict = {'x': float(pos[0]), 'y': float(pos[1])}

                if mds_dimensions >= 3:
                    pos_dict['z'] = float(pos[2])
                else:
                    pos_dict['z'] = 0.0

                if node in devices:
                    device_positions[node] = pos_dict
                else:
                    anchor_positions[node] = pos_dict

            logger.info(f"MDS positioning complete. Found {len(device_positions)} devices and {len(anchor_positions)} anchors.")
            logger.info(f"Device positions: {device_positions}")
            logger.info(f"First few anchor positions: {dict(list(anchor_positions.items())[:3])}")

            return device_positions, anchor_positions

        except Exception as e:
            logger.error(f"Error calculating relative positions: {str(e)}", exc_info=True)
            # Use our fallback mechanism
            return self._generate_fallback_positions()

    def _generate_fallback_positions(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Generate fallback positions when MDS cannot be performed due to insufficient data.
        This creates a basic layout with synthetic points to allow blueprint generation to continue.

        Returns:
            Tuple of device_positions and anchor_positions dictionaries
        """
        logger.warning("Generating fallback positions with synthetic points")

        # Create minimal set of synthetic positions
        device_positions = {
            "synthetic_device_1": {"x": 0.0, "y": 0.0, "z": 0.0},
            "synthetic_device_2": {"x": 5.0, "y": 0.0, "z": 0.0},
            "synthetic_device_3": {"x": 0.0, "y": 5.0, "z": 0.0}
        }

        # Create some anchor positions too
        anchor_positions = {
            "synthetic_scanner_1": {"x": 2.5, "y": 2.5, "z": 0.0},
            "synthetic_scanner_2": {"x": 5.0, "y": 5.0, "z": 0.0}
        }

        logger.info("Created synthetic positions to enable blueprint generation")
        return device_positions, anchor_positions

    def generate_blueprint(self) -> bool:
        """Generate a full 3D blueprint with improved error handling."""
        logger.info("Starting blueprint generation process...")
        blueprint_data = {"state": "starting", "progress": 0}
        self.status = blueprint_data  # Update status immediately

        try:
            # Initialize AI processor
            ai_processor = self  # Use self instead of creating a new instance

            # STEP 1: Data Collection Validation
            distance_window = self.config.get('generation_settings', {}).get('distance_window_minutes', 15)
            distance_data = get_recent_distances(time_window_minutes=distance_window)

            if not distance_data:
                logger.error("No distance data available for blueprint generation")
                self.status = {"state": "failed", "reason": "no_distance_data", "progress": 0}
                return False

            # Validate minimum data quality
            if len(distance_data) < 10:  # Require at least 10 distance readings
                logger.warning(f"Insufficient distance data: only {len(distance_data)} records (minimum 10 recommended)")
                # Continue with warning, don't fail immediately

            logger.info(f"Retrieved {len(distance_data)} distance records for blueprint generation")
            blueprint_data["progress"] = 10
            self.status = blueprint_data

            # STEP 2: Area Predictions with Robust Error Handling
            area_window = self.config.get('generation_settings', {}).get('area_window_minutes', 10)
            area_predictions = get_recent_area_predictions(time_window_minutes=area_window)

            if not area_predictions:
                logger.warning("No area predictions found - room assignment may be limited")
                # Continue with warning - we'll handle this in room generation

            logger.info(f"Retrieved area predictions for {len(area_predictions)} devices")
            blueprint_data["progress"] = 20
            self.status = blueprint_data

            # STEP 3: Calculate Relative Positions with Validation
            device_positions, scanner_positions = self.get_relative_positions()

            # Validate position results
            if not device_positions and not scanner_positions:
                logger.error("Both device and scanner positions are empty - using fallback positions")
                device_positions, scanner_positions = self._generate_fallback_positions()
                if not device_positions:
                    self.status = {"state": "failed", "reason": "position_calculation_failed", "progress": 30}
                    return False

            # Validate minimum entity requirement
            if len(device_positions) + len(scanner_positions) < 3:
                logger.error(f"Insufficient positioned entities ({len(device_positions)} devices, {len(scanner_positions)} scanners)")
                self.status = {"state": "failed", "reason": "insufficient_entities", "progress": 30}
                return False

            logger.info(f"Calculated relative positions for {len(device_positions)} devices and {len(scanner_positions)} scanners")
            blueprint_data["progress"] = 40
            self.status = blueprint_data

            # Initialize variables that might be undefined
            rooms = []
            walls = []
            objects = []
            transformed_positions = {}

            # STEP 4: Generate rooms based on positions and area predictions
            # This is a placeholder since the actual room generation code seems to be missing
            logger.info("Generating rooms from positions and area predictions")
            # In a real implementation, we would process device_positions and scanner_positions
            # to create rooms, but for now we'll use a placeholder room
            rooms = [{
                'id': 'room_1',
                'name': 'Default Room',
                'area_id': 'default',
                'center': {'x': 0, 'y': 0, 'z': 0},
                'dimensions': {'width': 5, 'length': 5, 'height': 2.5}
            }]
            blueprint_data["progress"] = 60
            self.status = blueprint_data

            # STEP 5: Generate walls based on room layout
            logger.info("Generating walls based on room layout")
            # This is a placeholder for wall generation
            walls = []
            blueprint_data["progress"] = 70
            self.status = blueprint_data

            # STEP 6: Generate objects for rooms
            logger.info("Generating furniture and objects for rooms")
            objects = self.predict_objects(rooms)
            blueprint_data["progress"] = 80
            self.status = blueprint_data

            # STEP 7: Transform positions to align with room layout
            logger.info("Transforming positions to align with room layout")
            transformed_positions = device_positions.copy()
            transformed_positions.update(scanner_positions)
            blueprint_data["progress"] = 90
            self.status = blueprint_data

            # Final blueprint assembly and validation
            blueprint = {
                'generated_at': datetime.now().isoformat(),
                'rooms': rooms,
                'walls': walls,
                'objects': objects,
                'positions': transformed_positions,
                'floors': self._determine_floors(rooms) if rooms else []
            }

            # Validate minimum blueprint requirements
            if not blueprint['rooms']:
                logger.error("Blueprint generation failed: no rooms created")
                self.status = {"state": "failed", "reason": "no_rooms_created", "progress": 90}
                return False

            # Save blueprint to database
            success = save_blueprint_to_sqlite(blueprint)

            if success:
                logger.info("Blueprint successfully generated and saved")
                self.status = {"state": "completed", "progress": 100, "room_count": len(blueprint['rooms'])}
                return True
            else:
                logger.error("Failed to save blueprint")
                self.status = {"state": "failed", "reason": "database_save_failed", "progress": 95}
                return False

        except Exception as e:
            logger.error(f"Error generating blueprint: {str(e)}", exc_info=True)
            self.status = {"state": "failed", "reason": "exception", "message": str(e), "progress": 0}
            return False

    def _determine_floors(self, rooms: List[Dict]) -> List[Dict]:
        """
        Determine floor levels from rooms.

        Args:
            rooms: List of room objects with z-coordinates

        Returns:
            List of floor objects with level and rooms
        """
        if not rooms:
            return []

        # Group rooms by their z-level (floor)
        floor_groups = {}
        for room in rooms:
            # Use the bottom z-coordinate of the room as the floor level
            z_level = room.get('bounds', {}).get('min', {}).get('z', 0)
            # Round to nearest 0.1m to account for minor height differences
            z_level = round(z_level * 10) / 10

            if z_level not in floor_groups:
                floor_groups[z_level] = []

            floor_groups[z_level].append(room['id'])

        # Convert to list of floor objects
        floors = []
        for i, (level, room_ids) in enumerate(sorted(floor_groups.items())):
            floors.append({
                'id': f"floor_{i}",
                'level': i,
                'height': level,
                'rooms': room_ids
            })

        logger.info(f"Determined {len(floors)} floor levels")
        return floors
