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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import Delaunay, procrustes, ConvexHull
from sklearn.manifold import MDS  # Added for relative positioning
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from shapely.geometry import Polygon, MultiPoint  # Added for room geometry generation

# Import specific DB functions needed
from .db import (
    get_area_observations,  # Needed for calculate_area_adjacency
    save_ai_model_sqlite,   # Needed by _save_model_info_to_sqlite (if used)
    execute_query,          # Needed if train_wall_prediction/refinement uses it
    execute_write_query,     # Potentially needed if saving intermediate AI data
    get_sqlite_connection, # Generally not needed directly, use helpers
    save_rssi_sample_to_sqlite, # Only if actively training RSSI model
    save_ai_model_sqlite,
    get_recent_distances    # Needed for get_rssi_data
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

            # Load blueprint refinement model if it exists
            refinement_model_path = MODEL_DIR / "blueprint_refinement_model.zip"
            if refinement_model_path.exists() and self.config.get('ai_settings', {}).get('enable_refinement', False):
                logger.info("Loading blueprint refinement model")
                # Refinement model uses Stable-Baselines3 PPO format
                self.blueprint_refinement_model = PPO.load(refinement_model_path)

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

            # Debug: Log the structure of the input data
            if distance_data and len(distance_data) > 0:
                sample = distance_data[0]
                logger.debug(f"Sample distance record: {sample}")
                logger.debug(f"Sample record keys: {list(sample.keys() if isinstance(sample, dict) else [])}")

            # ENHANCED ENTITY DETECTION
            # Extract all unique entity IDs (devices and scanners)
            devices = set()
            scanners = set()
            all_entities = set()

            # First pass - collect all entity IDs from distance records
            for record in distance_data:
                if not isinstance(record, dict):
                    logger.warning(f"Skipping non-dict record: {record}")
                    continue

                # Extract device ID and scanner ID from the record
                device_id = record.get('tracked_device_id')
                scanner_id = record.get('scanner_id')

                # Skip records with missing IDs
                if not device_id or not scanner_id:
                    continue

                # Add to our device and scanner sets
                devices.add(device_id)
                scanners.add(scanner_id)
                all_entities.add(device_id)
                all_entities.add(scanner_id)

            # Filter out any empty strings or None values
            all_entities = {entity for entity in all_entities if entity}

            # Log what we found - good for debugging
            logger.info(f"Found {len(devices)} tracked devices: {devices}")
            logger.info(f"Found {len(scanners)} scanners: {scanners}")
            logger.info(f"Found {len(all_entities)} total unique entities")

            # Create a sorted list of all entities for MDS
            device_list = sorted(list(all_entities))
            n_devices = len(device_list)

            if n_devices < 3:
                logger.error(f"Not enough entities ({n_devices}) for {dimensions}D positioning. Need at least 3 entities.")
                return {}

            logger.info(f"Creating distance matrix for {n_devices} entities")

            # Create an empty distance matrix (fill with large values initially)
            max_distance = 50  # A large default distance
            distance_matrix = np.ones((n_devices, n_devices)) * max_distance

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

                    # Find indices for these devices
                    if device_id in device_list and scanner_id in device_list:
                        device_idx = device_list.index(device_id)
                        scanner_idx = device_list.index(scanner_id)

                        # Set the distance in both directions (symmetric matrix)
                        distance_matrix[device_idx, scanner_idx] = distance
                        distance_matrix[scanner_idx, device_idx] = distance
                        measurements_added += 1
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Error processing distance reading: {e}")
                    continue

            logger.info(f"Added {measurements_added} measurements to distance matrix")

            # Apply MDS to get relative positions
            mds = MDS(n_components=dimensions, dissimilarity='precomputed',
                      random_state=42, normalized_stress='auto')

            try:
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
                    positions = np.random.rand(n_devices, dimensions) * 10
                    logger.warning("Using random positions as fallback")

            # Create output dictionary mapping device IDs to coordinates
            result = {}
            for i, device_id in enumerate(device_list):
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
            return {}

    def get_relative_positions(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
        """
        Calculate the relative positions of devices and anchors (scanners) using Multidimensional Scaling (MDS).

        Returns:
            Tuple of two dictionaries:
            1. Device positions: {device_id: {'x': x, 'y': y, 'z': z}}
            2. Anchor positions: {scanner_id: {'x': x, 'y': y, 'z': z}}
        """
        logger.info("Calculating relative positions using MDS...")

        try:
            from .db import get_recent_distances

            # Get recent distance measurements
            distance_window = self.config.get('generation_settings', {}).get('distance_window_minutes', 15)
            distances = get_recent_distances(time_window_minutes=distance_window)

            if not distances:
                logger.warning("No distance data available for positioning")
                return {}, {}

            # Debug log to see what we're working with
            logger.info(f"Retrieved {len(distances)} distance records for positioning")
            if distances and len(distances) > 0:
                logger.debug(f"First distance record structure: {distances[0]}")

            # Extract all unique device and scanner IDs directly
            devices = set()
            scanners = set()
            device_scanner_pairs = set()  # To track unique pairs

            # IMPROVED CLASSIFICATION LOGIC
            # Device patterns - these are likely tracked physical devices (phones, watches, etc.)
            device_patterns = ['iphone', 'phone', 'pixel', 'watch', 'tag', 'tracker', 'tile']
            # Scanner patterns - these are likely fixed BLE scanners or sensors
            scanner_prefixes = ['ble_', 'bt_', 'beacon_', 'rssi_', 'scanner_', 'to_']
            scanner_suffixes = ['_ble', '_bt', '_beacon', '_scanner', '_proxy', '_rssi']
            sensor_suffixes = ['_battery', '_mac', '_humidity', '_rssi', '_measured_power', '_minor', '_major']

            # Patterns to detect non-user devices that are actually sensors or technical entities
            technical_patterns = ['ble_mac', 'ble_minor', 'ble_major', 'ble_measured_power', 'ble_humidity', 'ble_battery']

            # First pass - collect all entity IDs from distance records
            all_entities = set()
            for record in distances:
                if not isinstance(record, dict):
                    continue

                device_id = record.get('tracked_device_id')
                scanner_id = record.get('scanner_id')

                if device_id:
                    all_entities.add(device_id)
                if scanner_id:
                    all_entities.add(scanner_id)

            # Reference point handling - always treat these as devices with known positions
            reference_points = {entity for entity in all_entities if entity.startswith('reference_point_')}
            devices.update(reference_points)
            all_entities -= reference_points  # Remove from entities needing classification

            # Second pass - improved categorization of remaining entities
            for entity in all_entities:
                # Skip empty entity IDs
                if not entity:
                    continue

                entity_lower = entity.lower()

                # First check if it's explicitly a technical entity
                if any(pattern in entity_lower for pattern in technical_patterns):
                    scanners.add(entity)
                    continue

                # Is this likely a scanner/sensor?
                is_scanner = False

                # Check for scanner patterns
                if any(prefix in entity_lower for prefix in scanner_prefixes) or \
                   any(entity_lower.endswith(suffix) for suffix in scanner_suffixes) or \
                   any(entity_lower.endswith(suffix) for suffix in sensor_suffixes):
                    is_scanner = True

                # Check if this is explicitly a tracked device
                is_device = any(pattern in entity_lower for pattern in device_patterns)

                # Add to appropriate category
                if is_device and not is_scanner:
                    devices.add(entity)
                elif is_scanner:
                    scanners.add(entity)
                else:
                    # For ambiguous cases, check name length and format
                    # Longer, complex IDs with underscores are likely scanners
                    if len(entity) > 15 and '_' in entity:
                        scanners.add(entity)
                    else:
                        # Short names without technical patterns are likely user devices
                        devices.add(entity)

            # Special handling for test_device
            if 'test_device' in all_entities:
                devices.add('test_device')
                if 'test_device' in scanners:
                    scanners.remove('test_device')

            # Handle special case for ESPresense BLE proxies (they use format to_XXX_ble)
            for entity in list(devices):
                if entity.startswith('to_') and entity.endswith('_ble'):
                    devices.remove(entity)
                    scanners.add(entity)

            # Add test_scanner to scanners if present
            if 'test_scanner' in all_entities:
                scanners.add('test_scanner')
                if 'test_scanner' in devices:
                    devices.remove('test_scanner')

            # Build device-scanner pairs from distance measurements
            for record in distances:
                if not isinstance(record, dict):
                    continue

                device_id = record.get('tracked_device_id')
                scanner_id = record.get('scanner_id')

                if not device_id or not scanner_id:
                    continue

                # If our classification is good, we should find pairs where
                # one is in devices and one is in scanners
                if device_id in devices and scanner_id in scanners:
                    device_scanner_pairs.add((device_id, scanner_id))
                elif device_id in scanners and scanner_id in devices:
                    # Swap if they're incorrectly categorized
                    device_scanner_pairs.add((scanner_id, device_id))
                else:
                    # Ambiguous case - fallback to rule-based assignment
                    # For safety, we'll categorize both in our matrices
                    device_scanner_pairs.add((device_id, scanner_id))

            # Log what we found - good for debugging
            logger.info(f"Found {len(devices)} unique devices: {devices}")
            logger.info(f"Found {len(scanners)} unique scanners: {scanners}")
            logger.info(f"Found {len(device_scanner_pairs)} unique device-scanner pairs")

            # If we don't have enough entities for 2D positioning, handle gracefully
            min_entities = 3
            total_entities = len(devices) + len(scanners)

            if total_entities < min_entities:
                logger.error(f"Not enough entities ({total_entities}) for 2D positioning. Minimum required: {min_entities}")
                return {}, {}

            # If we don't have enough devices specifically, consider some scanners as devices
            min_devices_required = 1
            if len(devices) < min_devices_required and len(scanners) > min_devices_required:
                logger.warning(f"Only {len(devices)} devices found. Converting some scanners to devices.")
                # Pick scanners that don't match strong scanner patterns
                potential_devices = [s for s in scanners if not any(p in s.lower() for p in scanner_prefixes[:3])][:2]
                for scanner in potential_devices:
                    devices.add(scanner)
                    scanners.remove(scanner)
                logger.info(f"After conversion: {len(devices)} devices and {len(scanners)} scanners")

            # Combine both sets for MDS
            all_nodes = list(devices) + list(scanners)
            n_nodes = len(all_nodes)

            if n_nodes < 3:
                logger.error(f"Insufficient nodes for positioning (need at least 3, found {n_nodes})")
                return {}, {}

            # Create node index mapping for the distance matrix
            node_indices = {node: i for i, node in enumerate(all_nodes)}

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

            # Apply MDS
            mds_dimensions = self.config.get('generation_settings', {}).get('mds_dimensions', 2)
            if mds_dimensions > 3:
                mds_dimensions = 3  # Cap at 3D

            seed = 42  # For reproducibility

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
            return device_positions, anchor_positions

        except Exception as e:
            logger.error(f"Error calculating relative positions: {str(e)}", exc_info=True)
            return {}, {}
