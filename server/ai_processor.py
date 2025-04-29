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
    save_ai_model_sqlite
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

        for room in rooms:
            room_id = room.get('id', '')
            area_id = room.get('area_id', '').lower()

            # Skip rooms without proper area_id
            if not area_id:
                logger.warning(f"Room {room_id} has no area_id, skipping object prediction")
                continue

            # Find common objects for this room type
            room_type = next((key for key in self.common_objects.keys() if key in area_id), 'default')
            possible_objects = self.common_objects.get(room_type, self.common_objects['default'])

            # Get room dimensions and bounds
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

            # Create list of all unique device IDs
            all_devices = set()
            for reading in distance_data:
                # Check for the correct key format based on database schema
                # The data should have tracked_device_id and scanner_id
                tracked_device_id = reading.get('tracked_device_id')
                scanner_id = reading.get('scanner_id')

                if tracked_device_id and scanner_id:
                    all_devices.add(tracked_device_id)
                    all_devices.add(scanner_id)
                # Fallback for alternative key names if needed
                elif 'device_id' in reading and 'other_id' in reading:
                    all_devices.add(reading['device_id'])
                    all_devices.add(reading['other_id'])
                else:
                    logger.warning(f"Unrecognized distance reading format: {reading}")

            # Sort them for consistent ordering
            device_list = sorted(list(all_devices))
            n_devices = len(device_list)

            if n_devices < 3:
                logger.error(f"Need at least 3 devices for relative positioning (found {n_devices})")
                return {}

            logger.info(f"Creating distance matrix for {n_devices} devices")

            # Create an empty distance matrix (fill with large values initially)
            max_distance = 50  # A large default distance
            distance_matrix = np.ones((n_devices, n_devices)) * max_distance

            # Fill in the diagonal with zeros (distance to self is 0)
            np.fill_diagonal(distance_matrix, 0)

            # Fill distance matrix with known measurements
            for reading in distance_data:
                try:
                    # Get the device IDs using the correct keys
                    if 'tracked_device_id' in reading and 'scanner_id' in reading:
                        device_a = reading['tracked_device_id']
                        device_b = reading['scanner_id']
                        distance = reading['distance']
                    elif 'device_id' in reading and 'other_id' in reading:
                        device_a = reading['device_id']
                        device_b = reading['other_id']
                        distance = reading['distance']
                    else:
                        continue  # Skip readings with unrecognized format

                    # Find indices for these devices
                    device_idx_a = device_list.index(device_a)
                    device_idx_b = device_list.index(device_b)

                    # Set the distance in both directions (symmetric matrix)
                    distance_matrix[device_idx_a, device_idx_b] = distance
                    distance_matrix[device_idx_b, device_idx_a] = distance
                except (ValueError, KeyError, TypeError) as e:
                    logger.warning(f"Error processing distance reading: {e}")
                    continue

            # Apply MDS to get relative positions
            mds = MDS(n_components=dimensions, dissimilarity='precomputed',
                      random_state=42, normalized_stress='auto')

            try:
                positions = mds.fit_transform(distance_matrix)
                logger.info("MDS calculation successful")
            except Exception as e:
                logger.error(f"MDS calculation failed: {e}")
                # Fallback to simpler approach
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

            logger.info(f"Relative positioning completed successfully for {len(result)} devices")
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

            # Organize distance data
            distance_matrix = {}
            devices = set()
            scanners = set()

            for record in distances:
                device_id = record['tracked_device_id']
                scanner_id = record['scanner_id']
                distance = record['distance']

                devices.add(device_id)
                scanners.add(scanner_id)

                if device_id not in distance_matrix:
                    distance_matrix[device_id] = {}

                distance_matrix[device_id][scanner_id] = distance

            # Prepare for MDS
            all_nodes = list(devices) + list(scanners)
            n_nodes = len(all_nodes)

            if n_nodes <= 1:
                logger.warning("Insufficient nodes for positioning (need at least 2)")
                return {}, {}

            # Create node index mapping
            node_indices = {node: i for i, node in enumerate(all_nodes)}

            # Create dissimilarity matrix for MDS
            dissimilarity = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if i == j:
                        continue

                    node_i = all_nodes[i]
                    node_j = all_nodes[j]

                    # If we have a direct measurement between these nodes
                    if node_i in distance_matrix and node_j in distance_matrix[node_i]:
                        dissimilarity[i, j] = distance_matrix[node_i][node_j]
                    elif node_j in distance_matrix and node_i in distance_matrix[node_j]:
                        dissimilarity[i, j] = distance_matrix[node_j][node_i]
                    else:
                        # If we don't have a direct measurement, use a large value
                        dissimilarity[i, j] = 1000.0

            # Fill in missing values using shortest path algorithm (Floyd-Warshall)
            for k in range(n_nodes):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if dissimilarity[i, j] > dissimilarity[i, k] + dissimilarity[k, j]:
                            dissimilarity[i, j] = dissimilarity[i, k] + dissimilarity[k, j]

            # Apply MDS
            mds_dimensions = self.config.get('generation_settings', {}).get('mds_dimensions', 2)
            if mds_dimensions > 3:
                mds_dimensions = 3  # Cap at 3D

            seed = 42  # For reproducibility
            mds = MDS(n_components=mds_dimensions, dissimilarity='precomputed', random_state=seed, n_init=10)
            positions = mds.fit_transform(dissimilarity)

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

    def generate_rooms_from_points(self, device_coords_by_area: Dict[str, List[Dict[str, Any]]], all_ha_areas: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Generate rooms based on device positions grouped by area.
        Now also includes all Home Assistant areas even if they don't have device positions.

        Args:
            device_coords_by_area: Dictionary mapping area_id to list of device coordinates
            all_ha_areas: Optional list of all Home Assistant areas

        Returns:
            List of room dictionaries
        """
        try:
            # Create a dictionary to map area_ids to area names from all_ha_areas
            area_id_to_name = {}
            if all_ha_areas:
                for area in all_ha_areas:
                    area_id = area.get('area_id')
                    if area_id:
                        area_id_to_name[area_id] = area.get('name', area_id)
                logger.info(f"Using {len(area_id_to_name)} area names from Home Assistant")

            # First, generate rooms based on device coordinates
            rooms = []
            floor_counter = 1  # Default all rooms to floor 1 initially

            logger.info(f"Generating rooms from points in {len(device_coords_by_area)} areas")

            # First pass - create rooms that have device positions
            for area_id, device_coords in device_coords_by_area.items():
                # Skip empty areas (shouldn't happen, but just in case)
                if not device_coords:
                    continue

                # Calculate room center based on device positions
                center_x = sum(d['x'] for d in device_coords) / len(device_coords)
                center_y = sum(d['y'] for d in device_coords) / len(device_coords)
                center_z = sum(d['z'] for d in device_coords) / len(device_coords)

                # Calculate room dimensions based on device spread
                x_values = [d['x'] for d in device_coords]
                y_values = [d['y'] for d in device_coords]
                z_values = [d['z'] for d in device_coords]

                min_x, max_x = min(x_values), max(x_values)
                min_y, max_y = min(y_values), max(y_values)

                # Set minimum room size to 2.5 meters for small rooms
                width = max(max_x - min_x + 2.0, 2.5)  # Add margin and enforce minimum size
                length = max(max_y - min_y + 2.0, 2.5)

                # Set room height to standard 2.4m
                height = 2.4

                # Get area name from mapping or use area_id if not found
                room_name = area_id_to_name.get(area_id, area_id)

                # Create room dictionary
                room = {
                    'id': str(uuid.uuid4()),
                    'name': room_name,
                    'type': self._predict_room_type(room_name),
                    'area_id': area_id,
                    'floor': floor_counter,
                    'center': {'x': center_x, 'y': center_y, 'z': center_z},
                    'dimensions': {'width': width, 'length': length, 'height': height},
                    'devices': [{'id': d['device_id'], 'x': d['x'], 'y': d['y'], 'z': d['z']} for d in device_coords]
                }

                rooms.append(room)
                logger.info(f"Generated room for {room_name} with {len(device_coords)} devices on floor {floor_counter}")

            # If we have Home Assistant areas, add empty rooms for areas without device positions
            added_rooms = 0
            if all_ha_areas:
                # Create a set of area_ids we've already processed
                processed_areas = set(device_coords_by_area.keys())

                # Get the average position of existing rooms to use as a reference
                avg_x, avg_y, avg_z = 0, 0, 0
                if rooms:
                    avg_x = sum(room['center']['x'] for room in rooms) / len(rooms)
                    avg_y = sum(room['center']['y'] for room in rooms) / len(rooms)
                    avg_z = sum(room['center']['z'] for room in rooms) / len(rooms)

                # Place rooms without devices in a spiral pattern around this center
                spiral_distance = 5.0  # Base distance between rooms (meters)
                angle_increment = 45  # Degrees between each room placement
                current_angle = 0
                current_distance = spiral_distance

                # Add rooms for areas that don't have device positions
                for area in all_ha_areas:
                    area_id = area.get('area_id')
                    if not area_id or area_id in processed_areas:
                        continue  # Skip already processed areas

                    area_name = area.get('name', area_id)

                    # Calculate position in a spiral pattern
                    angle_rad = math.radians(current_angle)
                    x = avg_x + current_distance * math.cos(angle_rad)
                    y = avg_y + current_distance * math.sin(angle_rad)

                    # Create a room with default dimensions
                    room = {
                        'id': str(uuid.uuid4()),
                        'name': area_name,
                        'type': self._predict_room_type(area_name),
                        'area_id': area_id,
                        'floor': floor_counter,
                        'center': {'x': x, 'y': y, 'z': avg_z},
                        'dimensions': {'width': 3.0, 'length': 3.0, 'height': 2.4},  # Default size
                        'devices': []  # No devices in this room yet
                    }

                    rooms.append(room)
                    added_rooms += 1

                    # Update spiral parameters for next room
                    current_angle = (current_angle + angle_increment) % 360
                    if current_angle < angle_increment:  # Completed a full circle
                        current_distance += spiral_distance  # Move outward for the next circle

                if added_rooms > 0:
                    logger.info(f"Added {added_rooms} additional rooms without devices from Home Assistant areas")

            logger.info(f"Successfully generated {len(rooms)} rooms")
            return rooms

        except Exception as e:
            logger.error(f"Error generating rooms from points: {str(e)}", exc_info=True)
            return []

    def generate_walls_between_rooms(self, rooms: List[Dict]) -> List[Dict]:
        """
        Generate walls between adjacent rooms.

        Parameters:
            rooms: List of room definitions

        Returns:
            List of wall definitions
        """
        logger.info(f"Generating walls between {len(rooms)} rooms")

        walls = []
        wall_id = 1

        # Group rooms by floor
        rooms_by_floor = {}
        for room in rooms:
            floor = room.get('floor', 0)
            if floor not in rooms_by_floor:
                rooms_by_floor[floor] = []
            rooms_by_floor[floor].append(room)

        # Process each floor separately
        for floor, floor_rooms in rooms_by_floor.items():
            # Skip floors with less than 2 rooms
            if len(floor_rooms) < 2:
                continue

            # For each room, check adjacency with other rooms on the same floor
            for i, room1 in enumerate(floor_rooms):
                for room2 in floor_rooms[i+1:]:
                    # Check if rooms are adjacent
                    if self._are_rooms_adjacent(room1, room2):
                        # Calculate wall segments between the rooms
                        new_walls = self._calculate_wall_segments(room1, room2, wall_id)
                        walls.extend(new_walls)
                        wall_id += len(new_walls)

        # Add external walls for each room
        for room in rooms:
            external_walls = self._generate_external_walls(room, wall_id)
            walls.extend(external_walls)
            wall_id += len(external_walls)

        logger.info(f"Generated {len(walls)} walls")
        return walls

    def _are_rooms_adjacent(self, room1: Dict, room2: Dict) -> bool:
        """Check if two rooms are adjacent to each other."""
        # Get room bounds
        r1_min_x = room1['bounds']['min']['x']
        r1_max_x = room1['bounds']['max']['x']
        r1_min_y = room1['bounds']['min']['y']
        r1_max_y = room1['bounds']['max']['y']

        r2_min_x = room2['bounds']['min']['x']
        r2_max_x = room2['bounds']['max']['x']
        r2_min_y = room2['bounds']['min']['y']
        r2_max_y = room2['bounds']['max']['y']

        # Check for x-overlap
        x_overlap = (r1_min_x <= r2_max_x and r1_max_x >= r2_min_x)

        # Check for y-overlap
        y_overlap = (r1_min_y <= r2_max_y and r1_max_y >= r2_min_y)

        # Rooms are adjacent if they overlap in one dimension and are close in the other
        max_gap = 0.1  # Maximum gap between rooms (10cm)

        # Adjacent along x-axis
        x_adjacent = y_overlap and (
            abs(r1_max_x - r2_min_x) <= max_gap or
            abs(r1_min_x - r2_max_x) <= max_gap
        )

        # Adjacent along y-axis
        y_adjacent = x_overlap and (
            abs(r1_max_y - r2_min_y) <= max_gap or
            abs(r1_min_y - r2_max_y) <= max_gap
        )

        return x_adjacent or y_adjacent

    def _calculate_wall_segments(self, room1: Dict, room2: Dict, start_id: int) -> List[Dict]:
        """Calculate wall segments between adjacent rooms."""
        # Basic implementation - just place a wall at the average position between rooms
        wall_height = 2.4  # Default wall height
        wall_thickness = 0.15  # Default wall thickness

        # Find the shared edge
        r1_min_x = room1['bounds']['min']['x']
        r1_max_x = room1['bounds']['max']['x']
        r1_min_y = room1['bounds']['min']['y']
        r1_max_y = room1['bounds']['max']['y']

        r2_min_x = room2['bounds']['min']['x']
        r2_max_x = room2['bounds']['max']['x']
        r2_min_y = room2['bounds']['min']['y']
        r2_max_y = room2['bounds']['max']['y']

        walls = []

        # Check for vertical wall (rooms adjacent horizontally)
        if abs(r1_max_x - r2_min_x) < 0.2 or abs(r1_min_x - r2_max_x) < 0.2:
            wall_x = (r1_max_x + r2_min_x) / 2 if r1_max_x < r2_max_x else (r1_min_x + r2_max_x) / 2

            # Find y-overlap
            start_y = max(r1_min_y, r2_min_y)
            end_y = min(r1_max_y, r2_max_y)

            walls.append({
                'id': f"wall_{start_id}",
                'start': {'x': wall_x, 'y': start_y},
                'end': {'x': wall_x, 'y': end_y},
                'thickness': wall_thickness,
                'height': wall_height
            })

        # Check for horizontal wall (rooms adjacent vertically)
        elif abs(r1_max_y - r2_min_y) < 0.2 or abs(r1_min_y - r2_max_y) < 0.2:
            wall_y = (r1_max_y + r2_min_y) / 2 if r1_max_y < r2_max_y else (r1_min_y + r2_max_y) / 2

            # Find x-overlap
            start_x = max(r1_min_x, r2_min_x)
            end_x = min(r1_max_x, r2_max_x)

            walls.append({
                'id': f"wall_{start_id}",
                'start': {'x': start_x, 'y': wall_y},
                'end': {'x': end_x, 'y': wall_y},
                'thickness': wall_thickness,
                'height': wall_height
            })

        return walls

    def _generate_external_walls(self, room: Dict, start_id: int) -> List[Dict]:
        """Generate external walls for a room."""
        min_x = room['bounds']['min']['x']
        max_x = room['bounds']['max']['x']
        min_y = room['bounds']['min']['y']
        max_y = room['bounds']['max']['y']

        wall_height = room.get('dimensions', {}).get('height', 2.4)
        wall_thickness = 0.15

        walls = [
            {
                'id': f"wall_{start_id}",
                'start': {'x': min_x, 'y': min_y},
                'end': {'x': max_x, 'y': min_y},
                'thickness': wall_thickness,
                'height': wall_height,
                'is_external': True
            },
            {
                'id': f"wall_{start_id + 1}",
                'start': {'x': max_x, 'y': min_y},
                'end': {'x': max_x, 'y': max_y},
                'thickness': wall_thickness,
                'height': wall_height,
                'is_external': True
            },
            {
                'id': f"wall_{start_id + 2}",
                'start': {'x': max_x, 'y': max_y},
                'end': {'x': min_x, 'y': max_y},
                'thickness': wall_thickness,
                'height': wall_height,
                'is_external': True
            },
            {
                'id': f"wall_{start_id + 3}",
                'start': {'x': min_x, 'y': max_y},
                'end': {'x': min_x, 'y': min_y},
                'thickness': wall_thickness,
                'height': wall_height,
                'is_external': True
            }
        ]

        return walls
