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

# Define model directory path
MODEL_DIR = Path("/data/models")

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

    # Existing methods...

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
