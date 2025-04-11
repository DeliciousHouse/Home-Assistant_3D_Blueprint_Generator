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

    def _create_tables(self) -> bool:
        """Create necessary database tables."""
        try:
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Check if tables already exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rssi_distance_samples';")
            table_exists = cursor.fetchone() is not None

            if table_exists:
                logger.info("AI database tables already exist, skipping creation")
                conn.close()
                return True

            # Create enhanced RSSI distance samples table with additional fields
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS rssi_distance_samples (
                id INTEGER PRIMARY KEY,
                device_id TEXT,
                sensor_id TEXT,
                rssi INTEGER,
                distance REAL,
                tx_power INTEGER,
                frequency REAL,
                environment_type TEXT,
                device_type TEXT,
                time_of_day INTEGER,
                day_of_week INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            # Create device positions table (for blueprint generator)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS device_positions (
                id INTEGER PRIMARY KEY,
                device_id TEXT,
                position_data TEXT,
                source TEXT,
                accuracy REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            conn.commit()
            conn.close()
            logger.info("AI database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            return False

    def _load_models(self) -> None:
        """Load trained models from disk."""
        try:
            # Load RSSI-to-distance model
            rssi_model_path = MODEL_DIR / 'rssi_distance_model.pkl'
            if rssi_model_path.exists():
                self.rssi_distance_model = joblib.load(rssi_model_path)
                logger.info("Loaded RSSI-to-distance model")

            # Load room clustering model
            room_model_path = MODEL_DIR / 'room_clustering_model.pkl'
            if room_model_path.exists():
                self.room_clustering_model = joblib.load(room_model_path)
                logger.info("Loaded room clustering model")

            # Load wall prediction model
            wall_model_path = MODEL_DIR / 'wall_prediction_model.pt'
            if wall_model_path.exists() and torch.cuda.is_available():
                self.wall_prediction_model = torch.load(wall_model_path)
                self.wall_prediction_model.eval()
                logger.info("Loaded wall prediction model")
            elif wall_model_path.exists():
                self.wall_prediction_model = torch.load(wall_model_path, map_location=torch.device('cpu'))
                self.wall_prediction_model.eval()
                logger.info("Loaded wall prediction model (CPU)")

            # Load blueprint refinement model
            bp_model_path = MODEL_DIR / 'blueprint_refinement_model.zip'
            if bp_model_path.exists():
                self.blueprint_refinement_model = PPO.load(bp_model_path)
                logger.info("Loaded blueprint refinement model")

        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")

    def get_models_status(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all AI models."""
        status = {
            "rssi_distance_model": {
                "loaded": self.rssi_distance_model is not None,
                "type": type(self.rssi_distance_model).__name__ if self.rssi_distance_model else None,
                "training_samples": self._get_training_sample_count("ai_rssi_distance_data")
            },
            "room_clustering_model": {
                "loaded": self.room_clustering_model is not None,
                "type": type(self.room_clustering_model).__name__ if self.room_clustering_model else None,
                "training_samples": self._get_training_sample_count("ai_room_clustering_data")
            },
            "wall_prediction_model": {
                "loaded": self.wall_prediction_model is not None,
                "type": "PyTorch Neural Network" if self.wall_prediction_model else None,
                "training_samples": self._get_training_sample_count("ai_wall_prediction_data")
            },
            "blueprint_refinement_model": {
                "loaded": self.blueprint_refinement_model is not None,
                "type": "Reinforcement Learning (PPO)" if self.blueprint_refinement_model else None,
                "feedback_samples": self._get_training_sample_count("ai_blueprint_feedback")
            }
        }
        return status

    def _get_training_sample_count(self, table_name: str) -> int:
        """Get the count of training samples for a model."""
        try:
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Check if table exists first
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if cursor.fetchone() is None:
                conn.close()
                return 0

            query = f"SELECT COUNT(*) FROM {table_name}"
            cursor.execute(query)
            result = cursor.fetchone()
            conn.close()

            return result[0] if result else 0
        except Exception as e:
            logger.error(f"Error getting sample count for {table_name}: {str(e)}")
            return 0

    # RSSI-to-Distance ML Model methods

    def save_rssi_distance_sample(self, device_id, sensor_id, rssi, distance, **kwargs):
        """Save RSSI-distance sample using the DB helper."""
        try:
            # Add these additional features if not passed in kwargs
            current_time = datetime.now()
            time_of_day = kwargs.get('time_of_day', current_time.hour)
            day_of_week = kwargs.get('day_of_week', current_time.weekday())

            # Extract device type from device_id if possible
            device_type = kwargs.get('device_type')
            if not device_type:
                if 'phone' in device_id.lower() or 'iphone' in device_id.lower():
                    device_type = 'smartphone'
                elif 'watch' in device_id.lower():
                    device_type = 'wearable'
                else:
                    device_type = 'unknown'

            return save_rssi_sample_to_sqlite(
                device_id=device_id,
                sensor_id=sensor_id, # Pass sensor_id
                rssi=rssi,
                distance=distance,
                tx_power=kwargs.get('tx_power'),
                frequency=kwargs.get('frequency'),
                environment_type=kwargs.get('environment_type'),
                device_type=device_type,
                time_of_day=time_of_day,
                day_of_week=day_of_week
            )
        except Exception as e:
            logger.error(f"Failed to save RSSI sample via helper: {e}")
            return False

    def train_models(self):
        """Train AI models with collected data."""
        try:
            # Get training data from database
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Get RSSI to distance training data
            cursor.execute("""
                SELECT rssi, distance, environment_type
                FROM rssi_distance_samples
                ORDER BY timestamp DESC
                LIMIT 10000
            """)
            rssi_distance_data = cursor.fetchall()
            conn.close()

            if not rssi_distance_data or len(rssi_distance_data) < 10:
                logger.info("Not enough training data for distance model")
                return False

            logger.info(f"Training models with {len(rssi_distance_data)} RSSI samples")

            # This is a placeholder - actual model training would happen here
            # In a real implementation, you'd train ML models with the data

            return True

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return False

    def train_rssi_distance_model(self, model_type: str = 'random_forest',
                                test_size: float = 0.2,
                                features: List[str] = None,
                                hyperparams: Dict = None) -> Dict:
        """Train the RSSI-to-distance regression model."""
        if features is None:
            features = ['rssi']
        if hyperparams is None:
            hyperparams = {}

        try:
            # Get training data from SQLite
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Build query based on available features
            query = """
            SELECT rssi, distance, tx_power, frequency, environment_type
            FROM rssi_distance_samples
            WHERE distance > 0
            """
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()

            if not results or len(results) < 10:
                return {
                    "success": False,
                    "error": "Not enough training data (minimum 10 samples required)"
                }

            # Prepare data
            data = []
            for row in results:
                sample = {'rssi': row[0], 'distance': row[1]}
                if 'tx_power' in features and row[2] is not None:
                    sample['tx_power'] = row[2]
                if 'frequency' in features and row[3] is not None:
                    sample['frequency'] = row[3]
                if 'environment_type' in features and row[4] is not None:
                    sample['environment_type'] = 1 if row[4] == 'indoor' else 0
                data.append(sample)

            df = pd.DataFrame(data)

            # Handle missing values
            for feature in features:
                if feature in df.columns and df[feature].isnull().any():
                    if feature == 'environment_type':
                        df[feature].fillna(0, inplace=True)
                    else:
                        df[feature].fillna(df[feature].mean(), inplace=True)

            # Prepare features and target
            X = df[[f for f in features if f in df.columns]]
            y = df['distance']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train model based on type
            if model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 10),
                    random_state=42
                )
            elif model_type == 'xgboost':
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=hyperparams.get('n_estimators', 100),
                    max_depth=hyperparams.get('max_depth', 6),
                    learning_rate=hyperparams.get('learning_rate', 0.1),
                    random_state=42
                )
            elif model_type == 'neural_network':
                from sklearn.neural_network import MLPRegressor
                model = MLPRegressor(
                    hidden_layer_sizes=hyperparams.get('hidden_layer_sizes', (100, 50)),
                    max_iter=hyperparams.get('max_iter', 500),
                    random_state=42
                )
            else:
                return {
                    "success": False,
                    "error": f"Unsupported model type: {model_type}"
                }

            # Train the model
            model.fit(X_train_scaled, y_train)

            # Evaluate the model
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Save the model
            model_data = {
                'model': model,
                'scaler': scaler,
                'features': features,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'r2': r2
                },
                'model_type': model_type,
                'hyperparams': hyperparams
            }

            model_path = MODEL_DIR / 'rssi_distance_model.pkl'
            joblib.dump(model_data, model_path)

            # Update the model in memory
            self.rssi_distance_model = model_data

            # Save model info to database
            metrics = {
                'mse': float(mse),
                'mae': float(mae),
                'r2': float(r2),
                'samples': len(df),
                'features': features,
                'hyperparams': hyperparams
            }

            self._save_model_info_to_sqlite('rssi_distance', model_type, str(model_path), metrics)

            logger.info(f"Trained RSSI-distance model: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

            return {
                "success": True,
                "metrics": {
                    "mse": float(mse),
                    "mae": float(mae),
                    "r2": float(r2)
                },
                "samples": len(df),
                "features": features,
                "model_type": model_type
            }

        except Exception as e:
            logger.error(f"Failed to train RSSI-distance model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def estimate_distance(self, rssi, tx_power=None, device_type=None, environment_type=None, use_imperial=True):
        """
        Estimate distance from RSSI using trained model or physics model with enhanced fallbacks.

        Args:
            rssi: The RSSI value (signal strength)
            tx_power: Transmission power in dBm (optional)
            device_type: Type of device ('smartphone', 'beacon', etc.) (optional)
            environment_type: Environment type ('indoor', 'outdoor') (optional)
            use_imperial: Whether to return distance in feet (True) or meters (False)

        Returns:
            Estimated distance in feet (if use_imperial=True) or meters (otherwise)
        """
        try:
            # First try ML model if available and has required features
            if self.rssi_distance_model is not None:
                try:
                    # Prepare features
                    features = {'rssi': float(rssi)}
                    model_data = self.rssi_distance_model

                    # Add additional features if they match what the model expects
                    if 'features' in model_data:
                        if 'tx_power' in model_data['features'] and tx_power is not None:
                            features['tx_power'] = float(tx_power)
                        if 'device_type' in model_data['features'] and device_type is not None:
                            # Simple encoding: map device types to integers
                            type_mapping = {'smartphone': 1, 'beacon': 2, 'wearable': 3, 'tag': 4}
                            features['device_type'] = type_mapping.get(device_type.lower(), 0)
                        if 'environment_type' in model_data['features'] and environment_type is not None:
                            # Simple encoding: 1 for indoor, 0 for outdoor
                            features['environment_type'] = 1 if environment_type.lower() == 'indoor' else 0

                    # Create feature vector
                    X = []
                    for feature in model_data['features']:
                        if feature in features:
                            X.append(features[feature])
                        else:
                            # If missing a required feature, fall back to physics model
                            logger.debug(f"Missing required feature: {feature}, falling back to physics model")
                            raise ValueError(f"Missing required feature: {feature}")

                    # Scale features if needed
                    if 'scaler' in model_data:
                        X = model_data['scaler'].transform([X])
                    else:
                        X = [X]

                    # Predict distance in meters
                    distance_meters = model_data['model'].predict(X)[0]

                    # Ensure reasonable result (in meters)
                    distance_meters = max(min(distance_meters, 30.0), 0.1)  # Clamp between 10cm and 30m

                    # Convert to feet if requested
                    if use_imperial:
                        from .unit_converter import meters_to_feet
                        distance_feet = meters_to_feet(distance_meters)
                        logger.debug(f"ML model estimated distance: {distance_feet:.2f}ft from RSSI {rssi}")
                        return distance_feet
                    else:
                        logger.debug(f"ML model estimated distance: {distance_meters:.2f}m from RSSI {rssi}")
                        return distance_meters

                except Exception as e:
                    logger.warning(f"ML distance estimation failed: {e}, falling back to physics model")
                    # Fall through to physics model

            # Physics-based calculation with enhanced protection
            # Ensure parameters are reasonable
            ref_power = getattr(self, 'reference_power', -65)  # Default reference RSSI at 1m
            path_loss = getattr(self, 'path_loss_exponent', 2.8)  # Default path loss exponent

            # Apply environment-specific adjustments
            if environment_type == 'indoor':
                # Indoor environments have higher path loss
                path_loss = max(path_loss, 3.0)
            elif environment_type == 'outdoor':
                # Outdoor environments have lower path loss
                path_loss = min(path_loss, 2.5)

            # If tx_power is provided, use it to adjust the reference power
            if tx_power is not None:
                try:
                    # Adjust reference power based on tx_power
                    # Reference power is typically measured at 1m with default tx_power
                    ref_power_adjustment = tx_power - (-70)  # Assuming -70 is default tx_power
                    ref_power += ref_power_adjustment
                except (ValueError, TypeError):
                    # Ignore if tx_power isn't a valid number
                    pass

            # Safe RSSI clamping
            try:
                safe_rssi = max(min(float(rssi), -20), -100)
            except (ValueError, TypeError):
                logger.warning(f"Invalid RSSI value: {rssi}, using default")
                safe_rssi = -65

            # Quick checks for extreme RSSI values
            if safe_rssi > -35:  # Very close
                distance_meters = 0.3  # 30cm
            elif safe_rssi < -95:  # Very far
                distance_meters = 25.0  # 25m
            else:
                try:
                    # Calculate with solid overflow protection
                    exponent = (ref_power - safe_rssi) / (10 * path_loss)

                    # Safe calculation
                    distance_meters = 10 ** exponent

                    # Ensure reasonable result
                    distance_meters = min(max(distance_meters, 0.1), 30.0)  # Between 10cm and 30m
                except Exception as e:
                    logger.warning(f"Physics-based distance calculation failed: {str(e)}")

                    # Map RSSI to distance ranges as ultimate fallback (in meters)
                    if safe_rssi > -50: distance_meters = 1.0
                    elif safe_rssi > -65: distance_meters = 3.0
                    elif safe_rssi > -75: distance_meters = 5.0
                    elif safe_rssi > -85: distance_meters = 10.0
                    else: distance_meters = 15.0

            # Convert to feet if requested
            if use_imperial:
                from .unit_converter import meters_to_feet
                distance_feet = meters_to_feet(distance_meters)
                logger.debug(f"Physics model estimated distance: {distance_feet:.2f}ft from RSSI {rssi}")
                return distance_feet
            else:
                logger.debug(f"Physics model estimated distance: {distance_meters:.2f}m from RSSI {rssi}")
                return distance_meters

        except Exception as e:
            logger.error(f"Error in distance estimation: {e}", exc_info=True)
            # Default reasonable distance - convert to feet if requested
            default_meters = 5.0
            if use_imperial:
                from .unit_converter import meters_to_feet
                return meters_to_feet(default_meters)  # ~16.4 feet
            else:
                return default_meters  # 5 meters

    def calibrate_rssi_reference_values(self):
        """Dynamically calibrate RSSI reference values based on collected data."""
        try:
            # Get RSSI at known 1-meter distances
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()
            cursor.execute('''
            SELECT rssi FROM rssi_distance_samples
            WHERE distance BETWEEN 0.8 AND 1.2  -- Around 1 meter
            AND timestamp > datetime('now', '-7 days')
            ''')
            results = cursor.fetchall()

            if results and len(results) >= 5:
                # Calculate median RSSI at 1m
                rssi_values = [row[0] for row in results]
                from statistics import median
                reference_power = median(rssi_values)

                # Update the configuration
                logger.info(f"Calibrated reference power to {reference_power} dBm based on {len(results)} samples")

                # Also calculate path loss exponent from varying distances
                cursor.execute('''
                SELECT rssi, distance FROM rssi_distance_samples
                WHERE distance > 0.5 AND distance < 10
                AND timestamp > datetime('now', '-7 days')
                ''')
                dist_results = cursor.fetchall()

                if dist_results and len(dist_results) >= 20:
                    # Calculate path loss exponent using linear regression
                    x_vals = []  # 10*log10(distance)
                    y_vals = []  # RSSI values

                    for rssi, distance in dist_results:
                        if distance > 0:
                            x_vals.append(10 * math.log10(distance))
                            y_vals.append(rssi)

                    # Simple linear regression
                    x_mean = sum(x_vals) / len(x_vals)
                    y_mean = sum(y_vals) / len(y_vals)

                    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
                    denominator = sum((x - x_mean) ** 2 for x in x_vals)

                    if denominator != 0:
                        slope = numerator / denominator
                        path_loss_exponent = -slope / 10

                        # Update path loss exponent if reasonable
                        if 1.5 <= path_loss_exponent <= 6.0:
                            logger.info(f"Calibrated path loss exponent to {path_loss_exponent}")

            conn.close()
            return True
        except Exception as e:
            logger.error(f"Error calibrating RSSI values: {e}")
            return False

    def train_models_with_hyperparameter_tuning(self):
        """Train models with advanced hyperparameter tuning."""
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.model_selection import GridSearchCV
            from sklearn.preprocessing import StandardScaler
            import pickle

            # Get training data
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Get available fields from the table
            cursor.execute("PRAGMA table_info(rssi_distance_samples)")
            table_info = cursor.fetchall()
            column_names = [col[1] for col in table_info]

            # Build query based on available columns
            select_fields = ["rssi", "distance"]
            if "tx_power" in column_names: select_fields.append("tx_power")
            if "environment_type" in column_names: select_fields.append("environment_type")
            if "device_type" in column_names: select_fields.append("device_type")
            if "time_of_day" in column_names: select_fields.append("time_of_day")

            query = f"SELECT {', '.join(select_fields)} FROM rssi_distance_samples WHERE timestamp > datetime('now', '-30 days')"
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()

            if len(results) < 50:
                logger.info(f"Not enough training data for model tuning: {len(results)} samples")
                return False

            # Prepare data
            X = []
            y = []

            for row in results:
                rssi, distance = row[0], row[1]

                # Skip invalid data
                if rssi is None or distance is None or distance <= 0:
                    continue

                # Create feature vector
                features = [rssi]

                # Add available additional features
                for i in range(2, len(row)):
                    if row[i] is not None:
                        # Convert categorical to numeric if needed
                        if isinstance(row[i], str):
                            features.append(hash(row[i]) % 10)
                        else:
                            features.append(float(row[i]))

                X.append(features)
                y.append(distance)

            # Train a simple model if we have enough data
            if len(X) >= 50:
                # Standardize features
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

                # Define a simple model with limited parameters
                model = RandomForestRegressor(n_estimators=100)
                model.fit(X, y)

                # Save the model
                with open('/data/distance_estimation_model.pkl', 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'scaler': scaler,
                        'features': len(X[0])
                    }, f)

                logger.info(f"Trained distance estimation model with {len(X)} samples")
                return True

            return False
        except Exception as e:
            logger.error(f"Error during model training with hyperparameters: {e}")
            return False

    # Room Clustering methods

    def configure_room_clustering(self, algorithm: str = 'dbscan',
                                 eps: float = 2.0, min_samples: int = 3,
                                 features: List[str] = None,
                                 temporal_weight: float = 0.2) -> Dict:
        """Configure the room clustering model."""
        if features is None:
            features = ['x', 'y', 'z']

        try:
            # Create clustering model based on algorithm
            if algorithm == 'dbscan':
                model = DBSCAN(eps=eps, min_samples=min_samples)
            elif algorithm == 'kmeans':
                model = KMeans(n_clusters=min_samples, random_state=42)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported clustering algorithm: {algorithm}"
                }

            # Save model configuration
            model_data = {
                'model': model,
                'algorithm': algorithm,
                'params': {
                    'eps': eps,
                    'min_samples': min_samples
                },
                'features': features,
                'temporal_weight': temporal_weight
            }

            model_path = MODEL_DIR / 'room_clustering_model.pkl'
            joblib.dump(model_data, model_path)

            # Update the model in memory
            self.room_clustering_model = model_data

            # Save model info to database using the helper function
            metrics = {
                'algorithm': algorithm,
                'eps': eps,
                'min_samples': min_samples,
                'features': features,
                'temporal_weight': temporal_weight
            }

            self._save_model_info_to_sqlite('room_clustering', algorithm, str(model_path), metrics)

            logger.info(f"Configured room clustering model: {algorithm}")

            return {
                "success": True,
                "algorithm": algorithm,
                "params": {
                    "eps": eps,
                    "min_samples": min_samples
                },
                "features": features,
                "temporal_weight": temporal_weight
            }

        except Exception as e:
            logger.error(f"Failed to configure room clustering model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def detect_rooms_ml(self, positions: Dict[str, Dict[str, float]]) -> List[Dict]:
        """Detect rooms using ML-based clustering."""
        if not positions or not self.room_clustering_model:
            return []

        try:
            # Extract position coordinates
            device_ids = list(positions.keys())
            coords = np.array([[p['x'], p['y'], p['z']] for p in positions.values()])

            # Apply clustering
            model = self.room_clustering_model['model']
            labels = model.fit_predict(coords)

            # Group positions by cluster
            rooms = []
            for label in set(labels):
                if label == -1:  # Skip noise points
                    continue

                # Get positions in this cluster
                room_positions = coords[labels == label]
                room_device_ids = [device_ids[i] for i, l in enumerate(labels) if l == label]

                # Calculate room properties
                min_coords = np.min(room_positions, axis=0)
                max_coords = np.max(room_positions, axis=0)
                center = np.mean(room_positions, axis=0)

                # Create room object
                room = {
                    'id': f"room_{label}",
                    'name': f"Room {label + 1}",
                    'center': {
                        'x': float(center[0]),
                        'y': float(center[1]),
                        'z': float(center[2])
                    },
                    'dimensions': {
                        'width': float(max_coords[0] - min_coords[0]),
                        'length': float(max_coords[1] - min_coords[1]),
                        'height': float(max_coords[2] - min_coords[2])
                    },
                    'bounds': {
                        'min': {
                            'x': float(min_coords[0]),
                            'y': float(min_coords[1]),
                            'z': float(min_coords[2])
                        },
                        'max': {
                            'x': float(max_coords[0]),
                            'y': float(max_coords[1]),
                            'z': float(max_coords[2])
                        }
                    },
                    'devices': room_device_ids
                }

                rooms.append(room)

            return rooms

        except Exception as e:
            logger.error(f"Error in ML room detection: {str(e)}")
            return []

    # Wall Prediction Neural Network methods

    class WallPredictionDataset(Dataset):
        """Dataset for wall prediction model."""

        def __init__(self, positions_data, walls_data, transform=None):
            self.positions = positions_data
            self.walls = walls_data
            self.transform = transform

        def __len__(self):
            return len(self.positions)

        def __getitem__(self, idx):
            positions = self.positions[idx]
            walls = self.walls[idx]

            # Convert to tensors
            positions_tensor = torch.tensor(positions, dtype=torch.float32)
            walls_tensor = torch.tensor(walls, dtype=torch.float32)

            if self.transform:
                positions_tensor = self.transform(positions_tensor)

            return positions_tensor, walls_tensor

    class WallPredictionCNN(nn.Module):
        """CNN model for wall prediction."""

        def __init__(self, input_channels=1, output_channels=1):
            super().__init__()

            # Encoder
            self.encoder = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU()
            )

            # Decoder
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(32, output_channels, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    def train_wall_prediction_model(self, model_type: str = 'cnn',
                                   training_data: List = None,
                                   epochs: int = 50,
                                   batch_size: int = 32,
                                   learning_rate: float = 0.001) -> Dict:
        """Train the wall prediction neural network."""
        try:
            # Get training data from database if not provided
            if not training_data:
                query = """
                SELECT positions_data, walls_data FROM ai_wall_prediction_data
                """
                results = execute_query(query)

                if not results or len(results) < 5:
                    return {
                        "success": False,
                        "error": "Not enough training data (minimum 5 samples required)"
                    }

                positions_data = [json.loads(row[0]) for row in results]
                walls_data = [json.loads(row[1]) for row in results]
            else:
                positions_data = [item['positions'] for item in training_data]
                walls_data = [item['walls'] for item in training_data]

            # Create dataset
            dataset = self.WallPredictionDataset(positions_data, walls_data)

            # Split into train/validation
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = self.WallPredictionCNN().to(device)

            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            # Training loop
            best_val_loss = float('inf')
            metrics = {'train_loss': [], 'val_loss': []}

            for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0

                for positions, walls in train_loader:
                    positions, walls = positions.to(device), walls.to(device)

                    optimizer.zero_grad()
                    outputs = model(positions)
                    loss = criterion(outputs, walls)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * positions.size(0)

                train_loss /= len(train_loader.dataset)
                metrics['train_loss'].append(train_loss)

                # Validation
                model.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for positions, walls in val_loader:
                        positions, walls = positions.to(device), walls.to(device)
                        outputs = model(positions)
                        loss = criterion(outputs, walls)
                        val_loss += loss.item() * positions.size(0)

                val_loss /= len(val_loader.dataset)
                metrics['val_loss'].append(val_loss)

                logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    model_path = MODEL_DIR / 'wall_prediction_model.pt'
                    torch.save(model, model_path)

            # Load the best model
            model_path = MODEL_DIR / 'wall_prediction_model.pt'
            self.wall_prediction_model = torch.load(model_path)

            # Save model info using the helper function
            metrics = {
                'model_type': model_type,
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'final_train_loss': metrics['train_loss'][-1],
                'final_val_loss': metrics['val_loss'][-1],
                'best_val_loss': best_val_loss,
                'samples': len(dataset)
            }

            self._save_model_info_to_sqlite('wall_prediction', model_type, str(model_path), metrics)

            logger.info(f"Trained wall prediction model: best_val_loss={best_val_loss:.4f}")

            return {
                "success": True,
                "metrics": {
                    "train_loss": metrics['train_loss'][-1],
                    "val_loss": metrics['val_loss'][-1],
                    "best_val_loss": best_val_loss
                },
                "samples": len(dataset),
                "model_type": model_type
            }

        except Exception as e:
            logger.error(f"Failed to train wall prediction model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def predict_walls(self, positions: Dict[str, Dict[str, float]], rooms: List[Dict]) -> List[Dict]:
        """Predict walls using the trained neural network."""
        if not positions or not rooms or not self.wall_prediction_model:
            return []

        try:
            # Convert positions to grid representation
            grid_size = 64
            grid = np.zeros((grid_size, grid_size), dtype=np.float32)

            # Find bounds of the space
            all_coords = np.array([[p['x'], p['y']] for p in positions.values()])
            min_x, min_y = np.min(all_coords, axis=0)
            max_x, max_y = np.max(all_coords, axis=0)

            # Add padding
            padding = 0.1 * max(max_x - min_x, max_y - min_y)
            min_x -= padding
            min_y -= padding
            max_x += padding
            max_y += padding

            # Scale function to convert real coordinates to grid indices
            def scale_to_grid(x, y):
                grid_x = int((x - min_x) / (max_x - min_x) * (grid_size - 1))
                grid_y = int((y - min_y) / (max_y - min_y) * (grid_size - 1))
                return max(0, min(grid_size - 1, grid_x)), max(0, min(grid_size - 1, grid_y))

            # Place positions on grid
            for pos in positions.values():
                grid_x, grid_y = scale_to_grid(pos['x'], pos['y'])
                grid[grid_y, grid_x] = 1.0

            # Add room boundaries to grid
            for room in rooms:
                bounds = room['bounds']
                min_room_x, min_room_y = scale_to_grid(bounds['min']['x'], bounds['min']['y'])
                max_room_x, max_room_y = scale_to_grid(bounds['max']['x'], bounds['max']['y'])

                # Draw room boundaries on grid
                for x in range(min_room_x, max_room_x + 1):
                    grid[min_room_y, x] = 0.5
                    grid[max_room_y, x] = 0.5

                for y in range(min_room_y, max_room_y + 1):
                    grid[y, min_room_x] = 0.5
                    grid[y, max_room_x] = 0.5

            # Prepare input for model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Get prediction
            self.wall_prediction_model.eval()
            with torch.no_grad():
                output = self.wall_prediction_model(input_tensor)
                wall_grid = output.squeeze().cpu().numpy()

            # Convert wall grid to wall segments
            walls = []
            threshold = 0.5

            # Function to convert grid indices back to real coordinates
            def grid_to_real(grid_x, grid_y):
                x = min_x + grid_x / (grid_size - 1) * (max_x - min_x)
                y = min_y + grid_y / (grid_size - 1) * (max_y - min_y)
                return x, y

            # Detect horizontal walls
            for y in range(grid_size):
                wall_start = None
                for x in range(grid_size):
                    if wall_grid[y, x] > threshold:
                        if wall_start is None:
                            wall_start = x
                    elif wall_start is not None:
                        if x - wall_start > 2:  # Minimum wall length
                            start_x, start_y = grid_to_real(wall_start, y)
                            end_x, end_y = grid_to_real(x - 1, y)

                            # Calculate wall properties
                            length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                            angle = np.arctan2(end_y - start_y, end_x - start_x)

                            walls.append({
                                'start': {'x': float(start_x), 'y': float(start_y)},
                                'end': {'x': float(end_x), 'y': float(end_y)},
                                'thickness': 0.1,
                                'height': 2.4,
                                'angle': float(angle),
                                'confidence': float(np.mean(wall_grid[y, wall_start:x]))
                            })
                        wall_start = None

            # Detect vertical walls
            for x in range(grid_size):
                wall_start = None
                for y in range(grid_size):
                    if wall_grid[y, x] > threshold:
                        if wall_start is None:
                            wall_start = y
                    elif wall_start is not None:
                        if y - wall_start > 2:  # Minimum wall length
                            start_x, start_y = grid_to_real(x, wall_start)
                            end_x, end_y = grid_to_real(x, y - 1)

                            # Calculate wall properties
                            length = np.sqrt((end_x - start_x)**2 + (end_y - start_y)**2)
                            angle = np.arctan2(end_y - start_y, end_x - start_x)

                            walls.append({
                                'start': {'x': float(start_x), 'y': float(start_y)},
                                'end': {'x': float(end_x), 'y': float(end_y)},
                                'thickness': 0.1,
                                'height': 2.4,
                                'angle': float(angle),
                                'confidence': float(np.mean(wall_grid[wall_start:y, x]))
                            })
                        wall_start = None

            return walls

        except Exception as e:
            logger.error(f"Error in wall prediction: {str(e)}")
            return []

    # Blueprint Refinement methods

    def train_blueprint_refinement_model(self, feedback_data: List = None,
                                        reward_weights: Dict = None,
                                        learning_rate: float = 0.01,
                                        discount_factor: float = 0.9) -> Dict:
        """Train the blueprint refinement model using reinforcement learning."""
        if reward_weights is None:
            reward_weights = {
                'room_size': 0.3,
                'wall_alignment': 0.4,
                'flow_efficiency': 0.3
            }

        try:
            # Get feedback data from database if not provided
            if not feedback_data:
                query = """
                SELECT original_blueprint, modified_blueprint, feedback_score
                FROM ai_blueprint_feedback
                WHERE feedback_score IS NOT NULL
                """
                results = execute_query(query)

                if not results or len(results) < 5:
                    return {
                        "success": False,
                        "error": "Not enough feedback data (minimum 5 samples required)"
                    }

                feedback_data = [
                    {
                        'original': json.loads(row[0]),
                        'modified': json.loads(row[1]),
                        'score': row[2]
                    }
                    for row in results
                ]

            # Create blueprint environment
            env = self._create_blueprint_environment(reward_weights)

            # Initialize model
            model = PPO('MlpPolicy', env, learning_rate=learning_rate, gamma=discount_factor, verbose=1)

            # Train model
            model.learn(total_timesteps=10000)

            # Save model
            model_path = MODEL_DIR / 'blueprint_refinement_model.zip'
            model.save(model_path)

            # Update model in memory
            self.blueprint_refinement_model = model

            # Save model info to database
            metrics_json = json.dumps({
                'reward_weights': reward_weights,
                'learning_rate': learning_rate,
                'discount_factor': discount_factor,
                'samples': len(feedback_data)
            })

            self._save_model_info_to_sqlite('blueprint_refinement', 'ppo', str(model_path), metrics)

            logger.info(f"Trained blueprint refinement model with {len(feedback_data)} feedback samples")

            return {
                "success": True,
                "samples": len(feedback_data),
                "reward_weights": reward_weights
            }

        except Exception as e:
            logger.error(f"Failed to train blueprint refinement model: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    def _create_blueprint_environment(self, reward_weights):
        """Create a gym environment for blueprint refinement."""
        # This is a simplified environment for blueprint refinement
        class BlueprintEnv(gym.Env):
            def __init__(self, reward_weights):
                super().__init__()
                self.reward_weights = reward_weights

                # Define action and observation space
                self.action_space = gym.spaces.Box(
                    low=-1.0, high=1.0, shape=(10,), dtype=np.float32
                )

                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32
                )

                self.blueprint = None
                self.step_count = 0

            def reset(self):
                self.blueprint = self._generate_random_blueprint()
                self.step_count = 0
                return self._get_observation()

            def step(self, action):
                self.step_count += 1

                # Apply action to modify blueprint
                self._apply_action(action)

                # Calculate reward
                reward = self._calculate_reward()

                # Check if done
                done = self.step_count >= 20

                return self._get_observation(), reward, done, {}

            def _generate_random_blueprint(self):
                # Generate a simple random blueprint for training
                return {
                    'rooms': [
                        {
                            'center': {'x': np.random.uniform(0, 10), 'y': np.random.uniform(0, 10), 'z': 0},
                            'dimensions': {'width': np.random.uniform(2, 5), 'length': np.random.uniform(2, 5), 'height': 2.4}
                        }
                        for _ in range(np.random.randint(1, 5))
                    ],
                    'walls': [
                        {
                            'start': {'x': np.random.uniform(0, 10), 'y': np.random.uniform(0, 10)},
                            'end': {'x': np.random.uniform(0, 10), 'y': np.random.uniform(0, 10)},
                            'thickness': 0.1,
                            'height': 2.4
                        }
                        for _ in range(np.random.randint(5, 15))
                    ]
                }

            def _apply_action(self, action):
                # Apply action to modify blueprint
                if len(self.blueprint['rooms']) > 0:
                    room_idx = np.random.randint(0, len(self.blueprint['rooms']))
                    room = self.blueprint['rooms'][room_idx]

                    # Modify room dimensions
                    room['dimensions']['width'] = max(1.5, room['dimensions']['width'] + action[0])
                    room['dimensions']['length'] = max(1.5, room['dimensions']['length'] + action[1])

                    # Modify room position
                    room['center']['x'] += action[2]
                    room['center']['y'] += action[3]

                if len(self.blueprint['walls']) > 0:
                    wall_idx = np.random.randint(0, len(self.blueprint['walls']))
                    wall = self.blueprint['walls'][wall_idx]

                    # Modify wall position
                    wall['start']['x'] += action[4]
                    wall['start']['y'] += action[5]
                    wall['end']['x'] += action[6]
                    wall['end']['y'] += action[7]

            def _get_observation(self):
                # Convert blueprint to observation vector
                obs = np.zeros(20)

                # Add room features
                room_count = min(2, len(self.blueprint['rooms']))
                for i in range(room_count):
                    room = self.blueprint['rooms'][i]
                    idx = i * 5
                    obs[idx] = room['center']['x']
                    obs[idx+1] = room['center']['y']
                    obs[idx+2] = room['dimensions']['width']
                    obs[idx+3] = room['dimensions']['length']
                    obs[idx+4] = room['dimensions']['height']

                # Add wall features
                wall_count = min(2, len(self.blueprint['walls']))
                for i in range(wall_count):
                    wall = self.blueprint['walls'][i]
                    idx = 10 + i * 5
                    obs[idx] = wall['start']['x']
                    obs[idx+1] = wall['start']['y']
                    obs[idx+2] = wall['end']['x']
                    obs[idx+3] = wall['end']['y']
                    obs[idx+4] = wall['thickness']

                return obs

            def _calculate_reward(self):
                # Calculate reward based on blueprint quality
                reward = 0

                # Room size reward
                room_size_reward = 0
                for room in self.blueprint['rooms']:
                    area = room['dimensions']['width'] * room['dimensions']['length']
                    if 6 <= area <= 30:  # Reasonable room size
                        room_size_reward += 1
                    else:
                        room_size_reward -= 1

                # Wall alignment reward
                wall_alignment_reward = 0
                for wall in self.blueprint['walls']:
                    # Check if wall is horizontal or vertical
                    dx = wall['end']['x'] - wall['start']['x']
                    dy = wall['end']['y'] - wall['start']['y']

                    if abs(dx) > abs(dy):  # Horizontal wall
                        wall_alignment_reward += 1 if abs(dy) < 0.1 else -1
                    else:  # Vertical wall
                        wall_alignment_reward += 1 if abs(dx) < 0.1 else -1

                # Flow efficiency reward (simplified)
                flow_efficiency_reward = 0

                # Combine rewards using weights
                reward = (
                    self.reward_weights['room_size'] * room_size_reward +
                    self.reward_weights['wall_alignment'] * wall_alignment_reward +
                    self.reward_weights['flow_efficiency'] * flow_efficiency_reward
                )

                return reward

        # Create and return the environment
        env = BlueprintEnv(reward_weights)
        return DummyVecEnv([lambda: env])

    def refine_blueprint(self, blueprint: Dict) -> Dict:
         logger.warning("AI refine_blueprint not implemented, returning original.")
         return blueprint # Placeholder

    def _save_blueprint_refinement(self, original: Dict, refined: Dict) -> None:
         logger.warning("_save_blueprint_refinement called, but RL not fully implemented.")
         # Call DB helper if feedback table exists
         # from .db import save_ai_feedback_to_sqlite
         # save_ai_feedback_to_sqlite(original.get('id','unknown'), {}, original, refined)
         pass

    def detect_movement_patterns(self):
        """Detect device movement patterns to improve position accuracy."""
        try:
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Get recent device positions over time
            cursor.execute('''
            SELECT device_id, position_data, timestamp FROM device_positions
            WHERE timestamp > datetime('now', '-24 hours')
            ORDER BY device_id, timestamp
            ''')
            results = cursor.fetchall()

            # Group by device
            device_tracks = {}
            for row in results:
                device_id, position_data, timestamp = row
                if device_id not in device_tracks:
                    device_tracks[device_id] = []

                position = json.loads(position_data)
                device_tracks[device_id].append({
                    'x': position.get('x', 0),
                    'y': position.get('y', 0),
                    'z': position.get('z', 0),
                    'timestamp': timestamp
                })

            # Analyze movement patterns
            patterns = {}
            for device_id, positions in device_tracks.items():
                if len(positions) < 5:  # Need enough data points
                    continue

                # Calculate average speed
                speeds = []
                for i in range(1, len(positions)):
                    p1, p2 = positions[i-1], positions[i]
                    distance = math.sqrt((p2['x']-p1['x'])**2 + (p2['y']-p1['y'])**2)
                    # Parse timestamps and calculate time difference
                    t1 = datetime.strptime(p1['timestamp'], '%Y-%m-%d %H:%M:%S')
                    t2 = datetime.strptime(p2['timestamp'], '%Y-%m-%d %H:%M:%S')
                    time_diff = (t2 - t1).total_seconds()
                    if time_diff > 0:
                        speeds.append(distance/time_diff)

                # Store pattern data
                if speeds:
                    patterns[device_id] = {
                        'avg_speed': sum(speeds)/len(speeds),
                        'max_speed': max(speeds),
                        'static': sum(speeds)/len(speeds) < 0.1  # Less than 10cm/sec = static
                    }

            return patterns
        except Exception as e:
            logger.error(f"Error detecting movement patterns: {e}")
            return {}

    def apply_spatial_memory(self, device_positions):
        """Apply spatial memory to improve position estimates."""
        try:
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            for device_id, position in device_positions.items():
                # Get recent positions for this device
                cursor.execute('''
                SELECT position_data FROM device_positions
                WHERE device_id = ? AND timestamp > datetime('now', '-60 minutes')
                ORDER BY timestamp DESC LIMIT 10
                ''', (device_id,))
                recent_positions = cursor.fetchall()

                if recent_positions:
                    # Average recent positions to smooth out errors
                    x_vals, y_vals, z_vals = [], [], []
                    weights = []

                    # Apply exponential decay weights (newer positions matter more)
                    for i, row in enumerate(recent_positions):
                        pos = json.loads(row[0])
                        weight = math.exp(-0.3 * i)  # Decay factor

                        x_vals.append(pos.get('x', 0) * weight)
                        y_vals.append(pos.get('y', 0) * weight)
                        z_vals.append(pos.get('z', 0) * weight)
                        weights.append(weight)

                    # Calculate weighted average
                    total_weight = sum(weights)
                    if total_weight > 0:
                        avg_x = sum(x_vals) / total_weight
                        avg_y = sum(y_vals) / total_weight
                        avg_z = sum(z_vals) / total_weight

                        # Blend current position with historical average (80/20)
                        position['x'] = position['x'] * 0.8 + avg_x * 0.2
                        position['y'] = position['y'] * 0.8 + avg_y * 0.2
                        position['z'] = position['z'] * 0.8 + avg_z * 0.2

            return device_positions
        except Exception as e:
            logger.error(f"Error applying spatial memory: {e}")
            return device_positions

    def _save_model_info_to_sqlite(self, model_name: str, model_type: str,
                              model_path: str, metrics: Dict) -> bool:
        """Save model information to SQLite database."""
        try:
            from .db import get_sqlite_connection
            conn = get_sqlite_connection()
            cursor = conn.cursor()

            # Check if model exists
            cursor.execute(
                "SELECT id FROM ai_models WHERE model_name = ?",
                (model_name,)
            )
            model_exists = cursor.fetchone() is not None

            # Serialize metrics
            metrics_json = json.dumps(metrics)

            if model_exists:
                # Update existing model
                cursor.execute("""
                UPDATE ai_models
                SET model_type = ?, model_data = ?, metrics = ?, version = version + 1
                WHERE model_name = ?
                """, (model_type, model_path, metrics_json, model_name))
            else:
                # Insert new model
                cursor.execute("""
                INSERT INTO ai_models
                (model_name, model_type, model_data, metrics, version, created_at)
                VALUES (?, ?, ?, ?, 1, datetime('now'))
                """, (model_name, model_type, model_path, metrics_json))

            conn.commit()
            conn.close()

            logger.info(f"Saved {model_name} model info to database")
            return True

        except Exception as e:
            logger.error(f"Failed to save model info to database: {str(e)}")
            return False

    # --- NEW/PLACEHOLDER AI/HEURISTIC METHODS ---

    def calculate_area_adjacency(self, min_transitions=2) -> Dict[str, List[str]]:
        """Analyze transitions to determine likely adjacent areas."""
        logger.info("Calculating area adjacency from observations...")
        adjacency: Dict[str, Counter[str]] = {}
        last_area: Dict[str, str] = {}

        try:
            # Use the new DB helper function
            observations = get_area_observations(limit=10000) # Get recent observations
            if not observations:
                logger.warning("No area observations found in database to calculate adjacency.")
                return {}

            # Process chronologically
            for obs in sorted(observations, key=lambda x: x['timestamp']):
                dev = obs['tracked_device_id']
                current_area = obs['predicted_area_id']

                # Skip if area is None or empty
                if not current_area:
                    last_area[dev] = None # Reset last area if prediction lost
                    continue

                prev_area = last_area.get(dev)

                # Check for a valid transition (must have a previous area different from current)
                if prev_area and prev_area != current_area:
                    # Ensure areas exist in adjacency dict
                    adjacency.setdefault(prev_area, Counter())
                    adjacency.setdefault(current_area, Counter())
                    # Increment count for both directions
                    adjacency[prev_area][current_area] += 1
                    adjacency[current_area][prev_area] += 1
                    logger.debug(f"Observed transition: {prev_area} <-> {current_area} for {dev}")

                # Update last known area for the device
                last_area[dev] = current_area

            # Filter adjacencies by minimum transition count
            final_adjacency: Dict[str, List[str]] = {}
            for area1, neighbors in adjacency.items():
                valid_neighbors = [area2 for area2, count in neighbors.items() if count >= min_transitions]
                if valid_neighbors:
                    final_adjacency[area1] = valid_neighbors

            logger.info(f"Calculated adjacency for {len(final_adjacency)} areas (min transitions: {min_transitions}).")
            logger.debug(f"Adjacency result: {final_adjacency}")
            return final_adjacency

        except Exception as e:
            logger.error(f"Error calculating area adjacency: {e}", exc_info=True)
            return {}

    def generate_heuristic_layout(self, area_ids: List[str], adjacency: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """Generate a basic 2D layout using adjacency graph via BFS."""
        logger.info("Generating heuristic layout for areas...")
        layout = {}
        if not area_ids:
            logger.warning("No area IDs provided for layout generation.")
            return {}

        q = []
        placed = set()

        # Start BFS from the first area ID
        start_node = area_ids[0]
        layout[start_node] = {'x': 0.0, 'y': 0.0}
        q.append(start_node)
        placed.add(start_node)

        head = 0
        processed_neighbors = set() # To avoid placing based on the same neighbor twice

        while head < len(q):
            current_area = q[head]
            head += 1
            cx, cy = layout[current_area]['x'], layout[current_area]['y']

            neighbors = adjacency.get(current_area, [])
            unplaced_neighbors = [n for n in neighbors if n in area_ids and n not in placed]

            # Simple angular placement (can be improved)
            angle_step = (2 * math.pi) / max(1, len(unplaced_neighbors))
            placement_distance = 5.0 # Default distance

            for i, neighbor in enumerate(unplaced_neighbors):
                # Check if this specific neighbor relationship has been processed
                pair = tuple(sorted((current_area, neighbor)))
                if pair in processed_neighbors:
                    continue

                angle = i * angle_step + (random.random() * 0.1 - 0.05) # Add slight jitter
                nx = cx + placement_distance * math.cos(angle)
                ny = cy + placement_distance * math.sin(angle)

                layout[neighbor] = {'x': round(nx, 2), 'y': round(ny, 2)}
                placed.add(neighbor)
                q.append(neighbor)
                processed_neighbors.add(pair) # Mark pair as processed

        # Place any disconnected areas arbitrarily
        current_x = (max(d['x'] for d in layout.values()) if layout else 0) + placement_distance * 2
        current_y = 0
        for area_id in area_ids:
            if area_id not in placed:
                 logger.warning(f"Area {area_id} seems disconnected, placing arbitrarily.")
                 layout[area_id] = {'x': current_x, 'y': current_y}
                 placed.add(area_id)
                 current_x += placement_distance


        logger.info(f"Generated heuristic layout for {len(layout)} areas.")
        logger.debug(f"Layout result: {layout}")
        return layout

    def estimate_room_dimensions(self, area_id: str, devices_in_area: List[str]) -> Dict[str, float]:
        """Placeholder: Estimate room size based on device count."""
        # Get defaults/constraints from config
        validation_config = self.config.get('blueprint_validation', {})
        min_dim = validation_config.get('min_room_dimension', 1.5)
        max_dim = validation_config.get('max_room_dimension', 15.0)
        min_height = validation_config.get('min_ceiling_height', 2.2)
        max_height = validation_config.get('max_ceiling_height', 4.0)

        # Simple heuristic: base size + increment per device
        num_devices = len(devices_in_area)
        base_size = 3.0 # meters
        # Size increases slower after the first few devices
        size_factor = base_size + math.log1p(max(0, num_devices)) * 0.75

        width = round(size_factor, 1)
        length = round(size_factor * 1.2, 1) # Slightly rectangular default
        height = min_height # Default height

        # Apply constraints
        width = max(min_dim, min(max_dim, width))
        length = max(min_dim, min(max_dim, length))
        height = max(min_height, min(max_height, height))

        logger.debug(f"Estimated dimensions for {area_id} ({num_devices} devices): W={width}, L={length}, H={height}")
        return {'width': width, 'length': length, 'height': height}

    def predict_walls_between_areas(self, area1_id, area2_id, transition_data) -> float:
        """Placeholder: Predict wall probability. Returns 0.0 (no wall)."""
        # Phase 1: No wall prediction yet.
        # Phase 2: Implement ML model using transition_data (RSSI changes).
        logger.debug(f"Placeholder: Predicting NO wall between {area1_id} and {area2_id}")
        return 0.0 # Return 0 probability for now

    # New methods for relative positioning and room geometry generation

    def run_relative_positioning(self, distance_data: List[Dict[str, Any]], n_dimensions=3) -> Dict[str, Dict[str, float]]:
        """Calculates relative positions using MDS. Expects distance_data as a list of dictionaries."""
        logger.info(f"Running relative positioning with {len(distance_data)} distance entries...")
        if not distance_data:
            logger.warning("No distance data provided for relative positioning.")
            return {}

        # --- NEW: Log type and first few elements of distance_data ---
        logger.debug(f"Type of distance_data received: {type(distance_data)}")
        if isinstance(distance_data, list) and len(distance_data) > 0:
            logger.debug(f"Type of first element: {type(distance_data[0])}")
            logger.debug(f"Value of first element: {distance_data[0]}")
            if len(distance_data) > 1:
                 logger.debug(f"Value of second element: {distance_data[1]}")
        # --- END NEW LOGGING ---

        # 1. Identify all unique entities (devices + scanners) from the list of dictionaries
        entities = set()
        measurements = {} # {(id1, id2): [dist1, dist2, ...]}

        logger.debug("--- Raw distance data sample & Entity Extraction ---")
        processed_records = 0
        valid_records_for_debug = 0

        for idx, record in enumerate(distance_data):
            dev_id = None
            scan_id = None
            dist = None
            timestamp = 'N/A'

            # --- UPDATED: Explicitly handle tuple vs dict and validate ---
            if isinstance(record, dict):
                dev_id = record.get('tracked_device_id')
                scan_id = record.get('scanner_id')
                dist = record.get('distance')
                timestamp = record.get('timestamp', 'N/A')
            elif isinstance(record, (tuple, list)) and len(record) == 4:
                # Assuming order: timestamp, tracked_device_id, scanner_id, distance
                timestamp, dev_id, scan_id, dist = record
                if idx < 5: # Log warning for first few tuple records
                     logger.warning(f"Received record as tuple/list at index {idx}, expected dict. Data: {record}")
            else:
                if idx < 20: # Log error only for the first few records to avoid spam
                    logger.warning(f"Skipping invalid record format at index {idx}: type={type(record)}, value={record}")
                continue
            # --- END UPDATED ---

            # Validate IDs - ensure they are not None or empty strings
            if not dev_id or not isinstance(dev_id, str):
                if idx < 20:
                    logger.warning(f"Skipping record with invalid tracked_device_id at index {idx}: {dev_id}")
                continue
            if not scan_id or not isinstance(scan_id, str):
                if idx < 20:
                    logger.warning(f"Skipping record with invalid scanner_id at index {idx}: {scan_id}")
                continue

            # Log first 20 valid records for debugging
            if valid_records_for_debug < 20:
                # Ensure dist is a float - handle the case where it's a string or None
                try:
                    if dist is None:
                        dist_value = None
                        dist_display = "None"
                    elif isinstance(dist, (int, float)):
                        dist_value = float(dist)
                        dist_display = f"{dist_value:.2f}"
                    elif isinstance(dist, str):
                        dist_value = float(dist) # Attempt conversion
                        dist_display = f"{dist_value:.2f}"
                    else:
                        dist_value = None # Treat other types as invalid
                        dist_display = f"InvalidType({type(dist).__name__})"
                except (ValueError, TypeError):
                    dist_value = None # Treat conversion errors as invalid
                    dist_display = f"InvalidValue({dist})"

                logger.debug(f"  Record (idx {idx}): dev='{dev_id}', scan='{scan_id}', dist={dist_display}, ts='{timestamp}'")
                valid_records_for_debug += 1
            else:
                # Convert distance for processing after debug limit
                 try:
                    if dist is None:
                        dist_value = None
                    elif isinstance(dist, (int, float)):
                        dist_value = float(dist)
                    elif isinstance(dist, str):
                        dist_value = float(dist)
                    else:
                        dist_value = None
                 except (ValueError, TypeError):
                    dist_value = None

            # Add valid entities and measurements
            entities.add(dev_id)
            entities.add(scan_id)
            pair = tuple(sorted((dev_id, scan_id)))

            if dist_value is not None and dist_value >= 0: # Only add valid, non-negative distances
                measurements.setdefault(pair, []).append(dist_value)
            elif valid_records_for_debug < 20: # Log warning only for first few invalid distances
                 logger.warning(f"Skipping record with invalid distance value at index {idx}: {dist} (type: {type(dist).__name__}) for pair ({dev_id}, {scan_id})")

            processed_records += 1

        # Diagnostic logging about entities found
        entity_list = sorted(list(entities))
        logger.debug("--- Entity Identification Summary ---")
        logger.debug(f"Processed {processed_records} records from distance_data.")
        logger.debug(f"Total unique entities identified: {len(entity_list)}")
        logger.debug(f"Entities identified (full list): {entity_list}")
        logger.debug(f"Total measurement pairs with valid distances: {len(measurements)}")

        # Add reference positions to entities and measurements if needed
        if len(entity_list) < n_dimensions + 1:
            logger.warning(f"Not enough unique entities ({len(entity_list)}) identified from processed records for {n_dimensions}D positioning.")
            logger.warning(f"Entities found: {entity_list}")

            # Try to retrieve reference positions from the database
            try:
                from .db import get_reference_positions_from_sqlite
                ref_positions = get_reference_positions_from_sqlite()
                if ref_positions:
                    logger.info(f"Found {len(ref_positions)} reference positions to supplement entity list")

                    # Check which reference positions are not already in our entity list
                    new_refs = [ref_id for ref_id in ref_positions.keys() if ref_id not in entities]
                    logger.info(f"Adding {len(new_refs)} new reference positions to entity calculation")

                    # Add synthetic distance measurements between reference points
                    for i, ref_id1 in enumerate(new_refs):
                        entities.add(ref_id1)
                        pos1 = ref_positions[ref_id1]

                        # Connect to other reference points with synthetic distances
                        for j, ref_id2 in enumerate(new_refs[i+1:], i+1):
                            entities.add(ref_id2)
                            pos2 = ref_positions[ref_id2]

                            # Calculate Euclidean distance between reference points
                            dist = np.sqrt((pos1['x'] - pos2['x'])**2 +
                                          (pos1['y'] - pos2['y'])**2 +
                                          (pos1['z'] - pos2['z'])**2)

                            pair = tuple(sorted((ref_id1, ref_id2)))
                            measurements[pair] = [float(dist)]
                            logger.debug(f"Added synthetic distance: {ref_id1} to {ref_id2} = {dist:.2f}m")

                        # Also connect to real devices if possible (using distance > 0)
                        # Find *actual* entities identified from distance_data (excluding refs)
                        actual_entities_found = [e for e in entity_list if e not in ref_positions]
                        for real_entity in actual_entities_found:
                            pair = tuple(sorted((ref_id1, real_entity)))
                            # Estimate distance based on area_id if available
                            # For now just use arbitrary distance for testing
                            measurements.setdefault(pair, []).append(10.0) # Use setdefault to avoid overwriting real data
                            logger.debug(f"Added synthetic distance: {ref_id1} to {real_entity} = 10.0m")

                    # Re-check entity count after adding references
                    entity_list = sorted(list(entities))
                    logger.info(f"After adding reference positions: {len(entity_list)} entities available")
            except Exception as e:
                logger.error(f"Failed to add reference positions to positioning calculation: {e}", exc_info=True)

        entity_map = {name: i for i, name in enumerate(entity_list)}
        n_entities = len(entity_list)
        if n_entities < n_dimensions + 1:
             logger.error(f"Still not enough entities ({n_entities}) for {n_dimensions}D positioning even after adding references.")
             return {}

        # 2. Create Dissimilarity Matrix (use median distance)
        dissimilarity_matrix = np.zeros((n_entities, n_entities))
        valid_measurement_count = 0
        for (id1, id2), dists in measurements.items():
            if id1 in entity_map and id2 in entity_map:
                 idx1, idx2 = entity_map[id1], entity_map[id2]
                 # Filter out potential non-numeric values that might have slipped through
                 numeric_dists = [d for d in dists if isinstance(d, (int, float)) and d >= 0]
                 if numeric_dists:
                     median_dist = np.median(numeric_dists)
                     dissimilarity_matrix[idx1, idx2] = median_dist
                     dissimilarity_matrix[idx2, idx1] = median_dist
                     valid_measurement_count += 1
                 else:
                     logger.warning(f"No valid numeric distances found for pair ({id1}, {id2})")

        logger.debug(f"Created dissimilarity matrix with {valid_measurement_count} valid median distance pairs.")

        # Set maximum dissimilarity for missing measurements
        # This helps MDS handle sparse matrices better
        valid_distances = dissimilarity_matrix[dissimilarity_matrix > 0]
        if valid_distances.size > 0:
            max_dist = np.max(valid_distances)
            # Use max_dist + 10% for missing values (but not on diagonal)
            missing_value = max_dist * 1.1
            for i in range(n_entities):
                for j in range(n_entities):
                    if i != j and dissimilarity_matrix[i, j] == 0:
                        dissimilarity_matrix[i, j] = missing_value
            logger.debug(f"Filled missing dissimilarity values with {missing_value:.2f}")
        else:
            logger.error("No valid distances found to build dissimilarity matrix. Cannot run MDS.")
            return {}

        # 3. Run MDS
        try:
            mds = MDS(n_components=n_dimensions, dissimilarity='precomputed',
                     random_state=42, n_init=4, max_iter=300, verbose=0, n_jobs=-1) # Use n_jobs=-1 for parallel
            relative_pos_array = mds.fit_transform(dissimilarity_matrix)
            logger.info(f"MDS stress: {mds.stress_:.4f}")
        except Exception as e:
             logger.error(f"MDS failed: {e}", exc_info=True)
             return {}

        # 4. Format output
        relative_coords = {}
        for i, name in enumerate(entity_list):
            relative_coords[name] = {'x': float(relative_pos_array[i, 0]), 'y': float(relative_pos_array[i, 1])}
            if n_dimensions > 2:
                relative_coords[name]['z'] = float(relative_pos_array[i, 2])
            else:
                relative_coords[name]['z'] = 0.0 # Default Z if 2D

        logger.info(f"Calculated relative positions for {len(relative_coords)} entities.")
        return relative_coords

    def calculate_anchoring_transform(self, relative_coords: Dict[str, Dict[str, float]],
                                     device_area_map: Dict[str, str],
                                     target_area_layout: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Finds the transformation to map relative coords to target layout via area centroids."""
        logger.info("Calculating anchoring transformation...")
        relative_centroids = {}
        target_centroids = {}
        points_per_area = {}

        # 1. Group relative device points by area and calculate relative centroids
        for dev_id, area_id in device_area_map.items():
            if area_id and dev_id in relative_coords and area_id in target_area_layout:
                 points_per_area.setdefault(area_id, []).append([
                     relative_coords[dev_id]['x'],
                     relative_coords[dev_id]['y']
                     # Ignoring Z for 2D anchoring initially
                 ])

        if not points_per_area:
             logger.warning("No devices with both relative coordinates and area predictions found.")
             return None

        # Calculate centroids only for areas with points
        valid_area_ids = list(points_per_area.keys())
        for area_id in valid_area_ids:
            points = np.array(points_per_area[area_id])
            if points.shape[0] > 0:
                centroid = np.mean(points, axis=0)
                relative_centroids[area_id] = centroid
                # Ensure the target layout has this area
                if area_id in target_area_layout:
                     target_centroids[area_id] = [target_area_layout[area_id]['x'], target_area_layout[area_id]['y']]

        # Ensure we have matching centroids in both sets
        common_areas = sorted(list(set(relative_centroids.keys()) & set(target_centroids.keys())))
        if len(common_areas) < 2: # Need at least 2 points for Procrustes (or 3 for non-degenerate)
             logger.warning(f"Not enough common areas ({len(common_areas)}) with centroids for anchoring.")
             return None

        matrix_a = np.array([relative_centroids[aid] for aid in common_areas])
        matrix_b = np.array([target_centroids[aid] for aid in common_areas])

        # 2. Perform Procrustes Analysis
        try:
            mtx1, mtx2, disparity = procrustes(matrix_b, matrix_a)

            # Estimate scale: ratio of distances between corresponding points
            dist_a = np.linalg.norm(matrix_a[0] - matrix_a[1])
            dist_b = np.linalg.norm(matrix_b[0] - matrix_b[1])
            scale = dist_b / dist_a if dist_a > 1e-6 else 1.0

            # Estimate translation: difference between centroids of centroids
            centroid_a = np.mean(matrix_a, axis=0)
            centroid_b = np.mean(matrix_b, axis=0)

            # Estimate Rotation using SVD on covariance matrix
            # Center the point sets around the origin
            centered_a = matrix_a - centroid_a
            centered_b = matrix_b - centroid_b

            # Compute the covariance matrix
            covariance = centered_a.T @ centered_b

            # SVD decomposition
            try:
                U, S, Vt = np.linalg.svd(covariance)
                # Ensure proper rotation (not reflection)
                rotation = Vt.T @ U.T
                if np.linalg.det(rotation) < 0:
                    Vt[-1, :] *= -1
                    rotation = Vt.T @ U.T
            except np.linalg.LinAlgError:
                # Fallback if SVD fails
                logger.warning("SVD failed for rotation estimation, using identity matrix")
                rotation = np.identity(2)

            # Calculate translation based on scale and rotation
            translation = centroid_b - scale * (centroid_a @ rotation)

            transform_params = {
                'scale': float(scale),
                'rotation': rotation.tolist(),  # Convert to list for serialization
                'translation': translation.tolist(),
                'disparity': float(disparity)
            }

            logger.info(f"Anchoring transform calculated. Scale: {scale:.3f}, Disparity: {disparity:.4f}")
            return transform_params

        except Exception as e:
            logger.error(f"Procrustes analysis failed: {e}", exc_info=True)
            return None

    def apply_transform(self, coords: Dict[str, Dict[str, float]], transform_params: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Applies scale, rotation, translation to coordinates."""
        if not transform_params:
            return coords

        anchored_coords = {}
        s = transform_params['scale']
        R = np.array(transform_params['rotation'])  # Convert from list
        t = np.array(transform_params['translation'])  # Convert from list

        for entity_id, pos in coords.items():
            # Assuming 2D transform for now
            point_relative = np.array([pos['x'], pos['y']])
            point_anchored = s * (point_relative @ R) + t
            anchored_coords[entity_id] = {
                'x': float(point_anchored[0]),
                'y': float(point_anchored[1]),
                'z': float(pos.get('z', 0))  # Keep original Z or transform if 3D
            }

        logger.info(f"Applied anchoring transform to {len(anchored_coords)} entities.")
        return anchored_coords

    def generate_rooms_from_points(self, anchored_device_coords_by_area: Dict[str, List[Dict[str, float]]], *args) -> List[Dict]:
        """Generate rooms from device coordinates grouped by area using AI/ML techniques.
        Note: Added *args to handle any extra arguments from legacy code.
        """
        logger.info(f"Generating rooms from {len(anchored_device_coords_by_area)} areas with device coordinates.")
        rooms = []
        room_id = 0

        # Get validation settings from config
        validation_config = self.config.get('blueprint_validation', {})
        min_dim = validation_config.get('min_room_dimension', 1.5)
        max_dim = validation_config.get('max_room_dimension', 15.0)
        min_area = validation_config.get('min_room_area', 4.0)
        min_points = self.config.get('generation_settings', {}).get('min_points_per_room', 3)

        # Process each area
        for area_id, devices in anchored_device_coords_by_area.items():
            if len(devices) < min_points:
                logger.warning(f"Area {area_id} has only {len(devices)} devices, which is below the minimum threshold of {min_points}. Skipping.")
                continue

            room_id += 1

            try:
                # Extract device coordinates
                points = [(d['x'], d['y']) for d in devices]

                # Use ML clustering to identify potential sub-rooms if there are many points
                if len(points) > 10:
                    try:
                        # Try DBSCAN for cluster detection
                        clustering = DBSCAN(eps=1.5, min_samples=3).fit(points)
                        labels = clustering.labels_

                        # If we found meaningful clusters, use them
                        if len(set(labels)) > 1 and -1 not in labels:
                            logger.info(f"ML clustering identified {len(set(labels))} sub-areas in {area_id}")
                            # Process sub-clusters (implementation omitted for brevity)
                    except Exception as e:
                        logger.warning(f"ML clustering failed for {area_id}: {e}")

                # Create a convex hull from the points
                if len(points) >= 3:
                    # Use ML-enhanced convex hull with outlier detection
                    try:
                        # Convert to numpy array for processing
                        points_array = np.array(points)

                        # Detect and remove outliers using statistical methods
                        distances = np.zeros(len(points_array))
                        centroid = np.mean(points_array, axis=0)

                        for i, point in enumerate(points_array):
                            distances[i] = np.linalg.norm(point - centroid)

                        # Use IQR method for outlier detection
                        q1, q3 = np.percentile(distances, [25, 75])
                        iqr = q3 - q1
                        threshold = q3 + 1.5 * iqr

                        # Filter points
                        filtered_points = points_array[distances <= threshold]

                        # If we have enough points after filtering, create hull
                        if len(filtered_points) >= 3:
                            hull = ConvexHull(filtered_points)
                            polygon_coords = [
                                (float(filtered_points[i, 0]), float(filtered_points[i, 1]))
                                for i in hull.vertices
                            ]
                        else:
                            # Fallback to original points if filtering removed too many
                            hull = ConvexHull(points_array)
                            polygon_coords = [
                                (float(points_array[i, 0]), float(points_array[i, 1]))
                                for i in hull.vertices
                            ]
                    except Exception as e:
                        logger.warning(f"ML convex hull generation failed: {e}, using simple bounding box")
                        # Fallback to bounding box
                        polygon_coords = None
                else:
                    polygon_coords = None

                # Calculate room bounds (min/max)
                xs = [d['x'] for d in devices]
                ys = [d['y'] for d in devices]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)

                # Apply minimum size constraints
                width = max(min_dim, max_x - min_x)
                length = max(min_dim, max_y - min_y)

                # Apply padding for better visualization
                padding = min(2.0, width * 0.1, length * 0.1)  # 10% padding, max 2m
                min_x -= padding
                min_y -= padding
                max_x += padding
                max_y += padding

                # Create room object
                room = {
                    'id': f"room_{room_id}",
                    'name': area_id.replace('_', ' ').title(),
                    'area_id': area_id,
                    'bounds': {
                        'min': {'x': float(min_x), 'y': float(min_y), 'z': 0},
                        'max': {'x': float(max_x), 'y': float(max_y), 'z': 2.4}
                    },
                    'devices': [d.get('device_id', f"device_{i}") for i, d in enumerate(devices)]
                }

                # Add polygon coordinates if available
                if polygon_coords:
                    room['polygon_coords'] = polygon_coords

                rooms.append(room)
                logger.debug(f"Generated room: {room['name']} with {len(devices)} devices")

            except Exception as e:
                logger.error(f"Failed to generate room for area {area_id}: {e}", exc_info=True)

        logger.info(f"Generated {len(rooms)} rooms from device coordinates")
        return rooms

    def generate_walls_between_rooms(self, rooms: List[Dict]) -> List[Dict]:
        """Infers walls based on proximity and alignment of room boundaries."""
        logger.info("Generating walls between rooms...")
        walls = []
        wall_id = 0

        # Skip if there are not enough rooms
        if len(rooms) < 2:
            logger.warning("Not enough rooms to generate walls.")
            return []

        try:
            # Create polygons for each room
            room_polygons = []
            for room in rooms:
                if 'polygon_coords' in room:
                    poly = Polygon(room['polygon_coords'])
                    room_polygons.append((room['id'], poly))

            # Set distance threshold for detecting walls (in meters)
            wall_distance_threshold = 1.5

            # Compare each pair of rooms
            for i in range(len(room_polygons)):
                room1_id, poly1 = room_polygons[i]

                for j in range(i+1, len(room_polygons)):
                    room2_id, poly2 = room_polygons[j]

                    # Check if rooms are close but not overlapping
                    if poly1.distance(poly2) < wall_distance_threshold and not poly1.intersects(poly2):
                        # Find points on boundaries that are closest
                        # For simplicity, this example uses the closest points of the polygons
                        # A more complex implementation would find the line segments

                        # Get exterior lines of both polygons
                        lines1 = list(zip(poly1.exterior.coords[:-1], poly1.exterior.coords[1:]))
                        lines2 = list(zip(poly2.exterior.coords[:-1], poly2.exterior.coords[1:]))

                        # Find closest pair of lines
                        min_dist = float('inf')
                        closest_pair = None

                        for line1 in lines1:
                            line1_poly = Polygon([line1[0], line1[1], line1[1], line1[0]])

                            for line2 in lines2:
                                line2_poly = Polygon([line2[0], line2[1], line2[1], line2[0]])
                                dist = line1_poly.distance(line2_poly)

                                if dist < min_dist:
                                    min_dist = dist
                                    closest_pair = (line1, line2)

                        if closest_pair and min_dist < wall_distance_threshold:
                            # Create wall from midpoints of closest lines
                            line1, line2 = closest_pair

                            mid1 = ((line1[0][0] + line1[1][0])/2, (line1[0][1] + line1[1][1])/2)
                            mid2 = ((line2[0][0] + line2[1][0])/2, (line2[0][1] + line2[1][1])/2)

                            # Calculate wall properties
                            wall_id += 1
                            thickness = 0.1  # Default wall thickness
                            height = 2.4     # Default wall height

                            # Calculate angle of the wall
                            angle = math.atan2(mid2[1] - mid1[1], mid2[0] - mid1[0])

                            wall = {
                                'id': f"wall_{wall_id}",
                                'start': {'x': float(mid1[0]), 'y': float(mid1[1]), 'z': 0},
                                'end': {'x': float(mid2[0]), 'y': float(mid2[1]), 'z': 0},
                                'thickness': thickness,
                                'height': height,
                                'angle': float(angle),
                                'connects': [room1_id, room2_id]
                            }

                            walls.append(wall)

            logger.info(f"Generated {len(walls)} walls between rooms.")
            return walls

        except Exception as e:
            logger.error(f"Failed to generate walls: {e}", exc_info=True)
            return []