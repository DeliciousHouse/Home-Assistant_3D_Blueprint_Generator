#!/usr/bin/env python3
"""
Static Device Detector Module

This module provides functionality to detect static (non-moving) BLE devices
that can be used as temporary high-confidence reference points for positioning.
"""

import logging
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
import time

from .config_loader import load_config
from .db import (
    save_device_position_history,
    get_device_position_history,
    get_all_device_position_history,
    save_static_device,
    get_active_static_devices,
    update_static_device_confidence
)

logger = logging.getLogger(__name__)

class StaticDeviceDetector:
    """
    Class for detecting static BLE devices based on their movement patterns.
    """

    def __init__(self):
        """Initialize the static device detector with configuration."""
        self.config = load_config()
        self.static_config = self.config.get('static_device_detection', {})
        self.enabled = self.static_config.get('enable_dynamic_anchors', True)

        # Get configuration values with defaults
        self.movement_threshold = self.static_config.get('movement_threshold_meters', 0.5)
        self.time_window_seconds = self.static_config.get('time_window_seconds', 300)
        self.min_observations = self.static_config.get('min_observations_for_static', 5)
        self.confidence_decay_hours = self.static_config.get('static_anchor_confidence_decay_hours', 1.0)
        self.max_dynamic_anchors = self.static_config.get('max_dynamic_anchors', 10)

        logger.info(f"Static Device Detector initialized (enabled: {self.enabled})")
        if self.enabled:
            logger.info(f"Movement threshold: {self.movement_threshold}m, "
                      f"Time window: {self.time_window_seconds}s, "
                      f"Min observations: {self.min_observations}")

    def record_device_position(self, device_id: str, x: float, y: float, z: float,
                             accuracy: Optional[float] = None, source: str = 'calculated') -> bool:
        """
        Record a device position in the history database for movement analysis.

        Args:
            device_id: Unique identifier for the device
            x, y, z: Coordinates in meters
            accuracy: Estimated position accuracy in meters
            source: Source of the position data

        Returns:
            True if the position was recorded successfully
        """
        if not self.enabled:
            return False

        return save_device_position_history(
            device_id=device_id,
            x=x, y=y, z=z,
            accuracy=accuracy,
            source=source
        )

    def calculate_movement_score(self, positions: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate a movement score for a device based on its position history.
        A lower score indicates a more static device.

        Args:
            positions: List of position records for a device

        Returns:
            Tuple of (movement_score, stats_dict)
        """
        if not positions or len(positions) < 2:
            return float('inf'), {'std_dev': float('inf'), 'max_distance': float('inf'), 'positions': 0}

        # Extract position coordinates
        coords = np.array([[p['x'], p['y'], p['z']] for p in positions])

        # Calculate centroid (average position)
        centroid = np.mean(coords, axis=0)

        # Calculate distances from centroid for each position
        distances = np.sqrt(np.sum((coords - centroid)**2, axis=1))

        # Various metrics for movement
        std_dev = np.std(distances)
        max_distance = np.max(distances)
        median_distance = np.median(distances)

        # Combined movement score (can be tuned as needed)
        # Higher weight on max_distance to be more sensitive to outlier movements
        movement_score = (0.3 * std_dev) + (0.6 * max_distance) + (0.1 * median_distance)

        stats = {
            'std_dev': float(std_dev),
            'max_distance': float(max_distance),
            'median_distance': float(median_distance),
            'positions': len(positions)
        }

        return float(movement_score), stats

    def process_device_history(self) -> List[Dict[str, Any]]:
        """
        Process the history of all devices to detect static devices.

        Returns:
            List of static devices that were detected or updated
        """
        if not self.enabled:
            logger.debug("Static device detection is disabled")
            return []

        # Update confidence of existing static devices
        update_static_device_confidence(decay_hours=self.confidence_decay_hours)

        # Get all device positions within the time window that have enough observations
        device_histories = get_all_device_position_history(
            time_window_seconds=self.time_window_seconds,
            min_observations=self.min_observations
        )

        if not device_histories:
            logger.debug("No devices with sufficient position history found")
            return []

        detected_devices = []

        # Process each device
        for device_id, positions in device_histories.items():
            # Calculate movement score
            movement_score, stats = self.calculate_movement_score(positions)

            # Skip devices with too much movement
            if movement_score > self.movement_threshold:
                logger.debug(f"Device {device_id} movement score {movement_score:.3f} > threshold {self.movement_threshold}")
                continue

            # Calculate confidence based on observations and movement score
            # More observations and less movement = higher confidence
            observation_factor = min(1.0, stats['positions'] / (self.min_observations * 2))
            movement_factor = max(0.1, 1.0 - (movement_score / self.movement_threshold))
            confidence = observation_factor * movement_factor

            # Calculate average position
            avg_x = sum(p['x'] for p in positions) / len(positions)
            avg_y = sum(p['y'] for p in positions) / len(positions)
            avg_z = sum(p['z'] for p in positions) / len(positions)

            # Save as static device
            if save_static_device(
                device_id=device_id,
                x=avg_x,
                y=avg_y,
                z=avg_z,
                confidence=confidence,
                movement_score=movement_score,
                observations_count=len(positions)
            ):
                detected_devices.append({
                    'device_id': device_id,
                    'position': {'x': avg_x, 'y': avg_y, 'z': avg_z},
                    'confidence': confidence,
                    'movement_score': movement_score,
                    'observations': len(positions)
                })
                logger.info(f"Detected static device {device_id} at ({avg_x:.2f}, {avg_y:.2f}, {avg_z:.2f}) "
                          f"confidence: {confidence:.2f}, movement: {movement_score:.3f}")

        return detected_devices

    def get_dynamic_anchors(self, min_confidence: float = 0.5) -> List[Dict[str, Any]]:
        """
        Get a list of static devices to use as dynamic anchors.

        Args:
            min_confidence: Minimum confidence threshold

        Returns:
            List of static devices with their positions
        """
        if not self.enabled:
            return []

        # Get active static devices with positions
        static_devices = get_active_static_devices(
            min_confidence=min_confidence,
            limit=self.max_dynamic_anchors
        )

        if not static_devices:
            return []

        # Format for blueprint generation
        dynamic_anchors = []
        for device in static_devices:
            dynamic_anchors.append({
                'device_id': device['device_id'],
                'position': {
                    'x': device['x'],
                    'y': device['y'],
                    'z': device['z']
                },
                'confidence': device['confidence'],
                'is_dynamic': True  # Flag to indicate this is a dynamic anchor
            })

        logger.info(f"Found {len(dynamic_anchors)} dynamic anchors with confidence >= {min_confidence}")
        return dynamic_anchors

# For convenience
detector = StaticDeviceDetector()