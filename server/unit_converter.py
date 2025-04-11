#!/usr/bin/env python3

"""
Unit converter for the Blueprint Generator
Handles conversions between metric and imperial units
"""

import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Conversion constants
METERS_TO_FEET = 3.28084
FEET_TO_METERS = 0.3048
SQ_METERS_TO_SQ_FEET = 10.7639
SQ_FEET_TO_SQ_METERS = 0.092903

def meters_to_feet(meters: float) -> float:
    """Convert meters to feet"""
    return meters * METERS_TO_FEET

def feet_to_meters(feet: float) -> float:
    """Convert feet to meters"""
    return feet * FEET_TO_METERS

def sq_meters_to_sq_feet(sq_meters: float) -> float:
    """Convert square meters to square feet"""
    return sq_meters * SQ_METERS_TO_SQ_FEET

def sq_feet_to_sq_meters(sq_feet: float) -> float:
    """Convert square feet to square meters"""
    return sq_feet * SQ_FEET_TO_SQ_METERS

def convert_position(position: Dict[str, float], to_imperial: bool = True) -> Dict[str, float]:
    """
    Convert a position dictionary between metric and imperial

    Args:
        position: Dictionary with x, y, z coordinates
        to_imperial: Whether to convert to imperial (True) or metric (False)

    Returns:
        Dictionary with converted coordinates
    """
    if not position or not isinstance(position, dict):
        return position

    result = position.copy()
    for key in ['x', 'y', 'z']:
        if key in result and isinstance(result[key], (int, float)):
            if to_imperial:
                result[key] = meters_to_feet(result[key])
            else:
                result[key] = feet_to_meters(result[key])

    return result

def convert_dimensions(dimensions: Dict[str, float], to_imperial: bool = True) -> Dict[str, float]:
    """
    Convert dimension dictionary between metric and imperial

    Args:
        dimensions: Dictionary with width, length, height
        to_imperial: Whether to convert to imperial (True) or metric (False)

    Returns:
        Dictionary with converted dimensions
    """
    if not dimensions or not isinstance(dimensions, dict):
        return dimensions

    result = dimensions.copy()
    for key in ['width', 'length', 'height', 'thickness']:
        if key in result and isinstance(result[key], (int, float)):
            if to_imperial:
                result[key] = meters_to_feet(result[key])
            else:
                result[key] = feet_to_meters(result[key])

    # Convert area if present
    if 'area' in result and isinstance(result['area'], (int, float)):
        if to_imperial:
            result['area'] = sq_meters_to_sq_feet(result['area'])
        else:
            result['area'] = sq_feet_to_sq_meters(result['area'])

    return result

def convert_blueprint_to_imperial(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an entire blueprint from metric to imperial units

    Args:
        blueprint: The blueprint dictionary with metric measurements

    Returns:
        The blueprint with all measurements converted to imperial units
    """
    if not blueprint or not isinstance(blueprint, dict):
        logger.warning("Invalid blueprint provided for conversion")
        return blueprint

    # Create a deep copy to avoid modifying the original
    result = blueprint.copy()

    # Mark as imperial
    result['units'] = 'imperial'

    # Convert rooms
    if 'rooms' in result and isinstance(result['rooms'], list):
        for room in result['rooms']:
            if not isinstance(room, dict):
                continue

            # Convert center position
            if 'center' in room:
                room['center'] = convert_position(room['center'])

            # Convert dimensions
            if 'dimensions' in room:
                room['dimensions'] = convert_dimensions(room['dimensions'])

            # Convert bounds
            if 'bounds' in room and isinstance(room['bounds'], dict):
                if 'min' in room['bounds']:
                    room['bounds']['min'] = convert_position(room['bounds']['min'])
                if 'max' in room['bounds']:
                    room['bounds']['max'] = convert_position(room['bounds']['max'])

    # Convert walls
    if 'walls' in result and isinstance(result['walls'], list):
        for wall in result['walls']:
            if not isinstance(wall, dict):
                continue

            # Convert start and end positions
            if 'start' in wall:
                wall['start'] = convert_position(wall['start'])
            if 'end' in wall:
                wall['end'] = convert_position(wall['end'])

            # Convert height and thickness
            for key in ['height', 'thickness']:
                if key in wall and isinstance(wall[key], (int, float)):
                    wall[key] = meters_to_feet(wall[key])

    # Convert any other fields as needed

    logger.info("Successfully converted blueprint to imperial units")
    return result

def convert_blueprint_to_metric(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert an entire blueprint from imperial to metric units

    Args:
        blueprint: The blueprint dictionary with imperial measurements

    Returns:
        The blueprint with all measurements converted to metric units
    """
    if not blueprint or not isinstance(blueprint, dict):
        logger.warning("Invalid blueprint provided for conversion")
        return blueprint

    # Create a deep copy to avoid modifying the original
    result = blueprint.copy()

    # Mark as metric
    result['units'] = 'metric'

    # Convert rooms
    if 'rooms' in result and isinstance(result['rooms'], list):
        for room in result['rooms']:
            if not isinstance(room, dict):
                continue

            # Convert center position
            if 'center' in room:
                room['center'] = convert_position(room['center'], to_imperial=False)

            # Convert dimensions
            if 'dimensions' in room:
                room['dimensions'] = convert_dimensions(room['dimensions'], to_imperial=False)

            # Convert bounds
            if 'bounds' in room and isinstance(room['bounds'], dict):
                if 'min' in room['bounds']:
                    room['bounds']['min'] = convert_position(room['bounds']['min'], to_imperial=False)
                if 'max' in room['bounds']:
                    room['bounds']['max'] = convert_position(room['bounds']['max'], to_imperial=False)

    # Convert walls
    if 'walls' in result and isinstance(result['walls'], list):
        for wall in result['walls']:
            if not isinstance(wall, dict):
                continue

            # Convert start and end positions
            if 'start' in wall:
                wall['start'] = convert_position(wall['start'], to_imperial=False)
            if 'end' in wall:
                wall['end'] = convert_position(wall['end'], to_imperial=False)

            # Convert height and thickness
            for key in ['height', 'thickness']:
                if key in wall and isinstance(wall[key], (int, float)):
                    wall[key] = feet_to_meters(wall[key])

    # Convert any other fields as needed

    logger.info("Successfully converted blueprint to metric units")
    return result