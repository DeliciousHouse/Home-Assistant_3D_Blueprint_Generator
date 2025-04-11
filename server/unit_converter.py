#!/usr/bin/env python3

"""
Unit converter for the Blueprint Generator
Handles conversions between metric and imperial units
"""

import logging
from typing import Dict, List, Any, Union

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
    """Convert a position coordinate between metric and imperial"""
    if not position:
        return position

    result = position.copy()

    for key in ['x', 'y', 'z']:
        if key in result and result[key] is not None:
            if to_imperial:
                result[key] = meters_to_feet(result[key])
            else:
                result[key] = feet_to_meters(result[key])

    return result

def convert_dimensions(dimensions: Dict[str, float], to_imperial: bool = True) -> Dict[str, float]:
    """Convert room dimensions between metric and imperial"""
    if not dimensions:
        return dimensions

    result = dimensions.copy()

    for key in ['width', 'length', 'height', 'thickness']:
        if key in result and result[key] is not None:
            if to_imperial:
                result[key] = meters_to_feet(result[key])
            else:
                result[key] = feet_to_meters(result[key])

    return result

def convert_bounds(bounds: Dict[str, Dict[str, float]], to_imperial: bool = True) -> Dict[str, Dict[str, float]]:
    """Convert boundary positions between metric and imperial"""
    if not bounds:
        return bounds

    result = bounds.copy()

    for bound_type in ['min', 'max']:
        if bound_type in result:
            result[bound_type] = convert_position(result[bound_type], to_imperial)

    return result

def convert_blueprint_to_imperial(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a blueprint from metric (meters) to imperial (feet) units"""
    if not blueprint:
        logger.warning("Cannot convert empty blueprint")
        return blueprint

    # Make a deep copy to avoid modifying the original
    import copy
    imperial_blueprint = copy.deepcopy(blueprint)

    # Set the units flag
    imperial_blueprint['units'] = 'imperial'

    # Convert rooms
    if 'rooms' in imperial_blueprint and isinstance(imperial_blueprint['rooms'], list):
        for room in imperial_blueprint['rooms']:
            # Convert center coordinates
            if 'center' in room:
                room['center'] = convert_position(room['center'])

            # Convert dimensions
            if 'dimensions' in room:
                room['dimensions'] = convert_dimensions(room['dimensions'])

            # Convert bounds
            if 'bounds' in room:
                room['bounds'] = convert_bounds(room['bounds'])

    # Convert walls
    if 'walls' in imperial_blueprint and isinstance(imperial_blueprint['walls'], list):
        for wall in imperial_blueprint['walls']:
            # Convert start and end coordinates
            for point in ['start', 'end']:
                if point in wall:
                    wall[point] = convert_position(wall[point])

            # Convert thickness and height
            for dim in ['thickness', 'height']:
                if dim in wall and wall[dim] is not None:
                    wall[dim] = meters_to_feet(wall[dim])

    # Convert floors
    if 'floors' in imperial_blueprint and isinstance(imperial_blueprint['floors'], list):
        for floor in imperial_blueprint['floors']:
            if 'height' in floor and floor['height'] is not None:
                floor['height'] = meters_to_feet(floor['height'])

    logger.info(f"Converted blueprint from metric to imperial units with {len(imperial_blueprint.get('rooms', []))} rooms")
    return imperial_blueprint

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

def convert_coordinates(coords: Union[List[float], Dict[str, float]], to_imperial: bool = True) -> Union[List[float], Dict[str, float]]:
    """Convert coordinates from metric to imperial or vice versa."""
    if isinstance(coords, list):
        if to_imperial:
            return [meters_to_feet(val) for val in coords]
        else:
            return [feet_to_meters(val) for val in coords]

    elif isinstance(coords, dict):
        if to_imperial:
            return {k: meters_to_feet(v) if isinstance(v, (int, float)) else v for k, v in coords.items()}
        else:
            return {k: feet_to_meters(v) if isinstance(v, (int, float)) else v for k, v in coords.items()}

    return coords  # Return unchanged if format not recognized

def format_measurements(value: float, imperial: bool = False, precision: int = 1) -> str:
    """Format a measurement value with the appropriate unit suffix."""
    if imperial:
        return f"{round(value, precision)} ft"
    else:
        return f"{round(value, precision)} m"

def format_area(value: float, imperial: bool = False, precision: int = 1) -> str:
    """Format an area value with the appropriate unit suffix."""
    if imperial:
        return f"{round(value, precision)} sq ft"
    else:
        return f"{round(value, precision)} m²"