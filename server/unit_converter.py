#!/usr/bin/env python3
"""
Unit Converter Module for the 3D Blueprint Generator

This module provides functions to convert between metric and imperial units.
Primarily used to ensure compatibility with different measurement systems in Home Assistant.
"""

import logging
import math
from typing import Dict, List, Tuple, Union, Optional

logger = logging.getLogger(__name__)

# Common conversion factors
METERS_TO_FEET = 3.28084
FEET_TO_METERS = 0.3048
METERS_TO_INCHES = 39.3701
INCHES_TO_METERS = 0.0254
SQUARE_METERS_TO_SQUARE_FEET = 10.7639
SQUARE_FEET_TO_SQUARE_METERS = 0.092903
CM_TO_INCHES = 0.393701
INCHES_TO_CM = 2.54

def meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters * METERS_TO_FEET

def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet * FEET_TO_METERS

def meters_to_feet_inches(meters: float) -> Tuple[int, float]:
    """
    Convert meters to a tuple of (feet, inches).

    Returns:
        Tuple containing feet (integer) and inches (float)
    """
    total_inches = meters * METERS_TO_INCHES
    feet = int(total_inches // 12)
    inches = total_inches % 12
    return (feet, inches)

def feet_inches_to_meters(feet: int, inches: float) -> float:
    """
    Convert feet and inches to meters.

    Args:
        feet: Number of feet (integer part)
        inches: Number of inches (fractional part)

    Returns:
        The equivalent length in meters
    """
    total_inches = (feet * 12) + inches
    return total_inches * INCHES_TO_METERS

def square_meters_to_square_feet(square_meters: float) -> float:
    """Convert square meters to square feet."""
    return square_meters * SQUARE_METERS_TO_SQUARE_FEET

def square_feet_to_square_meters(square_feet: float) -> float:
    """Convert square feet to square meters."""
    return square_feet * SQUARE_FEET_TO_SQUARE_METERS

def cm_to_inches(cm: float) -> float:
    """Convert centimeters to inches."""
    return cm * CM_TO_INCHES

def inches_to_cm(inches: float) -> float:
    """Convert inches to centimeters."""
    return inches * INCHES_TO_CM

def format_distance_for_display(meters: float, unit_system: str = "metric") -> str:
    """
    Format a distance for display in the appropriate unit system.

    Args:
        meters: Distance in meters
        unit_system: Either 'metric' or 'imperial'

    Returns:
        Formatted string with appropriate units
    """
    if unit_system.lower() == "imperial":
        feet, inches = meters_to_feet_inches(meters)
        if inches < 0.1:  # Close enough to zero inches
            return f"{feet} ft"
        else:
            return f"{feet}' {inches:.1f}\""
    else:
        if meters >= 1:
            return f"{meters:.2f} m"
        else:
            return f"{meters * 100:.1f} cm"

def format_area_for_display(square_meters: float, unit_system: str = "metric") -> str:
    """
    Format an area measurement for display in the appropriate unit system.

    Args:
        square_meters: Area in square meters
        unit_system: Either 'metric' or 'imperial'

    Returns:
        Formatted string with appropriate units
    """
    if unit_system.lower() == "imperial":
        square_feet = square_meters_to_square_feet(square_meters)
        return f"{square_feet:.1f} sq ft"
    else:
        if square_meters >= 1:
            return f"{square_meters:.2f} m²"
        else:
            return f"{square_meters * 10000:.1f} cm²"

def convert_coordinates(coordinates: Dict[str, float],
                        source_system: str = "metric",
                        target_system: str = "imperial") -> Dict[str, float]:
    """
    Convert coordinates between metric and imperial systems.

    Args:
        coordinates: Dict with 'x', 'y', and possibly 'z' keys (in meters or feet)
        source_system: Either 'metric' or 'imperial'
        target_system: Either 'metric' or 'imperial'

    Returns:
        Dict with converted coordinates
    """
    if source_system == target_system:
        return coordinates.copy()

    result = {}

    if source_system == "metric" and target_system == "imperial":
        for key in coordinates:
            if key in ['x', 'y', 'z']:
                result[key] = meters_to_feet(coordinates[key])
            else:
                result[key] = coordinates[key]  # Preserve non-coordinate values

    elif source_system == "imperial" and target_system == "metric":
        for key in coordinates:
            if key in ['x', 'y', 'z']:
                result[key] = feet_to_meters(coordinates[key])
            else:
                result[key] = coordinates[key]  # Preserve non-coordinate values

    return result

def convert_blueprint_dimensions(blueprint: Dict,
                                target_system: str = "imperial") -> Dict:
    """
    Convert all dimensions in a blueprint to the target unit system.

    Args:
        blueprint: Blueprint data structure with rooms, walls, etc.
        target_system: Either 'metric' or 'imperial'

    Returns:
        Blueprint with converted dimensions
    """
    # Make a deep copy to avoid modifying the original
    import copy
    result = copy.deepcopy(blueprint)

    source_system = blueprint.get('unit_system', 'metric')

    # If already in target system, return as is
    if source_system == target_system:
        return result

    # Update unit system
    result['unit_system'] = target_system

    # Process room dimensions
    if 'rooms' in result:
        for room in result['rooms']:
            # Convert dimensions
            if 'dimensions' in room:
                dims = room['dimensions']

                if source_system == 'metric' and target_system == 'imperial':
                    if 'width' in dims:
                        dims['width'] = meters_to_feet(dims['width'])
                    if 'length' in dims:
                        dims['length'] = meters_to_feet(dims['length'])
                    if 'height' in dims:
                        dims['height'] = meters_to_feet(dims['height'])
                    if 'area' in dims:
                        dims['area'] = square_meters_to_square_feet(dims['area'])

                elif source_system == 'imperial' and target_system == 'metric':
                    if 'width' in dims:
                        dims['width'] = feet_to_meters(dims['width'])
                    if 'length' in dims:
                        dims['length'] = feet_to_meters(dims['length'])
                    if 'height' in dims:
                        dims['height'] = feet_to_meters(dims['height'])
                    if 'area' in dims:
                        dims['area'] = square_feet_to_square_meters(dims['area'])

            # Convert coordinates
            if 'coordinates' in room:
                room['coordinates'] = convert_coordinates(
                    room['coordinates'], source_system, target_system
                )

            # Convert walls
            if 'walls' in room:
                for wall in room['walls']:
                    if 'start' in wall:
                        wall['start'] = convert_coordinates(
                            wall['start'], source_system, target_system
                        )
                    if 'end' in wall:
                        wall['end'] = convert_coordinates(
                            wall['end'], source_system, target_system
                        )

    # Convert furniture or other elements if present
    if 'furniture' in result:
        for item in result['furniture']:
            if 'position' in item:
                item['position'] = convert_coordinates(
                    item['position'], source_system, target_system
                )

            # Convert dimensions
            if 'dimensions' in item:
                dims = item['dimensions']

                if source_system == 'metric' and target_system == 'imperial':
                    if 'width' in dims:
                        dims['width'] = meters_to_feet(dims['width'])
                    if 'length' in dims:
                        dims['length'] = meters_to_feet(dims['length'])
                    if 'height' in dims:
                        dims['height'] = meters_to_feet(dims['height'])

                elif source_system == 'imperial' and target_system == 'metric':
                    if 'width' in dims:
                        dims['width'] = feet_to_meters(dims['width'])
                    if 'length' in dims:
                        dims['length'] = feet_to_meters(dims['length'])
                    if 'height' in dims:
                        dims['height'] = feet_to_meters(dims['height'])

    logger.info(f"Converted blueprint dimensions from {source_system} to {target_system}")
    return result

def get_preferred_unit_system(config: Optional[Dict] = None) -> str:
    """
    Determine the preferred unit system based on configuration.
    Defaults to metric if not specified.

    Args:
        config: Configuration dictionary

    Returns:
        Either 'metric' or 'imperial'
    """
    if config is None:
        return "metric"

    # Check for unit system in config
    unit_system = config.get('unit_system',
                            config.get('display', {}).get('unit_system', 'metric'))

    # Normalize the value
    if unit_system.lower() in ['imperial', 'us', 'customary']:
        return 'imperial'
    else:
        return 'metric'  # Default to metric for any other value

def round_to_precision(value: float, precision: float = 0.01) -> float:
    """
    Round a value to a specified precision.

    Args:
        value: The value to round
        precision: The precision to round to (default 0.01 for 2 decimal places)

    Returns:
        Rounded value
    """
    return round(value / precision) * precision