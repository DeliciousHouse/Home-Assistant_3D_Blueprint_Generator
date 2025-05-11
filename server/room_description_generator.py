#!/usr/bin/env python3
"""
Room Description Generator Module

This module generates detailed room and floor descriptions based on blueprint data,
which can then be used as prompts for AI image generation models.
"""

import logging
from typing import Dict, List, Any, Optional
import random
import json

from .config_loader import load_config

logger = logging.getLogger(__name__)

class RoomDescriptionGenerator:
    """
    Class for generating detailed textual descriptions of rooms and floor layouts
    that can be used as prompts for AI image generation models.
    """

    def __init__(self):
        """Initialize the room description generator with configuration."""
        self.config = load_config()
        self.description_config = self.config.get('room_description', {})

        # Load style presets
        self.style_presets = self.description_config.get('style_presets', {
            'modern': "modern, clean lines, minimalist, natural light, neutral colors",
            'traditional': "traditional, classic, warm colors, detailed woodwork, cozy",
            'industrial': "industrial, exposed brick, metal accents, concrete floors, open concept",
            'scandinavian': "scandinavian, light woods, white walls, functional, hygge",
            'farmhouse': "farmhouse style, rustic elements, wooden beams, vintage accents, warm",
            'mid_century': "mid-century modern, clean lines, organic curves, bold colors",
            'coastal': "coastal, light blues, white, natural textures, airy, beachy"
        })

        # Default style if not specified
        self.default_style = self.description_config.get('default_style', 'modern')

        # Material descriptions for different room types
        self.room_materials = {
            'kitchen': [
                "granite countertops", "marble countertops", "quartz countertops",
                "stainless steel appliances", "hardwood floors", "tile floors",
                "white cabinets", "dark wood cabinets", "glass cabinet doors",
                "subway tile backsplash", "mosaic tile backsplash"
            ],
            'living_room': [
                "hardwood floors", "area rug", "carpeted floors",
                "leather sofa", "fabric sofa", "coffee table", "bookshelves",
                "fireplace", "large windows", "recessed lighting"
            ],
            'bedroom': [
                "carpeted floors", "hardwood floors", "queen bed", "king bed",
                "nightstands", "reading lamp", "dresser", "walk-in closet",
                "blackout curtains", "soft lighting"
            ],
            'bathroom': [
                "tile floors", "marble countertop", "double vanity", "single vanity",
                "glass shower", "shower/tub combo", "freestanding tub", "toilet",
                "subway tile", "mosaic tile"
            ],
            'dining_room': [
                "hardwood floors", "dining table", "chairs", "chandelier",
                "china cabinet", "buffet", "large windows", "wainscoting"
            ],
            'office': [
                "desk", "office chair", "bookshelves", "hardwood floors",
                "area rug", "filing cabinet", "desk lamp", "computer setup"
            ],
            'hallway': [
                "hardwood floors", "runner rug", "wall art", "console table",
                "pendant light", "recessed lighting"
            ],
            'staircase': [
                "wooden stairs", "carpet runner", "handrail", "balusters",
                "newel post", "landing", "recessed lighting"
            ],
            'laundry_room': [
                "tile floors", "washer", "dryer", "utility sink",
                "storage cabinets", "folding counter"
            ],
            'entryway': [
                "tile floors", "hardwood floors", "console table", "coat rack",
                "bench", "mirror", "pendant light", "area rug"
            ],
            'garage': [
                "concrete floor", "garage door", "storage shelves",
                "workbench", "tool storage", "floor drain"
            ]
        }

        # Default materials for generic rooms
        self.default_materials = [
            "hardwood floors", "recessed lighting", "neutral wall color",
            "windows with natural light", "baseboards", "crown molding"
        ]

        # Lighting descriptions
        self.lighting_options = [
            "filled with natural light", "well-lit with recessed lighting",
            "illuminated by a stylish pendant light", "warmly lit with wall sconces",
            "bright and airy", "with ambient lighting", "with task lighting",
            "with accent lighting highlighting key features"
        ]

        logger.info("Room Description Generator initialized")

    def generate_room_description(self, room_data: Dict[str, Any], style: Optional[str] = None) -> str:
        """
        Generate a detailed textual description of a room based on its data.

        Args:
            room_data: Dictionary containing room details (name, dimensions, furniture, etc.)
            style: Optional style preset to use (modern, traditional, etc.)

        Returns:
            Detailed textual description of the room for use as an image generation prompt
        """
        room_name = room_data.get('name', 'Room')
        room_type = self._detect_room_type(room_name)
        dimensions = room_data.get('dimensions', {})

        width = dimensions.get('width', 0)
        length = dimensions.get('length', 0)
        height = dimensions.get('height', 2.4)  # Default ceiling height

        # Get the room style
        selected_style = style or self.default_style
        style_description = self.style_presets.get(selected_style, self.style_presets['modern'])

        # Get appropriate materials for this room type
        materials = self.room_materials.get(room_type, self.default_materials)
        selected_materials = random.sample(materials, min(3, len(materials)))
        materials_desc = ", ".join(selected_materials)

        # Select a lighting description
        lighting = random.choice(self.lighting_options)

        # Area calculation
        area = width * length
        size_descriptor = "small" if area < 10 else "medium-sized" if area < 20 else "spacious"

        # Build the basic description
        description = f"A {size_descriptor} {room_name.lower()} in {style_description} style, "
        description += f"approximately {width:.1f} meters wide by {length:.1f} meters long "
        description += f"with a ceiling height of {height:.1f} meters. "
        description += f"Features include {materials_desc}. "
        description += f"The room is {lighting}. "

        # Add furniture details if available
        if 'furniture' in room_data and room_data['furniture']:
            furniture_list = ", ".join([f"{item.get('name', 'item')}" for item in room_data['furniture'][:5]])
            description += f"The room contains {furniture_list}. "

        # Add a perspective description for the AI to render
        description += "This is a photorealistic 3D rendering shown from a corner perspective "
        description += "to give depth and spatial awareness to the room."

        return description

    def generate_floor_plan_description(self, floor_data: Dict[str, Any], style: Optional[str] = None) -> str:
        """
        Generate a detailed textual description of a floor plan based on its data.

        Args:
            floor_data: Dictionary containing floor details (rooms, layout, etc.)
            style: Optional style preset to use (modern, traditional, etc.)

        Returns:
            Detailed textual description of the floor plan for use as an image generation prompt
        """
        floor_number = floor_data.get('floor_number', 0)
        floor_name = floor_data.get('name', self._get_floor_name(floor_number))
        rooms = floor_data.get('rooms', [])

        # Get the floor style
        selected_style = style or self.default_style
        style_description = self.style_presets.get(selected_style, self.style_presets['modern'])

        # Basic floor description
        description = f"A detailed architectural floor plan for the {floor_name.lower()} of a home in {style_description} style. "

        # Add room details
        if rooms:
            room_count = len(rooms)
            description += f"This floor contains {room_count} rooms including "
            room_names = [room.get('name', 'room') for room in rooms[:5]]
            if len(rooms) > 5:
                room_names_text = f"{', '.join(room_names[:-1])}, and others"
            else:
                room_names_text = f"{', '.join(room_names[:-1])}, and {room_names[-1]}" if len(room_names) > 1 else room_names[0]
            description += f"{room_names_text}. "

        # Add floor area if available
        total_area = sum(room.get('dimensions', {}).get('width', 0) * room.get('dimensions', {}).get('length', 0) for room in rooms)
        if total_area > 0:
            description += f"The total floor area is approximately {total_area:.1f} square meters. "

        # Visualization instructions for the AI
        description += "This should be a top-down architectural floor plan with walls, doorways, and "
        description += "room labels clearly marked. Show wall thickness, window placements, and include "
        description += "basic furniture outlines to indicate room functions. "
        description += "Use a clean, professional architectural style with subtle shadows for depth."

        return description

    def generate_home_exterior_description(self, blueprint_data: Dict[str, Any], style: Optional[str] = None) -> str:
        """
        Generate a detailed textual description of a home's exterior based on blueprint data.

        Args:
            blueprint_data: Dictionary containing overall blueprint details
            style: Optional style preset to use (modern, traditional, etc.)

        Returns:
            Detailed textual description of the home exterior for use as an image generation prompt
        """
        # Get the house style
        selected_style = style or self.default_style
        style_description = self.style_presets.get(selected_style, self.style_presets['modern'])

        # Count floors and get total area
        floors = blueprint_data.get('floors', [])
        floor_count = len(floors)

        # Calculate approximate total area
        total_area = 0
        for floor in floors:
            floor_area = sum(room.get('dimensions', {}).get('width', 0) *
                            room.get('dimensions', {}).get('length', 0)
                            for room in floor.get('rooms', []))
            total_area += floor_area

        # Size descriptor based on area
        size_descriptor = "small" if total_area < 100 else "medium-sized" if total_area < 200 else "large"

        # Build the exterior description
        description = f"A {size_descriptor} {floor_count}-story home in {style_description} style. "

        # Add architectural features based on style
        if selected_style == 'modern':
            description += "Features clean lines, large windows, flat or low-pitched roof, "
            description += "and a minimalist facade with a mix of materials like concrete, glass, and wood. "
        elif selected_style == 'traditional':
            description += "Features symmetrical windows, covered entry, pitched roof with shingles, "
            description += "brick or wood siding exterior with classic detailing and trim. "
        elif selected_style == 'farmhouse':
            description += "Features a front porch, metal roof, board and batten siding, "
            description += "large windows, and rustic wooden accents. "
        elif selected_style == 'industrial':
            description += "Features exposed structural elements, large metal-framed windows, "
            description += "a mix of concrete, metal, and reclaimed brick with minimal ornamentation. "
        else:
            description += "Features an attractive exterior with well-designed architectural elements "
            description += "consistent with contemporary home design, including tasteful landscaping. "

        # Add visualization instructions
        description += "This is a photorealistic 3D rendering of the home's exterior from the front, "
        description += "showing the complete facade with architectural details, entry, and surrounding landscaping. "
        description += "The image should have natural lighting with subtle shadows to highlight the architectural features."

        return description

    def _detect_room_type(self, room_name: str) -> str:
        """
        Detect the room type from the room name.

        Args:
            room_name: Name of the room

        Returns:
            Standardized room type identifier
        """
        room_name_lower = room_name.lower()

        if any(keyword in room_name_lower for keyword in ['kitchen', 'cook']):
            return 'kitchen'
        elif any(keyword in room_name_lower for keyword in ['living', 'lounge', 'family']):
            return 'living_room'
        elif any(keyword in room_name_lower for keyword in ['bed', 'master', 'guest room']):
            return 'bedroom'
        elif any(keyword in room_name_lower for keyword in ['bath', 'shower', 'toilet', 'wc']):
            return 'bathroom'
        elif any(keyword in room_name_lower for keyword in ['dining', 'dinner']):
            return 'dining_room'
        elif any(keyword in room_name_lower for keyword in ['office', 'study', 'work']):
            return 'office'
        elif any(keyword in room_name_lower for keyword in ['hall', 'corridor', 'passage']):
            return 'hallway'
        elif any(keyword in room_name_lower for keyword in ['stair', 'steps']):
            return 'staircase'
        elif any(keyword in room_name_lower for keyword in ['laundry', 'utility']):
            return 'laundry_room'
        elif any(keyword in room_name_lower for keyword in ['entry', 'foyer', 'vestibule']):
            return 'entryway'
        elif any(keyword in room_name_lower for keyword in ['garage', 'car']):
            return 'garage'
        else:
            return 'generic'

    def _get_floor_name(self, floor_number: int) -> str:
        """
        Get a standardized name for a floor based on its number.

        Args:
            floor_number: The floor number (0 = ground floor, negative = basement)

        Returns:
            Standardized floor name
        """
        if floor_number == 0:
            return "Ground Floor"
        elif floor_number > 0:
            if floor_number == 1:
                return "First Floor"
            elif floor_number == 2:
                return "Second Floor"
            else:
                return f"{floor_number}th Floor"
        else:
            if floor_number == -1:
                return "Basement"
            else:
                return f"Basement {abs(floor_number)}"

# Singleton instance for convenience
description_generator = RoomDescriptionGenerator()
