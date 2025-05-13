#!/usr/bin/env python3
"""
AI Image Generator Module

This module provides functionality to generate realistic images of rooms and
floor plans using AI image generation models.
"""

import logging
import requests
import os
import base64
import time
import json
from typing import Dict, Any, Optional, List
from pathlib import Path
from io import BytesIO

from .config_loader import load_config
from .room_description_generator import description_generator    # Add Google Gemini imports
try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger(__name__)

class AIImageGenerator:
    """
    Class for generating realistic images of rooms and floor plans using AI models.
    """

    def __init__(self):
        """Initialize the AI image generator with configuration."""
        self.config = load_config()
        self.image_gen_config = self.config.get('ai_image_generation', {})
        self.enabled = self.image_gen_config.get('enabled', False)
        self.provider = self.image_gen_config.get('provider', 'ollama')  # ollama, openai, replicate, gemini
        self.api_key = self.image_gen_config.get('api_key', os.environ.get('AI_IMAGE_API_KEY', ''))
        self.api_url = self.image_gen_config.get('api_url', 'http://localhost:11434/api/generate')  # Default for Ollama
        self.model = self.image_gen_config.get('model', 'llava')  # Default model
        self.image_size = self.image_gen_config.get('image_size', '1024x1024')
        self.quality = self.image_gen_config.get('quality', 'standard')
        # Try multiple possible paths in order of preference:
        # 1. Container volume path
        # 2. Local path relative to code
        # 3. Fallback to /tmp
        possible_paths = [
            '/data/generated_images',  # Container volume path
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'generated_images'),
            '/tmp/blueprint_generator_images'
        ]

        # Find the first writable directory
        default_output_dir = None
        for path in possible_paths:
            try:
                parent_dir = os.path.dirname(path)
                if os.path.exists(parent_dir) and os.access(parent_dir, os.W_OK):
                    default_output_dir = path
                    break
            except Exception:
                continue

        # If no writable directory found, fall back to /tmp
        if not default_output_dir:
            default_output_dir = '/tmp/blueprint_generator_images'

        # Get the directory from config or use the default
        self.output_dir = self.image_gen_config.get('output_dir', default_output_dir)

        # Initialize Google Gemini client if needed
        self.gemini_client = None
        if self.provider == 'gemini':
            self._initialize_gemini_client()

    def _initialize_gemini_client(self):
        """Initialize the Google Gemini client with error handling"""
        if not GEMINI_AVAILABLE:
            logger.error("Google Generative AI package not installed. Install with: pip install google-generativeai")
            return

        # Check all possible environment variables for the API key
        gemini_api_key = (self.api_key or
                         os.environ.get('GOOGLE_API_KEY', '') or
                         os.environ.get('AI_IMAGE_API_KEY', ''))

        if not gemini_api_key:
            logger.error("No Google API key found for Gemini image generation. "
                        "Please set it in config.json or as GOOGLE_API_KEY/AI_IMAGE_API_KEY environment variable")
            return

        try:
            import google.generativeai as genai
            # Configure the API key
            genai.configure(api_key=gemini_api_key)

            # Create client using the correct model name for image generation
            # Use minimal configuration to avoid response modality conflicts
            self.gemini_client = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")
            logger.info("Google Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            # Try one more time with a minimal approach and different model name
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                # Try the model name without the preview suffix as fallback
                self.gemini_client = genai.GenerativeModel("gemini-2.0-flash-image-generation")
                logger.info("Google Gemini client initialized with fallback model name")
            except Exception as retry_error:
                logger.error(f"Gemini client retry initialization failed: {str(retry_error)}")

        # Create output directory if it doesn't exist and we have permissions
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
                logger.info(f"Created image output directory: {self.output_dir}")
        except PermissionError:
            # If we can't create the specified directory, fall back to /tmp
            self.output_dir = '/tmp/blueprint_generator_images'
            os.makedirs(self.output_dir, exist_ok=True)
            logger.warning(f"Permission denied for original output directory, using: {self.output_dir}")

        logger.info(f"AI Image Generator initialized (enabled: {self.enabled}, provider: {self.provider}, model: {self.model})")

    def generate_room_image(self, room_data: Dict[str, Any], style: Optional[str] = None) -> Optional[str]:
        """
        Generate a realistic image of a room based on its data.

        Args:
            room_data: Dictionary containing room details (name, dimensions, furniture, etc.)
            style: Optional style preset to use (modern, traditional, etc.)

        Returns:
            Path to the generated image file or None if generation failed
        """
        if not self.enabled:
            logger.warning("AI image generation is disabled")
            return None

        # Generate description using the dedicated generator
        prompt = description_generator.generate_room_description(room_data, style)
        room_name = room_data.get('name', 'Unknown Room')
        floor = room_data.get('floor', 0)

        logger.info(f"Generating image for {room_name} with prompt: {prompt[:100]}...")

        try:
            image_path = self._call_ai_service(prompt, f"{room_name.lower().replace(' ', '_')}_floor{floor}")
            logger.info(f"Generated image for {room_name} at {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"Failed to generate image for {room_name}: {str(e)}")
            return None

    def generate_floor_plan(self, floor_data: Dict[str, Any], style: Optional[str] = None) -> Optional[str]:
        """
        Generate a realistic floor plan image based on floor data.

        Args:
            floor_data: Dictionary containing floor details (rooms, layout, etc.)
            style: Optional style preset to use (modern, traditional, etc.)

        Returns:
            Path to the generated image file or None if generation failed
        """
        if not self.enabled:
            logger.warning("AI image generation is disabled")
            return None

        # Generate description using the dedicated generator
        prompt = description_generator.generate_floor_plan_description(floor_data, style)
        floor_number = floor_data.get('floor_number', 0)
        floor_name = floor_data.get('name', description_generator._get_floor_name(floor_number))

        logger.info(f"Generating floor plan for {floor_name} with prompt: {prompt[:100]}...")

        try:
            image_path = self._call_ai_service(prompt, f"floor_plan_{floor_number}")
            logger.info(f"Generated floor plan image for {floor_name} at {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"Failed to generate floor plan image for {floor_name}: {str(e)}")
            return None

    def generate_home_exterior(self, blueprint_data: Dict[str, Any], style: Optional[str] = None) -> Optional[str]:
        """
        Generate a realistic image of the home exterior based on blueprint data.

        Args:
            blueprint_data: Dictionary containing overall blueprint details
            style: Optional style preset to use (modern, traditional, etc.)

        Returns:
            Path to the generated image file or None if generation failed
        """
        if not self.enabled:
            logger.warning("AI image generation is disabled")
            return None

        # Generate description using the dedicated generator
        prompt = description_generator.generate_home_exterior_description(blueprint_data, style)

        logger.info(f"Generating home exterior image with prompt: {prompt[:100]}...")

        try:
            image_path = self._call_ai_service(prompt, "home_exterior")
            logger.info(f"Generated home exterior image at {image_path}")
            return image_path
        except Exception as e:
            logger.error(f"Failed to generate home exterior image: {str(e)}")
            return None

    def _call_ai_service(self, prompt: str, filename_base: str) -> str:
        """
        Call the AI image generation service API.

        Args:
            prompt: Text prompt for image generation
            filename_base: Base filename for the output image

        Returns:
            Path to the generated image file
        """
        if self.provider == 'openai':
            return self._call_openai_dalle(prompt, filename_base)
        elif self.provider == 'ollama':
            return self._call_ollama(prompt, filename_base)
        elif self.provider == 'replicate':
            return self._call_replicate(prompt, filename_base)
        elif self.provider == 'local':
            return self._call_local_model(prompt, filename_base)
        elif self.provider == 'gemini':
            if self.gemini_client:
                return self._call_gemini(prompt, filename_base)
            else:
                raise ValueError("Gemini client not initialized. Install google-generativeai package and check API key.")
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")

    def _call_ollama(self, prompt: str, filename_base: str) -> str:
        """Call local Ollama instance for image generation (LLaVa, etc.)"""
        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "prompt": f"Generate a photorealistic image based on this description: {prompt}",
            "stream": False
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            raise Exception(f"Ollama API error: {response.status_code}")

        # Parse the response to find the image URL or base64 data
        # This is a placeholder - actual implementation will depend on Ollama's API
        response_data = response.json()

        # Extract image data (specifics depend on the API response format)
        # For this example, assuming a URL is returned
        image_url = response_data.get("image_url")

        if image_url:
            # Download the image
            image_response = requests.get(image_url)
            image_response.raise_for_status()

            # Save the image
            filename = f"{filename_base}_{int(time.time())}.png"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(image_response.content)

            return filepath
        else:
            raise Exception("No image data received from Ollama")

    def _call_openai_dalle(self, prompt: str, filename_base: str) -> str:
        """Call OpenAI's DALL-E API for image generation"""
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not provided")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # Parse size dimensions
        width, height = map(int, self.image_size.split('x'))

        data = {
            "model": "dall-e-3",
            "prompt": prompt,
            "n": 1,
            "size": f"{width}x{height}",
            "quality": self.quality
        }

        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
            raise Exception(f"OpenAI API error: {response.status_code}")

        response_data = response.json()
        image_url = response_data['data'][0]['url']

        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        # Save the image
        filename = f"{filename_base}_{int(time.time())}.png"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(image_response.content)

        return filepath

    def _call_replicate(self, prompt: str, filename_base: str) -> str:
        """Call Replicate API for image generation using models like Stability Diffusion"""
        if not self.api_key:
            raise ValueError("Replicate API key is required but not provided")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Token {self.api_key}"
        }

        # Parse size dimensions
        width, height = map(int, self.image_size.split('x'))

        # Use Stable Diffusion XL model on Replicate (or another appropriate model)
        data = {
            "version": "a00d0b7dcbb9c3fbb34ba87d2d5b46c56969c84a628bf778a7fdaec30b1b99c5",
            "input": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_outputs": 1,
                "guidance_scale": 7.5,
                "num_inference_steps": 50
            }
        }

        # Start the prediction
        response = requests.post(
            "https://api.replicate.com/v1/predictions",
            headers=headers,
            json=data
        )

        if response.status_code != 201:
            logger.error(f"Replicate API error: {response.status_code} - {response.text}")
            raise Exception(f"Replicate API error: {response.status_code}")

        prediction = response.json()
        prediction_id = prediction["id"]

        # Poll for completion
        while True:
            response = requests.get(
                f"https://api.replicate.com/v1/predictions/{prediction_id}",
                headers=headers
            )

            prediction = response.json()
            if prediction["status"] == "succeeded":
                break
            elif prediction["status"] == "failed":
                raise Exception("Replicate prediction failed")

            time.sleep(1)

        # Get the output image URL
        image_url = prediction["output"][0]

        # Download the image
        image_response = requests.get(image_url)
        image_response.raise_for_status()

        # Save the image
        filename = f"{filename_base}_{int(time.time())}.png"
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'wb') as f:
            f.write(image_response.content)

        return filepath

    def _call_local_model(self, prompt: str, filename_base: str) -> str:
        """
        Call a locally running model server (e.g., via ComfyUI, InvokeAI, etc.)
        This is a placeholder implementation - actual implementation will depend on your local setup
        """
        # Local model server URL (example for ComfyUI running on default port)
        local_url = self.image_gen_config.get('local_url', 'http://127.0.0.1:8188/api')

        # This is a simplified example - actual implementation will depend on the local API
        headers = {
            "Content-Type": "application/json"
        }

        # Parse size dimensions
        width, height = map(int, self.image_size.split('x'))

        data = {
            "prompt": prompt,
            "width": width,
            "height": height
        }

        response = requests.post(
            f"{local_url}/generate",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            logger.error(f"Local model API error: {response.status_code} - {response.text}")
            raise Exception(f"Local model API error: {response.status_code}")

        # Extract image data (specifics depend on the API response format)
        response_data = response.json()
        image_data = response_data.get("image_data")

        if image_data:
            # Save the image
            filename = f"{filename_base}_{int(time.time())}.png"
            filepath = os.path.join(self.output_dir, filename)

            # Decode base64 image data and save
            with open(filepath, 'wb') as f:
                f.write(base64.b64decode(image_data))

            return filepath
        else:
            raise Exception("No image data received from local model")

    def _call_gemini(self, prompt: str, filename_base: str) -> str:
        """Call Google Gemini model to generate an image."""
        if not self.gemini_client:
            logger.error("Gemini client not initialized")
            raise Exception("Gemini client not initialized")

        try:
            import google.generativeai as genai

            logger.info(f"Calling Gemini API with prompt: {prompt[:100]}...")

            # Use the initialized client directly without additional parameters
            # This is to avoid the "The requested combination of response modalities is not supported" error
            response = self.gemini_client.generate_content(prompt)

            # Check if we got a valid response
            if not response:
                logger.error("Gemini API returned an empty response")
                raise Exception("Empty response from Gemini API")

            # Debug response structure
            logger.debug(f"Response type: {type(response)}")

            # Different versions of the API have different response structures
            # Handle both the old and new response formats
            if hasattr(response, 'parts'):
                parts = response.parts
            elif hasattr(response, 'candidates') and response.candidates:
                parts = []
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        parts.extend(candidate.content.parts)
            else:
                logger.error("Gemini API returned an invalid response structure")
                raise Exception("Invalid response format from Gemini API")

            # Find the image part in the response using the variable 'parts' we defined above
            image_data = None
            mime_type = None

            # First try the standard image attribute
            for part in parts:
                if hasattr(part, 'image') and part.image:
                    image_data = part.image.to_bytes()
                    mime_type = 'image/png'  # Assume PNG for direct image objects
                    break

                # Alternative format with inline_data (used in some versions)
                if hasattr(part, 'inline_data') and part.inline_data:
                    inline_data = part.inline_data
                    if hasattr(inline_data, 'mime_type') and inline_data.mime_type.startswith('image/'):
                        mime_type = inline_data.mime_type
                        image_data = base64.b64decode(inline_data.data)
                        break

            if not image_data:
                logger.error("No image found in Gemini API response")
                raise Exception("No image found in Gemini API response")

            # Generate a filename with timestamp to avoid collisions
            extension = 'png' if mime_type == 'image/png' else 'jpg'
            filename = f"{filename_base}_{int(time.time())}.{extension}"
            full_path = os.path.join(self.output_dir, filename)

            # Write the image data to a file
            with open(full_path, 'wb') as f:
                f.write(image_data)

            logger.info(f"Saved image to {full_path}")
            return full_path  # Return full path rather than just filename for better usability

        except Exception as e:
            error_message = str(e)
            logger.error(f"Gemini API error: {error_message}")

            # Check for specific error messages and provide more helpful information
            if "response modalities" in error_message:
                logger.error("This is likely due to the Gemini model configuration not supporting the requested output format.")
                logger.error("Try updating the google-generativeai package to the latest version: pip install -U google-generativeai")
                raise Exception("Gemini API modality error: The model doesn't support the requested output format. Update the google-generativeai package.")
            elif "API key" in error_message or "authentication" in error_message.lower():
                logger.error("This is likely due to an invalid API key. Check that your GOOGLE_API_KEY is valid and has access to Gemini.")
                raise Exception("Gemini API authentication error: Check your API key.")
            else:
                raise Exception(f"Gemini API error: {error_message}")

# Singleton instance for convenience
image_generator = AIImageGenerator()
