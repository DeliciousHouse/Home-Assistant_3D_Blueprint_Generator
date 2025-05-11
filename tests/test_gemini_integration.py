#!/usr/bin/env python3
"""
Test script for Google Gemini image generation integration.
This script tests the AI image generation functionality using Google Gemini.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.ai_image_generator import AIImageGenerator
from server.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gemini_image_generation(api_key=None, prompt=None, output_dir=None):
    """Test Gemini image generation with specified parameters."""
    # Use provided API key or the one from the environment
    api_key = api_key or os.environ.get('GOOGLE_API_KEY', 'AIzaSyCL_0VcKYtGzYI-KQYbRvBPL4bp3VtbxGM')

    # Create test configuration - use /tmp directory by default for testing
    output_dir = output_dir or '/tmp/blueprint_generator_images'

    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Using output directory: {output_dir}")
    except PermissionError:
        # If we can't create the specified directory, fall back to /tmp
        output_dir = '/tmp/blueprint_generator_images'
        os.makedirs(output_dir, exist_ok=True)
        logger.warning(f"Permission denied for original output directory, using: {output_dir}")

    # Create instance with test configuration
    generator = AIImageGenerator()

    # Override config with command line parameters
    if api_key:
        generator.api_key = api_key
        # Re-initialize Gemini client with new API key
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            generator.gemini_client = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")
            logger.info("Re-initialized Google Gemini client with provided API key")
        except Exception as e:
            logger.error(f"Failed to initialize Google Gemini client: {str(e)}")
            return False

    # Set provider to Gemini
    generator.provider = "gemini"
    generator.enabled = True

    # Use default prompt if none provided
    test_prompt = prompt or "A modern living room with large windows, hardwood floors, a comfortable gray sofa, and minimalist decor. Natural light streaming in."

    try:
        logger.info(f"Testing Gemini image generation with prompt: {test_prompt}")
        image_path = generator._call_gemini(test_prompt, "gemini_test")
        logger.info(f"Test successful! Image generated at: {image_path}")
        return True
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        return False

def main():
    """Main function to run the test."""
    parser = argparse.ArgumentParser(description="Test Google Gemini image generation")
    parser.add_argument("--api-key", type=str, help="Google API key for Gemini")
    parser.add_argument("--prompt", type=str, help="Test prompt for image generation")
    parser.add_argument("--output-dir", type=str, help="Output directory for generated images")
    args = parser.parse_args()

    # Use environment variable if no API key provided
    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("AI_IMAGE_API_KEY")

    if not api_key:
        logger.error("No API key provided. Please provide an API key using --api-key or set GOOGLE_API_KEY environment variable.")
        return 1

    success = test_gemini_image_generation(api_key, args.prompt, args.output_dir)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
