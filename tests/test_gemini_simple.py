#!/usr/bin/env python3
"""
Simple test script for Gemini image generation.
This is a stripped-down version to test just the core functionality.
"""

import os
import sys
import logging
from pathlib import Path
import base64
from io import BytesIO

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gemini_direct():
    """Test Gemini image generation directly without any complex configuration."""
    # Use API key from environment or hardcode for testing
    api_key = os.environ.get('GOOGLE_API_KEY', 'AIzaSyCL_0VcKYtGzYI-KQYbRvBPL4bp3VtbxGM')

    try:
        import google.generativeai as genai
        from PIL import Image
    except ImportError:
        logger.error("Required packages not installed. Install with: pip install google-generativeai Pillow")
        return False

    # Simple prompt for testing
    prompt = "A modern living room with large windows and minimalist decor"

    try:
        # Configure Gemini
        genai.configure(api_key=api_key)

        # Create model with minimal configuration
        model = genai.GenerativeModel("gemini-2.0-flash-preview-image-generation")

        # Generate image with just the prompt
        logger.info(f"Generating image with prompt: '{prompt}'")
        response = model.generate_content(prompt)

        # Check for image in response
        image_data = None
        if hasattr(response, 'candidates') and response.candidates:
            for candidate in response.candidates:
                if hasattr(candidate, 'content') and candidate.content:
                    for part in candidate.content.parts:
                        if hasattr(part, 'inline_data') and part.inline_data:
                            mime_type = getattr(part.inline_data, 'mime_type', '')
                            if mime_type and mime_type.startswith('image/'):
                                image_data = part.inline_data.data
                                logger.info(f"Found image data with MIME type: {mime_type}")
                                break

        if not image_data:
            logger.error("No image data found in response")
            logger.info(f"Response structure: {dir(response)}")
            if hasattr(response, 'candidates'):
                logger.info(f"First candidate structure: {dir(response.candidates[0])}")
            return False

        # Save image to /tmp for inspection
        output_path = "/tmp/gemini_test_image.png"
        image_bytes = base64.b64decode(image_data)
        with open(output_path, 'wb') as f:
            f.write(image_bytes)

        logger.info(f"Image saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Test failed with exception: {str(e)}")
        return False

if __name__ == "__main__":
    if test_gemini_direct():
        print("Test successful! Gemini image generation is working correctly.")
        sys.exit(0)
    else:
        print("Test failed. Gemini image generation has issues.")
        sys.exit(1)
