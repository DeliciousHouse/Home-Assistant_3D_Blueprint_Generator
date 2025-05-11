# AI Image Generation with Gemini 2.0

This document explains how to use the AI image generation feature in the Home Assistant 3D Blueprint Generator using Google's Gemini 2.0 Flash Preview API.

## Setup Instructions

1. **Obtain a Google API Key**:
   - Visit [Google AI Studio](https://makersuite.google.com/) to create an account
   - Create a new API key in the API Keys section
   - Copy the API key for configuration

2. **Configure the Blueprint Generator**:
   - Edit `config/config.json` to enable AI image generation:
   ```json
   "ai_image_generation": {
     "enabled": true,
     "provider": "gemini",
     "api_key": "YOUR_API_KEY_HERE",
     "model": "gemini-2.0-flash-preview-image-generation",
     "image_size": "1024x1024",
     "quality": "high"
   }
   ```
   - Alternatively, set your API key as an environment variable:
   ```bash
   export GOOGLE_API_KEY="your_api_key_here"
   ```

3. **Install Required Python Dependencies**:
   ```bash
   pip install google-generativeai Pillow
   ```

   Or use the npm script:
   ```bash
   npm run install-ai-deps
   ```

## Usage

Once configured, the Blueprint Generator will automatically generate images for:

1. Individual rooms based on their dimensions and features
2. Floor plans showing the layout of each floor
3. An exterior view of your home

These images will be accessible in the 3D Blueprint Viewer via the "AI Images" button.

## Customizing Image Styles

You can customize the style of the generated images by setting the `default_style` in the `room_description` section of your config:

```json
"room_description": {
  "default_style": "modern",
  "style_presets": {
    "modern": "modern, clean lines, minimalist, natural light, neutral colors",
    "traditional": "traditional, classic, warm colors, detailed woodwork, cozy",
    "industrial": "industrial, exposed brick, metal accents, concrete floors, open concept",
    "scandinavian": "scandinavian, light woods, white walls, functional, hygge",
    "farmhouse": "farmhouse style, rustic elements, wooden beams, vintage accents, warm",
    "mid_century": "mid-century modern, clean lines, organic curves, bold colors",
    "coastal": "coastal, light blues, white, natural textures, airy, beachy"
  }
}
```

You can add your own style presets by extending the `style_presets` object.

## Testing the Integration

Use the provided test script to verify your Gemini API integration:

```bash
python tests/test_gemini_integration.py --api-key YOUR_API_KEY
```

This will generate a test image to confirm that the Gemini API is working correctly.

## Troubleshooting

If you encounter issues with image generation:

1. **API Key Issues**: Verify your API key is correct and has access to the Gemini Pro Vision API
2. **Import Errors**: Make sure you've installed the required dependencies (`google-generativeai` and `Pillow`)
3. **Resource Limits**: Check if you've hit API rate limits or quotas
4. **Image Storage**: Ensure the `data/generated_images` directory exists and is writable

Check the logs for specific error messages that can help diagnose issues.

## Alternative Providers

The system also supports these alternative image generation providers:

1. OpenAI DALL-E
2. Replicate API (various models)
3. Ollama (local LLaVA model)

Refer to the main documentation for details on configuring these alternative providers.
