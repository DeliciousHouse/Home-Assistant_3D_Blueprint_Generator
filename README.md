# Home Assistant 3D Blueprint Generator

Generate dynamic 3D home blueprints from Bluetooth sensor data in Home Assistant. This add-on processes Bluetooth signal strengths to create and maintain an accurate spatial map of your home.

[![Open your Home Assistant instance and show the add add-on repository dialog with a specific repository URL pre-filled.](https://my.home-assistant.io/badges/supervisor_add_addon_repository.svg)](https://my.home-assistant.io/redirect/supervisor_add_addon_repository/?repository_url=https%3A%2F%2Fgithub.com%2FDeliciousHouse%2FHome-Assistant_3D_Blueprint_Generator)

## Features

- **Dynamic Blueprint Generation**: Automatically creates 3D home blueprints from Bluetooth sensor data
- **Real-time Processing**: Continuously updates spatial mapping based on sensor readings
- **Interactive UI**: Modern web interface for viewing and adjusting blueprints
- **Local Processing**: All data processing happens locally for maximum privacy
- **Home Assistant Integration**: Direct integration with Home Assistant's Bluetooth devices
- **Manual Adjustments**: Interface for fine-tuning room layouts and dimensions
- **AI Image Generation**: Creates realistic room and floor plan images using AI
  - **Google Gemini 2.0 Image Generation**: Generate photorealistic room, floor plan, and exterior visualizations
  - **Multiple Provider Support**: Options for OpenAI DALL-E, Replicate, and local models
  - **Style Customization**: Generate images in different interior design styles

## Quick Start

1. Click the "Add to Home Assistant" button above
2. Click "Install" in the Home Assistant Add-on Store
3. Configure the required settings
4. Start the add-on
5. Click "OPEN WEB UI" to access the interface

## Documentation

- [Installation Guide](blueprint_generator/DOCS.md#installation)
- [Configuration](blueprint_generator/DOCS.md#configuration)
- [Usage Guide](blueprint_generator/DOCS.md#usage)
- [AI Image Generation](docs/ai_image_generation.md)
- [Troubleshooting](blueprint_generator/DOCS.md#troubleshooting)
- [Contributing](CONTRIBUTING.md)

## AI Image Generation

The Blueprint Generator can create photorealistic visualizations of your home using AI image generation:

- **Room Visualizations**: See how each room could look based on its dimensions and purpose
- **Floor Plan Images**: Generate visual floor plans for each level of your home
- **Exterior Views**: Create exterior visualizations of your home
- **Style Options**: Choose from modern, traditional, industrial, scandinavian, farmhouse, and other interior design styles

To configure AI image generation, see the [dedicated documentation](docs/ai_image_generation.md).

## Support

- [Open an issue](https://github.com/DeliciousHouse/Home-Assistant_3D_Blueprint_Generator/issues)
- [Discord Chat](https://discord.gg/c5DvZ4e)
- [Home Assistant Community](https://community.home-assistant.io)

## Development

### Prerequisites

- Docker
- Python 3.9 or higher
- Git

### Local Testing

```bash
cd blueprint_generator
./test_locally.sh
```

### Running Tests

```bash
pytest                 # Run all tests
pytest -v             # Verbose output
pytest --cov         # With coverage report
```

## AI Image Generation

The 3D Blueprint Generator now includes AI-powered image generation capabilities that create realistic visualizations of:

1. Individual rooms based on their dimensions and features
2. Floor plans with accurate room layouts
3. Exterior views of your home

### Supported AI Providers

- **Google Gemini** (Default) - Using the new Gemini 2.0 Flash Preview Image Generation model
- **OpenAI DALL-E** - Compatible with DALL-E 3 for photorealistic room images
- **Replicate** - Supports various models like Stable Diffusion XL
- **Ollama** - For local image generation using models like LLaVA

### Configuration

AI image generation is disabled by default. To enable it, update your `config.json`:

```json
{
  "ai_image_generation": {
    "enabled": true,
    "provider": "gemini",
    "api_key": "YOUR_API_KEY",
    "model": "gemini-2.0-flash-preview-image-generation",
    "image_size": "1024x1024",
    "quality": "high"
  },
  "room_description": {
    "default_style": "modern"
  }
}
```

Available style presets include: modern, traditional, industrial, scandinavian, farmhouse, mid_century, and coastal.

### API Keys

To use the AI image generation feature, you'll need to provide an API key for your chosen provider:

- **Google Gemini**: Obtain from [Google AI Studio](https://makersuite.google.com/)
- **OpenAI**: Create at [OpenAI Platform](https://platform.openai.com/)
- **Replicate**: Get from [Replicate](https://replicate.com/)

For better security, you can set the API key as an environment variable `GOOGLE_API_KEY` or `AI_IMAGE_API_KEY` instead of putting it directly in the config file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Home Assistant community
- Three.js for 3D rendering
- Contributors and testers

## Security

- All data processed locally
- No external API calls
- Database access restricted
- Input validation on all endpoints
- Secure password storage
