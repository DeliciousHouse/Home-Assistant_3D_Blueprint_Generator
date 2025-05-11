FROM python:3.9-slim

# Set environment variables for Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies and nginx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    nginx \
    curl \
    jq \
    vim \
    net-tools \
    libopenblas-dev \
    libcurl4 \
    libnghttp2-14 \
    libpsl5 \
    librtmp1 \
    libssh2-1 \
    publicsuffix \
    libjq1 \
    sqlite3 \
    libonig5 \
    && rm -rf /var/lib/apt/lists/*

# Create a nginx user
RUN adduser --system --no-create-home --shell /bin/false --group --disabled-login nginx

# Create working directory
WORKDIR /opt/blueprint_generator

# Upgrade pip and install requirements
COPY requirements.txt .
RUN pip install --upgrade pip wheel setuptools
RUN pip install --no-cache-dir -r requirements.txt
# Add missing packages
RUN pip install python-json-logger google-generativeai Pillow

# Copy your application code
COPY . .

# Fix the config import issue
# RUN echo '#!/usr/bin/env python3\n\nimport json\nimport logging\nimport os\nfrom pathlib import Path\n\nlogger = logging.getLogger(__name__)\n\ndef load_config():\n    """Load configuration from file."""\n    try:\n        config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json"))\n        if config_path.exists():\n            with open(config_path, "r") as f:\n                return json.load(f)\n        return {}\n    except Exception as e:\n        logger.error(f"Failed to load config: {str(e)}")\n        return {}' > /opt/blueprint_generator/config.py

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create log directories
RUN mkdir -p /var/log/ /var/log/nginx

# Create data directory
RUN mkdir -p /data
# Explicitly set ownership to root (common for container processes)
# Or use 'nginx:nginx' if your entrypoint switches user
RUN chown -R root:root /data
# Grant write permissions broadly (adjust if security is paramount)
RUN chmod -R 777 /data
VOLUME /data

# Make scripts executable and move entrypoint
RUN chmod +x run.py docker-entrypoint.sh
RUN cp docker-entrypoint.sh /usr/local/bin/

# Expose port
EXPOSE 8000

# Set up entrypoint
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# Labels for Home Assistant
LABEL \
    io.hass.name="3D Blueprint Generator" \
    io.hass.description="Generate 3D home blueprints from Bluetooth sensor data" \
    io.hass.type="addon" \
    io.hass.version="${BUILD_VERSION}" \
    maintainer="brendan3394@gmail.com"