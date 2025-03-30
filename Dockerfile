FROM python:3.9-slim

# Install build dependencies and nginx
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libmariadb-dev \
    nginx \
    curl \
    jq \
    vim \
    net-tools \
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

# Upgrade pip and install requirements
RUN pip install protobuf==3.20.0
RUN pip install --upgrade pip wheel setuptools
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Add missing package
RUN pip install python-json-logger

# Copy your application code
COPY . /opt/blueprint_generator/
WORKDIR /opt/blueprint_generator

# Fix the config import issue
RUN echo '#!/usr/bin/env python3\n\nimport json\nimport logging\nimport os\nfrom pathlib import Path\n\nlogger = logging.getLogger(__name__)\n\ndef load_config():\n    """Load configuration from file."""\n    try:\n        config_path = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.json"))\n        if config_path.exists():\n            with open(config_path, "r") as f:\n                return json.load(f)\n        return {}\n    except Exception as e:\n        logger.error(f"Failed to load config: {str(e)}")\n        return {}' > /opt/blueprint_generator/config.py

# Create symlink to make the module findable
RUN ln -s /opt/blueprint_generator/config.py /usr/local/lib/python3.9/site-packages/config.py

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Create log directories
RUN mkdir -p /var/log/blueprint_generator /var/log/nginx
RUN mkdir -p /data
VOLUME /data

# Make scripts executable and move entrypoint
RUN chmod +x /opt/blueprint_generator/run.py /opt/blueprint_generator/docker-entrypoint.sh
RUN cp /opt/blueprint_generator/docker-entrypoint.sh /usr/local/bin/

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