#!/bin/bash

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration
PORT=8002  # Changed from 8001 to avoid conflict
CONTAINER_NAME="blueprint-generator-external"
VOLUME_PATH="$(pwd)/data"

# Check if the environment variable for HA token is set
if [ -n "$HA_TOKEN" ]; then
  echo -e "${GREEN}Home Assistant token detected in environment variable${NC}"
else
  echo -e "${RED}Warning: No HA_TOKEN environment variable set. Authentication with Home Assistant may fail.${NC}"
  echo -e "${YELLOW}You can set it with: export HA_TOKEN=your_long_lived_access_token${NC}"
fi

# Check if HA_URL is set
if [ -n "$HA_URL" ]; then
  echo -e "${GREEN}Home Assistant URL detected: $HA_URL${NC}"
else
  echo -e "${YELLOW}No HA_URL environment variable set. Using default: http://localhost:8123${NC}"
  echo -e "${YELLOW}You can set it with: export HA_URL=http://your_home_assistant_ip:8123${NC}"
  export HA_URL="http://localhost:8123"
fi

# Copy the deployment script to the root directory
cp "$(dirname "$0")/deploy-external.sh" "$(dirname "$0")/../" 2>/dev/null || true

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
cd "$(dirname "$0")/.."
docker build -t blueprint-generator-external:latest -f Dockerfile.external .

# Stop and remove existing container if it exists
if docker ps -a | grep -q $CONTAINER_NAME; then
    echo -e "${YELLOW}Stopping and removing existing container...${NC}"
    docker stop $CONTAINER_NAME || true
    docker rm $CONTAINER_NAME || true
fi

# Function to check if a port is available
function is_port_available() {
    local port=$1
    if netstat -tuln | grep ":$port " >/dev/null; then
        return 1
    else
        return 0
    fi
}

# Find an available port starting from the configured one
ORIGINAL_PORT=$PORT
while ! is_port_available $PORT; do
    echo -e "${YELLOW}Port $PORT is already in use, trying next port...${NC}"
    PORT=$((PORT + 1))
    if [ $PORT -gt $((ORIGINAL_PORT + 10)) ]; then
        echo -e "${RED}Could not find an available port in range $ORIGINAL_PORT-$PORT. Please free a port and try again.${NC}"
        exit 1
    fi
done

# Run the new container
echo -e "${YELLOW}Starting new container using port $PORT...${NC}"
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8001 \
    -v "$VOLUME_PATH:/data" \
    -e "LOG_LEVEL=info" \
    -e "HA_TOKEN=$HA_TOKEN" \
    -e "HA_URL=$HA_URL" \
    blueprint-generator-external:latest

# Check if container is running
if docker ps | grep -q $CONTAINER_NAME; then
    echo -e "${GREEN}Container started successfully!${NC}"
    echo -e "${GREEN}Blueprint Generator is available at: http://localhost:$PORT${NC}"
    echo -e ""
    echo -e "${YELLOW}Connection details:${NC}"
    echo -e "  Home Assistant URL: ${GREEN}$HA_URL${NC}"
    echo -e "  Web interface: ${GREEN}http://localhost:$PORT${NC}"
else
    echo -e "${RED}Failed to start container. Check logs with: docker logs $CONTAINER_NAME${NC}"
    exit 1
fi