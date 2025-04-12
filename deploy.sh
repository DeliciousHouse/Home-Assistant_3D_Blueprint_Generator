#!/bin/bash
# filepath: /home/bkam/automations/Home-Assistant_3D_Blueprint_Generator/deploy.sh

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get current version from config.yaml
CURRENT_VERSION=$(grep 'version:' config.yaml | sed 's/.*"\([0-9.]*\)".*/\1/')
echo -e "${GREEN}Deploying version: $CURRENT_VERSION${NC}"

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
docker build -t blueprint-generator:$CURRENT_VERSION .

# Stop and remove existing container if it exists
if docker ps -a | grep -q blueprint-generator; then
    echo -e "${YELLOW}Stopping and removing existing container...${NC}"
    docker stop blueprint-generator || true
    docker rm blueprint-generator || true
fi

# Run the new container
echo -e "${YELLOW}Starting new container...${NC}"
docker run -d \
    --name blueprint-generator \
    -p 8001:8000 \
    -v "$(pwd)/data:/data" \
    -e "LOG_LEVEL=info" \
    blueprint-generator:$CURRENT_VERSION

# Check if container is running
if docker ps | grep -q blueprint-generator; then
    echo -e "${GREEN}Container started successfully!${NC}"
    echo -e "${GREEN}Blueprint Generator is available at: http://localhost:8001${NC}"
else
    echo -e "${RED}Failed to start container. Check logs with: docker logs blueprint-generator${NC}"
    exit 1
fi