#!/bin/bash
# build-and-push.sh

set -e # Exit on error

# Load GitHub token
if [ -f ~/.github_token ]; then
  GITHUB_TOKEN=$(cat ~/.github_token)
else
  echo "Error: GitHub token file not found at ~/.github_token"
  exit 1
fi

# Get current version from config.yaml
CURRENT_VERSION=$(grep 'version:' config.yaml | sed 's/.*"\([0-9.]*\)".*/\1/')
echo "Current version: $CURRENT_VERSION"

# Increment version by 0.01
NEW_VERSION=$(awk -v ver="$CURRENT_VERSION" 'BEGIN { printf("%.2f", ver + 0.01) }')
echo "New version: $NEW_VERSION"

# Update version in config.yaml - using more specific pattern
sed -i "s/version: \"$CURRENT_VERSION\"/version: \"$NEW_VERSION\"/" config.yaml
echo "Updated config.yaml to version $NEW_VERSION"

# Update repository.json
if [ -f repository.json ]; then
    echo "Updating repository.json..."
    # More precise pattern to update version in repository.json
    sed -i "s/\"version\": \"$CURRENT_VERSION\"/\"version\": \"$NEW_VERSION\"/" repository.json
    echo "Updated repository.json to version $NEW_VERSION"
else
    echo "Warning: repository.json not found."
fi

# Update build.yaml if it exists
if [ -f build.yaml ]; then
    echo "Updating build.yaml..."
    sed -i "s/org.opencontainers.image.version: \"$CURRENT_VERSION\"/org.opencontainers.image.version: \"$NEW_VERSION\"/" build.yaml
    echo "Updated build.yaml to version $NEW_VERSION"
else
    echo "Warning: build.yaml not found."
fi

# Set version for building
VERSION=$NEW_VERSION
echo "Building version $VERSION"

# Build the image with no cache
echo "Building Docker image (clean build)..."
docker build --no-cache --platform linux/amd64 -t blueprint-generator:$VERSION .

# Tag for GitHub
docker tag blueprint-generator:$VERSION ghcr.io/delicioushouse/blueprint-generator-amd64:$VERSION
docker tag blueprint-generator:$VERSION ghcr.io/delicioushouse/blueprint-generator-amd64:latest

# Push to GitHub Container Registry
echo "Logging in to GitHub Container Registry..."
echo "$GITHUB_TOKEN" | docker login ghcr.io -u DeliciousHouse --password-stdin

# Push images
echo "Pushing images to GitHub Container Registry..."
docker push ghcr.io/delicioushouse/blueprint-generator-amd64:$VERSION
docker push ghcr.io/delicioushouse/blueprint-generator-amd64:latest

# Add all changes to git
echo "Committing changes to git..."
git add -A
git commit -m "Update Blueprint Generator to version $VERSION"
git push origin main

echo "Successfully built, pushed and updated version $VERSION"
# Wait for GitHub to process the push
echo "Waiting 10 seconds for GitHub to process the changes..."
sleep 10

# SSH into Home Assistant and update the addon
echo "SSHing into Home Assistant to update the addons..."
HASS_PASSWORD="Xenia1031"

sshpass -p "$HASS_PASSWORD" ssh bkam@192.168.86.91 << 'EOF'
  echo "Connected to Home Assistant, updating addons..."
  cd /addons/Home-Assistant_3D_Blueprint_Generator
  git pull

  # Run addon update check - assuming you have the Home Assistant CLI available
  # If using a standard Home Assistant instance with CLI:
  ha addons check-update

  # If using a different setup, you might need:
  # ha addons rebuild local_blueprint_generator

  echo "Addon update complete!"
EOF

# End of SSH session
echo "Home Assistant addon updated successfully!"
echo "Successfully built, pushed and updated version $VERSION"