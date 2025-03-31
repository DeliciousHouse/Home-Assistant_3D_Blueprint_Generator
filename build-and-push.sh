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

# Build the image
echo "Building Docker image..."
docker build --platform linux/amd64 -t blueprint-generator:$VERSION .

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

# Add this to your build script before git operations
git config --global credential.helper 'store --timeout=3600'
git config --global user.name "DeliciousHouse"
git config --global user.email "brendan3394@gmail.com"

# Set up token for git operations
echo "https://DeliciousHouse:${GITHUB_TOKEN}@github.com" > ~/.git-credentials
chmod 600 ~/.git-credentials

# Add all changes to git
echo "Committing changes to git..."
git add -A
git commit -m "Update Blueprint Generator to version $VERSION"
git push origin main

echo "Successfully built, pushed and updated version $VERSION"