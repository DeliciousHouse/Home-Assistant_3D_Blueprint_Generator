#!/bin/bash
# build-and-push.sh - Use this script for building and publishing release Docker images
# For day-to-day development, use dev-push.sh instead

set -e # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load GitHub token
if [ -f ~/.github_token ]; then
  GITHUB_TOKEN=$(cat ~/.github_token)
else
  echo -e "${RED}Error: GitHub token file not found at ~/.github_token${NC}"
  exit 1
fi

# Get current version from config.yaml
CURRENT_VERSION=$(grep 'version:' config.yaml | sed 's/.*"\([0-9.]*\)".*/\1/')
echo -e "${YELLOW}Current version: $CURRENT_VERSION${NC}"

# Ask if the user wants to increment the version
read -p "Do you want to increment the version number? (y/n): " INCREMENT_VERSION

if [[ $INCREMENT_VERSION == "y" || $INCREMENT_VERSION == "Y" ]]; then
    # Increment version by 0.01
    NEW_VERSION=$(awk -v ver="$CURRENT_VERSION" 'BEGIN { printf("%.2f", ver + 0.01) }')
    echo -e "${GREEN}New version: $NEW_VERSION${NC}"

    # Update version in config.yaml - using more specific pattern
    sed -i "s/version: \"$CURRENT_VERSION\"/version: \"$NEW_VERSION\"/" config.yaml
    echo -e "${GREEN}Updated config.yaml to version $NEW_VERSION${NC}"

    # Update repository.json
    if [ -f repository.json ]; then
        echo "Updating repository.json..."
        # More precise pattern to update version in repository.json
        sed -i "s/\"version\": \"$CURRENT_VERSION\"/\"version\": \"$NEW_VERSION\"/" repository.json
        echo -e "${GREEN}Updated repository.json to version $NEW_VERSION${NC}"
    else
        echo -e "${YELLOW}Warning: repository.json not found.${NC}"
    fi

    # Update build.yaml if it exists
    if [ -f build.yaml ]; then
        echo "Updating build.yaml..."
        sed -i "s/org.opencontainers.image.version: \"$CURRENT_VERSION\"/org.opencontainers.image.version: \"$NEW_VERSION\"/" build.yaml
        echo -e "${GREEN}Updated build.yaml to version $NEW_VERSION${NC}"
    else
        echo -e "${YELLOW}Warning: build.yaml not found.${NC}"
    fi

    VERSION=$NEW_VERSION
else
    VERSION=$CURRENT_VERSION
    echo -e "${GREEN}Using current version: $VERSION${NC}"
fi

echo -e "${YELLOW}Building version $VERSION${NC}"

# Ask if the user wants to build and push Docker images
read -p "Do you want to build and push Docker images? (y/n): " BUILD_CHOICE

if [[ $BUILD_CHOICE == "y" || $BUILD_CHOICE == "Y" ]]; then
    # Build the image with no cache
    echo -e "${YELLOW}Building Docker image (clean build)...${NC}"
    docker build --no-cache --platform linux/amd64 -t blueprint-generator:$VERSION .

    # Tag for GitHub
    docker tag blueprint-generator:$VERSION ghcr.io/delicioushouse/blueprint-generator-amd64:$VERSION
    docker tag blueprint-generator:$VERSION ghcr.io/delicioushouse/blueprint-generator-amd64:latest

    # Push to GitHub Container Registry
    echo -e "${YELLOW}Logging in to GitHub Container Registry...${NC}"
    echo "$GITHUB_TOKEN" | docker login ghcr.io -u DeliciousHouse --password-stdin

    # Push images
    echo -e "${YELLOW}Pushing images to GitHub Container Registry...${NC}"
    docker push ghcr.io/delicioushouse/blueprint-generator-amd64:$VERSION
    docker push ghcr.io/delicioushouse/blueprint-generator-amd64:latest

    echo -e "${GREEN}Successfully built and pushed Docker images for version $VERSION${NC}"
else
    echo -e "${YELLOW}Skipping Docker build and push.${NC}"
fi

# Ask if the user wants to commit and push to git
read -p "Do you want to commit changes to git? (y/n): " COMMIT_CHOICE

if [[ $COMMIT_CHOICE == "y" || $COMMIT_CHOICE == "Y" ]]; then
    # Add all changes to git
    echo -e "${YELLOW}Committing changes to git...${NC}"
    git add -A
    git commit -m "Update Blueprint Generator to version $VERSION"

    # Configure git credentials for this push
    git config --local credential.helper "!f() { echo username=DeliciousHouse; echo password=$GITHUB_TOKEN; }; f"

    echo -e "${YELLOW}Pushing to GitHub...${NC}"
    git push origin main

    # Reset credential helper
    git config --local --unset credential.helper

    echo -e "${GREEN}Successfully committed and pushed changes to GitHub${NC}"
else
    echo -e "${YELLOW}Skipping git commit and push.${NC}"
fi

echo -e "${GREEN}Build and push process complete for version $VERSION${NC}"
echo -e "${YELLOW}Note: For development workflow, use ./dev-push.sh instead${NC}"