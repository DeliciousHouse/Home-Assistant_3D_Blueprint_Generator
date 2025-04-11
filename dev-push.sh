#!/bin/bash
# filepath: /home/bkam/automations/Home-Assistant_3D_Blueprint_Generator/dev-push.sh

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

# Configuration
REPO_PATH=$(pwd)
REMOTE_NAME=${1:-origin}
BRANCH_NAME=${2:-main}
HA_HOST=${HA_HOST:-"192.168.86.91"}  # Default HA host - change or set env var
HA_USER=${HA_USER:-"bkam"}           # Default HA user - change or set env var
HA_ADDON_PATH=${HA_ADDON_PATH:-"/addons/Home-Assistant_3D_Blueprint_Generator"}  # Default addon path on HA

echo -e "${YELLOW}Starting development push workflow...${NC}"

# Check if user wants to update version
read -p "Do you want to update the version number? (y/n): " VERSION_CHOICE

if [[ $VERSION_CHOICE == "y" || $VERSION_CHOICE == "Y" ]]; then
    # Get current version from config.yaml
    CURRENT_VERSION=$(grep 'version:' config.yaml | sed 's/.*"\([0-9.]*\)".*/\1/')
    echo -e "${YELLOW}Current version: $CURRENT_VERSION${NC}"

    # Increment version by 0.01
    NEW_VERSION=$(awk -v ver="$CURRENT_VERSION" 'BEGIN { printf("%.2f", ver + 0.01) }')
    echo -e "${GREEN}New version: $NEW_VERSION${NC}"

    # Update version in config.yaml
    sed -i "s/version: \"$CURRENT_VERSION\"/version: \"$NEW_VERSION\"/" config.yaml
    echo -e "${GREEN}Updated config.yaml to version $NEW_VERSION${NC}"

    # Update repository.json
    if [ -f repository.json ]; then
        echo "Updating repository.json..."
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

    echo -e "${GREEN}Version updated successfully.${NC}"
fi

# Check if we have uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}Uncommitted changes detected.${NC}"

    # Show status
    git status -s

    # Ask if user wants to commit
    read -p "Do you want to commit these changes? (y/n): " COMMIT_CHOICE

    if [[ $COMMIT_CHOICE == "y" || $COMMIT_CHOICE == "Y" ]]; then
        # If we updated the version, suggest a commit message
        DEFAULT_MSG=""
        if [[ $VERSION_CHOICE == "y" || $VERSION_CHOICE == "Y" ]]; then
            DEFAULT_MSG="Update to version $NEW_VERSION"
        fi

        # Ask for commit message
        read -p "Enter commit message [$DEFAULT_MSG]: " COMMIT_MSG
        COMMIT_MSG=${COMMIT_MSG:-$DEFAULT_MSG}

        # Add and commit changes
        git add .
        git commit -m "$COMMIT_MSG"
        echo -e "${GREEN}Changes committed.${NC}"
    else
        echo -e "${YELLOW}Skipping commit.${NC}"
        exit 0
    fi
fi

# Configure git credentials for this push
git config --local credential.helper "!f() { echo username=DeliciousHouse; echo password=$GITHUB_TOKEN; }; f"

# Push changes
echo -e "${YELLOW}Pushing changes to $REMOTE_NAME/$BRANCH_NAME...${NC}"
git push $REMOTE_NAME $BRANCH_NAME

# Reset credential helper
git config --local --unset credential.helper

echo -e "${GREEN}Push complete.${NC}"

# Ask if the user wants to deploy to Home Assistant
read -p "Do you want to deploy these changes to Home Assistant? (y/n): " DEPLOY_CHOICE

if [[ $DEPLOY_CHOICE == "y" || $DEPLOY_CHOICE == "Y" ]]; then
    echo -e "${YELLOW}Deploying to Home Assistant at $HA_HOST...${NC}"

    # Ask for password or use SSH key authentication
    read -s -p "Enter SSH password for $HA_USER@$HA_HOST (leave empty for key auth): " HA_PASSWORD
    echo ""

    if [ -n "$HA_PASSWORD" ]; then
        # Using password authentication
        if ! command -v sshpass &> /dev/null; then
            echo -e "${RED}Error: sshpass not installed. Please install it or use SSH key authentication.${NC}"
            exit 1
        fi
        SSH_CMD="sshpass -p '$HA_PASSWORD' ssh -o StrictHostKeyChecking=no $HA_USER@$HA_HOST"
    else
        # Using key authentication
        SSH_CMD="ssh -o StrictHostKeyChecking=no $HA_USER@$HA_HOST"
    fi

    # Execute commands on Home Assistant
    $SSH_CMD << EOF
        echo "Connected to Home Assistant, updating add-on..."
        cd $HA_ADDON_PATH
        git pull

        echo "Deployment complete!"
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully deployed to Home Assistant!${NC}"
    else
        echo -e "${RED}Failed to deploy to Home Assistant. Check SSH connection and try again.${NC}"
    fi
else
    echo -e "${YELLOW}Skipping deployment to Home Assistant.${NC}"
    echo -e "${YELLOW}To apply changes without a full rebuild:${NC}"
    echo -e "  1. SSH into Home Assistant"
    echo -e "  2. Run: ${GREEN}cd $HA_ADDON_PATH && git pull${NC}"
    echo -e "  3. Restart the add-on from Home Assistant"
fi

exit 0