#!/bin/bash
# filepath: /home/bkam/automations/Home-Assistant_3D_Blueprint_Generator/dev-push.sh

set -e # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Load GitHub token (using the same approach as build-and-push.sh)
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

echo -e "${YELLOW}Starting development push workflow...${NC}"

# Check if we have uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}Uncommitted changes detected.${NC}"

    # Show status
    git status -s

    # Ask if user wants to commit
    read -p "Do you want to commit these changes? (y/n): " COMMIT_CHOICE

    if [[ $COMMIT_CHOICE == "y" || $COMMIT_CHOICE == "Y" ]]; then
        # Ask for commit message
        read -p "Enter commit message: " COMMIT_MSG

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
echo -e "${YELLOW}Changes will be available in the repository but won't update the running container.${NC}"
echo -e "${YELLOW}To apply changes without a full rebuild:${NC}"
echo -e "  1. SSH into Home Assistant"
echo -e "  2. Run: ${GREEN}docker exec -it addon_blueprint_generator bash${NC}"
echo -e "  3. Run: ${GREEN}cd /opt/blueprint_generator && git pull${NC}"
echo -e "  4. Run: ${GREEN}exit${NC}"
echo -e "  5. Restart the add-on from Home Assistant"

exit 0