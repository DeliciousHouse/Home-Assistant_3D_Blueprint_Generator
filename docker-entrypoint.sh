#!/bin/bash
set -e

# Create log directories
mkdir -p /var/log/blueprint_generator /var/log/nginx

# Start nginx in the background
echo "Starting nginx..."
nginx &
sleep 1

# Start Flask application
echo "Starting Blueprint Generator at $(date)"
cd /opt/blueprint_generator
exec python3 run.py