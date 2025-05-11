#!/usr/bin/env python3
"""
Setup directories for the Home Assistant 3D Blueprint Generator
This script creates necessary directories for the application
"""

import os
import sys

def setup_directories():
    """Create necessary directories for the application."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dirs_to_create = [
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "data", "generated_images"),
    ]

    for directory in dirs_to_create:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
            else:
                print(f"Directory already exists: {directory}")
        except Exception as e:
            print(f"Error creating directory {directory}: {str(e)}", file=sys.stderr)
            return False

    return True

if __name__ == "__main__":
    if setup_directories():
        print("Directory setup completed successfully")
    else:
        print("Failed to set up directories", file=sys.stderr)
        sys.exit(1)
