#!/usr/bin/env python3
# filepath: /home/bkam/automations/Home-Assistant_3D_Blueprint_Generator/dependency_checker.py

import os
import re
import sys
import importlib
import subprocess
from pathlib import Path

# ANSI colors for output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
ENDC = '\033[0m'

def get_project_files(directory='.'):
    """Get all Python files in the project."""
    python_files = []
    for root, _, files in os.walk(directory):
        # Skip virtual environments and hidden directories
        if '/venv/' in root or '/.venv/' in root or '/\.' in root:
            continue
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    return python_files

def extract_imports(file_path):
    """Extract import statements from a Python file."""
    imports = set()

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Match standard imports
    import_pattern = re.compile(r'^import\s+([\w\.]+)', re.MULTILINE)
    for match in import_pattern.finditer(content):
        package = match.group(1).split('.')[0]
        imports.add(package)

    # Match from ... import
    from_pattern = re.compile(r'^from\s+([\w\.]+)\s+import', re.MULTILINE)
    for match in from_pattern.finditer(content):
        package = match.group(1).split('.')[0]
        imports.add(package)

    return imports

def check_package_installed(package):
    """Check if a package is installed."""
    try:
        importlib.import_module(package)
        return True
    except ImportError:
        return False

def get_docker_requirements():
    """Extract packages from requirements.txt."""
    requirements = set()
    req_file = Path('requirements.txt')

    if req_file.exists():
        with open(req_file, 'r') as f:
            for line in f:
                # Clean up and extract package name
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle version specifiers
                    package = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].strip()
                    requirements.add(package)

    return requirements

def main():
    print(f"{YELLOW}Checking project dependencies...{ENDC}")

    # Get all Python files
    python_files = get_project_files()
    print(f"Found {len(python_files)} Python files")

    # Extract unique imports
    all_imports = set()
    for file in python_files:
        file_imports = extract_imports(file)
        all_imports.update(file_imports)

    # Remove standard library modules
    standard_libs = set(sys.builtin_module_names)
    try:
        stdlib_path = os.path.dirname(os.__file__)
        standard_libs.update([name for name in os.listdir(stdlib_path)
                             if os.path.isdir(os.path.join(stdlib_path, name)) or
                             name.endswith('.py')])
    except:
        pass

    third_party_imports = all_imports - standard_libs

    # Compare with requirements.txt
    requirements = get_docker_requirements()

    print(f"\n{YELLOW}Checking {len(third_party_imports)} third-party imports against requirements.txt...{ENDC}")

    missing_in_requirements = []
    for package in sorted(third_party_imports):
        if package.lower() not in [req.lower() for req in requirements]:
            missing_in_requirements.append(package)

    # Check which packages are actually installed
    missing_in_env = []
    for package in sorted(third_party_imports):
        if not check_package_installed(package):
            missing_in_env.append(package)

    # Print results
    if missing_in_requirements:
        print(f"\n{RED}Packages imported but not in requirements.txt:{ENDC}")
        for package in missing_in_requirements:
            print(f"  - {package}")
    else:
        print(f"\n{GREEN}All imported packages are in requirements.txt{ENDC}")

    if missing_in_env:
        print(f"\n{RED}Packages imported but not installed in current environment:{ENDC}")
        for package in missing_in_env:
            print(f"  - {package}")
    else:
        print(f"\n{GREEN}All imported packages are installed in current environment{ENDC}")

    # Suggest pip command
    if missing_in_env:
        print(f"\n{YELLOW}Run this to install missing packages:{ENDC}")
        print(f"pip install {' '.join(missing_in_env)}")

        print(f"\n{YELLOW}To update requirements.txt, add:{ENDC}")
        for package in missing_in_env:
            print(f"{package}")

    return len(missing_in_env) > 0

if __name__ == "__main__":
    sys.exit(1 if main() else 0)