#!/usr/bin/env python3
"""
Update script that runs the Garmin data update pipeline:
1. get_tokens.py - Verify Garmin credentials and authentication
2. db_filler.py - Fetch and cache run data from Garmin API
3. db_to_csv.py - Export cached data to CSV
"""
import subprocess
import sys
import os
from pathlib import Path

# Check Python version (requires 3.6+ for f-strings)
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required. Current version: {}.{}".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

def get_venv_python():
    """Find and return the path to the venv Python interpreter."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    venv_python = script_dir / "venv" / "bin" / "python3"
    
    # Check if venv Python exists
    if venv_python.exists():
        return str(venv_python)
    
    # Fallback: check if we're already in a venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.executable
    
    # If no venv found, return None to use system Python
    return None

def run_script(script_name):
    """Run a Python script and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}\n")
    
    # Use venv Python if available, otherwise use current interpreter
    python_executable = get_venv_python() or sys.executable
    
    try:
        result = subprocess.run(
            [python_executable, script_name],
            check=True
        )
        print(f"\n✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Error: {script_name} not found")
        return False

if __name__ == "__main__":
    scripts = [
        "db_filler.py",
        "db_to_csv.py"
    ]
    
    print("Starting Garmin data update pipeline...")
    
    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"\nPipeline stopped due to failure in {script}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("All scripts completed successfully!")
    print(f"{'='*60}")

