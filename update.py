#!/usr/bin/env python3
"""
Update script that runs the Fitbit data update pipeline:
1. get_tokens.py - Get/refresh OAuth tokens
2. db_filler.py - Fetch and cache run data from Fitbit API
3. db_to_csv.py - Export cached data to CSV
"""
import subprocess
import sys
import os
from pathlib import Path

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
        "get_tokens.py",
        "db_filler.py",
        "db_to_csv.py"
    ]
    
    print("Starting Fitbit data update pipeline...")
    
    for script in scripts:
        success = run_script(script)
        if not success:
            print(f"\nPipeline stopped due to failure in {script}")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("All scripts completed successfully!")
    print(f"{'='*60}")

