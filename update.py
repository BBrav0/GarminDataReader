#!/usr/bin/env python3
"""
Update script that runs the Garmin data update pipeline:
1. db_filler.py - Fetch and cache run data from Garmin API
2. db_to_csv.py - Export cached data to CSV
"""
import json
import sqlite3
import subprocess
import sys
import os
from datetime import date
from pathlib import Path

from garmin_store import format_run_summary, get_latest_run

SCRIPT_DIR = Path(__file__).parent.absolute()
RHR_LOG_PATH = Path.home() / ".hermes" / "workspace" / "rhr_log.jsonl"

# Check Python version (requires 3.6+ for f-strings)
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required. Current version: {}.{}".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

def get_venv_python():
    """Find and return the path to the venv Python interpreter."""
    for env_name in (".venv", "venv"):
        venv_python = SCRIPT_DIR / env_name / "bin" / "python3"
        if venv_python.exists():
            return str(venv_python)
    
    # Fallback: check if we're already in a venv
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return sys.executable
    
    # If no venv found, return None to use system Python
    return None

def get_cached_resting_heart_rate(date_str, db_path=None):
    """Read a cached resting heart rate for the given date from cache.db."""
    db_path = db_path or (SCRIPT_DIR / "cache.db")
    if not db_path.exists():
        return None

    con = None
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute(
            """
            SELECT resting_hr
            FROM runs
            WHERE date = ? AND resting_hr IS NOT NULL AND resting_hr > 0
            ORDER BY CASE WHEN activity_type = 'None' THEN 1 ELSE 0 END, rowid DESC
            LIMIT 1
            """,
            (date_str,),
        )
        row = cur.fetchone()
        return row[0] if row else None
    except sqlite3.Error as e:
        print(f"WARN: could not read resting HR from cache.db: {e}")
        return None
    finally:
        if con is not None:
            con.close()

def append_rhr_log_entry(date_str, rhr, log_path=None):
    """Append an RHR log entry if the date is not already present."""
    log_path = log_path or RHR_LOG_PATH

    try:
        rhr = int(rhr)
    except (TypeError, ValueError):
        return False

    if rhr <= 0:
        return False

    existing_dates = set()
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_dates.add(json.loads(line)["date"])
            except Exception:
                pass

    if date_str in existing_dates:
        print(f"RHR {date_str}: {rhr} bpm already logged")
        return False

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": date_str, "rhr": rhr}) + "\n")

    print(f"RHR {date_str}: {rhr} bpm logged to {log_path}")
    return True

def sync_today_rhr_log(db_path=None, log_path=None):
    """Write today's cached RHR to rhr_log.jsonl so pull_rhr.py can skip later."""
    today = date.today().isoformat()
    rhr = get_cached_resting_heart_rate(today, db_path=db_path)
    if rhr is None:
        return False
    return append_rhr_log_entry(today, rhr, log_path=log_path)

def run_script(script_name):
    """Run a Python script and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Running {script_name}...")
    print(f"{'='*60}\n")
    
    # Use venv Python if available, otherwise use current interpreter
    python_executable = get_venv_python() or sys.executable
    
    script_path = SCRIPT_DIR / script_name

    try:
        result = subprocess.run(
            [python_executable, str(script_path)],
            check=True,
            cwd=str(SCRIPT_DIR)
        )
        print(f"\n✓ {script_name} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {script_name} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Error: {script_name} not found")
        return False


def print_latest_run_summary(db_path=None):
    """Print the latest locally cached run after the pipeline succeeds."""
    latest_run = get_latest_run(db_path or (SCRIPT_DIR / "cache.db"))
    print(f"Latest cached run: {format_run_summary(latest_run)}")

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
        if script == "db_filler.py":
            sync_today_rhr_log()
    
    print(f"\n{'='*60}")
    print("All scripts completed successfully!")
    print(f"{'='*60}")
    print_latest_run_summary()

