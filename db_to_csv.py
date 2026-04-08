#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Check Python version (requires 3.6+ for f-strings)
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required. Current version: {}.{}".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

# Auto-detect and use venv Python if available
def ensure_venv():
    """Re-execute script with venv Python if not already using it."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return  # Already in a venv
    
    script_dir = Path(__file__).parent.absolute()
    for env_name in (".venv", "venv"):
        venv_python = script_dir / env_name / "bin" / "python3"
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

ensure_venv()

import csv
import os
import sqlite3

from garmin_store import initialize_database

def export_runs_to_csv(db_path="cache.db", csv_path="runs_data.csv"):
    """
    Connects to the SQLite database, reads data from the 'runs' table,
    filters for records where activity_type in ('Run', 'Treadmill run'), and writes them to a CSV file.
    """
    initialize_database(db_path)
    if not os.path.exists(db_path):
        print(f"Error: Database file not found at '{db_path}'")
        return 0

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("PRAGMA table_info(runs)")
        columns_info = cursor.fetchall()
        column_names = [col[1] for col in columns_info]

        order_by = (
            "COALESCE(start_time_local, date || 'T00:00:00') DESC"
            if "start_time_local" in column_names
            else "date(date) DESC"
        )
        query = (
            "SELECT "
            + ", ".join([col for col in column_names])
            + " FROM runs WHERE activity_type IN ('Run', 'Treadmill run')"
            + f" ORDER BY {order_by}"
        )
        cursor.execute(query)
        records = cursor.fetchall()

        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(column_names)
            csv_writer.writerows(records)

        if not records:
            print("No run records found; wrote CSV header only.")
            return 0

        print(f"Successfully exported {len(records)} records to '{csv_path}'")
        return len(records)

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Define the database and CSV file paths
    database_file = "cache.db"
    output_csv_file = "runs_data.csv"

    print(f"Attempting to export data from '{database_file}' to '{output_csv_file}'...")
    export_runs_to_csv(database_file, output_csv_file)
