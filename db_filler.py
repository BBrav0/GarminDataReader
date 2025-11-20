#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Check Python version (requires 3.6+ for f-strings and garminconnect)
if sys.version_info < (3, 6):
    print("Error: Python 3.6 or higher is required. Current version: {}.{}".format(sys.version_info.major, sys.version_info.minor))
    sys.exit(1)

# Auto-detect and use venv Python if available
def ensure_venv():
    """Re-execute script with venv Python if not already using it."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return  # Already in a venv
    
    script_dir = Path(__file__).parent.absolute()
    venv_python = script_dir / "venv" / "bin" / "python3"
    
    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

ensure_venv()

from garminconnect import Garmin
import pandas as pd
from datetime import date, timedelta, datetime
import os
from dotenv import load_dotenv
import sqlite3 as sql
import time
import requests

def format_duration(ms):
    """Convert milliseconds to H:MM:SS string."""
    try:
        total_seconds = max(0, int(ms) // 1000)
    except Exception:
        return None
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"

def format_pace(distance_miles, ms):
    """Compute average pace as HH:MM:SS string (zero-padded hours).
    Returns None if distance is invalid or ms is None.
    """
    if distance_miles is None or ms is None:
        return None
    try:
        distance = float(distance_miles)
        if distance <= 0:
            return None
        total_seconds = max(0, int(ms) // 1000)
        seconds_per_mile = int(round(total_seconds / distance))
        hours = seconds_per_mile // 3600
        minutes = (seconds_per_mile % 3600) // 60
        seconds = seconds_per_mile % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        return None

def elevation_gain_from_tcx(xml_text: str) -> float:
    """Sum positive altitude deltas from TCX content. Returns meters."""
    try:
        import re
        alts = [float(x) for x in re.findall(r"<AltitudeMeters>([-+]?[0-9]*\.?[0-9]+)</AltitudeMeters>", xml_text or "")]
        if not alts:
            return 0.0
        gain = 0.0
        prev = alts[0]
        for a in alts[1:]:
            delta = a - prev
            if delta > 0:
                gain += delta
            prev = a
        return gain
    except Exception:
        return 0.0

def map_activity_type(garmin_type: str) -> str:
    """Map Garmin activity type to standardized 'Run' or 'Treadmill run'."""
    if not garmin_type:
        return "Run"
    
    garmin_type_lower = garmin_type.lower().strip()
    
    # Treadmill activities
    if 'treadmill' in garmin_type_lower:
        return "Treadmill run"
    
    # All other running activities map to "Run"
    running_keywords = ['running', 'run', 'street', 'track', 'trail', 'ultra', 'virtual', 'indoor', 'road']
    if any(keyword in garmin_type_lower for keyword in running_keywords):
        return "Run"
    
    # Default to "Run" for any unrecognized activity type
    return "Run"

def compute_elevation_gain(activity: dict, client: Garmin) -> float:
    """Get elevation gain in feet using API field if available, else TCX fallback."""
    # Try to get elevation gain from activity summary
    elev_m = activity.get('elevationGain') or activity.get('elevationGainMeters') or activity.get('elevationGained') or activity.get('elevationAscent')
    try:
        if elev_m is not None:
            return float(elev_m) * 3.28084
    except Exception:
        pass
    
    # Fallback: try to get TCX/GPX data
    activity_id = activity.get('activityId')
    if activity_id:
        try:
            # Try to get TCX data - handle different method signatures
            tcx_data = None
            if hasattr(client, 'download_activity'):
                try:
                    if hasattr(client, 'ActivityDownloadFormat'):
                        tcx_data = client.download_activity(activity_id, dl_fmt=client.ActivityDownloadFormat.TCX)
                    else:
                        tcx_data = client.download_activity(activity_id, dl_fmt='tcx')
                except Exception:
                    try:
                        tcx_data = client.download_activity(activity_id)
                    except Exception:
                        pass
            
            if tcx_data:
                elev_from_tcx_m = elevation_gain_from_tcx(tcx_data)
                return elev_from_tcx_m * 3.28084
        except Exception:
            pass
    
    return 0.0

def cache_run(date_str, distance, duration, steps, minhr, maxhr, avghr, calories, resting_hr=0, elev_gain=None, activity_type="Run"):
    con = sql.connect("cache.db")
    cur = con.cursor()

    # Store duration as formatted H:MM:SS string from milliseconds
    formatted_duration = format_duration(duration) if duration is not None else None
    # Compute cadence (steps per minute) as integer (rounded)
    cadence = None
    try:
        minutes = (int(duration) / 60000.0) if duration is not None else None
        if minutes and minutes > 0 and steps is not None:
            cadence_value = float(steps) / float(minutes)
            cadence = int(round(cadence_value))
    except Exception:
        cadence = None
    average_pace = format_pace(distance, duration)
    elev_gain_per_mile = None
    try:
        if elev_gain is not None and distance not in (None, 0):
            elev_gain_per_mile = float(elev_gain) / float(distance)
    except Exception:
        elev_gain_per_mile = None

    # Round REALs to two decimal places
    def round2(value):
        try:
            return round(float(value), 2)
        except Exception:
            return value
    distance = round2(distance) if distance is not None else None
    elev_gain = round2(elev_gain) if elev_gain is not None else None
    elev_gain_per_mile = round2(elev_gain_per_mile) if elev_gain_per_mile is not None else None

    cur.execute("""
        INSERT OR REPLACE INTO runs (date, distance, duration, avg_pace, elev_gain, elev_gain_per_mile, steps, cadence, minhr, maxhr, avghr, calories, resting_hr, activity_type) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (date_str, distance, formatted_duration, average_pace, elev_gain, elev_gain_per_mile, steps, cadence, minhr, maxhr, avghr, calories, resting_hr, activity_type))

    con.commit()
    con.close()

def cache_no_run(date_str):
    """Insert a placeholder for a date with no runs to avoid future API calls."""
    con = sql.connect("cache.db")
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO runs (date, distance, duration, avg_pace, elev_gain, elev_gain_per_mile, steps, cadence, minhr, maxhr, avghr, calories, resting_hr, activity_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (date_str, None, None, None, None, None, None, None, None, None, None, None, None, "None"),
    )
    con.commit()
    con.close()

def cache_pending(date_str):
    """Insert a placeholder for a date to ensure an entry exists.
    Uses activity_type = 'None' (no run) as the default state.
    """
    con = sql.connect("cache.db")
    cur = con.cursor()
    cur.execute(
        """
        INSERT OR REPLACE INTO runs (date, distance, duration, avg_pace, elev_gain, elev_gain_per_mile, steps, cadence, minhr, maxhr, avghr, calories, resting_hr, activity_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (date_str, None, None, None, None, None, None, None, None, None, None, None, None, "None"),
    )
    con.commit()
    con.close()

def get_resting_heart_rate(date_str, client: Garmin):
    """Get resting heart rate for a specific date"""
    try:
        # Parse date string
        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
        # Try different methods to get heart rate data
        hr_data = None
        if hasattr(client, 'get_heart_rates'):
            hr_data = client.get_heart_rates(date_obj.isoformat())
        elif hasattr(client, 'get_daily_summary'):
            summary = client.get_daily_summary(date_obj.isoformat())
            if summary and 'restingHeartRate' in summary:
                return summary.get('restingHeartRate', 0)
        elif hasattr(client, 'get_daily_summary_v2'):
            summary = client.get_daily_summary_v2(date_obj.isoformat())
            if summary and 'restingHeartRate' in summary:
                return summary.get('restingHeartRate', 0)
        
        if hr_data and 'restingHeartRate' in hr_data:
            return hr_data.get('restingHeartRate', 0)
        return 0
    except Exception as e:
        print(f"    Error getting resting heart rate: {e}")
        return 0

def get_treadmill_manual_data(date_str, distance):
    """Get manual data for treadmill runs from user input"""
    print(f"\n    Treadmill run detected for {date_str}")
    print(f"    Distance: {distance:.2f} miles")
    print(f"    Please enter heart rate and elevation data:")
    
    # Get heart rate data
    min_hr = None
    max_hr = None
    avg_hr = None
    
    try:
        min_hr_input = input("    Enter minimum heart rate (or press Enter to skip): ").strip()
        if min_hr_input:
            min_hr = int(min_hr_input)
    except ValueError:
        print("    Invalid minimum heart rate, skipping...")
    
    try:
        max_hr_input = input("    Enter maximum heart rate (or press Enter to skip): ").strip()
        if max_hr_input:
            max_hr = int(max_hr_input)
    except ValueError:
        print("    Invalid maximum heart rate, skipping...")
    
    try:
        avg_hr_input = input("    Enter average heart rate (or press Enter to skip): ").strip()
        if avg_hr_input:
            avg_hr = int(avg_hr_input)
    except ValueError:
        print("    Invalid average heart rate, skipping...")
    
    # Get elevation percentage
    elevation_percent = None
    try:
        elev_input = input("    Enter treadmill elevation percentage (e.g., 2.5 for 2.5%): ").strip()
        if elev_input:
            elevation_percent = float(elev_input)
    except ValueError:
        print("    Invalid elevation percentage, using 0%...")
        elevation_percent = 0.0
    
    # Calculate elevation gain
    elev_gain = 0.0
    if elevation_percent is not None and distance > 0:
        elev_gain = distance * elevation_percent * 52.8  # 5280/100 = 52.8
        print(f"    Calculated elevation gain: {elev_gain:.1f} feet ({elevation_percent}% over {distance:.2f} miles)")
    
    return {
        'min_hr': min_hr,
        'max_hr': max_hr,
        'avg_hr': avg_hr,
        'elev_gain': elev_gain
    }

# ===== ENVIRONNENTALS =======

load_dotenv()

GARMIN_EMAIL = os.getenv('GARMIN_EMAIL')
GARMIN_PASSWORD = os.getenv('GARMIN_PASSWORD')

if not GARMIN_EMAIL or not GARMIN_PASSWORD:
    print("fix your .env - set GARMIN_EMAIL and GARMIN_PASSWORD")
    exit(1)

# ============================

# ===== API SETUP =======

try:
    client = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
    client.login()
    print("✓ Garmin authentication successful")
except Exception as e:
    print(f"✗ Garmin authentication failed: {e}")
    exit(1)

# =======================


# ===== SQL SETUP =======


con = sql.connect("cache.db")
cur = con.cursor()

cur.execute("""

        CREATE TABLE IF NOT EXISTS runs (
            date TEXT PRIMARY KEY,
            distance REAL,
            duration TEXT,
            avg_pace TEXT,
            elev_gain REAL,
            elev_gain_per_mile REAL,
            steps INTEGER,
            cadence INTEGER,
            minhr INTEGER,
            maxhr INTEGER,
            avghr INTEGER,
            calories INTEGER,
            resting_hr INTEGER,
            activity_type TEXT
        )

""")

con.commit()
con.close()

# Ensure schema has avg_pace and elev_gain columns for existing databases
try:
    con = sql.connect("cache.db")
    cur = con.cursor()
    cur.execute("PRAGMA table_info(runs)")
    columns = [row[1] for row in cur.fetchall()]
    if 'avg_pace' not in columns:
        cur.execute("ALTER TABLE runs ADD COLUMN avg_pace TEXT")
        con.commit()
    if 'elev_gain' not in columns:
        cur.execute("ALTER TABLE runs ADD COLUMN elev_gain REAL")
        con.commit()
    if 'elev_gain_per_mile' not in columns:
        cur.execute("ALTER TABLE runs ADD COLUMN elev_gain_per_mile REAL")
        con.commit()
    if 'activity_type' not in columns:
        cur.execute("ALTER TABLE runs ADD COLUMN activity_type TEXT")
        con.commit()
    # Ensure cadence column exists and is INTEGER type; migrate if needed
    cur.execute("PRAGMA table_info(runs)")
    info = cur.fetchall()
    col_types = {row[1]: (row[2] or '').upper() for row in info}
    if 'cadence' not in col_types:
        cur.execute("ALTER TABLE runs ADD COLUMN cadence INTEGER")
        con.commit()
    elif col_types.get('cadence') != 'INTEGER':
        # Migrate table to make cadence INTEGER via table recreate
        try:
            cur.execute("BEGIN TRANSACTION")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS runs_new (
                    date TEXT PRIMARY KEY,
                    distance REAL,
                    duration TEXT,
                    avg_pace TEXT,
                    elev_gain REAL,
                    elev_gain_per_mile REAL,
                    steps INTEGER,
                    cadence INTEGER,
                    minhr INTEGER,
                    maxhr INTEGER,
                    avghr INTEGER,
                    calories INTEGER,
                    resting_hr INTEGER,
                    activity_type TEXT
                )
                """
            )
            # Copy with cadence cast to INTEGER (rounded)
            cur.execute(
                """
                INSERT OR REPLACE INTO runs_new (
                    date, distance, duration, avg_pace, elev_gain, elev_gain_per_mile,
                    steps, cadence, minhr, maxhr, avghr, calories, resting_hr, activity_type
                )
                SELECT
                    date, distance, duration, avg_pace, elev_gain, elev_gain_per_mile,
                    steps,
                    CASE WHEN cadence IS NULL THEN NULL ELSE CAST(ROUND(cadence) AS INTEGER) END,
                    minhr, maxhr, avghr, calories, resting_hr,
                    CASE WHEN activity_type IS NULL THEN 'Run' ELSE activity_type END
                FROM runs
                """
            )
            cur.execute("DROP TABLE runs")
            cur.execute("ALTER TABLE runs_new RENAME TO runs")
            con.commit()
        except Exception as me:
            con.rollback()
            print(f"warning: cadence type migration failed: {me}")
    con.close()
except Exception as e:
    print(f"warning: could not ensure required columns exist: {e}")

print("db ready")

# =======================


# ===== MAIN ========

# Set start date to February 20th of current year
current_year = date.today().year
start_date = date(2025, 2, 20)
curr = date.today()
request_count = 0

# Date helpers
def padded_date_string(d):
    try:
        return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
    except Exception:
        return str(d)

def unpadded_date_string(d):
    try:
        return f"{d.year}-{d.month}-{d.day}"
    except Exception:
        return str(d)

# Preload all existing dates for fast membership checks (normalized variants)
def load_existing_dates():
    try:
        con = sql.connect("cache.db")
        cur = con.cursor()
        cur.execute("SELECT date FROM runs")
        rows = cur.fetchall()
        con.close()
        s = set()
        for (raw,) in rows:
            if raw is None:
                continue
            raw_stripped = str(raw).strip()
            s.add(raw_stripped)
            # Try to parse basic YYYY-MM-DD and also store unpadded/padded variants
            try:
                parts = raw_stripped.split("-")
                if len(parts) == 3:
                    y = int(parts[0])
                    m = int(parts[1])
                    d2 = int(parts[2])
                    s.add(f"{y:04d}-{m:02d}-{d2:02d}")
                    s.add(f"{y}-{m}-{d2}")
            except Exception:
                pass
        return s
    except Exception:
        return set()

existing_dates = load_existing_dates()
failure_counts = {}

# Check if we have complete data for the current date
def date_is_complete(check_date):
    """Return True if the row exists and required fields are present.
    Rule: if activity_type=='None' it's complete; if activity_type in ('Run', 'Treadmill run'), require elev_gain not NULL.
    """
    try:
        con = sql.connect("cache.db")
        cur = con.cursor()
        cur.execute("SELECT activity_type, elev_gain, elev_gain_per_mile FROM runs WHERE date = ?", (str(check_date),))
        row = cur.fetchone()
        con.close()
        if row is None:
            return False
        activity_type_val, elev_gain_val, elev_gain_per_mile_val = row
        if activity_type_val == 'None':
            return True
        return (elev_gain_val is not None) and (elev_gain_per_mile_val is not None)
    except Exception:
        return False

def date_exists(check_date):
    """Return True if any row exists for the given date, regardless of completeness."""
    try:
        con = sql.connect("cache.db")
        cur = con.cursor()
        cur.execute("SELECT 1 FROM runs WHERE date = ? LIMIT 1", (str(check_date),))
        row = cur.fetchone()
        con.close()
        return row is not None
    except Exception:
        return False

def date_exists_loose(check_date):
    """Return True if any row exists for the date, allowing suffixes (e.g., time).
    Matches exact date or values starting with the date string.
    """
    try:
        s_padded = padded_date_string(check_date)
        s_unpadded = unpadded_date_string(check_date)
        con = sql.connect("cache.db")
        cur = con.cursor()
        cur.execute(
            """
            SELECT 1 FROM runs
            WHERE date = ? OR date = ? OR date LIKE ? OR date LIKE ?
            LIMIT 1
            """,
            (s_padded, s_unpadded, s_padded + '%', s_unpadded + '%')
        )
        row = cur.fetchone()
        con.close()
        return row is not None
    except Exception:
        return False

while curr >= start_date:
    # Skip only if the date is fully complete
    padded = padded_date_string(curr)
    unpadded = unpadded_date_string(curr)
    if date_is_complete(curr):
        print(f"✓ Data already complete for {padded} - skipping API")
        curr = curr - timedelta(1)
        continue

    # Ensure a pending placeholder exists so interruptions still leave a row
    try:
        con = sql.connect("cache.db")
        cur = con.cursor()
        cur.execute("SELECT 1 FROM runs WHERE date = ? OR date = ? OR date LIKE ? OR date LIKE ? LIMIT 1", (padded, unpadded, padded + '%', unpadded + '%'))
        exists_row = cur.fetchone() is not None
        con.close()
    except Exception:
        exists_row = False
    if not exists_row:
        cache_pending(padded)
    
    days_remaining = (curr - start_date).days + 1
    print(f"Processing {curr} ({days_remaining} days remaining)...")
    
    try:
        # Add delay between requests to respect rate limits
        if request_count > 0:
            print("  Waiting 2 seconds before next request...")
            time.sleep(2)
        
        # Get activities for current date
        # Try different methods depending on garminconnect version
        activities = None
        try:
            # Try get_activities_by_date first (if available)
            if hasattr(client, 'get_activities_by_date'):
                activities = client.get_activities_by_date(curr.isoformat())
            elif hasattr(client, 'get_activities'):
                # Get activities and filter by date
                all_activities = client.get_activities(0, 100)  # Get first 100 activities
                activities = [a for a in all_activities if a.get('startTimeLocal', '').startswith(curr.isoformat())]
            else:
                # Fallback: try to get activities for the date range
                activities = client.get_activities_by_date(curr.isoformat(), curr.isoformat())
        except Exception as e:
            # If method doesn't exist or fails, try alternative
            try:
                all_activities = client.get_activities(0, 100)
                activities = [a for a in all_activities if a.get('startTimeLocal', '').startswith(curr.isoformat())]
            except Exception as e2:
                print(f"  ⚠ Error fetching activities: {e2}")
                activities = []
        
        request_count += 1
        
        if activities:
            for activity in activities:
                # Get activity type from Garmin
                garmin_activity_type = activity.get('activityType', {}).get('typeKey', '') or activity.get('activityType', '')
                activity_type = map_activity_type(garmin_activity_type)
                
                # Only process Run or Treadmill run activities
                if activity_type in ["Run", "Treadmill run"]:
                    # Get distance in meters, convert to miles
                    distance_m = activity.get('distance', 0) or activity.get('distanceMeters', 0)
                    distance = distance_m / 1609.34  # Convert meters to miles
                    
                    # Skip runs with 0 distance
                    if distance == 0:
                        print(f"  ⚠ Skipping run with 0 distance for {curr}")
                        continue
                    
                    print(f"  ✓ Found {activity_type.lower()} for {curr}")
                    date_str = padded_date_string(curr)
                    print(f"    {activity_type} {date_str}:")
                    print(f"    Activity ID: {activity.get('activityId', 'N/A')}")
                    print(f"    Start Time: {activity.get('startTimeLocal', 'N/A')}")
                    
                    # Get duration in seconds, convert to milliseconds
                    duration_sec = activity.get('duration', 0) or activity.get('elapsedDuration', 0)
                    duration_ms = int(duration_sec * 1000) if duration_sec else 0
                    print(f"    Duration: {format_duration(duration_ms)}")
                    print(f"    Distance: {round(distance, 2)} miles")
                    
                    # Get steps
                    steps = activity.get('steps', 0) or activity.get('stepsCount', 0)
                    print(f"    Steps: {steps}")
                    
                    # Get calories
                    calories = activity.get('calories', 0) or activity.get('caloriesConsumed', 0)
                    print(f"    Calories: {calories}")
                    
                    # Branch processing based on activity type
                    if activity_type == "Run":
                        # Outdoor run logic
                        avg_hr = activity.get('averageHeartRate', 0) or activity.get('avgHeartRate', 0) or activity.get('heartRate', {}).get('averageHeartRate', 0)
                        max_hr = activity.get('maxHeartRate', 0) or activity.get('heartRate', {}).get('maxHeartRate', 0)
                        min_hr = activity.get('minHeartRate', 0) or activity.get('heartRate', {}).get('minHeartRate', 0)
                        
                        # Try to get detailed activity data for heart rate
                        activity_id = activity.get('activityId')
                        if activity_id and (not avg_hr or not max_hr):
                            try:
                                # Try different methods to get activity details
                                activity_details = None
                                if hasattr(client, 'get_activity'):
                                    activity_details = client.get_activity(activity_id)
                                elif hasattr(client, 'get_activity_summary'):
                                    activity_details = client.get_activity_summary(activity_id)
                                
                                if activity_details:
                                    if not avg_hr:
                                        avg_hr = activity_details.get('averageHeartRate', 0) or activity_details.get('avgHeartRate', 0) or activity_details.get('heartRate', {}).get('averageHeartRate', 0)
                                    if not max_hr:
                                        max_hr = activity_details.get('maxHeartRate', 0) or activity_details.get('heartRate', {}).get('maxHeartRate', 0)
                                    if not min_hr:
                                        min_hr = activity_details.get('minHeartRate', 0) or activity_details.get('heartRate', {}).get('minHeartRate', 0)
                            except Exception as e:
                                print(f"    Could not fetch activity details: {e}")
                        
                        # Compute elevation gain in feet
                        elev_gain = compute_elevation_gain(activity, client)
                        
                    elif activity_type == "Treadmill run":
                        # Treadmill run logic - get manual data from user
                        manual_data = get_treadmill_manual_data(date_str, distance)
                        min_hr = manual_data['min_hr']
                        max_hr = manual_data['max_hr']
                        avg_hr = manual_data['avg_hr']
                        elev_gain = manual_data['elev_gain']
                    
                    print(f"    Average HR: {avg_hr if avg_hr else 'N/A'}")
                    print(f"    Max HR: {max_hr if max_hr else 'N/A'}")
                    print(f"    Min HR: {min_hr if min_hr else 'N/A'}")
                    
                    # Get resting heart rate for the day
                    resting_hr = get_resting_heart_rate(date_str, client)
                    print(f"    Resting HR: {resting_hr}")
                    print(f"    Elevation Gain (ft): {elev_gain:.1f if elev_gain else 0:.1f}")
                    
                    print("-" * 50)
                    # Cache the run data
                    cache_run(date_str, distance, duration_ms, steps, min_hr, max_hr, avg_hr, calories, resting_hr, elev_gain, activity_type=activity_type)
                    existing_dates.add(date_str.strip())
                    try:
                        # Also add alt normalized forms
                        parts = date_str.strip().split("-")
                        if len(parts) == 3:
                            y = int(parts[0]); m = int(parts[1]); d2 = int(parts[2])
                            existing_dates.add(f"{y:04d}-{m:02d}-{d2:02d}")
                            existing_dates.add(f"{y}-{m}-{d2}")
                    except Exception:
                        pass
                    break
        else:
            print(f"  No runs found for {curr}")
            # Cache a placeholder to avoid re-querying this date
            cache_no_run(padded_date_string(curr))
            existing_dates.add(padded)
            existing_dates.add(unpadded)
            
        # Move to previous day after successful processing (regardless of runs found)
        curr = curr - timedelta(1)
        
    except requests.exceptions.Timeout:
        print(f"  ⚠ Timeout getting activities for {curr}")
        key = padded_date_string(curr)
        failure_counts[key] = failure_counts.get(key, 0) + 1
        if failure_counts[key] >= 2:
            print(f"  ⚠ Marking {key} as no-run after repeated timeouts")
            cache_no_run(key)
            existing_dates.add(key)
        # Move to previous day
        curr = curr - timedelta(1)
    except requests.exceptions.RequestException as e:
        print(f"  ⚠ Network error for {curr}: {e}")
        print("  Waiting 30 seconds before retry...")
        time.sleep(30)
        # Don't move to next date - retry the same date
        continue
    except Exception as e:
        error_msg = str(e)
        if 'rate limit' in error_msg.lower() or 'too many' in error_msg.lower():
            print(f"  ⚠ Rate limit hit for {curr}: {e}")
            print("  Waiting 100 seconds before retry...")
            time.sleep(100)
            # Don't move to next date - retry the same date
            continue
        else:
            print(f"  ❌ Error getting activities for {curr}: {e}")
            key = padded_date_string(curr)
            failure_counts[key] = failure_counts.get(key, 0) + 1
            if failure_counts[key] >= 2:
                print(f"  ⚠ Marking {key} as no-run after repeated errors")
                cache_no_run(key)
                existing_dates.add(key)
            # Move to previous day for other errors
            curr = curr - timedelta(1)

print(f"\nCompleted processing {request_count} API requests")
