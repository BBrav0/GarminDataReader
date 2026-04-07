#!/usr/bin/env python3
"""
pull_rhr.py — Pull today's resting heart rate from Garmin Connect.
Runs daily (run day or not). Appends to ~/.hermes/workspace/rhr_log.jsonl
and updates cache.db if a run row exists for today.
"""
import os, sys, json, sqlite3
from pathlib import Path
from datetime import date, datetime
from dotenv import load_dotenv

# ── Load credentials ──────────────────────────────────────────────────
load_dotenv(Path(__file__).parent / ".env")
GARMIN_EMAIL    = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")

if not GARMIN_EMAIL or not GARMIN_PASSWORD:
    print("ERROR: GARMIN_EMAIL / GARMIN_PASSWORD not set in .env")
    sys.exit(1)

# ── Skip if already logged today ─────────────────────────────────────
today = date.today().isoformat()
log_path = Path.home() / ".hermes" / "workspace" / "rhr_log.jsonl"
if log_path.exists():
    for line in log_path.read_text().splitlines():
        try:
            if json.loads(line).get("date") == today:
                rhr = json.loads(line).get("rhr")
                print(f"RHR {today}: {rhr} bpm (already logged, skipping Garmin call)")
                sys.exit(0)
        except Exception:
            pass

# ── Auth ──────────────────────────────────────────────────────────────
try:
    from garminconnect import Garmin
    client = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
    client.login()
except Exception as e:
    print(f"ERROR: Garmin auth failed: {e}")
    sys.exit(1)

# ── Fetch RHR ─────────────────────────────────────────────────────────
rhr = None

try:
    hr_data = client.get_heart_rates(today)
    rhr = hr_data.get("restingHeartRate")
except Exception:
    pass

if rhr is None:
    # Fallback: daily summary
    try:
        summary = client.get_daily_summary(today)
        rhr = summary.get("restingHeartRate")
    except Exception:
        pass

if rhr is None:
    print(f"RHR {today}: no data available from Garmin (rest day or sync pending)")
    sys.exit(0)

# ── Append to rhr_log.jsonl ───────────────────────────────────────────
log_path = Path.home() / ".hermes" / "workspace" / "rhr_log.jsonl"
log_path.parent.mkdir(parents=True, exist_ok=True)

# Avoid duplicate entries for the same date
existing_dates = set()
if log_path.exists():
    for line in log_path.read_text().splitlines():
        try:
            existing_dates.add(json.loads(line)["date"])
        except Exception:
            pass

if today not in existing_dates:
    with log_path.open("a") as f:
        f.write(json.dumps({"date": today, "rhr": rhr}) + "\n")

# ── Update cache.db if a run row exists for today ─────────────────────
db_path = Path(__file__).parent / "cache.db"
if db_path.exists():
    try:
        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        cur.execute("UPDATE runs SET resting_hr = ? WHERE date = ?", (rhr, today))
        con.commit()
        con.close()
    except Exception as e:
        print(f"WARN: could not update cache.db: {e}")

print(f"RHR {today}: {rhr} bpm")
