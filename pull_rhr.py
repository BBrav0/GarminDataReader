#!/usr/bin/env python3
"""
pull_rhr.py — Pull today's resting heart rate from Garmin Connect.
Runs daily (run day or not). Appends to ~/.hermes/workspace/rhr_log.jsonl
and updates cache.db if a run row exists for today.
"""
import os, sys, json, sqlite3
from pathlib import Path
from datetime import date

SCRIPT_DIR = Path(__file__).parent.absolute()

# Auto-detect and use venv Python if available
def ensure_venv():
    """Re-execute script with venv Python if not already using it."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return  # Already in a venv

    venv_python = SCRIPT_DIR / "venv" / "bin" / "python3"

    if venv_python.exists():
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)

ensure_venv()

from dotenv import load_dotenv
from garmin_auth import login_to_garmin

# ── Load credentials ──────────────────────────────────────────────────
load_dotenv(SCRIPT_DIR / ".env")
GARMIN_EMAIL    = os.getenv("GARMIN_EMAIL")
GARMIN_PASSWORD = os.getenv("GARMIN_PASSWORD")

if not GARMIN_EMAIL or not GARMIN_PASSWORD:
    print("ERROR: GARMIN_EMAIL / GARMIN_PASSWORD not set in .env")
    sys.exit(1)

# ── Skip if already logged today ─────────────────────────────────────
today = date.today().isoformat()
log_path = Path.home() / ".hermes" / "workspace" / "rhr_log.jsonl"
if log_path.exists():
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            if entry.get("date") == today:
                rhr = entry.get("rhr")
                print(f"RHR {today}: {rhr} bpm (already logged, skipping Garmin call)")
                sys.exit(0)
        except Exception:
            pass

# ── Auth ──────────────────────────────────────────────────────────────
try:
    client = login_to_garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
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
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            existing_dates.add(json.loads(line)["date"])
        except Exception:
            pass

if today not in existing_dates:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"date": today, "rhr": rhr}) + "\n")

# ── Update cache.db if a run row exists for today ─────────────────────
db_path = SCRIPT_DIR / "cache.db"
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
