#!/usr/bin/env python3
"""Summarize recent Garmin cross-training activities without mixing them into run logs.

Intended for bike/walk cross-training checks during return-to-run blocks. It reads the
local cache populated by update.py and prints recent non-run activities as JSON.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "cache.db"
RUN_TYPES = {"run", "running", "treadmill_running", "track_running"}


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    # Normalize the HR key names for quick coach reads while preserving originals.
    if "avghr" in data and "avg_hr" not in data:
        data["avg_hr"] = data["avghr"]
    if "maxhr" in data and "max_hr" not in data:
        data["max_hr"] = data["maxhr"]
    return data


def _is_run(activity_type: str | None) -> bool:
    if not activity_type:
        return False
    normalized = activity_type.strip().lower().replace(" ", "_")
    return normalized in RUN_TYPES or "run" in normalized


def fetch_activities(days: int, activity_type: str | None = None) -> list[dict[str, Any]]:
    if not DB_PATH.exists():
        raise SystemExit(f"cache DB not found: {DB_PATH}; run update.py first")

    since = (date.today() - timedelta(days=days - 1)).isoformat()
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        cur = con.cursor()
        cur.execute(
            """
            SELECT *
            FROM runs
            WHERE date >= ?
              AND COALESCE(activity_type, '') NOT IN ('', 'None')
            ORDER BY start_time_local DESC, date DESC
            """,
            (since,),
        )
        rows = [_row_to_dict(row) for row in cur.fetchall()]
    finally:
        con.close()

    activities = [row for row in rows if not _is_run(row.get("activity_type"))]
    if activity_type:
        needle = activity_type.strip().lower()
        activities = [
            row
            for row in activities
            if needle in str(row.get("activity_type", "")).strip().lower()
            or needle in str(row.get("activity_name", "")).strip().lower()
        ]
    return activities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print recent non-run Garmin activities for cross-training review."
    )
    parser.add_argument("--days", type=int, default=2, help="Lookback window in days")
    parser.add_argument(
        "--type",
        dest="activity_type",
        default=None,
        help="Optional activity/name filter, e.g. bike, cycling, ride",
    )
    args = parser.parse_args()

    activities = fetch_activities(args.days, args.activity_type)
    output = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(DB_PATH),
        "note": "Cross-training only. Do not treat these as runs or use them to advance RTR progression.",
        "count": len(activities),
        "activities": activities,
    }
    print(json.dumps(output, indent=2, default=str))


if __name__ == "__main__":
    main()
