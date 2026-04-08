#!/usr/bin/env python3
"""Shared helpers for the local Garmin activity store."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).parent.absolute()
DB_PATH = SCRIPT_DIR / "cache.db"
DEFAULT_BOOTSTRAP_START_DATE = date(2026, 4, 1)
DEFAULT_SYNC_LOOKBACK_DAYS = 14
RUN_ACTIVITY_TYPES = ("Run", "Treadmill run")


def format_duration(milliseconds: int | float | None) -> str | None:
    """Convert milliseconds to H:MM:SS."""
    if milliseconds is None:
        return None
    try:
        total_seconds = max(0, int(milliseconds) // 1000)
    except (TypeError, ValueError):
        return None
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def format_pace(distance_miles: float | int | None, milliseconds: int | float | None) -> str | None:
    """Return average pace as HH:MM:SS per mile."""
    if distance_miles is None or milliseconds is None:
        return None
    try:
        distance = float(distance_miles)
        if distance <= 0:
            return None
        total_seconds = max(0, int(milliseconds) // 1000)
    except (TypeError, ValueError):
        return None

    seconds_per_mile = int(round(total_seconds / distance))
    hours = seconds_per_mile // 3600
    minutes = (seconds_per_mile % 3600) // 60
    seconds = seconds_per_mile % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def initialize_database(db_path: str | Path | None = None) -> Path:
    """Create/migrate the local SQLite store."""
    resolved_db_path = Path(db_path) if db_path else DB_PATH
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)

    with closing(sqlite3.connect(str(resolved_db_path))) as con:
        cur = con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                date TEXT PRIMARY KEY,
                activity_id INTEGER,
                start_time_local TEXT,
                activity_name TEXT,
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
                activity_type TEXT,
                last_synced_at TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        con.commit()

        cur.execute("PRAGMA table_info(runs)")
        existing_columns = {row[1] for row in cur.fetchall()}
        required_columns = {
            "activity_id": "INTEGER",
            "start_time_local": "TEXT",
            "activity_name": "TEXT",
            "avg_pace": "TEXT",
            "elev_gain": "REAL",
            "elev_gain_per_mile": "REAL",
            "cadence": "INTEGER",
            "resting_hr": "INTEGER",
            "activity_type": "TEXT",
            "last_synced_at": "TEXT",
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                cur.execute(f"ALTER TABLE runs ADD COLUMN {column_name} {column_type}")

        cur.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_activity_id
            ON runs(activity_id)
            WHERE activity_id IS NOT NULL
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_start_time_local
            ON runs(start_time_local DESC, date DESC)
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_runs_activity_type
            ON runs(activity_type, date DESC)
            """
        )
        con.commit()

    return resolved_db_path


def _round2(value: Any) -> float | None:
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _normalize_optional_int(value: Any) -> int | None:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return None
    return normalized if normalized > 0 else None


def set_sync_state(key: str, value: str, db_path: str | Path | None = None) -> None:
    resolved_db_path = initialize_database(db_path)
    with closing(sqlite3.connect(str(resolved_db_path))) as con:
        con.execute(
            """
            INSERT INTO sync_state(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        con.commit()


def get_sync_state(key: str, db_path: str | Path | None = None) -> str | None:
    resolved_db_path = initialize_database(db_path)
    with closing(sqlite3.connect(str(resolved_db_path))) as con:
        row = con.execute("SELECT value FROM sync_state WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def get_sync_window(
    db_path: str | Path | None = None,
    today: date | None = None,
    bootstrap_start_date: date = DEFAULT_BOOTSTRAP_START_DATE,
    lookback_days: int = DEFAULT_SYNC_LOOKBACK_DAYS,
) -> tuple[date, date]:
    """Return the bounded sync window for the next activity refresh."""
    resolved_db_path = initialize_database(db_path)
    sync_end = today or date.today()
    last_sync = get_sync_state("last_activity_sync_end", resolved_db_path)

    if last_sync:
        sync_start = max(
            bootstrap_start_date,
            date.fromisoformat(last_sync) - timedelta(days=lookback_days),
        )
        return sync_start, sync_end

    with closing(sqlite3.connect(str(resolved_db_path))) as con:
        row = con.execute(
            """
            SELECT date
            FROM runs
            WHERE activity_type IN (?, ?)
            ORDER BY COALESCE(start_time_local, date || 'T00:00:00') DESC
            LIMIT 1
            """,
            RUN_ACTIVITY_TYPES,
        ).fetchone()

    if row and row[0]:
        latest_local_date = date.fromisoformat(row[0])
        sync_start = max(bootstrap_start_date, latest_local_date - timedelta(days=lookback_days))
    else:
        sync_start = bootstrap_start_date

    return sync_start, sync_end


def upsert_run(run: dict[str, Any], db_path: str | Path | None = None) -> None:
    """Insert or update one locally cached run row."""
    resolved_db_path = initialize_database(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()

    distance = _round2(run.get("distance"))
    elev_gain = _round2(run.get("elev_gain"))
    elev_gain_per_mile = None
    if elev_gain is not None and distance not in (None, 0):
        elev_gain_per_mile = _round2(elev_gain / distance)

    duration_ms = run.get("duration_ms")
    steps = _normalize_optional_int(run.get("steps"))
    cadence = None
    try:
        if steps is not None and duration_ms is not None:
            minutes = int(duration_ms) / 60000.0
            if minutes > 0:
                cadence = int(round(steps / minutes))
    except (TypeError, ValueError, ZeroDivisionError):
        cadence = None

    activity_id = run.get("activity_id")
    try:
        activity_id = int(activity_id) if activity_id is not None else None
    except (TypeError, ValueError):
        activity_id = None

    record = {
        "date": run["date"],
        "activity_id": activity_id,
        "start_time_local": run.get("start_time_local"),
        "activity_name": run.get("activity_name"),
        "distance": distance,
        "duration": format_duration(duration_ms),
        "avg_pace": format_pace(distance, duration_ms),
        "elev_gain": elev_gain,
        "elev_gain_per_mile": elev_gain_per_mile,
        "steps": steps,
        "cadence": cadence,
        "minhr": _normalize_optional_int(run.get("minhr")),
        "maxhr": _normalize_optional_int(run.get("maxhr")),
        "avghr": _normalize_optional_int(run.get("avghr")),
        "calories": _normalize_optional_int(run.get("calories")),
        "resting_hr": _normalize_optional_int(run.get("resting_hr")),
        "activity_type": run.get("activity_type"),
        "last_synced_at": now_iso,
    }

    columns = list(record.keys())
    placeholders = ", ".join("?" for _ in columns)
    values = [record[column] for column in columns]
    optional_columns = {
        "activity_name",
        "minhr",
        "maxhr",
        "avghr",
        "calories",
        "resting_hr",
        "elev_gain",
        "elev_gain_per_mile",
        "avg_pace",
        "duration",
        "steps",
        "cadence",
        "start_time_local",
        "distance",
    }
    update_assignments = []
    for column in columns:
        if column == "date":
            continue
        if column in optional_columns:
            update_assignments.append(f"{column} = COALESCE(excluded.{column}, runs.{column})")
        else:
            update_assignments.append(f"{column} = excluded.{column}")

    with closing(sqlite3.connect(str(resolved_db_path))) as con:
        cur = con.cursor()
        if activity_id is not None:
            existing = cur.execute(
                "SELECT date FROM runs WHERE activity_id = ?",
                (activity_id,),
            ).fetchone()
            if existing and existing[0] != record["date"]:
                cur.execute("DELETE FROM runs WHERE activity_id = ?", (activity_id,))

        cur.execute(
            f"""
            INSERT INTO runs ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(date) DO UPDATE SET
                {", ".join(update_assignments)}
            """,
            values,
        )
        con.commit()


def store_resting_heart_rate(date_str: str, resting_hr: int | None, db_path: str | Path | None = None) -> None:
    resolved_db_path = initialize_database(db_path)
    normalized_rhr = _normalize_optional_int(resting_hr)
    if normalized_rhr is None:
        return

    with closing(sqlite3.connect(str(resolved_db_path))) as con:
        con.execute(
            "UPDATE runs SET resting_hr = ? WHERE date = ?",
            (normalized_rhr, date_str),
        )
        con.commit()


def get_latest_run(db_path: str | Path | None = None) -> dict[str, Any] | None:
    resolved_db_path = initialize_database(db_path)
    with closing(sqlite3.connect(str(resolved_db_path))) as con:
        con.row_factory = sqlite3.Row
        row = con.execute(
            """
            SELECT *
            FROM runs
            WHERE activity_type IN (?, ?)
            ORDER BY COALESCE(start_time_local, date || 'T00:00:00') DESC,
                     COALESCE(activity_id, 0) DESC
            LIMIT 1
            """,
            RUN_ACTIVITY_TYPES,
        ).fetchone()
    return dict(row) if row else None


def format_run_summary(run: dict[str, Any] | None) -> str:
    if not run:
        return "No run data available."

    start_fragment = run.get("start_time_local") or run.get("date")
    activity_type = run.get("activity_type") or "Run"
    distance = run.get("distance")
    duration = run.get("duration") or "unknown duration"
    pace = run.get("avg_pace")

    pieces = [str(start_fragment), str(activity_type)]
    if distance is not None:
        pieces.append(f"{float(distance):.2f} mi")
    pieces.append(str(duration))
    if pace:
        pieces.append(f"{pace}/mi")
    return " — ".join(pieces)
