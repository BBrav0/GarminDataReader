#!/usr/bin/env python3
"""Fetch recent Garmin activities and keep the local SQLite cache usable."""

from __future__ import annotations

import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Any

if sys.version_info < (3, 6):
    print(
        "Error: Python 3.6 or higher is required. Current version: {}.{}".format(
            sys.version_info.major,
            sys.version_info.minor,
        )
    )
    sys.exit(1)


SCRIPT_DIR = Path(__file__).parent.absolute()


def ensure_venv():
    """Re-execute with the project virtualenv if available."""
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        return

    for env_name in (".venv", "venv"):
        venv_python = SCRIPT_DIR / env_name / "bin" / "python3"
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)


ensure_venv()

import requests
from dotenv import load_dotenv

from garmin_auth import is_rate_limit_error, login_to_garmin
from garmin_store import (
    DEFAULT_BOOTSTRAP_START_DATE,
    DB_PATH,
    format_run_summary,
    get_latest_run,
    get_sync_window,
    initialize_database,
    set_sync_state,
    upsert_run,
)


def first_value(*values: Any) -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def garmin_request(description: str, action, max_attempts: int = 3):
    """Run one Garmin API request with bounded retries."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return action()
        except Exception as error:  # pragma: no cover - exercised in integration use
            last_error = error
            retryable = is_rate_limit_error(error) or isinstance(
                error,
                (
                    requests.exceptions.Timeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.RequestException,
                ),
            )
            if not retryable or attempt >= max_attempts:
                raise

            if is_rate_limit_error(error):
                delay_seconds = min(120, 30 * (2 ** (attempt - 1)))
                print(
                    f"⚠ {description} hit Garmin rate limits "
                    f"(attempt {attempt}/{max_attempts}): {error}"
                )
            else:
                delay_seconds = min(20, 5 * attempt)
                print(
                    f"⚠ {description} had a transient network error "
                    f"(attempt {attempt}/{max_attempts}): {error}"
                )
            print(f"  Waiting {delay_seconds}s before retrying...")
            time.sleep(delay_seconds)
    raise RuntimeError(f"{description} failed: {last_error}")


def elevation_gain_from_tcx(xml_text: str | bytes | bytearray | None) -> float | None:
    """Sum positive altitude deltas from TCX content and return feet."""
    try:
        import re

        if isinstance(xml_text, (bytes, bytearray)):
            xml_text = xml_text.decode("utf-8", errors="replace")
        altitudes = [
            float(value)
            for value in re.findall(
                r"<AltitudeMeters>([-+]?[0-9]*\.?[0-9]+)</AltitudeMeters>",
                xml_text or "",
            )
        ]
    except Exception:
        return None

    if not altitudes:
        return None

    gain_meters = 0.0
    previous = altitudes[0]
    for altitude in altitudes[1:]:
        delta = altitude - previous
        if delta > 0:
            gain_meters += delta
        previous = altitude

    return gain_meters * 3.28084


def heart_rate_from_tcx(xml_text: str | bytes | bytearray | None) -> dict[str, int | None]:
    """Extract avg/max/min HR from TCX content."""
    try:
        import re

        if isinstance(xml_text, (bytes, bytearray)):
            xml_text = xml_text.decode("utf-8", errors="replace")
        values = re.findall(r"<HeartRateBpm><Value>(\d+)</Value></HeartRateBpm>", xml_text or "")
        if not values:
            values = re.findall(r"<HeartRateBpm>(\d+)</HeartRateBpm>", xml_text or "")
    except Exception:
        values = []

    if not values:
        return {"avghr": None, "maxhr": None, "minhr": None}

    heart_rates = [int(value) for value in values]
    ignored_start = min(120, len(heart_rates))
    min_slice = heart_rates[ignored_start:] or heart_rates
    return {
        "avghr": sum(heart_rates) // len(heart_rates),
        "maxhr": max(heart_rates),
        "minhr": min(min_slice),
    }


def extract_activity_type_key(activity: dict[str, Any]) -> str:
    raw_type = activity.get("activityType", "")
    if isinstance(raw_type, dict):
        return str(
            raw_type.get("typeKey")
            or raw_type.get("typeId")
            or raw_type.get("parentTypeId")
            or ""
        )
    return str(raw_type or "")


def map_activity_type(garmin_type: str | None) -> str | None:
    if not garmin_type:
        return None

    garmin_type_lower = garmin_type.lower().strip()
    cycling_keywords = (
        "cycling",
        "bike",
        "biking",
        "bicycle",
        "indoor_cycling",
        "road_biking",
        "mountain_biking",
        "virtual_cycling",
        "e_bike",
        "ebike",
    )
    if any(keyword in garmin_type_lower for keyword in cycling_keywords):
        return None
    if "treadmill" in garmin_type_lower:
        return "Treadmill run"

    running_keywords = ("running", "run", "street", "track", "trail", "ultra", "virtual", "indoor", "road")
    if any(keyword in garmin_type_lower for keyword in running_keywords):
        return "Run"
    return None


def detect_gps_signal(activity: dict[str, Any]) -> bool | None:
    gps_negative_signal = False
    candidates = [activity]
    summary_dto = activity.get("summaryDTO")
    if isinstance(summary_dto, dict):
        candidates.append(summary_dto)

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate.get("hasPolyline") is True or candidate.get("hasMap") is True:
            return True
        if candidate.get("polyline"):
            return True
        for lat_key, lon_key in (
            ("startLatitude", "startLongitude"),
            ("beginLatitude", "beginLongitude"),
            ("endLatitude", "endLongitude"),
        ):
            latitude = candidate.get(lat_key)
            longitude = candidate.get(lon_key)
            if latitude not in (None, "", 0, 0.0) and longitude not in (None, "", 0, 0.0):
                return True
        if candidate.get("hasPolyline") is False or candidate.get("hasMap") is False:
            gps_negative_signal = True

    return False if gps_negative_signal else None


def infer_activity_type(activity: dict[str, Any]) -> str | None:
    garmin_type = extract_activity_type_key(activity)
    mapped = map_activity_type(garmin_type)
    if mapped != "Run":
        return mapped

    garmin_type_lower = garmin_type.lower().strip()
    if any(keyword in garmin_type_lower for keyword in ("treadmill", "indoor", "virtual")):
        return "Treadmill run"

    gps_signal = detect_gps_signal(activity)
    if gps_signal is True:
        return "Run"
    if gps_signal is False:
        return "Treadmill run"
    return mapped


def iter_metric_sources(*activities: dict[str, Any] | None):
    for activity in activities:
        if not isinstance(activity, dict):
            continue
        yield activity
        summary_dto = activity.get("summaryDTO")
        if isinstance(summary_dto, dict):
            yield summary_dto
        heart_rate = activity.get("heartRate")
        if isinstance(heart_rate, dict):
            yield heart_rate


def extract_heart_rate_metrics(*activities: dict[str, Any] | None) -> dict[str, int | None]:
    avg_keys = ("averageHR", "averageHeartRate", "avgHeartRate")
    max_keys = ("maxHR", "maxHeartRate", "maximumHeartRate")
    min_keys = ("minHR", "minHeartRate", "minimumHeartRate")

    def lookup(*keys: str) -> int | None:
        for candidate in iter_metric_sources(*activities):
            for key in keys:
                value = as_int(candidate.get(key))
                if value and value > 0:
                    return value
        return None

    return {
        "avghr": lookup(*avg_keys),
        "maxhr": lookup(*max_keys),
        "minhr": lookup(*min_keys),
    }


def extract_elevation_gain_feet(*activities: dict[str, Any] | None) -> float | None:
    elevation_keys = (
        "elevationGain",
        "elevationGainMeters",
        "elevationGained",
        "elevationAscent",
        "elevationGainInMeters",
    )
    for activity in iter_metric_sources(*activities):
        for key in elevation_keys:
            meters = as_float(activity.get(key))
            if meters is not None:
                return meters * 3.28084
    return None


def parse_start_time_local(activity: dict[str, Any]) -> str | None:
    value = first_value(
        activity.get("startTimeLocal"),
        activity.get("summaryDTO", {}).get("startTimeLocal") if isinstance(activity.get("summaryDTO"), dict) else None,
    )
    if value is None:
        return None
    return str(value).replace(" ", "T")


def parse_activity_date(activity: dict[str, Any]) -> str:
    start_time_local = parse_start_time_local(activity)
    if start_time_local:
        return start_time_local[:10]
    return date.today().isoformat()


def download_activity_tcx(client, activity_id: int | str) -> str | bytes | None:
    if not hasattr(client, "download_activity"):
        return None

    def download():
        if hasattr(client, "ActivityDownloadFormat"):
            return client.download_activity(activity_id, dl_fmt=client.ActivityDownloadFormat.TCX)
        return client.download_activity(activity_id, dl_fmt="tcx")

    try:
        return garmin_request(f"activity download {activity_id}", download, max_attempts=2)
    except Exception:
        return None


def get_resting_heart_rate(date_str: str, client) -> int | None:
    """Fetch RHR for a single day, using the least expensive available endpoint."""
    if hasattr(client, "get_heart_rates"):
        try:
            response = garmin_request(
                f"resting heart rate lookup {date_str}",
                lambda: client.get_heart_rates(date_str),
                max_attempts=2,
            )
            if isinstance(response, dict):
                value = as_int(response.get("restingHeartRate"))
                if value:
                    return value
        except Exception:
            pass

    for method_name in ("get_daily_summary", "get_daily_summary_v2"):
        if hasattr(client, method_name):
            try:
                response = garmin_request(
                    f"daily summary lookup {date_str}",
                    lambda method_name=method_name: getattr(client, method_name)(date_str),
                    max_attempts=2,
                )
                if isinstance(response, dict):
                    value = as_int(response.get("restingHeartRate"))
                    if value:
                        return value
            except Exception:
                continue
    return None


def normalize_run_record(activity: dict[str, Any], client) -> dict[str, Any] | None:
    activity_type = infer_activity_type(activity)
    if activity_type not in ("Run", "Treadmill run"):
        return None

    activity_id = as_int(activity.get("activityId"))
    details = None
    if activity_id and hasattr(client, "get_activity"):
        try:
            details = garmin_request(
                f"activity detail lookup {activity_id}",
                lambda: client.get_activity(activity_id),
                max_attempts=2,
            )
        except Exception as error:
            print(f"  ⚠ Could not fetch details for activity {activity_id}: {error}")

    merged_activity = details if isinstance(details, dict) else activity
    date_str = parse_activity_date(merged_activity)
    start_time_local = parse_start_time_local(merged_activity) or parse_start_time_local(activity)

    distance_meters = as_float(
        first_value(
            merged_activity.get("distance"),
            merged_activity.get("distanceMeters"),
            activity.get("distance"),
            activity.get("distanceMeters"),
        )
    )
    if not distance_meters or distance_meters <= 0:
        print(f"  ⚠ Skipping activity {activity_id or 'unknown'} with missing distance")
        return None

    duration_seconds = as_float(
        first_value(
            merged_activity.get("duration"),
            merged_activity.get("elapsedDuration"),
            merged_activity.get("movingDuration"),
            activity.get("duration"),
            activity.get("elapsedDuration"),
        )
    )
    duration_ms = int(round(duration_seconds * 1000)) if duration_seconds else None

    heart_rate = extract_heart_rate_metrics(activity, details)
    elevation_gain_feet = extract_elevation_gain_feet(activity, details)
    needs_tcx = elevation_gain_feet is None or any(value is None for value in heart_rate.values())
    tcx_payload = download_activity_tcx(client, activity_id) if needs_tcx and activity_id else None

    if tcx_payload:
        tcx_heart_rate = heart_rate_from_tcx(tcx_payload)
        for key, value in tcx_heart_rate.items():
            if heart_rate.get(key) is None and value is not None:
                heart_rate[key] = value
        if elevation_gain_feet is None:
            elevation_gain_feet = elevation_gain_from_tcx(tcx_payload)

    if activity_type == "Treadmill run" and elevation_gain_feet is None:
        elevation_gain_feet = 0.0

    return {
        "date": date_str,
        "activity_id": activity_id,
        "start_time_local": start_time_local,
        "activity_name": first_value(merged_activity.get("activityName"), activity.get("activityName")),
        "distance": distance_meters / 1609.34,
        "duration_ms": duration_ms,
        "steps": as_int(first_value(merged_activity.get("steps"), activity.get("steps"), merged_activity.get("stepsCount"), activity.get("stepsCount"))),
        "calories": as_int(first_value(merged_activity.get("calories"), activity.get("calories"), merged_activity.get("caloriesConsumed"), activity.get("caloriesConsumed"))),
        "resting_hr": get_resting_heart_rate(date_str, client),
        "elev_gain": elevation_gain_feet,
        "activity_type": activity_type,
        **heart_rate,
    }


def sync_garmin_data(client, db_path: str | Path | None = None, today: date | None = None) -> dict[str, Any]:
    """Synchronize the recent Garmin activity window into the local SQLite DB."""
    resolved_db_path = initialize_database(db_path or DB_PATH)
    sync_start, sync_end = get_sync_window(resolved_db_path, today=today)
    print(f"Syncing Garmin activities from {sync_start.isoformat()} to {sync_end.isoformat()}...")

    activities = garmin_request(
        "activity search",
        lambda: client.get_activities_by_date(sync_start.isoformat(), sync_end.isoformat()),
    )
    activities = list(activities or [])
    print(f"Fetched {len(activities)} total Garmin activities in window")

    synced_runs = []
    seen_activity_ids = set()
    for activity in sorted(activities, key=lambda item: parse_start_time_local(item) or ""):
        activity_id = as_int(activity.get("activityId"))
        if activity_id is not None and activity_id in seen_activity_ids:
            continue

        normalized = normalize_run_record(activity, client)
        if not normalized:
            continue

        upsert_run(normalized, resolved_db_path)
        if activity_id is not None:
            seen_activity_ids.add(activity_id)
        synced_runs.append(normalized)
        print(
            "  ✓ Synced",
            normalized["date"],
            normalized["activity_type"],
            f"{float(normalized['distance']):.2f} mi",
            f"(activity {normalized.get('activity_id') or 'n/a'})",
        )

    set_sync_state("last_activity_sync_end", sync_end.isoformat(), resolved_db_path)
    latest_run = get_latest_run(resolved_db_path)
    return {
        "db_path": str(resolved_db_path),
        "sync_start": sync_start.isoformat(),
        "sync_end": sync_end.isoformat(),
        "activity_count": len(activities),
        "run_count": len(synced_runs),
        "latest_run": latest_run,
    }


def main() -> int:
    load_dotenv(SCRIPT_DIR / ".env")
    email = os.getenv("GARMIN_EMAIL")
    password = os.getenv("GARMIN_PASSWORD")
    if not email or not password:
        print("ERROR: set GARMIN_EMAIL and GARMIN_PASSWORD in .env")
        return 1

    initialize_database(DB_PATH)
    print(f"Local store ready at {DB_PATH}")
    print(f"Bootstrap start date: {DEFAULT_BOOTSTRAP_START_DATE.isoformat()}")

    try:
        client = login_to_garmin(email, password)
        print("✓ Garmin authentication successful")
    except Exception as error:
        print(f"✗ Garmin authentication failed: {error}")
        return 1

    try:
        result = sync_garmin_data(client, db_path=DB_PATH)
    except Exception as error:
        print(f"✗ Garmin activity sync failed: {error}")
        return 1

    print(
        f"✓ Sync finished: {result['run_count']} run(s) updated "
        f"from {result['sync_start']} to {result['sync_end']}"
    )
    print(f"Latest cached run: {format_run_summary(result['latest_run'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
