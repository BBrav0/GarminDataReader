#!/usr/bin/env python3
"""Pull latest Withings smart scale/body metrics.

Appends one JSON line per Withings measurement group to:
  ~/.hermes/workspace/body_metrics_log.jsonl

Requires:
  /Users/bensopenclaw/garmin/.env with WITHINGS_CLIENT_ID / WITHINGS_CLIENT_SECRET
  ~/.hermes/secrets/withings_tokens.json created by withings_auth.py

Optional in .env:
  BEN_HEIGHT_M=1.78  # used to compute BMI locally from weight
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).parent.absolute()
TOKEN_PATH = Path.home() / ".hermes" / "secrets" / "withings_tokens.json"
LOG_PATH = Path.home() / ".hermes" / "workspace" / "body_metrics_log.jsonl"
API_ENDPOINT = "https://wbsapi.withings.net"
MEASURE_TYPES = {
    1: ("weight_kg", "kg"),
    4: ("height_m", "m"),
    5: ("fat_free_mass_kg", "kg"),
    6: ("fat_ratio_pct", "%"),
    8: ("fat_mass_kg", "kg"),
    11: ("heart_pulse_bpm", "bpm"),
    76: ("muscle_mass_kg", "kg"),
    77: ("hydration_kg", "kg"),
    88: ("bone_mass_kg", "kg"),
    91: ("pulse_wave_velocity_m_s", "m/s"),
    170: ("visceral_fat_index", "index"),
}
MEASTYPES_PARAM = ",".join(str(k) for k in sorted(MEASURE_TYPES))


def ensure_venv() -> None:
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        return
    for env_name in (".venv", "venv"):
        venv_python = SCRIPT_DIR / env_name / "bin" / "python3"
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)


def sign(params: dict[str, object], client_secret: str) -> str:
    params_to_sign = {
        "action": params["action"],
        "client_id": params["client_id"],
    }
    if "nonce" in params and params["nonce"]:
        params_to_sign["nonce"] = params["nonce"]
    if "timestamp" in params and params["timestamp"]:
        params_to_sign["timestamp"] = params["timestamp"]
    data = ",".join(str(params_to_sign[key]) for key in sorted(params_to_sign))
    return hmac.new(client_secret.encode(), data.encode(), hashlib.sha256).hexdigest()


def get_nonce(client_id: str, client_secret: str) -> str:
    params: dict[str, object] = {
        "action": "getnonce",
        "client_id": client_id,
        "timestamp": int(time.time()),
    }
    params["signature"] = sign(params, client_secret)
    resp = requests.post(f"{API_ENDPOINT}/v2/signature", data=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 0:
        raise RuntimeError(f"Withings getnonce failed: status={payload.get('status')} error={payload.get('error')}")
    return payload["body"]["nonce"]


def load_config() -> tuple[str, str, float | None]:
    load_dotenv(SCRIPT_DIR / ".env")
    client_id = os.getenv("WITHINGS_CLIENT_ID")
    client_secret = os.getenv("WITHINGS_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise SystemExit("ERROR: WITHINGS_CLIENT_ID / WITHINGS_CLIENT_SECRET not set in /Users/bensopenclaw/garmin/.env")
    height_raw = os.getenv("BEN_HEIGHT_M")
    height_m = None
    if height_raw:
        try:
            height_m = float(height_raw)
        except ValueError:
            raise SystemExit("ERROR: BEN_HEIGHT_M must be a number in meters, e.g. 1.78")
    return client_id, client_secret, height_m


def load_tokens() -> dict[str, object]:
    if not TOKEN_PATH.exists():
        raise SystemExit(f"ERROR: missing {TOKEN_PATH}; run withings_auth.py first")
    return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))


def save_tokens(tokens: dict[str, object]) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(tokens, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    TOKEN_PATH.chmod(0o600)


def refresh_tokens(client_id: str, client_secret: str, tokens: dict[str, object]) -> dict[str, object]:
    refresh_token = str(tokens.get("refresh_token") or "")
    if not refresh_token:
        raise RuntimeError("Withings refresh_token missing")
    params: dict[str, object] = {
        "action": "requesttoken",
        "client_id": client_id,
        "client_secret": client_secret,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    resp = requests.post(f"{API_ENDPOINT}/v2/oauth2", data=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("status") != 0:
        raise RuntimeError(f"Withings refresh failed: status={payload.get('status')} error={payload.get('error')}")
    new_tokens = dict(tokens)
    new_tokens.update(payload["body"])
    new_tokens["obtained_at"] = int(time.time())
    save_tokens(new_tokens)
    return new_tokens


def token_expired(tokens: dict[str, object]) -> bool:
    obtained_at = int(tokens.get("obtained_at") or 0)
    expires_in = int(tokens.get("expires_in") or 0)
    if not obtained_at or not expires_in:
        return True
    return time.time() >= obtained_at + expires_in - 300


def withings_value(measure: dict[str, object]) -> float:
    return float(measure["value"]) * (10 ** int(measure["unit"]))


def fetch_measure_groups(access_token: str, start_ts: int, end_ts: int) -> list[dict[str, object]]:
    groups: list[dict[str, object]] = []
    offset = 0
    while True:
        data = {
            "action": "getmeas",
            "meastypes": MEASTYPES_PARAM,
            "category": 1,
            "startdate": start_ts,
            "enddate": end_ts,
            "offset": offset,
        }
        resp = requests.post(
            f"{API_ENDPOINT}/measure",
            data=data,
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != 0:
            raise RuntimeError(f"Withings getmeas failed: status={payload.get('status')} error={payload.get('error')}")
        body = payload.get("body", {})
        groups.extend(body.get("measuregrps", []))
        if body.get("more") == 1 and body.get("offset") is not None:
            offset = body["offset"]
            continue
        return groups


def load_log_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if LOG_PATH.exists():
        for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
            except Exception:
                pass
    return rows


def existing_grpids(rows: list[dict[str, object]]) -> set[str]:
    seen: set[str] = set()
    for row in rows:
        if row.get("grpid") is not None:
            seen.add(str(row["grpid"]))
    return seen


def latest_height_m(rows: list[dict[str, object]]) -> float | None:
    for row in reversed(rows):
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and metrics.get("height_m"):
            try:
                height_m = float(metrics["height_m"])
            except (TypeError, ValueError):
                continue
            if height_m > 0:
                return height_m
    return None


def height_from_groups(groups: list[dict[str, object]]) -> float | None:
    for group in sorted(groups, key=lambda g: int(g.get("date", 0)), reverse=True):
        for measure in group.get("measures", []):
            if not isinstance(measure, dict):
                continue
            if int(measure.get("type", 0)) == 4:
                height_m = withings_value(measure)
                if height_m > 0:
                    return height_m
    return None


def add_missing_derived_metrics(rows: list[dict[str, object]], height_m: float | None) -> bool:
    changed = False
    for row in rows:
        metrics = row.get("metrics", {})
        if not isinstance(metrics, dict):
            continue

        weight_kg = metrics.get("weight_kg")
        if height_m and height_m > 0 and weight_kg is not None and metrics.get("bmi") is None:
            try:
                metrics["bmi"] = round(float(weight_kg) / (height_m * height_m), 2)
                changed = True
            except (TypeError, ValueError):
                pass

        muscle_mass_kg = metrics.get("muscle_mass_kg")
        if muscle_mass_kg is not None and metrics.get("muscle_mass_lb") is None:
            try:
                metrics["muscle_mass_lb"] = round(float(muscle_mass_kg) * 2.2046226218, 2)
                changed = True
            except (TypeError, ValueError):
                pass
    return changed


def group_to_record(group: dict[str, object], height_m: float | None) -> dict[str, object]:
    metrics: dict[str, float] = {}
    for measure in group.get("measures", []):
        if not isinstance(measure, dict):
            continue
        measure_type = int(measure.get("type", 0))
        if measure_type not in MEASURE_TYPES:
            continue
        key, _unit = MEASURE_TYPES[measure_type]
        metrics[key] = round(withings_value(measure), 4)

    weight_kg = metrics.get("weight_kg")
    height_m = metrics.get("height_m") or height_m
    if weight_kg is not None:
        metrics["weight_lb"] = round(weight_kg * 2.2046226218, 2)
        if height_m and height_m > 0:
            metrics["bmi"] = round(weight_kg / (height_m * height_m), 2)

    muscle_mass_kg = metrics.get("muscle_mass_kg")
    if muscle_mass_kg is not None:
        metrics["muscle_mass_lb"] = round(muscle_mass_kg * 2.2046226218, 2)

    ts = int(group["date"])
    recorded_at = datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().isoformat()
    return {
        "date": datetime.fromtimestamp(ts, tz=timezone.utc).astimezone().date().isoformat(),
        "recorded_at": recorded_at,
        "timestamp": ts,
        "grpid": str(group.get("grpid")),
        "category": group.get("category"),
        "attrib": group.get("attrib"),
        "model": group.get("model"),
        "modelid": group.get("modelid"),
        "deviceid_hash": group.get("hash_deviceid"),
        "metrics": metrics,
    }


def fmt(value: object, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value}{suffix}"


def main() -> None:
    ensure_venv()
    client_id, client_secret, height_m = load_config()
    tokens = load_tokens()
    if token_expired(tokens):
        tokens = refresh_tokens(client_id, client_secret, tokens)

    now = datetime.now(timezone.utc)
    start = now - timedelta(days=14)
    groups = fetch_measure_groups(str(tokens["access_token"]), int(start.timestamp()), int(now.timestamp()))
    existing_records = load_log_records()
    effective_height_m = height_m or height_from_groups(groups) or latest_height_m(existing_records)
    if add_missing_derived_metrics(existing_records, effective_height_m):
        LOG_PATH.write_text(
            "".join(json.dumps(record, sort_keys=True) + "\n" for record in existing_records),
            encoding="utf-8",
        )

    seen = existing_grpids(existing_records)
    new_records = [
        group_to_record(group, effective_height_m)
        for group in sorted(groups, key=lambda g: int(g.get("date", 0)))
    ]
    new_records = [record for record in new_records if record["grpid"] not in seen]

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if new_records:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            for record in new_records:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    all_records = existing_records + new_records

    if not all_records:
        print("Withings: no scale data yet")
        return

    latest = next(
        (
            record
            for record in reversed(all_records)
            if isinstance(record.get("metrics"), dict)
            and (
                record["metrics"].get("weight_kg") is not None  # type: ignore[index, union-attr]
                or record["metrics"].get("fat_ratio_pct") is not None  # type: ignore[index, union-attr]
                or record["metrics"].get("muscle_mass_kg") is not None  # type: ignore[index, union-attr]
            )
        ),
        all_records[-1],
    )
    metrics_raw = latest.get("metrics", {})
    metrics = metrics_raw if isinstance(metrics_raw, dict) else {}
    print(
        "Withings "
        f"{latest.get('date')}: "
        f"{fmt(metrics.get('weight_lb'), ' lb')}"
        f" / {fmt(metrics.get('weight_kg'), ' kg')}"
        f" • BMI {fmt(metrics.get('bmi'))}"
        f" • fat {fmt(metrics.get('fat_ratio_pct'), '%')}"
        f" • muscle {fmt(metrics.get('muscle_mass_lb'), ' lb')}"
        + (f" ({len(new_records)} new)" if new_records else " (already logged)")
    )


if __name__ == "__main__":
    main()
