#!/usr/bin/env python3
"""Build a memory-aware weekly running coach context for the cron LLM.

This script intentionally does zero Garmin/Withings API calls. It reads local logs,
Garmin cache.db, Honcho/session-search excerpts when the cron agent chooses to add
those, and durable running context files.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
import subprocess
import sys

SCRIPT_DIR = Path(__file__).parent.absolute()
DB_PATH = SCRIPT_DIR / "cache.db"
WORKSPACE = Path.home() / ".hermes" / "workspace"
CONTEXT_RUNNING = Path.home() / ".hermes" / "context" / "running.md"
RHR_LOG = WORKSPACE / "rhr_log.jsonl"
BODY_METRICS_LOG = WORKSPACE / "body_metrics_log.jsonl"
GARMIN_LOG = WORKSPACE / "garmin_log.jsonl"
OUTPUT_PATH = Path("/tmp/weekly_summary/weekly_coach_context.md")


def week_bounds(anchor: date | None = None) -> tuple[date, date]:
    today = anchor or date.today()
    monday = today - timedelta(days=today.weekday())
    return monday, monday + timedelta(days=6)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Some older garmin_log rows were accidentally saved with prefixed line numbers.
        if "|" in line and line.split("|", 1)[0].strip().isdigit():
            line = line.split("|", 1)[1].strip()
        try:
            value = json.loads(line)
        except Exception:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def load_runs(db_path: Path = DB_PATH) -> list[dict]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM runs WHERE activity_type != 'None' ORDER BY date DESC LIMIT 200")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows


def latest_body_by_day(rows: list[dict]) -> dict[str, dict]:
    by_day: dict[str, dict] = {}
    for row in sorted(rows, key=lambda r: (str(r.get("date") or ""), str(r.get("grpid") or ""))):
        date_str = row.get("date")
        metrics = row.get("metrics")
        if not date_str or not isinstance(metrics, dict):
            continue
        merged = by_day.setdefault(str(date_str), {})
        for key, value in metrics.items():
            if value is not None:
                merged[key] = value
    return by_day


def pace_to_seconds(pace: str | None) -> int | None:
    if not pace:
        return None
    try:
        parts = [int(p) for p in pace.split(":")]
    except Exception:
        return None
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    return None


def seconds_to_pace(seconds: float | int | None) -> str:
    if seconds is None:
        return "—"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}/mi"


def fmt(value, suffix="", decimals=1) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return "—"


def collect_honcho_running_memory() -> str:
    """Best-effort retrieval of conversational running memory from Hermes/Honcho CLI.

    This is intentionally optional: the weekly cron's LLM prompt also tells the agent
    to call Honcho/session memory tools directly when available. This subprocess path
    gives the markdown context file a chance to include those excerpts too without
    making the chart/data generation depend on Hermes internals.
    """
    commands = [
        ["hermes", "honcho", "search", "running OR run OR Garmin OR PT OR stress fracture OR gait", "--max-tokens", "1200"],
        [sys.executable, "-m", "hermes", "honcho", "search", "running OR run OR Garmin OR PT OR stress fracture OR gait", "--max-tokens", "1200"],
    ]
    for command in commands:
        try:
            proc = subprocess.run(command, capture_output=True, text=True, timeout=20)
        except Exception:
            continue
        output = (proc.stdout or "").strip()
        if proc.returncode == 0 and output:
            return output[:4000]
    return "Honcho CLI excerpts unavailable here. Cron agent should use Honcho/session_search tools directly before writing the final coaching digest."


def main() -> int:
    monday, sunday = week_bounds()
    start = monday.isoformat()
    end = sunday.isoformat()

    runs = load_runs()
    week_runs = [r for r in runs if start <= str(r.get("date")) <= end]
    week_runs = sorted(week_runs, key=lambda r: str(r.get("date")))
    total_miles = sum(float(r.get("distance") or 0) for r in week_runs)
    avg_hr_values = [int(r["avghr"]) for r in week_runs if r.get("avghr")]
    avg_pace_values = [seconds for r in week_runs if (seconds := pace_to_seconds(r.get("avg_pace"))) is not None]

    rhr_rows = read_jsonl(RHR_LOG)
    week_rhr = [r for r in rhr_rows if start <= str(r.get("date")) <= end]
    latest_rhr = week_rhr[-1] if week_rhr else (rhr_rows[-1] if rhr_rows else None)

    body_rows = read_jsonl(BODY_METRICS_LOG)
    body_by_day = latest_body_by_day(body_rows)
    week_body = {d: m for d, m in body_by_day.items() if start <= d <= end}
    latest_body_date = max(body_by_day.keys()) if body_by_day else None
    latest_body = body_by_day.get(latest_body_date, {}) if latest_body_date else {}

    debrief_rows = read_jsonl(GARMIN_LOG)
    week_debriefs = [r for r in debrief_rows if start <= str(r.get("date")) <= end]
    recent_debriefs = debrief_rows[-8:]

    running_context = CONTEXT_RUNNING.read_text(encoding="utf-8") if CONTEXT_RUNNING.exists() else ""
    honcho_memory = collect_honcho_running_memory()

    lines: list[str] = []
    lines.append("# Weekly Running Coach Context")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Week: {start} to {end}")
    lines.append("")
    lines.append("## Durable running context / memory")
    lines.append(running_context.strip() or "No running.md context found.")
    lines.append("")
    lines.append("## Conversational running memory excerpts")
    lines.append(honcho_memory.strip())
    lines.append("")
    lines.append("## This week's run stats from Garmin cache.db")
    lines.append(f"- Runs: {len(week_runs)}")
    lines.append(f"- Total mileage: {total_miles:.2f} mi")
    lines.append(f"- Average run HR: {round(sum(avg_hr_values)/len(avg_hr_values)) if avg_hr_values else '—'} bpm")
    lines.append(f"- Average pace: {seconds_to_pace(sum(avg_pace_values)/len(avg_pace_values)) if avg_pace_values else '—'}")
    if week_runs:
        lines.append("")
        for run in week_runs:
            lines.append(
                f"- {run.get('date')}: {fmt(run.get('distance'), ' mi', 2)}, "
                f"pace {run.get('avg_pace') or '—'}, avg HR {run.get('avghr') or '—'}, "
                f"max HR {run.get('maxhr') or '—'}, cadence {run.get('steps') or '—'} steps total"
            )
    else:
        lines.append("- No Garmin runs logged in cache.db this week.")
    lines.append("")
    lines.append("## This week's resting HR")
    if week_rhr:
        for row in week_rhr:
            lines.append(f"- {row.get('date')}: {row.get('rhr')} bpm")
    elif latest_rhr:
        lines.append(f"- No RHR rows this week yet. Latest: {latest_rhr.get('date')} {latest_rhr.get('rhr')} bpm")
    else:
        lines.append("- No RHR data found.")
    lines.append("")
    lines.append("## Withings body metrics")
    if week_body:
        for d in sorted(week_body):
            m = week_body[d]
            lines.append(
                f"- {d}: weight {fmt(m.get('weight_lb'), ' lb', 1)} / {fmt(m.get('weight_kg'), ' kg', 2)}, "
                f"BMI {fmt(m.get('bmi'), '', 2)}, fat {fmt(m.get('fat_ratio_pct'), '%', 1)}, "
                f"muscle {fmt(m.get('muscle_mass_kg'), ' kg', 1)}, bone {fmt(m.get('bone_mass_kg'), ' kg', 2)}, "
                f"visceral fat {fmt(m.get('visceral_fat_index'), '', 1)}"
            )
    elif latest_body_date:
        m = latest_body
        lines.append(
            f"- No body metrics in current week window. Latest {latest_body_date}: "
            f"weight {fmt(m.get('weight_lb'), ' lb', 1)}, BMI {fmt(m.get('bmi'), '', 2)}, "
            f"fat {fmt(m.get('fat_ratio_pct'), '%', 1)}, muscle {fmt(m.get('muscle_mass_kg'), ' kg', 1)}"
        )
    else:
        lines.append("- No Withings body metrics found.")
    lines.append("")
    lines.append("## Garmin run debrief memory from garmin_log.jsonl")
    source_debriefs = week_debriefs or recent_debriefs
    if source_debriefs:
        label = "this week" if week_debriefs else "recent fallback because this week has no logged debrief rows"
        lines.append(f"Using {label}:")
        for row in source_debriefs:
            lines.append(
                f"- {row.get('date')}: {row.get('activity_type', 'Run')}, {row.get('distance')} mi, "
                f"avg HR {row.get('avg_hr')}, RHR {row.get('resting_hr')}. Notes: {row.get('notes', '')}"
            )
    else:
        lines.append("- No garmin_log.jsonl run debrief rows found.")
    lines.append("")
    lines.append("## Required coaching style")
    lines.append("- Act like an elite running coach but keep it concise and direct.")
    lines.append("- Stress fracture return remains the key constraint unless durable context says he was cleared.")
    lines.append("- Do not prescribe speedwork/hills/hard sessions unless cleared by PT.")
    lines.append("- Always account for Ben's earlier-week subjective context/debrief notes if present.")
    lines.append("- Include: what went well, what is concerning if anything, breakdown of week, next-week outlook/expected plan.")
    lines.append("- No markdown tables; Discord-friendly bullets only.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
