# GarminDataReader

Personal running data pipeline for Ben Bravo. Pulls activity data from Garmin Connect, stores it locally, and feeds a weekly summary chart delivered to Discord every Sunday — all automated, zero manual steps.

## What this actually does

I'm returning from bilateral tibial stress fractures (Jan 2026) and training toward the Annapolis 10 Mile in August 2026. This repo is the data layer that backs my AI training assistant (Hermes). Every run I log on my Garmin FR165 flows through here automatically.

**The pipeline:**

```
Garmin Connect → update.py → cache.db → weekly_summary.py → Discord
                                      ↘ runs_data.csv
```

- `update.py` — orchestrator. Runs auth, fetches new activities, exports CSV. Run this after any workout.
- `db_filler.py` — pulls activities from Garmin API, fills `cache.db` (SQLite). Handles treadmill vs outdoor detection automatically from GPS/activity type metadata (no user prompts).
- `db_to_csv.py` — exports `cache.db` to `runs_data.csv`
- `pull_rhr.py` — nightly cron job (11pm) that pulls resting HR from Garmin and appends to `~/.hermes/workspace/rhr_log.jsonl`. Skips if already logged today.
- `weekly_summary.py` — generates a PNG chart (mileage, HR, pace trends, RHR trend). Triggered by a Sunday 9pm cron job that posts the image to Discord.
- `garmin_auth.py` — centralized auth with exponential backoff/retry on 429s. All scripts route through here.

## Automation

Two cron jobs handle everything:

| Job | Schedule | What it does |
|-----|----------|--------------|
| RHR pull | Nightly 11pm | `pull_rhr.py` → appends RHR to `rhr_log.jsonl`, skips if run was already logged today |
| Weekly chart | Sundays 9pm | `weekly_summary.py` → posts PNG chart to Discord |

After a run, Hermes runs `update.py`, debriefs the session, and pushes the updated DB and CSV to this repo.

## Setup

1. Clone the repo and create a venv:
   ```bash
   python3 -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Create `.env` in the project root:
   ```
   GARMIN_EMAIL=your-email@example.com
   GARMIN_PASSWORD=your-password
   ```

3. Run the pipeline:
   ```bash
   python update.py
   ```

That's it. Scripts auto-bootstrap the venv on re-entry so cron jobs work without manual activation.

## Data

- `cache.db` — SQLite, source of truth. Contains all runs with HR, pace, cadence, elevation, calories, activity type.
- `runs_data.csv` — flat export for spreadsheets or quick analysis
- `~/.hermes/workspace/rhr_log.jsonl` — daily RHR log (lives outside the repo, managed by Hermes)
- `~/.hermes/workspace/garmin_log.jsonl` — per-run debrief log written by Hermes after each session

## Notes

- Garmin rate-limits aggressively (HTTP 429). `garmin_auth.py` handles this with backoff, but avoid hammering the API multiple times in one day.
- Treadmill runs are detected automatically from Garmin activity type and GPS presence — no prompts, safe for cron.
- `update.py` also writes today's RHR into `rhr_log.jsonl` after `db_filler.py` runs, so the nightly cron skips cleanly on run days.
- Start date is set in `db_filler.py` (currently Apr 1 2026 — fresh start post-injury).
