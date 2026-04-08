#!/usr/bin/env python3
"""Regression tests for the Garmin sync pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from db_filler import sync_garmin_data
from db_to_csv import export_runs_to_csv
from garmin_store import get_latest_run, get_sync_window, initialize_database, set_sync_state


class FakeGarminClient:
    def __init__(self, activities, details=None, tcx_payloads=None, heart_rates=None):
        self.activities = activities
        self.details = details or {}
        self.tcx_payloads = tcx_payloads or {}
        self.heart_rates = heart_rates or {}

    def get_activities_by_date(self, startdate, enddate):
        return list(self.activities)

    def get_activity(self, activity_id):
        return self.details.get(activity_id, {})

    def download_activity(self, activity_id, dl_fmt=None):
        return self.tcx_payloads.get(activity_id, b"")

    def get_heart_rates(self, cdate):
        return {"restingHeartRate": self.heart_rates.get(cdate)}


class GarminSyncTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tempdir.name) / "cache.db"
        initialize_database(self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_sync_stores_only_runs_and_records_latest_run(self):
        activities = [
            {
                "activityId": 101,
                "activityType": {"typeKey": "running"},
                "distance": 8046.72,
                "duration": 2400,
                "startTimeLocal": "2026-04-08 07:10:00",
                "steps": 5100,
                "calories": 580,
                "hasMap": True,
            },
            {
                "activityId": 202,
                "activityType": {"typeKey": "cycling"},
                "distance": 12000,
                "duration": 1800,
                "startTimeLocal": "2026-04-08 12:30:00",
            },
        ]
        details = {
            101: {
                "activityName": "Morning Run",
                "summaryDTO": {
                    "startTimeLocal": "2026-04-08 07:10:00",
                    "averageHR": 152,
                    "maxHR": 167,
                    "minHR": 133,
                    "elevationGain": 62,
                },
            }
        }
        client = FakeGarminClient(
            activities=activities,
            details=details,
            heart_rates={"2026-04-08": 47},
        )

        result = sync_garmin_data(client, db_path=self.db_path, today=date(2026, 4, 8))

        self.assertEqual(result["activity_count"], 2)
        self.assertEqual(result["run_count"], 1)

        latest_run = get_latest_run(self.db_path)
        self.assertIsNotNone(latest_run)
        self.assertEqual(latest_run["date"], "2026-04-08")
        self.assertEqual(latest_run["activity_id"], 101)
        self.assertEqual(latest_run["activity_type"], "Run")
        self.assertEqual(latest_run["resting_hr"], 47)
        self.assertEqual(latest_run["avghr"], 152)

    def test_treadmill_run_uses_tcx_fallback_and_csv_remains_queryable(self):
        activities = [
            {
                "activityId": 303,
                "activityType": {"typeKey": "treadmill_running"},
                "distance": 6437.36,
                "duration": 2100,
                "startTimeLocal": "2026-04-09 18:25:00",
                "steps": 4300,
                "hasMap": False,
            }
        ]
        tcx_payload = b"""
        <TrainingCenterDatabase>
          <Trackpoint><HeartRateBpm><Value>145</Value></HeartRateBpm></Trackpoint>
          <Trackpoint><HeartRateBpm><Value>148</Value></HeartRateBpm></Trackpoint>
          <Trackpoint><HeartRateBpm><Value>150</Value></HeartRateBpm></Trackpoint>
        </TrainingCenterDatabase>
        """
        client = FakeGarminClient(
            activities=activities,
            details={303: {"summaryDTO": {"startTimeLocal": "2026-04-09 18:25:00"}}},
            tcx_payloads={303: tcx_payload},
            heart_rates={"2026-04-09": 49},
        )

        sync_garmin_data(client, db_path=self.db_path, today=date(2026, 4, 9))

        latest_run = get_latest_run(self.db_path)
        self.assertEqual(latest_run["activity_type"], "Treadmill run")
        self.assertEqual(latest_run["resting_hr"], 49)
        self.assertGreater(latest_run["avghr"], 0)
        self.assertEqual(latest_run["elev_gain"], 0.0)

        csv_path = Path(self.tempdir.name) / "runs_data.csv"
        record_count = export_runs_to_csv(self.db_path, csv_path)
        self.assertEqual(record_count, 1)
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertGreaterEqual(len(rows), 2)
        self.assertIn("start_time_local", rows[0])

    def test_sync_window_uses_recent_lookback_instead_of_full_rescan(self):
        set_sync_state("last_activity_sync_end", "2026-04-21", self.db_path)
        sync_start, sync_end = get_sync_window(self.db_path, today=date(2026, 4, 22))
        self.assertEqual(sync_start.isoformat(), "2026-04-07")
        self.assertEqual(sync_end.isoformat(), "2026-04-22")


if __name__ == "__main__":
    unittest.main()
