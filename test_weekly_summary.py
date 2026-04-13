#!/usr/bin/env python3
"""Tests for standalone weekly summary chart generation."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from garmin_store import initialize_database, upsert_run
from weekly_summary import CHART_SPECS, generate_weekly_charts


class WeeklySummaryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.tempdir.name)
        self.db_path = self.temp_path / "cache.db"
        self.rhr_log_path = self.temp_path / "rhr_log.jsonl"
        self.output_dir = self.temp_path / "charts"
        self.legacy_output_path = self.temp_path / "weekly_summary.png"
        initialize_database(self.db_path)

    def tearDown(self):
        self.tempdir.cleanup()

    def seed_runs(self):
        runs = [
            ("2026-02-23", 3.0, 1560_000, 145, 160, 48),
            ("2026-03-02", 3.5, 1770_000, 148, 165, 47),
            ("2026-03-09", 4.0, 1980_000, 150, 168, 47),
            ("2026-03-16", 4.4, 2130_000, 151, 170, 46),
            ("2026-03-23", 4.8, 2280_000, 152, 171, 46),
            ("2026-03-30", 5.0, 2340_000, 153, 172, 45),
            ("2026-04-06", 4.2, 1920_000, 149, 166, 45),
            ("2026-04-08", 5.1, 2340_000, 154, 173, 45),
            ("2026-04-11", 6.0, 2760_000, 158, 178, 44),
        ]
        for index, (run_date, miles, duration_ms, avg_hr, max_hr, resting_hr) in enumerate(runs, start=1):
            upsert_run(
                {
                    "date": run_date,
                    "activity_id": 1000 + index,
                    "start_time_local": f"{run_date} 07:00:00",
                    "activity_name": f"Run {index}",
                    "distance": miles,
                    "duration_ms": duration_ms,
                    "steps": int(miles * 1800),
                    "minhr": avg_hr - 15,
                    "maxhr": max_hr,
                    "avghr": avg_hr,
                    "calories": int(miles * 100),
                    "resting_hr": resting_hr,
                    "elev_gain": miles * 40,
                    "activity_type": "Run",
                },
                db_path=self.db_path,
            )

        self.rhr_log_path.write_text(
            "\n".join(
                [
                    '{"date":"2026-04-06","rhr":45}',
                    '{"date":"2026-04-07","rhr":46}',
                    '{"date":"2026-04-08","rhr":45}',
                    '{"date":"2026-04-09","rhr":44}',
                    '{"date":"2026-04-10","rhr":44}',
                    '{"date":"2026-04-11","rhr":44}',
                    '{"date":"2026-04-12","rhr":43}',
                ]
            ),
            encoding="utf-8",
        )

    def test_generate_weekly_charts_writes_separate_pngs(self):
        self.seed_runs()
        self.legacy_output_path.write_text("legacy", encoding="utf-8")

        saved_paths = generate_weekly_charts(
            db_path=self.db_path,
            rhr_log_path=self.rhr_log_path,
            output_dir=self.output_dir,
            today=date(2026, 4, 12),
            legacy_output_path=self.legacy_output_path,
        )

        self.assertEqual(len(saved_paths), len(CHART_SPECS))
        self.assertEqual({path.name for path in saved_paths}, {name for name, _ in CHART_SPECS})
        for path in saved_paths:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 1_000, path)
        self.assertFalse(self.legacy_output_path.exists())

    def test_generate_weekly_charts_handles_empty_state(self):
        saved_paths = generate_weekly_charts(
            db_path=self.db_path,
            rhr_log_path=self.rhr_log_path,
            output_dir=self.output_dir,
            today=date(2026, 4, 12),
            legacy_output_path=self.legacy_output_path,
        )

        self.assertEqual(len(saved_paths), len(CHART_SPECS))
        for path in saved_paths:
            self.assertTrue(path.exists(), path)
            self.assertGreater(path.stat().st_size, 1_000, path)


if __name__ == "__main__":
    unittest.main()
