#!/usr/bin/env python3
"""Re-render the weekly charts as if run at the scheduled time.

Usage: python replay_weekly_summary.py 2026-08-23

Anchors generation to the given Sunday so 'this week' means the week the
Sunday-night cron would have covered (anchor 2026-08-23 -> week Aug 18-24),
instead of whenever the re-run happens. Overwrites the PNGs in
/tmp/weekly_summary/.
"""
import sys
from datetime import date

import weekly_summary

anchor = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else None
print("anchor:", anchor)
paths = weekly_summary.generate_weekly_charts(today=anchor)
print(f"Saved {len(paths)} charts")
for p in paths:
    print(f" - {p}")
