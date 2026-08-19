#!/usr/bin/env python3
"""Daily running calendar strip for Discord.

Renders a 6-day window (2 past, today highlighted, 3 future) from the running
calendar markdown into a square dark-theme PNG. Used by the daily running coach cron.

Usage:
    python3 daily_calendar_strip.py                     # today, 2 past, 3 future
    python3 daily_calendar_strip.py --date 2026-08-18   # specific anchor day
Output default: /tmp/running_calendar_strip.png
"""

import argparse
import re
import textwrap
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

CALENDAR_PATH = "/Users/bensopenclaw/.hermes/context/running-calendar.md"
BLOCK_START = date(2026, 8, 18)  # marathon block day 1
RACE_DATE_EST = date(2027, 11, 14)  # YMCA Harrisburg Marathon 2027 (estimated)

# Discord dark theme palette
BG = "#1e1f22"
CARD = "#2b2d31"
CARD_DIM = "#26272b"
CARD_TODAY = "#31333a"
TEXT = "#f2f3f5"
TEXT_DIM = "#9ba0a6"
DONE = "#57f287"      # green, completed
RUN = "#5865f2"       # blurple, planned run
REST = "#b5bac1"      # gray, rest/strength
HOLD = "#fee75c"      # amber, changed/hold
SKIP = "#ed4245"      # red, skipped
GAP = "#4e5058"       # dashed, no plan
TODAY_BORDER = "#f0b232"  # gold

TZ = ZoneInfo("America/New_York")


def parse_calendar(path):
    """Parse calendar markdown table into {date: dict}. Later rows win."""
    rows = {}
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return rows
    for line in lines:
        line = line.strip()
        if not line.startswith("| 20"):
            continue
        fields = [f.strip() for f in line.split("|")]
        if len(fields) < 6:
            continue
        dspec, status, session, plan = fields[1], fields[2], fields[3], fields[4]
        m = re.match(r"(20\d\d-\d\d-\d\d)(?: to (20\d\d-\d\d-\d\d))?", dspec)
        if not m:
            continue
        if m.group(2):
            d0 = date.fromisoformat(m.group(1))
            d1 = date.fromisoformat(m.group(2))
            if (d1 - d0).days > 30 or (d1 - d0).days < 0:
                dates = [d0]
            else:
                dates = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
        else:
            dates = [date.fromisoformat(m.group(1))]
        for d in dates:
            rows[d] = {
                "status": status,
                "session": session,
                "plan": plan,
            }
    return rows


def categorize(row):
    """Return (kind, color). Days with no calendar entry default to rest."""
    if row is None:
        return "rest", REST
    s = row["status"].lower()
    sess = row["session"].lower()
    if "bike" in sess or "cross-train" in sess:
        is_run = False
    elif "rest" in sess.split("/")[0] or sess.startswith("full rest"):
        is_run = False
    else:
        is_run = (
            "run" in sess
            or "treadmill" in sess
            or "level" in sess
            or re.search(r"\d+(\.\d+)?\s?(-\s?\d+(\.\d+)?\s?)?mi\b", sess)
        )
    if "completed" in s:
        return "done", DONE
    if "skipped" in s:
        return "skip", SKIP
    if "changed" in s or "hold" in s:
        return "hold", HOLD
    if "pending" in s:
        return "hold", HOLD
    if is_run:
        return "run", RUN
    return "rest", REST


def wrap_clipped(text, width, max_lines):
    """Wrap text; if it exceeds max_lines, clip and end last line with an ellipsis."""
    lines = textwrap.wrap(text, width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][: width - 1].rstrip() + "…"
    return lines


def shorten(text, limit):
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def render(anchor, past_n, future_n, rows, out_path):
    days = [anchor + timedelta(days=i) for i in range(-past_n, future_n + 1)]
    n = len(days)
    cols = 3
    rows_n = -(-n // cols)  # ceil

    fig = plt.figure(figsize=(10.0, 10.0), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header
    week_no = (anchor - BLOCK_START).days // 7 + 1
    days_to_go = (RACE_DATE_EST - anchor).days
    fig.text(0.03, 0.965, "MARATHON BLOCK · WEEK %d" % week_no,
             color=TODAY_BORDER, fontsize=17, fontweight="bold", family="DejaVu Sans")
    fig.text(0.03, 0.938, "PHASE 0 — CONSISTENCY", color=TEXT_DIM, fontsize=11,
             family="DejaVu Sans")
    fig.text(0.97, 0.965, "T-%d DAYS" % days_to_go,
             color=TODAY_BORDER, fontsize=15, fontweight="bold", ha="right",
             family="DejaVu Sans")
    fig.text(0.97, 0.938, "HARRISBURG 2027 · est. Nov 14", color=TEXT_DIM,
             fontsize=9.5, ha="right", family="DejaVu Sans")

    # Card grid geometry (3 columns x 2 rows)
    left, right = 0.025, 0.975
    col_gap = 0.016
    row_gap = 0.024
    grid_top, grid_bot = 0.895, 0.065
    cw = (right - left - col_gap * (cols - 1)) / cols
    ch = (grid_top - grid_bot - row_gap * (rows_n - 1)) / rows_n

    for i, d in enumerate(days):
        col = i % cols
        rw = i // cols
        x0 = left + col * (cw + col_gap)
        ct = grid_top - rw * (ch + row_gap)          # card top
        cb = ct - ch                                  # card bottom
        row = rows.get(d)
        kind, color = categorize(row)
        is_today = d == anchor
        is_past = d < anchor

        bg = CARD_TODAY if is_today else (CARD_DIM if is_past else CARD)
        card = FancyBboxPatch(
            (x0, cb), cw, ch,
            boxstyle="round,pad=0.004,rounding_size=0.018",
            linewidth=2.8 if is_today else 1.0,
            edgecolor=TODAY_BORDER if is_today else color,
            facecolor=bg,
            linestyle="-",
            alpha=0.95 if is_past else 1.0,
        )
        ax.add_patch(card)

        # solid color band across card top (no text inside)
        band = FancyBboxPatch(
            (x0 + 0.007, ct - 0.048), cw - 0.014, 0.038,
            boxstyle="round,pad=0.002,rounding_size=0.010",
            linewidth=0, facecolor=color, alpha=0.92 if not is_past else 0.45,
        )
        ax.add_patch(band)

        cx = x0 + cw / 2

        # TODAY badge
        if is_today:
            fig.text(cx, ct + 0.017, "TODAY", color=TODAY_BORDER,
                     fontsize=11, fontweight="bold", ha="center", family="DejaVu Sans")

        # weekday + date
        wd = d.strftime("%a").upper()
        dt = d.strftime("%b %d")
        fig.text(cx, ct - 0.078, wd, color=TEXT, fontsize=14,
                 fontweight="bold", ha="center", family="DejaVu Sans")
        fig.text(cx, ct - 0.108, dt, color=TEXT_DIM, fontsize=10.5,
                 ha="center", family="DejaVu Sans")

        # session title (wrapped bold)
        if row:
            sess_lines = wrap_clipped(shorten(row["session"], 80), 24, 3)
        else:
            sess_lines = ["Rest day"]
        y = ct - 0.148
        for ln in sess_lines:
            fig.text(cx, y, ln, color=TEXT, fontsize=10.5, ha="center",
                     fontweight="bold", family="DejaVu Sans")
            y -= 0.031
        title_end = y + 0.031  # y of the last drawn title line

        # detail lines (snug under the title, never past the card bottom)
        if row:
            detail = shorten(row["plan"], 200)
        else:
            detail = "no calendar entry yet - defaulted to rest"
        det_y0 = title_end - 0.038
        floor = cb + 0.028
        max_det = max(1, int((det_y0 - floor) / 0.027) + 1)
        det_lines = wrap_clipped(detail, 30, min(7, max_det))
        y = det_y0
        for ln in det_lines:
            fig.text(cx, y, ln, color=TEXT_DIM, fontsize=8.5, ha="center",
                     family="DejaVu Sans")
            y -= 0.027

    # Footer legend
    legend = [("●", DONE, "done"), ("●", RUN, "run"), ("●", REST, "rest/PT"),
              ("●", HOLD, "hold"), ("●", SKIP, "skipped")]
    lx = 0.03
    for sym, col, name in legend:
        fig.text(lx, 0.030, sym, color=col, fontsize=10, family="DejaVu Sans")
        fig.text(lx + 0.013, 0.030, name, color=TEXT_DIM, fontsize=9.5,
                 family="DejaVu Sans")
        lx += 0.014 + 0.0105 * len(name) + 0.016

    fig.savefig(out_path, facecolor=BG)
    plt.close(fig)
    print("wrote %s (anchor %s, %d past / %d future, %dx%d grid)"
          % (out_path, anchor, past_n, future_n, cols, rows_n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=None, help="anchor date YYYY-MM-DD (default today ET)")
    ap.add_argument("--past", type=int, default=2)
    ap.add_argument("--future", type=int, default=3)
    ap.add_argument("--calendar", default=CALENDAR_PATH)
    ap.add_argument("--out", default="/tmp/running_calendar_strip.png")
    args = ap.parse_args()

    anchor = (
        date.fromisoformat(args.date)
        if args.date
        else datetime.now(TZ).date()
    )
    rows = parse_calendar(args.calendar)
    render(anchor, args.past, args.future, rows, args.out)


if __name__ == "__main__":
    main()
