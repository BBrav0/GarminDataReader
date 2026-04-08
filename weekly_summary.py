#!/usr/bin/env python3
"""
weekly_summary.py — Generate a weekly running summary chart from cache.db.
Reads purely from local SQLite — zero Garmin API calls.
Saves to /tmp/weekly_summary.png
"""
import os
import sys
import sqlite3, json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()

# Auto-detect and use venv Python if available
def ensure_venv():
    """Re-execute script with venv Python if not already using it."""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        return  # Already in a venv

    for env_name in (".venv", "venv"):
        venv_python = SCRIPT_DIR / env_name / "bin" / "python3"
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)

ensure_venv()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Config ────────────────────────────────────────────────────────────
DB_PATH  = SCRIPT_DIR / "cache.db"
RHR_LOG  = Path.home() / ".hermes" / "workspace" / "rhr_log.jsonl"
OUT_PATH = "/tmp/weekly_summary.png"

BG       = '#0f0f1a'
PANEL    = '#1a1a2e'
ACCENT   = '#4cc9f0'
ACCENT2  = '#f72585'
ACCENT3  = '#7bed9f'
MUTED    = '#888ea8'
TEXT     = '#e0e0f0'
REST_COL = '#2a2a3e'

# ── Load data ─────────────────────────────────────────────────────────
def load_runs():
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(str(DB_PATH))
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM runs WHERE activity_type != 'None' ORDER BY date DESC LIMIT 200")
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def load_rhr_log():
    """Load rhr_log.jsonl — used for resting HR trend on non-run days too."""
    entries = {}
    if RHR_LOG.exists():
        for line in RHR_LOG.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                entries[d["date"]] = d["rhr"]
            except Exception:
                pass
    return entries

def pace_to_seconds(pace_str):
    """Convert 'MM:SS' or 'H:MM:SS' pace string to seconds per mile."""
    if not pace_str:
        return None
    try:
        parts = pace_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return None

def week_bounds():
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday

def style_ax(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.spines[['top','right','left','bottom']].set_color('#2a2a3e')
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color='#2a2a3e', linewidth=0.5, zorder=0)

# ── Main ──────────────────────────────────────────────────────────────
runs     = load_runs()
rhr_data = load_rhr_log()

monday, sunday = week_bounds()

# Index runs by date
runs_by_date = {r["date"]: r for r in runs}

# ── Panel 1: This week daily distance ────────────────────────────────
week_days  = []
week_miles = []
for i in range(7):
    d = monday + timedelta(days=i)
    label = d.strftime('%a\n%b %-d')
    week_days.append(label)
    run = runs_by_date.get(d.isoformat())
    week_miles.append(run["distance"] or 0 if run else 0)

total_mi = sum(week_miles)
num_runs = sum(1 for m in week_miles if m > 0)

# ── Panel 2: HR-at-pace trend (last 8 weeks with runs) ───────────────
hr_pace_runs = [(r["date"], r["avghr"], pace_to_seconds(r["avg_pace"]))
                for r in runs if r.get("avghr") and r.get("avg_pace") and pace_to_seconds(r["avg_pace"])]
hr_pace_runs = sorted(hr_pace_runs, key=lambda x: x[0])[-42:]  # last 6 weeks max

# ── Panel 3: Resting HR trend (last 30 days, merge run + rhr_log) ────
rhr_trend = {}
for r in runs:
    if r.get("resting_hr") and r["resting_hr"] > 0:
        rhr_trend[r["date"]] = r["resting_hr"]
rhr_trend.update(rhr_data)  # rhr_log wins on conflict

today = date.today()
rhr_dates = []
rhr_vals  = []
for i in range(29, -1, -1):
    d = (today - timedelta(days=i)).isoformat()
    if d in rhr_trend:
        rhr_dates.append(d)
        rhr_vals.append(rhr_trend[d])

# ── Panel 4: Aerobic efficiency (max-avg HR) last 8 weeks ────────────
decouple_runs = [(r["date"], r["maxhr"] - r["avghr"])
                 for r in runs if r.get("maxhr") and r.get("avghr") and r["maxhr"] > r["avghr"]]
decouple_runs = sorted(decouple_runs, key=lambda x: x[0])[-12:]

# ── Panel 5: Weekly mileage last 8 weeks ─────────────────────────────
weekly = {}
for r in runs:
    try:
        d = date.fromisoformat(r["date"])
        wk_monday = d - timedelta(days=d.weekday())
        weekly[wk_monday] = weekly.get(wk_monday, 0) + (r["distance"] or 0)
    except Exception:
        pass

wk_keys   = sorted(weekly.keys())[-8:]
wk_labels = [d.strftime('%b %-d') for d in wk_keys]
wk_miles  = [weekly[d] for d in wk_keys]

# ── Avg pace for summary ──────────────────────────────────────────────
this_week_runs = [runs_by_date[d.isoformat()] for d in [monday + timedelta(days=i) for i in range(7)]
                  if d.isoformat() in runs_by_date]
avg_pace_str = "—"
avg_hr_str   = "—"
if this_week_runs:
    paces = [pace_to_seconds(r["avg_pace"]) for r in this_week_runs if r.get("avg_pace")]
    hrs   = [r["avghr"] for r in this_week_runs if r.get("avghr")]
    if paces:
        avg_s = int(sum(paces) / len(paces))
        avg_pace_str = f"{avg_s//60}:{avg_s%60:02d}/mi"
    if hrs:
        avg_hr_str = f"{int(sum(hrs)/len(hrs))} bpm"

# ── Empty state ───────────────────────────────────────────────────────
NO_DATA = not runs

# ── Draw ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor=BG)
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35,
                       left=0.07, right=0.97, top=0.88, bottom=0.07)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])

week_range = f"{monday.strftime('%b %-d')} – {sunday.strftime('%b %-d, %Y')}"
fig.text(0.5, 0.945, f'Weekly Running Summary  —  {week_range}',
         ha='center', va='center', fontsize=14, color=TEXT, fontweight='bold')

if NO_DATA:
    for ax in [ax1,ax2,ax3,ax4,ax5]:
        ax.set_facecolor(PANEL)
        ax.text(0.5, 0.5, 'No run data yet', ha='center', va='center',
                transform=ax.transAxes, color=MUTED, fontsize=11)
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f"Saved (empty state) → {OUT_PATH}")
    exit(0)

summary = f'  {total_mi:.1f} mi total    {num_runs} runs    Avg pace {avg_pace_str}    Avg HR {avg_hr_str}  '
fig.text(0.5, 0.912, summary, ha='center', va='center', fontsize=10, color=ACCENT,
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#16213e', edgecolor=ACCENT, linewidth=1))

# Panel 1
style_ax(ax1)
colors = [ACCENT if m > 0 else REST_COL for m in week_miles]
bars = ax1.bar(week_days, week_miles, color=colors, width=0.55, zorder=3)
for bar, m in zip(bars, week_miles):
    if m > 0:
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.05,
                 f'{m:.1f}', ha='center', va='bottom', color=TEXT, fontsize=9, fontweight='bold')
    else:
        ax1.text(bar.get_x()+bar.get_width()/2, 0.05,
                 'rest', ha='center', va='bottom', color=MUTED, fontsize=8)
ax1.set_title('This Week — Daily Distance', fontsize=10, pad=6)
ax1.set_ylabel('Miles', fontsize=9)
ax1.set_ylim(0, max(week_miles + [1]) * 1.4)

# Panel 2: HR-at-pace
style_ax(ax2)
if hr_pace_runs:
    labels = [r[0][-5:] for r in hr_pace_runs]
    hrs    = [r[1] for r in hr_pace_runs]
    ax2.plot(range(len(hrs)), hrs, color=ACCENT2, linewidth=2, marker='o', markersize=5, zorder=3)
    step = max(1, len(labels)//6)
    ax2.set_xticks(range(0, len(labels), step))
    ax2.set_xticklabels([labels[i] for i in range(0, len(labels), step)], fontsize=7, rotation=20)
    ax2.set_ylabel('Avg HR (bpm)', fontsize=9)
else:
    ax2.text(0.5, 0.5, 'Not enough data yet', ha='center', va='center',
             transform=ax2.transAxes, color=MUTED, fontsize=10)
ax2.set_title('HR-at-Pace Trend', fontsize=10, pad=6)

# Panel 3: RHR trend
style_ax(ax3)
if rhr_vals:
    x3 = list(range(len(rhr_vals)))
    ax3.plot(x3, rhr_vals, color=ACCENT3, linewidth=1.5, zorder=3)
    ax3.fill_between(x3, rhr_vals, min(rhr_vals)-1, alpha=0.15, color=ACCENT3)
    tick_pos = [0, len(rhr_vals)//2, len(rhr_vals)-1]
    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels([rhr_dates[i][-5:] for i in tick_pos], fontsize=7)
    ax3.annotate(f'Today: {rhr_vals[-1]} bpm', xy=(len(rhr_vals)-1, rhr_vals[-1]),
                 xytext=(len(rhr_vals)*0.6, rhr_vals[-1]+1.5),
                 fontsize=8, color=ACCENT3,
                 arrowprops=dict(arrowstyle='->', color=ACCENT3, lw=1))
    ax3.set_ylabel('RHR (bpm)', fontsize=9)
else:
    ax3.text(0.5, 0.5, 'Not enough data yet', ha='center', va='center',
             transform=ax3.transAxes, color=MUTED, fontsize=10)
ax3.set_title('Resting HR — Last 30 Days', fontsize=10, pad=6)

# Panel 4: Aerobic efficiency
style_ax(ax4)
if decouple_runs:
    d_labels = [r[0][-5:] for r in decouple_runs]
    d_vals   = [r[1] for r in decouple_runs]
    ax4.bar(range(len(d_vals)), d_vals,
            color=[ACCENT if v < 15 else ACCENT2 for v in d_vals], width=0.6, zorder=3)
    for i, v in enumerate(d_vals):
        ax4.text(i, v+0.2, str(v), ha='center', va='bottom', color=TEXT, fontsize=8)
    ax4.set_xticks(range(len(d_labels)))
    ax4.set_xticklabels(d_labels, fontsize=7, rotation=20)
    ax4.axhline(14, color=ACCENT3, linewidth=1, linestyle='--', alpha=0.6)
    ax4.text(len(d_vals)-0.5, 14.3, 'target', color=ACCENT3, fontsize=8)
    ax4.set_ylabel('HR Spread (bpm)', fontsize=9)
else:
    ax4.text(0.5, 0.5, 'Not enough data yet', ha='center', va='center',
             transform=ax4.transAxes, color=MUTED, fontsize=10)
ax4.set_title('Aerobic Efficiency (Max - Avg HR)', fontsize=10, pad=6)

# Panel 5: Weekly mileage
style_ax(ax5)
if wk_keys:
    bar_colors = [ACCENT if m > 0 else REST_COL for m in wk_miles]
    ax5.bar(range(len(wk_miles)), wk_miles, color=bar_colors, width=0.6, zorder=3)
    for i, m in enumerate(wk_miles):
        if m > 0:
            ax5.text(i, m+0.1, f'{m:.1f}', ha='center', va='bottom',
                     color=TEXT, fontsize=8, fontweight='bold')
    ax5.set_xticks(range(len(wk_labels)))
    ax5.set_xticklabels(wk_labels, fontsize=7, rotation=30, ha='right')
    ax5.set_ylabel('Miles', fontsize=9)
else:
    ax5.text(0.5, 0.5, 'Not enough data yet', ha='center', va='center',
             transform=ax5.transAxes, color=MUTED, fontsize=10)
ax5.set_title('Weekly Mileage — Last 8 Weeks', fontsize=10, pad=6)

plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close(fig)
print(f"Saved → {OUT_PATH}")
