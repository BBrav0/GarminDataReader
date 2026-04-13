#!/usr/bin/env python3
"""
weekly_summary.py — generate standalone weekly running charts from cache.db.
Reads purely from local SQLite — zero Garmin API calls.
Saves shareable PNG files to /tmp/weekly_summary/
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).parent.absolute()


def ensure_venv() -> None:
    """Re-execute script with venv Python if not already using it."""
    if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix):
        return

    for env_name in (".venv", "venv"):
        venv_python = SCRIPT_DIR / env_name / "bin" / "python3"
        if venv_python.exists():
            os.execv(str(venv_python), [str(venv_python)] + sys.argv)


# ── Config ────────────────────────────────────────────────────────────
DB_PATH = SCRIPT_DIR / "cache.db"
RHR_LOG = Path.home() / ".hermes" / "workspace" / "rhr_log.jsonl"
OUTPUT_DIR = Path("/tmp/weekly_summary")
LEGACY_OUT_PATH = Path("/tmp/weekly_summary.png")
EXPORT_DPI = 220

BG = "#0f0f1a"
PANEL = "#1a1a2e"
ACCENT = "#4cc9f0"
ACCENT2 = "#f72585"
ACCENT3 = "#7bed9f"
ACCENT4 = "#f4d35e"
MUTED = "#9aa3bd"
TEXT = "#eef2ff"
REST_COL = "#2a2a3e"
GRID = "#2f3654"

CHART_SPECS = (
    ("this_week_daily_distance.png", "This Week Daily Distance"),
    ("heart_rate_vs_pace.png", "Heart Rate vs Pace"),
    ("resting_heart_rate_30_days.png", "Resting Heart Rate"),
    ("aerobic_efficiency.png", "Aerobic Efficiency"),
    ("weekly_mileage_8_weeks.png", "Weekly Mileage"),
)


# ── Load data ─────────────────────────────────────────────────────────
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


def load_rhr_log(rhr_log_path: Path = RHR_LOG) -> dict[str, int]:
    """Load rhr_log.jsonl — used for resting HR trend on non-run days too."""
    entries: dict[str, int] = {}
    if rhr_log_path.exists():
        for line in rhr_log_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                date_str = record.get("date")
                rhr = int(record.get("rhr"))
                if date_str and rhr > 0:
                    entries[date_str] = rhr
            except Exception:
                pass
    return entries


def pace_to_seconds(pace_str: str | None) -> int | None:
    """Convert 'MM:SS' or 'H:MM:SS' pace string to seconds per mile."""
    if not pace_str:
        return None
    try:
        parts = pace_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return None
    return None


def seconds_to_pace(total_seconds: int | float | None) -> str:
    if total_seconds is None:
        return "—"
    total_seconds = int(round(total_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}/mi"


def week_bounds(today: date | None = None) -> tuple[date, date]:
    anchor = today or date.today()
    monday = anchor - timedelta(days=anchor.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday


def day_label(day: date) -> str:
    return f"{day.strftime('%a')}\n{day.strftime('%b')} {day.day}"


def short_date_label(day: date) -> str:
    return f"{day.strftime('%b')} {day.day}"


def style_ax(ax) -> None:
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(1)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)


def add_chart_header(ax, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", fontsize=16, fontweight="bold", pad=22, color=TEXT)
    ax.text(
        0,
        1.03,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED,
    )


def draw_empty(ax, message: str) -> None:
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=12,
        fontweight="semibold",
    )
    ax.set_xticks([])
    ax.set_yticks([])


def create_figure():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    fig.patch.set_facecolor(BG)
    return fig, ax


def save_chart(fig, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=EXPORT_DPI, bbox_inches="tight", pad_inches=0.3, facecolor=BG)
    plt.close(fig)
    return destination


def tick_positions(length: int, max_ticks: int = 5) -> list[int]:
    if length <= 0:
        return []
    if length <= max_ticks:
        return list(range(length))
    return sorted({int(round(i)) for i in np.linspace(0, length - 1, num=max_ticks)})


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    if count == 1:
        return singular
    return plural or f"{singular}s"


def build_context(
    runs: list[dict],
    rhr_data: dict[str, int],
    today: date | None = None,
) -> dict:
    anchor = today or date.today()
    monday, sunday = week_bounds(anchor)
    runs = sorted(runs, key=lambda run: run["date"])
    runs_by_date = {run["date"]: run for run in runs}

    week_days: list[str] = []
    week_miles: list[float] = []
    this_week_runs: list[dict] = []
    for offset in range(7):
        current_day = monday + timedelta(days=offset)
        week_days.append(day_label(current_day))
        run = runs_by_date.get(current_day.isoformat())
        distance = float(run.get("distance") or 0) if run else 0.0
        week_miles.append(distance)
        if run and distance > 0:
            this_week_runs.append(run)

    total_mi = round(sum(week_miles), 1)
    num_runs = sum(1 for miles in week_miles if miles > 0)

    avg_pace_str = "—"
    avg_hr_str = "—"
    if this_week_runs:
        paces = [pace_to_seconds(run.get("avg_pace")) for run in this_week_runs if pace_to_seconds(run.get("avg_pace"))]
        hrs = [int(run["avghr"]) for run in this_week_runs if run.get("avghr")]
        if paces:
            avg_pace_str = seconds_to_pace(sum(paces) / len(paces))
        if hrs:
            avg_hr_str = f"{int(round(sum(hrs) / len(hrs)))} bpm"

    hr_pace_runs = [
        {
            "date": run["date"],
            "avg_hr": int(run["avghr"]),
            "pace_seconds": pace_to_seconds(run.get("avg_pace")),
        }
        for run in runs
        if run.get("avghr") and pace_to_seconds(run.get("avg_pace"))
    ][-18:]

    rhr_trend: dict[str, int] = {}
    for run in runs:
        if run.get("resting_hr") and int(run["resting_hr"]) > 0:
            rhr_trend[run["date"]] = int(run["resting_hr"])
    rhr_trend.update(rhr_data)

    rhr_dates: list[str] = []
    rhr_vals: list[int] = []
    for offset in range(29, -1, -1):
        date_str = (anchor - timedelta(days=offset)).isoformat()
        if date_str in rhr_trend:
            rhr_dates.append(date_str)
            rhr_vals.append(rhr_trend[date_str])

    decouple_runs = [
        {
            "date": run["date"],
            "spread": int(run["maxhr"]) - int(run["avghr"]),
        }
        for run in runs
        if run.get("maxhr") and run.get("avghr") and int(run["maxhr"]) > int(run["avghr"])
    ][-12:]

    weekly: dict[date, float] = {}
    for run in runs:
        try:
            run_day = date.fromisoformat(run["date"])
            week_start = run_day - timedelta(days=run_day.weekday())
            weekly[week_start] = weekly.get(week_start, 0.0) + float(run.get("distance") or 0)
        except Exception:
            pass

    week_keys = sorted(weekly.keys())[-8:]
    weekly_points = [{"week_start": week_key, "miles": round(weekly[week_key], 1)} for week_key in week_keys]

    return {
        "today": anchor,
        "monday": monday,
        "sunday": sunday,
        "week_days": week_days,
        "week_miles": week_miles,
        "total_mi": total_mi,
        "num_runs": num_runs,
        "avg_pace_str": avg_pace_str,
        "avg_hr_str": avg_hr_str,
        "hr_pace_runs": hr_pace_runs,
        "rhr_dates": rhr_dates,
        "rhr_vals": rhr_vals,
        "decouple_runs": decouple_runs,
        "weekly_points": weekly_points,
        "has_runs": bool(runs),
    }


def plot_weekly_distance(ax, context: dict) -> None:
    style_ax(ax)
    week_range = f"Week of {short_date_label(context['monday'])} – {short_date_label(context['sunday'])}, {context['sunday'].year}"
    add_chart_header(ax, "This Week Daily Distance", week_range)

    week_miles = context["week_miles"]
    week_days = context["week_days"]
    colors = [ACCENT if miles > 0 else REST_COL for miles in week_miles]
    bars = ax.bar(week_days, week_miles, color=colors, width=0.58, edgecolor="#111627", linewidth=1.1, zorder=3)

    ceiling = max(week_miles + [1.0]) * 1.3
    ax.set_ylim(0, ceiling)
    ax.set_ylabel("Miles")
    ax.margins(x=0.03)

    for bar, miles in zip(bars, week_miles):
        x_pos = bar.get_x() + bar.get_width() / 2
        if miles > 0:
            ax.text(
                x_pos,
                miles + max(ceiling * 0.02, 0.08),
                f"{miles:.1f}",
                ha="center",
                va="bottom",
                color=TEXT,
                fontsize=10,
                fontweight="bold",
            )
        else:
            ax.text(x_pos, 0.07, "Rest", ha="center", va="bottom", color=MUTED, fontsize=9)

    summary = (
        f"{context['total_mi']:.1f} mi total   •   {context['num_runs']} runs   •   "
        f"Avg pace {context['avg_pace_str']}   •   Avg HR {context['avg_hr_str']}"
    )
    ax.text(
        1,
        1.12,
        summary,
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=9,
        color=ACCENT,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#16213e", edgecolor=ACCENT, linewidth=1),
    )


def plot_hr_vs_pace(ax, context: dict) -> None:
    style_ax(ax)
    points = context["hr_pace_runs"]
    add_chart_header(
        ax,
        "Heart Rate vs Pace",
        (
            f"Last {len(points)} qualifying {pluralize(len(points), 'run')} • lower HR at the same pace is better"
            if points
            else "Needs runs with pace and HR data"
        ),
    )

    if not points:
        draw_empty(ax, "Not enough pace and HR data yet")
        return

    paces = np.array([point["pace_seconds"] for point in points], dtype=float)
    hrs = np.array([point["avg_hr"] for point in points], dtype=float)
    sizes = np.linspace(70, 130, num=len(points))
    scatter = ax.scatter(
        paces,
        hrs,
        c=np.arange(len(points)),
        cmap="cool",
        s=sizes,
        alpha=0.9,
        edgecolors="#111627",
        linewidth=1,
        zorder=4,
    )

    if len(points) >= 3:
        slope, intercept = np.polyfit(paces, hrs, 1)
        x_line = np.linspace(paces.min(), paces.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=ACCENT4, linewidth=2, linestyle="--", zorder=3)

    latest = points[-1]
    ax.annotate(
        f"Latest • {latest['date'][5:]}",
        xy=(latest["pace_seconds"], latest["avg_hr"]),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        color=TEXT,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="#16213e", edgecolor=ACCENT2, linewidth=1),
    )

    pace_ticks = tick_positions(len(np.unique(paces)), max_ticks=5)
    pace_min = int(np.floor(paces.min() / 15.0) * 15)
    pace_max = int(np.ceil(paces.max() / 15.0) * 15)
    tick_values = sorted({int(value) for value in np.linspace(pace_min, pace_max, num=max(3, len(pace_ticks)))})
    ax.set_xticks(tick_values)
    ax.set_xticklabels([seconds_to_pace(value).replace("/mi", "") for value in tick_values])
    ax.set_xlabel("Pace (min/mi)")
    ax.set_ylabel("Average HR (bpm)")
    ax.margins(x=0.08, y=0.12)
    if len(points) >= 3:
        colorbar = plt.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.outline.set_edgecolor(GRID)
        colorbar.ax.tick_params(colors=MUTED, labelsize=8)
        colorbar.set_label("Run order", color=MUTED, fontsize=9)


def plot_rhr(ax, context: dict) -> None:
    style_ax(ax)
    values = context["rhr_vals"]
    dates = context["rhr_dates"]
    add_chart_header(
        ax,
        "Resting Heart Rate",
        "Last 30 days • includes non-run days from rhr_log.jsonl" if values else "Needs recent resting HR data",
    )

    if not values:
        draw_empty(ax, "No resting HR data found")
        return

    x_values = np.arange(len(values))
    ax.plot(x_values, values, color=ACCENT3, linewidth=2.4, marker="o", markersize=5, zorder=4)
    ax.fill_between(x_values, values, min(values) - 1, alpha=0.14, color=ACCENT3, zorder=2)

    if len(values) >= 4:
        window = min(7, len(values))
        kernel = np.ones(window) / window
        trend = np.convolve(values, kernel, mode="valid")
        trend_x = np.arange(window - 1, len(values))
        ax.plot(trend_x, trend, color=ACCENT4, linewidth=2, linestyle="--", zorder=5, label=f"{window}-point avg")
        ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=MUTED)

    latest_idx = len(values) - 1
    ax.annotate(
        f"{values[-1]} bpm",
        xy=(latest_idx, values[-1]),
        xytext=(-4, 14),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color=TEXT,
        fontweight="bold",
    )

    tick_idx = tick_positions(len(dates), max_ticks=5)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([dates[idx][5:] for idx in tick_idx])
    ax.set_ylabel("RHR (bpm)")
    ax.margins(x=0.03, y=0.18)


def plot_aerobic_efficiency(ax, context: dict) -> None:
    style_ax(ax)
    points = context["decouple_runs"]
    add_chart_header(
        ax,
        "Aerobic Efficiency",
        (
            f"Last {len(points)} {pluralize(len(points), 'run')} • lower max/avg HR spread is steadier"
            if points
            else "Needs runs with average and max HR"
        ),
    )

    if not points:
        draw_empty(ax, "Not enough max HR and avg HR data yet")
        return

    spreads = [point["spread"] for point in points]
    labels = [point["date"][5:] for point in points]
    x_values = np.arange(len(spreads))
    colors = [ACCENT if spread <= 14 else ACCENT2 for spread in spreads]
    bars = ax.bar(x_values, spreads, color=colors, width=0.62, edgecolor="#111627", linewidth=1, zorder=3)

    for bar, spread in zip(bars, spreads):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            spread + 0.35,
            str(spread),
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(14, color=ACCENT3, linewidth=1.5, linestyle="--", alpha=0.8)
    ax.text(0.99, 14.25, "target", color=ACCENT3, fontsize=9, ha="right", transform=ax.get_yaxis_transform())
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Max HR - Avg HR (bpm)")
    ax.margins(x=0.03, y=0.15)


def plot_weekly_mileage(ax, context: dict) -> None:
    style_ax(ax)
    points = context["weekly_points"]
    add_chart_header(
        ax,
        "Weekly Mileage",
        (
            f"Last {len(points)} training {pluralize(len(points), 'week')}"
            if points
            else "Needs at least one completed run week"
        ),
    )

    if not points:
        draw_empty(ax, "Not enough weekly mileage data yet")
        return

    miles = [point["miles"] for point in points]
    labels = [short_date_label(point["week_start"]) for point in points]
    x_values = np.arange(len(miles))
    colors = [ACCENT] * len(miles)
    colors[-1] = ACCENT2
    bar_width = 0.5 if len(miles) == 1 else 0.62
    bars = ax.bar(x_values, miles, color=colors, width=bar_width, edgecolor="#111627", linewidth=1, zorder=3)

    if len(miles) >= 2:
        slope, intercept = np.polyfit(x_values, miles, 1)
        ax.plot(x_values, slope * x_values + intercept, color=ACCENT4, linewidth=2, linestyle="--", zorder=4)

    for bar, miles_value in zip(bars, miles):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            miles_value + 0.15,
            f"{miles_value:.1f}",
            ha="center",
            va="bottom",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_ylabel("Miles")
    ax.margins(x=0.03, y=0.15)
    if len(miles) == 1:
        ax.set_xlim(-0.4, 0.4)


def generate_weekly_charts(
    db_path: Path = DB_PATH,
    rhr_log_path: Path = RHR_LOG,
    output_dir: Path = OUTPUT_DIR,
    today: date | None = None,
    legacy_output_path: Path = LEGACY_OUT_PATH,
) -> list[Path]:
    runs = load_runs(db_path)
    rhr_data = load_rhr_log(rhr_log_path)
    context = build_context(runs, rhr_data, today=today)

    if legacy_output_path.exists():
        legacy_output_path.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    chart_builders = (
        ("this_week_daily_distance.png", plot_weekly_distance),
        ("heart_rate_vs_pace.png", plot_hr_vs_pace),
        ("resting_heart_rate_30_days.png", plot_rhr),
        ("aerobic_efficiency.png", plot_aerobic_efficiency),
        ("weekly_mileage_8_weeks.png", plot_weekly_mileage),
    )

    saved_paths: list[Path] = []
    for filename, builder in chart_builders:
        fig, ax = create_figure()
        builder(ax, context)
        saved_paths.append(save_chart(fig, output_dir / filename))
    return saved_paths


def main() -> int:
    saved_paths = generate_weekly_charts()
    print(f"Saved {len(saved_paths)} charts → {OUTPUT_DIR}")
    for path in saved_paths:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    ensure_venv()
    raise SystemExit(main())
