#!/usr/bin/env python3
"""
Script to analyze runs by pace intervals, showing:
- Quantity of runs per pace interval (minute intervals, e.g., 9-10, 10-11)
- Average heart rate for runs at each pace
- Average elevation gain for runs at each pace
Excludes treadmill runs.
"""
import sqlite3
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib is not installed. Install it with: pip install matplotlib")
from collections import defaultdict

def parse_pace_to_minutes(pace_str):
    """
    Parse pace string in format 'HH:MM:SS' to minutes per mile.
    Returns None if parsing fails.
    """
    if not pace_str:
        return None
    try:
        parts = pace_str.split(':')
        if len(parts) == 3:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            total_minutes = hours * 60 + minutes + seconds / 60.0
            return total_minutes
    except (ValueError, AttributeError):
        return None
    return None

def get_pace_interval(minutes_per_mile):
    """
    Convert minutes per mile to a minute interval string (e.g., '9-10').
    Returns None if minutes_per_mile is None.
    """
    if minutes_per_mile is None:
        return None
    # Round down to get the interval start
    interval_start = int(minutes_per_mile)
    interval_end = interval_start + 1
    return f"{interval_start}-{interval_end}"

def analyze_runs_by_pace(db_path="cache.db"):
    """
    Analyze runs from database, grouping by pace intervals.
    Excludes treadmill runs.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Query runs excluding treadmill runs
    query = """
        SELECT avg_pace, avghr, elev_gain
        FROM runs
        WHERE activity_type = 'Run'
        AND avg_pace IS NOT NULL
        AND distance IS NOT NULL
        AND distance > 0
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()
    
    # Group data by pace interval
    pace_data = defaultdict(lambda: {'count': 0, 'hr_sum': 0, 'hr_count': 0, 'elev_sum': 0, 'elev_count': 0})
    
    for pace_str, avghr, elev_gain in rows:
        minutes_per_mile = parse_pace_to_minutes(pace_str)
        if minutes_per_mile is None:
            continue
        
        interval = get_pace_interval(minutes_per_mile)
        if interval is None:
            continue
        
        pace_data[interval]['count'] += 1
        
        # Accumulate heart rate (only if not NULL and > 0)
        if avghr is not None and avghr > 0:
            pace_data[interval]['hr_sum'] += avghr
            pace_data[interval]['hr_count'] += 1
        
        # Accumulate elevation gain (only if not NULL)
        if elev_gain is not None:
            pace_data[interval]['elev_sum'] += elev_gain
            pace_data[interval]['elev_count'] += 1
    
    # Convert to lists for plotting, sorted by interval start
    intervals = []
    counts = []
    avg_hrs = []
    avg_elevs = []
    
    # Sort intervals by their start value
    sorted_intervals = sorted(pace_data.keys(), key=lambda x: int(x.split('-')[0]))
    
    for interval in sorted_intervals:
        data = pace_data[interval]
        intervals.append(interval)
        counts.append(data['count'])
        
        # Calculate average heart rate
        if data['hr_count'] > 0:
            avg_hrs.append(data['hr_sum'] / data['hr_count'])
        else:
            avg_hrs.append(0)
        
        # Calculate average elevation gain
        if data['elev_count'] > 0:
            avg_elevs.append(data['elev_sum'] / data['elev_count'])
        else:
            avg_elevs.append(0)
    
    return intervals, counts, avg_hrs, avg_elevs

def create_bar_chart(intervals, counts, avg_hrs, avg_elevs):
    """
    Create a bar chart showing runs per pace interval with average HR and elevation gain.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("\nCannot create chart: matplotlib is not installed.")
        print("Install it with: pip install matplotlib")
        return
    
    if not intervals:
        print("No data to plot. Make sure you have runs in the database.")
        return
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    x = np.arange(len(intervals))
    width = 0.6
    
    # Chart 1: Quantity of runs per pace interval
    bars1 = ax1.bar(x, counts, width, color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Pace Interval (minutes per mile)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Number of Runs', fontsize=11, fontweight='bold')
    ax1.set_title('Quantity of Runs per Pace Interval (Excluding Treadmill Runs)', 
                  fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(intervals, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    # Chart 2: Average heart rate per pace interval
    bars2 = ax2.bar(x, avg_hrs, width, color='crimson', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Pace Interval (minutes per mile)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Heart Rate (bpm)', fontsize=11, fontweight='bold')
    ax2.set_title('Average Heart Rate per Pace Interval', 
                  fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(intervals, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars (only if > 0)
    for bar, hr in zip(bars2, avg_hrs):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(round(height))}',
                    ha='center', va='bottom', fontsize=9)
    
    # Chart 3: Average elevation gain per pace interval
    bars3 = ax3.bar(x, avg_elevs, width, color='forestgreen', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Pace Interval (minutes per mile)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Average Elevation Gain (feet)', fontsize=11, fontweight='bold')
    ax3.set_title('Average Elevation Gain per Pace Interval', 
                  fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(intervals, rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars (only if > 0)
    for bar, elev in zip(bars3, avg_elevs):
        height = bar.get_height()
        if height > 0:
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(round(height))}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('pace_analysis.png', dpi=300, bbox_inches='tight')
    print(f"Chart saved as 'pace_analysis.png'")
    plt.show()

def main():
    print("Analyzing runs by pace interval...")
    print("Excluding treadmill runs...")
    
    intervals, counts, avg_hrs, avg_elevs = analyze_runs_by_pace()
    
    if not intervals:
        print("No valid run data found in the database.")
        return
    
    print(f"\nFound {sum(counts)} runs across {len(intervals)} pace intervals")
    print("\nSummary by pace interval:")
    print("-" * 60)
    print(f"{'Interval':<12} {'Runs':<8} {'Avg HR':<10} {'Avg Elev (ft)':<15}")
    print("-" * 60)
    
    for interval, count, hr, elev in zip(intervals, counts, avg_hrs, avg_elevs):
        hr_str = f"{int(round(hr))}" if hr > 0 else "N/A"
        elev_str = f"{int(round(elev))}" if elev > 0 else "N/A"
        print(f"{interval:<12} {count:<8} {hr_str:<10} {elev_str:<15}")
    
    create_bar_chart(intervals, counts, avg_hrs, avg_elevs)

if __name__ == "__main__":
    main()

