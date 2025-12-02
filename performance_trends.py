#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance Trends Analysis
Analyzes pace progression, PRs, consistency, and volume trends over time.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from collections import defaultdict

# Configure UTF-8 output for Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def parse_time_to_seconds(time_str):
    """Convert time string HH:MM:SS to seconds."""
    if pd.isna(time_str) or not time_str:
        return np.nan
    parts = str(time_str).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    return np.nan

def seconds_to_time_str(seconds):
    """Convert seconds to HH:MM:SS format."""
    if np.isnan(seconds):
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def load_and_prepare_data(csv_path='runs_data.csv'):
    """Load CSV and prepare data for analysis."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Convert pace and duration to seconds
    df['pace_seconds'] = df['avg_pace'].apply(parse_time_to_seconds)
    df['duration_seconds'] = df['duration'].apply(parse_time_to_seconds)
    
    # Calculate speed (mph)
    df['speed_mph'] = np.where(df['pace_seconds'] > 0, 60 / (df['pace_seconds'] / 60), np.nan)
    
    # Filter out treadmill runs for outdoor analysis
    outdoor_df = df[df['activity_type'] == 'Run'].copy()
    
    return df, outdoor_df

def analyze_personal_records(df):
    """Find personal records for different distance brackets."""
    distance_brackets = {
        '5K (3.0-3.5 mi)': (3.0, 3.5),
        '10K (6.0-6.5 mi)': (6.0, 6.5),
        'Half Marathon (13-14 mi)': (13.0, 14.0),
        '15-20 mi': (15.0, 20.0),
        'Marathon (26+ mi)': (26.0, 50.0)
    }
    
    prs = {}
    for bracket_name, (min_dist, max_dist) in distance_brackets.items():
        bracket_runs = df[(df['distance'] >= min_dist) & (df['distance'] <= max_dist) & 
                          (df['pace_seconds'] > 0)].copy()
        if len(bracket_runs) > 0:
            fastest_idx = bracket_runs['pace_seconds'].idxmin()
            fastest_run = bracket_runs.loc[fastest_idx]
            prs[bracket_name] = {
                'date': fastest_run['date'],
                'distance': fastest_run['distance'],
                'pace': fastest_run['avg_pace'],
                'pace_seconds': fastest_run['pace_seconds'],
                'duration': fastest_run['duration']
            }
    
    return prs

def calculate_consistency_metrics(df):
    """Calculate consistency and variation metrics."""
    outdoor_runs = df[(df['pace_seconds'] > 0) & (df['distance'] > 0)]
    
    if len(outdoor_runs) == 0:
        return {}
    
    pace_cv = np.std(outdoor_runs['pace_seconds']) / np.mean(outdoor_runs['pace_seconds'])
    distance_cv = np.std(outdoor_runs['distance']) / np.mean(outdoor_runs['distance'])
    
    # Calculate training streaks (consecutive weeks with runs)
    df_sorted = outdoor_runs.sort_values('date')
    df_sorted['week'] = df_sorted['date'].dt.isocalendar().week
    df_sorted['year'] = df_sorted['date'].dt.year
    weekly_runs = df_sorted.groupby(['year', 'week']).size()
    
    # Find longest streak of consecutive weeks
    current_streak = 1
    max_streak = 1
    prev_week = None
    
    for (year, week), count in weekly_runs.items():
        if prev_week is not None:
            if (year, week) == (prev_week[0], prev_week[1] + 1) or \
               (week == 1 and year == prev_week[0] + 1 and prev_week[1] >= 52):
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 1
        prev_week = (year, week)
    
    return {
        'pace_cv': pace_cv,
        'distance_cv': distance_cv,
        'mean_pace_seconds': np.mean(outdoor_runs['pace_seconds']),
        'std_pace_seconds': np.std(outdoor_runs['pace_seconds']),
        'mean_distance': np.mean(outdoor_runs['distance']),
        'total_runs': len(outdoor_runs),
        'max_weekly_streak': max_streak
    }

def analyze_volume_trends(df):
    """Analyze weekly and monthly mileage trends."""
    outdoor_runs = df[df['distance'] > 0].copy()
    outdoor_runs['week'] = outdoor_runs['date'].dt.to_period('W')
    outdoor_runs['month'] = outdoor_runs['date'].dt.to_period('M')
    
    # Weekly totals
    weekly_volume = outdoor_runs.groupby('week')['distance'].sum()
    weekly_dates = weekly_volume.index.to_timestamp()
    
    # Monthly totals
    monthly_volume = outdoor_runs.groupby('month')['distance'].sum()
    monthly_dates = monthly_volume.index.to_timestamp()
    
    # Calculate 4-week moving average
    if len(weekly_volume) >= 4:
        moving_avg = np.convolve(weekly_volume.values, np.ones(4)/4, mode='valid')
        moving_avg_dates = weekly_dates[3:]
    else:
        moving_avg = None
        moving_avg_dates = None
    
    return {
        'weekly_volume': weekly_volume.values,
        'weekly_dates': weekly_dates,
        'monthly_volume': monthly_volume.values,
        'monthly_dates': monthly_dates,
        'moving_avg': moving_avg,
        'moving_avg_dates': moving_avg_dates,
        'peak_week': weekly_volume.max(),
        'avg_weekly': weekly_volume.mean(),
        'peak_month': monthly_volume.max()
    }

def classify_run_type(pace_seconds, threshold_pace):
    """Classify run as tempo or easy based on pace."""
    if np.isnan(pace_seconds) or np.isnan(threshold_pace):
        return 'Unknown'
    if pace_seconds < threshold_pace * 1.15:
        return 'Tempo/Fast'
    else:
        return 'Easy/Long'

def analyze_speed_distribution(df):
    """Analyze distribution of tempo vs easy runs."""
    outdoor_runs = df[(df['pace_seconds'] > 0)].copy()
    
    if len(outdoor_runs) == 0:
        return {}
    
    # Use median pace as threshold
    threshold_pace = np.median(outdoor_runs['pace_seconds'])
    outdoor_runs['run_type'] = outdoor_runs['pace_seconds'].apply(
        lambda x: classify_run_type(x, threshold_pace)
    )
    
    type_counts = outdoor_runs['run_type'].value_counts()
    
    return {
        'threshold_pace': threshold_pace,
        'tempo_count': type_counts.get('Tempo/Fast', 0),
        'easy_count': type_counts.get('Easy/Long', 0),
        'tempo_pct': (type_counts.get('Tempo/Fast', 0) / len(outdoor_runs)) * 100,
        'easy_pct': (type_counts.get('Easy/Long', 0) / len(outdoor_runs)) * 100
    }

def create_visualizations(df, prs, consistency, volume_trends, speed_dist):
    """Create comprehensive visualization charts."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    outdoor_runs = df[df['activity_type'] == 'Run']
    
    # 1. Pace progression over time
    ax1 = fig.add_subplot(gs[0, :])
    valid_pace_runs = outdoor_runs[outdoor_runs['pace_seconds'] > 0]
    
    if len(valid_pace_runs) > 0:
        scatter = ax1.scatter(valid_pace_runs['date'], valid_pace_runs['pace_seconds'] / 60,
                             c=valid_pace_runs['distance'], cmap='viridis', 
                             s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Add trend line
        x_numeric = (valid_pace_runs['date'] - valid_pace_runs['date'].min()).dt.days
        z = np.polyfit(x_numeric, valid_pace_runs['pace_seconds'] / 60, 2)
        p = np.poly1d(z)
        trend_x = np.linspace(x_numeric.min(), x_numeric.max(), 100)
        trend_dates = valid_pace_runs['date'].min() + pd.to_timedelta(trend_x, unit='D')
        ax1.plot(trend_dates, p(trend_x), 'r--', linewidth=2, alpha=0.7, label='Trend')
        
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('Pace (min/mile)', fontweight='bold')
        ax1.set_title('Pace Progression Over Time (Outdoor Runs)', fontweight='bold', fontsize=14)
        ax1.grid(alpha=0.3)
        ax1.invert_yaxis()
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Distance (miles)', fontweight='bold')
        ax1.legend()
    
    # 2. Weekly volume trends
    ax2 = fig.add_subplot(gs[1, 0])
    if volume_trends['weekly_volume'] is not None:
        ax2.bar(volume_trends['weekly_dates'], volume_trends['weekly_volume'], 
               width=5, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        if volume_trends['moving_avg'] is not None:
            ax2.plot(volume_trends['moving_avg_dates'], volume_trends['moving_avg'],
                    'r-', linewidth=2, label='4-week MA')
        
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.set_ylabel('Weekly Mileage', fontweight='bold')
        ax2.set_title(f"Weekly Volume (Avg: {volume_trends['avg_weekly']:.1f} mi, Peak: {volume_trends['peak_week']:.1f} mi)", 
                     fontweight='bold')
        ax2.grid(alpha=0.3, axis='y')
        ax2.legend()
    
    # 3. Monthly volume
    ax3 = fig.add_subplot(gs[1, 1])
    if volume_trends['monthly_volume'] is not None:
        ax3.bar(volume_trends['monthly_dates'], volume_trends['monthly_volume'],
               width=25, color='forestgreen', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Month', fontweight='bold')
        ax3.set_ylabel('Monthly Mileage', fontweight='bold')
        ax3.set_title(f"Monthly Volume (Peak: {volume_trends['peak_month']:.1f} mi)", 
                     fontweight='bold')
        ax3.grid(alpha=0.3, axis='y')
    
    # 4. Pace distribution histogram
    ax4 = fig.add_subplot(gs[2, 0])
    if len(valid_pace_runs) > 0:
        pace_minutes = valid_pace_runs['pace_seconds'] / 60
        ax4.hist(pace_minutes, bins=20, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax4.axvline(pace_minutes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pace_minutes.mean():.2f}')
        ax4.axvline(pace_minutes.median(), color='blue', linestyle='--', linewidth=2, label=f'Median: {pace_minutes.median():.2f}')
        ax4.set_xlabel('Pace (min/mile)', fontweight='bold')
        ax4.set_ylabel('Frequency', fontweight='bold')
        ax4.set_title('Pace Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
    
    # 5. Run type distribution
    ax5 = fig.add_subplot(gs[2, 1])
    if speed_dist:
        labels = ['Tempo/Fast', 'Easy/Long']
        sizes = [speed_dist['tempo_count'], speed_dist['easy_count']]
        colors = ['#ff6b6b', '#4ecdc4']
        explode = (0.05, 0.05)
        
        ax5.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontweight': 'bold'})
        ax5.set_title(f"Run Type Distribution (Threshold: {seconds_to_time_str(speed_dist['threshold_pace'])})", 
                     fontweight='bold')
    
    plt.savefig('performance_trends.png', dpi=300, bbox_inches='tight')
    print("[OK] Chart saved as 'performance_trends.png'")
    plt.close()

def print_report(df, prs, consistency, volume_trends, speed_dist):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print(" PERFORMANCE TRENDS ANALYSIS REPORT ".center(80, "="))
    print("="*80)
    
    print("\n📊 PERSONAL RECORDS BY DISTANCE")
    print("-" * 80)
    if prs:
        for bracket, pr_data in prs.items():
            print(f"\n{bracket}:")
            print(f"  Date: {pr_data['date'].strftime('%Y-%m-%d')}")
            print(f"  Distance: {pr_data['distance']:.2f} miles")
            print(f"  Pace: {pr_data['pace']} min/mile")
            print(f"  Time: {pr_data['duration']}")
    else:
        print("  No PRs found in standard distance brackets")
    
    print("\n\n📈 CONSISTENCY METRICS")
    print("-" * 80)
    if consistency:
        print(f"Total Outdoor Runs: {consistency['total_runs']}")
        print(f"Average Pace: {seconds_to_time_str(consistency['mean_pace_seconds'])} min/mile")
        print(f"Pace Variability (CV): {consistency['pace_cv']:.2%}")
        print(f"  → {'Low variance - very consistent!' if consistency['pace_cv'] < 0.15 else 'Moderate variance' if consistency['pace_cv'] < 0.25 else 'High variance - varying efforts'}")
        print(f"Average Distance: {consistency['mean_distance']:.2f} miles")
        print(f"Distance Variability (CV): {consistency['distance_cv']:.2%}")
        print(f"Longest Weekly Streak: {consistency['max_weekly_streak']} weeks")
    
    print("\n\n📦 VOLUME ANALYSIS")
    print("-" * 80)
    if volume_trends:
        print(f"Peak Weekly Mileage: {volume_trends['peak_week']:.1f} miles")
        print(f"Average Weekly Mileage: {volume_trends['avg_weekly']:.1f} miles")
        print(f"Peak Monthly Mileage: {volume_trends['peak_month']:.1f} miles")
        print(f"Total Weeks Tracked: {len(volume_trends['weekly_volume'])}")
        
        recent_4_weeks = volume_trends['weekly_volume'][-4:] if len(volume_trends['weekly_volume']) >= 4 else volume_trends['weekly_volume']
        print(f"Recent 4-Week Average: {np.mean(recent_4_weeks):.1f} miles")
    
    print("\n\n⚡ SPEED DISTRIBUTION")
    print("-" * 80)
    if speed_dist:
        print(f"Threshold Pace: {seconds_to_time_str(speed_dist['threshold_pace'])} min/mile")
        print(f"Tempo/Fast Runs: {speed_dist['tempo_count']} ({speed_dist['tempo_pct']:.1f}%)")
        print(f"Easy/Long Runs: {speed_dist['easy_count']} ({speed_dist['easy_pct']:.1f}%)")
        
        if speed_dist['easy_pct'] >= 70:
            print("  → Excellent 80/20 ratio - good base building!")
        elif speed_dist['easy_pct'] >= 60:
            print("  → Good balance for most training phases")
        else:
            print("  → Consider more easy runs for recovery and aerobic base")
    
    print("\n\n💡 KEY INSIGHTS & RECOMMENDATIONS")
    print("-" * 80)
    
    if consistency and consistency['pace_cv'] < 0.15:
        print("✓ Very consistent pacing - shows good control and discipline")
    
    if volume_trends and len(volume_trends['weekly_volume']) >= 4:
        recent_trend = np.polyfit(range(len(volume_trends['weekly_volume'][-8:])), 
                                  volume_trends['weekly_volume'][-8:], 1)[0]
        if recent_trend > 0.5:
            print("⚠ Volume increasing - monitor for 10% rule to prevent injury")
        elif recent_trend < -0.5:
            print("↓ Volume decreasing - taper or recovery phase?")
    
    if speed_dist and speed_dist['easy_pct'] < 50:
        print("⚠ High intensity ratio - ensure adequate recovery to prevent burnout")
    
    if consistency and consistency['max_weekly_streak'] >= 8:
        print(f"✓ Strong consistency with {consistency['max_weekly_streak']}-week streak!")
    
    print("\n" + "="*80 + "\n")

def main():
    print("Loading data...")
    df, outdoor_df = load_and_prepare_data()
    
    print("Analyzing personal records...")
    prs = analyze_personal_records(outdoor_df)
    
    print("Calculating consistency metrics...")
    consistency = calculate_consistency_metrics(outdoor_df)
    
    print("Analyzing volume trends...")
    volume_trends = analyze_volume_trends(df)
    
    print("Analyzing speed distribution...")
    speed_dist = analyze_speed_distribution(outdoor_df)
    
    print("Creating visualizations...")
    create_visualizations(outdoor_df, prs, consistency, volume_trends, speed_dist)
    
    print_report(outdoor_df, prs, consistency, volume_trends, speed_dist)

if __name__ == "__main__":
    main()
