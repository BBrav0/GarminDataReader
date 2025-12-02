#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training Load & Recovery Analysis
Analyzes ACWR, training monotony, recovery patterns, and injury risk factors.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

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

def load_and_prepare_data(csv_path='runs_data.csv'):
    """Load CSV and prepare data for analysis."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df['duration_seconds'] = df['duration'].apply(parse_time_to_seconds)
    df['duration_hours'] = df['duration_seconds'] / 3600
    
    # Training load = distance * intensity factor (using HR if available)
    df['intensity_factor'] = np.where(
        (df['avghr'] > 0) & (df['maxhr'] > 0),
        df['avghr'] / df['maxhr'],
        1.0
    )
    df['training_load'] = df['distance'] * df['intensity_factor']
    
    return df

def calculate_acwr(df, acute_days=7, chronic_days=28):
    """
    Calculate Acute:Chronic Workload Ratio (ACWR).
    ACWR < 0.8: undertraining
    ACWR 0.8-1.3: optimal (sweet spot)
    ACWR > 1.5: high injury risk
    """
    df = df.sort_values('date')
    
    acwr_data = []
    dates = []
    acute_loads = []
    chronic_loads = []
    
    for idx, row in df.iterrows():
        current_date = row['date']
        
        # Acute workload (last 7 days)
        acute_start = current_date - timedelta(days=acute_days)
        acute_window = df[(df['date'] > acute_start) & (df['date'] <= current_date)]
        acute_load = acute_window['training_load'].sum()
        
        # Chronic workload (last 28 days)
        chronic_start = current_date - timedelta(days=chronic_days)
        chronic_window = df[(df['date'] > chronic_start) & (df['date'] <= current_date)]
        chronic_load = chronic_window['training_load'].sum()
        
        if chronic_load > 0:
            acwr = acute_load / chronic_load
            acwr_data.append(acwr)
            dates.append(current_date)
            acute_loads.append(acute_load)
            chronic_loads.append(chronic_load)
    
    return np.array(dates), np.array(acwr_data), np.array(acute_loads), np.array(chronic_loads)

def calculate_training_monotony(df, window_days=7):
    """
    Calculate training monotony and strain.
    Monotony = mean / std of training load
    High monotony (>2.0) + high load = increased injury risk
    """
    df = df.sort_values('date')
    
    monotony_data = []
    strain_data = []
    dates = []
    
    for idx, row in df.iterrows():
        current_date = row['date']
        window_start = current_date - timedelta(days=window_days)
        window = df[(df['date'] > window_start) & (df['date'] <= current_date)]
        
        if len(window) >= 3:
            loads = window['training_load'].values
            mean_load = np.mean(loads)
            std_load = np.std(loads)
            
            if std_load > 0:
                monotony = mean_load / std_load
                strain = np.sum(loads) * monotony
                
                monotony_data.append(monotony)
                strain_data.append(strain)
                dates.append(current_date)
    
    return np.array(dates), np.array(monotony_data), np.array(strain_data)

def analyze_resting_hr_trends(df):
    """
    Analyze resting heart rate trends as cardiovascular fitness indicator.
    Decreasing RHR = improving fitness
    Elevated RHR = fatigue/overtraining/illness
    """
    rhr_data = df[df['resting_hr'] > 0].copy()
    rhr_data = rhr_data.sort_values('date')
    
    if len(rhr_data) < 5:
        return None
    
    # Calculate rolling average
    rhr_data['rhr_7day_avg'] = rhr_data['resting_hr'].rolling(window=7, min_periods=3).mean()
    
    # Calculate trend
    x_numeric = (rhr_data['date'] - rhr_data['date'].min()).dt.days
    valid_idx = ~np.isnan(rhr_data['rhr_7day_avg'])
    
    if np.sum(valid_idx) >= 2:
        trend_coef = np.polyfit(x_numeric[valid_idx], rhr_data['rhr_7day_avg'][valid_idx], 1)[0]
    else:
        trend_coef = 0
    
    return {
        'dates': rhr_data['date'].values,
        'rhr': rhr_data['resting_hr'].values,
        'rhr_smooth': rhr_data['rhr_7day_avg'].values,
        'trend_coef': trend_coef,
        'mean_rhr': np.mean(rhr_data['resting_hr']),
        'min_rhr': np.min(rhr_data['resting_hr']),
        'max_rhr': np.max(rhr_data['resting_hr'])
    }

def calculate_hard_easy_ratio(df):
    """
    Calculate ratio of hard to easy runs.
    Hard = high HR (>80% max HR) or fast pace
    """
    outdoor_runs = df[(df['activity_type'] == 'Run') & (df['avghr'] > 0) & (df['maxhr'] > 0)].copy()
    
    if len(outdoor_runs) == 0:
        return None
    
    # Classify as hard if avg HR > 80% of max HR
    outdoor_runs['hr_percentage'] = (outdoor_runs['avghr'] / outdoor_runs['maxhr']) * 100
    outdoor_runs['is_hard'] = outdoor_runs['hr_percentage'] > 80
    
    hard_count = outdoor_runs['is_hard'].sum()
    easy_count = len(outdoor_runs) - hard_count
    
    return {
        'hard_count': hard_count,
        'easy_count': easy_count,
        'hard_pct': (hard_count / len(outdoor_runs)) * 100,
        'easy_pct': (easy_count / len(outdoor_runs)) * 100,
        'total_runs': len(outdoor_runs)
    }

def calculate_fatigue_index(df):
    """
    Calculate cumulative fatigue index.
    Considers load accumulation and recovery time between runs.
    """
    df = df.sort_values('date')
    
    fatigue_values = []
    dates = []
    recovery_decay = 0.15  # Daily recovery rate
    
    cumulative_fatigue = 0
    prev_date = None
    
    for idx, row in df.iterrows():
        current_date = row['date']
        
        if prev_date is not None:
            days_since_last = (current_date - prev_date).days
            cumulative_fatigue *= (1 - recovery_decay) ** days_since_last
        
        # Add today's load
        cumulative_fatigue += row['training_load']
        
        fatigue_values.append(cumulative_fatigue)
        dates.append(current_date)
        prev_date = current_date
    
    return np.array(dates), np.array(fatigue_values)

def identify_overtraining_signals(acwr_data, monotony_data, rhr_trends, fatigue_data):
    """Identify potential overtraining signals."""
    signals = []
    
    # High ACWR
    if len(acwr_data) > 0:
        recent_acwr = acwr_data[-5:] if len(acwr_data) >= 5 else acwr_data
        if np.any(recent_acwr > 1.5):
            signals.append("⚠ High ACWR detected (>1.5) - elevated injury risk")
    
    # High monotony
    if len(monotony_data) > 0:
        recent_monotony = monotony_data[-5:] if len(monotony_data) >= 5 else monotony_data
        if np.any(recent_monotony > 2.0):
            signals.append("⚠ High training monotony (>2.0) - add variety to prevent burnout")
    
    # Elevated RHR
    if rhr_trends is not None:
        recent_rhr = rhr_trends['rhr'][-7:] if len(rhr_trends['rhr']) >= 7 else rhr_trends['rhr']
        if np.mean(recent_rhr) > rhr_trends['mean_rhr'] + 5:
            signals.append("⚠ Elevated resting HR - possible fatigue or illness")
        
        if rhr_trends['trend_coef'] > 0.1:
            signals.append("↑ Rising RHR trend - monitor recovery quality")
    
    # High fatigue
    if len(fatigue_data) > 0:
        recent_fatigue = fatigue_data[-5:] if len(fatigue_data) >= 5 else fatigue_data
        if len(fatigue_data) >= 10:
            historical_avg = np.mean(fatigue_data[:-5])
            if np.mean(recent_fatigue) > historical_avg * 1.5:
                signals.append("⚠ Elevated cumulative fatigue - consider recovery week")
    
    return signals

def create_visualizations(df, acwr_dates, acwr_data, acute_loads, chronic_loads,
                         mono_dates, monotony_data, strain_data,
                         rhr_trends, fatigue_dates, fatigue_data, hard_easy):
    """Create comprehensive visualization charts."""
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.3)
    
    # 1. ACWR over time
    ax1 = fig.add_subplot(gs[0, :])
    if len(acwr_data) > 0:
        ax1.plot(acwr_dates, acwr_data, 'b-', linewidth=2, label='ACWR')
        ax1.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Undertraining (<0.8)')
        ax1.axhline(y=1.3, color='orange', linestyle='--', alpha=0.5, label='Upper optimal (1.3)')
        ax1.axhline(y=1.5, color='red', linestyle='--', alpha=0.5, label='Injury risk (>1.5)')
        ax1.fill_between(acwr_dates, 0.8, 1.3, alpha=0.2, color='green', label='Sweet spot')
        ax1.fill_between(acwr_dates, 1.5, np.max(acwr_data) * 1.1, alpha=0.2, color='red')
        ax1.set_xlabel('Date', fontweight='bold')
        ax1.set_ylabel('ACWR', fontweight='bold')
        ax1.set_title('Acute:Chronic Workload Ratio (7:28 day)', fontweight='bold', fontsize=14)
        ax1.grid(alpha=0.3)
        ax1.legend(loc='upper left', fontsize=8)
    
    # 2. Acute vs Chronic loads
    ax2 = fig.add_subplot(gs[1, 0])
    if len(acwr_data) > 0:
        ax2.plot(acwr_dates, acute_loads, 'r-', linewidth=2, label='Acute (7-day)', marker='o', markersize=3)
        ax2.plot(acwr_dates, chronic_loads, 'b-', linewidth=2, label='Chronic (28-day)', marker='s', markersize=3)
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.set_ylabel('Training Load', fontweight='bold')
        ax2.set_title('Acute vs Chronic Training Loads', fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.legend()
    
    # 3. Training monotony
    ax3 = fig.add_subplot(gs[1, 1])
    if len(monotony_data) > 0:
        ax3.plot(mono_dates, monotony_data, 'purple', linewidth=2, marker='o', markersize=4)
        ax3.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='High monotony (>2.0)')
        ax3.fill_between(mono_dates, 0, 2.0, alpha=0.2, color='green')
        ax3.fill_between(mono_dates, 2.0, np.max(monotony_data) * 1.1, alpha=0.2, color='red')
        ax3.set_xlabel('Date', fontweight='bold')
        ax3.set_ylabel('Monotony', fontweight='bold')
        ax3.set_title('Training Monotony (Mean/SD of 7-day load)', fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.legend()
    
    # 4. Training strain
    ax4 = fig.add_subplot(gs[2, 0])
    if len(strain_data) > 0:
        ax4.plot(mono_dates, strain_data, 'darkred', linewidth=2, marker='s', markersize=4)
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_ylabel('Strain', fontweight='bold')
        ax4.set_title('Training Strain (Load × Monotony)', fontweight='bold')
        ax4.grid(alpha=0.3)
    
    # 5. Resting heart rate trends
    ax5 = fig.add_subplot(gs[2, 1])
    if rhr_trends is not None:
        ax5.scatter(rhr_trends['dates'], rhr_trends['rhr'], alpha=0.5, s=30, color='lightcoral', label='Daily RHR')
        
        valid_idx = ~np.isnan(rhr_trends['rhr_smooth'])
        ax5.plot(rhr_trends['dates'][valid_idx], rhr_trends['rhr_smooth'][valid_idx],
                'r-', linewidth=2.5, label='7-day average')
        
        ax5.axhline(y=rhr_trends['mean_rhr'], color='blue', linestyle='--', alpha=0.5, label=f"Mean: {rhr_trends['mean_rhr']:.0f}")
        ax5.set_xlabel('Date', fontweight='bold')
        ax5.set_ylabel('Resting HR (bpm)', fontweight='bold')
        ax5.set_title('Resting Heart Rate Trends', fontweight='bold')
        ax5.grid(alpha=0.3)
        ax5.legend()
    
    # 6. Cumulative fatigue
    ax6 = fig.add_subplot(gs[3, 0])
    if len(fatigue_data) > 0:
        ax6.fill_between(fatigue_dates, 0, fatigue_data, alpha=0.3, color='orange')
        ax6.plot(fatigue_dates, fatigue_data, 'darkorange', linewidth=2)
        ax6.set_xlabel('Date', fontweight='bold')
        ax6.set_ylabel('Fatigue Index', fontweight='bold')
        ax6.set_title('Cumulative Fatigue Index (with 15% daily recovery)', fontweight='bold')
        ax6.grid(alpha=0.3)
    
    # 7. Hard/Easy ratio
    ax7 = fig.add_subplot(gs[3, 1])
    if hard_easy is not None:
        labels = ['Easy (<80% max HR)', 'Hard (>80% max HR)']
        sizes = [hard_easy['easy_count'], hard_easy['hard_count']]
        colors = ['#4ecdc4', '#ff6b6b']
        explode = (0.05, 0.05)
        
        ax7.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontweight': 'bold', 'fontsize': 10})
        ax7.set_title(f"Hard vs Easy Run Distribution (n={hard_easy['total_runs']})", fontweight='bold')
    
    plt.savefig('training_load_recovery.png', dpi=300, bbox_inches='tight')
    print("[OK] Chart saved as 'training_load_recovery.png'")
    plt.close()

def print_report(acwr_data, monotony_data, rhr_trends, hard_easy, signals):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print(" TRAINING LOAD & RECOVERY ANALYSIS REPORT ".center(80, "="))
    print("="*80)
    
    print("\n⚖️ ACUTE:CHRONIC WORKLOAD RATIO (ACWR)")
    print("-" * 80)
    if len(acwr_data) > 0:
        current_acwr = acwr_data[-1]
        recent_acwr = acwr_data[-4:] if len(acwr_data) >= 4 else acwr_data
        
        print(f"Current ACWR: {current_acwr:.2f}")
        print(f"Recent average (last 4 runs): {np.mean(recent_acwr):.2f}")
        print(f"Range (all time): {np.min(acwr_data):.2f} - {np.max(acwr_data):.2f}")
        
        if current_acwr < 0.8:
            status = "📉 Undertraining - consider increasing volume"
        elif 0.8 <= current_acwr <= 1.3:
            status = "✓ Optimal range - low injury risk"
        elif 1.3 < current_acwr <= 1.5:
            status = "⚠ Moderate - monitor closely"
        else:
            status = "🚨 High injury risk - reduce load immediately"
        
        print(f"Status: {status}")
    else:
        print("Insufficient data for ACWR calculation")
    
    print("\n\n🔄 TRAINING MONOTONY")
    print("-" * 80)
    if len(monotony_data) > 0:
        current_monotony = monotony_data[-1]
        avg_monotony = np.mean(monotony_data)
        
        print(f"Current Monotony: {current_monotony:.2f}")
        print(f"Average Monotony: {avg_monotony:.2f}")
        
        if current_monotony < 1.5:
            print("Status: ✓ Good variety in training")
        elif current_monotony < 2.0:
            print("Status: → Moderate - acceptable")
        else:
            print("Status: ⚠ High - add more variety to prevent burnout")
    else:
        print("Insufficient data for monotony calculation")
    
    print("\n\n❤️ RESTING HEART RATE ANALYSIS")
    print("-" * 80)
    if rhr_trends is not None:
        print(f"Mean RHR: {rhr_trends['mean_rhr']:.0f} bpm")
        print(f"Lowest RHR: {rhr_trends['min_rhr']:.0f} bpm")
        print(f"Highest RHR: {rhr_trends['max_rhr']:.0f} bpm")
        print(f"Trend: {rhr_trends['trend_coef']:.3f} bpm/day")
        
        if rhr_trends['trend_coef'] < -0.05:
            print("Status: ✓ Improving cardiovascular fitness (RHR decreasing)")
        elif rhr_trends['trend_coef'] > 0.1:
            print("Status: ⚠ RHR increasing - monitor recovery and stress")
        else:
            print("Status: → Stable RHR")
    else:
        print("Insufficient resting HR data")
    
    print("\n\n💪 HARD/EASY RUN BALANCE")
    print("-" * 80)
    if hard_easy is not None:
        print(f"Hard Runs (>80% max HR): {hard_easy['hard_count']} ({hard_easy['hard_pct']:.1f}%)")
        print(f"Easy Runs (<80% max HR): {hard_easy['easy_count']} ({hard_easy['easy_pct']:.1f}%)")
        
        if hard_easy['easy_pct'] >= 80:
            print("Status: ✓ Excellent 80/20 ratio - optimal for base building")
        elif hard_easy['easy_pct'] >= 70:
            print("Status: ✓ Good balance for most training phases")
        elif hard_easy['easy_pct'] >= 60:
            print("Status: → Moderate - ensure adequate recovery")
        else:
            print("Status: ⚠ Too many hard runs - increase easy mileage")
    else:
        print("Insufficient heart rate data")
    
    print("\n\n🚨 OVERTRAINING SIGNALS")
    print("-" * 80)
    if signals:
        for signal in signals:
            print(f"  {signal}")
    else:
        print("  ✓ No overtraining signals detected - training appears well-managed")
    
    print("\n\n💡 RECOVERY RECOMMENDATIONS")
    print("-" * 80)
    
    if len(acwr_data) > 0 and acwr_data[-1] > 1.5:
        print("  • URGENT: Reduce training volume by 20-30% next week")
        print("  • Focus on easy runs and active recovery")
    
    if hard_easy is not None and hard_easy['easy_pct'] < 70:
        print("  • Increase proportion of easy runs (target 80% easy, 20% hard)")
        print("  • Easy pace should be conversational - able to talk in full sentences")
    
    if len(monotony_data) > 0 and monotony_data[-1] > 2.0:
        print("  • Add variety: mix different distances, paces, and terrains")
        print("  • Include cross-training or rest days")
    
    if rhr_trends is not None and rhr_trends['trend_coef'] > 0.1:
        print("  • Prioritize sleep quality and quantity (7-9 hours)")
        print("  • Monitor stress levels and consider reducing volume temporarily")
        print("  • Stay hydrated and maintain proper nutrition")
    
    print("\n" + "="*80 + "\n")

def main():
    print("Loading data...")
    df = load_and_prepare_data()
    
    print("Calculating ACWR...")
    acwr_dates, acwr_data, acute_loads, chronic_loads = calculate_acwr(df)
    
    print("Calculating training monotony...")
    mono_dates, monotony_data, strain_data = calculate_training_monotony(df)
    
    print("Analyzing resting HR trends...")
    rhr_trends = analyze_resting_hr_trends(df)
    
    print("Calculating hard/easy ratio...")
    hard_easy = calculate_hard_easy_ratio(df)
    
    print("Calculating fatigue index...")
    fatigue_dates, fatigue_data = calculate_fatigue_index(df)
    
    print("Identifying overtraining signals...")
    signals = identify_overtraining_signals(acwr_data, monotony_data, rhr_trends, fatigue_data)
    
    print("Creating visualizations...")
    create_visualizations(df, acwr_dates, acwr_data, acute_loads, chronic_loads,
                         mono_dates, monotony_data, strain_data,
                         rhr_trends, fatigue_dates, fatigue_data, hard_easy)
    
    print_report(acwr_data, monotony_data, rhr_trends, hard_easy, signals)

if __name__ == "__main__":
    main()
