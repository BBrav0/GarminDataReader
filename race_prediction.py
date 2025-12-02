#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Race Prediction & Performance Benchmarking
Estimates VO2max, VDOT scores, and predicts race times for various distances.
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

def seconds_to_time_str(seconds):
    """Convert seconds to HH:MM:SS format."""
    if np.isnan(seconds):
        return "N/A"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def load_and_prepare_data(csv_path='runs_data.csv'):
    """Load CSV and prepare data for analysis."""
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    df['pace_seconds'] = df['avg_pace'].apply(parse_time_to_seconds)
    df['duration_seconds'] = df['duration'].apply(parse_time_to_seconds)
    df['pace_per_km'] = df['pace_seconds'] / 1.60934  # Convert to pace per km
    df['speed_mps'] = np.where(df['pace_seconds'] > 0, 
                                1609.34 / df['pace_seconds'], np.nan)  # meters per second
    
    return df

def estimate_vo2max_daniels(pace_seconds_per_mile, duration_minutes):
    """
    Estimate VO2max using Jack Daniels' formula.
    Based on race performance (pace and duration).
    
    VO2 = -4.60 + 0.182258 × velocity + 0.000104 × velocity²
    Where velocity is in meters/minute
    """
    if np.isnan(pace_seconds_per_mile) or pace_seconds_per_mile <= 0:
        return np.nan
    
    # Convert to velocity in meters per minute
    velocity_mpm = 1609.34 / (pace_seconds_per_mile / 60)
    
    # Daniels formula
    vo2 = -4.60 + 0.182258 * velocity_mpm + 0.000104 * (velocity_mpm ** 2)
    
    # Adjust for duration (longer races use lower % of VO2max)
    if duration_minutes <= 8:  # < 8 min (1500m-ish)
        pct_vo2max = 1.0
    elif duration_minutes <= 15:  # 8-15 min (5K-ish)
        pct_vo2max = 0.98
    elif duration_minutes <= 30:  # 15-30 min (10K-ish)
        pct_vo2max = 0.95
    elif duration_minutes <= 60:  # 30-60 min (half-ish)
        pct_vo2max = 0.90
    elif duration_minutes <= 150:  # 1-2.5 hr (marathon-ish)
        pct_vo2max = 0.85
    else:  # Ultra
        pct_vo2max = 0.80
    
    vo2max = vo2 / pct_vo2max
    
    return vo2max

def calculate_vdot(vo2max):
    """
    Calculate VDOT score from VO2max.
    VDOT is a simpler representation of VO2max for training purposes.
    """
    # VDOT ≈ VO2max for practical purposes
    return vo2max

def predict_race_times_daniels(vdot):
    """
    Predict race times for various distances using VDOT tables.
    Based on Jack Daniels' Running Formula.
    
    Approximate formulas derived from VDOT tables:
    """
    if np.isnan(vdot) or vdot <= 0:
        return None
    
    # Velocity at VO2max (m/min)
    v_vo2max = (vdot + 4.60 - np.sqrt((vdot + 4.60)**2 - 4 * 0.000104 * vdot)) / (2 * 0.000104)
    
    predictions = {}
    
    # Race predictions with % of VO2max
    race_specs = {
        '5K': (5000, 0.98),
        '10K': (10000, 0.95),
        'Half Marathon': (21097.5, 0.88),
        'Marathon': (42195, 0.85)
    }
    
    for race_name, (distance_m, pct_vo2max) in race_specs.items():
        race_velocity = v_vo2max * pct_vo2max
        race_time_sec = distance_m / (race_velocity / 60)  # Convert to sec/meter then invert
        
        predictions[race_name] = {
            'time_seconds': race_time_sec,
            'time_formatted': seconds_to_time_str(race_time_sec),
            'pace_per_mile': (race_time_sec / (distance_m / 1609.34)) / 60,  # min/mile
            'pace_formatted': seconds_to_time_str((race_time_sec / (distance_m / 1609.34)))
        }
    
    return predictions

def calculate_vdot_from_performance(distance_miles, time_seconds):
    """Calculate VDOT directly from a race performance."""
    if time_seconds <= 0 or distance_miles <= 0:
        return np.nan
    
    pace_per_mile = time_seconds / distance_miles
    duration_minutes = time_seconds / 60
    
    vo2max = estimate_vo2max_daniels(pace_per_mile, duration_minutes)
    return calculate_vdot(vo2max)

def analyze_vdot_progression(df):
    """Analyze VDOT progression over time using best efforts."""
    runs = df[(df['distance'] >= 3) & (df['pace_seconds'] > 0) & 
              (df['activity_type'] == 'Run')].copy()
    
    if len(runs) == 0:
        return None
    
    vdots = []
    dates = []
    distances = []
    
    for idx, row in runs.iterrows():
        vdot = calculate_vdot_from_performance(row['distance'], row['duration_seconds'])
        if not np.isnan(vdot) and vdot > 20 and vdot < 90:  # Reasonable bounds
            vdots.append(vdot)
            dates.append(row['date'])
            distances.append(row['distance'])
    
    if len(vdots) == 0:
        return None
    
    vdots = np.array(vdots)
    dates = np.array(dates)
    distances = np.array(distances)
    
    # Calculate trend
    x_numeric = (pd.to_datetime(dates) - pd.to_datetime(dates).min()).days
    if len(vdots) >= 3:
        trend_coef = np.polyfit(x_numeric, vdots, 1)[0]
    else:
        trend_coef = 0
    
    return {
        'dates': dates,
        'vdots': vdots,
        'distances': distances,
        'current_vdot': vdots[-1],
        'max_vdot': np.max(vdots),
        'mean_vdot': np.mean(vdots),
        'recent_vdot': np.mean(vdots[-5:]) if len(vdots) >= 5 else np.mean(vdots),
        'trend_coef': trend_coef
    }

def analyze_pacing_strategy(df):
    """
    Analyze pacing consistency on long runs.
    Even pacing = race readiness.
    """
    long_runs = df[(df['distance'] >= 10) & (df['pace_seconds'] > 0) & 
                   (df['activity_type'] == 'Run')].copy()
    
    if len(long_runs) == 0:
        return None
    
    # Use pace variance as proxy for pacing consistency
    # (Real analysis would need split data, but we approximate)
    pace_cv = np.std(long_runs['pace_seconds']) / np.mean(long_runs['pace_seconds'])
    
    return {
        'long_run_count': len(long_runs),
        'avg_long_run_pace': np.mean(long_runs['pace_seconds']),
        'pace_consistency': 1 / (1 + pace_cv),  # 0-1 scale, higher is better
        'longest_run': long_runs['distance'].max(),
        'avg_long_run_distance': long_runs['distance'].mean()
    }

def calculate_competition_readiness(df, vdot_data, pacing_data, target_distance='Marathon'):
    """
    Calculate competition readiness score (0-100).
    Based on recent volume, intensity, long runs, and recovery.
    """
    if vdot_data is None:
        return None
    
    score_components = {}
    
    # 1. Recent volume (30 points)
    last_4_weeks = df[df['date'] >= (df['date'].max() - timedelta(days=28))]
    weekly_mileage = last_4_weeks['distance'].sum() / 4
    
    target_weekly = {
        '5K': 20,
        '10K': 30,
        'Half Marathon': 40,
        'Marathon': 50
    }
    
    volume_score = min(30, (weekly_mileage / target_weekly.get(target_distance, 40)) * 30)
    score_components['volume'] = volume_score
    
    # 2. Long run preparation (25 points)
    if pacing_data:
        target_long_run = {
            '5K': 8,
            '10K': 10,
            'Half Marathon': 15,
            'Marathon': 20
        }
        
        long_run_score = min(25, (pacing_data['longest_run'] / 
                                  target_long_run.get(target_distance, 15)) * 25)
        score_components['long_runs'] = long_run_score
    else:
        score_components['long_runs'] = 0
    
    # 3. VDOT trend (20 points)
    if vdot_data['trend_coef'] > 0.01:
        vdot_trend_score = 20
    elif vdot_data['trend_coef'] > -0.01:
        vdot_trend_score = 15
    else:
        vdot_trend_score = 10
    score_components['fitness_trend'] = vdot_trend_score
    
    # 4. Consistency (15 points)
    last_8_weeks = df[df['date'] >= (df['date'].max() - timedelta(days=56))]
    weeks_with_runs = len(last_8_weeks.groupby(last_8_weeks['date'].dt.isocalendar().week))
    consistency_score = (weeks_with_runs / 8) * 15
    score_components['consistency'] = consistency_score
    
    # 5. Recovery indicator (10 points)
    recent_rhr = df[df['resting_hr'] > 0].tail(5)
    if len(recent_rhr) >= 3:
        rhr_trend = np.polyfit(range(len(recent_rhr)), recent_rhr['resting_hr'], 1)[0]
        if rhr_trend < 0:  # Improving
            recovery_score = 10
        elif rhr_trend < 0.5:
            recovery_score = 7
        else:
            recovery_score = 4
    else:
        recovery_score = 5
    score_components['recovery'] = recovery_score
    
    total_score = sum(score_components.values())
    
    return {
        'total_score': total_score,
        'components': score_components,
        'rating': 'Excellent' if total_score >= 80 else 
                 'Good' if total_score >= 65 else 
                 'Fair' if total_score >= 50 else 'Needs Work'
    }

def create_visualizations(df, vdot_data, predictions, readiness):
    """Create comprehensive visualization charts."""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # 1. VDOT progression over time
    ax1 = fig.add_subplot(gs[0, :])
    if vdot_data:
        scatter = ax1.scatter(vdot_data['dates'], vdot_data['vdots'],
                             c=vdot_data['distances'], cmap='viridis',
                             s=80, alpha=0.6, edgecolors='black', linewidth=0.5)
        
        # Trend line
        x_numeric = (pd.to_datetime(vdot_data['dates']) - pd.to_datetime(vdot_data['dates']).min()).days
        z = np.polyfit(x_numeric, vdot_data['vdots'], 1)
        p = np.poly1d(z)
        trend_dates = pd.to_datetime(vdot_data['dates']).min() + pd.to_timedelta(np.linspace(0, x_numeric.max(), 100), unit='D')
        ax1.plot(trend_dates, p(np.linspace(0, x_numeric.max(), 100)),
                'r--', linewidth=2.5, alpha=0.7, label=f'Trend: {vdot_data["trend_coef"]:.3f} VDOT/day')
        
        ax1.axhline(vdot_data['max_vdot'], color='green', linestyle='--', 
                   alpha=0.5, label=f'Peak: {vdot_data["max_vdot"]:.1f}')
        
        ax1.set_xlabel('Date', fontweight='bold', fontsize=11)
        ax1.set_ylabel('VDOT Score', fontweight='bold', fontsize=11)
        ax1.set_title(f'VDOT Progression (Current: {vdot_data["current_vdot"]:.1f})', 
                     fontweight='bold', fontsize=14)
        ax1.grid(alpha=0.3)
        ax1.legend(fontsize=10)
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Distance (miles)', fontweight='bold')
    
    # 2. Race time predictions
    ax2 = fig.add_subplot(gs[1, 0])
    if predictions:
        races = list(predictions.keys())
        times = [predictions[r]['time_seconds'] / 60 for r in races]  # Convert to minutes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        bars = ax2.barh(races, times, color=colors, edgecolor='black', linewidth=1)
        ax2.set_xlabel('Time (minutes)', fontweight='bold', fontsize=11)
        ax2.set_title('Predicted Race Times (Based on Current VDOT)', 
                     fontweight='bold', fontsize=12)
        ax2.grid(alpha=0.3, axis='x')
        
        # Add time labels
        for bar, race in zip(bars, races):
            width = bar.get_width()
            ax2.text(width, bar.get_y() + bar.get_height()/2,
                    f' {predictions[race]["time_formatted"]}',
                    ha='left', va='center', fontweight='bold', fontsize=10)
    
    # 3. Race pace predictions
    ax3 = fig.add_subplot(gs[1, 1])
    if predictions:
        races = list(predictions.keys())
        paces = [predictions[r]['pace_per_mile'] for r in races]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        bars = ax3.barh(races, paces, color=colors, edgecolor='black', linewidth=1)
        ax3.set_xlabel('Pace (min/mile)', fontweight='bold', fontsize=11)
        ax3.set_title('Predicted Race Paces', fontweight='bold', fontsize=12)
        ax3.grid(alpha=0.3, axis='x')
        
        # Add pace labels
        for bar, race in zip(bars, races):
            width = bar.get_width()
            ax3.text(width, bar.get_y() + bar.get_height()/2,
                    f' {predictions[race]["pace_formatted"]}/mi',
                    ha='left', va='center', fontweight='bold', fontsize=10)
    
    # 4. Competition readiness breakdown
    ax4 = fig.add_subplot(gs[2, 0])
    if readiness:
        components = readiness['components']
        labels = ['Volume\n(30)', 'Long Runs\n(25)', 'Fitness\n(20)', 'Consistency\n(15)', 'Recovery\n(10)']
        values = [components['volume'], components['long_runs'], components['fitness_trend'],
                 components['consistency'], components['recovery']]
        max_values = [30, 25, 20, 15, 10]
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, max_values, width, label='Maximum', 
                       color='lightgray', edgecolor='black', linewidth=0.5)
        bars2 = ax4.bar(x + width/2, values, width, label='Current',
                       color='steelblue', edgecolor='black', linewidth=0.5)
        
        ax4.set_ylabel('Points', fontweight='bold', fontsize=11)
        ax4.set_title(f'Competition Readiness: {readiness["total_score"]:.0f}/100 ({readiness["rating"]})',
                     fontweight='bold', fontsize=12)
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels, fontsize=9)
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
    
    # 5. VDOT distribution
    ax5 = fig.add_subplot(gs[2, 1])
    if vdot_data:
        ax5.hist(vdot_data['vdots'], bins=15, color='coral', alpha=0.7, 
                edgecolor='black', linewidth=0.5)
        ax5.axvline(vdot_data['current_vdot'], color='red', linestyle='--', 
                   linewidth=2, label=f'Current: {vdot_data["current_vdot"]:.1f}')
        ax5.axvline(vdot_data['mean_vdot'], color='blue', linestyle='--',
                   linewidth=2, label=f'Mean: {vdot_data["mean_vdot"]:.1f}')
        ax5.set_xlabel('VDOT Score', fontweight='bold', fontsize=11)
        ax5.set_ylabel('Frequency', fontweight='bold', fontsize=11)
        ax5.set_title('VDOT Score Distribution', fontweight='bold', fontsize=12)
        ax5.legend()
        ax5.grid(alpha=0.3, axis='y')
    
    plt.savefig('race_prediction.png', dpi=300, bbox_inches='tight')
    print("[OK] Chart saved as 'race_prediction.png'")
    plt.close()

def print_report(vdot_data, predictions, pacing_data, readiness):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print(" RACE PREDICTION & PERFORMANCE BENCHMARKING ".center(80, "="))
    print("="*80)
    
    print("\n📊 VDOT SCORE & VO2MAX ESTIMATION")
    print("-" * 80)
    if vdot_data:
        print(f"Current VDOT: {vdot_data['current_vdot']:.1f}")
        print(f"Peak VDOT: {vdot_data['max_vdot']:.1f}")
        print(f"Mean VDOT: {vdot_data['mean_vdot']:.1f}")
        print(f"Recent Average (last 5): {vdot_data['recent_vdot']:.1f}")
        print(f"Trend: {vdot_data['trend_coef']:.3f} VDOT points/day")
        
        # VDOT interpretation
        current = vdot_data['current_vdot']
        if current >= 70:
            level = "Elite / National class"
        elif current >= 60:
            level = "Very competitive / Sub-elite"
        elif current >= 50:
            level = "Competitive runner"
        elif current >= 40:
            level = "Above average recreational"
        elif current >= 30:
            level = "Average recreational"
        else:
            level = "Beginner / Developing fitness"
        
        print(f"\nFitness Level: {level}")
        
        if vdot_data['trend_coef'] > 0.01:
            print("Trend: ✓ Improving fitness - training is working!")
        elif vdot_data['trend_coef'] > -0.01:
            print("Trend: → Maintaining fitness - stable performance")
        else:
            print("Trend: ↓ Declining fitness - check training load and recovery")
    
    print("\n\n🏁 RACE TIME PREDICTIONS (Based on Current VDOT)")
    print("-" * 80)
    if predictions:
        print(f"{'Distance':<20} {'Time':<12} {'Pace':<15}")
        print("-" * 80)
        for race, pred in predictions.items():
            print(f"{race:<20} {pred['time_formatted']:<12} {pred['pace_formatted']}/mi")
        
        print("\nNote: Predictions assume:")
        print("  • Proper race-specific training")
        print("  • Appropriate taper")
        print("  • Good race day conditions")
        print("  • Even pacing strategy")
    
    print("\n\n📏 PACING STRATEGY ANALYSIS")
    print("-" * 80)
    if pacing_data:
        print(f"Long Runs (≥10 miles): {pacing_data['long_run_count']}")
        print(f"Longest Run: {pacing_data['longest_run']:.1f} miles")
        print(f"Average Long Run: {pacing_data['avg_long_run_distance']:.1f} miles")
        print(f"Pacing Consistency: {pacing_data['pace_consistency']:.2%}")
        
        if pacing_data['pace_consistency'] > 0.85:
            print("Status: ✓ Excellent pacing discipline")
        elif pacing_data['pace_consistency'] > 0.70:
            print("Status: → Good pacing, room for improvement")
        else:
            print("Status: ⚠ Work on even-pacing in long runs")
    else:
        print("Insufficient long run data (need runs ≥10 miles)")
    
    print("\n\n🎯 COMPETITION READINESS SCORE")
    print("-" * 80)
    if readiness:
        print(f"Overall Score: {readiness['total_score']:.0f}/100")
        print(f"Rating: {readiness['rating']}")
        print("\nBreakdown:")
        print(f"  Volume (30 max):      {readiness['components']['volume']:.1f}")
        print(f"  Long Runs (25 max):   {readiness['components']['long_runs']:.1f}")
        print(f"  Fitness Trend (20 max): {readiness['components']['fitness_trend']:.1f}")
        print(f"  Consistency (15 max): {readiness['components']['consistency']:.1f}")
        print(f"  Recovery (10 max):    {readiness['components']['recovery']:.1f}")
        
        if readiness['total_score'] >= 80:
            print("\n✓ You are in excellent racing shape!")
        elif readiness['total_score'] >= 65:
            print("\n→ Good fitness - should be able to race well")
        elif readiness['total_score'] >= 50:
            print("\n⚠ Fair fitness - continue building for best results")
        else:
            print("\n📈 Build more base before racing - focus on volume and consistency")
    
    print("\n\n💡 TRAINING RECOMMENDATIONS")
    print("-" * 80)
    
    if vdot_data:
        current_vdot = vdot_data['current_vdot']
        
        # Calculate training paces based on VDOT
        print("Recommended Training Paces (based on current VDOT):")
        
        # Easy pace: 60-70% effort (add ~2 min to 5K pace)
        if predictions and '5K' in predictions:
            k5_pace = predictions['5K']['pace_per_mile']
            easy_pace = k5_pace + 2.0
            tempo_pace = k5_pace + 0.5
            interval_pace = k5_pace - 0.3
            
            print(f"  Easy/Recovery:  {seconds_to_time_str(easy_pace * 60)}/mile (conversational)")
            print(f"  Marathon Pace:  {predictions['Marathon']['pace_formatted']}/mile")
            print(f"  Tempo/Threshold: {seconds_to_time_str(tempo_pace * 60)}/mile (comfortably hard)")
            print(f"  Interval/5K:    {seconds_to_time_str(interval_pace * 60)}/mile (hard)")
    
    if pacing_data and pacing_data['longest_run'] < 20:
        print("\n  • Build long runs: gradually increase to 20+ miles for marathon prep")
    
    if readiness and readiness['components']['consistency'] < 12:
        print("  • Focus on consistency: aim for regular weekly running schedule")
    
    if readiness and readiness['components']['volume'] < 20:
        print("  • Increase weekly volume gradually (10% per week max)")
    
    print("\n" + "="*80 + "\n")

def main():
    print("Loading data...")
    df = load_and_prepare_data()
    
    print("Analyzing VDOT progression...")
    vdot_data = analyze_vdot_progression(df)
    
    if vdot_data:
        print("Predicting race times...")
        predictions = predict_race_times_daniels(vdot_data['current_vdot'])
    else:
        predictions = None
    
    print("Analyzing pacing strategy...")
    pacing_data = analyze_pacing_strategy(df)
    
    print("Calculating competition readiness...")
    readiness = calculate_competition_readiness(df, vdot_data, pacing_data)
    
    print("Creating visualizations...")
    create_visualizations(df, vdot_data, predictions, readiness)
    
    print_report(vdot_data, predictions, pacing_data, readiness)

if __name__ == "__main__":
    main()
