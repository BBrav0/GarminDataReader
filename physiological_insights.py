#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Physiological Insights Analysis
Analyzes HR zones, aerobic efficiency, cardiac drift, and cadence patterns.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
    
    df['pace_seconds'] = df['avg_pace'].apply(parse_time_to_seconds)
    df['duration_seconds'] = df['duration'].apply(parse_time_to_seconds)
    df['pace_minutes'] = df['pace_seconds'] / 60
    
    return df

def estimate_max_hr(df):
    """Estimate maximum heart rate from data."""
    recorded_max = df['maxhr'].max()
    if recorded_max > 0:
        return recorded_max
    
    # Fallback: 220 - age (can't determine age, so use recorded max)
    return 200  # Conservative estimate

def calculate_hr_zones(df, max_hr):
    """
    Calculate time/runs in different HR zones.
    Zone 1: 50-60% max HR (Recovery)
    Zone 2: 60-70% max HR (Aerobic base)
    Zone 3: 70-80% max HR (Tempo)
    Zone 4: 80-90% max HR (Threshold)
    Zone 5: 90-100% max HR (VO2 max)
    """
    hr_runs = df[(df['avghr'] > 0) & (df['maxhr'] > 0)].copy()
    
    if len(hr_runs) == 0:
        return None
    
    hr_runs['hr_percentage'] = (hr_runs['avghr'] / max_hr) * 100
    
    # Classify into zones
    def get_zone(hr_pct):
        if hr_pct < 60:
            return 'Z1 Recovery'
        elif hr_pct < 70:
            return 'Z2 Aerobic'
        elif hr_pct < 80:
            return 'Z3 Tempo'
        elif hr_pct < 90:
            return 'Z4 Threshold'
        else:
            return 'Z5 VO2max'
    
    hr_runs['zone'] = hr_runs['hr_percentage'].apply(get_zone)
    
    # Calculate time in each zone
    zone_time = hr_runs.groupby('zone')['duration_seconds'].sum() / 3600  # Convert to hours
    zone_counts = hr_runs['zone'].value_counts()
    
    zone_info = {}
    for zone in ['Z1 Recovery', 'Z2 Aerobic', 'Z3 Tempo', 'Z4 Threshold', 'Z5 VO2max']:
        zone_info[zone] = {
            'count': zone_counts.get(zone, 0),
            'hours': zone_time.get(zone, 0),
            'pct_runs': (zone_counts.get(zone, 0) / len(hr_runs)) * 100
        }
    
    return zone_info, hr_runs

def calculate_aerobic_efficiency(df):
    """
    Calculate aerobic efficiency as pace/HR ratio.
    Lower HR at same pace = better aerobic fitness.
    Track improvement over time.
    """
    efficient_runs = df[(df['avghr'] > 0) & (df['pace_seconds'] > 0) & 
                        (df['activity_type'] == 'Run')].copy()
    
    if len(efficient_runs) == 0:
        return None
    
    # Aerobic efficiency = (pace in min/mile) / (avghr)
    # Lower is better (faster pace at lower HR)
    efficient_runs['efficiency'] = efficient_runs['pace_minutes'] / efficient_runs['avghr']
    
    # Normalize by multiplying by 10000 for readability
    efficient_runs['efficiency_score'] = 10000 / efficient_runs['efficiency']
    
    # Calculate trend over time
    x_numeric = (efficient_runs['date'] - efficient_runs['date'].min()).dt.days
    
    if len(efficient_runs) >= 5:
        trend_coef = np.polyfit(x_numeric, efficient_runs['efficiency_score'], 1)[0]
    else:
        trend_coef = 0
    
    return {
        'dates': efficient_runs['date'].values,
        'efficiency_scores': efficient_runs['efficiency_score'].values,
        'pace': efficient_runs['pace_minutes'].values,
        'hr': efficient_runs['avghr'].values,
        'trend_coef': trend_coef,
        'mean_efficiency': np.mean(efficient_runs['efficiency_score']),
        'recent_efficiency': np.mean(efficient_runs['efficiency_score'].values[-5:]) if len(efficient_runs) >= 5 else np.mean(efficient_runs['efficiency_score'])
    }

def analyze_cardiac_drift(df):
    """
    Analyze cardiac drift on long runs (>60 min).
    Cardiac drift = HR increase over duration despite steady pace.
    Indicates heat stress, dehydration, or aerobic fitness.
    """
    long_runs = df[(df['duration_seconds'] > 3600) & (df['avghr'] > 0) & 
                   (df['minhr'] > 0) & (df['maxhr'] > 0)].copy()
    
    if len(long_runs) == 0:
        return None
    
    # Estimate drift: (maxhr - minhr) / avghr
    long_runs['drift_index'] = ((long_runs['maxhr'] - long_runs['minhr']) / long_runs['avghr']) * 100
    
    return {
        'dates': long_runs['date'].values,
        'drift_indices': long_runs['drift_index'].values,
        'distances': long_runs['distance'].values,
        'durations': long_runs['duration_seconds'].values / 3600,
        'mean_drift': np.mean(long_runs['drift_index']),
        'max_drift': np.max(long_runs['drift_index'])
    }

def calculate_hr_reserve_utilization(df, max_hr):
    """
    Calculate heart rate reserve (HRR) utilization.
    HRR = max_hr - resting_hr
    % HRR used = (avghr - resting_hr) / HRR
    """
    hrr_runs = df[(df['avghr'] > 0) & (df['resting_hr'] > 0)].copy()
    
    if len(hrr_runs) == 0:
        return None
    
    hrr_runs['hrr'] = max_hr - hrr_runs['resting_hr']
    hrr_runs['hrr_utilization'] = ((hrr_runs['avghr'] - hrr_runs['resting_hr']) / hrr_runs['hrr']) * 100
    
    # Clip to 0-100%
    hrr_runs['hrr_utilization'] = hrr_runs['hrr_utilization'].clip(0, 100)
    
    return {
        'dates': hrr_runs['date'].values,
        'hrr_utilization': hrr_runs['hrr_utilization'].values,
        'mean_utilization': np.mean(hrr_runs['hrr_utilization']),
        'max_utilization': np.max(hrr_runs['hrr_utilization'])
    }

def analyze_cadence_patterns(df):
    """
    Analyze cadence patterns and correlation with pace and terrain.
    Optimal cadence typically 170-180 spm for injury prevention.
    """
    cadence_runs = df[(df['cadence'] > 0) & (df['pace_seconds'] > 0)].copy()
    
    if len(cadence_runs) == 0:
        return None
    
    # Analyze cadence by pace groups
    cadence_runs['pace_group'] = pd.cut(cadence_runs['pace_minutes'], 
                                        bins=[0, 9, 10, 11, 12, 13, 100],
                                        labels=['<9 min/mi', '9-10', '10-11', '11-12', '12-13', '>13'])
    
    cadence_by_pace = cadence_runs.groupby('pace_group')['cadence'].mean()
    
    # Analyze cadence by elevation
    cadence_runs['elev_group'] = pd.cut(cadence_runs['elev_gain_per_mile'],
                                        bins=[0, 50, 100, 150, 1000],
                                        labels=['Flat (0-50)', 'Rolling (50-100)', 
                                               'Hilly (100-150)', 'Very hilly (>150)'])
    
    cadence_by_elev = cadence_runs.groupby('elev_group')['cadence'].mean()
    
    # Correlation between cadence and pace
    correlation = np.corrcoef(cadence_runs['cadence'], cadence_runs['pace_minutes'])[0, 1]
    
    return {
        'dates': cadence_runs['date'].values,
        'cadence': cadence_runs['cadence'].values,
        'pace': cadence_runs['pace_minutes'].values,
        'mean_cadence': np.mean(cadence_runs['cadence']),
        'std_cadence': np.std(cadence_runs['cadence']),
        'min_cadence': np.min(cadence_runs['cadence']),
        'max_cadence': np.max(cadence_runs['cadence']),
        'cadence_by_pace': cadence_by_pace,
        'cadence_by_elev': cadence_by_elev,
        'pace_correlation': correlation
    }

def analyze_hr_recovery_rate(df):
    """
    Analyze HR recovery by comparing resting HR to average HR.
    Better fitness = lower resting HR and better recovery.
    """
    recovery_runs = df[(df['resting_hr'] > 0) & (df['avghr'] > 0)].copy()
    
    if len(recovery_runs) == 0:
        return None
    
    recovery_runs['hr_elevation'] = recovery_runs['avghr'] - recovery_runs['resting_hr']
    
    return {
        'dates': recovery_runs['date'].values,
        'resting_hr': recovery_runs['resting_hr'].values,
        'avg_hr': recovery_runs['avghr'].values,
        'hr_elevation': recovery_runs['hr_elevation'].values,
        'mean_resting': np.mean(recovery_runs['resting_hr']),
        'mean_elevation': np.mean(recovery_runs['hr_elevation'])
    }

def create_visualizations(max_hr, zone_info, hr_runs, efficiency_data, 
                         drift_data, hrr_data, cadence_data, recovery_data):
    """Create comprehensive visualization charts."""
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.35)
    
    # 1. HR Zone distribution (pie)
    ax1 = fig.add_subplot(gs[0, 0])
    if zone_info:
        labels = list(zone_info.keys())
        sizes = [zone_info[z]['count'] for z in labels]
        colors = ['#90EE90', '#4CAF50', '#FFA500', '#FF6347', '#8B0000']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               shadow=True, startangle=90, textprops={'fontsize': 9, 'fontweight': 'bold'})
        ax1.set_title(f'HR Zone Distribution by Runs (Max HR: {max_hr:.0f})', fontweight='bold', fontsize=11)
    
    # 2. HR Zone time (bar)
    ax2 = fig.add_subplot(gs[0, 1])
    if zone_info:
        zones = list(zone_info.keys())
        hours = [zone_info[z]['hours'] for z in zones]
        colors_bar = ['#90EE90', '#4CAF50', '#FFA500', '#FF6347', '#8B0000']
        
        bars = ax2.bar(range(len(zones)), hours, color=colors_bar, edgecolor='black', linewidth=0.5)
        ax2.set_xticks(range(len(zones)))
        ax2.set_xticklabels(zones, rotation=15, ha='right', fontsize=9)
        ax2.set_ylabel('Time (hours)', fontweight='bold')
        ax2.set_title('Time Spent in Each HR Zone', fontweight='bold', fontsize=11)
        ax2.grid(alpha=0.3, axis='y')
        
        for bar, h in zip(bars, hours):
            if h > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, h, f'{h:.1f}h',
                        ha='center', va='bottom', fontsize=8)
    
    # 3. HR % vs Pace scatter
    ax3 = fig.add_subplot(gs[0, 2])
    if hr_runs is not None:
        scatter = ax3.scatter(hr_runs['pace_minutes'], hr_runs['hr_percentage'],
                             c=hr_runs['distance'], cmap='plasma', s=50, alpha=0.6,
                             edgecolors='black', linewidth=0.5)
        ax3.set_xlabel('Pace (min/mile)', fontweight='bold')
        ax3.set_ylabel('HR % of Max', fontweight='bold')
        ax3.set_title('Heart Rate % vs Pace', fontweight='bold', fontsize=11)
        ax3.grid(alpha=0.3)
        
        # Add zone boundaries
        ax3.axhline(y=60, color='green', linestyle='--', alpha=0.3, linewidth=1)
        ax3.axhline(y=70, color='orange', linestyle='--', alpha=0.3, linewidth=1)
        ax3.axhline(y=80, color='red', linestyle='--', alpha=0.3, linewidth=1)
        ax3.axhline(y=90, color='darkred', linestyle='--', alpha=0.3, linewidth=1)
        
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Distance (mi)', fontweight='bold', fontsize=9)
    
    # 4. Aerobic efficiency over time
    ax4 = fig.add_subplot(gs[1, :])
    if efficiency_data:
        ax4.scatter(efficiency_data['dates'], efficiency_data['efficiency_scores'],
                   alpha=0.6, s=40, color='steelblue', edgecolors='black', linewidth=0.5)
        
        # Trend line
        x_numeric = (pd.to_datetime(efficiency_data['dates']) - pd.to_datetime(efficiency_data['dates']).min()).days
        z = np.polyfit(x_numeric, efficiency_data['efficiency_scores'], 1)
        p = np.poly1d(z)
        trend_dates = pd.to_datetime(efficiency_data['dates']).min() + pd.to_timedelta(np.linspace(0, x_numeric.max(), 100), unit='D')
        ax4.plot(trend_dates, p(np.linspace(0, x_numeric.max(), 100)), 
                'r--', linewidth=2, alpha=0.7, label=f'Trend (slope={efficiency_data["trend_coef"]:.2f}/day)')
        
        ax4.set_xlabel('Date', fontweight='bold')
        ax4.set_ylabel('Aerobic Efficiency Score', fontweight='bold')
        ax4.set_title('Aerobic Efficiency Over Time (Higher = Better)', fontweight='bold', fontsize=12)
        ax4.grid(alpha=0.3)
        ax4.legend()
    
    # 5. Cardiac drift on long runs
    ax5 = fig.add_subplot(gs[2, 0])
    if drift_data:
        scatter = ax5.scatter(drift_data['durations'], drift_data['drift_indices'],
                             s=drift_data['distances']*10, alpha=0.6, 
                             c=drift_data['distances'], cmap='autumn',
                             edgecolors='black', linewidth=0.5)
        ax5.set_xlabel('Duration (hours)', fontweight='bold')
        ax5.set_ylabel('Cardiac Drift Index (%)', fontweight='bold')
        ax5.set_title('Cardiac Drift on Long Runs (>1 hr)', fontweight='bold', fontsize=11)
        ax5.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Distance (mi)', fontweight='bold', fontsize=9)
    
    # 6. HRR utilization over time
    ax6 = fig.add_subplot(gs[2, 1])
    if hrr_data:
        ax6.scatter(hrr_data['dates'], hrr_data['hrr_utilization'], 
                   alpha=0.6, s=40, color='crimson', edgecolors='black', linewidth=0.5)
        ax6.axhline(y=hrr_data['mean_utilization'], color='blue', linestyle='--', 
                   alpha=0.5, label=f"Mean: {hrr_data['mean_utilization']:.1f}%")
        ax6.set_xlabel('Date', fontweight='bold')
        ax6.set_ylabel('HRR Utilization (%)', fontweight='bold')
        ax6.set_title('Heart Rate Reserve Utilization', fontweight='bold', fontsize=11)
        ax6.grid(alpha=0.3)
        ax6.legend()
    
    # 7. Cadence distribution
    ax7 = fig.add_subplot(gs[2, 2])
    if cadence_data:
        ax7.hist(cadence_data['cadence'], bins=20, color='forestgreen', 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax7.axvline(cadence_data['mean_cadence'], color='red', linestyle='--', 
                   linewidth=2, label=f"Mean: {cadence_data['mean_cadence']:.0f}")
        ax7.axvline(180, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Optimal: 180')
        ax7.set_xlabel('Cadence (steps/min)', fontweight='bold')
        ax7.set_ylabel('Frequency', fontweight='bold')
        ax7.set_title('Cadence Distribution', fontweight='bold', fontsize=11)
        ax7.legend()
        ax7.grid(alpha=0.3, axis='y')
    
    # 8. Cadence vs pace
    ax8 = fig.add_subplot(gs[3, 0])
    if cadence_data:
        ax8.scatter(cadence_data['pace'], cadence_data['cadence'], 
                   alpha=0.6, s=50, color='purple', edgecolors='black', linewidth=0.5)
        
        # Trend line
        z = np.polyfit(cadence_data['pace'], cadence_data['cadence'], 1)
        p = np.poly1d(z)
        pace_range = np.linspace(cadence_data['pace'].min(), cadence_data['pace'].max(), 100)
        ax8.plot(pace_range, p(pace_range), 'r--', linewidth=2, alpha=0.7,
                label=f'Corr: {cadence_data["pace_correlation"]:.2f}')
        
        ax8.set_xlabel('Pace (min/mile)', fontweight='bold')
        ax8.set_ylabel('Cadence (steps/min)', fontweight='bold')
        ax8.set_title('Cadence vs Pace Relationship', fontweight='bold', fontsize=11)
        ax8.grid(alpha=0.3)
        ax8.legend()
    
    # 9. Cadence by pace group
    ax9 = fig.add_subplot(gs[3, 1])
    if cadence_data and not cadence_data['cadence_by_pace'].empty:
        pace_groups = cadence_data['cadence_by_pace'].index.astype(str)
        cadence_vals = cadence_data['cadence_by_pace'].values
        
        bars = ax9.bar(range(len(pace_groups)), cadence_vals, 
                      color='teal', alpha=0.7, edgecolor='black', linewidth=0.5)
        ax9.set_xticks(range(len(pace_groups)))
        ax9.set_xticklabels(pace_groups, rotation=30, ha='right', fontsize=9)
        ax9.set_ylabel('Avg Cadence (spm)', fontweight='bold')
        ax9.set_title('Cadence by Pace Range', fontweight='bold', fontsize=11)
        ax9.grid(alpha=0.3, axis='y')
        
        for bar, val in zip(bars, cadence_vals):
            if not np.isnan(val):
                ax9.text(bar.get_x() + bar.get_width()/2, val, f'{val:.0f}',
                        ha='center', va='bottom', fontsize=8)
    
    # 10. Resting HR vs Average HR
    ax10 = fig.add_subplot(gs[3, 2])
    if recovery_data:
        ax10.scatter(recovery_data['dates'], recovery_data['resting_hr'],
                    label='Resting HR', alpha=0.6, s=30, color='blue')
        ax10.scatter(recovery_data['dates'], recovery_data['avg_hr'],
                    label='Avg Run HR', alpha=0.6, s=30, color='red')
        ax10.set_xlabel('Date', fontweight='bold')
        ax10.set_ylabel('Heart Rate (bpm)', fontweight='bold')
        ax10.set_title('Resting vs Running Heart Rate', fontweight='bold', fontsize=11)
        ax10.grid(alpha=0.3)
        ax10.legend()
    
    plt.savefig('physiological_insights.png', dpi=300, bbox_inches='tight')
    print("[OK] Chart saved as 'physiological_insights.png'")
    plt.close()

def print_report(max_hr, zone_info, efficiency_data, drift_data, hrr_data, cadence_data):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print(" PHYSIOLOGICAL INSIGHTS ANALYSIS REPORT ".center(80, "="))
    print("="*80)
    
    print(f"\n💓 HEART RATE ANALYSIS (Estimated Max HR: {max_hr:.0f} bpm)")
    print("-" * 80)
    
    if zone_info:
        print("\nZone Distribution:")
        for zone, data in zone_info.items():
            print(f"  {zone:15} {data['count']:3} runs ({data['pct_runs']:5.1f}%) - {data['hours']:5.1f} hours")
        
        # Training polarization assessment
        z1_z2_pct = zone_info['Z1 Recovery']['pct_runs'] + zone_info['Z2 Aerobic']['pct_runs']
        z4_z5_pct = zone_info['Z4 Threshold']['pct_runs'] + zone_info['Z5 VO2max']['pct_runs']
        
        print("\nTraining Polarization:")
        print(f"  Low intensity (Z1-Z2): {z1_z2_pct:.1f}%")
        print(f"  High intensity (Z4-Z5): {z4_z5_pct:.1f}%")
        
        if z1_z2_pct >= 75:
            print("  Status: ✓ Excellent base-building approach")
        elif z1_z2_pct >= 60:
            print("  Status: → Good aerobic foundation")
        else:
            print("  Status: ⚠ Consider more low-intensity volume")
    
    print("\n\n🏃 AEROBIC EFFICIENCY")
    print("-" * 80)
    if efficiency_data:
        print(f"Mean Efficiency Score: {efficiency_data['mean_efficiency']:.1f}")
        print(f"Recent Efficiency (last 5): {efficiency_data['recent_efficiency']:.1f}")
        print(f"Trend: {efficiency_data['trend_coef']:.2f} points/day")
        
        if efficiency_data['trend_coef'] > 1:
            print("Status: ✓ Improving aerobic fitness - running faster at lower HR")
        elif efficiency_data['trend_coef'] > -1:
            print("Status: → Stable aerobic fitness")
        else:
            print("Status: ↓ Decreasing efficiency - check recovery and training stress")
        
        improvement = ((efficiency_data['recent_efficiency'] - efficiency_data['mean_efficiency']) / 
                      efficiency_data['mean_efficiency']) * 100
        print(f"Recent vs Overall: {improvement:+.1f}%")
    
    print("\n\n🌡️ CARDIAC DRIFT (Long Runs >1 hour)")
    print("-" * 80)
    if drift_data:
        print(f"Mean Drift Index: {drift_data['mean_drift']:.1f}%")
        print(f"Max Drift Index: {drift_data['max_drift']:.1f}%")
        print(f"Long runs analyzed: {len(drift_data['drift_indices'])}")
        
        if drift_data['mean_drift'] < 5:
            print("Status: ✓ Excellent cardiac stability - well-conditioned")
        elif drift_data['mean_drift'] < 10:
            print("Status: → Good stability for most runs")
        else:
            print("Status: ⚠ High drift - focus on hydration, pacing, heat adaptation")
    else:
        print("No long runs (>1 hour) with HR data available")
    
    print("\n\n💪 HEART RATE RESERVE UTILIZATION")
    print("-" * 80)
    if hrr_data:
        print(f"Mean HRR Utilization: {hrr_data['mean_utilization']:.1f}%")
        print(f"Max HRR Utilization: {hrr_data['max_utilization']:.1f}%")
        
        if hrr_data['mean_utilization'] < 60:
            print("Status: → Most runs are easy/moderate intensity")
        elif hrr_data['mean_utilization'] < 75:
            print("Status: → Balanced intensity distribution")
        else:
            print("Status: ⚠ High average intensity - ensure recovery runs are truly easy")
    
    print("\n\n🦵 CADENCE ANALYSIS")
    print("-" * 80)
    if cadence_data:
        print(f"Mean Cadence: {cadence_data['mean_cadence']:.0f} steps/min")
        print(f"Range: {cadence_data['min_cadence']:.0f} - {cadence_data['max_cadence']:.0f} steps/min")
        print(f"Std Dev: {cadence_data['std_cadence']:.1f} (consistency)")
        print(f"Correlation with pace: {cadence_data['pace_correlation']:.2f}")
        
        if 170 <= cadence_data['mean_cadence'] <= 180:
            print("Status: ✓ Optimal cadence range for injury prevention")
        elif 165 <= cadence_data['mean_cadence'] < 170:
            print("Status: → Slightly low - consider increasing cadence 5-10%")
        elif cadence_data['mean_cadence'] < 165:
            print("Status: ⚠ Low cadence - aim for 170+ to reduce impact forces")
        else:
            print("Status: → High cadence - ensure you're not overstriding")
        
        print("\nCadence by Pace:")
        for pace_group, cadence in cadence_data['cadence_by_pace'].items():
            if not np.isnan(cadence):
                print(f"  {pace_group}: {cadence:.0f} spm")
        
        if not cadence_data['cadence_by_elev'].empty:
            print("\nCadence by Terrain:")
            for elev_group, cadence in cadence_data['cadence_by_elev'].items():
                if not np.isnan(cadence):
                    print(f"  {elev_group}: {cadence:.0f} spm")
    
    print("\n\n💡 KEY PHYSIOLOGICAL INSIGHTS")
    print("-" * 80)
    
    if zone_info:
        z2_hours = zone_info['Z2 Aerobic']['hours']
        if z2_hours < 5:
            print("  • Build aerobic base: aim for more Zone 2 volume (conversational pace)")
    
    if efficiency_data and efficiency_data['trend_coef'] > 1:
        print("  ✓ Your aerobic system is adapting well - keep current training approach")
    
    if cadence_data and cadence_data['mean_cadence'] < 170:
        print("  • Work on cadence drills: aim for 170-180 spm to reduce injury risk")
        print("    - Quick feet drills, metronome runs, downhill strides")
    
    if drift_data and drift_data['mean_drift'] > 10:
        print("  • High cardiac drift on long runs:")
        print("    - Focus on hydration (start early, every 15-20 min)")
        print("    - Slow your pace 10-15 sec/mile on long runs")
        print("    - Train in heat gradually to improve adaptation")
    
    if hrr_data and hrr_data['mean_utilization'] > 70:
        print("  ⚠ High intensity on average - 80% of runs should feel easy")
        print("    - Easy pace = can hold full conversation")
    
    print("\n" + "="*80 + "\n")

def main():
    print("Loading data...")
    df = load_and_prepare_data()
    
    print("Estimating max heart rate...")
    max_hr = estimate_max_hr(df)
    
    print("Calculating HR zones...")
    zone_result = calculate_hr_zones(df, max_hr)
    if zone_result:
        zone_info, hr_runs = zone_result
    else:
        zone_info, hr_runs = None, None
    
    print("Analyzing aerobic efficiency...")
    efficiency_data = calculate_aerobic_efficiency(df)
    
    print("Analyzing cardiac drift...")
    drift_data = analyze_cardiac_drift(df)
    
    print("Calculating HR reserve utilization...")
    hrr_data = calculate_hr_reserve_utilization(df, max_hr)
    
    print("Analyzing cadence patterns...")
    cadence_data = analyze_cadence_patterns(df)
    
    print("Analyzing HR recovery rate...")
    recovery_data = analyze_hr_recovery_rate(df)
    
    print("Creating visualizations...")
    create_visualizations(max_hr, zone_info, hr_runs, efficiency_data,
                         drift_data, hrr_data, cadence_data, recovery_data)
    
    print_report(max_hr, zone_info, efficiency_data, drift_data, hrr_data, cadence_data)

if __name__ == "__main__":
    main()
