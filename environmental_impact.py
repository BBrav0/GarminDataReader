#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environmental Impact Analysis
Analyzes how elevation, terrain, and distance affect running performance.
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
    
    df['pace_seconds'] = df['avg_pace'].apply(parse_time_to_seconds)
    df['pace_minutes'] = df['pace_seconds'] / 60
    df['duration_seconds'] = df['duration'].apply(parse_time_to_seconds)
    
    return df

def analyze_elevation_impact(df):
    """
    Analyze how elevation gain affects pace.
    Calculate grade-adjusted pace (GAP).
    """
    outdoor_runs = df[(df['activity_type'] == 'Run') & (df['pace_seconds'] > 0) & 
                      (df['elev_gain_per_mile'] >= 0)].copy()
    
    if len(outdoor_runs) == 0:
        return None
    
    # Categorize by elevation
    outdoor_runs['terrain_type'] = pd.cut(outdoor_runs['elev_gain_per_mile'],
                                          bins=[0, 30, 75, 150, 1000],
                                          labels=['Flat (0-30)', 'Rolling (30-75)', 
                                                 'Hilly (75-150)', 'Very Hilly (>150)'])
    
    # Calculate correlation between elevation and pace
    valid_data = outdoor_runs[(outdoor_runs['elev_gain_per_mile'] > 0) & 
                              (outdoor_runs['pace_seconds'] > 0)]
    
    if len(valid_data) >= 5:
        correlation, p_value = stats.pearsonr(valid_data['elev_gain_per_mile'], 
                                              valid_data['pace_minutes'])
    else:
        correlation, p_value = 0, 1
    
    # Pace by terrain type
    pace_by_terrain = outdoor_runs.groupby('terrain_type')['pace_minutes'].agg(['mean', 'std', 'count'])
    
    # Calculate grade adjustment factor (approximate)
    # Rule of thumb: +12-15 sec/mile per 100ft elevation gain per mile
    outdoor_runs['pace_adjustment'] = outdoor_runs['elev_gain_per_mile'] * 0.12 / 100  # minutes
    outdoor_runs['gap'] = outdoor_runs['pace_minutes'] - outdoor_runs['pace_adjustment']
    
    return {
        'terrain_df': outdoor_runs,
        'pace_by_terrain': pace_by_terrain,
        'correlation': correlation,
        'p_value': p_value,
        'mean_flat_pace': pace_by_terrain.loc['Flat (0-30)', 'mean'] if 'Flat (0-30)' in pace_by_terrain.index else np.nan,
        'mean_hilly_pace': pace_by_terrain.loc['Hilly (75-150)', 'mean'] if 'Hilly (75-150)' in pace_by_terrain.index else np.nan
    }

def analyze_terrain_preferences(df):
    """
    Analyze performance on different terrain types.
    Identify strength areas (flat vs climbing).
    """
    outdoor_runs = df[(df['activity_type'] == 'Run') & (df['pace_seconds'] > 0)].copy()
    
    if len(outdoor_runs) == 0:
        return None
    
    # Calculate pace percentile for each run
    outdoor_runs['pace_percentile'] = outdoor_runs['pace_seconds'].rank(pct=True)
    
    # Split into flat vs hilly
    flat_runs = outdoor_runs[outdoor_runs['elev_gain_per_mile'] < 50]
    hilly_runs = outdoor_runs[outdoor_runs['elev_gain_per_mile'] >= 75]
    
    if len(flat_runs) >= 3 and len(hilly_runs) >= 3:
        flat_performance = flat_runs['pace_percentile'].mean()
        hilly_performance = hilly_runs['pace_percentile'].mean()
        
        # Lower percentile = faster (better performance)
        flat_strength = 1 - flat_performance
        hilly_strength = 1 - hilly_performance
        
        if flat_strength > hilly_strength + 0.15:
            terrain_preference = "Flat terrain specialist - work on hill climbing"
        elif hilly_strength > flat_strength + 0.15:
            terrain_preference = "Strong climber - leverage hilly courses"
        else:
            terrain_preference = "Well-rounded - performs equally on varied terrain"
    else:
        flat_performance = np.nan
        hilly_performance = np.nan
        terrain_preference = "Insufficient data for terrain comparison"
    
    return {
        'flat_performance': flat_performance,
        'hilly_performance': hilly_performance,
        'flat_runs': len(flat_runs),
        'hilly_runs': len(hilly_runs),
        'terrain_preference': terrain_preference
    }

def analyze_distance_sweet_spot(df):
    """
    Identify optimal race distance based on performance.
    Compare pace across different distance ranges.
    """
    outdoor_runs = df[(df['activity_type'] == 'Run') & (df['pace_seconds'] > 0) & 
                      (df['distance'] >= 2)].copy()
    
    if len(outdoor_runs) == 0:
        return None
    
    # Categorize by distance
    outdoor_runs['distance_category'] = pd.cut(outdoor_runs['distance'],
                                               bins=[0, 4, 7, 14, 100],
                                               labels=['Short (2-4mi)', 'Medium (4-7mi)', 
                                                      'Long (7-14mi)', 'Very Long (>14mi)'])
    
    # Calculate relative performance (z-score of pace within each category)
    distance_stats = []
    
    for category in outdoor_runs['distance_category'].unique():
        if pd.isna(category):
            continue
        
        cat_runs = outdoor_runs[outdoor_runs['distance_category'] == category]
        if len(cat_runs) >= 3:
            avg_pace = cat_runs['pace_minutes'].mean()
            best_pace = cat_runs['pace_minutes'].min()
            count = len(cat_runs)
            
            # Performance score: how much faster than average pace?
            performance_score = (outdoor_runs['pace_minutes'].mean() - avg_pace) / outdoor_runs['pace_minutes'].std()
            
            distance_stats.append({
                'category': category,
                'avg_pace': avg_pace,
                'best_pace': best_pace,
                'count': count,
                'performance_score': performance_score
            })
    
    if distance_stats:
        best_distance = max(distance_stats, key=lambda x: x['performance_score'])
    else:
        best_distance = None
    
    return {
        'distance_stats': distance_stats,
        'best_distance': best_distance,
        'distance_df': outdoor_runs
    }

def analyze_cadence_terrain_relationship(df):
    """
    Analyze how cadence changes with terrain.
    Higher cadence on hills is typical for maintaining power.
    """
    cadence_runs = df[(df['cadence'] > 0) & (df['elev_gain_per_mile'] >= 0)].copy()
    
    if len(cadence_runs) == 0:
        return None
    
    # Split by terrain
    flat_cadence = cadence_runs[cadence_runs['elev_gain_per_mile'] < 50]['cadence'].mean()
    hilly_cadence = cadence_runs[cadence_runs['elev_gain_per_mile'] >= 75]['cadence'].mean()
    
    # Correlation
    valid_data = cadence_runs[(cadence_runs['elev_gain_per_mile'] > 0)]
    
    if len(valid_data) >= 5:
        correlation, p_value = stats.pearsonr(valid_data['elev_gain_per_mile'], 
                                              valid_data['cadence'])
    else:
        correlation, p_value = 0, 1
    
    return {
        'flat_cadence': flat_cadence,
        'hilly_cadence': hilly_cadence,
        'cadence_diff': hilly_cadence - flat_cadence if not np.isnan(hilly_cadence) and not np.isnan(flat_cadence) else 0,
        'correlation': correlation,
        'p_value': p_value
    }

def analyze_long_run_marathon_readiness(df):
    """
    Analyze marathon-pace long runs for race readiness.
    Compare long run performance and consistency.
    """
    long_runs = df[(df['distance'] >= 15) & (df['pace_seconds'] > 0) & 
                   (df['activity_type'] == 'Run')].copy()
    
    if len(long_runs) == 0:
        return None
    
    long_runs = long_runs.sort_values('date')
    
    # Calculate pace trend on long runs
    if len(long_runs) >= 3:
        x_numeric = (long_runs['date'] - long_runs['date'].min()).dt.days
        pace_trend = np.polyfit(x_numeric, long_runs['pace_minutes'], 1)[0]
    else:
        pace_trend = 0
    
    # Marathon readiness factors
    readiness_score = 0
    factors = {}
    
    # Factor 1: Number of long runs (15+ miles)
    if len(long_runs) >= 4:
        readiness_score += 30
        factors['long_run_volume'] = "✓ Sufficient long runs"
    elif len(long_runs) >= 2:
        readiness_score += 15
        factors['long_run_volume'] = "→ Moderate long run experience"
    else:
        factors['long_run_volume'] = "⚠ Need more long runs"
    
    # Factor 2: 20-miler completion
    if long_runs['distance'].max() >= 20:
        readiness_score += 25
        factors['peak_distance'] = "✓ Completed 20+ mile run"
    elif long_runs['distance'].max() >= 18:
        readiness_score += 15
        factors['peak_distance'] = "→ Close to 20-mile mark"
    else:
        factors['peak_distance'] = "⚠ Build to 20 miles"
    
    # Factor 3: Pace improvement on long runs
    if pace_trend < -0.01:
        readiness_score += 25
        factors['pace_trend'] = "✓ Getting faster on long runs"
    elif pace_trend < 0.01:
        readiness_score += 15
        factors['pace_trend'] = "→ Maintaining pace on long runs"
    else:
        factors['pace_trend'] = "⚠ Slowing on long runs - check fatigue"
    
    # Factor 4: Recent long run
    days_since_last = (df['date'].max() - long_runs['date'].max()).days
    if days_since_last <= 14:
        readiness_score += 20
        factors['recency'] = "✓ Recent long run completed"
    elif days_since_last <= 28:
        readiness_score += 10
        factors['recency'] = "→ Long run within past month"
    else:
        factors['recency'] = "⚠ No recent long runs"
    
    return {
        'long_runs': long_runs,
        'count': len(long_runs),
        'max_distance': long_runs['distance'].max(),
        'avg_pace': long_runs['pace_minutes'].mean(),
        'best_pace': long_runs['pace_minutes'].min(),
        'pace_trend': pace_trend,
        'readiness_score': readiness_score,
        'factors': factors
    }

def create_visualizations(elevation_data, terrain_pref, distance_data, 
                         cadence_terrain, marathon_ready):
    """Create comprehensive visualization charts."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    # 1. Elevation vs Pace scatter
    ax1 = fig.add_subplot(gs[0, :2])
    if elevation_data:
        terrain_df = elevation_data['terrain_df']
        scatter = ax1.scatter(terrain_df['elev_gain_per_mile'], terrain_df['pace_minutes'],
                             c=terrain_df['distance'], cmap='coolwarm', s=60, alpha=0.6,
                             edgecolors='black', linewidth=0.5)
        
        # Trend line
        valid = (terrain_df['elev_gain_per_mile'] > 0) & (terrain_df['pace_minutes'] > 0)
        if np.sum(valid) >= 5:
            z = np.polyfit(terrain_df.loc[valid, 'elev_gain_per_mile'], 
                          terrain_df.loc[valid, 'pace_minutes'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(terrain_df['elev_gain_per_mile'].min(), 
                                 terrain_df['elev_gain_per_mile'].max(), 100)
            ax1.plot(x_range, p(x_range), 'r--', linewidth=2.5, alpha=0.7,
                    label=f'Trend (r={elevation_data["correlation"]:.2f})')
        
        ax1.set_xlabel('Elevation Gain per Mile (ft)', fontweight='bold', fontsize=11)
        ax1.set_ylabel('Pace (min/mile)', fontweight='bold', fontsize=11)
        ax1.set_title('Impact of Elevation on Pace', fontweight='bold', fontsize=13)
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Distance (miles)', fontweight='bold')
    
    # 2. Pace by terrain type
    ax2 = fig.add_subplot(gs[0, 2])
    if elevation_data and not elevation_data['pace_by_terrain'].empty:
        terrain_types = elevation_data['pace_by_terrain'].index
        paces = elevation_data['pace_by_terrain']['mean'].values
        errors = elevation_data['pace_by_terrain']['std'].values
        
        colors = ['#90EE90', '#FFD700', '#FFA500', '#FF6347']
        bars = ax2.bar(range(len(terrain_types)), paces, yerr=errors, 
                      color=colors[:len(terrain_types)], edgecolor='black', 
                      linewidth=1, capsize=5, alpha=0.7)
        
        ax2.set_xticks(range(len(terrain_types)))
        ax2.set_xticklabels([str(t) for t in terrain_types], rotation=20, ha='right', fontsize=9)
        ax2.set_ylabel('Avg Pace (min/mile)', fontweight='bold', fontsize=10)
        ax2.set_title('Pace by Terrain Type', fontweight='bold', fontsize=11)
        ax2.grid(alpha=0.3, axis='y')
        
        for bar, pace in zip(bars, paces):
            if not np.isnan(pace):
                ax2.text(bar.get_x() + bar.get_width()/2, pace,
                        f'{pace:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 3. Grade Adjusted Pace (GAP) vs Actual Pace
    ax3 = fig.add_subplot(gs[1, 0])
    if elevation_data:
        terrain_df = elevation_data['terrain_df']
        hilly_runs = terrain_df[terrain_df['elev_gain_per_mile'] > 50]
        
        if len(hilly_runs) > 0:
            x = np.arange(len(hilly_runs))
            width = 0.35
            
            bars1 = ax3.bar(x - width/2, hilly_runs['pace_minutes'], width,
                           label='Actual Pace', color='coral', edgecolor='black', linewidth=0.5)
            bars2 = ax3.bar(x + width/2, hilly_runs['gap'], width,
                           label='Grade Adj Pace', color='steelblue', edgecolor='black', linewidth=0.5)
            
            ax3.set_xlabel('Run Index', fontweight='bold', fontsize=10)
            ax3.set_ylabel('Pace (min/mile)', fontweight='bold', fontsize=10)
            ax3.set_title('Actual vs Grade-Adjusted Pace (Hilly Runs)', fontweight='bold', fontsize=11)
            ax3.legend()
            ax3.grid(alpha=0.3, axis='y')
    
    # 4. Performance by distance category
    ax4 = fig.add_subplot(gs[1, 1])
    if distance_data and distance_data['distance_stats']:
        stats_list = distance_data['distance_stats']
        categories = [s['category'] for s in stats_list]
        avg_paces = [s['avg_pace'] for s in stats_list]
        best_paces = [s['best_pace'] for s in stats_list]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax4.bar(x - width/2, avg_paces, width, label='Avg Pace',
                       color='lightcoral', edgecolor='black', linewidth=0.5)
        bars2 = ax4.bar(x + width/2, best_paces, width, label='Best Pace',
                       color='limegreen', edgecolor='black', linewidth=0.5)
        
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(c) for c in categories], rotation=20, ha='right', fontsize=9)
        ax4.set_ylabel('Pace (min/mile)', fontweight='bold', fontsize=10)
        ax4.set_title('Performance by Distance Range', fontweight='bold', fontsize=11)
        ax4.legend()
        ax4.grid(alpha=0.3, axis='y')
    
    # 5. Cadence vs Elevation
    ax5 = fig.add_subplot(gs[1, 2])
    if cadence_terrain:
        terrain_labels = ['Flat\n(<50 ft/mi)', 'Hilly\n(≥75 ft/mi)']
        cadences = [cadence_terrain['flat_cadence'], cadence_terrain['hilly_cadence']]
        colors = ['#90EE90', '#FF6347']
        
        bars = ax5.bar(terrain_labels, cadences, color=colors, 
                      edgecolor='black', linewidth=1, alpha=0.7)
        
        ax5.set_ylabel('Cadence (steps/min)', fontweight='bold', fontsize=10)
        ax5.set_title(f'Cadence by Terrain (Δ={cadence_terrain["cadence_diff"]:.1f})', 
                     fontweight='bold', fontsize=11)
        ax5.grid(alpha=0.3, axis='y')
        
        for bar, cad in zip(bars, cadences):
            if not np.isnan(cad):
                ax5.text(bar.get_x() + bar.get_width()/2, cad,
                        f'{cad:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 6. Long run progression
    ax6 = fig.add_subplot(gs[2, :2])
    if marathon_ready and len(marathon_ready['long_runs']) > 0:
        long_runs = marathon_ready['long_runs']
        
        # Twin axis for distance and pace
        ax6_twin = ax6.twinx()
        
        ax6.plot(long_runs['date'], long_runs['distance'], 'b-o', 
                linewidth=2, markersize=8, label='Distance', alpha=0.7)
        ax6_twin.plot(long_runs['date'], long_runs['pace_minutes'], 'r-s',
                     linewidth=2, markersize=7, label='Pace', alpha=0.7)
        
        ax6.set_xlabel('Date', fontweight='bold', fontsize=11)
        ax6.set_ylabel('Distance (miles)', fontweight='bold', fontsize=11, color='blue')
        ax6_twin.set_ylabel('Pace (min/mile)', fontweight='bold', fontsize=11, color='red')
        ax6.set_title('Long Run Progression (15+ miles)', fontweight='bold', fontsize=13)
        ax6.grid(alpha=0.3)
        ax6.tick_params(axis='y', labelcolor='blue')
        ax6_twin.tick_params(axis='y', labelcolor='red')
        
        # Combine legends
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 7. Marathon readiness factors
    ax7 = fig.add_subplot(gs[2, 2])
    if marathon_ready:
        factors = ['Long Run\nVolume', 'Peak\nDistance', 'Pace\nTrend', 'Recent\nLong Run']
        max_scores = [30, 25, 25, 20]
        
        # Extract scores (approximation based on factors)
        score_map = {
            'long_run_volume': 30 if '✓' in marathon_ready['factors'].get('long_run_volume', '') else 
                              15 if '→' in marathon_ready['factors'].get('long_run_volume', '') else 0,
            'peak_distance': 25 if '✓' in marathon_ready['factors'].get('peak_distance', '') else
                           15 if '→' in marathon_ready['factors'].get('peak_distance', '') else 0,
            'pace_trend': 25 if '✓' in marathon_ready['factors'].get('pace_trend', '') else
                        15 if '→' in marathon_ready['factors'].get('pace_trend', '') else 0,
            'recency': 20 if '✓' in marathon_ready['factors'].get('recency', '') else
                     10 if '→' in marathon_ready['factors'].get('recency', '') else 0
        }
        
        scores = list(score_map.values())
        
        x = np.arange(len(factors))
        width = 0.35
        
        bars1 = ax7.bar(x - width/2, max_scores, width, label='Maximum',
                       color='lightgray', edgecolor='black', linewidth=0.5)
        bars2 = ax7.bar(x + width/2, scores, width, label='Current',
                       color='forestgreen', edgecolor='black', linewidth=0.5)
        
        ax7.set_xticks(x)
        ax7.set_xticklabels(factors, fontsize=9)
        ax7.set_ylabel('Score', fontweight='bold', fontsize=10)
        ax7.set_title(f'Marathon Readiness: {marathon_ready["readiness_score"]}/100',
                     fontweight='bold', fontsize=11)
        ax7.legend()
        ax7.grid(alpha=0.3, axis='y')
    
    plt.savefig('environmental_impact.png', dpi=300, bbox_inches='tight')
    print("[OK] Chart saved as 'environmental_impact.png'")
    plt.close()

def print_report(elevation_data, terrain_pref, distance_data, cadence_terrain, marathon_ready):
    """Print comprehensive analysis report."""
    print("\n" + "="*80)
    print(" ENVIRONMENTAL IMPACT ANALYSIS ".center(80, "="))
    print("="*80)
    
    print("\n⛰️ ELEVATION IMPACT ON PERFORMANCE")
    print("-" * 80)
    if elevation_data:
        print(f"Correlation (elevation vs pace): r = {elevation_data['correlation']:.2f}")
        if abs(elevation_data['correlation']) < 0.3:
            print("  → Weak correlation - pace not strongly affected by hills")
        elif abs(elevation_data['correlation']) < 0.6:
            print("  → Moderate correlation - hills slow you down somewhat")
        else:
            print("  → Strong correlation - significant pace impact from elevation")
        
        print("\nPace by Terrain Type:")
        for terrain, row in elevation_data['pace_by_terrain'].iterrows():
            if row['count'] > 0:
                print(f"  {terrain}: {row['mean']:.2f} min/mi (±{row['std']:.2f}, n={int(row['count'])})")
        
        if not np.isnan(elevation_data['mean_flat_pace']) and not np.isnan(elevation_data['mean_hilly_pace']):
            diff = elevation_data['mean_hilly_pace'] - elevation_data['mean_flat_pace']
            print(f"\nPace difference (hilly vs flat): +{diff:.2f} min/mile")
    
    print("\n\n🏔️ TERRAIN PREFERENCES & STRENGTHS")
    print("-" * 80)
    if terrain_pref:
        print(f"Flat terrain runs: {terrain_pref['flat_runs']}")
        print(f"Hilly terrain runs: {terrain_pref['hilly_runs']}")
        print(f"\nAssessment: {terrain_pref['terrain_preference']}")
        
        if not np.isnan(terrain_pref['flat_performance']) and not np.isnan(terrain_pref['hilly_performance']):
            print(f"\nPerformance percentiles (lower = faster):")
            print(f"  Flat terrain: {terrain_pref['flat_performance']:.1%}")
            print(f"  Hilly terrain: {terrain_pref['hilly_performance']:.1%}")
    
    print("\n\n📏 OPTIMAL DISTANCE ANALYSIS")
    print("-" * 80)
    if distance_data and distance_data['distance_stats']:
        print("Performance by distance category:")
        for stat in distance_data['distance_stats']:
            print(f"\n{stat['category']}:")
            print(f"  Average pace: {stat['avg_pace']:.2f} min/mile")
            print(f"  Best pace: {stat['best_pace']:.2f} min/mile")
            print(f"  Runs: {stat['count']}")
            print(f"  Performance score: {stat['performance_score']:.2f}")
        
        if distance_data['best_distance']:
            best = distance_data['best_distance']
            print(f"\n✓ Strongest distance: {best['category']}")
            print(f"  Average pace: {best['avg_pace']:.2f} min/mile")
    
    print("\n\n🦵 CADENCE ADAPTATION TO TERRAIN")
    print("-" * 80)
    if cadence_terrain:
        print(f"Flat terrain cadence: {cadence_terrain['flat_cadence']:.0f} spm")
        print(f"Hilly terrain cadence: {cadence_terrain['hilly_cadence']:.0f} spm")
        print(f"Difference: {cadence_terrain['cadence_diff']:+.0f} spm")
        
        if cadence_terrain['cadence_diff'] > 3:
            print("Status: ✓ Good adaptation - increasing cadence on hills")
        elif cadence_terrain['cadence_diff'] > -3:
            print("Status: → Minimal change - consider increasing cadence on climbs")
        else:
            print("Status: ⚠ Cadence drops on hills - maintain quick turnover")
        
        print(f"Correlation: r = {cadence_terrain['correlation']:.2f}")
    
    print("\n\n🏃 MARATHON READINESS (Long Run Analysis)")
    print("-" * 80)
    if marathon_ready:
        print(f"Total long runs (15+ miles): {marathon_ready['count']}")
        print(f"Peak long run: {marathon_ready['max_distance']:.1f} miles")
        print(f"Average long run pace: {marathon_ready['avg_pace']:.2f} min/mile")
        print(f"Best long run pace: {marathon_ready['best_pace']:.2f} min/mile")
        print(f"Long run pace trend: {marathon_ready['pace_trend']:.4f} min/mi per day")
        
        print(f"\nReadiness Score: {marathon_ready['readiness_score']}/100")
        
        print("\nReadiness Factors:")
        for factor, status in marathon_ready['factors'].items():
            print(f"  • {status}")
        
        if marathon_ready['readiness_score'] >= 80:
            print("\n✓ Excellent marathon preparation - ready to race!")
        elif marathon_ready['readiness_score'] >= 60:
            print("\n→ Good foundation - continue building")
        else:
            print("\n⚠ More preparation needed - focus on long run volume")
    else:
        print("No long runs (15+ miles) in dataset")
    
    print("\n\n💡 ENVIRONMENTAL TRAINING RECOMMENDATIONS")
    print("-" * 80)
    
    if elevation_data and elevation_data['correlation'] > 0.5:
        print("  • Hills significantly impact your pace - train on varied terrain")
        print("  • Use grade-adjusted pace (GAP) for hill runs to gauge effort")
        print("  • Hill repeats can improve strength and economy")
    
    if terrain_pref and 'Flat terrain specialist' in terrain_pref.get('terrain_preference', ''):
        print("  • Incorporate more hill training to develop climbing strength")
        print("  • Start with gradual inclines, progress to steeper grades")
        print("  • Hill work builds power and running economy")
    
    if terrain_pref and 'Strong climber' in terrain_pref.get('terrain_preference', ''):
        print("  • Leverage your climbing strength in hilly races")
        print("  • Balance with flat speed work to maintain turnover")
    
    if cadence_terrain and cadence_terrain['cadence_diff'] < 0:
        print("  • Focus on maintaining cadence on hills (don't slow turnover)")
        print("  • Shorten stride on climbs but keep feet moving quickly")
    
    if distance_data and distance_data['best_distance']:
        best_cat = distance_data['best_distance']['category']
        print(f"  • Your sweet spot appears to be {best_cat}")
        print(f"  • Consider targeting races in this distance range")
    
    if marathon_ready and marathon_ready['readiness_score'] < 60:
        print("  • Build long run volume: progress to 18-20 miles")
        print("  • Long runs should be 25-30% of weekly mileage")
        print("  • Practice race nutrition on runs >90 minutes")
    
    print("\n" + "="*80 + "\n")

def main():
    print("Loading data...")
    df = load_and_prepare_data()
    
    print("Analyzing elevation impact...")
    elevation_data = analyze_elevation_impact(df)
    
    print("Analyzing terrain preferences...")
    terrain_pref = analyze_terrain_preferences(df)
    
    print("Analyzing distance sweet spot...")
    distance_data = analyze_distance_sweet_spot(df)
    
    print("Analyzing cadence-terrain relationship...")
    cadence_terrain = analyze_cadence_terrain_relationship(df)
    
    print("Analyzing marathon readiness...")
    marathon_ready = analyze_long_run_marathon_readiness(df)
    
    print("Creating visualizations...")
    create_visualizations(elevation_data, terrain_pref, distance_data, 
                         cadence_terrain, marathon_ready)
    
    print_report(elevation_data, terrain_pref, distance_data, cadence_terrain, marathon_ready)

if __name__ == "__main__":
    main()
