# Running Analytics Suite - NumPy Analysis Scripts

This collection of 5 advanced NumPy-based analysis scripts provides comprehensive insights into your running performance using data from your Garmin device.

## Overview

The analytics suite analyzes your `runs_data.csv` file and generates both detailed console reports and visual charts (PNG files) for each analysis category.

## Scripts

### 1. **performance_trends.py** - Performance Evolution Analysis
Analyzes your running performance over time to identify trends and improvements.

**Key Metrics:**
- Personal records (PRs) by distance bracket (5K, 10K, Half Marathon, Marathon)
- Pace progression and consistency (coefficient of variation)
- Weekly and monthly volume trends with moving averages
- Training streak analysis
- Speed distribution (tempo vs easy runs)

**Output:**
- Console report with PRs, consistency metrics, volume analysis
- `performance_trends.png` with 5 visualizations

**Run:**
```bash
python3 performance_trends.py
```

---

### 2. **training_load_recovery.py** - Training Load & Recovery Patterns
Evaluates training stress and recovery to prevent overtraining and optimize performance.

**Key Metrics:**
- **ACWR** (Acute:Chronic Workload Ratio) - 7-day vs 28-day rolling averages
- **Training monotony** - Variation in training load (high = injury risk)
- **Resting heart rate trends** - Cardiovascular fitness indicator
- **Hard/Easy ratio** - Balance of high vs low intensity runs
- **Cumulative fatigue index** - Total training stress with recovery modeling
- **Overtraining signals** - Automated alerts for injury risk

**Output:**
- Console report with ACWR, monotony, recovery recommendations
- `training_load_recovery.png` with 7 visualizations

**Run:**
```bash
python3 training_load_recovery.py
```

**ACWR Zones:**
- < 0.8: Undertraining
- 0.8-1.3: Optimal (sweet spot)
- > 1.5: High injury risk

---

### 3. **physiological_insights.py** - Cardiovascular & Efficiency Metrics
Deep dive into heart rate data and running biomechanics.

**Key Metrics:**
- **HR zone distribution** (Z1-Z5) - Time spent in each training zone
- **Aerobic efficiency** - Pace/HR ratio trends (improving fitness = lower HR at same pace)
- **Cardiac drift** - HR increase during long runs (hydration/heat stress indicator)
- **Heart rate reserve utilization** - Effort level analysis
- **Cadence analysis** - Optimal turnover and terrain adaptation
- **HR recovery rate** - Resting HR vs running HR trends

**Output:**
- Console report with HR zones, efficiency scores, cadence analysis
- `physiological_insights.png` with 10 visualizations

**Run:**
```bash
python3 physiological_insights.py
```

**Training Zones:**
- Z1 (50-60%): Recovery
- Z2 (60-70%): Aerobic base
- Z3 (70-80%): Tempo
- Z4 (80-90%): Threshold
- Z5 (90-100%): VO2max

---

### 4. **race_prediction.py** - Race Capability & Benchmarking
Predicts race times and estimates VO2max using Jack Daniels' Running Formula.

**Key Metrics:**
- **VO2max estimation** - Cardiovascular fitness level
- **VDOT score** - Training benchmark (simplification of VO2max)
- **Race time predictions** - 5K, 10K, Half Marathon, Marathon
- **Recommended training paces** - Easy, Tempo, Threshold, Interval zones
- **Competition readiness score** - 0-100 scale based on volume, long runs, consistency
- **VDOT progression** - Fitness trends over time

**Output:**
- Console report with VDOT, race predictions, training paces
- `race_prediction.png` with 5 visualizations

**Run:**
```bash
python3 race_prediction.py
```

**VDOT Scale:**
- 70+: Elite / National class
- 60-69: Very competitive / Sub-elite
- 50-59: Competitive runner
- 40-49: Above average recreational
- 30-39: Average recreational
- <30: Beginner / Developing fitness

---

### 5. **environmental_impact.py** - Terrain & Distance Analysis
Analyzes how elevation, terrain, and distance affect performance.

**Key Metrics:**
- **Elevation impact** - Correlation between hills and pace
- **Grade-adjusted pace (GAP)** - Normalized pace for fair comparison
- **Terrain preferences** - Strengths on flat vs hilly routes
- **Optimal race distance** - Best performance by distance category
- **Cadence vs terrain** - Step rate adaptation to elevation
- **Marathon readiness** - Long run preparation score (0-100)

**Output:**
- Console report with terrain analysis, distance sweet spot, marathon readiness
- `environmental_impact.png` with 7 visualizations

**Run:**
```bash
python3 environmental_impact.py
```

---

## Requirements

All scripts require:
- Python 3.7+
- numpy
- pandas
- matplotlib
- scipy (only for environmental_impact.py)

Install dependencies:
```bash
pip install numpy pandas matplotlib scipy
```

## Data Requirements

All scripts read from `runs_data.csv` which should contain:
- `date` - Run date
- `distance` - Distance in miles
- `duration` - Duration in HH:MM:SS format
- `avg_pace` - Average pace in MM:SS format
- `elev_gain` - Elevation gain in feet
- `elev_gain_per_mile` - Elevation gain per mile
- `cadence` - Steps per minute
- `minhr`, `maxhr`, `avghr` - Heart rate data
- `resting_hr` - Resting heart rate
- `activity_type` - Run or Treadmill run

## Key Insights You'll Learn

### About Your Fitness
1. **Current VO2max and VDOT** - How fit you are compared to other runners
2. **Aerobic efficiency trends** - Whether training is improving your engine
3. **Heart rate zones** - Are you building an aerobic base or overdoing intensity?

### About Your Training
4. **ACWR score** - Are you at injury risk from rapid volume increases?
5. **Training monotony** - Do you need more variety in your workouts?
6. **Hard/Easy balance** - Is your 80/20 rule actually working?

### About Your Racing
7. **Race time predictions** - What can you run right now at various distances?
8. **Optimal race distance** - What distance are you naturally best at?
9. **Marathon readiness** - Are you prepared for your goal race?

### About Your Biomechanics
10. **Cadence patterns** - Are you at optimal turnover (170-180 spm)?
11. **Cardiac drift** - How well do you handle long efforts and heat?
12. **Terrain strengths** - Are you a flat speed runner or a hill climber?

## Usage Recommendations

1. **Run all 5 scripts** for a complete athlete profile
2. **Re-run weekly** to track progress and trends
3. **Pay attention to warnings** (⚠) in the reports - these indicate areas needing work
4. **Use recommendations** at the end of each report for training adjustments

## Sample Workflow

```bash
# Generate all analyses
python3 performance_trends.py
python3 training_load_recovery.py
python3 physiological_insights.py
python3 race_prediction.py
python3 environmental_impact.py

# Review all PNG files for visual insights
# Read console reports for detailed metrics
# Adjust training based on recommendations
```

## Interpreting Results

### Good Signs (✓)
- Improving VDOT trend
- ACWR in 0.8-1.3 range
- 70%+ easy runs (Z1-Z2)
- Decreasing resting HR
- Cadence 170-180 spm
- Low training monotony

### Warning Signs (⚠)
- ACWR > 1.5 (injury risk)
- <60% easy runs (too much intensity)
- Increasing resting HR (fatigue)
- High cardiac drift (dehydration/heat issues)
- Cadence < 165 spm (injury risk)
- High training monotony (burnout risk)

## Advanced Tips

1. **Track VDOT over time** - Best single metric for overall fitness
2. **Monitor ACWR before races** - Should be in sweet spot (0.8-1.3)
3. **Build aerobic base** - Aim for 70%+ of time in Z2
4. **Use GAP for hilly runs** - More accurate effort assessment
5. **Check competition readiness** before signing up for races
6. **Maintain 170+ cadence** - Reduces injury risk

## Troubleshooting

**Script fails:**
- Ensure `runs_data.csv` is in the same directory
- Check that required packages are installed
- Verify CSV has proper column names

**Strange results:**
- Check for data quality issues in CSV
- Ensure dates are in correct format
- Verify heart rate data is present for HR analyses

**No visualizations:**
- Make sure matplotlib is installed
- Check for file permissions in directory

## Output Files

Each script generates:
1. **Console report** - Detailed text analysis with metrics
2. **PNG visualization** - Multi-panel chart with graphs

Generated files:
- `performance_trends.png`
- `training_load_recovery.png`
- `physiological_insights.png`
- `race_prediction.png`
- `environmental_impact.png`

---

## Credits

Analysis methods based on:
- Jack Daniels' Running Formula (VDOT, VO2max)
- Tim Gabbett's ACWR research (injury prevention)
- Stephen Seiler's polarized training research (80/20 rule)
- Sports science literature on cardiac drift, cadence, and running economy

Built with NumPy for efficient numerical computations and Matplotlib for professional visualizations.
