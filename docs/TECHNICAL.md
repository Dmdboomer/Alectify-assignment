# Technical Reference

## Problem Statement

Long-haul truck drivers operate on irregular schedules that shift over time
due to route changes, delivery deadlines, and personal habits. Fatigue-related
accidents are a leading cause of trucking incidents. By leveraging GPS
telemetry data collected over ~1 year, we aim to **predict when a driver will
sleep** in the upcoming days/weeks, enabling fleet managers to:

- Proactively schedule rest stops and relay drivers
- Flag high-fatigue-risk shifts before they happen
- Comply with Hours of Service (HOS) regulations more effectively

## Approach

We use a **multi-model comparison** that mirrors the TimeGPT workflow
(Nixtla's `unique_id | ds | y` + exogenous features format), running entirely
locally with no API keys required:

| Model | Type | Exogenous Features |
|-------|------|--------------------|
| **MLForecast + LightGBM** | TimeGPT-equivalent (local) | Yes — calendar + auto lag/rolling |
| **MLForecast + LinearRegression** | TimeGPT-equivalent (local) | Yes — calendar + auto lag/rolling |
| **TimesFM 200M** (Google) | Foundation model | No (univariate only) |
| **AutoARIMA** | Classical baseline | No |
| **SeasonalNaive** | Naive baseline | No |

**Key technique**: Sleep start hour uses **circular encoding** (sin/cos of
hour/24) to handle midnight wraparound. The model predicts sin and cos
components separately, then reconstructs the hour via atan2.

To swap in the real TimeGPT, replace the MLForecast calls with:
```python
from nixtla import NixtlaClient
nixtla = NixtlaClient(api_key="YOUR_KEY")
forecast = nixtla.forecast(df=train_df, X_df=future_exog, h=14, freq="D")
```
The DataFrame format is identical.

## Results

Evaluation uses **circular MAE** for sleep start hour (so 23:50 vs 00:10 =
20 min, not 23.7 hours).

### Sleep Start Hour (best: TimesFM, MAE = 0.79 h / ~47 min)

| Model | MAE | RMSE |
|-------|-----|------|
| **TimesFM** | **0.79 h (~47 min)** | 1.02 h |
| AutoARIMA | 0.83 h (~50 min) | 1.05 h |
| LinearRegression | 0.84 h (~50 min) | 1.08 h |
| LightGBM | 0.86 h (~51 min) | 1.10 h |
| SeasonalNaive | 1.06 h (~64 min) | 1.37 h |

### Sleep Duration (best: LinearRegression, MAE = 1.01 h / ~61 min)

| Model | MAE | RMSE |
|-------|-----|------|
| **LinearRegression** | **1.01 h (~61 min)** | 1.30 h |
| TimesFM | 1.04 h (~62 min) | 1.32 h |
| LightGBM | 1.05 h (~63 min) | 1.34 h |
| AutoARIMA | 1.05 h (~63 min) | 1.35 h |
| SeasonalNaive | 1.45 h (~87 min) | 1.85 h |

The target standard deviations on this dataset are roughly 1.3 h (duration)
and 7.3 h (raw start hour). A naive "predict the global mean" baseline would
land around those numbers, so the non-naive models are learning real
structure — especially on start hour where TimesFM cuts the SeasonalNaive
error by ~25 %.

## Data

### Source (Synthetic)

Run `python generate_data.py` to produce three CSV files in `data/`:

| File | Description | Approx. Rows |
|------|-------------|--------------|
| `driver_gps_events.csv` | Raw GPS pings + event labels, every ~30–60 min per driver | ~66k |
| `sleep_events.csv` | One row per sleep session (start, end, duration, location) | ~4,200 |
| `driver_profiles.csv` | Per-driver metadata (home city, shift type, sleep habits) | 12 |

### Schema: `driver_gps_events.csv`

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Timestamp of the GPS ping |
| `driver_id` | string | Unique driver identifier (e.g. `DRV-001`) |
| `latitude` | float | GPS latitude |
| `longitude` | float | GPS longitude |
| `speed_mph` | float | Vehicle speed at ping time |
| `engine_on` | bool | Whether the engine is running |
| `event` | string | One of: `driving`, `driving_break`, `arrive`, `stationary`, `sleep_start`, `sleep_end`, `wake_up` |
| `location_label` | string | Nearest city name (populated on stops, empty while driving) |
| `is_sleeping` | bool | Ground-truth sleep label |

### Schema: `sleep_events.csv`

| Column | Type | Description |
|--------|------|-------------|
| `driver_id` | string | Driver identifier |
| `date` | date | Calendar date of sleep start |
| `day_of_week` | string | e.g. `Monday` |
| `sleep_start` | datetime | When sleep began |
| `sleep_end` | datetime | When sleep ended |
| `duration_hours` | float | Total sleep duration |
| `latitude` | float | Sleep location latitude |
| `longitude` | float | Sleep location longitude |
| `location_label` | string | City name where driver slept |

### Schema: `driver_profiles.csv`

| Column | Type | Description |
|--------|------|-------------|
| `driver_id` | string | Driver identifier |
| `home_city` | string | Base city |
| `shift_type` | string | `day`, `night`, or `split` |
| `preferred_sleep_hour` | float | Baseline bedtime (0–24) |
| `sleep_duration_mean` | float | Average sleep length |
| `sleep_duration_std` | float | Per-night std of sleep length |
| `weekend_rest_prob` | float | Probability of taking a weekend day off |
| `drive_days_per_week` | int | Target active days per week |

### Data characteristics

- **12 drivers** split across three shift types (day / night / split).
- **~365 days** of data (April 2025 – April 2026).
- **Bedtimes span all 24 hours** — day drivers 20:30–23:30, night drivers
  7:30–10:30, split drivers 0:30–3:30 (post-midnight).
- **Weekly cycle** — longer/later sleep Thu/Fri/Sat.
- **Annual seasonality** — winter sleep up to ~1 h longer; summer wake earlier.
- **Habit drift** — bounded random walk on preferred bedtime across the year.
- **Sleep debt** — previous-night deficit partly repaid on the next night.
- **Fatigue** — rolling 3-day driving hours increase sleep duration.
- **Long-haul trips** every ~3 weeks (short pre-haul night, long recovery).
- **Vacation window** of 1–2 weeks per driver with later, longer sleep.
- **Disruption days** (~2 %) — weather/breakdowns produce unusual timings.
- **Late arrivals** push bedtime later for day-shift drivers.

Duration std on the produced dataset is ~1.3 h (vs ~0.4 h in the earlier
toy version), so naive baselines leave meaningful room for better models.

## Pipeline

See [PIPELINE.md](PIPELINE.md) for a detailed pipeline diagram.

### Quick overview

```
generate_data.py --> data/*.csv --> forecast.py --> data/cv_*.csv --> evaluate.py --> plots/ + metrics
```

1. **`generate_data.py`** — generates 1 year of synthetic GPS + sleep data
   for 12 truck drivers with mixed shift types and realistic dynamics.
2. **`forecast.py`** — runs all 5 models with circular encoding for sleep
   start + 4-window rolling cross-validation.
3. **`evaluate.py`** — computes MAE/RMSE per model (circular-aware for
   hours), per driver, and generates comparison plots.

## Getting Started

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
source venv/Scripts/activate      # Windows (Git Bash)

# 2. Install dependencies
pip install pandas scikit-learn matplotlib lightgbm mlforecast statsforecast
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/google-research/timesfm.git
pip install huggingface_hub safetensors einops

# 3. Generate synthetic data
python generate_data.py

# 4. Run forecasting pipeline
python forecast.py

# 5. Evaluate and generate plots
python evaluate.py
```

## Directory Structure

```
alectify/
  README.md                 # Top-level overview
  docs/
    SUMMARY.md              # Short write-up
    TECHNICAL.md            # This file
    PIPELINE.md             # Pipeline diagrams
  generate_data.py          # Synthetic data generator
  forecast.py               # Forecasting pipeline (TimesFM + MLForecast + baselines)
  evaluate.py               # Evaluation metrics and plots
  data/
    driver_gps_events.csv   # Raw GPS telemetry + event labels
    sleep_events.csv        # Aggregated sleep sessions
    driver_profiles.csv     # Driver metadata
    cv_sleep_start.csv      # Cross-validation results (sleep start hour)
    cv_duration.csv         # Cross-validation results (sleep duration)
    metrics_*.csv           # Saved evaluation metrics
  plots/
    model_comparison.png    # Side-by-side MAE bar chart
    cv_*.png                # Actual vs predicted time series
    driver_heatmap_*.png    # Per-driver error heatmaps
```
