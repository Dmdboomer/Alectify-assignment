# Technical Reference

## Problem Statement

Long-haul truck drivers operate on irregular schedules that shift over time due to route changes, delivery deadlines, and personal habits. Fatigue-related accidents are a leading cause of trucking incidents. By leveraging GPS telemetry data collected over ~1 year, we aim to **predict when a driver will sleep** in the upcoming days/weeks, enabling fleet managers to:

- Proactively schedule rest stops and relay drivers
- Flag high-fatigue-risk shifts before they happen
- Comply with Hours of Service (HOS) regulations more effectively

## Approach

We use a **multi-model comparison** that mirrors the TimeGPT workflow (Nixtla's `unique_id | ds | y` + exogenous features format), running entirely locally with no API keys required:

| Model | Type | Exogenous Features |
|-------|------|--------------------|
| **MLForecast + LightGBM** | TimeGPT-equivalent (local) | Yes - calendar + auto lag/rolling |
| **MLForecast + LinearRegression** | TimeGPT-equivalent (local) | Yes - calendar + auto lag/rolling |
| **TimesFM 200M** (Google) | Foundation model | No (univariate only) |
| **AutoARIMA** | Classical baseline | No |
| **SeasonalNaive** | Naive baseline | No |

**Key technique**: Sleep start hour uses **circular encoding** (sin/cos of hour/24) to handle midnight wraparound. The model predicts sin and cos components separately, then reconstructs the hour via atan2.

To swap in the real TimeGPT, replace the MLForecast calls with:
```python
from nixtla import NixtlaClient
nixtla = NixtlaClient(api_key="YOUR_KEY")
forecast = nixtla.forecast(df=train_df, X_df=future_exog, h=14, freq="D")
```
The DataFrame format is identical.

## Results

All models predict within **~12-18 minutes** for both targets. Evaluation uses **circular MAE** for sleep start hour (so 23:50 vs 00:10 = 20 min, not 23.7 hours).

### Sleep Start Hour (best: AutoARIMA, MAE = 0.21h / ~12 min)

| Model | MAE | RMSE |
|-------|-----|------|
| **AutoARIMA** | **0.21h (~12 min)** | 0.27h |
| TimesFM | 0.22h (~13 min) | 0.27h |
| LinearRegression | 0.22h (~13 min) | 0.28h |
| LightGBM | 0.23h (~14 min) | 0.29h |
| SeasonalNaive | 0.30h (~18 min) | 0.38h |

### Sleep Duration (best: TimesFM + LinearRegression, MAE = 0.20h / ~12 min)

| Model | MAE | RMSE |
|-------|-----|------|
| **TimesFM** | **0.20h (~12 min)** | 0.27h |
| **LinearRegression** | **0.20h (~12 min)** | 0.27h |
| AutoARIMA | 0.21h (~13 min) | 0.27h |
| LightGBM | 0.22h (~13 min) | 0.28h |
| SeasonalNaive | 0.29h (~17 min) | 0.38h |

## Data

### Source (Synthetic)

Run `python generate_data.py` to produce three CSV files in `data/`:

| File | Description | Rows |
|------|-------------|------|
| `driver_gps_events.csv` | Raw GPS pings + event labels every ~30-60 min per driver | ~51k |
| `sleep_events.csv` | One row per sleep session (start, end, duration, location) | ~3,650 |
| `driver_profiles.csv` | Per-driver metadata (home city, sleep habits) | 10 |

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
| `date` | date | Date the sleep session started |
| `day_of_week` | string | e.g. `Monday` |
| `sleep_start` | datetime | When sleep began |
| `sleep_end` | datetime | When sleep ended |
| `duration_hours` | float | Total sleep duration |
| `latitude` | float | Sleep location latitude |
| `longitude` | float | Sleep location longitude |
| `location_label` | string | City name where driver slept |

### Data characteristics

- **10 drivers**, each with distinct but consistent habits (bedtimes 21:00-23:00, 6.5-8h avg sleep)
- **~365 days** of data (April 2025 - April 2026)
- **Seasonal variation**: slightly longer sleep in winter, shorter in summer
- **Weekly patterns**: higher probability of rest days on weekends
- **Route variability**: drivers move between 15 US cities, sleep location changes daily
- **Mild noise**: late arrivals can push sleep times slightly later; rare mid-week rest days

## Pipeline

See [PIPELINE.md](PIPELINE.md) for a detailed pipeline diagram.

### Quick overview

```
generate_data.py --> data/*.csv --> forecast.py --> data/cv_*.csv --> evaluate.py --> plots/ + metrics
```

1. **`generate_data.py`** - Generates 1 year of synthetic GPS + sleep data for 10 truck drivers
2. **`forecast.py`** - Runs all 5 models with circular encoding for sleep start + 4-window rolling cross-validation
3. **`evaluate.py`** - Computes MAE/RMSE per model (circular-aware for hours), per driver, generates comparison plots

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
  README.md                 # This file
  SUMMARY.md                # Pipeline diagram and architecture
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
