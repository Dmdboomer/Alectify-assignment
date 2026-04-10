# Truck Driver Sleep Prediction

## What This Does

Predicts two things for each truck driver, 14 days ahead:
- **When they'll fall asleep** (e.g., s9:18 PM)
- **How long they'll sleep** (e.g., 7.6 hours)

Uses 1 year of GPS telemetry from 10 drivers across 15 US cities.

## Models

| Model | Why |
|-------|-----|
| **LightGBM** (MLForecast) | Local replacement for TimeGPT. Uses calendar features, lag history, and rolling stats. Same Nixtla DataFrame format - swap one function call to use the real TimeGPT API. |
| **LinearRegression** (MLForecast) | Simple baseline with the same features as LightGBM to test if the complexity is worth it. |
| **TimesFM 200M** (Google) | Pretrained foundation model. No features needed - just feed it the raw time series. Tests whether a large pretrained model can compete without any feature engineering. |
| **AutoARIMA** | Classical statistical model. Auto-tunes its own parameters. Benchmark for whether we even need ML. |
| **SeasonalNaive** | "Same as last week." If the models can't beat this, they're useless. |

## Results

All models predict within **12-18 minutes** of the actual value.

| | Best Model | MAE |
|-|-----------|-----|
| **Sleep start hour** | AutoARIMA | 0.21h (~12 min) |
| **Sleep duration** | TimesFM | 0.20h (~12 min) |

## Plots

### Model Comparison

MAE in hours for each model, side by side for both targets. Shorter bar = better.

![Model Comparison](plots/model_comparison.png)

### Sleep Start Hour - Forecast vs Actual

Black line = actual sleep data. Colored lines = model predictions. 4 drivers shown across the cross-validation test windows.

![Sleep Start Hour](plots/cv_sleep_start_hour.png)

### Sleep Duration - Forecast vs Actual

![Sleep Duration](plots/cv_duration_hours.png)

### Per-Driver Error Heatmaps

Rows = drivers, columns = models. Darker red = higher error. Shows which drivers are harder to predict and which model handles them best.

![Heatmap - Sleep Start](plots/driver_heatmap_sleep_start_hour.png)

![Heatmap - Duration](plots/driver_heatmap_duration_hours.png)

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/SUMMARY.md](docs/SUMMARY.md) | Short summary - models, key techniques, takeaways |
| [docs/TECHNICAL.md](docs/TECHNICAL.md) | Full technical reference - data schemas, setup, directory structure |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Pipeline diagrams - data flow, model routing, cross-validation |

## How to Run

```bash
python -m venv venv && source venv/Scripts/activate
pip install pandas scikit-learn matplotlib lightgbm mlforecast statsforecast
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/google-research/timesfm.git
pip install huggingface_hub safetensors einops

python generate_data.py   # generate synthetic data
python forecast.py        # run all models
python evaluate.py        # compute metrics + plots
```
