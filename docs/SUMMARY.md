# Summary

## Goal

Predict **when** and **how long** a truck driver will sleep on any given day, using 1 year of historical GPS telemetry. Two prediction targets:

- **Sleep start hour** - e.g., "DRV-001 will fall asleep at ~9:18 PM"
- **Sleep duration** - e.g., "DRV-001 will sleep for ~7.6 hours"

## Models

Five models were compared, mirroring the TimeGPT workflow (Nixtla `unique_id | ds | y` format):

| Model | Category | Exogenous Features |
|-------|----------|--------------------|
| LightGBM (MLForecast) | TimeGPT-equivalent | Yes - calendar, lags, rolling stats |
| LinearRegression (MLForecast) | TimeGPT-equivalent | Yes - same as above |
| TimesFM 200M (Google) | Foundation model | No - univariate only |
| AutoARIMA | Classical | No |
| SeasonalNaive | Naive baseline | No |

**MLForecast** replicates the exact TimeGPT API locally: same DataFrame format, same exogenous regressor interface (`X_df`), just swap one function call to switch to the cloud API.

**TimesFM** is Google's 200M-parameter pretrained time series transformer. It runs locally on CPU via PyTorch with no API key.

## Key Technique: Circular Encoding

Sleep start hour has a midnight wraparound problem: 23:50 and 00:10 are 20 minutes apart, but raw numbers (23.83 vs 0.17) look 23.7 hours apart. This destroys model accuracy.

Solution: encode the hour as `sin(2*pi*h/24)` and `cos(2*pi*h/24)`, predict each component as a separate time series, then reconstruct via `atan2`. This makes the feature space continuous across midnight.

## Features

**Exogenous** (known in advance, passed for future dates):
- Day of week (sin/cos cyclical)
- Day of year (sin/cos seasonal)
- Weekend flag

**Auto-generated** by MLForecast:
- Lags at 1, 2, 3, 7, 14 days
- 7-day and 14-day rolling mean
- 7-day rolling std

## Evaluation

4-window rolling cross-validation over the last 2 months, stepping 7 days at a time with a 14-day forecast horizon. Sleep start hour uses **circular MAE** (wraps around midnight) instead of standard MAE.

### Sleep Start Hour

| Model | MAE | In Minutes |
|-------|-----|------------|
| **AutoARIMA** | **0.21h** | **~12 min** |
| TimesFM | 0.22h | ~13 min |
| LinearRegression | 0.22h | ~13 min |
| LightGBM | 0.23h | ~14 min |
| SeasonalNaive | 0.30h | ~18 min |

### Sleep Duration

| Model | MAE | In Minutes |
|-------|-----|------------|
| **TimesFM** | **0.20h** | **~12 min** |
| **LinearRegression** | **0.20h** | **~12 min** |
| AutoARIMA | 0.21h | ~13 min |
| LightGBM | 0.22h | ~13 min |
| SeasonalNaive | 0.29h | ~17 min |

All models except SeasonalNaive predict within **~12-14 minutes** of the actual value. The top models are nearly indistinguishable in accuracy on this dataset.

## Takeaways

1. **Circular encoding was the single biggest improvement** - it dropped sleep start MAE from ~2.85h to ~0.21h.
2. **TimesFM is competitive without any feature engineering** - it matched or beat the exogenous-feature models on duration, using only the raw time series.
3. **The pipeline is TimeGPT-ready** - swapping to the cloud API requires changing one function call; the data format is identical.
4. **Classical models hold up well** - AutoARIMA won on sleep start hour, showing that simple models with good encoding can beat complex ones on low-noise periodic data.
