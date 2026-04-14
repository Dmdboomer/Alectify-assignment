# Summary

## Goal

Predict **when** and **how long** a truck driver will sleep on any given day,
using 1 year of historical GPS telemetry. Two prediction targets:

- **Sleep start hour** — e.g., "DRV-003 will fall asleep at ~22:20"
- **Sleep duration** — e.g., "DRV-003 will sleep for ~7.6 hours"

## Models

Five models were compared, mirroring the TimeGPT workflow (Nixtla
`unique_id | ds | y` format):

| Model | Category | Exogenous Features |
|-------|----------|--------------------|
| LightGBM (MLForecast) | TimeGPT-equivalent | Yes — calendar, lags, rolling stats |
| LinearRegression (MLForecast) | TimeGPT-equivalent | Yes — same as above |
| TimesFM 200M (Google) | Foundation model | No — univariate only |
| AutoARIMA | Classical | No |
| SeasonalNaive | Naive baseline | No |

**MLForecast** replicates the exact TimeGPT API locally: same DataFrame
format, same exogenous regressor interface (`X_df`). Swap one function call
to switch to the cloud API.

**TimesFM** is Google's 200M-parameter pretrained time-series transformer.
Runs locally on CPU via PyTorch, no API key required.

## Key Technique: Circular Encoding

Sleep start hour has a midnight wraparound problem: 23:50 and 00:10 are 20
minutes apart, but raw numbers (23.83 vs 0.17) look 23.7 hours apart. This
destroys model accuracy — especially for split-shift drivers whose bedtimes
cluster around 01:00–03:00.

Solution: encode the hour as `sin(2*pi*h/24)` and `cos(2*pi*h/24)`, predict
each component as a separate time series, then reconstruct via `atan2`.

## Dataset

Synthetic, but deliberately non-trivial. 12 drivers split across three shift
types:

- Day-shift (sleep ~20:30–23:30)
- Night-shift (sleep starts 07:30–10:30, typical for long-haul runs)
- Split-shift (sleep starts 00:30–03:30, past midnight)

On top of each driver's baseline the generator layers weekly cycles, annual
seasonality, slow habit drift (bounded random walk), sleep debt
(short-night → long-night), driving fatigue from the prior 3 days, long-haul
trips every ~3 weeks, a 1–2 week vacation window, and occasional disruption
days (weather/breakdowns). Result: duration std ≈ 1.33 h and sleep start
covers the full 0–24 h range with real midnight wraparound.

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

4-window rolling cross-validation over the last 2 months, stepping 7 days at
a time with a 14-day forecast horizon. Sleep start hour uses **circular
MAE** (wraps around 24 h).

### Sleep Start Hour (circular MAE)

| Model | MAE | In Minutes |
|-------|-----|------------|
| **TimesFM** | **0.79 h** | **~47 min** |
| AutoARIMA | 0.83 h | ~50 min |
| LinearRegression | 0.84 h | ~50 min |
| LightGBM | 0.86 h | ~51 min |
| SeasonalNaive | 1.06 h | ~64 min |

### Sleep Duration

| Model | MAE | In Minutes |
|-------|-----|------------|
| **LinearRegression** | **1.01 h** | **~61 min** |
| TimesFM | 1.04 h | ~62 min |
| LightGBM | 1.05 h | ~63 min |
| AutoARIMA | 1.05 h | ~63 min |
| SeasonalNaive | 1.45 h | ~87 min |

All four non-naive models comfortably beat SeasonalNaive. On start hour,
TimesFM benefits most from the wider distribution created by the mixed shift
types — it's genuinely useful where feature-engineered models struggle with
wraparound.

## Takeaways

1. **Circular encoding is load-bearing** with split-shift drivers in the mix
   — without it, start-hour MAE explodes because of the 23 ↔ 0 gap.
2. **TimesFM earns its keep** on the start-hour target once the underlying
   series has real variance. On a near-constant duration target its
   apparent accuracy was mostly an artefact of low target variance.
3. **LinearRegression with the right features is hard to beat on duration.**
   The extra flexibility of LightGBM doesn't pay off on daily data of this
   size.
4. **SeasonalNaive is the baseline to beat.** On this dataset every model
   beats it by 20–40 %, but the margin tells you whether a model is
   actually learning anything vs. echoing last week.
5. **The pipeline is TimeGPT-ready** — swapping to the cloud API requires
   changing one function call; the DataFrame format is identical.
