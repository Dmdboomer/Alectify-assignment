"""
Truck Driver Sleep Forecasting Pipeline

Mirrors the TimeGPT workflow using Nixtla's local ecosystem:
  - mlforecast (LightGBM) with exogenous features  - replaces TimeGPT
  - TimesFM (univariate, no covariates)             - foundation model comparison
  - StatsForecast baselines                         - classical baselines

TimeGPT approach:
  nixtla.forecast(df=train_df, X_df=future_exog, h=14, freq="D")
    - df has columns: unique_id, ds, y, + exogenous columns
    - X_df has the future values of those same exogenous columns

We replicate this exactly with mlforecast, which uses the same format.

Inputs to the model (per driver, per day):
  y (target)        = sleep_start_hour OR duration_hours
  Exogenous features (known in the future):
    - day_of_week_sin/cos    (cyclical weekday encoding)
    - day_of_year_sin/cos    (cyclical seasonal encoding)
    - is_weekend             (binary)
  Auto-generated lag features (mlforecast handles these):
    - lag 1, 7, 14 days
    - rolling mean 7d, 14d
    - rolling std 7d

Output: predicted y for next 14 days + prediction intervals
"""

import os
import math
import pandas as pd
import numpy as np
import lightgbm as lgb
from mlforecast import MLForecast
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, SeasonalNaive
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
from timesfm.configs import ForecastConfig
from sklearn.linear_model import LinearRegression
from mlforecast.lag_transforms import RollingMean, RollingStd

# --- Config ---
DATA_DIR = "data"
OUTPUT_DIR = "data"
TRAIN_MONTHS = 10
HORIZON = 14
SEASON_LENGTH = 7

# Exogenous feature columns (all known in advance — calendar-based)
EXOG_COLS = [
    "day_of_week_sin", "day_of_week_cos",
    "day_of_year_sin", "day_of_year_cos",
    "is_weekend",
]


def hour_to_circular(hours):
    """Encode hours as sin/cos to handle midnight wraparound."""
    radians = 2 * math.pi * hours / 24.0
    return np.sin(radians), np.cos(radians)


def circular_to_hour(sin_vals, cos_vals):
    """Decode sin/cos back to hours [0, 24)."""
    hours = np.arctan2(sin_vals, cos_vals) * 24.0 / (2 * math.pi)
    return hours % 24.0


def load_and_prepare():
    """Load sleep_events.csv, merge with GPS data for driving features."""
    sleep = pd.read_csv(
        os.path.join(DATA_DIR, "sleep_events.csv"),
        parse_dates=["sleep_start", "sleep_end"],
    )
    sleep["date"] = pd.to_datetime(sleep["date"])
    sleep["sleep_start_hour"] = (
        sleep["sleep_start"].dt.hour + sleep["sleep_start"].dt.minute / 60.0
    )

    # Circular encoding of sleep start hour
    sin_h, cos_h = hour_to_circular(sleep["sleep_start_hour"].values)
    sleep["sleep_start_sin"] = sin_h
    sleep["sleep_start_cos"] = cos_h

    # Keep primary sleep event per driver per day
    sleep = (
        sleep.sort_values("duration_hours", ascending=False)
        .drop_duplicates(subset=["driver_id", "date"], keep="first")
    )

    # Compute driving stats per driver per day from GPS data
    gps = pd.read_csv(
        os.path.join(DATA_DIR, "driver_gps_events.csv"),
        parse_dates=["timestamp"],
    )
    gps["date"] = gps["timestamp"].dt.date.astype("datetime64[ns]")

    drive_stats = (
        gps[gps["event"] == "driving"]
        .groupby(["driver_id", "date"])
        .agg(
            drive_pings=("speed_mph", "count"),
            avg_speed=("speed_mph", "mean"),
            max_speed=("speed_mph", "max"),
        )
        .reset_index()
    )

    # Merge driving stats into sleep data
    sleep = sleep.merge(drive_stats, on=["driver_id", "date"], how="left")
    sleep[["drive_pings", "avg_speed", "max_speed"]] = (
        sleep[["drive_pings", "avg_speed", "max_speed"]].fillna(0)
    )

    return sleep


def add_calendar_features(df):
    """Add calendar-based exogenous features (all known in advance)."""
    dow = df["ds"].dt.dayofweek  # 0=Mon, 6=Sun
    doy = df["ds"].dt.dayofyear

    df["day_of_week_sin"] = np.sin(2 * math.pi * dow / 7)
    df["day_of_week_cos"] = np.cos(2 * math.pi * dow / 7)
    df["day_of_year_sin"] = np.sin(2 * math.pi * doy / 365.25)
    df["day_of_year_cos"] = np.cos(2 * math.pi * doy / 365.25)
    df["is_weekend"] = (dow >= 5).astype(float)

    return df


def build_timeseries(df, target_col):
    """Build daily time series with exogenous features."""
    ts = df[["driver_id", "date", target_col]].rename(
        columns={"driver_id": "unique_id", "date": "ds", target_col: "y"}
    )
    ts = ts.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    # Fill missing days
    filled = []
    for uid, grp in ts.groupby("unique_id"):
        full_range = pd.date_range(grp["ds"].min(), grp["ds"].max(), freq="D")
        grp = grp.set_index("ds").reindex(full_range).rename_axis("ds").reset_index()
        grp["unique_id"] = uid
        grp["y"] = grp["y"].ffill().bfill()
        filled.append(grp)

    ts = pd.concat(filled, ignore_index=True)
    ts = add_calendar_features(ts)

    return ts


def build_future_exog(train_df, horizon):
    """
    Build future exogenous DataFrame (X_df) — this is how TimeGPT works.
    Only calendar features since those are known in advance.
    """
    rows = []
    for uid in train_df["unique_id"].unique():
        last_date = train_df[train_df["unique_id"] == uid]["ds"].max()
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        for d in future_dates:
            rows.append({"unique_id": uid, "ds": d})

    X_df = pd.DataFrame(rows)
    X_df = add_calendar_features(X_df)
    return X_df


def train_test_split(ts_df):
    """Split: first TRAIN_MONTHS months for training, rest for testing."""
    min_date = ts_df["ds"].min()
    cutoff = min_date + pd.DateOffset(months=TRAIN_MONTHS)

    train = ts_df[ts_df["ds"] < cutoff].copy()
    test = ts_df[ts_df["ds"] >= cutoff].copy()

    print(f"  Train: {train['ds'].min().date()} to {train['ds'].max().date()} "
          f"({len(train)} rows)")
    print(f"  Test:  {test['ds'].min().date()} to {test['ds'].max().date()} "
          f"({len(test)} rows)")

    return train, test


# ─── MLForecast (TimeGPT-equivalent) ───────────────────────────

def create_mlforecast():
    """
    Create MLForecast with LightGBM — mirrors TimeGPT's approach:
    - Accepts exogenous features alongside the time series
    - Auto-generates lag and rolling features
    - Uses the same unique_id | ds | y DataFrame format
    """
    return MLForecast(
        models={
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=10,
                verbosity=-1,
            ),
            "LinearRegression": LinearRegression(),
        },
        freq="D",
        lags=[1, 2, 3, 7, 14],
        lag_transforms={
            7: [RollingMean(window_size=7), RollingStd(window_size=7)],
            14: [RollingMean(window_size=14)],
        },
        date_features=["dayofweek", "month"],
    )


def mlforecast_fit_predict(train_df, horizon):
    """Fit MLForecast and predict — equivalent to nixtla.forecast(df, X_df, h)."""
    mlf = create_mlforecast()

    # Build future exogenous (known-ahead features)
    X_df = build_future_exog(train_df, horizon)

    # Fit on training data with exogenous columns
    mlf.fit(train_df, static_features=[])

    # Predict with future exogenous — same as TimeGPT's X_df parameter
    forecasts = mlf.predict(h=horizon, X_df=X_df)

    return forecasts


def mlforecast_cv(ts_df, horizon, n_windows=4, step_size=7):
    """Cross-validation with MLForecast."""
    mlf = create_mlforecast()

    cv_results = mlf.cross_validation(
        df=ts_df,
        h=horizon,
        n_windows=n_windows,
        step_size=step_size,
        static_features=[],
    )

    return cv_results


# ─── TimesFM (univariate foundation model) ─────────────────────

def load_timesfm():
    """Load and compile the TimesFM model."""
    print("  Loading TimesFM 200M (PyTorch)...")
    model = TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    config = ForecastConfig(
        max_context=512,
        max_horizon=128,
        per_core_batch_size=32,
    )
    model.compile(config)
    print("  TimesFM model ready.")
    return model


def timesfm_forecast(model, train_df, horizon):
    """TimesFM univariate forecast (no exogenous support)."""
    results = []
    drivers = train_df["unique_id"].unique()

    inputs = []
    driver_order = []
    for uid in drivers:
        series = (
            train_df[train_df["unique_id"] == uid]
            .sort_values("ds")["y"]
            .values.astype(np.float32)
        )
        inputs.append(series)
        driver_order.append(uid)

    point_forecasts, quantile_forecasts = model.forecast(
        horizon=horizon, inputs=inputs
    )

    for i, uid in enumerate(driver_order):
        last_date = train_df[train_df["unique_id"] == uid]["ds"].max()
        future_dates = pd.date_range(
            last_date + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        for j in range(horizon):
            results.append({
                "unique_id": uid,
                "ds": future_dates[j],
                "TimesFM": point_forecasts[i, j],
            })

    return pd.DataFrame(results)


def timesfm_cv(model, ts_df, horizon, n_windows=4, step_size=7):
    """Manual rolling CV for TimesFM."""
    max_date = ts_df["ds"].max()
    drivers = ts_df["unique_id"].unique()

    all_results = []
    for w in range(n_windows):
        test_end = max_date - pd.Timedelta(days=w * step_size)
        cutoff = test_end - pd.Timedelta(days=horizon)

        train_window = ts_df[ts_df["ds"] <= cutoff]

        inputs, driver_order = [], []
        for uid in drivers:
            series = (
                train_window[train_window["unique_id"] == uid]
                .sort_values("ds")["y"]
                .values.astype(np.float32)
            )
            if len(series) < 14:
                continue
            inputs.append(series)
            driver_order.append(uid)

        point_forecasts, _ = model.forecast(horizon=horizon, inputs=inputs)

        for i, uid in enumerate(driver_order):
            future_dates = pd.date_range(
                cutoff + pd.Timedelta(days=1), periods=horizon, freq="D"
            )
            actuals = (
                ts_df[(ts_df["unique_id"] == uid) & (ts_df["ds"].isin(future_dates))]
                .set_index("ds")["y"]
            )
            for j in range(horizon):
                d = future_dates[j]
                all_results.append({
                    "unique_id": uid,
                    "ds": d,
                    "cutoff": cutoff,
                    "y": actuals.get(d, np.nan),
                    "TimesFM": point_forecasts[i, j],
                })

    return pd.DataFrame(all_results)


# ─── StatsForecast baselines ───────────────────────────────────

def statsforecast_cv(ts_df, horizon, n_windows=4, step_size=7):
    """CV with classical baselines."""
    # Only use unique_id, ds, y columns for StatsForecast
    ts_clean = ts_df[["unique_id", "ds", "y"]].copy()

    models = [
        AutoARIMA(season_length=SEASON_LENGTH),
        SeasonalNaive(season_length=SEASON_LENGTH),
    ]
    sf = StatsForecast(models=models, freq="D", n_jobs=1)
    return sf.cross_validation(
        df=ts_clean, h=horizon, step_size=step_size, n_windows=n_windows,
    ).reset_index()


# ─── Main ──────────────────────────────────────────────────────

def run_target(target_name, target_col, df, tfm_model):
    """Run full pipeline for one target variable."""
    print(f"\n{'-'*60}")
    print(f"  Target: {target_name}")
    print(f"{'-'*60}")

    print("\n  Building time series...")
    ts = build_timeseries(df, target_col)
    train, test = train_test_split(ts)

    # 1. MLForecast (TimeGPT-equivalent) with exogenous features
    print("\n  [MLForecast + LightGBM] Fitting with exogenous features...")
    print(f"    Features: {EXOG_COLS} + auto lags [1,2,3,7,14] + rolling stats")
    ml_forecast = mlforecast_fit_predict(train, HORIZON)
    print(f"    Forecast: {len(ml_forecast)} rows")

    print("  [MLForecast + LightGBM] Cross-validation...")
    ml_cv = mlforecast_cv(ts, HORIZON)
    print(f"    CV: {len(ml_cv)} rows")

    # 2. TimesFM (univariate - no exogenous)
    print("\n  [TimesFM] Univariate forecast...")
    tfm_fc = timesfm_forecast(tfm_model, train[["unique_id", "ds", "y"]], HORIZON)
    print(f"    Forecast: {len(tfm_fc)} rows")

    print("  [TimesFM] Cross-validation...")
    tfm_cv_result = timesfm_cv(
        tfm_model, ts[["unique_id", "ds", "y"]], HORIZON
    )
    print(f"    CV: {len(tfm_cv_result)} rows")

    # 3. StatsForecast baselines
    print("\n  [StatsForecast] Baselines CV...")
    sf_cv = statsforecast_cv(ts, HORIZON)
    print(f"    CV: {len(sf_cv)} rows")

    # -- Merge all CV results --
    cv = ml_cv.merge(
        tfm_cv_result[["unique_id", "ds", "cutoff", "TimesFM"]],
        on=["unique_id", "ds", "cutoff"],
        how="outer",
    )
    cv = cv.merge(
        sf_cv.drop(columns=["y"], errors="ignore"),
        on=["unique_id", "ds", "cutoff"],
        how="outer",
    )

    # Resolve y columns from merges
    y_cols = [c for c in cv.columns if c.startswith("y")]
    if len(y_cols) > 1:
        cv["y"] = cv[y_cols[0]]
        for yc in y_cols[1:]:
            cv["y"] = cv["y"].fillna(cv[yc])
        cv = cv.drop(columns=[c for c in y_cols if c != "y"])

    # Save
    cv.to_csv(os.path.join(OUTPUT_DIR, f"cv_{target_name}.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, f"test_{target_name}.csv"), index=False)

    print(f"\n  Saved: cv_{target_name}.csv ({len(cv)} rows)")
    return cv


def run_circular_sleep_start(df, tfm_model):
    """
    Run sleep_start prediction using circular encoding (sin/cos).

    Instead of predicting raw hours (where 23.5 and 0.5 look 23h apart),
    we predict sin(2*pi*h/24) and cos(2*pi*h/24) separately, then
    reconstruct the hour with atan2. This makes midnight smooth.
    """
    print(f"\n{'='*60}")
    print("  Target: sleep_start (CIRCULAR ENCODING)")
    print(f"{'='*60}")
    print("  Predicting sin and cos components separately,")
    print("  then reconstructing hours via atan2.")

    MODEL_NAMES = ["LightGBM", "LinearRegression", "TimesFM", "AutoARIMA", "SeasonalNaive"]

    # -- Run sin component --
    print("\n  --- sin component ---")
    ts_sin = build_timeseries(df, "sleep_start_sin")
    train_sin, _ = train_test_split(ts_sin)

    print("  [MLForecast] CV on sin...")
    ml_cv_sin = mlforecast_cv(ts_sin, HORIZON)
    print("  [TimesFM] CV on sin...")
    tfm_cv_sin = timesfm_cv(tfm_model, ts_sin[["unique_id", "ds", "y"]], HORIZON)
    print("  [StatsForecast] CV on sin...")
    sf_cv_sin = statsforecast_cv(ts_sin, HORIZON)

    # Merge sin CV
    cv_sin = ml_cv_sin.merge(
        tfm_cv_sin[["unique_id", "ds", "cutoff", "TimesFM"]],
        on=["unique_id", "ds", "cutoff"], how="outer",
    )
    cv_sin = cv_sin.merge(
        sf_cv_sin.drop(columns=["y"], errors="ignore"),
        on=["unique_id", "ds", "cutoff"], how="outer",
    )
    y_cols = [c for c in cv_sin.columns if c.startswith("y")]
    if len(y_cols) > 1:
        cv_sin["y"] = cv_sin[y_cols[0]]
        for yc in y_cols[1:]:
            cv_sin["y"] = cv_sin["y"].fillna(cv_sin[yc])
        cv_sin = cv_sin.drop(columns=[c for c in y_cols if c != "y"])

    # -- Run cos component --
    print("\n  --- cos component ---")
    ts_cos = build_timeseries(df, "sleep_start_cos")
    train_cos, _ = train_test_split(ts_cos)

    print("  [MLForecast] CV on cos...")
    ml_cv_cos = mlforecast_cv(ts_cos, HORIZON)
    print("  [TimesFM] CV on cos...")
    tfm_cv_cos = timesfm_cv(tfm_model, ts_cos[["unique_id", "ds", "y"]], HORIZON)
    print("  [StatsForecast] CV on cos...")
    sf_cv_cos = statsforecast_cv(ts_cos, HORIZON)

    # Merge cos CV
    cv_cos = ml_cv_cos.merge(
        tfm_cv_cos[["unique_id", "ds", "cutoff", "TimesFM"]],
        on=["unique_id", "ds", "cutoff"], how="outer",
    )
    cv_cos = cv_cos.merge(
        sf_cv_cos.drop(columns=["y"], errors="ignore"),
        on=["unique_id", "ds", "cutoff"], how="outer",
    )
    y_cols = [c for c in cv_cos.columns if c.startswith("y")]
    if len(y_cols) > 1:
        cv_cos["y"] = cv_cos[y_cols[0]]
        for yc in y_cols[1:]:
            cv_cos["y"] = cv_cos["y"].fillna(cv_cos[yc])
        cv_cos = cv_cos.drop(columns=[c for c in y_cols if c != "y"])

    # -- Reconstruct hours from sin/cos predictions --
    print("\n  Reconstructing hours from sin/cos predictions...")

    # Merge sin and cos on the join keys
    merge_keys = ["unique_id", "ds", "cutoff"]
    combined = cv_sin[merge_keys + ["y"]].rename(columns={"y": "y_sin"}).merge(
        cv_cos[merge_keys + ["y"]].rename(columns={"y": "y_cos"}),
        on=merge_keys, how="inner",
    )

    # Actual hours from actual sin/cos
    combined["y"] = circular_to_hour(combined["y_sin"].values, combined["y_cos"].values)

    # For each model, reconstruct predicted hours
    for model in MODEL_NAMES:
        if model not in cv_sin.columns or model not in cv_cos.columns:
            continue
        sin_preds = cv_sin.set_index(merge_keys)[model]
        cos_preds = cv_cos.set_index(merge_keys)[model]
        # Align on the same index
        both = pd.DataFrame({"sin": sin_preds, "cos": cos_preds}).dropna()
        hours = circular_to_hour(both["sin"].values, both["cos"].values)
        hour_series = pd.Series(hours, index=both.index, name=model)
        combined = combined.merge(
            hour_series.reset_index(),
            on=merge_keys, how="left",
        )

    # Save
    combined.to_csv(os.path.join(OUTPUT_DIR, "cv_sleep_start.csv"), index=False)
    print(f"\n  Saved: cv_sleep_start.csv ({len(combined)} rows)")

    # Also save the raw hour version for the test set
    ts_raw = build_timeseries(df, "sleep_start_hour")
    _, test_raw = train_test_split(ts_raw)
    test_raw.to_csv(os.path.join(OUTPUT_DIR, "test_sleep_start.csv"), index=False)

    return combined


def main():
    print("=" * 60)
    print("Truck Driver Sleep Forecasting Pipeline")
    print()
    print("  TimeGPT-equivalent : MLForecast + LightGBM (with exogenous)")
    print("  Foundation model   : TimesFM 200M (univariate)")
    print("  Classical baselines: AutoARIMA, SeasonalNaive")
    print("  Sleep start hour   : CIRCULAR ENCODING (sin/cos)")
    print("=" * 60)

    print("\n[1] Loading data...")
    df = load_and_prepare()
    print(f"  {len(df)} sleep events, {df['driver_id'].nunique()} drivers")

    print("\n[2] Loading TimesFM...")
    tfm_model = load_timesfm()

    # Sleep start with circular encoding
    run_circular_sleep_start(df, tfm_model)

    # Duration stays the same (no wraparound issue)
    run_target("duration", "duration_hours", df, tfm_model)

    print("\n" + "=" * 60)
    print("Pipeline complete. Outputs in data/:")
    print("  cv_sleep_start.csv  - circular-encoded sleep start predictions")
    print("  cv_duration.csv     - duration predictions")
    print("=" * 60)


if __name__ == "__main__":
    main()
