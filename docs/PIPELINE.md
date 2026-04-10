# Pipeline

## Overview

```
generate_data.py --> data/*.csv --> forecast.py --> data/cv_*.csv --> evaluate.py --> plots/ + metrics
```

## Step 1: Data Generation

```
generate_data.py
      |
      +--> data/driver_gps_events.csv    (51k GPS pings, 10 drivers, 365 days)
      +--> data/sleep_events.csv         (3,650 sleep sessions)
      +--> data/driver_profiles.csv      (10 driver profiles)
```

## Step 2: Forecasting (`forecast.py`)

```
                    data/sleep_events.csv
                    data/driver_gps_events.csv
                             |
                             v
                  +---------------------+
                  |  load_and_prepare() |
                  |  - extract sleep    |
                  |    start hour       |
                  |  - compute sin/cos  |
                  |  - merge drive stats|
                  +----------+----------+
                             |
              +--------------+--------------+
              |                             |
              v                             v
   sleep_start_hour                  duration_hours
   (circular encoding)               (direct)
              |                             |
              v                             v
   +--------------------+        +--------------------+
   | build_timeseries() |        | build_timeseries() |
   | for sin component  |        | target = duration  |
   | for cos component  |        +----------+---------+
   +----------+---------+                   |
              |                             |
              v                             v
   Run 3 model families             Run 3 model families
   on EACH component:               on duration directly:
              |                             |
     +--------+--------+          +--------+--------+
     |        |        |          |        |        |
     v        v        v          v        v        v
  MLFore-  TimesFM  Stats-     MLFore-  TimesFM  Stats-
  cast     (uni-    Fore-      cast     (uni-    Fore-
  +LightGBM variate) cast      +LightGBM variate) cast
  +LinReg          +AutoARIMA +LinReg          +AutoARIMA
                   +Seasonal                   +Seasonal
     |        |        |          |        |        |
     v        v        v          v        v        v
  CV sin   CV sin   CV sin     CV dur   CV dur   CV dur
  CV cos   CV cos   CV cos
     |        |        |
     v        v        v
   +--------------------+
   | circular_to_hour() |
   | atan2(sin, cos)    |
   | --> predicted hours|
   +----------+---------+
              |                             |
              v                             v
     data/cv_sleep_start.csv       data/cv_duration.csv
```

## Step 3: Evaluation (`evaluate.py`)

```
   data/cv_sleep_start.csv          data/cv_duration.csv
              |                             |
              v                             v
   circular_mae()                  mean_absolute_error()
   (wraps around 24h)              (standard)
              |                             |
              v                             v
   +--------------------+        +--------------------+
   | Per-model metrics  |        | Per-model metrics  |
   | Per-driver metrics |        | Per-driver metrics |
   +----------+---------+        +----------+---------+
              |                             |
              +-------------+---------------+
                            |
                            v
                  +---------+---------+
                  |  Generate plots   |
                  +---------+---------+
                            |
              +-------------+-------------+
              |             |             |
              v             v             v
     plots/cv_*.png  plots/model_   plots/driver_
     (actual vs      comparison.png heatmap_*.png
      predicted)     (bar chart)   (MAE per driver)
```

## Cross-Validation Detail

```
Full dataset: Apr 2025 -------------------------------------------- Apr 2026
                                                    |
Training (10 months)                           cutoff    Test (2 months)
|xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx|          |xxxxxxxxxxxxxxxx|

4 rolling windows, step = 7 days, horizon = 14 days:

Window 1:  train xxxxxxxxxxxxxxxxxxxxxx| forecast [14 days]
Window 2:  train xxxxxxxxxxxxxxxxxxxxxxxxxxx| forecast [14 days]
Window 3:  train xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx| forecast [14 days]
Window 4:  train xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx| forecast [14 days]

Each window: fit on all data up to cutoff, predict next 14 days, compare to actuals.
```

## Model Routing

```
                        +------------------+
                        |   Input data     |
                        | unique_id, ds, y |
                        | + exog features  |
                        +--------+---------+
                                 |
                +----------------+----------------+
                |                |                |
                v                v                v
        +-----------+    +-----------+    +-----------+
        | MLForecast|    |  TimesFM  |    |StatsFore- |
        |           |    |           |    |cast       |
        | LightGBM  |    | 200M param|    |           |
        | LinRegres |    | pretrained|    | AutoARIMA |
        |           |    | transform.|    | Seasonal  |
        | USES:     |    |           |    | Naive     |
        | - exog    |    | USES:     |    |           |
        | - lags    |    | - y only  |    | USES:     |
        | - rolling |    | - no exog |    | - y only  |
        | - calendar|    |           |    | - no exog |
        +-----------+    +-----------+    +-----------+
                |                |                |
                v                v                v
        +-----------+    +-----------+    +-----------+
        | LightGBM  |    |  TimesFM  |    | AutoARIMA |
        | LinRegres |    |           |    | Seasonal  |
        | columns   |    |  column   |    | Naive col |
        +-----------+    +-----------+    +-----------+
                |                |                |
                +----------------+----------------+
                                 |
                                 v
                        +------------------+
                        | Merged CV result |
                        | y, LightGBM,     |
                        | LinRegres,       |
                        | TimesFM,         |
                        | AutoARIMA,       |
                        | SeasonalNaive    |
                        +------------------+
```
