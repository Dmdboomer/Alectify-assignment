"""
Evaluate forecasting results.

Computes per-model and per-driver metrics from cross-validation outputs,
compares all models, and generates summary plots.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_DIR = "data"
PLOT_DIR = "plots"

MODEL_COLS = ["LightGBM", "LinearRegression", "TimesFM", "AutoARIMA", "SeasonalNaive"]

MODEL_COLORS = {
    "LightGBM": "#2ca02c",
    "LinearRegression": "#98df8a",
    "TimesFM": "#1f77b4",
    "AutoARIMA": "#aec7e8",
    "SeasonalNaive": "#c7c7c7",
}


def load_cv(target_name):
    path = os.path.join(DATA_DIR, f"cv_{target_name}.csv")
    df = pd.read_csv(path, parse_dates=["ds"])
    return df


def circular_mae(y_true, y_pred, period=24.0):
    """MAE that handles wraparound (e.g., 23.5 vs 0.5 = 1h, not 23h)."""
    diff = np.abs(y_true - y_pred)
    diff = np.minimum(diff, period - diff)
    return np.mean(diff)


def compute_metrics(cv_df, model_cols, circular=False):
    """Compute MAE, RMSE per model across all drivers."""
    results = []
    for model in model_cols:
        if model not in cv_df.columns:
            continue
        mask = cv_df["y"].notna() & cv_df[model].notna()
        y_true = cv_df.loc[mask, "y"].values
        y_pred = cv_df.loc[mask, model].values

        if circular:
            mae = circular_mae(y_true, y_pred)
            # Circular RMSE
            diff = np.abs(y_true - y_pred)
            diff = np.minimum(diff, 24.0 - diff)
            rmse = np.sqrt(np.mean(diff ** 2))
        else:
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        results.append({
            "model": model,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
        })

    return pd.DataFrame(results)


def compute_per_driver_metrics(cv_df, model_cols, circular=False):
    """Compute MAE per driver per model."""
    rows = []
    for uid, grp in cv_df.groupby("unique_id"):
        for model in model_cols:
            if model not in grp.columns:
                continue
            mask = grp["y"].notna() & grp[model].notna()
            if mask.sum() == 0:
                continue
            y_true = grp.loc[mask, "y"].values
            y_pred = grp.loc[mask, model].values
            if circular:
                mae = circular_mae(y_true, y_pred)
            else:
                mae = mean_absolute_error(y_true, y_pred)
            rows.append({"driver_id": uid, "model": model, "MAE": round(mae, 3)})
    return pd.DataFrame(rows)


def plot_forecasts_vs_actual(cv_df, target_name):
    """Plot actual vs predicted for sample drivers."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    drivers = sorted(cv_df["unique_id"].unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Forecast vs Actual: {target_name}", fontsize=14)

    plot_models = [
        ("LightGBM", "-", 1.4),
        ("TimesFM", "-", 1.0),
        ("AutoARIMA", "--", 0.8),
    ]

    for ax, driver in zip(axes.flat, drivers):
        drv = cv_df[cv_df["unique_id"] == driver].sort_values("ds")

        ax.plot(drv["ds"], drv["y"], "k-", linewidth=0.8, label="Actual", alpha=0.8)
        for model, style, lw in plot_models:
            if model in drv.columns:
                ax.plot(
                    drv["ds"], drv[model], style,
                    linewidth=lw, label=model,
                    color=MODEL_COLORS.get(model), alpha=0.7,
                )

        ax.set_title(driver, fontsize=10)
        ax.set_ylabel(target_name)
        ax.legend(fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"cv_{target_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot: {path}")


def plot_forecasts_circular(cv_df, target_name="sleep_start_hour"):
    """
    Plot sleep start hour in circular (sin/cos) space.

    Hours are mapped to the unit circle via sin(2π·h/24), cos(2π·h/24).
    This removes the 00↔24 wraparound that makes line plots of raw hours
    look like huge spikes when a driver's sleep time crosses midnight.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    drivers = sorted(cv_df["unique_id"].unique())[:4]
    fig, axes = plt.subplots(4, 2, figsize=(14, 14), sharex=True)
    fig.suptitle(
        f"Forecast vs Actual (circular encoding): {target_name}",
        fontsize=14,
    )

    plot_models = [
        ("LightGBM", "-", 1.4),
        ("TimesFM", "-", 1.0),
        ("AutoARIMA", "--", 0.8),
    ]

    def to_sin_cos(hours):
        rad = 2 * np.pi * hours / 24.0
        return np.sin(rad), np.cos(rad)

    for row, driver in enumerate(drivers):
        drv = cv_df[cv_df["unique_id"] == driver].sort_values("ds")
        y_sin, y_cos = to_sin_cos(drv["y"].values)

        ax_sin, ax_cos = axes[row, 0], axes[row, 1]
        ax_sin.plot(drv["ds"], y_sin, "k-", linewidth=0.8, label="Actual", alpha=0.8)
        ax_cos.plot(drv["ds"], y_cos, "k-", linewidth=0.8, label="Actual", alpha=0.8)

        for model, style, lw in plot_models:
            if model not in drv.columns:
                continue
            p_sin, p_cos = to_sin_cos(drv[model].values)
            color = MODEL_COLORS.get(model)
            ax_sin.plot(drv["ds"], p_sin, style, linewidth=lw, label=model,
                        color=color, alpha=0.7)
            ax_cos.plot(drv["ds"], p_cos, style, linewidth=lw, label=model,
                        color=color, alpha=0.7)

        ax_sin.set_title(f"{driver} — sin component", fontsize=10)
        ax_cos.set_title(f"{driver} — cos component", fontsize=10)
        ax_sin.set_ylabel("sin(2π·h/24)")
        ax_cos.set_ylabel("cos(2π·h/24)")
        ax_sin.set_ylim(-1.1, 1.1)
        ax_cos.set_ylim(-1.1, 1.1)
        ax_sin.axhline(0, color="grey", linewidth=0.4, alpha=0.4)
        ax_cos.axhline(0, color="grey", linewidth=0.4, alpha=0.4)
        if row == 0:
            ax_sin.legend(fontsize=7)
            ax_cos.legend(fontsize=7)
        if row == len(drivers) - 1:
            ax_sin.tick_params(axis="x", rotation=30, labelsize=7)
            ax_cos.tick_params(axis="x", rotation=30, labelsize=7)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"cv_{target_name}_circular.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot: {path}")


def plot_residuals(cv_df, target_name, circular=False):
    """
    Plot signed residuals (actual - predicted) for each driver.

    If circular=True, residuals are wrapped into [-12, 12] hours so
    the 00↔24 boundary doesn't create fake 23h errors.
    A flat line at 0 = perfect forecast.
    """
    os.makedirs(PLOT_DIR, exist_ok=True)

    drivers = sorted(cv_df["unique_id"].unique())[:4]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    kind = "circular" if circular else "linear"
    fig.suptitle(
        f"Forecast Residuals (actual − predicted, {kind}): {target_name}",
        fontsize=14,
    )

    plot_models = [
        ("LightGBM", "-", 1.2),
        ("TimesFM", "-", 1.0),
        ("AutoARIMA", "--", 0.9),
    ]

    all_resids = []
    for driver in drivers:
        drv = cv_df[cv_df["unique_id"] == driver].sort_values("ds")
        for model, _, _ in plot_models:
            if model not in drv.columns:
                continue
            diff = drv["y"].values - drv[model].values
            if circular:
                diff = ((diff + 12) % 24) - 12
            all_resids.append(diff[~np.isnan(diff)])
    if all_resids:
        flat = np.concatenate(all_resids)
        pad = max(1.0, np.nanpercentile(np.abs(flat), 99) * 1.1)
        ylim = (-pad, pad) if not circular else (-12, 12)
    else:
        ylim = (-12, 12) if circular else (-5, 5)

    for ax, driver in zip(axes.flat, drivers):
        drv = cv_df[cv_df["unique_id"] == driver].sort_values("ds")
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.7)

        for model, style, lw in plot_models:
            if model not in drv.columns:
                continue
            diff = drv["y"].values - drv[model].values
            if circular:
                diff = ((diff + 12) % 24) - 12
            ax.plot(
                drv["ds"], diff, style,
                linewidth=lw, label=model,
                color=MODEL_COLORS.get(model), alpha=0.8,
            )

        ax.set_title(driver, fontsize=10)
        ax.set_ylabel("Residual (hours)")
        ax.set_ylim(*ylim)
        ax.axhspan(-1, 1, color="green", alpha=0.08)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(axis="x", rotation=30, labelsize=7)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"cv_{target_name}_residuals.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot: {path}")


def plot_model_comparison(metrics_start, metrics_dur):
    """Bar chart comparing model MAE for both targets."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors_start = [MODEL_COLORS.get(m, "#999") for m in metrics_start["model"]]
    ax1.barh(metrics_start["model"], metrics_start["MAE"], color=colors_start)
    ax1.set_xlabel("MAE (hours)")
    ax1.set_title("Sleep Start Hour - Model Comparison")
    for i, mae in enumerate(metrics_start["MAE"]):
        ax1.text(mae + 0.02, i, f"{mae:.2f}h", va="center", fontsize=9)

    colors_dur = [MODEL_COLORS.get(m, "#999") for m in metrics_dur["model"]]
    ax2.barh(metrics_dur["model"], metrics_dur["MAE"], color=colors_dur)
    ax2.set_xlabel("MAE (hours)")
    ax2.set_title("Sleep Duration - Model Comparison")
    for i, mae in enumerate(metrics_dur["MAE"]):
        ax2.text(mae + 0.01, i, f"{mae:.2f}h", va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot: {path}")


def plot_per_driver_heatmap(per_driver_df, target_name):
    """Heatmap of MAE per driver per model."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    pivot = per_driver_df.pivot(index="driver_id", columns="model", values="MAE")
    ordered = [c for c in MODEL_COLS if c in pivot.columns]
    pivot = pivot[ordered]

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isnan(val):
                continue
            ax.text(
                j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                color="white" if val > np.nanmean(pivot.values) else "black",
            )

    plt.colorbar(im, ax=ax, label="MAE (hours)")
    ax.set_title(f"Per-Driver MAE: {target_name}")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"driver_heatmap_{target_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved plot: {path}")


def main():
    print("=" * 60)
    print("Evaluation Report")
    print("=" * 60)

    # --- Sleep Start Hour (circular-aware metrics) ---
    print("\n--- Sleep Start Hour (circular MAE) ---")
    cv_start = load_cv("sleep_start")
    metrics_start = compute_metrics(cv_start, MODEL_COLS, circular=True)
    print("\nOverall Metrics:")
    print(metrics_start.to_string(index=False))

    per_driver_start = compute_per_driver_metrics(cv_start, MODEL_COLS, circular=True)
    print("\nPer-Driver MAE (best model per driver):")
    best_per_driver = per_driver_start.loc[
        per_driver_start.groupby("driver_id")["MAE"].idxmin()
    ]
    print(best_per_driver[["driver_id", "model", "MAE"]].to_string(index=False))

    # --- Sleep Duration ---
    print("\n\n--- Sleep Duration ---")
    cv_dur = load_cv("duration")
    metrics_dur = compute_metrics(cv_dur, MODEL_COLS, circular=False)
    print("\nOverall Metrics:")
    print(metrics_dur.to_string(index=False))

    per_driver_dur = compute_per_driver_metrics(cv_dur, MODEL_COLS, circular=False)
    print("\nPer-Driver MAE (best model per driver):")
    best_per_driver_dur = per_driver_dur.loc[
        per_driver_dur.groupby("driver_id")["MAE"].idxmin()
    ]
    print(best_per_driver_dur[["driver_id", "model", "MAE"]].to_string(index=False))

    # --- Summary ---
    print("\n\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best_start = metrics_start.loc[metrics_start["MAE"].idxmin()]
    best_dur = metrics_dur.loc[metrics_dur["MAE"].idxmin()]
    print(f"\nBest for sleep start hour : {best_start['model']} "
          f"(MAE = {best_start['MAE']}h = ~{int(best_start['MAE']*60)} min)")
    print(f"Best for sleep duration   : {best_dur['model']} "
          f"(MAE = {best_dur['MAE']}h = ~{int(best_dur['MAE']*60)} min)")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_forecasts_vs_actual(cv_start, "sleep_start_hour")
    plot_forecasts_circular(cv_start, "sleep_start_hour")
    plot_residuals(cv_start, "sleep_start_hour", circular=True)
    plot_forecasts_vs_actual(cv_dur, "duration_hours")
    plot_residuals(cv_dur, "duration_hours", circular=False)
    plot_model_comparison(metrics_start, metrics_dur)
    plot_per_driver_heatmap(per_driver_start, "sleep_start_hour")
    plot_per_driver_heatmap(per_driver_dur, "duration_hours")

    metrics_start.to_csv(os.path.join(DATA_DIR, "metrics_sleep_start.csv"), index=False)
    metrics_dur.to_csv(os.path.join(DATA_DIR, "metrics_duration.csv"), index=False)
    print(f"\nMetrics saved to {DATA_DIR}/metrics_*.csv")
    print(f"Plots saved to {PLOT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
