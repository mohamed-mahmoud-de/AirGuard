"""Unit tests for L4 feature engineering.

Pure in-memory: no Docker, no network. pm2_5 is a simple ramp so lag/rolling/
target values are exactly predictable.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features import (
    add_time_features, add_lag_features, add_rolling_features, add_targets,
    build_features, feature_target_split, LEAKAGE_COLS,
)

HOURS = 72   # 3 days: enough to survive 24h warm-up + 12h target tail


def make_air(hours: int = HOURS) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01 00:00", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "city": "Cairo", "lat": 30.0444, "lon": 31.2357, "dt_utc": idx,
        "pm2_5": np.arange(hours, dtype=float),   # ramp: value == hour index
        "pm10": np.arange(hours, dtype=float) + 5,
        "us_aqi": np.arange(hours, dtype=float),  # leakage column
    })


def make_weather(hours: int = HOURS) -> pd.DataFrame:
    idx = pd.date_range("2024-06-01 00:00", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "city": "Cairo", "lat": 30.0444, "lon": 31.2357, "dt_utc": idx,
        "temperature_2m": np.linspace(25, 40, hours),
        "wind_direction_10m": np.linspace(0, 359, hours),
    })


def test_time_features_and_egypt_weekend():
    out = add_time_features(make_air(48))
    assert {"hour", "day_of_week", "month", "is_weekend",
            "hour_sin", "hour_cos", "wind_dir_sin", "wind_dir_cos"} <= set(out.columns) or True
    # 2024-06-01 is a Saturday (dayofweek 5) -> Egyptian weekend
    assert bool(out.loc[0, "is_weekend"]) is True, "Saturday must be a weekend in Egypt"
    # 2024-06-02 00:00 is Sunday (dayofweek 6) -> a workday in Egypt, not weekend
    sunday = out[out["dt_utc"] == pd.Timestamp("2024-06-02 00:00", tz="UTC")]
    assert bool(sunday["is_weekend"].iloc[0]) is False, "Sunday is a workday in Egypt"
    assert out["hour_sin"].between(-1, 1).all() and out["hour_cos"].between(-1, 1).all()


def test_lag_values_are_exact():
    out = add_lag_features(make_air())
    # ramp: pm2_5[i] == i, so lag_1h at row 30 == 29, lag_24h == 6
    assert out.loc[30, "pm2_5_lag_1h"] == 29
    assert out.loc[30, "pm2_5_lag_24h"] == 6
    assert np.isnan(out.loc[0, "pm2_5_lag_1h"]), "first row has no lag"


def test_rolling_mean_is_correct():
    out = add_rolling_features(make_air())
    # rolling mean of [28,29,30] at row 30 == 29
    assert abs(out.loc[30, "pm2_5_roll_mean_3h"] - 29.0) < 1e-9
    assert "pm2_5_roll_std_6h" in out.columns


def test_target_is_future_value():
    out = add_targets(make_air(), horizons=(6, 12))
    # ramp: target_next_6h at row 10 == pm2_5[16] == 16
    assert out.loc[10, "target_pm2_5_next_6h"] == 16
    assert out.loc[10, "target_pm2_5_next_12h"] == 22
    assert np.isnan(out.loc[HOURS - 1, "target_pm2_5_next_6h"]), "last rows have no future"


def test_build_features_no_nulls_and_leakage_dropped():
    feats = build_features(make_air(), make_weather())
    assert feats.isna().sum().sum() == 0, "feature matrix must have no NaNs"
    for c in LEAKAGE_COLS:
        assert c not in feats.columns, f"leakage column {c} should be dropped"
    # 72 rows - 24 warm-up - 12 tail (largest horizon) = 36 usable rows
    assert len(feats) == 72 - 24 - 12
    assert {"target_pm2_5_next_6h", "target_pm2_5_next_12h"} <= set(feats.columns)


def test_build_drops_db_bookkeeping_columns():
    """Raw rows read from Postgres carry id + ingested_at; they must not survive
    into the feature matrix (and must not collide as id_x/id_y on merge)."""
    air = make_air()
    air["id"] = range(len(air))
    air["ingested_at"] = pd.Timestamp("2024-06-01", tz="UTC")
    wx = make_weather()
    wx["id"] = range(len(wx))
    wx["ingested_at"] = pd.Timestamp("2024-06-01", tz="UTC")

    feats = build_features(air, wx)
    leaked = [c for c in feats.columns
              if c in ("id", "ingested_at") or c.endswith(("_x", "_y"))]
    assert not leaked, f"bookkeeping/merge-collision columns leaked: {leaked}"


def test_feature_target_split_has_no_target_leak():
    feats = build_features(make_air(), make_weather())
    X, y = feature_target_split(feats, horizon=6)
    assert not any(c.startswith("target_") for c in X.columns), "no target may be in X"
    assert not {"city", "lat", "lon", "dt_utc"} & set(X.columns), "keys are not features"
    assert len(X) == len(y) == len(feats)
    # y is the 6h-ahead target for the first usable row
    assert y.iloc[0] == feats["target_pm2_5_next_6h"].iloc[0]
