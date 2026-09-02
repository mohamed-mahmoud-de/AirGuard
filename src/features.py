"""
AirGuard — L4 Feature engineering.

Turns the cleaned raw tables (L3 output) into a single model-ready table: current
pollutants + weather, plus the engineered signal that actually lets a model
forecast — lags, rolling stats, cyclical time features — and the forecast
targets. This is the input to L5 (LazyPredict → XGBoost).

Design decisions (and the honest reasons)
------------------------------------------
- **Lags 1/3/6/24h** are the strongest predictors: where PM2.5 was heading.
- **Rolling mean/std** smooth noise and capture volatility. They include the
  current hour, which is legitimate — at prediction time t we know everything up
  to and including t. No future information is used.
- **Cyclical time** (hour_sin/cos) matters *specifically here*: the L3 EDA showed
  the persistence error peaks at 12h and drops again at 24h — a strong daily
  cycle. sin/cos let the model see that 23:00 and 01:00 are close, which raw
  integers cannot.
- **Egypt weekend is Friday–Saturday**, not Sat–Sun — `is_weekend` reflects that.
- **Wind direction is circular** (0°≈360°), so we add sin/cos of it; the raw
  degrees are kept for interpretability.
- **Leakage guardrail — drop `us_aqi`.** us_aqi is a deterministic transform of
  the pollutant concentrations (an AQI formula), so as a feature it is a near
  copy of the target's inputs and would flatter the score. We drop it.
- **Two targets** (`next_6h`, `next_12h`): the horizons the EDA identified as the
  sweet spot (short horizons the naive baseline already wins; 24h the daily cycle
  makes naive strong again). One features table serves both L5 models.
- **Time-ordered, no leakage:** every lag/rolling/target is computed per city in
  time order; warm-up rows (no lag yet) and tail rows (no future target yet) are
  dropped so the model never sees a NaN or a fabricated value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns dropped before modelling because they leak / duplicate the signal.
LEAKAGE_COLS: tuple[str, ...] = ("us_aqi",)

# The forecast horizons we build targets for (hours ahead). See module docstring.
DEFAULT_HORIZONS: tuple[int, ...] = (6, 12)

LAGS: tuple[int, ...] = (1, 3, 6, 24)
ROLL_MEAN_WINDOWS: tuple[int, ...] = (3, 6, 24)
ROLL_STD_WINDOWS: tuple[int, ...] = (6,)

JOIN_KEYS = ["city", "lat", "lon", "dt_utc"]

# DB bookkeeping columns that must not enter the feature matrix (and would
# otherwise collide as id_x/id_y when air and weather are merged).
BOOKKEEPING_COLS: tuple[str, ...] = ("id", "ingested_at")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar + cyclical time features derived from dt_utc.

    hour_sin/cos encode the daily cycle so the model treats 23:00 and 01:00 as
    neighbours. is_weekend uses the Egyptian weekend (Friday=4, Saturday=5).
    """
    out = df.copy()
    t = out["dt_utc"].dt
    out["hour"] = t.hour
    out["day_of_week"] = t.dayofweek          # Mon=0 ... Sun=6
    out["month"] = t.month
    out["is_weekend"] = t.dayofweek.isin([4, 5])   # Egypt: Fri + Sat
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    if "wind_direction_10m" in out.columns:
        rad = np.deg2rad(out["wind_direction_10m"])
        out["wind_dir_sin"] = np.sin(rad)
        out["wind_dir_cos"] = np.cos(rad)
    return out


def add_lag_features(df: pd.DataFrame, col: str = "pm2_5",
                     lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Add lagged values of `col` (per city, in time order)."""
    out = df.sort_values(JOIN_KEYS).copy()
    g = out.groupby("city")[col]
    for k in lags:
        out[f"{col}_lag_{k}h"] = g.shift(k)
    return out


def add_rolling_features(df: pd.DataFrame, col: str = "pm2_5",
                         mean_windows: tuple[int, ...] = ROLL_MEAN_WINDOWS,
                         std_windows: tuple[int, ...] = ROLL_STD_WINDOWS) -> pd.DataFrame:
    """Add rolling mean/std of `col` (per city). Windows use past+current hours."""
    out = df.sort_values(JOIN_KEYS).copy()
    g = out.groupby("city")[col]
    for w in mean_windows:
        out[f"{col}_roll_mean_{w}h"] = g.transform(lambda s, w=w: s.rolling(w).mean())
    for w in std_windows:
        out[f"{col}_roll_std_{w}h"] = g.transform(lambda s, w=w: s.rolling(w).std())
    return out


def add_targets(df: pd.DataFrame, col: str = "pm2_5",
                horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """Add forecast targets: `col` shifted N hours into the future, per city.

    target_{col}_next_Nh at time t equals {col} at time t+N — what we want to
    predict. Shifting by -N is what makes this genuine forecasting.
    """
    out = df.sort_values(JOIN_KEYS).copy()
    g = out.groupby("city")[col]
    for h in horizons:
        out[f"target_{col}_next_{h}h"] = g.shift(-h)
    return out


def build_features(air: pd.DataFrame, weather: pd.DataFrame,
                   horizons: tuple[int, ...] = DEFAULT_HORIZONS,
                   drop_leakage: bool = True) -> pd.DataFrame:
    """Full L4: join air+weather → time/lag/rolling features → targets → dropna.

    Expects already-cleaned inputs (L3). Returns a fully-populated feature matrix
    with no NaNs: warm-up rows (lags not yet available) and tail rows (future
    target not yet available) are dropped.
    """
    # Drop DB bookkeeping first so the merge doesn't create id_x/id_y collisions.
    air = air.drop(columns=[c for c in BOOKKEEPING_COLS if c in air.columns])
    weather = weather.drop(columns=[c for c in BOOKKEEPING_COLS if c in weather.columns])

    df = air.merge(weather, on=JOIN_KEYS, how="inner")
    df = df.sort_values(JOIN_KEYS).reset_index(drop=True)

    if drop_leakage:
        df = df.drop(columns=[c for c in LEAKAGE_COLS if c in df.columns])

    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_targets(df, horizons=horizons)

    # Drop rows made invalid by the shifts: engineered features + targets must
    # all be present. This removes the warm-up head and the target-less tail.
    engineered = [c for c in df.columns
                  if c.startswith("pm2_5_lag_")
                  or c.startswith("pm2_5_roll_")
                  or c.startswith("target_pm2_5_next_")]
    df = df.dropna(subset=engineered).reset_index(drop=True)
    return df


def feature_target_split(df: pd.DataFrame, horizon: int = 6) -> tuple[pd.DataFrame, pd.Series]:
    """Split the feature table into X (model inputs) and y for one horizon.

    Drops every target column from X (so no future value leaks in) plus the keys
    that are identifiers, not predictors.
    """
    target = f"target_pm2_5_next_{horizon}h"
    drop = [c for c in df.columns if c.startswith("target_pm2_5_next_")]
    drop += ["city", "lat", "lon", "dt_utc"]
    X = df.drop(columns=[c for c in drop if c in df.columns])
    y = df[target]
    return X, y


if __name__ == "__main__":
    # Build features from the DB (fallback to a live fetch) and store them.
    from src.clean import clean_dataframe
    from src.storage import init_db, upsert_dataframe, read_table, count_rows

    S, E = "2024-01-01", "2024-12-31"
    try:
        air = read_table("raw_air_pollution", city="Cairo", start=S, end=f"{E} 23:00")
        wx = read_table("raw_weather", city="Cairo", start=S, end=f"{E} 23:00")
        assert len(air) and len(wx)
        print(f"raw from DB: air {len(air)}, weather {len(wx)}")
    except Exception as e:
        print(f"DB unavailable ({type(e).__name__}); live fetch")
        from src.ingest import fetch_air_quality, fetch_weather
        air, wx = fetch_air_quality(S, E), fetch_weather(S, E)

    feats = build_features(clean_dataframe(air), clean_dataframe(wx))
    print(f"feature rows: {len(feats)}, columns: {feats.shape[1]}")
    print("any nulls:", int(feats.isna().sum().sum()))

    init_db()
    n = upsert_dataframe(feats, "features")
    print(f"upserted {n} feature rows; table now has {count_rows('features')}")
    X, y = feature_target_split(feats, horizon=6)
    print(f"X shape {X.shape}, y (target_next_6h) mean {y.mean():.2f}")
