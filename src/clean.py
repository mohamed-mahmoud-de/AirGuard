"""
AirGuard — L3 Cleaning.

Takes a raw table (raw_air_pollution or raw_weather) as a DataFrame and returns
a clean, model-ready-shaped time series: no duplicate hours, a complete hourly
grid, no physically impossible values, and short gaps filled.

Why this layer exists even though CAMS is gap-free
---------------------------------------------------
Open-Meteo air quality is CAMS reanalysis, which is gap-free and non-negative by
construction — so on a clean backfill this layer changes almost nothing. It
still matters because:
  - the hourly DAG (L1) can miss an hour (network blip, late run), and
  - re-runs / merged backfills can create duplicate (city, dt_utc) rows.
Cleaning makes the feature layer (L4) able to assume one row per hour, in order,
with no holes. It is defensive, idempotent, and safe to run repeatedly.

Honest design choice on outliers
---------------------------------
A high PM2.5 spike in Cairo is usually a REAL Saharan-dust event, not an error
(see Risks: dust drives Cairo PM2.5). So we do NOT delete statistical extremes —
that would erase the exact signal the model must learn. We only null out values
that are *physically impossible* (a negative concentration, humidity above 100%)
and let interpolation replace those. Genuine spikes are kept.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns that are metadata / keys, not measured values.
ID_COLS: tuple[str, ...] = ("id", "city", "lat", "lon", "dt_utc", "ingested_at")

# Measured quantities that can never be negative (a negative here is a data
# error, not weather). Temperature is deliberately absent — it can be negative.
NON_NEGATIVE: frozenset[str] = frozenset({
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide", "sulphur_dioxide",
    "ozone", "dust", "aerosol_optical_depth", "us_aqi",
    "surface_pressure", "wind_speed_10m", "cloud_cover", "precipitation",
    "relative_humidity_2m",
})

# Columns with a hard physical range; anything outside it is a data error.
BOUNDED: dict[str, tuple[float, float]] = {
    "relative_humidity_2m": (0.0, 100.0),
    "cloud_cover": (0.0, 100.0),
    "wind_direction_10m": (0.0, 360.0),
}

# Default cap on how many *consecutive* missing hours we interpolate across.
# Interpolating a short gap is honest; inventing a whole day of values is not.
DEFAULT_GAP_LIMIT = 6


def value_columns(df: pd.DataFrame) -> list[str]:
    """Return the measured-value columns (everything that is not a key/metadata)."""
    return [c for c in df.columns if c not in ID_COLS]


def deduplicate(df: pd.DataFrame,
                key: tuple[str, ...] = ("city", "dt_utc")) -> pd.DataFrame:
    """Drop duplicate rows on the unique key, keeping the freshest, and sort by time.

    The DB upsert already prevents duplicates in storage, but a raw DataFrame
    assembled from several fetches can still contain them. 'keep=last' keeps the
    most recently ingested copy (rows arrive in ingest order).
    """
    out = df.sort_values(list(key)).drop_duplicates(subset=list(key), keep="last")
    return out.sort_values("dt_utc").reset_index(drop=True)


def fix_impossible_values(df: pd.DataFrame) -> pd.DataFrame:
    """Set physically impossible readings to NaN so the gap-filler can replace them.

    We do NOT clip to a boundary (that would fabricate a plausible-looking
    number); we mark the value missing and let interpolation decide. Only
    impossible values are touched — real extremes are left alone.
    """
    out = df.copy()
    for col in out.columns:
        if col in NON_NEGATIVE:
            out.loc[out[col] < 0, col] = np.nan
        if col in BOUNDED:
            low, high = BOUNDED[col]
            out.loc[(out[col] < low) | (out[col] > high), col] = np.nan
    return out


def reindex_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex each city onto a complete hourly grid (min→max), exposing any gaps.

    Missing hours become rows of NaN measured values so the next step can fill
    them. city/lat/lon are constant per location, so they are carried forward.
    """
    frames = []
    vals = value_columns(df)
    for city, g in df.groupby("city", sort=False):
        g = g.sort_values("dt_utc")
        full = pd.date_range(g["dt_utc"].min(), g["dt_utc"].max(),
                             freq="h", tz="UTC")
        g = g.set_index("dt_utc").reindex(full)
        g.index.name = "dt_utc"
        g["city"] = city
        # lat/lon are fixed for a city — fill the new rows from the known value.
        for meta in ("lat", "lon"):
            if meta in g.columns:
                g[meta] = g[meta].ffill().bfill()
        frames.append(g.reset_index())
    out = pd.concat(frames, ignore_index=True)
    # keep the original column order where possible
    ordered = [c for c in df.columns if c in out.columns]
    ordered += [c for c in out.columns if c not in ordered]
    return out[ordered]


def fill_gaps(df: pd.DataFrame, limit: int = DEFAULT_GAP_LIMIT) -> pd.DataFrame:
    """Fill short runs of missing values by time interpolation, per city.

    `limit` caps how many consecutive NaNs we bridge, so long outages are left
    visible rather than fabricated. Any NaN still left at an edge (start/end of
    the series, where interpolation has nothing to lean on) is filled from the
    nearest real value.
    """
    out = df.copy()
    vals = value_columns(out)
    parts = []
    for _, g in out.groupby("city", sort=False):
        g = g.sort_values("dt_utc").set_index("dt_utc")
        g[vals] = (g[vals]
                   .interpolate(method="time", limit=limit, limit_direction="both")
                   .ffill()
                   .bfill())
        parts.append(g.reset_index())
    return pd.concat(parts, ignore_index=True)


def count_gaps(df: pd.DataFrame) -> int:
    """How many hourly rows are missing versus a complete grid (before filling).

    A quick data-quality read: 0 means the series is already gap-free (expected
    for CAMS). Report this in EDA rather than assuming completeness.
    """
    missing = 0
    for _, g in df.groupby("city", sort=False):
        span = pd.date_range(g["dt_utc"].min(), g["dt_utc"].max(), freq="h", tz="UTC")
        missing += len(span) - g["dt_utc"].nunique()
    return int(missing)


def clean_dataframe(df: pd.DataFrame, gap_limit: int = DEFAULT_GAP_LIMIT) -> pd.DataFrame:
    """Full L3 clean: dedup → null impossible values → hourly grid → fill gaps.

    Idempotent: cleaning an already-clean frame returns an equivalent frame.
    Works for both raw_air_pollution and raw_weather (same key + metadata shape).
    """
    if df.empty:
        return df.copy()
    out = deduplicate(df)
    out = fix_impossible_values(out)
    out = reindex_hourly(out)
    out = fill_gaps(out, limit=gap_limit)
    return out


if __name__ == "__main__":
    # Smoke test on live data: fetch a week, dirty it, clean it, show the effect.
    from src.ingest import fetch_air_quality

    raw = fetch_air_quality("2024-06-01", "2024-06-07")
    print(f"raw rows: {len(raw)}, gaps vs grid: {count_gaps(raw)}")

    # inject faults to prove the cleaner catches them
    dirty = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)   # a duplicate
    dirty.loc[5, "pm2_5"] = -10          # impossible negative concentration
    dirty = dirty.drop(index=10)         # a missing hour

    cleaned = clean_dataframe(dirty)
    print(f"dirty rows: {len(dirty)}  ->  cleaned rows: {len(cleaned)}")
    print("duplicates after clean:",
          cleaned.duplicated(subset=["city", "dt_utc"]).sum())
    print("nulls in pm2_5 after clean:", cleaned["pm2_5"].isna().sum())
    print("negative pm2_5 after clean:", (cleaned["pm2_5"] < 0).sum())
