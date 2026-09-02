"""Unit tests for L3 cleaning.

Pure in-memory tests: no Docker, no network. We build a small synthetic hourly
frame, dirty it in known ways, and check each cleaning guarantee.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clean import (
    deduplicate, fix_impossible_values, reindex_hourly, fill_gaps,
    clean_dataframe, count_gaps, value_columns,
)


def make_frame(hours: int = 48) -> pd.DataFrame:
    """A clean, complete hourly air-quality-shaped frame for one city."""
    idx = pd.date_range("2024-06-01 00:00", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({
        "city": "Cairo",
        "lat": 30.0444,
        "lon": 31.2357,
        "dt_utc": idx,
        "pm2_5": np.linspace(20, 60, hours),
        "relative_humidity_2m": np.linspace(30, 70, hours),
    })


def test_value_columns_excludes_keys():
    cols = value_columns(make_frame())
    assert "pm2_5" in cols and "relative_humidity_2m" in cols
    assert "city" not in cols and "dt_utc" not in cols and "lat" not in cols


def test_deduplicate_removes_repeated_key_and_sorts():
    df = make_frame(5)
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)   # repeat first hour
    out = deduplicate(dup)
    assert len(out) == 5, "duplicate (city, dt_utc) row was not removed"
    assert out["dt_utc"].is_monotonic_increasing, "rows must come out time-sorted"


def test_fix_impossible_values_nulls_only_the_impossible():
    df = make_frame(5)
    df.loc[1, "pm2_5"] = -5             # impossible: negative concentration
    df.loc[2, "relative_humidity_2m"] = 150   # impossible: >100%
    df.loc[3, "pm2_5"] = 999            # extreme but PHYSICALLY POSSIBLE (dust)
    out = fix_impossible_values(df)
    assert np.isnan(out.loc[1, "pm2_5"]), "negative pm2_5 should become NaN"
    assert np.isnan(out.loc[2, "relative_humidity_2m"]), "humidity>100 should be NaN"
    assert out.loc[3, "pm2_5"] == 999, "a real dust spike must be preserved, not clipped"


def test_reindex_hourly_exposes_a_missing_hour():
    df = make_frame(6).drop(index=3).reset_index(drop=True)  # remove one hour
    assert count_gaps(df) == 1, "the removed hour should be counted as a gap"
    out = reindex_hourly(df)
    assert len(out) == 6, "grid should restore the missing hour as a row"
    assert out["pm2_5"].isna().sum() == 1, "the restored hour starts as NaN"
    # metadata must be carried onto the new row
    assert out["city"].notna().all() and out["lat"].notna().all()


def test_fill_gaps_interpolates_short_holes():
    df = make_frame(6)
    df.loc[3, "pm2_5"] = np.nan
    out = fill_gaps(df)
    assert out["pm2_5"].notna().all(), "short gap should be interpolated away"
    # value 20..60 over 6 points -> index 3 should be the interpolated midpoint
    expected = np.linspace(20, 60, 6)[3]
    assert abs(out.loc[3, "pm2_5"] - expected) < 1e-6


def test_clean_dataframe_end_to_end():
    df = make_frame(24)
    dirty = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate
    dirty.loc[5, "pm2_5"] = -1            # impossible
    dirty = dirty.drop(index=10)          # missing hour
    out = clean_dataframe(dirty)
    assert len(out) == 24, "should be exactly one row per hour on the grid"
    assert out.duplicated(subset=["city", "dt_utc"]).sum() == 0
    assert out["pm2_5"].notna().all() and (out["pm2_5"] >= 0).all()


def test_clean_dataframe_is_idempotent():
    df = make_frame(24)
    once = clean_dataframe(df)
    twice = clean_dataframe(once)
    pd.testing.assert_frame_equal(once, twice)


def test_clean_dataframe_empty_input():
    empty = make_frame(0)
    out = clean_dataframe(empty)
    assert out.empty, "cleaning an empty frame should return an empty frame"
