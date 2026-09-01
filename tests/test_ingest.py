"""Integration tests for L1 ingestion (hits the live Open-Meteo API)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest import fetch_air_quality, fetch_weather, AIR_QUALITY_VARS, WEATHER_VARS

START, END = "2024-06-01", "2024-06-02"   # 2 days = 48 hourly rows


def test_air_quality_shape_and_columns():
    df = fetch_air_quality(START, END)
    assert len(df) == 48, "expected 48 hourly rows for a 2-day range"
    for col in ["city", "lat", "lon", "dt_utc", *AIR_QUALITY_VARS]:
        assert col in df.columns, f"missing column: {col}"
    assert pd.api.types.is_datetime64_any_dtype(df["dt_utc"])
    assert df["pm2_5"].notna().all(), "pm2_5 should be complete (CAMS is gap-free)"


def test_weather_shape_and_columns():
    df = fetch_weather(START, END)
    assert len(df) == 48
    for col in ["city", "lat", "lon", "dt_utc", *WEATHER_VARS]:
        assert col in df.columns, f"missing column: {col}"


def test_air_and_weather_join_on_timestamp():
    aq = fetch_air_quality(START, END)
    wx = fetch_weather(START, END)
    merged = aq.merge(wx, on=["city", "lat", "lon", "dt_utc"], how="inner")
    assert len(merged) == 48, "air quality and weather must align on the same grid"
