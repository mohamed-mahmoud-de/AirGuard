"""
AirGuard — L1 Ingestion.

Fetches hourly air-quality and weather data for a location from the Open-Meteo
APIs and returns clean pandas DataFrames, ready for the storage layer (L2).

Data source facts (verified 2026-08):
- Air quality: https://air-quality-api.open-meteo.com/v1/air-quality
- Weather:     https://archive-api.open-meteo.com/v1/archive
- No API key. Hourly resolution. Both return the same hourly grid, so they
  join cleanly on the timestamp.
- Air-quality values are CAMS reanalysis (model-based, gap-free), NOT raw sensors.
"""
from __future__ import annotations

import requests
import pandas as pd

# --- Open-Meteo endpoints ---
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"

# --- the variables we pull (see Data-Source note in Obsidian) ---
AIR_QUALITY_VARS = [
    "pm2_5", "pm10", "carbon_monoxide", "nitrogen_dioxide",
    "sulphur_dioxide", "ozone", "dust", "aerosol_optical_depth", "us_aqi",
]
WEATHER_VARS = [
    "temperature_2m", "relative_humidity_2m", "surface_pressure",
    "wind_speed_10m", "wind_direction_10m", "cloud_cover", "precipitation",
]

# Cairo default (used when caller does not override)
CAIRO_LAT, CAIRO_LON = 30.0444, 31.2357

REQUEST_TIMEOUT = 60  # seconds


def _fetch(url: str, lat: float, lon: float, hourly_vars: list[str],
           start_date: str, end_date: str, timezone: str = "UTC") -> pd.DataFrame:
    """Call an Open-Meteo endpoint and turn its 'hourly' block into a DataFrame.

    The API returns hourly data as parallel arrays:
        {"hourly": {"time": [...], "pm2_5": [...], "pm10": [...], ...}}
    We convert that into rows, one per hour.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(hourly_vars),
        "start_date": start_date,
        "end_date": end_date,
        "timezone": timezone,
    }
    resp = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:  # Open-Meteo signals bad requests with an 'error' flag
        raise ValueError(f"Open-Meteo error: {data.get('reason', 'unknown')}")

    hourly = data["hourly"]
    df = pd.DataFrame(hourly)
    # 'time' is an ISO string per hour -> a real UTC timestamp column
    df = df.rename(columns={"time": "dt_utc"})
    df["dt_utc"] = pd.to_datetime(df["dt_utc"], utc=True)
    return df


def fetch_air_quality(start_date: str, end_date: str,
                      lat: float = CAIRO_LAT, lon: float = CAIRO_LON,
                      city: str = "Cairo") -> pd.DataFrame:
    """Fetch hourly air-quality data for a date range.

    Args:
        start_date, end_date: 'YYYY-MM-DD' (inclusive).
        lat, lon: location (defaults to Cairo).
        city: label stored with each row.

    Returns:
        DataFrame with columns: city, lat, lon, dt_utc + AIR_QUALITY_VARS.
    """
    df = _fetch(AIR_QUALITY_URL, lat, lon, AIR_QUALITY_VARS, start_date, end_date)
    df.insert(0, "city", city)
    df.insert(1, "lat", lat)
    df.insert(2, "lon", lon)
    return df


def fetch_weather(start_date: str, end_date: str,
                  lat: float = CAIRO_LAT, lon: float = CAIRO_LON,
                  city: str = "Cairo") -> pd.DataFrame:
    """Fetch hourly weather data for a date range (same shape as air quality)."""
    df = _fetch(WEATHER_URL, lat, lon, WEATHER_VARS, start_date, end_date)
    df.insert(0, "city", city)
    df.insert(1, "lat", lat)
    df.insert(2, "lon", lon)
    return df


if __name__ == "__main__":
    # quick manual smoke test
    aq = fetch_air_quality("2024-06-01", "2024-06-02")
    wx = fetch_weather("2024-06-01", "2024-06-02")
    print("AIR QUALITY:", aq.shape)
    print(aq.head(3).to_string())
    print("\nWEATHER:", wx.shape)
    print(wx.head(3).to_string())
