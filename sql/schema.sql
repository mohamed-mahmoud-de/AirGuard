-- AirGuard database schema (L2).
-- Idempotent: safe to run repeatedly (CREATE TABLE IF NOT EXISTS).
-- The features table is added in L4 once its columns are final.

CREATE TABLE IF NOT EXISTS raw_air_pollution (
    id                    BIGSERIAL PRIMARY KEY,
    city                  TEXT              NOT NULL,
    lat                   DOUBLE PRECISION  NOT NULL,
    lon                   DOUBLE PRECISION  NOT NULL,
    dt_utc                TIMESTAMPTZ       NOT NULL,
    pm2_5                 DOUBLE PRECISION,
    pm10                  DOUBLE PRECISION,
    carbon_monoxide       DOUBLE PRECISION,
    nitrogen_dioxide      DOUBLE PRECISION,
    sulphur_dioxide       DOUBLE PRECISION,
    ozone                 DOUBLE PRECISION,
    dust                  DOUBLE PRECISION,
    aerosol_optical_depth DOUBLE PRECISION,
    us_aqi                DOUBLE PRECISION,
    ingested_at           TIMESTAMPTZ       NOT NULL DEFAULT now(),
    UNIQUE (city, dt_utc)
);

CREATE TABLE IF NOT EXISTS raw_weather (
    id                   BIGSERIAL PRIMARY KEY,
    city                 TEXT              NOT NULL,
    lat                  DOUBLE PRECISION  NOT NULL,
    lon                  DOUBLE PRECISION  NOT NULL,
    dt_utc               TIMESTAMPTZ       NOT NULL,
    temperature_2m       DOUBLE PRECISION,
    relative_humidity_2m DOUBLE PRECISION,
    surface_pressure     DOUBLE PRECISION,
    wind_speed_10m       DOUBLE PRECISION,
    wind_direction_10m   DOUBLE PRECISION,
    cloud_cover          DOUBLE PRECISION,
    precipitation        DOUBLE PRECISION,
    ingested_at          TIMESTAMPTZ       NOT NULL DEFAULT now(),
    UNIQUE (city, dt_utc)
);

-- Indexes for time-range queries (the model reads by time).
CREATE INDEX IF NOT EXISTS idx_air_dt   ON raw_air_pollution (city, dt_utc);
CREATE INDEX IF NOT EXISTS idx_weather_dt ON raw_weather      (city, dt_utc);
