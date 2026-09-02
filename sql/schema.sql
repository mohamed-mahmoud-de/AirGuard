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

-- Model-ready feature table (L4). Columns match src/features.build_features().
-- us_aqi is intentionally absent (dropped as a leakage/duplicate column).
CREATE TABLE IF NOT EXISTS features (
    id                     BIGSERIAL PRIMARY KEY,
    city                   TEXT              NOT NULL,
    lat                    DOUBLE PRECISION  NOT NULL,
    lon                    DOUBLE PRECISION  NOT NULL,
    dt_utc                 TIMESTAMPTZ       NOT NULL,
    -- current pollutants (raw, minus us_aqi)
    pm2_5                  DOUBLE PRECISION,
    pm10                   DOUBLE PRECISION,
    carbon_monoxide        DOUBLE PRECISION,
    nitrogen_dioxide       DOUBLE PRECISION,
    sulphur_dioxide        DOUBLE PRECISION,
    ozone                  DOUBLE PRECISION,
    dust                   DOUBLE PRECISION,
    aerosol_optical_depth  DOUBLE PRECISION,
    -- current weather
    temperature_2m         DOUBLE PRECISION,
    relative_humidity_2m   DOUBLE PRECISION,
    surface_pressure       DOUBLE PRECISION,
    wind_speed_10m         DOUBLE PRECISION,
    wind_direction_10m     DOUBLE PRECISION,
    cloud_cover            DOUBLE PRECISION,
    precipitation          DOUBLE PRECISION,
    wind_dir_sin           DOUBLE PRECISION,
    wind_dir_cos           DOUBLE PRECISION,
    -- lags
    pm2_5_lag_1h           DOUBLE PRECISION,
    pm2_5_lag_3h           DOUBLE PRECISION,
    pm2_5_lag_6h           DOUBLE PRECISION,
    pm2_5_lag_24h          DOUBLE PRECISION,
    -- rolling stats
    pm2_5_roll_mean_3h     DOUBLE PRECISION,
    pm2_5_roll_mean_6h     DOUBLE PRECISION,
    pm2_5_roll_mean_24h    DOUBLE PRECISION,
    pm2_5_roll_std_6h      DOUBLE PRECISION,
    -- time features
    hour                   INTEGER,
    day_of_week            INTEGER,
    month                  INTEGER,
    is_weekend             BOOLEAN,
    hour_sin               DOUBLE PRECISION,
    hour_cos               DOUBLE PRECISION,
    -- forecast targets
    target_pm2_5_next_6h   DOUBLE PRECISION,
    target_pm2_5_next_12h  DOUBLE PRECISION,
    ingested_at            TIMESTAMPTZ       NOT NULL DEFAULT now(),
    UNIQUE (city, dt_utc)
);

CREATE INDEX IF NOT EXISTS idx_features_dt ON features (city, dt_utc);
