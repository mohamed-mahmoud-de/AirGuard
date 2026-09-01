# AirGuard 🛡️🌫️

**Know before you breathe.** An end-to-end MLOps pipeline that forecasts
**PM2.5 air pollution for Cairo** and turns it into a plain-language health
advisory — so asthmatics, heart patients, children, and the elderly know when
the air is unsafe *before* it gets dangerous.

> NTI × Huawei — AI Track Capstone Project

---

## The problem
Cairo is among the most polluted megacities on earth. People most at risk have
no simple tool telling them when to stay indoors, skip outdoor exercise, or keep
the kids in. AirGuard forecasts pollution and warns ahead of time.

## What it does
1. **Ingests** hourly air-quality + weather data (Open-Meteo, keyless, free)
2. **Stores** it in PostgreSQL as a clean time-series
3. **Engineers features** (lags, rolling stats, time, weather, Saharan dust)
4. **Forecasts** PM2.5 several hours ahead with XGBoost
5. **Advises** — maps the forecast to a health category (Good → Hazardous)
6. **Serves** it in a Streamlit app, and **retrains** as new data arrives

## Architecture
```
Open-Meteo → PostgreSQL → clean/features → XGBoost (+MLflow) → Streamlit
     ▲                                                              │
     └──────────── Airflow schedules ingest + retrain ◄────────────┘
```

## Tech stack
`Python` · `pandas` · `PostgreSQL` · `XGBoost` · `scikit-learn` ·
`LazyPredict` · `MLflow` · `Airflow` · `Streamlit` · `Docker`

## Data source
[Open-Meteo](https://open-meteo.com/) Air Quality + Historical Weather APIs —
free, no API key, complete hourly history back to 2018 for Cairo. Air-quality
values are CAMS reanalysis (model-based), used for complete, gap-free coverage.

## Status
🚧 In development — data source verified, pipeline being built.

## Setup
```bash
cp .env.example .env      # fill in DB credentials (Open-Meteo needs no key)
pip install -r requirements.txt
docker-compose up -d      # postgres + airflow
```
*(Full setup + usage instructions added as the pipeline is built.)*

## Team
2-person capstone team.

## Disclaimer
Educational project. Air-quality forecasts are estimates for planning, not a
substitute for official government air-quality advisories.
