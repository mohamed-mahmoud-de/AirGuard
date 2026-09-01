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
2. **Stores** it in PostgreSQL as a clean, de-duplicated time-series
3. **Engineers features** (lags, rolling stats, time, Saharan dust)
4. **Forecasts** PM2.5 several hours ahead with XGBoost
5. **Advises** — maps the forecast to a health category (Good → Hazardous)
6. **Serves** it in a Streamlit app, and **retrains** as new data arrives

## Architecture
```
Open-Meteo → PostgreSQL → clean/features → XGBoost (+MLflow) → Streamlit
     ▲                                                            │
     └──────────── Airflow schedules ingest + retrain ◄───────────┘
```

## Tech stack
`Python` · `pandas` · `PostgreSQL` · `SQLAlchemy` · `XGBoost` ·
`scikit-learn` · `LazyPredict` · `MLflow` · `Airflow` · `Streamlit` · `Docker`

## Data source
[Open-Meteo](https://open-meteo.com/) Air Quality + Historical Weather APIs —
free, no API key, complete hourly history back to 2018 for Cairo. Air-quality
values are CAMS reanalysis (model-based) for complete, gap-free coverage.

---

## Setup

**1. Python environment** (reproducible via venv + pinned versions):
```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**2. Config** — copy the template and fill in DB credentials (Open-Meteo needs
no key):
```bash
cp .env.example .env
```

**3. Database** — start PostgreSQL in Docker:
```bash
docker compose up -d
```

## Usage (so far)

**Ingest data (L1):**
```bash
python src/ingest.py                 # smoke test: prints sample air + weather
```

**Store to Postgres (L2):**
```bash
python -m src.storage                # creates schema, upserts, reads back
```

**Run tests:**
```bash
python -m pytest -v                  # requires Docker up for storage tests
```

---

## Project structure
```
AirGuard/
├── docker-compose.yml     # PostgreSQL service
├── requirements.txt       # pinned Python deps
├── .env.example           # config template
├── sql/schema.sql         # database tables
├── src/
│   ├── ingest.py          # L1 — Open-Meteo API clients        ✅
│   └── storage.py         # L2 — PostgreSQL upsert + read       ✅
└── tests/                 # integration tests
```

## Roadmap
- [x] L1 — Ingestion (Open-Meteo air quality + weather)
- [x] L2 — Storage (PostgreSQL, idempotent upserts)
- [ ] L3 — Cleaning + EDA (+ persistence baseline)
- [ ] L4 — Feature engineering
- [ ] L5 — Model (XGBoost forecast + classifier, MLflow)
- [ ] L6 — Streamlit app
- [ ] L7 — Airflow scheduling + retrain loop

## Team
- **Mohamed Mahmoud** — MLOps Engineer
- **Nourseen Saeed** — MLOps Engineer

## Disclaimer
Educational project. Air-quality forecasts are estimates for planning, not a
substitute for official government air-quality advisories.
