"""
AirGuard — L2 Storage.

Handles the PostgreSQL data layer: connection, schema creation, idempotent
upserts, and reads. Every write is an UPSERT on the table's UNIQUE(city, dt_utc)
key, so re-running the pipeline never creates duplicates (safe backfills + hourly
DAG runs).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

load_dotenv()

SCHEMA_FILE = Path(__file__).resolve().parents[1] / "sql" / "schema.sql"

_engine: Engine | None = None


def get_engine() -> Engine:
    """Create (once) and return the SQLAlchemy engine from .env settings."""
    global _engine
    if _engine is None:
        url = (
            f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:"
            f"{os.environ['POSTGRES_PASSWORD']}@{os.environ['POSTGRES_HOST']}:"
            f"{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
        )
        _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def init_db() -> None:
    """Run sql/schema.sql to create tables + indexes (idempotent)."""
    ddl = SCHEMA_FILE.read_text(encoding="utf-8")
    with get_engine().begin() as conn:
        conn.execute(text(ddl))


def upsert_dataframe(df: pd.DataFrame, table_name: str,
                     conflict_cols: tuple[str, ...] = ("city", "dt_utc")) -> int:
    """Insert rows, updating on conflict with the unique key. Idempotent.

    Returns the number of rows sent. NaN is converted to SQL NULL.
    """
    if df.empty:
        return 0

    # pandas NaN -> Python None so Postgres stores NULL, not 'NaN'
    clean = df.astype(object).where(pd.notnull(df), None)
    records = clean.to_dict(orient="records")

    engine = get_engine()
    table = Table(table_name, MetaData(), autoload_with=engine)

    stmt = pg_insert(table).values(records)
    # on conflict, refresh every column except the key, id, and ingested_at
    update_cols = {
        c.name: stmt.excluded[c.name]
        for c in table.columns
        if c.name not in conflict_cols and c.name not in ("id", "ingested_at")
    }
    stmt = stmt.on_conflict_do_update(index_elements=list(conflict_cols),
                                      set_=update_cols)
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(records)


def read_table(table_name: str, city: str | None = None,
               start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Read a table into a DataFrame, optionally filtered by city + time range."""
    q = f"SELECT * FROM {table_name}"
    conds, params = [], {}
    if city:
        conds.append("city = :city"); params["city"] = city
    if start:
        conds.append("dt_utc >= :start"); params["start"] = start
    if end:
        conds.append("dt_utc <= :end"); params["end"] = end
    if conds:
        q += " WHERE " + " AND ".join(conds)
    q += " ORDER BY dt_utc"
    with get_engine().connect() as conn:
        return pd.read_sql(text(q), conn, params=params)


def count_rows(table_name: str) -> int:
    with get_engine().connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar_one()


if __name__ == "__main__":
    from src.ingest import fetch_air_quality, fetch_weather

    init_db()
    print("schema ready")

    aq = fetch_air_quality("2024-06-01", "2024-06-02")
    wx = fetch_weather("2024-06-01", "2024-06-02")
    print("upserted air:", upsert_dataframe(aq, "raw_air_pollution"))
    print("upserted weather:", upsert_dataframe(wx, "raw_weather"))

    # re-run to prove idempotency (count must not double)
    upsert_dataframe(aq, "raw_air_pollution")
    print("air rows after re-upsert:", count_rows("raw_air_pollution"))
    print(read_table("raw_air_pollution").head(3).to_string())
