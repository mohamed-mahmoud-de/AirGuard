"""Integration tests for L2 storage. Requires the Postgres container running:
    docker compose up -d
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.ingest import fetch_air_quality
from src.storage import init_db, upsert_dataframe, read_table

START, END = "2024-06-01", "2024-06-02"   # 48 hourly rows


def test_upsert_and_read():
    init_db()
    aq = fetch_air_quality(START, END)
    n = upsert_dataframe(aq, "raw_air_pollution")
    assert n == 48

    back = read_table("raw_air_pollution", city="Cairo",
                      start=f"{START} 00:00", end=f"{END} 23:00")
    assert len(back) == 48
    assert back["pm2_5"].notna().all()


def test_upsert_is_idempotent():
    """Upserting the same range twice must NOT create duplicates."""
    init_db()
    aq = fetch_air_quality(START, END)
    upsert_dataframe(aq, "raw_air_pollution")
    upsert_dataframe(aq, "raw_air_pollution")  # second time

    back = read_table("raw_air_pollution", city="Cairo",
                      start=f"{START} 00:00", end=f"{END} 23:00")
    assert len(back) == 48, "idempotency broken: duplicates were inserted"
