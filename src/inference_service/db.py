"""Prediction-log persistence to Postgres. This DB is never read at inference
time (a new order has no DB row yet, see README) — it exists only so a
prediction can be evaluated later against ground truth and to feed §10's
monitoring. Because of that, a DB outage must degrade *logging*, not
predictions: every function here is best-effort and swallows its own
failures (logged as a warning), never raises into the request path."""

from __future__ import annotations

import json
from typing import Dict, Optional

import psycopg2
from psycopg2 import sql

from inference_service.config import get_config
from inference_service.logger import get_logger

logger = get_logger(__name__)

CONNECT_TIMEOUT_SECONDS = 2  # fail fast — this must never make a request slow


def _connection_string() -> Optional[str]:
    cfg = get_config()
    db = cfg.database
    host = getattr(db, "host", None)
    if not host or "$" in str(host):
        # ${POSTGRES_HOST} never got expanded -> no .env value set, e.g. local
        # dev outside Docker. Treated as "not configured", not an error.
        return None
    return (
        f"host={db.host} port={db.port} dbname={db.name} "
        f"user={db.user} password={db.password}"
    )


def ensure_table() -> None:
    """Idempotent — also created by db/init.sql on first container boot, but
    calling this at app startup too means local dev (against a Postgres
    outside Docker) doesn't need the init script to have run."""
    conn_str = _connection_string()
    if conn_str is None:
        return
    cfg = get_config()
    table = sql.Identifier(cfg.database.predictions_table)
    try:
        with psycopg2.connect(
            conn_str, connect_timeout=CONNECT_TIMEOUT_SECONDS
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {table} (
                        id SERIAL PRIMARY KEY,
                        input JSONB NOT NULL,
                        late BOOLEAN NOT NULL,
                        probability DOUBLE PRECISION NOT NULL,
                        model_version TEXT NOT NULL,
                        latency_ms DOUBLE PRECISION NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """).format(table=table))
    except Exception as exc:
        logger.warning(
            "could not ensure prediction_logs table exists (non-fatal): %s", exc
        )


def log_prediction(order: Dict, result: Dict, latency_ms: float) -> None:
    """Best-effort insert of one prediction. Never raises."""
    conn_str = _connection_string()
    if conn_str is None:
        return
    cfg = get_config()
    table = sql.Identifier(cfg.database.predictions_table)
    try:
        with psycopg2.connect(
            conn_str, connect_timeout=CONNECT_TIMEOUT_SECONDS
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {table} "
                        "(input, late, probability, model_version, latency_ms) "
                        "VALUES (%s, %s, %s, %s, %s)"
                    ).format(table=table),
                    (
                        json.dumps(order, default=str),
                        result["late"],
                        result["probability"],
                        result["model_version"],
                        latency_ms,
                    ),
                )
    except Exception as exc:
        logger.warning("failed to write prediction log to DB (non-fatal): %s", exc)
