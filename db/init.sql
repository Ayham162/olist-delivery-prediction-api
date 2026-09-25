-- Mounted into /docker-entrypoint-initdb.d/ (docker-compose.yml). The
-- official postgres image runs every .sql file there automatically, but
-- only on a truly first boot (empty data volume). Schema kept identical to
-- src/inference_service/db.py's ensure_table(), which also runs this at
-- app startup as a second, idempotent guarantee for local dev without Docker.
CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    input JSONB NOT NULL,
    late BOOLEAN NOT NULL,
    probability DOUBLE PRECISION NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
