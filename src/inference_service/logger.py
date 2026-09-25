"""Process-wide logging setup — console + rotating file handler, configured
once from config.yaml. Every other module gets its logger from here; nothing
in this service should use print()."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from inference_service.config import get_config, resolve_path

_configured = False

# Configuring the root logger (below) means every third-party library's
# loggers flow into our file too via propagation, not just our own modules'.
# Most are quiet, but great_expectations logs INFO-level internal plugin
# registration chatter ("Skipping registering function X because it is a
# closure") on every import — measured directly: ~46KB of log from 2 HTTP
# requests, almost entirely this. Silenced explicitly rather than lowering
# our own level (which would also hide our own INFO logs).
_NOISY_THIRD_PARTY_LOGGERS = ["great_expectations"]


def _configure() -> None:
    global _configured
    if _configured:
        return

    cfg = get_config()
    log_path = resolve_path(cfg.logging.file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(cfg.logging.format)
    level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    for name in _NOISY_THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
