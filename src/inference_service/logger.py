"""Process-wide logging setup — console + rotating file handler, configured
once from config.yaml. Every other module gets its logger from here; nothing
in this service should use print()."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from inference_service.config import get_config, resolve_path

_configured = False


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
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
