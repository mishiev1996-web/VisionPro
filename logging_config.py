"""
logging_config.py — Centralized logging with file rotation.

Replaces ad-hoc .log/.txt files with rotating log handlers.
"""
from __future__ import annotations

import logging
import logging.handlers
import os

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def setup_logging(level: int = logging.INFO):
    """Configure root logger with console + rotating file handlers.

    - console: INFO level, concise format
    - server.log: INFO level, 5 MB rotation, 3 backups
    - errors.log: WARNING level, 2 MB rotation, 5 backups
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers on re-init
    if root.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # Rotating server log
    server_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "server.log"),
        maxBytes=5 * 1024 * 1024,  # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    server_handler.setLevel(level)
    server_handler.setFormatter(fmt)
    root.addHandler(server_handler)

    # Rotating error log (WARNING+ only)
    error_handler = logging.handlers.RotatingFileHandler(
        os.path.join(LOG_DIR, "errors.log"),
        maxBytes=2 * 1024 * 1024,  # 2 MB
        backupCount=5,
        encoding="utf-8",
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(fmt)
    root.addHandler(error_handler)
