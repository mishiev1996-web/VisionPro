"""
state.py — Shared mutable state: JOB, MODEL, helpers.

Import from here in routers to avoid circular deps.
"""
from __future__ import annotations

import collections
import datetime as dt
import logging
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib

logger = logging.getLogger("state")

MAX_SSE_EVENTS = 10000

MODEL = None
TENNIS_MODEL = None


# ── Job manager ──────────────────────────────────────────────────────────────

class Job:
    """Tracks one long-running task (collect or predict-with-refresh)."""
    def __init__(self):
        self.kind: Optional[str] = None
        self.running: bool = False
        self.continuous: bool = False
        self.cancel = threading.Event()
        self.events: List[Dict[str, Any]] = []
        self.thread: Optional[threading.Thread] = None
        self.result: Optional[Dict[str, Any]] = None
        self.lock = threading.Lock()
        self.job_id: Optional[int] = None

    def reset(self, kind: str) -> None:
        with self.lock:
            self.kind = kind
            self.running = True
            self.continuous = False
            self.cancel.clear()
            self.events = []
            self.result = None
            self.job_id = (self.job_id or 0) + 1

    def emit(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", dt.datetime.now().isoformat(timespec="seconds"))
        event["job_id"] = self.job_id
        with self.lock:
            if len(self.events) > MAX_SSE_EVENTS:
                self.events = self.events[-MAX_SSE_EVENTS // 2:]
            self.events.append(event)

    def finalize(self) -> None:
        with self.lock:
            self.running = False

    def is_actually_running(self) -> bool:
        if not self.running:
            return False
        if self.thread and self.thread.is_alive():
            return True
        with self.lock:
            self.running = False
            logger.warning("[JOB] Auto-reset stuck running flag (thread dead)")
            return False

    def snapshot(self, since: int = 0) -> Tuple[List[Dict], bool]:
        with self.lock:
            return list(self.events[since:]), self.running


JOB = Job()


# ── Model loading ────────────────────────────────────────────────────────────

def load_model():
    global MODEL
    if os.path.exists("model.pkl"):
        MODEL = joblib.load("model.pkl")
        logger.info("Model loaded from model.pkl")
    else:
        MODEL = None
        logger.warning("No model.pkl — run 'python train.py' to train")


def load_tennis_model():
    global TENNIS_MODEL
    tennis_model_path = os.path.join("tennis", "tennis_model.pkl")
    if os.path.exists(tennis_model_path):
        TENNIS_MODEL = joblib.load(tennis_model_path)
        logger.info("Tennis model loaded from tennis/tennis_model.pkl")
    else:
        TENNIS_MODEL = None
        logger.warning("No tennis/tennis_model.pkl — run 'python -m tennis.tennis_trainer' to train")
