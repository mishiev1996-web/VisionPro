"""
app.py — FastAPI server: REST API + frontend hosting + background data refresh.

Run via start.bat, or manually:
    uvicorn app:app --reload
"""
from __future__ import annotations

import collections
import datetime as dt
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("app")

# Set up rotating file logs (logs/server.log, logs/errors.log)
try:
    from logging_config import setup_logging
    setup_logging()
except Exception:
    pass

import db
import config
import tennis.tennis_db as tennis_db
import data_collector
from state import JOB, MODEL, load_model, load_tennis_model


# ── Rate limiter ─────────────────────────────────────────────────────────────

class _RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, collections.deque] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if key not in self._hits:
                self._hits[key] = collections.deque()
            q = self._hits[key]
            while q and q[0] < now - self.window:
                q.popleft()
            if len(q) >= self.max_requests:
                return False
            q.append(now)
            return True

RATE_LIMITER = _RateLimiter(max_requests=120, window_seconds=60)


# ── Scheduler hooks ──────────────────────────────────────────────────────────

SCHEDULER = None


def _scheduled_refresh():
    if JOB.is_actually_running():
        return
    try:
        data_collector.refresh_current_season()
        logger.info(f"[{dt.datetime.now():%H:%M:%S}] Scheduled Understat refresh ok.")
    except Exception as e:
        logger.info(f"[scheduler] understat refresh failed: {e}")


def _scheduled_live_refresh():
    if JOB.is_actually_running():
        return
    try:
        data_collector.refresh_live_only()
        logger.info(f"[{dt.datetime.now():%H:%M:%S}] Scheduled FlashScore live refresh ok.")
    except Exception as e:
        logger.warning(f"[scheduler] live refresh failed: {e}")


def _scheduled_espn_refresh():
    if JOB.is_actually_running():
        return
    try:
        from web_scraper import gather_team_data
        teams = ["Arsenal", "Chelsea", "Liverpool", "Barcelona", "Real Madrid"]
        for team in teams:
            gather_team_data(team)
        logger.info(f"[{dt.datetime.now():%H:%M:%S}] Scheduled ESPN refresh ok.")
    except Exception as e:
        logger.warning(f"[scheduler] ESPN refresh failed: {e}")


def _scheduled_settle():
    """Daily settlement: match predictions to actual results."""
    try:
        result = db.settle_predictions()
        if result["settled"] > 0:
            logger.info(f"[settle] Settled {result['settled']} predictions, "
                        f"correct: {result['correct']}, hit rate: {result['hit_rate']}%")
        if result["not_found"] > 0:
            logger.info(f"[settle] {result['not_found']} predictions pending (no result data yet)")
    except Exception as e:
        logger.warning(f"[scheduler] settlement failed: {e}")


# ── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    tennis_db.init_db()
    load_model()
    load_tennis_model()

    from apscheduler.schedulers.background import BackgroundScheduler
    global SCHEDULER
    SCHEDULER = BackgroundScheduler(daemon=True)
    SCHEDULER.add_job(_scheduled_refresh, "interval", hours=6, id="understat")
    SCHEDULER.add_job(_scheduled_live_refresh, "interval", minutes=5, id="fs_live")
    SCHEDULER.add_job(_scheduled_espn_refresh, "interval", minutes=30, id="espn")
    SCHEDULER.add_job(_scheduled_settle, "cron", hour=3, minute=0, id="settle")
    SCHEDULER.start()
    logger.info("Scheduler started: Understat every 6h, FlashScore live every 5min, ESPN every 30min, Settlement daily at 03:00.")

    if MODEL is not None:
        def _precompute():
            try:
                from routers.admin import api_model_stats
                api_model_stats()
                logger.info("Model stats pre-computed.")
            except Exception:
                pass
        threading.Thread(target=_precompute, daemon=True).start()
    yield
    if SCHEDULER:
        SCHEDULER.shutdown(wait=False)


# ── FastAPI app ──────────────────────────────────────────────────────────────

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Football AI Predictor", lifespan=lifespan)

_ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "http://localhost:8000").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not RATE_LIMITER.is_allowed(client_ip):
        return StreamingResponse(
            content=b'{"error":"Rate limit exceeded. Try again in a minute."}',
            status_code=429,
            media_type="application/json",
        )
    response = await call_next(request)
    return response


# ── Root / Mini App ──────────────────────────────────────────────────────────

@app.get("/")
def root():
    import time as _time
    v = int(_time.time())
    return FileResponse("frontend/index.html", headers={
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "X-VisionPRO-Version": str(v),
    })


@app.get("/mini-app")
def mini_app():
    return FileResponse("mini_app.html", media_type="text/html")


# ── Tennis startup ───────────────────────────────────────────────────────────

@app.on_event("startup")
async def _tennis_startup():
    tennis_db.init_db()


# ── Include routers ──────────────────────────────────────────────────────────

from routers.football import router as football_router
from routers.tennis import router as tennis_router
from routers.ai import router as ai_router
from routers.admin import router as admin_router

app.include_router(football_router)
app.include_router(tennis_router)
app.include_router(ai_router)
app.include_router(admin_router)
