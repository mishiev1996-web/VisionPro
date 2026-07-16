"""
routers/admin.py — Admin endpoints: collect, train, backtest, status, model-stats.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

import db
import data_collector
import state as _state
from state import JOB

logger = logging.getLogger("router.admin")

router = APIRouter(prefix="/api", tags=["admin"])


# ── Refresh status ────────────────────────────────────────────────────────────

@router.get("/refresh-status")
def api_refresh_status():
    with db.connect() as conn:
        last = db.get_meta(conn, "last_refresh")
        n_matches = conn.execute("SELECT COUNT(*) AS c FROM matches").fetchone()["c"]
        n_teams = conn.execute("SELECT COUNT(*) AS c FROM teams").fetchone()["c"]
        n_leagues = conn.execute("SELECT COUNT(*) AS c FROM leagues").fetchone()["c"]
        n_elo = conn.execute("SELECT COUNT(*) AS c FROM team_elo").fetchone()["c"]
        n_inj = conn.execute("SELECT COUNT(*) AS c FROM injuries").fetchone()["c"]
    fs = db.fs_stats()
    return {
        "last_refresh": last,
        "matches": n_matches, "teams": n_teams, "leagues": n_leagues,
        "elo_teams": n_elo, "injuries": n_inj,
        "fs_matches": fs.get("total") or 0,
        "fs_countries": fs.get("countries") or 0,
        "fs_leagues": fs.get("leagues") or 0,
        "fs_live": fs.get("live") or 0,
        "model_loaded": _state.MODEL is not None,
    }


# ── Model stats ──────────────────────────────────────────────────────────────

_MODEL_STATS_CACHE: Optional[Dict[str, Any]] = None
_MODEL_STATS_CACHE_TS: float = 0


@router.get("/model-stats")
def api_model_stats():
    global _MODEL_STATS_CACHE, _MODEL_STATS_CACHE_TS
    import time as _time
    if _state.MODEL is None:
        return {"model_loaded": False}
    try:
        fmt = _state.MODEL.get("format", "v1")
        features = _state.MODEL.get("features", [])
        n_features = len(features)
        model_type = fmt
        if fmt == "ensemble_v3":
            model_type = "DC Ensemble"
        elif fmt == "ensemble_v2":
            model_type = "XGB+LGBM"
        now = _time.time()
        if _MODEL_STATS_CACHE and (now - _MODEL_STATS_CACHE_TS) < 300:
            return {**_MODEL_STATS_CACHE, "model_loaded": True, "format": fmt,
                    "model_type": model_type, "n_features": n_features}
        import backtest
        bt = backtest.run_backtest(seasons=2, max_matches_per_league=100)
        overall = bt.get("overall", {})
        result = {
            "accuracy": overall.get("accuracy", 0),
            "total_matches": overall.get("total", 0),
            "avg_log_loss": overall.get("avg_log_loss", 0),
            "top_20pct_accuracy": overall.get("top_20pct_accuracy", 0),
        }
        _MODEL_STATS_CACHE = result
        _MODEL_STATS_CACHE_TS = now
        return {**result, "model_loaded": True, "format": fmt,
                "model_type": model_type, "n_features": n_features}
    except Exception as e:
        return {"model_loaded": True, "error": str(e)}


# ── Feature importance ────────────────────────────────────────────────────────

@router.get("/feature-importance")
def api_feature_importance():
    if _state.MODEL is None:
        return {"error": "Model not loaded"}
    try:
        base_xgb = _state.MODEL["ensemble"].global_model["xgb"].calibrated_classifiers_[0].estimator
        importances = sorted(zip(_state.MODEL["features"], base_xgb.feature_importances_), key=lambda x: -x[1])
        return {"features": [{"name": n, "importance": round(float(v), 4)} for n, v in importances]}
    except Exception as e:
        return {"error": str(e)}


# ── Data collection ──────────────────────────────────────────────────────────

@router.post("/collect/start")
def api_collect_start(mode: str = "understat", continuous: bool = False):
    if JOB.is_actually_running():
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания или нажмите Стоп")
    current = data_collector._current_season_year()
    seasons = list(range(current - 7, current + 1))
    JOB.reset("collect")
    JOB.continuous = continuous

    def worker():
        try:
            while True:
                if mode == "all" or mode == "continuous":
                    data_collector.collect_all(seasons, progress_cb=JOB.emit, cancel_event=JOB.cancel)
                elif mode == "flashscore":
                    data_collector.collect_flashscore(progress_cb=JOB.emit, cancel_event=JOB.cancel)
                elif mode == "live":
                    data_collector.refresh_live_only(progress_cb=JOB.emit, cancel_event=JOB.cancel)
                else:
                    data_collector.collect_understat(seasons, progress_cb=JOB.emit, cancel_event=JOB.cancel)
                if not continuous or JOB.cancel.is_set():
                    break
                JOB.emit({"type": "info", "msg": "Повторный сбор через 5 минут…"})
                for _ in range(300):
                    if JOB.cancel.is_set():
                        break
                    time.sleep(1)
            JOB.emit({"type": "done", "msg": "Сбор завершён"})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Неожиданная ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = threading.Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "collect", "mode": mode, "continuous": continuous, "seasons": seasons, "job_id": JOB.job_id}


@router.post("/collect/stop")
def api_collect_stop():
    if not JOB.is_actually_running():
        return {"ok": True, "msg": "Ничего не запущено"}
    JOB.cancel.set()
    return {"ok": True, "msg": "Сигнал остановки отправлен — закончит текущую лигу"}


@router.get("/collect/status")
def api_collect_status():
    return {
        "running": JOB.running, "kind": JOB.kind,
        "events": len(JOB.events), "result": JOB.result,
    }


@router.get("/collect/stream")
async def api_collect_stream(request: Request, since: int = 0):
    async def event_gen():
        last_seen = max(0, since)
        last_heartbeat = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            new_events, is_running = JOB.snapshot(last_seen)
            for ev in new_events:
                yield f"data: {__import__('json').dumps({'idx': last_seen, **ev})}\n\n"
                last_seen += 1
            if not is_running and not new_events:
                return
            if time.monotonic() - last_heartbeat > 10:
                yield ": heartbeat\n\n"
                last_heartbeat = time.monotonic()
            await asyncio.sleep(0.3)

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache", "X-Accel-Buffering": "no",
    })


# ── Predict-with-refresh ─────────────────────────────────────────────────────

from pydantic import BaseModel

class PredictRequest(BaseModel):
    home_id: int
    away_id: int


@router.post("/predict-with-refresh/start")
def api_predict_refresh_start(body: PredictRequest):
    from helpers import predict_pair
    if JOB.is_actually_running():
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания")
    home = db.get_team(body.home_id)
    away = db.get_team(body.away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    if body.home_id == body.away_id:
        raise HTTPException(400, "Хозяева и гости должны быть разными")
    JOB.reset("predict")

    def worker():
        try:
            JOB.emit({"type": "info", "msg": f"Поиск матча: {home['name']} vs {away['name']}"})
            current = data_collector._current_season_year()
            leagues_to_refresh = list({home["league_slug"], away["league_slug"]})
            JOB.emit({"type": "info", "msg": f"Обновляю свежие данные: {', '.join(leagues_to_refresh)} · сезон {current}/{current+1}"})
            def predict_cb(ev):
                if ev.get("type") == "done":
                    ev = {**ev, "type": "info", "msg": "✓ Свежие данные собраны"}
                JOB.emit(ev)
            data_collector.collect([current], leagues=leagues_to_refresh, progress_cb=predict_cb, cancel_event=JOB.cancel)
            if JOB.cancel.is_set():
                return
            JOB.emit({"type": "info", "msg": "Считаю прогноз…"})
            prediction = predict_pair(body.home_id, body.away_id, home, away)
            JOB.result = prediction
            JOB.emit({"type": "result", "msg": "Прогноз готов", "prediction": prediction})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = threading.Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "predict", "job_id": JOB.job_id}


# ── Model retrain ─────────────────────────────────────────────────────────────

@router.post("/train")
def api_train():
    if JOB.is_actually_running():
        raise HTTPException(409, "Идёт сбор данных — сначала дождитесь его окончания")
    import train
    from state import load_model
    try:
        train.main()
    except SystemExit as e:
        raise HTTPException(400, str(e))
    load_model()
    return {"ok": True, "model_loaded": _state.MODEL is not None}


# ── Backtest ──────────────────────────────────────────────────────────────────

@router.get("/backtest")
def api_backtest(seasons: int = 3):
    import backtest
    return backtest.run_backtest(seasons=seasons)


@router.post("/backtest/start")
def api_backtest_start(seasons: int = 3):
    if JOB.is_actually_running():
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания")
    JOB.reset("backtest")
    import backtest

    def worker():
        try:
            result = backtest.run_backtest(seasons=seasons, progress_cb=JOB.emit)
            JOB.result = result
            JOB.emit({"type": "result", "msg": "Backtest завершён", "result": result})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка backtest: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = threading.Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "backtest", "job_id": JOB.job_id}
