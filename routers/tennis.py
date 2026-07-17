"""
routers/tennis.py — Tennis endpoints: rankings, matches, search, predict, analyze.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

import tennis.tennis_db as tennis_db
import tennis.tennis_collector as tennis_collector
import tennis.tennis_ai as tennis_ai
import state as _state
from state import JOB

logger = logging.getLogger("router.tennis")

router = APIRouter(prefix="/api/tennis", tags=["tennis"])


class TennisAnalysisRequest(BaseModel):
    player1: str
    player2: str
    model: str = "deepseek/deepseek-v4-flash"


@router.get("/rankings")
def api_tennis_rankings(tour: str = "atp", limit: int = 50):
    players = tennis_db.list_players(tour, limit)
    return {"tour": tour, "rankings": players}


@router.get("/matches/today")
def api_tennis_matches_today():
    matches = tennis_db.upcoming_matches(50)
    return {"date": dt.date.today().isoformat(), "matches": matches}


@router.get("/matches/live")
def api_tennis_matches_live():
    matches = tennis_db.live_matches()
    return {"matches": matches}


@router.get("/live")
def api_tennis_live():
    """Fetch live tennis matches from FlashScore + Tennis API."""
    import tennis_live
    api_matches = tennis_live.fetch_live_from_api()
    fs_matches = tennis_live.fetch_live_from_flashscore()
    # Merge, dedup by player names
    seen = set()
    all_matches = []
    for m in api_matches + fs_matches:
        key = tuple(sorted([m["player1"].lower(), m["player2"].lower()]))
        if key not in seen:
            seen.add(key)
            all_matches.append(m)
    return {"matches": all_matches, "count": len(all_matches)}


@router.get("/matches/results")
def api_tennis_results(limit: int = 20):
    matches = tennis_db.recent_results(limit)
    return {"matches": matches}


@router.get("/search")
def api_tennis_search(q: str = Query(min_length=1)):
    results = tennis_db.search_player(q)
    return {"results": results}


@router.get("/player/{player_id}")
def api_tennis_player(player_id: int):
    player = tennis_db.get_player(player_id)
    if not player:
        raise HTTPException(404, "Игрок не найден")
    matches = tennis_db.player_matches(player_id, 10)
    return {"player": player, "recent_matches": matches}


@router.get("/h2h")
def api_tennis_h2h(player1_id: int, player2_id: int):
    matches = tennis_db.head_to_head(player1_id, player2_id)
    return {"player1_id": player1_id, "player2_id": player2_id, "matches": matches}


@router.get("/stats")
def api_tennis_stats():
    return tennis_db.db_stats()


@router.get("/predictions")
def api_tennis_predictions(limit: int = 50):
    return {"predictions": tennis_db.list_predictions(limit)}


@router.post("/collect/start")
def api_tennis_collect_start(mode: str = "all"):
    if JOB.is_actually_running():
        raise HTTPException(409, "Уже идёт другая задача")
    JOB.reset("tennis_collect")

    def worker():
        try:
            if mode == "rankings":
                tennis_collector.collect_rankings(progress_cb=JOB.emit)
            elif mode == "today":
                tennis_collector.collect_today(progress_cb=JOB.emit)
            elif mode == "live":
                tennis_collector.collect_live(progress_cb=JOB.emit)
            else:
                tennis_collector.collect_all(progress_cb=JOB.emit, cancel_event=JOB.cancel)
            JOB.emit({"type": "done", "msg": "Сбор теннисных данных завершён"})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = __import__("threading").Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "tennis_collect", "mode": mode, "job_id": JOB.job_id}


@router.post("/analyze")
def api_tennis_analyze(body: TennisAnalysisRequest):
    result = tennis_ai.search_and_analyze(body.player1, body.player2, model=body.model)
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/analyze/start")
def api_tennis_analyze_start(body: TennisAnalysisRequest):
    if JOB.is_actually_running():
        raise HTTPException(409, "Уже идёт другая задача")
    JOB.reset("tennis_ai")

    def worker():
        try:
            result = tennis_ai.search_and_analyze(body.player1, body.player2, model=body.model, progress_cb=JOB.emit)
            JOB.result = result
            JOB.emit({"type": "result", "msg": "Анализ готов", "prediction": result})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = __import__("threading").Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "tennis_ai", "job_id": JOB.job_id}


@router.get("/search-api")
def api_tennis_search_api(q: str = Query(min_length=1)):
    results = tennis_collector.search(q)
    return {"results": results}


@router.get("/prematch/upcoming")
def api_tennis_prematch_upcoming():
    import tennis_prematch
    matches = tennis_prematch.fetch_tennis_upcoming()
    return {"matches": matches, "count": len(matches)}


@router.get("/prematch/live")
def api_tennis_prematch_live():
    import tennis_prematch
    matches = tennis_prematch.fetch_tennis_live()
    return {"matches": matches, "count": len(matches)}


@router.get("/prematch/tournament/{name}")
def api_tennis_prematch_tournament(name: str):
    import tennis_prematch
    return tennis_prematch.fetch_tournament_info(name)


@router.get("/prematch/player/{name}")
def api_tennis_prematch_player(name: str):
    import tennis_prematch
    return tennis_prematch.fetch_player_profile(name)


@router.get("/predict")
def api_tennis_predict(player1_id: int, player2_id: int):
    if _state.TENNIS_MODEL is None:
        raise HTTPException(503, "Теннисная модель не загружена. Запустите tennis_trainer.py")
    p1 = tennis_db.get_player(player1_id)
    p2 = tennis_db.get_player(player2_id)
    if not p1 or not p2:
        raise HTTPException(404, "Игрок не найден")
    all_matches = tennis_db.all_finished_matches()
    from tennis_trainer import build_features_fast, _build_player_stats, _build_ranking_lookup, _build_h2h, _build_form
    import pandas as pd
    player_stats = _build_player_stats(all_matches)
    rankings = _build_ranking_lookup(all_matches)
    h2h = _build_h2h(all_matches)
    form = _build_form(all_matches, 20)
    features = build_features_fast({}, player_stats, rankings, h2h, form, all_matches, p1_id=player1_id, p2_id=player2_id)
    X = pd.DataFrame([features], columns=_state.TENNIS_MODEL["features"])
    p1_win_prob = (
        0.5 * _state.TENNIS_MODEL["xgb"].predict_proba(X)[:, 1][0] +
        0.5 * _state.TENNIS_MODEL["lgbm"].predict_proba(X)[:, 1][0]
    )
    h2h_matches = tennis_db.head_to_head(player1_id, player2_id)
    return {
        "player1": p1, "player2": p2,
        "probabilities": {
            "player1_win": round(float(p1_win_prob) * 100, 1),
            "player2_win": round(float(1 - p1_win_prob) * 100, 1),
        },
        "h2h_last5": h2h_matches,
        "model_used": "XGB+LGBM Ensemble",
    }


@router.post("/predict")
def api_tennis_predict_by_name(body: TennisAnalysisRequest):
    if _state.TENNIS_MODEL is None:
        raise HTTPException(503, "Теннисная модель не загружена")
    p1_results = tennis_db.search_player(body.player1, limit=3)
    p2_results = tennis_db.search_player(body.player2, limit=3)
    if not p1_results:
        raise HTTPException(404, f"Игрок '{body.player1}' не найден")
    if not p2_results:
        raise HTTPException(404, f"Игрок '{body.player2}' не найден")
    p1 = p1_results[0]
    p2 = p2_results[0]
    all_matches = tennis_db.all_finished_matches()
    from tennis_trainer import build_features_fast, _build_player_stats, _build_ranking_lookup, _build_h2h, _build_form
    import pandas as pd
    player_stats = _build_player_stats(all_matches)
    rankings = _build_ranking_lookup(all_matches)
    h2h = _build_h2h(all_matches)
    form = _build_form(all_matches, 20)
    features = build_features_fast({}, player_stats, rankings, h2h, form, all_matches, p1_id=p1["id"], p2_id=p2["id"])
    X = pd.DataFrame([features], columns=_state.TENNIS_MODEL["features"])
    p1_win_prob = (
        0.5 * _state.TENNIS_MODEL["xgb"].predict_proba(X)[:, 1][0] +
        0.5 * _state.TENNIS_MODEL["lgbm"].predict_proba(X)[:, 1][0]
    )
    h2h_matches = tennis_db.head_to_head(p1["id"], p2["id"])
    return {
        "player1": p1, "player2": p2,
        "probabilities": {
            "player1_win": round(float(p1_win_prob) * 100, 1),
            "player2_win": round(float(1 - p1_win_prob) * 100, 1),
        },
        "h2h_last5": h2h_matches,
        "model_used": "XGB+LGBM Ensemble",
    }


@router.get("/ai-and-predict")
def api_tennis_ai_predict(player1: str, player2: str, model: str = "deepseek/deepseek-v4-flash"):
    ml_result = None
    if _state.TENNIS_MODEL is not None:
        try:
            p1_results = tennis_db.search_player(player1, limit=3)
            p2_results = tennis_db.search_player(player2, limit=3)
            if p1_results and p2_results:
                p1 = p1_results[0]
                p2 = p2_results[0]
                all_matches = tennis_db.all_finished_matches()
                from tennis_trainer import build_features_fast, _build_player_stats, _build_ranking_lookup, _build_h2h, _build_form
                import pandas as pd
                player_stats = _build_player_stats(all_matches)
                rankings = _build_ranking_lookup(all_matches)
                h2h = _build_h2h(all_matches)
                form = _build_form(all_matches, 20)
                features = build_features_fast({}, player_stats, rankings, h2h, form, all_matches, p1_id=p1["id"], p2_id=p2["id"])
                X = pd.DataFrame([features], columns=_state.TENNIS_MODEL["features"])
                p1_win_prob = (
                    0.5 * _state.TENNIS_MODEL["xgb"].predict_proba(X)[:, 1][0] +
                    0.5 * _state.TENNIS_MODEL["lgbm"].predict_proba(X)[:, 1][0]
                )
                ml_result = {
                    "player1_win": round(float(p1_win_prob) * 100, 1),
                    "player2_win": round(float(1 - p1_win_prob) * 100, 1),
                }
        except Exception as e:
            logger.warning(f"Tennis ML prediction error: {e}")
    ai_result = tennis_ai.search_and_analyze(player1, player2, model=model)
    if "error" in ai_result:
        raise HTTPException(400, ai_result["error"])
    ai_result["ml_prediction"] = ml_result
    return ai_result
