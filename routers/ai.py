"""
routers/ai.py — AI analysis endpoints (football), bot predict, predictions CRUD.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import db
import config
import state as _state
from state import JOB
from helpers import predict_pair

logger = logging.getLogger("router.ai")

router = APIRouter(prefix="/api", tags=["ai"])


# ── Bot predict ──────────────────────────────────────────────────────────────

class BotPredictRequest(BaseModel):
    home_name: str
    away_name: str
    init_data: Optional[str] = None


@router.post("/bot/predict")
def api_bot_predict(body: BotPredictRequest):
    import ai_analyzer
    import telegram_auth
    telegram_user = None
    if body.init_data:
        telegram_user = telegram_auth.validate_init_data(body.init_data)
        if telegram_user is None:
            raise HTTPException(403, "Невалидные данные Telegram. Откройте приложение заново.")
    result = ai_analyzer.search_and_predict(body.home_name, body.away_name, progress_cb=lambda e: None)
    if not result:
        return {"error": "Команды не найдены"}
    if telegram_user:
        result["telegram_user_id"] = telegram_user.get("id")
    return result


# ── AI Analysis ──────────────────────────────────────────────────────────────

@router.get("/ai/analyze")
def api_ai_analyze(home_id: int, away_id: int, model: str = "deepseek/deepseek-v3.2"):
    import ai_analyzer
    import traceback
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    if home_id == away_id:
        raise HTTPException(400, "Хозяева и гости должны быть разными")
    try:
        result = ai_analyzer.generate_preview(home_id, away_id, model=model)
        if "error" in result:
            raise HTTPException(500, result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI analyze error: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))


@router.get("/ai/models")
def api_ai_models():
    return {"models": config.AI_MODELS, "default": config.DEFAULT_AI_MODEL}


@router.get("/ai/quick-analysis")
def api_ai_quick(home_id: int, away_id: int, model: str = "deepseek/deepseek-v3.2"):
    import ai_analyzer
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        raise HTTPException(404, "Команда не найдена")
    result = ai_analyzer.generate_preview(home_id, away_id, model=model)
    if "error" in result:
        raise HTTPException(500, result["error"])
    full_analysis = result.get("analysis", "")
    short_prompt = [
        {"role": "system", "content": "Дай краткий вердикт (2-3 предложения) по матчу. Только главный инсайт и ГЛАВНЫЙ ПРОГНОЗ VisionPRO."},
        {"role": "user", "content": full_analysis},
    ]
    short = ai_analyzer._chat(short_prompt, model=model, temperature=0.5, max_tokens=200)
    return {"home": home, "away": away, "verdict": short, "model_used": model}


# ── Search & Auto-ingest + AI ────────────────────────────────────────────────

class SearchPredictRequest(BaseModel):
    home_name: str
    away_name: str
    model: str = "deepseek/deepseek-v3.2"
    init_data: Optional[str] = None


@router.post("/ai/search-and-predict")
def api_ai_search_predict(body: SearchPredictRequest):
    import ai_analyzer
    import telegram_auth
    if body.init_data:
        user = telegram_auth.validate_init_data(body.init_data)
        if user is None:
            raise HTTPException(403, "Невалидные данные Telegram. Откройте приложение заново.")
    return ai_analyzer.search_and_predict(body.home_name, body.away_name, model=body.model)


@router.post("/ai/search-and-predict/start")
def api_ai_search_predict_start(body: SearchPredictRequest):
    if JOB.is_actually_running():
        raise HTTPException(409, "Уже идёт другая задача — дождитесь окончания")
    JOB.reset("ai")
    import ai_analyzer

    def worker():
        try:
            result = ai_analyzer.search_and_predict(body.home_name, body.away_name, model=body.model, progress_cb=JOB.emit)
            JOB.result = result
            JOB.emit({"type": "result", "msg": "Анализ готов", "prediction": result})
        except Exception as e:
            JOB.emit({"type": "error", "msg": f"Ошибка: {e}"})
        finally:
            JOB.finalize()

    JOB.thread = threading.Thread(target=worker, daemon=True)
    JOB.thread.start()
    return {"ok": True, "kind": "ai", "job_id": JOB.job_id}


@router.get("/ai/search-db")
def api_ai_search_db(q: str = Query(min_length=1)):
    results = db.search_team_fuzzy(q, limit=10)
    return {"results": results}


# ── Save / Load Predictions ────────────────────────────────────────────────

class SavePredictionRequest(BaseModel):
    home_name: str
    away_name: str
    league: str = ""
    match_date: str = ""
    analysis: str = ""
    main_bet: str = ""
    confidence: str = ""
    home_win: Optional[float] = None
    draw_prob: Optional[float] = None
    away_win: Optional[float] = None
    total_over: Optional[float] = None
    total_under: Optional[float] = None
    btts_yes: Optional[float] = None
    btts_no: Optional[float] = None
    exact_score: str = ""
    model_used: str = ""
    prediction_type: str = "analysis"
    game_id: Optional[int] = None


@router.post("/predictions/save")
def api_save_prediction(body: SavePredictionRequest):
    import datetime as _dt
    pred = body.model_dump()
    pred["created_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    pred_id = db.save_prediction(pred)
    return {"ok": True, "id": pred_id}


@router.get("/predictions/list")
def api_list_predictions(limit: int = 50):
    return {"predictions": db.list_predictions(limit)}


@router.get("/predictions/stats")
def api_predictions_stats():
    """Legacy stats: by league, confidence breakdown, recent trend."""


@router.post("/predictions/settle")
def api_settle_predictions():
    """Settle predictions against actual match results."""
    result = db.settle_predictions()
    return result


@router.get("/predictions/hitrate")
def api_prediction_hitrate():
    """Hit-rate stats for settled predictions (by confidence, by month)."""
    return db.prediction_stats()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT 500"
        ).fetchall()
    preds = [dict(r) for r in rows]
    total = len(preds)
    if not total:
        return {"total": 0, "win_rate": 0, "by_league": {}, "confidence_breakdown": {}, "recent_trend": []}
    by_league = {}
    conf_breakdown = {"Высокая": 0, "Средняя": 0, "Низкая": 0, "None": 0}
    recent_trend = []
    for p in preds:
        lg = p.get("league") or "Unknown"
        conf = p.get("confidence") or "None"
        if lg not in by_league:
            by_league[lg] = {"total": 0, "high_conf": 0}
        by_league[lg]["total"] += 1
        if conf == "Высокая":
            by_league[lg]["high_conf"] += 1
        if conf in conf_breakdown:
            conf_breakdown[conf] += 1
        created = p.get("created_at", "")
        recent_trend.append({
            "date": created[:10] if created else "",
            "home": p.get("home_name", ""),
            "away": p.get("away_name", ""),
            "confidence": conf,
            "main_bet": p.get("main_bet", ""),
        })
    return {
        "total": total, "by_league": by_league,
        "confidence_breakdown": conf_breakdown,
        "recent_trend": recent_trend[:20],
    }


@router.get("/predictions/{pred_id}")
def api_get_prediction(pred_id: int):
    pred = db.get_prediction(pred_id)
    if not pred:
        raise HTTPException(404, "Прогноз не найден")
    return pred


@router.delete("/predictions/{pred_id}")
def api_delete_prediction(pred_id: int):
    ok = db.delete_prediction(pred_id)
    if not ok:
        raise HTTPException(404, "Прогноз не найден")
    return {"ok": True}


@router.get("/predictions/{pred_id}/print")
def api_prediction_print(pred_id: int):
    pred = db.get_prediction(pred_id)
    if not pred:
        raise HTTPException(404, "Прогноз не найден")
    p = pred
    conf_color = "#22c55e" if p.get("confidence") == "Высокая" else "#eab308" if p.get("confidence") == "Средняя" else "#ef4444"
    html = f"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="UTF-8"><title>VisionPRO — {p['home_name']} vs {p['away_name']}</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#fff;color:#161826;padding:40px;max-width:600px;margin:0 auto}}
  .header{{display:flex;align-items:center;gap:10px;margin-bottom:24px;border-bottom:2px solid #06b6d4;padding-bottom:16px}}
  .logo{{font-size:22px;font-weight:700}} .logo span{{color:#06b6d4}}
  .match{{font-size:24px;font-weight:700;margin-bottom:4px}}
  .meta{{font-size:12px;color:#6b6f8a;margin-bottom:20px}}
  .bet{{background:linear-gradient(135deg,rgba(6,182,212,.08),rgba(139,92,246,.08));border:1px solid rgba(6,182,212,.2);border-radius:12px;padding:16px;margin-bottom:20px}}
  .bet-label{{font-size:10px;text-transform:uppercase;letter-spacing:1px;color:#06b6d4;font-weight:700;margin-bottom:6px}}
  .bet-text{{font-size:20px;font-weight:700}}
  .bet-conf{{font-size:12px;color:{conf_color};margin-top:4px;text-align:right}}
  .bar{{display:flex;height:28px;border-radius:8px;overflow:hidden;margin:12px 0}}
  .bar div{{display:flex;align-items:center;justify-content:center;font-size:12px;font-weight:700;color:#fff}}
  .bar .h{{background:#22c55e}} .bar .d{{background:#eab308}} .bar .a{{background:#ef4444}}
  .analysis{{font-size:13px;line-height:1.7;color:#4b5563;white-space:pre-wrap;margin:16px 0}}
  .footer{{font-size:10px;color:#9a9db3;text-align:center;margin-top:24px;border-top:1px solid #e2e4ee;padding-top:12px}}
  @media print{{body{{padding:20px}}}}
</style></head><body>
  <div class="header"><div class="logo">Vision<span>PRO</span></div><div style="font-size:11px;color:#6b6f8a">AI Football Analytics</div></div>
  <div class="match">{p['home_name']} vs {p['away_name']}</div>
  <div class="meta">{p.get('league','')} · {p.get('model_used','')} · {p.get('created_at','')[:16]}</div>
  {"<div class='bet'><div class='bet-label'>ГЛАВНЫЙ ПРОГНОЗ</div><div class='bet-text'>"+str(p.get('main_bet',''))+"</div><div class='bet-conf'>"+str(p.get('confidence',''))+"</div></div>" if p.get('main_bet') else ""}
  {"<div class='bar'><div class='h' style='width:"+str(p['home_win'])+"%'>"+str(p['home_win'])+"%</div><div class='d' style='width:"+str(p['draw_prob'])+"%'>"+str(p['draw_prob'])+"%</div><div class='a' style='width:"+str(p['away_win'])+"%'>"+str(p['away_win'])+"%</div></div>" if p.get('home_win') is not None else ""}
  {"<div class='analysis'>"+str(p.get('analysis',''))+"</div>" if p.get('analysis') else ""}
  <div class="footer">VisionPRO · Информационный характер · {p.get('model_used','')}</div>
  <script>window.onload=()=>window.print()</script>
</body></html>"""
    return HTMLResponse(content=html)
