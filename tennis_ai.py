"""
tennis_ai.py — AI-powered tennis match analysis via Polza.ai.

Generates natural language insights: form analysis, H2H breakdown,
surface preferences, key factors, prediction commentary.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
from typing import Any, Dict, List, Optional

from ai_core import chat as _chat, get_api_key as _get_api_key, parse_prob_line

import config
import tennis_db

logger = logging.getLogger("tennis_ai")


def _fetch_player_data_from_api(player_name: str) -> dict:
    """Fetch real player data from Tennis API."""
    import tennis_collector
    results = tennis_collector.search(player_name)
    if not results:
        return {}

    entity = results[0]
    player_id = entity.get("id")
    if not player_id:
        return entity

    # Fetch full details from API
    headers = {
        "X-RapidAPI-Host": config.TENNIS_API_HOST,
        "X-RapidAPI-Key": config.TENNIS_API_KEY,
    }
    try:
        resp = requests.get(
            f"https://{config.TENNIS_API_HOST}/api/tennis/team/{player_id}",
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            data = resp.json()
            team = data.get("team", {})
            entity["country"] = team.get("country", {}).get("name", entity.get("country", ""))
            entity["current_tournament"] = team.get("tournament", {}).get("name", "")
            entity["gender"] = team.get("gender", "")
    except Exception:
        pass

    return entity


def _build_match_context(player1: str, player2: str, match_data: dict = None,
                         live_data: dict = None) -> str:
    """Build context string for LLM analysis with real API data."""
    ctx_parts = []
    ctx_parts.append(f"МАТЧ: {player1} vs {player2}")

    # Live data (FlashScore / Tennis API)
    if live_data and live_data.get("is_live"):
        ctx_parts.append("\n--- LIVE ДАННЫЕ (FlashScore) ---")
        ctx_parts.append(f"  СЧЁТ: {live_data.get('score_display', '?')}")
        ctx_parts.append(f"  Турнир: {live_data.get('tournament', '?')}")
        if live_data.get("round"):
            ctx_parts.append(f"  Раунд: {live_data['round']}")
        if live_data.get("surface"):
            ctx_parts.append(f"  Покрытие: {live_data['surface']}")

        sets = live_data.get("current_sets", [])
        if sets:
            ctx_parts.append("\n  --- ПО НАБОРАМ ---")
            for i, (h, a) in enumerate(sets, 1):
                marker = " ← текущий" if i == len(sets) else ""
                ctx_parts.append(f"    Сет {i}: {h}-{a}{marker}")

        current = live_data.get("current_game", "0-0")
        ctx_parts.append(f"\n  Текущий гейм: {current}")

        h_sets = live_data.get("home_sets", 0)
        a_sets = live_data.get("away_sets", 0)
        ctx_parts.append(f"  Счёт по сетам: {h_sets}:{a_sets}")

        # Momentum
        if len(sets) >= 2:
            prev = sets[-2]
            curr = sets[-1]
            if curr[0] > curr[1] and prev[0] < prev[1]:
                ctx_parts.append(f"  Моментум: {player1} перехватил инициативу")
            elif curr[1] > curr[0] and prev[1] < prev[0]:
                ctx_parts.append(f"  Моментум: {player2} перехватил инициативу")
            elif curr[0] > curr[1]:
                ctx_parts.append(f"  Моментум: {player1} доминирует")
            elif curr[1] > curr[0]:
                ctx_parts.append(f"  Моментум: {player2} доминирует")

        ctx_parts.append("")

    # Fetch real data from API
    p1_data = _fetch_player_data_from_api(player1)
    p2_data = _fetch_player_data_from_api(player2)

    if p1_data:
        ctx_parts.append(f"\n--- ДАННЫЕ ИГРОКА 1 (из API) ---")
        ctx_parts.append(f"  Имя: {p1_data.get('name', player1)}")
        if p1_data.get("ranking"):
            ctx_parts.append(f"  Рейтинг: #{p1_data['ranking']}")
        if p1_data.get("country"):
            ctx_parts.append(f"  Страна: {p1_data['country']}")
        if p1_data.get("current_tournament"):
            ctx_parts.append(f"  Текущий турнир: {p1_data['current_tournament']}")

    if p2_data:
        ctx_parts.append(f"\n--- ДАННЫЕ ИГРОКА 2 (из API) ---")
        ctx_parts.append(f"  Имя: {p2_data.get('name', player2)}")
        if p2_data.get("ranking"):
            ctx_parts.append(f"  Рейтинг: #{p2_data['ranking']}")
        if p2_data.get("country"):
            ctx_parts.append(f"  Страна: {p2_data['country']}")
        if p2_data.get("current_tournament"):
            ctx_parts.append(f"  Текущий турнир: {p2_data['current_tournament']}")

    if match_data:
        if match_data.get("tournament_name"):
            ctx_parts.append(f"\nТурнир: {match_data['tournament_name']}")
        if match_data.get("surface"):
            ctx_parts.append(f"Покрытие: {match_data['surface']}")
        if match_data.get("round_name"):
            ctx_parts.append(f"Раунд: {match_data['round_name']}")

        # Rankings from DB
        if match_data.get("player1_ranking") or match_data.get("player2_ranking"):
            ctx_parts.append("\n--- РЕЙТИНГ (из БД) ---")
            ctx_parts.append(f"  {player1}: #{match_data.get('player1_ranking', '?')}")
            ctx_parts.append(f"  {player2}: #{match_data.get('player2_ranking', '?')}")

        # H2H
        h2h_p1 = match_data.get("h2h_player1_wins", 0)
        h2h_p2 = match_data.get("h2h_player2_wins", 0)
        h2h_total = match_data.get("h2h_total", 0)
        if h2h_total > 0:
            ctx_parts.append(f"\n--- H2H (последние {h2h_total}) ---")
            ctx_parts.append(f"  {player1}: {h2h_p1} побед")
            ctx_parts.append(f"  {player2}: {h2h_p2} побед")

        # Odds
        if match_data.get("odds_player1") and match_data.get("odds_player2"):
            ctx_parts.append("\n--- КОЭФФИЦИЕНТЫ ---")
            ctx_parts.append(f"  {player1}: {match_data['odds_player1']}")
            ctx_parts.append(f"  {player2}: {match_data['odds_player2']}")

        # ML Prediction (from tennis_model.pkl)
        ml_pred = match_data.get("ml_prediction")
        if ml_pred:
            ctx_parts.append("\n--- ПРОГНОЗ МОДЕЛИ (XGB+LGBM) ---")
            ctx_parts.append(f"  Победа {player1}: {ml_pred.get('player1_win', '?')}%")
            ctx_parts.append(f"  Победа {player2}: {ml_pred.get('player2_win', '?')}%")
            ctx_parts.append("  Используй эти вероятности как ОСНОВУ для вердикта.")

    ctx_parts.append("\nВАЖНО: Используй ТОЛЬКО реальные данные выше. Если данных нет — скажи 'нет данных'. Не выдумывай статистику!")

    return "\n".join(ctx_parts)


TENNIS_SYSTEM_PROMPT = """Ты — Oracle AI, аналитик теннисных прогнозов. Обращаешься к пользователю по имени Залман.

**СТИЛЬ ОБЩЕНИЯ:**
- Обращайся: "Залман, слушай..."
- Тон: уверенный эксперт, который знает толк в теннисе
- Без воды, сразу к делу
- Конкретные цифры и факты
- Разговорный, но профессиональный стиль

**ЖЁСТКИЕ ПРАВИЛА:**

1. **МОДЕЛЬ:** Если в контексте есть блок "ПРОГНОЗ МОДЕЛИ (XGB+LGBM)" — используй эти вероятности как ОСНОВУ для ГЛАВНОГО ПРОГНОЗА и строки PROB. Математическая модель точнее твоей интуиции. Запрещено предсказывать исход, противоречащий модели. Если модель даёт 55% игроку 1 — ты пишешь "Победа игрока 1", даже если кажется иначе. PROB:player1/player2 ОБЯЗАНЫ совпадать с цифрами модели (подели на 100).

1.1. **ОГРАНИЧЕНИЕ ПО КОЭФФИЦИЕНТАМ:** Если кф на исход меньше 1.5 — НЕ предлагай ставку на этот исход. Кф < 1.5 означает, что рынок уже оценивает вероятность > 67%, выигрыш слишком мал, чтобы оправдать риск. Вместо этого предложи альтернативу с более интересным кф.

2. **ФОРМАТ ОТВЕТА:**
   - Начни с персонального обращения: "Залман, слушай..."
   - Далее: анализ матча с конкретными цифрами и фактами
   - Статистика игроков, форма, последние результаты
   - В конце: ГЛАВНЫЙ ПРОГНОЗ Oracle с обоснованием
   - В самом конце: PROB:p1=0.XX:p2=0.XX:bet=СТАВКА:confidence=УРОВЕНЬ
   - PROB — это ВЕРОЯТНОСТИ от 0.00 до 1.00 (сумма = 1.00), НЕ коэффициенты!

2. **СТИЛЬ:** Русский, 250-400 слов. Уверенный тон эксперта.
   - НИКОГДА не используй звёздочки, решётки или другой markdown-разметки
   - Пиши простым текстом
   - Используй факты и цифры для обоснования
   - Не добавляй "Прогноз носит информационный характер"

3. **ДОПОЛНИТЕЛЬНЫЕ РЫНКИ:** После ГЛАВНОГО ПРОГНОЗа дай краткий вердикт по:
   - ПОБЕДА: Кто победит с обоснованием
   - ТОТАЛ ГЕЙМОВ: Больше/меньше X.5 геймов с обоснованием
   - ФОРА: Рекомендуемая фора с обоснованием

   Формат в конце:
   ГЛАВНЫЙ ПРОГНОЗ: ...
   ПОБЕДА: Игрок1 / Игрок2 — обоснование
   ТОТАЛ ГЕЙМОВ: Больше/меньше X.5 — обоснование
   ФОРА: Игрок1 (-X.5) / Игрок2 (+X.5) — обоснование
   PROB:p1=X.XX:p2=X.XX:bet=СТАВКА:confidence=УРОВЕНЬ"""


TENNIS_LIVE_PROMPT = """Ты — Oracle AI, аналитик LIVE теннисных матчей. Обращаешься к пользователю по имени Залман.

**СТИЛЬ:** Как в предматчевом промпте — уверенный эксперт, без воды, конкретика.

**ЖЁСТКИЕ ПРАВИЛА ДЛЯ LIVE:**

1. **ТЕКУЩИЙ СЧЁТ:** Учитывай его ВСЕГДА. Если счёт 6-4 3-2 — это факт, а не прогноз. Не предсказывай то, что уже случилось.

2. **СЕТЫ:** Если один игрок ведёт 2-0 по сетам — это решающее преимущество. Учти психологическое давление на отстающего.

3. **МОМЕНТУМ:** Если в контексте сказано "Моментум: Игрок1 перехватил инициативу" — это ключевой фактор. Игрок, перехвативший инерцию, часто доводит сет до победы.

4. **ТЕКУЩИЙ ГЕЙМ:** Учитывай счёт в текущем гейме. Если 40-0 — почти выигранный гейм. Если 30-30 — паритет.

5. **МОДЕЛЬ (ДО МАТЧА):** Если есть блок "ПРОГНОЗ МОДЕЛИ" — это стартовая оценка. LIVE-данные могут её скорректировать, но не отменять. Если модель давала 60% игроку 1, а сейчас он проигрывает 0-2 по сетам — скорректируй вниз, но не ниже 25%.

5.1. **ОГРАНИЧЕНИЕ ПО КОЭФФИЦИЕНТАМ:** Если кф на исход меньше 1.5 — НЕ предлагай ставку на этот исход. Кф < 1.5 означает, что рынок уже оценивает вероятность > 67%, выигрыш слишком мал, чтобы оправдать риск. Вместо этого предложи альтернативу с более интересным кф.

6. **ФОРМАТ ОТВЕТА:**
   - Реакция на текущую ситуацию (без шаблона)
   - Анализ по сетам и геймам
   - Прогноз на оставшуюся часть матча
   - PROB:p1=X.XX:p2=X.XX:bet=СТАВКА:confidence=УРОВЕНЬ

7. **СТИЛЬ:** Русский, 200-350 слов. Без "прогноз носит информационный характер".

Формат в конце:
ГЛАВНЫЙ ПРОГНОЗ: ...
ПОБЕДА: Игрок1 / Игрок2 — обоснование с учётом текущего счёта
ТОТАЛ ГЕЙМОВ: Больше/меньше X.5 — обоснование
PROB:p1=X.XX:p2=X.XX:bet=СТАВКА:confidence=УРОВЕНЬ"""


def _extract_tennis_predictions(text: str) -> dict:
    """Extract structured predictions from LLM text response."""
    # Try unified PROB parser first (handles tennis PROB:p1=...:p2=...)
    prob = parse_prob_line(text)
    if prob:
        return prob

    # Fallback: try to find percentages in text
    import re
    predictions = {}

    m = re.search(r'Победа\s+.*?(\d+)%.*?Победа\s+.*?(\d+)%', text)
    if m:
        predictions["player1_win"] = int(m.group(1))
        predictions["player2_win"] = int(m.group(2))

    m = re.search(r'ГЛАВНЫЙ ПРОГНОЗ.*?:\s*(.+)', text, re.IGNORECASE)
    if m:
        predictions["main_bet"] = m.group(1).strip()[:100]

    return predictions


def generate_analysis(player1: str, player2: str, match_data: dict = None,
                      model: str = None, live_data: dict = None) -> str:
    """Generate AI analysis for a tennis match. Returns plain text like football."""
    context = _build_match_context(player1, player2, match_data, live_data=live_data)

    # Use live prompt if match is live
    is_live = live_data and live_data.get("is_live")
    prompt = TENNIS_LIVE_PROMPT if is_live else TENNIS_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Проанализируй этот теннисный матч:\n\n{context}"},
    ]

    raw = _chat(messages, model=model, temperature=0.7, max_tokens=1500)
    if not raw:
        return None

    return raw


def _parse_analysis_response(text: str) -> dict:
    """Parse the plain text analysis into structured data."""
    if not text:
        return {"analysis": "", "prediction": "", "main_bet": "", "confidence": "Средняя"}

    predictions = _extract_tennis_predictions(text)

    # Extract main sections
    analysis = text
    main_bet = predictions.get("main_bet", "")
    confidence = predictions.get("confidence", "Средняя")
    p1_win = predictions.get("player1_win", 50)
    p2_win = predictions.get("player2_win", 50)

    # Try to extract ГЛАВНЫЙ ПРОГНОЗ
    import re
    m = re.search(r'ГЛАВНЫЙ ПРОГНОЗ[:\s]+(.+?)(?:\n|$)', text, re.IGNORECASE)
    prediction = m.group(1).strip() if m else ""

    return {
        "analysis": text,
        "prediction": prediction,
        "main_bet": main_bet,
        "confidence": confidence,
        "player1_win_prob": p1_win,
        "player2_win_prob": p2_win,
    }


def search_and_analyze(player1_name: str, player2_name: str,
                       model: str = None, progress_cb=None) -> dict:
    """Search for players, build context, generate analysis."""
    import tennis_collector

    def _emit(ev):
        if progress_cb:
            try:
                progress_cb(ev)
            except Exception:
                pass

    _emit({"type": "info", "msg": f"Поиск игроков: {player1_name} vs {player2_name}"})

    # First try local DB search
    p1_results = tennis_db.search_player(player1_name, limit=3)
    p2_results = tennis_db.search_player(player2_name, limit=3)

    # If not found in DB, try API
    if not p1_results:
        _emit({"type": "info", "msg": f"Ищу '{player1_name}' в API..."})
        api_results = tennis_collector.search(player1_name)
        if api_results:
            p1_results = [{"id": r.get("id"), "name": r.get("name"), "country": r.get("country", "")}
                          for r in api_results[:3]]

    if not p2_results:
        _emit({"type": "info", "msg": f"Ищу '{player2_name}' в API..."})
        api_results = tennis_collector.search(player2_name)
        if api_results:
            p2_results = [{"id": r.get("id"), "name": r.get("name"), "country": r.get("country", "")}
                          for r in api_results[:3]]

    p1 = p1_results[0] if p1_results else None
    p2 = p2_results[0] if p2_results else None

    if not p1:
        return {"error": f"Игрок '{player1_name}' не найден"}
    if not p2:
        return {"error": f"Игрок '{player2_name}' не найден"}

    _emit({"type": "info", "msg": f"Найдены: {p1['name']} vs {p2['name']}"})

    # Build match data from DB or API
    match_data = {
        "player1_name": p1["name"],
        "player2_name": p2["name"],
    }

    # Get rankings from DB
    with tennis_db.connect() as conn:
        r1 = conn.execute(
            "SELECT ranking FROM tennis_players WHERE id=?", (p1["id"],)
        ).fetchone()
        r2 = conn.execute(
            "SELECT ranking FROM tennis_players WHERE id=?", (p2["id"],)
        ).fetchone()
        if r1:
            match_data["player1_ranking"] = r1["ranking"]
        if r2:
            match_data["player2_ranking"] = r2["ranking"]

    _emit({"type": "info", "msg": "Генерирую AI-анализ..."})

    # Check for live match data (FlashScore / Tennis API)
    live_data = None
    try:
        import tennis_live
        live_data = tennis_live.fetch_live_context(p1["name"], p2["name"])
        if live_data:
            _emit({"type": "info", "msg": f"LIVE: {live_data.get('score_display', '?')}"})
    except Exception as e:
        _emit({"type": "info", "msg": f"Live-данные недоступны: {e}"})

    # Get ML prediction if model is loaded
    ml_prediction = None
    try:
        import os
        if os.path.exists("tennis_model.pkl"):
            import joblib
            import pandas as pd
            import numpy as np
            from tennis_trainer import build_features_fast, _build_player_stats, _build_ranking_lookup, _build_h2h, _build_form

            tennis_model = joblib.load("tennis_model.pkl")
            all_matches = tennis_db.all_finished_matches()

            player_stats = _build_player_stats(all_matches)
            rankings = _build_ranking_lookup(all_matches)
            h2h_data = _build_h2h(all_matches)
            form = _build_form(all_matches, 20)

            features = build_features_fast({}, player_stats, rankings, h2h_data, form, all_matches,
                                           p1_id=p1["id"], p2_id=p2["id"])
            X = pd.DataFrame([features], columns=tennis_model["features"])
            p1_win_prob = (
                0.5 * tennis_model["xgb"].predict_proba(X)[:, 1][0] +
                0.5 * tennis_model["lgbm"].predict_proba(X)[:, 1][0]
            )
            ml_prediction = {
                "player1_win": round(float(p1_win_prob) * 100, 1),
                "player2_win": round(float(1 - p1_win_prob) * 100, 1),
            }
            match_data["ml_prediction"] = ml_prediction
            _emit({"type": "info", "msg": f"ML-прогноз: {p1['name']} {ml_prediction['player1_win']}% vs {p2['name']} {ml_prediction['player2_win']}%"})
    except Exception as e:
        _emit({"type": "info", "msg": f"ML-модель недоступна: {e}"})

    # Generate analysis (returns plain text like football)
    raw_analysis = generate_analysis(
        p1["name"], p2["name"], match_data, model=model, live_data=live_data
    )

    if not raw_analysis:
        return {"error": "Не удалось получить ответ от AI"}

    # Parse structured data from plain text
    parsed = _parse_analysis_response(raw_analysis)

    # Save prediction
    pred_data = {
        "player1_name": p1["name"],
        "player2_name": p2["name"],
        "analysis": raw_analysis,
        "main_bet": parsed.get("main_bet", ""),
        "confidence": parsed.get("confidence", ""),
        "player1_win": parsed.get("player1_win_prob"),
        "player2_win": parsed.get("player2_win_prob"),
        "model_used": model or config.DEFAULT_AI_MODEL,
    }
    pred_id = tennis_db.save_prediction(pred_data)

    _emit({"type": "done", "msg": "Анализ готов"})

    return {
        "player1": p1,
        "player2": p2,
        "match_data": match_data,
        "analysis": raw_analysis,
        "prediction": parsed.get("prediction", ""),
        "main_bet": parsed.get("main_bet", ""),
        "confidence": parsed.get("confidence", ""),
        "player1_win_prob": parsed.get("player1_win_prob", 50),
        "player2_win_prob": parsed.get("player2_win_prob", 50),
        "ml_prediction": ml_prediction,
        "live_data": live_data,
        "prediction_id": pred_id,
    }


# ── CLI entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    import sys
    if len(sys.argv) >= 3:
        p1 = sys.argv[1]
        p2 = sys.argv[2]
    else:
        p1 = "Sinner"
        p2 = "Alcaraz"

    print(f"=== Tennis AI Analysis: {p1} vs {p2} ===")

    def _cli_cb(ev):
        print(f"  [{ev.get('type', '?')}] {ev.get('msg', '')}")

    result = search_and_analyze(p1, p2, progress_cb=_cli_cb)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"\n{result.get('analysis', 'N/A')}")
        print(f"\n---")
        print(f"Main bet: {result.get('main_bet', 'N/A')}")
        print(f"Confidence: {result.get('confidence', 'N/A')}")
        print(f"P1 win: {result.get('player1_win_prob', '?')}%")
        print(f"P2 win: {result.get('player2_win_prob', '?')}%")
