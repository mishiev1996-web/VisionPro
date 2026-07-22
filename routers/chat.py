"""
routers/chat.py — Free-form sports chat.
Single input replaces the old two-field form.
Uses the EXISTING search_and_predict() pipeline from ai_analyzer.py.
"""
from __future__ import annotations

import json
import re
import time
import random
import logging
import datetime as dt
from typing import Any, Dict, List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

import db
import helpers
import ai_analyzer
from ai_core import chat as _chat

logger = logging.getLogger("router.chat")
router = APIRouter(prefix="/api/chat", tags=["chat"])

# ── Funny loading messages ──────────────────────────────────────────────────
LOADING_MSGS = [
    "Залман, Сейчас будет жарко, подожди секунду...",
    "Залман, Закатываю рукава, считаю...",
    "Залман, Загружаю данные, не отвлекай...",
    "Залман, Мозги включаются, подожди...",
    "Залман, Работаю, уже почти...",
    "Залман, Анализирую как босс, подожди...",
    "Залман, Считаю вероятности, ща будет...",
    "Залман, Погнал, не мешай...",
]


# ── Conversation filter ──────────────────────────────────────────────────────

def _is_conversation(msg: str) -> bool:
    """Detect non-sports conversational messages."""
    msg_lower = msg.lower().strip()
    # Common conversational / non-sport patterns
    conv_patterns = [
        # Profanity/insults (short messages)
        "ебу", "хуй", "пиздец", "бля", "охуенно", "пиздато",
        # Questions not about teams
        "как дела", "привет", "пока", "спасибо", "что нового",
        # Single short word without context
    ]
    # If message is very short and contains profanity
    if len(msg_lower) < 30 and any(p in msg_lower for p in conv_patterns):
        return True
    # If message has no football-related keywords at all AND is short
    football_kws = [
        "прогноз", "матч", "команда", "гол", "победа", "ничья", "тотал",
        "фора", "кф", "коэффициент", "забьёт", "тайм", "liga", "epl",
        "футбол", "игра", "ставка", "счёт", "счет", "удар", "угловой",
        # Football team patterns
        "—", "-", "vs", "против",
    ]
    if len(msg_lower) < 40 and not any(kw in msg_lower for kw in football_kws):
        # Likely conversational
        return True
    return False


def _conversation_reply(msg: str) -> str:
    """Generate a response for conversational messages."""
    msg_lower = msg.lower().strip()
    if "привет" in msg_lower:
        return "Привет, Залман! Напиши матч — например «Спартак — ЦСКА», и я дам прогноз."
    if "ебу" in msg_lower or "хуй" in msg_lower:
        return "Сам такой) Давай по делу — пиши матч, дам прогноз."
    if "пока" in msg_lower:
        return "Давай, Залман! Возвращайся с матчами."
    if "спасибо" in msg_lower:
        return "Не за что! Пиши ещё."
    return "Я тут только про футбол. Напиши матч, например «Спартак — ЦСКА»."


# ── Intent parser ────────────────────────────────────────────────────────────

INTENT_SYSTEM = """Ты — парсер команд. Верни ТОЛЬКО JSON без комментариев.

На основе сообщения пользователя определи:
1. home_team — название команды хозяев
2. away_team — название команды гостей
3. intent: "prematch" / "live" / "1h" / "2h" / "question" / "not_sport"
4. focus: "total" / "btts" / "winner" / "general" / "next_scorer"
5. one_team: true если пользователь назвал только одну команду (ищет все матчи на сегодня)

Примеры:
Вход: "дай прогноз Омония Никосия Кайрат Алматы"
Выход: {"home_team":"Омония Никосия","away_team":"Кайрат Алматы","intent":"prematch","focus":"general","one_team":false}

Вход: "прогноз на 1 тайм Спартак ЦСКА"
Выход: {"home_team":"Спартак","away_team":"ЦСКА","intent":"1h","focus":"general","one_team":false}

Вход: "тотал больше 2.5 Челси Арсенал"
Выход: {"home_team":"Челси","away_team":"Арсенал","intent":"prematch","focus":"total","one_team":false}

Вход: "кто следующий забьёт в Реал Барселона"
Выход: {"home_team":"Реал Мадрид","away_team":"Барселона","intent":"live","focus":"next_scorer","one_team":false}

Вход: "какие матчи у Арсенала сегодня"
Выход: {"home_team":"Арсенал","away_team":null,"intent":"prematch","focus":"general","one_team":true}

Вход: "сколько будет голов в матче"
Выход: {"intent":"question","home_team":null,"away_team":null,"focus":"general","one_team":false}"""


def _parse_intent(message: str) -> dict:
    """Parse user message. First try fast regex patterns, then LLM."""
    # Fast path: simple "Team A — Team B" or "Team A Team B" patterns
    # These are the most common chat queries
    fast = _fast_parse_teams(message)
    if fast:
        return fast

    # Slow path: LLM intent parse for complex queries
    messages = [
        {"role": "system", "content": INTENT_SYSTEM},
        {"role": "user", "content": message},
    ]
    raw = _chat(messages, temperature=0.0, max_tokens=500, timeout=15)
    if not raw:
        return {"intent": "question", "home_team": None, "away_team": None,
                "focus": "general", "one_team": False}
    try:
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "question", "home_team": None, "away_team": None,
                "focus": "general", "one_team": False}


def _fast_parse_teams(message: str) -> Optional[dict]:
    """Fast regex-based team extraction for common patterns."""
    msg = message.strip()

    # Determine focus/intent from keywords FIRST (before team extraction)
    msg_lower = msg.lower()
    focus = "general"
    intent_type = "prematch"

    if "1 тайм" in msg_lower or "первый тайм" in msg_lower:
        intent_type = "1h"
    elif "2 тайм" in msg_lower or "второй тайм" in msg_lower:
        intent_type = "2h"

    if "тотал" in msg_lower or "total" in msg_lower or "over" in msg_lower:
        focus = "total"
    elif "btts" in msg_lower or "обе забьют" in msg_lower:
        focus = "btts"
    elif "кто забьёт" in msg_lower or "кто следующий" in msg_lower:
        focus = "next_scorer"
        intent_type = "live"

    # Remove common prefixes to find team names
    clean = msg
    for prefix in ["дай прогноз на матч", "дай прогноз", "прогноз на матч",
                   "прогноз на", "прогноз", "матч", "считай", "покажи",
                   "какие матчи у", "какие матчи", "матчи у",
                   "кто забьёт следующим в", "кто забьёт в", "кто следующий в"]:
        clean = re.sub(rf'^{prefix}\s+', '', clean, flags=re.IGNORECASE).strip()

    # Remove intent keywords from clean text (so "1 тайм Спартак ЦСКА" → "Спартак ЦСКА")
    for kw in ["1 тайм", "первый тайм", "2 тайм", "второй тайм",
               "тотал больше", "тотал меньше", "тотал", "btts", "обе забьют",
               "кто забьёт", "кто следующий", "сегодня", "завтра"]:
        clean = re.sub(rf'\b{kw}\b\s*', '', clean, flags=re.IGNORECASE).strip()

    # Remove standalone numbers (odds like "2.5")
    clean = re.sub(r'\b\d+\.\d+\b\s*', '', clean).strip()

    # Pattern 1: "Team A — Team B" or "Team A - Team B" or "Team A vs Team B" or "Team A против Team B"
    for sep in [" — ", " – ", " vs ", " VS ", " - ", " против "]:
        if sep in clean:
            parts = clean.split(sep, 1)
            if len(parts) == 2:
                home = parts[0].strip()
                away = parts[1].strip()
                if home and away and len(home) > 2 and len(away) > 2:
                    return {"intent": intent_type, "home_team": home, "away_team": away,
                            "focus": focus, "one_team": False}

    # Pattern 2: "Team A Team B" (two team names separated by space, no other separator)
    # Check clean (after prefix removal) not msg
    clean_lower = clean.lower()
    if not any(kw in clean_lower for kw in ["как", "что", "сколько", "где", "когда", "кто", "забьёт"]):
        words = clean.split()
        if len(words) == 2 and all(len(w) > 2 for w in words):
            home = words[0].strip()
            away = words[1].strip()
            combined = home + " " + away
            # Common multi-word teams that shouldn't be split
            multiword = ["Реал Мадрид", "Атлетико Мадрид", "Манчестер Юнайтед",
                         "Манчестер Сити", "Тоттенхэм Хотспур", "Вест Хэм",
                         "Боруссия Дортмунд", "Боруссия Мёнхенгладбах",
                         "Пари Сен-Жермен", "Олимпик Лион", "Олимпик Марсель",
                         "Интер Милан", "Ювентус Турин", "Милан",
                         "Ливерпуль", "Эвертон", "Лестер Сити", "Вулверхэмптон"]
            is_multiword = any(mw.lower() in combined.lower() for mw in multiword)
            if not is_multiword and len(home) > 2 and len(away) > 2:
                return {"intent": intent_type, "home_team": home, "away_team": away,
                        "focus": focus, "one_team": False}

    # Pattern 3: Single team (for today's matches search)
    # Only if it looks like just a team name (2-4 words, no question words)
    if not any(kw in clean_lower for kw in ["как", "что", "сколько", "где", "когда", "кто", "забьёт"]):
        words = clean.split()
        if 1 <= len(words) <= 4 and all(len(w) > 1 for w in words):
            # Check if this is likely a team name (not a question)
            if not any(w.lower() in ["матч", "прогноз", "счёт", "счет", "тотал", "btts"] for w in words):
                team_name = " ".join(words)
                # Strip Russian genitive suffixes ONLY from common Russian team names
                # Not from proper nouns like "Барселона", "Мадрид"
                russian_teams = ["Спартак", "ЦСКА", "Динамо", "Локомотив", "Зенит", "Краснодар",
                                 "Рубин", "Ростов", "Сочи", "Урал", "Ахмат", "Оренбург",
                                 "Нижний Новгород", "Пари НН", "Факел", "Торпедо", "Балтика",
                                 "Самара", "Крылья Советов", "Арсенал", "Манчестер", "Ливерпуль"]
                for team in russian_teams:
                    if team_name.lower().startswith(team.lower()):
                        for suffix in ["а", "у", "е", "ой", "ей", "ом", "ем", "ам", "ям"]:
                            if team_name.endswith(suffix) and len(team_name) > len(suffix) + 2:
                                team_name = team_name[:-len(suffix)]
                                break
                        break
                if len(team_name) > 3:
                    return {"intent": intent_type, "home_team": team_name, "away_team": None,
                            "focus": focus, "one_team": True}

    return None


# ── Endpoint ─────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str


@router.post("")
def api_chat(req: ChatRequest):
    t0 = time.time()
    msg = req.message.strip()

    logger.info(f"[chat] Incoming: {msg[:80]}")

    try:
        return _api_chat_inner(msg, t0)
    except Exception as e:
        logger.error(f"[chat] Unhandled error: {e}", exc_info=True)
        return {
            "reply": "Произошла ошибка при обработке запроса. Попробуй ещё раз.",
            "type": "text",
            "parse_time": round(time.time() - t0, 2),
        }


def _api_chat_inner(msg: str, t0: float):
    # Step 0: Conversation filter — profanity, greetings, non-sport
    if _is_conversation(msg):
        reply = _conversation_reply(msg)
        logger.info(f"[chat] Conversation detected, replying: {reply[:50]}")
        return {
            "reply": reply,
            "type": "text",
            "parse_time": round(time.time() - t0, 2),
        }

    # Step 1: Parse intent
    try:
        intent = _parse_intent(msg)
        logger.info(f"[chat] Intent: {intent}")
    except Exception as e:
        logger.error(f"[chat] Intent parse failed: {e}")
        intent = {"intent": "question", "home_team": None, "away_team": None,
                  "focus": "general", "one_team": False}

    intent_type = intent.get("intent", "question")

    # Not sport
    if intent_type == "not_sport":
        return {
            "reply": "Это не про спорт) Я тут только про футбол — спроси про матч или команду!",
            "type": "text",
            "parse_time": round(time.time() - t0, 2),
        }

    # Step 2: Resolve teams
    home_name = intent.get("home_team")
    away_name = intent.get("away_team")
    one_team = intent.get("one_team", False)

    # If user named only one team — show today's matches
    if (one_team or (home_name and not away_name)) and home_name:
        candidates = _search_team_all_sources(home_name)
        if not candidates:
            return {
                "reply": f"Не нашёл команду «{home_name}» ни в базе, ни в sstats. Проверь название и попробуй ещё раз.",
                "type": "text",
                "parse_time": round(time.time() - t0, 2),
            }
        if len(candidates) > 1:
            return {
                "reply": f"Какую «{home_name}» ты имеешь в виду?",
                "type": "suggestions",
                "suggestions": [{"id": c["id"], "name": c["name"], "league": c.get("league_slug", "")}
                               for c in candidates[:6]],
                "parse_time": round(time.time() - t0, 2),
            }
        team = candidates[0]
        matches = _find_today_matches(team["id"])
        if not matches:
            return {
                "reply": f"У {team['name']} сегодня матчей нет. Попробуй завтра или поищи другой матч.",
                "type": "text",
                "parse_time": round(time.time() - t0, 2),
            }
        # If exactly one match — predict it
        if len(matches) == 1:
            m = matches[0]
            home_id = m["home_id"]
            away_id = m["away_id"]
            return _run_prediction(home_id, away_id, intent, msg, t0)
        # Multiple matches — show list
        return {
            "reply": f"У {team['name']} сегодня несколько матчей:",
            "type": "match_list",
            "matches": [{"id": m["id"], "home": m["home_name"], "away": m["away_name"],
                         "date": m["date"][:16], "league": m.get("league_slug", "")}
                        for m in matches],
            "parse_time": round(time.time() - t0, 2),
        }

    # If no team names at all — general question
    if not home_name and not away_name:
        messages = [
            {"role": "system", "content": "Ты — футбольный аналитик VisionPro. Отвечай кратко, по-русски, по делу."},
            {"role": "user", "content": msg},
        ]
        reply = _chat(messages, temperature=0.3, max_tokens=500, timeout=15)
        return {
            "reply": reply or "Спроси про конкретный матч — и я дам прогноз!",
            "type": "text",
            "parse_time": round(time.time() - t0, 2),
        }

    # Resolve home team using the FULL pipeline (DB + transliteration + sstats API)
    home_team = _resolve_team_full(home_name) if home_name else None
    if home_name and not home_team:
        candidates = _search_team_all_sources(home_name)
        if candidates:
            return {
                "reply": f"Не нашёл точное совпадение для «{home_name}». Ты имел в виду:",
                "type": "suggestions",
                "suggestions": [{"id": c["id"], "name": c["name"], "league": c.get("league_slug", "")}
                               for c in candidates[:8]],
                "parse_time": round(time.time() - t0, 2),
            }
        return {
            "reply": f"Не нашёл команду «{home_name}» ни в базе, ни в sstats. Проверь название и попробуй ещё раз — может по-другому называется?",
            "type": "text",
            "parse_time": round(time.time() - t0, 2),
        }

    # Resolve away team using the FULL pipeline (DB + transliteration + sstats API)
    away_team = _resolve_team_full(away_name) if away_name else None
    if away_name and not away_team:
        candidates = _search_team_all_sources(away_name)
        if candidates:
            return {
                "reply": f"Не нашёл точное совпадение для «{away_name}». Ты имел в виду:",
                "type": "suggestions",
                "suggestions": [{"id": c["id"], "name": c["name"], "league": c.get("league_slug", "")}
                               for c in candidates[:8]],
                "parse_time": round(time.time() - t0, 2),
            }
        return {
            "reply": f"Не нашёл команду «{away_name}» ни в базе, ни в sstats. Проверь название и попробуй ещё раз.",
            "type": "text",
            "parse_time": round(time.time() - t0, 2),
        }

    if not home_team or not away_team:
        logger.info(f"[chat] Missing teams: home={home_team}, away={away_team}")
        return {
            "reply": "Не удалось определить обе команды. Напиши названия точнее.",
            "type": "text",
            "parse_time": round(time.time() - t0, 2),
        }

    # Both teams found — run prediction
    logger.info(f"[chat] Running prediction: {home_team['name']} vs {away_team['name']}")
    return _run_prediction(home_team["id"], away_team["id"], intent, msg, t0)


def _resolve_team_full(name: str) -> Optional[dict]:
    """Full team resolution: DB + transliteration + sstats API search.
    Uses the EXISTING ai_analyzer._find_team_in_db() + _search_sstats_team() pipeline.
    """
    if not name:
        return None

    # 1. Try ai_analyzer's full DB search (includes transliteration + TEAM_NAME_MAP)
    result = ai_analyzer._find_team_in_db(name)
    if result:
        logger.info(f"[chat] Team '{name}' found in DB: {result.get('name')} (id={result.get('id')})")
        return result

    # 2. Try sstats.net live API search (saves team + matches to DB)
    logger.info(f"[chat] Team '{name}' not in DB, trying sstats.net...")
    try:
        result = ai_analyzer._search_sstats_team(name, progress_cb=None, max_seconds=15.0)
        if result:
            team = db.get_team(result["team_id"])
            if team:
                logger.info(f"[chat] Team '{name}' found in sstats: {team['name']} (id={team['id']})")
                return team
    except Exception as e:
        logger.error(f"[chat] sstats search failed: {e}")

    return None


def _search_team_all_sources(name: str) -> List[dict]:
    """Search team candidates across DB for suggestions.
    Returns list of team dicts (id, name, league_slug).
    NOTE: sstats is NOT searched here — it was already tried in _resolve_team_full.
    """
    if not name:
        return []

    # 1. Try DB fuzzy search (uses db.search_team_fuzzy which prioritizes by match count)
    results = db.search_team_fuzzy(name, limit=8)
    if results:
        return results

    # 2. Try transliterated name in DB
    try:
        import ai_analyzer
        en_name = ai_analyzer._transliterate_ru_to_en(name)
        if en_name != name:
            results = db.search_team_fuzzy(en_name, limit=8)
            if results:
                return results
    except Exception:
        pass

    return []


def _find_today_matches(team_id: int) -> List[dict]:
    """Find today's matches for a team."""
    today = dt.date.today().isoformat()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT m.*, th.name AS home_name, ta.name AS away_name "
            "FROM matches m "
            "JOIN teams th ON th.id = m.home_id "
            "JOIN teams ta ON ta.id = m.away_id "
            "WHERE (m.home_id = ? OR m.away_id = ?) AND m.date LIKE ? "
            "ORDER BY m.date",
            (team_id, team_id, f"{today}%")
        ).fetchall()
        return [dict(r) for r in rows]


def _run_prediction(home_id: int, away_id: int, intent: dict, user_msg: str, t0: float) -> dict:
    """Run the full prediction pipeline using ai_analyzer.generate_preview()."""
    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        return {"reply": "Команда не найдена в базе.", "type": "text"}

    focus = intent.get("focus", "general")
    intent_type = intent.get("intent", "prematch")

    logger.info(f"[chat] Running generate_preview: {home['name']} vs {away['name']}")

    # Use the EXISTING generate_preview pipeline (includes ML + sstats enrichment + AI analysis)
    try:
        result = ai_analyzer.generate_preview(home_id, away_id)
    except Exception as e:
        logger.error(f"[chat] generate_preview failed: {e}")
        # Fallback to predict_pair (ML only, no sstats enrichment)
        try:
            prediction = helpers.predict_pair(home_id, away_id, home, away)
            return {
                "reply": f"Прогноз: {home['name']} vs {away['name']}",
                "type": "analysis",
                "teams": {"home": home["name"], "away": away["name"]},
                "prediction": prediction.get("probabilities"),
                "intent": intent_type,
                "focus": focus,
                "parse_time": round(time.time() - t0, 2),
            }
        except Exception as e2:
            logger.error(f"[chat] predict_pair fallback failed: {e2}")
            return {"reply": f"Ошибка расчёта: {e2}", "type": "text"}

    analysis = result.get("analysis", "")
    prediction = result.get("prediction")

    loading_msg = random.choice(LOADING_MSGS)

    return {
        "reply": analysis or f"Прогноз: {home['name']} vs {away['name']}",
        "type": "analysis",
        "loading_msg": loading_msg,
        "teams": {"home": home["name"], "away": away["name"]},
        "prediction": prediction.get("probabilities") if prediction else None,
        "intent": intent_type,
        "focus": focus,
        "match_info": result.get("match_info"),
        "parse_time": round(time.time() - t0, 2),
    }
