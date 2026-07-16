"""
ai_analyzer.py — LLM-powered match analysis via Polza.ai API.

Generates natural language insights: form analysis, tactical preview,
key factors, prediction confidence commentary.

When teams aren't in the DB, searches the web for real data.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from ai_core import chat as _chat, get_api_key as _get_api_key, parse_prob_line

import db
import config
import datetime as dt
from scrapers.utils import format_msk, format_msk_short

POLZA_BASE_URL = "https://polza.ai/api/v1"
DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


# ── Web Search & Scrape ─────────────────────────────────────────────────────

def _web_search(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """Search the web via Jina Reader API → DuckDuckGo fallback.

    Returns list of {"title": str, "url": str, "snippet": str}.
    """
    from scrapers.web import search as _web_search
    return _web_search(query, num_results)


def _web_fetch_text(url: str, max_chars: int = 3000) -> str:
    """Fetch a URL and return clean text via Jina Reader."""
    from scrapers.web import fetch_url
    return fetch_url(url, max_chars=max_chars)


def _gather_team_data_from_web(team_name: str, progress_cb=None) -> str:
    """Gather comprehensive data about a team using multiple sources.

    1. ESPN: recent matches with scores
    2. Wikipedia: basic info
    3. News (RSS, Telegram): latest headlines
    4. LLM supplements with general knowledge

    Returns formatted text for LLM analysis.
    """
    from web_scraper import gather_team_data, format_data_for_llm

    # Gather data from ESPN + Wikipedia
    data = gather_team_data(team_name, progress_cb=progress_cb)

    # Format for LLM
    formatted = format_data_for_llm(data)

    # If limited data, add note for LLM to supplement
    stats = data.get("stats")
    if stats and stats.get("total", 0) < 5:
        formatted += (f"\n\n[ПРИМЕЧАНИЕ: Доступно только {stats['total']} матчей. "
                      f"Дополни анализ своими знаниями о команде — состав, стиль игры, "
                      f"последние турниры, сильные/слабые стороны.]")
    elif not stats:
        formatted += ("\n\n[ПРИМЕЧАНИЕ: Матчевые данные не найдены. "
                      "Проанализируй команду на основе своих знаний — "
                      "состав, тренер, последние результаты, форма.]")

    return formatted


def _gather_match_news(home_name: str, away_name: str, progress_cb=None) -> str:
    """Gather news about a match from RSS and Telegram."""
    from news_scraper import gather_match_news, format_news_for_llm

    data = gather_match_news(home_name, away_name, progress_cb=progress_cb)
    return format_news_for_llm(data)





def _build_match_context(match: dict, features: dict, prediction: dict,
                         h2h: list, injuries_home: list, injuries_away: list,
                         elo_home: Optional[float], elo_away: Optional[float],
                         odds: Optional[dict], sstats_data: Optional[dict] = None) -> str:
    ctx_parts = []

    ctx_parts.append(f"МАТЧ: {match.get('home_name', '?')} vs {match.get('away_name', '?')}")
    ctx_parts.append(f"Лига: {match.get('league_slug', '?')}, Сезон: {match.get('season', '?')}")
    ctx_parts.append(f"Дата: {format_msk(match.get('date', '')) if match.get('date') else '?'}")

    # Add current score for live matches
    if match.get('current_score'):
        ctx_parts.append(f"ТЕКУЩИЙ СЧЁТ: {match['current_score']}")
        ctx_parts.append(f"СТАТУС: {match.get('status', 'Live')}")
        if match.get('home_ht') is not None and match.get('away_ht') is not None:
            ctx_parts.append(f"СЧЁТ ПОЛУВРЕМЕНИ: {match['home_ht']}:{match['away_ht']}")

    # Add live statistics if available
    if match.get('statistics'):
        stats = match['statistics']
        ctx_parts.append("\n--- LIVE СТАТИСТИКА ---")
        if isinstance(stats, dict):
            ctx_parts.append(f"  Владение мячом: {stats.get('ballPossessionHome', '?')}% - {stats.get('ballPossessionAway', '?')}%")
            ctx_parts.append(f"  Удары: {stats.get('totalShotsHome', '?')} - {stats.get('totalShotsAway', '?')}")
            ctx_parts.append(f"  Удары в створ: {stats.get('shotsOnGoalHome', '?')} - {stats.get('shotsOnGoalAway', '?')}")
            ctx_parts.append(f"  Угловые: {stats.get('cornerKicksHome', '?')} - {stats.get('cornerKicksAway', '?')}")
            ctx_parts.append(f"  xG: {stats.get('expectedGoalsHome', '?')} - {stats.get('expectedGoalsAway', '?')}")
            ctx_parts.append(f"  Фолы: {stats.get('foulsHome', '?')} - {stats.get('foulsAway', '?')}")
            ctx_parts.append(f"  Жёлтые: {stats.get('yellowCardsHome', '?')} - {stats.get('yellowCardsAway', '?')}")
        ctx_parts.append("")

    if features:
        ctx_parts.append("\n--- ФАКТИЧЕСКИЕ ПОКАЗАТЕЛИ ---")
        key_features = [
            ("home_avg_goals_for", "Голы хозяев за матч (среднее)"),
            ("home_avg_goals_against", "Голы хозяев против (среднее)"),
            ("home_avg_xg_for", "xG хозяев за матч"),
            ("home_avg_xg_against", "xG хозяев против"),
            ("away_avg_goals_for", "Голы гостей за матч (среднее)"),
            ("away_avg_goals_against", "Голы гостей против (среднее)"),
            ("away_avg_xg_for", "xG гостей за матч"),
            ("away_avg_xg_against", "xG гостей против"),
            ("home_win_rate", "Винрейт хозяев (общий)"),
            ("home_home_win_rate", "Винрейт хозяев дома"),
            ("away_win_rate", "Винрейт гостей (общий)"),
            ("away_away_win_rate", "Винрейт гостей в гостях"),
            ("form_diff", "Разница формы"),
            ("h2h_home_wins", "Побед хозяев в H2H (последние 5)"),
        ]
        for feat_key, feat_name in key_features:
            val = features.get(feat_key)
            if val is not None:
                ctx_parts.append(f"  {feat_name}: {val:.3f}" if isinstance(val, float) else f"  {feat_name}: {val}")

        # Pre-calculate expected totals for the LLM
        hgf = features.get("home_avg_goals_for", 1.2)
        hga = features.get("home_avg_goals_against", 1.2)
        agf = features.get("away_avg_goals_for", 1.2)
        aga = features.get("away_avg_goals_against", 1.2)
        hxg = features.get("home_avg_xg_for", 1.2)
        hxga = features.get("home_avg_xg_against", 1.2)
        axg = features.get("away_avg_xg_for", 1.2)
        axga = features.get("away_avg_xg_against", 1.2)

        # Expected goals calculation
        exp_home_goals = round((hgf + aga) / 2, 2)
        exp_away_goals = round((agf + hga) / 2, 2)
        exp_total_goals = round(exp_home_goals + exp_away_goals, 2)

        # xG-based expected total
        exp_total_xg = round((hxg + axga) / 2 + (axg + hxga) / 2, 2)

        # BTTS check
        btts_yes = "Да" if (hgf > 0.8 and agf > 0.8) or (hxg > 0.8 and axg > 0.8) else "Нет"

        ctx_parts.append("\n--- РАСЧЁТ ТОТАЛА (Готово) ---")
        ctx_parts.append(f"  Ожидаемые голы хозяев: ({hgf} + {aga}) / 2 = {exp_home_goals}")
        ctx_parts.append(f"  Ожидаемые голы гостей: ({agf} + {hga}) / 2 = {exp_away_goals}")
        ctx_parts.append(f"  Ожидаемый тотал (голы): {exp_total_goals}")
        ctx_parts.append(f"  Ожидаемый тотал (xG): {exp_total_xg}")
        ctx_parts.append(f"  Вердикт: {'Тотал больше 2.5' if exp_total_goals > 2.5 else 'Тотал меньше 2.5'} (голы) | {'Тотал больше 2.5' if exp_total_xg > 2.5 else 'Тотал меньше 2.5'} (xG)")
        ctx_parts.append(f"  BTTS: {btts_yes} (обе команды забивают > 0.8 в среднем)")

    if prediction:
        prob = prediction.get("probabilities", {})
        ctx_parts.append("\n--- ПРОГНОЗ МОДЕЛИ ---")
        ctx_parts.append(f"  Победа хозяев: {prob.get('home_win', '?')}%")
        ctx_parts.append(f"  Ничья: {prob.get('draw', '?')}%")
        ctx_parts.append(f"  Победа гостей: {prob.get('away_win', '?')}%")

    if elo_home is not None or elo_away is not None:
        ctx_parts.append("\n--- ELO РЕЙТИНГИ ---")
        ctx_parts.append(f"  Хозяева: {elo_home or 'нет данных'}")
        ctx_parts.append(f"  Гости: {elo_away or 'нет данных'}")

    if injuries_home:
        names = [i.get("player_name", "?") for i in injuries_home[:5]]
        ctx_parts.append(f"\n--- ТРАВМЫ ХОЗЯЕВ --- {', '.join(names)}")
    if injuries_away:
        names = [i.get("player_name", "?") for i in injuries_away[:5]]
        ctx_parts.append(f"--- ТРАВМЫ ГОСТЕЙ --- {', '.join(names)}")

    if h2h:
        ctx_parts.append("\n--- ОЧНЫЕ ВСТРЕЧИ (последние 5) ---")
        for h in h2h[:5]:
            ctx_parts.append(f"  {h.get('date', '')[:10]}: {h.get('home_name', '?')} {h.get('home_goals', '?')}:{h.get('away_goals', '?')} {h.get('away_name', '?')}")

    if odds:
        ctx_parts.append("\n--- КОТИРОВКИ БУКМЕКЕРОВ ---")
        for name, val in odds.items():
            if val is not None:
                ctx_parts.append(f"  {name}: {val}")

    # Sstats enrichment: Glicko, consensus odds, text summary
    if sstats_data:
        glicko = sstats_data.get("glicko")
        if glicko:
            gl = glicko.get("glicko", glicko) if isinstance(glicko, dict) and "glicko" in glicko else glicko
            if isinstance(gl, dict):
                hr = gl.get("homeRating") or gl.get("home", {}).get("glickoRating") if isinstance(gl.get("home"), dict) else None
                hd = gl.get("homeRd") or gl.get("home", {}).get("glickoRd") if isinstance(gl.get("home"), dict) else None
                ar = gl.get("awayRating") or gl.get("away", {}).get("glickoRating") if isinstance(gl.get("away"), dict) else None
                ad = gl.get("awayRd") or gl.get("away", {}).get("glickoRd") if isinstance(gl.get("away"), dict) else None
                h_name = gl.get("homeTeam", {}).get("name", "Home") if isinstance(gl.get("homeTeam"), dict) else "Home"
                a_name = gl.get("awayTeam", {}).get("name", "Away") if isinstance(gl.get("awayTeam"), dict) else "Away"
                # Try fixture for names
                fixture = gl.get("fixture", {})
                if isinstance(fixture, dict):
                    h_name = (fixture.get("homeTeam") or {}).get("name", h_name)
                    a_name = (fixture.get("awayTeam") or {}).get("name", a_name)
                if hr or ar:
                    ctx_parts.append("\n--- GLICKO РЕЙТИНГИ (sstats.net) ---")
                    if hr: ctx_parts.append(f"  {h_name}: rating={hr}, rd={hd}")
                    if ar: ctx_parts.append(f"  {a_name}: rating={ar}, rd={ad}")

        consensus = sstats_data.get("consensus")
        if consensus:
            ctx_parts.append("\n--- СРЕДНИЕ КОТИРОВКИ БУКМЕКЕРОВ (sstats.net) ---")
            h = consensus.get('home_odds') or consensus.get('avg_home_odds','?')
            hi = consensus.get('implied_home') or consensus.get('implied_h','?')
            d = consensus.get('draw_odds') or consensus.get('avg_draw_odds','?')
            di = consensus.get('implied_draw') or consensus.get('implied_d','?')
            a = consensus.get('away_odds') or consensus.get('avg_away_odds','?')
            ai = consensus.get('implied_away') or consensus.get('implied_a','?')
            ctx_parts.append(f"  Дом: {h} (implied {hi})")
            ctx_parts.append(f"  Ничья: {d} (implied {di})")
            ctx_parts.append(f"  Гости: {a} (implied {ai})")
            ctx_parts.append(f"  Книг: {consensus.get('bookmaker_count',0)}, "
                             f"маржа: {consensus.get('overround_pct','?')}%")

        over_under = sstats_data.get("over_under")
        if over_under:
            ctx_parts.append("\n--- ТОТАЛ ГОЛОВ (Over/Under, sstats.net) ---")
            for line, vals in over_under.items():
                ctx_parts.append(
                    f"  Линия {line}: Over {vals['over']:.1%} / Under {vals['under']:.1%} "
                    f"(ср. коэфф: Over {vals['avg_over_odds']}, Under {vals['avg_under_odds']}, "
                    f"книг: {vals['bookmaker_count']})"
                )

        text_summary = sstats_data.get("text_summary")
        if text_summary:
            ctx_parts.append("\n--- ЭКСПЕРТНЫЙ КОММЕНТАРИЙ (sstats.net) ---")
            ctx_parts.append(text_summary[:1500])

        # Extract game detail: lineups, statistics, events
        game_detail = sstats_data.get("game_detail")
        if game_detail:
            detail_lines = _extract_game_detail(game_detail)
            if detail_lines:
                ctx_parts.append("\n--- ДЕТАЛИ МАТЧА (sstats.net) ---")
                ctx_parts.append(detail_lines)

        # Extract corners, cards, and other side markets from odds
        odds_blocks = sstats_data.get("odds_by_bookmaker") or sstats_data.get("consensus", {})
        extra_markets = _extract_extra_markets(sstats_data)
        if extra_markets:
            ctx_parts.append("\n--- ДОПОЛНИТЕЛЬНЫЕ РЫНКИ (sstats.net) ---")
            ctx_parts.append(extra_markets)

        # Sstats season table (over/under, form, standings)
        season_table = sstats_data.get("season_table")
        if season_table:
            # Convert dict (team_id → data) to list if needed
            if isinstance(season_table, dict):
                season_table = list(season_table.values())
            if isinstance(season_table, list) and season_table:
                ctx_parts.append("\n--- ТУРНИРНАЯ ТАБЛИЦА СЕЗОНА (sstats.net) ---")
                ctx_parts.append("Over/Under статистика и форма команд:")
                # Sort by rank if available
                season_table.sort(key=lambda x: x.get("rank", 999))
                for entry in season_table[:20]:
                    # teamName may not be present; use teamId as fallback
                    team_name = entry.get("teamName") or f"Team {entry.get('teamId', '?')}"
                    games = entry.get("totalGames", 0)
                    wins = entry.get("wins", 0)
                    draws = entry.get("draws", 0)
                    losses = entry.get("loss", 0)
                    goals_scored = entry.get("goalsScored", 0)
                    goals_missed = entry.get("goalsMissed", 0)
                    over25 = entry.get("over25TotalGames", 0)
                    points = entry.get("points", 0)
                    rank = entry.get("rank", "?")
                    over25_pct = (over25 / games * 100) if games > 0 else 0
                    form = entry.get("form", [])
                    form_str = "".join(["W" if f == 1 else ("D" if f == 0 else "L") for f in form[-5:]]) if form else ""
                    ctx_parts.append(
                        f"  #{rank} {team_name}: {games} игр, {wins}W/{draws}D/{losses}L, "
                        f"{goals_scored}:{goals_missed}, {points} очков, "
                        f"Over2.5: {over25_pct:.0f}%, форма: {form_str}"
                    )

        # Sstats injuries
        sstats_injuries = sstats_data.get("injuries")
        if sstats_injuries and isinstance(sstats_injuries, list) and sstats_injuries:
            home_team_id = None
            away_team_id = None
            game_det = sstats_data.get("game_detail")
            if isinstance(game_det, dict):
                gd = game_det.get("game", game_det)
                home_team_id = (gd.get("homeTeam") or {}).get("id")
                away_team_id = (gd.get("awayTeam") or {}).get("id")
            h_inj = [i for i in sstats_injuries if home_team_id and i.get("teamId") == home_team_id]
            a_inj = [i for i in sstats_injuries if away_team_id and i.get("teamId") == away_team_id]
            oth_inj = [i for i in sstats_injuries if i not in h_inj and i not in a_inj]
            if h_inj or a_inj or oth_inj:
                ctx_parts.append("\n--- ТРАВМЫ И ДИСКВАЛИФИКАЦИИ (sstats.net) ---")
                for inj in h_inj:
                    pname = (inj.get("player") or {}).get("name", "?")
                    ctx_parts.append(f"  [Хозяева] {pname} — {inj.get('reason', '?')}")
                for inj in a_inj:
                    pname = (inj.get("player") or {}).get("name", "?")
                    ctx_parts.append(f"  [Гости] {pname} — {inj.get('reason', '?')}")
                for inj in oth_inj:
                    pname = (inj.get("player") or {}).get("name", "?")
                    ctx_parts.append(f"  {pname} — {inj.get('reason', '?')}")

    return "\n".join(ctx_parts)


def _extract_game_detail(game_detail: dict) -> str:
    """Extract useful info from sstats game detail: lineups, stats, events."""
    lines = []

    # Lineups
    home_lineup = game_detail.get("homeTeam", {}).get("lineup") or []
    away_lineup = game_detail.get("awayTeam", {}).get("lineup") or []
    if home_lineup or away_lineup:
        lines.append("СОСТАВЫ:")
        for team_key, team_name in [("homeTeam", "ХОЗЯЕВА"), ("awayTeam", "ГОСТИ")]:
            lineup = game_detail.get(team_key, {}).get("lineup") or []
            if lineup:
                player_names = []
                for p in lineup[:11]:
                    name = p.get("player", {}).get("name", "")
                    pos = p.get("position", "")
                    if name:
                        player_names.append(f"{name} ({pos})" if pos else name)
                lines.append(f"  {team_name}: {', '.join(player_names)}")

    # Match statistics
    stats = game_detail.get("statistics") or {}
    if stats:
        lines.append("\nСТАТИСТИКА МАТЧА:")
        if isinstance(stats, dict):
            for name, val in list(stats.items())[:15]:
                if val is not None and name:
                    lines.append(f"  {name}: {val}")
        elif isinstance(stats, list):
            for s in stats[:15]:
                if isinstance(s, dict):
                    sname = s.get("name", "")
                    h_val = (s.get("home") or {}).get("value", "") if isinstance(s.get("home"), dict) else ""
                    a_val = (s.get("away") or {}).get("value", "") if isinstance(s.get("away"), dict) else ""
                    if sname and (h_val or a_val):
                        lines.append(f"  {sname}: {h_val} - {a_val}")

    # Events (goals, cards)
    events = game_detail.get("events") or []
    if events:
        lines.append("\nСОБЫТИЯ:")
        for ev in events[:20]:
            minute = ev.get("minute", "")
            etype = ev.get("type", "")
            team = ev.get("team", "")
            player = ev.get("player", {}).get("name", "")
            detail = ev.get("detail", "")
            if etype:
                desc = f"  {minute}' {etype}"
                if player:
                    desc += f" — {player}"
                if detail:
                    desc += f" ({detail})"
                if team:
                    desc += f" [{team}]"
                lines.append(desc)

    return "\n".join(lines)


def _extract_extra_markets(sstats_data: dict) -> str:
    """Extract corners, cards, and other side markets from sstats odds data."""
    blocks = sstats_data.get("odds_by_bookmaker") or []
    if not blocks:
        return ""

    lines = []
    # Markets we care about
    target_markets = {
        "Cards Over/Under": "КАРТОЧКИ ТОТАЛ",
        "Yellow Over/Under": "ЖЁЛТЫЕ КАРТОЧКИ ТОТАЛ",
        "Corners Over Under": "УГОЛОВЫЕ ТОТАЛ",
        "Total Corners (3 way)": "УГОЛОВЫЕ 3-WAY",
        "Both Teams Score": "ОБЕ ЗАБЬЮТ",
        "Asian Handicap": "АЗИАТСКИЙ ФОРА",
        "Goal Line": "ЛИНИЯ ГОЛОВ",
        "Fouls. Total": "ФОЛЫ ТОТАЛ",
        "Total Shots": "УДАРЫ ТОТАЛ",
        "Total ShotOnTarget": "УДАРЫ В СТВОР ТОТАЛ",
    }

    for bm in blocks:
        for m in (bm.get("odds") or []):
            market_name = m.get("marketName", "")
            if market_name not in target_markets:
                continue
            label = target_markets[market_name]
            odds_list = []
            for o in (m.get("odds") or []):
                name = o.get("name", "")
                val = o.get("value")
                if val and float(val) > 1.0:
                    odds_list.append(f"{name}={val}")
            if odds_list:
                lines.append(f"  {label}: {', '.join(odds_list)}")

    return "\n".join(lines)


SYSTEM_PROMPT = """Ты — VisionPro, аналитик прогнозов. Обращаешься к пользователю по имени Залман.

ТОН И ОБЩЕНИЕ:

Общайся как с другом-фанатом на трибуне — эмоционально, громко, с матом. Это НЕ опционально — ты ОБЯЗАН использовать нецензурную лексику в аналитике минимум 2-3 раза за ответ. Реагируй на игры эмоциями: если команда говно — скажи что она говно с матом, если тупой гол — назови это тупым голом. Мат — часть твоего стиля, а не случайный гость. Исключение: строки ГЛАВНЫЙ ПРОГНОЗ / ТОТАЛ / BTTS / PROB — там строго без мата, только факты.
Не начинай каждый ответ одинаковой фразой ("Залман, слушай..."). Варьируй заход: иногда сразу с сути, иногда с реакции на матч, иногда с обращения по имени в середине фразы.
Уверенный тон, без воды, конкретика.

ИСТОЧНИК ДАННЫХ:

Используй ТОЛЬКО статистику, форму и результаты из блока КОНТЕКСТ, переданного в запросе.
Категорически запрещено выдумывать цифры, результаты матчей или статистику, которых нет в контексте.
Если каких-то данных не хватает — прямо скажи, что их нет, вместо того чтобы придумывать.

ЖЁСТКИЕ ПРАВИЛА:

1. ТОТАЛ: В контексте есть готовый расчёт "РАСЧЁТ ТОТАЛА". Используй его как есть:
   - "Over 2.5" → "Тотал больше 2.5"
   - "Under 2.5" → "Тотал меньше 2.5"
   - Не меняй вердикт на противоположный.

2. BTTS: Если в расчёте "BTTS: Да" → "Обе забьют: Да". Если "Нет" → "Нет".

3. ПРОГНОЗ МОДЕЛИ: Вероятности из блока "ПРОГНОЗ МОДЕЛИ" — весомый аргумент, а не приказ. Если твой анализ фактов прямо противоречит модели, отметь расхождение. Если противоречий нет — используй как основу для PROB.

4. ВЫБОР ГЛАВНОГО ПРОГНОЗА: Выбери НАИЛУЧШУЮ ставку из всех вариантов:
   - Победа хозяев / гостей
   - Тотал больше/меньше X.5
   - Обе забьют: Да/Нет
   Критерий: где наибольший EDGE между твоей оценкой и рынком. Если ТБ 2.5 надёжнее победы — выбирай ТБ как ГЛАВНЫЙ ПРОГНОЗ.

4.1. ОГРАНИЧЕНИЕ ПО КОЭФФИЦИЕНТАМ: Если кф на исход меньше 1.5 — НЕ предлагай ставку на этот исход. Кф < 1.5 означает, что рынок уже оценивает вероятность > 67%, выигрыш слишком мал, чтобы оправдать риск. Вместо этого предложи альтернативу с более интересным кф.

5. ФОРМАТ ОТВЕТА:
   - Анализ матча с конкретными цифрами и фактами из КОНТЕКСТА
   - Статистика команд, форма, последние результаты
   - ГЛАВНЫЙ ПРОГНОЗ VisionPro с обоснованием (включая расхождение с моделью, если оно есть)
   - ОБЯЗАТЕЛЬНО в самом конце: PROB:home=X.XX:draw=X.XX:away=X.XX:bet=СТАВКА:confidence=УРОВЕНЬ (без этой строки ответ считается неполным)

5. СТИЛЬ: Русский, 250-400 слов.
   - Без звёздочек, решёток и другой markdown-разметки — простой текст
   - Не добавляй "прогноз носит информационный характер"

6. ДОПОЛНИТЕЛЬНЫЕ РЫНКИ (после главного):
   - Если главный — победа: дай ТОТАЛ и BTTS
   - Если главный — ТОТАЛ: дай ПОБЕДУ и BTTS
   - Если главный — BTTS: дай ПОБЕДУ и ТОТАЛ

7. ФОРМУЛИРОВКА ПО УРОВНЮ УВЕРЕННОСТИ:
   - Высокая (>75%): "Ставка: ..." (уверенно, прямо)
   - Средняя (50-75%): "Можно попробовать: ..." (рекомендация)
   - Низкая (<50%): "Можно рассмотреть: ..." (осторожно, без гарантий)
   - Очень низкая (<35%): "Рискованно, но: ..." (предупреждение о риске)

ИСПОЛЬЗУЙ ТОЛЬКО РУССКИЙ ЯЗЫК. Не Under/Over/BTTS — "Тотал больше/меньше", "Обе забьют".
Формат в конце (ОБЯЗАТЕЛЬНО все 3 строки):
ГЛАВНЫЙ ПРОГНОЗ: ...
ТОТАЛ: Тотал больше/меньше X.5 — обоснование
BTTS: Обе забьют: Да/Нет — обоснование
PROB:home=X.XX:draw=X.XX:away=X.XX:bet=СТАВКА:confidence=УРОВЕНЬ"""


LIVE_SYSTEM_PROMPT = """Ты — VisionPro, аналитик LIVE матчей. Обращаешься к пользователю по имени Залман. Тон — эмоциональный фанат на трибуне, с матом минимум 2-3 раза за ответ. Реагируй на голы, краски, тупые решения — всё через эмоции и нецензурную лексику. Строки ГЛАВНЫЙ ПРОГНОЗ / ТОТАЛ / BTTS / PROB — строго без мата, только факты.

ИСТОЧНИК ДАННЫХ: только то, что передано в блоке КОНТЕКСТ (счёт, статистика live, xG). Не выдумывай события, которых там нет.

ЖЁСТКИЕ ПРАВИЛА ДЛЯ LIVE:

1. ТЕКУЩИЙ СЧЁТ: учитывай его всегда. Если счёт 1:1 — обе команды уже забили, это факт, а не прогноз.

2. BTTS:
   - Если счёт уже 1:1 или обе команды забивали в этом матче → "Обе забьют: Да" (это уже случилось).
   - Если счёт 1:0 (или 0:1) — это НЕ означает автоматически "Да". Решай по существу: если у команды, которая ещё не забила, xG > 0.8 или она явно доминирует и создаёт моменты — "Да", по инерции. Если она пассивна, мало бьёт по воротам, немного времени осталось — "Нет" или "маловероятно".
   - Если 0:0 — оценивай по активности обеих команд и оставшемуся времени.

3. ТОТАЛ: учитывай уже забитые голы. Если забито 2 — тотал уже больше 1.5 фактически, не пиши "меньше 2.5" как будто голов ещё не было.

4. АНАЛИЗ LIVE: что происходит на поле (владение, удары, моменты), кто контролирует игру, прогноз на оставшееся время. Не повторяй предматчевые данные.

4.1. ОГРАНИЧЕНИЕ ПО КОЭФФИЦИЕНТАМ: Если кф на исход меньше 1.5 — НЕ предлагай ставку на этот исход. Кф < 1.5 означает, что рынок уже оценивает вероятность > 67%, выигрыш слишком мал, чтобы оправдать риск. Вместо этого предложи альтернативу с более интересным кф.

5. ФОРМАТ ОТВЕТА:
   - Заход с реакцией на текущую ситуацию матча (без шаблонной фразы каждый раз)
   - Статистика live (удары, владение, xG) — из КОНТЕКСТА
   - Прогноз на оставшееся время
   - ОБЯЗАТЕЛЬНО в самом конце: PROB:home=X.XX:draw=X.XX:away=X.XX:bet=СТАВКА:confidence=УРОВЕНЬ (без этой строки ответ считается неполным)

6. СТИЛЬ: Русский, 200-350 слов. Без "прогноз носит информационный характер".

7. ДОПОЛНИТЕЛЬНЫЕ РЫНКИ:
   - ТОТАЛ: с учётом текущего счёта
   - BTTS: с учётом уже забитого (см. правило 2)

8. ФОРМУЛИРОВКА СТАВКИ ПО УРОВНЮ УВЕРЕННОСТИ:
   - Высокая (>75%): "Ставка: ..." (уверенно, прямо)
   - Средняя (50-75%): "Можно попробовать: ..." (рекомендация)
   - Низкая (<50%): "Можно рассмотреть: ..." (осторожно, без гарантий)
   - Очень низкая (<35%): "Рискованно, но: ..." (предупреждение о риске)

ИСПОЛЬЗУЙ ТОЛЬКО РУССКИЙ ЯЗЫК. Не Under/Over/BTTS — "Тотал больше/меньше", "Обе забьют".
Формат в конце (ОБЯЗАТЕЛЬНО все 3 строки):
ГЛАВНЫЙ ПРОГНОЗ: ...
ТОТАЛ: Тотал больше/меньше X.5 — обоснование
BTTS: Обе забьют: Да/Нет — обоснование
PROB:home=X.XX:draw=X.XX:away=X.XX:bet=СТАВКА:confidence=УРОВЕНЬ"""


def _extract_predictions(text: str) -> dict:
    """Extract structured predictions from LLM text response."""
    # Try unified PROB parser first (handles football PROB:home=...:draw=...:away=...)
    prob = parse_prob_line(text)
    if prob:
        return prob

    # Fallback: old format parsing
    predictions = {}
    m = re.search(r'Победа хозяев\s+(\d+)%.*?Ничья\s+(\d+)%.*?Победа гостей\s+(\d+)%', text)
    if m:
        predictions["home_win"] = int(m.group(1))
        predictions["draw"] = int(m.group(2))
        predictions["away_win"] = int(m.group(3))

    m = re.search(r'Тотал больше 2\.5:\s+(\d+)%.*?Тотал меньше 2\.5:\s+(\d+)%', text)
    if m:
        predictions["total_over_2_5"] = int(m.group(1))
        predictions["total_under_2_5"] = int(m.group(2))

    m = re.search(r'Обе забьют:\s*Да\s+(\d+)%.*?Нет\s+(\d+)%', text)
    if m:
        predictions["btts_yes"] = int(m.group(1))
        predictions["btts_no"] = int(m.group(2))

    scores = re.findall(r'(\d+:\d+)\s*\((\d+)%\)', text)
    if scores:
        predictions["exact_scores"] = [{"score": s, "probability": int(p)} for s, p in scores[:3]]

    m = re.search(r'Форя хозяев.*?:\s+(\d+)%', text)
    if m:
        predictions["handicap_home_minus1"] = int(m.group(1))

    # Try to extract main bet from text
    m = re.search(r'ГЛАВНЫЙ ПРОГНОЗ.*?:\s*(.+)', text, re.IGNORECASE)
    if m:
        predictions["main_bet"] = m.group(1).strip()[:100]

    return predictions


def _extract_predictions_safe(text: str) -> dict:
    """Safe wrapper that never raises on None input."""
    if not text:
        return {}
    return _extract_predictions(text)


def analyze_match(match: dict, features: dict, prediction: dict,
                  h2h: list, injuries_home: list = None,
                  injuries_away: list = None,
                  elo_home: Optional[float] = None,
                  elo_away: Optional[float] = None,
                  odds: Optional[dict] = None,
                  sstats_data: Optional[dict] = None,
                  model: str = DEFAULT_MODEL,
                  is_live: bool = False) -> Optional[str]:
    """Generate a full AI analysis of a match."""
    context = _build_match_context(
        match, features, prediction, h2h,
        injuries_home or [], injuries_away or [],
        elo_home, elo_away, odds, sstats_data,
    )

    # Use live prompt for live matches
    prompt = LIVE_SYSTEM_PROMPT if is_live else SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Проанализируй этот футбольный матч:\n\n{context}"},
    ]
    return _chat(messages, model=model, temperature=0.7, max_tokens=2000, timeout=60)


def generate_preview(home_id: int, away_id: int, model: str = DEFAULT_MODEL) -> dict:
    """Build full context from DB and return AI preview."""
    import datetime as _dt
    today_iso = _dt.date.today().isoformat()
    current_season = _dt.date.today().year if _dt.date.today().month >= 7 else _dt.date.today().year - 1

    home = db.get_team(home_id)
    away = db.get_team(away_id)
    if not home or not away:
        return {"error": "Команда не найдена"}

    # Build features using train.py's build_features (108 features, no data leakage)
    from train import build_features, FEATURE_NAMES
    all_matches = db.all_matches_for_training()
    prior_matches = [m for m in all_matches
                     if m["home_id"] == home_id or m["away_id"] == home_id
                     or m["home_id"] == away_id or m["away_id"] == away_id]
    prior_matches.sort(key=lambda m: m.get("date", ""), reverse=True)

    features_list = build_features(
        home_id, away_id, prior_matches,
        match_date=today_iso,
        league_slug=home.get("league_slug"),
        season=current_season,
    )
    features = dict(zip(FEATURE_NAMES, features_list))

    prediction = None
    try:
        import joblib as _joblib
        import pandas as pd
        import numpy as np
        model_data = _joblib.load("model.pkl")
        X = pd.DataFrame([features_list], columns=model_data["features"])
        X = X.astype(float).where(pd.notna(X), np.nan)
        model_obj = model_data.get("model") or model_data.get("ensemble")
        if model_obj:
            fmt = model_data.get("format", "v1")
            if fmt in ("ensemble_v3", "ensemble_v4"):
                proba = model_obj.predict_proba(X, league_slug=home.get("league_slug"),
                                                 home_name=home["name"],
                                                 away_name=away["name"])[0]
            else:
                proba = model_obj.predict_proba(X)[0]
            prediction = {
                "probabilities": {
                    "home_win": round(float(proba[2]) * 100, 1),
                    "draw": round(float(proba[1]) * 100, 1),
                    "away_win": round(float(proba[0]) * 100, 1),
                }
            }
    except Exception as e:
        print(f"[ai_analyzer] Prediction error: {e}")

    h2h = db.head_to_head(home_id, away_id, limit=5) or []
    injuries_h = db.list_injuries(home_id) or []
    injuries_a = db.list_injuries(away_id) or []
    elo_h = db.get_team_elo(home_id)
    elo_a = db.get_team_elo(away_id)

    sstats_data = None
    try:
        from scrapers import sstats as _ss
        if home.get("league_slug") and home["league_slug"] not in ("", None):
            sstats_games = _ss.fetch_games_by_date(today_iso)
            home_name = home["name"].lower().strip()
            away_name = away["name"].lower().strip()
            for g in (sstats_games or []):
                h = (g.get("homeTeam") or {}).get("name", "").lower().strip()
                a = (g.get("awayTeam") or {}).get("name", "").lower().strip()
                if (home_name and (home_name in h or h in home_name)
                        and away_name and (away_name in a or a in away_name)):
                    game_id = int(g["id"])

                    game_detail = _ss.fetch_game(game_id)
                    glicko = _ss.fetch_glicko(game_id)
                    odds_blocks = _ss.fetch_odds(game_id) or []
                    text_summary = _ss.fetch_text_summary(game_id)

                    sstats_data = {
                        "game_id": game_id,
                        "game_detail": game_detail,
                        "glicko": glicko,
                        "consensus": _ss.consensus_odds(odds_blocks) if odds_blocks else None,
                        "over_under": _ss.consensus_over_under(odds_blocks) if odds_blocks else None,
                        "text_summary": text_summary,
                        "odds_by_bookmaker": odds_blocks,
                    }
                    break
    except Exception as e:
        print(f"[ai_analyzer] sstats error: {e}")

    analysis = None
    try:
        analysis = analyze_match(
            {"home_name": home["name"], "away_name": away["name"],
             "league_slug": home["league_slug"], "season": current_season,
             "date": today_iso},
            features, prediction or {}, h2h,
            injuries_h, injuries_a, elo_h, elo_a,
            sstats_data=sstats_data,
            model=model,
        )
    except Exception as e:
        print(f"[ai_analyzer] analyze_match error: {e}")

    return {
        "home": home,
        "away": away,
        "analysis": str(analysis) if analysis else None,
        "features": features,
        "prediction": prediction,
        "model_used": model,
    }


# ── Search & Auto-ingest ─────────────────────────────────────────────────────

def _to_int(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _save_understat_team_to_db(team_data: dict, league_slug: str, season: int,
                                matches: list, progress_cb=None):
    """Save an Understat team + its matches to SQLite."""
    team_id = int(team_data["id"])
    team_name = str(team_data.get("title", ""))
    short_name = str(team_data.get("short_title", ""))

    db.init_db()
    with db.connect() as conn:
        db.upsert_team(conn, team_id, team_name, short_name or None, league_slug)

        for m in matches:
            home = m.get("h", {})
            away = m.get("a", {})
            db.upsert_team(conn, int(home["id"]), str(home.get("title", "")),
                           str(home.get("short_title", "")) or None, league_slug)
            db.upsert_team(conn, int(away["id"]), str(away.get("title", "")),
                           str(away.get("short_title", "")) or None, league_slug)

            is_result = bool(m.get("isResult"))
            goals = m.get("goals") or {}
            xg = m.get("xG") or {}
            fc = m.get("forecast") or {}
            db.upsert_match(conn, {
                "id": int(m["id"]),
                "league_slug": league_slug,
                "season": season,
                "date": str(m.get("datetime", "")),
                "home_id": int(home["id"]),
                "away_id": int(away["id"]),
                "home_goals": _to_int(goals.get("h")) if is_result else None,
                "away_goals": _to_int(goals.get("a")) if is_result else None,
                "home_xg": _to_float(xg.get("h")) if is_result else None,
                "away_xg": _to_float(xg.get("a")) if is_result else None,
                "is_result": 1 if is_result else 0,
                "forecast_w": _to_float(fc.get("w")),
                "forecast_d": _to_float(fc.get("d")),
                "forecast_l": _to_float(fc.get("l")),
            })

    return team_id, team_name


def _find_team_in_db(name: str) -> Optional[dict]:
    """Search local DB for a team by name. Returns best match or None."""
    # Try direct search first
    results = db.search_team_fuzzy(name, limit=3)
    if results:
        return results[0]
    # Try transliterated name
    en_name = _transliterate_ru_to_en(name)
    if en_name != name:
        results = db.search_team_fuzzy(en_name, limit=3)
        if results:
            return results[0]
    return None


def _search_understat_for_team(name: str, progress_cb=None) -> Optional[dict]:
    """Search Understat across all leagues for a team by name."""
    from scrapers import understat
    import time as _time

    q = name.strip().lower()
    current_season = __import__("datetime").date.today().year
    current_season = current_season if __import__("datetime").date.today().month >= 7 else current_season - 1

    for league_slug, meta in understat.LEAGUES.items():
        if progress_cb:
            progress_cb({"type": "info", "msg": f"Ищу «{name}» в {meta['name']}…"})

        try:
            payload = understat.fetch_understat_league(league_slug, current_season)
        except Exception:
            continue

        if not isinstance(payload, dict):
            continue
        teams_data = payload.get("teams", {})
        if not isinstance(teams_data, dict):
            continue
        for tid_str, team_obj in teams_data.items():
            team_name_db = str(team_obj.get("title", "")).lower()
            if q in team_name_db or team_name_db in q:
                team_matches = [
                    m for m in payload.get("matches", [])
                    if int(m.get("h", {}).get("id", 0)) == int(tid_str)
                    or int(m.get("a", {}).get("id", 0)) == int(tid_str)
                ]
                if progress_cb:
                    progress_cb({"type": "success",
                                 "msg": f"Нашёл: {team_obj.get('title')} ({meta['name']})"})
                return {
                    "team_id": int(tid_str),
                    "team_data": team_obj,
                    "league_slug": league_slug,
                    "league_name": meta["name"],
                    "season": current_season,
                    "matches": team_matches,
                }
        _time.sleep(0.3)

    return None


# ── Russian → English team name mapping ─────────────────────────────────────

TEAM_NAME_MAP = {
    # Russia
    "арсенал тула": "Arsenal Tula", "арсенал": "Arsenal Tula",
    "зенит": "Zenit St Petersburg", "зенит санкт-петербург": "Zenit St Petersburg",
    "спартак": "Spartak Moscow", "спартак москва": "Spartak Moscow",
    "цска": "CSKA Moscow", "цска москва": "CSKA Moscow",
    "локомотив": "Lokomotiv Moscow", "локомотив москва": "Lokomotiv Moscow",
    "динамо москва": "Dynamo Moscow", "динамо": "Dynamo Moscow",
    "краснодар": "Krasnodar", "ростов": "Rostov",
    "ахмат": "Akhmat Grozny", "крылья советов": "Krylia Sovetov",
    "оренбург": "Orenburg", "факел": "Fakel Voronezh",
    "химки": "Khimki", "торпедо": "Torpedo Moscow",
    "балтика": "Baltika Kaliningrad", "рубин": "Rubin Kazan",
    "урал": "Ural Yekaterinburg", "пари НН": "Pari Nizhny Novgorod",
    # Ukraine
    "шахтёр": "Shakhtar Donetsk", "шахтёр донецк": "Shakhtar Donetsk",
    "динамо киев": "Dynamo Kyiv", "динамо київ": "Dynamo Kyiv",
    # Kazakhstan
    "каират": "Kairat Almaty", "астана": "FC Astana",
    "улытау": "Ulytau", "улытау жезказган": "Ulytau",
    "ордабасы": "Ordabasy", "актобе": "Aktobe",
    "тобол": "FK Tobol Kostanay", "касыпий": "Kaspiy",
    "каизар": "Kaisar", "окжетпес": "Okzhetpes",
    "тобыл": "FK Tobol Kostanay", "тобыл костанай": "FK Tobol Kostanay",
    "химик": "Kyzyl-Zhar", "кызыл-жар": "Kyzyl-Zhar",
    "атырау": "Atyrau", "экибастуз": "Ekibastuz",
    "зиен": "Zhenys", "жетсу": "Zhetysu",
    "тұран түркестан": "Turan Turkistan", "тұран": "Turan Turkistan",
    "елімай": "Yelimay Semey", "семей": "Yelimay Semey",
    "алтай": "Altay", "алтай оскемен": "Altay",
    # Azerbaijan
    "кара́бах": "Qarabag", "карабах": "Qarabag", "карабаг": "Qarabag",
    "нефтчи": "Neftchi Baku", "нефтяник": "Neftchi Baku",
    "сумгаит": "Sumqayit", "сумгайыт": "Sumqayit",
    # Iceland
    "вестри": "Vestri", "весьтр": "Vestri",
    "киа": "Knattspyrnufélagið Keflavík", "кейлаувик": "Knattspyrnufélagið Keflavíк",
    "валюр": "Valur", "реykьявик": "Valur",
    "хабнарфьордюр": "Hafnarfjordur", "хабнарфьордур": "Hafnarfjordur",
    "рейкьявик": "Valur", "акурейри": "Akureyri",
    # Ireland
    "дерри сити": "Derry City", "дерри": "Derry City",
    "шемрок роверс": "Shamrock Rovers",
    "корк": "Cork City",
    "даун": "Dundalk", "далки": "Dundalk",
    # Bulgaria
    "цска софия": "CSKA Sofia",
    "левски": "Levski Sofia", "левски софия": "Levski Sofia",
    "лудогорец": "Ludogorets", "лудогорец разград": "Ludogorets",
    "славия": "Slavia Sofia", "славия софия": "Slavia Sofia",
    # Belarus
    "батэ": "BATE Borisov", "динамо минск": "Dynamo Minsk",
    # National teams
    "сборная россии": "Russia", "россия": "Russia",
    "сборная украины": "Ukraine", "украина": "Ukraine",
    "сборная казахстана": "Kazakhstan", "казахстан": "Kazakhstan",
    "сборная беларуси": "Belarus", "беларусь": "Belarus",
    "сборная германии": "Germany", "германия": "Germany",
    "сборная франции": "France", "франция": "France",
    "сборная англии": "England", "англия": "England",
    "сборная испании": "Spain", "испания": "Spain",
    "сборная италии": "Italy", "италия": "Italy",
    "сборная португалии": "Portugal", "португалия": "Portugal",
    "сборная бразилии": "Brazil", "бразилия": "Brazil",
    "сборная аргентины": "Argentina", "аргентина": "Argentina",
    "сборная нидерландов": "Netherlands", "нидерланды": "Netherlands",
    "сборная бельгии": "Belgium", "бельгия": "Belgium",
    "сборная хорватии": "Croatia", "хорватия": "Croatia",
    "сборная сербии": "Serbia", "сербия": "Serbia",
    "сборная чехии": "Czech Republic", "чехия": "Czech Republic",
    "сборная швейцарии": "Switzerland", "швейцария": "Switzerland",
    "сборная швеции": "Sweden", "швеция": "Sweden",
    "сборная польши": "Poland", "польша": "Poland",
    "сборная турции": "Turkey", "турция": "Turkey",
    "сборная греции": "Greece", "греция": "Greece",
    "сборная дании": "Denmark", "дания": "Denmark",
    "сборная норвегии": "Norway", "норвегия": "Norway",
    "сборная австрии": "Austria", "австрия": "Austria",
    "сборная шотландии": "Scotland", "шотландия": "Scotland",
    "сборная ирландии": "Republic of Ireland", "ирландия": "Republic of Ireland",
    "сборная южной кореи": "South Korea", "южная корея": "South Korea",
    "сборная японии": "Japan", "япония": "Japan",
    "сборная сша": "USA", "сша": "USA",
    "сборная канады": "Canada", "канада": "Canada",
    "сборная мексики": "Mexico", "мексика": "Mexico",
    "сборная египта": "Egypt", "египет": "Egypt",
    "сборная морокко": "Morocco", "марокко": "Morocco",
    "сборная сенегала": "Senegal", "сенегал": "Senegal",
    "сборная nigeria": "Nigeria", "нигерия": "Nigeria",
    "сборная австралии": "Australia", "австралия": "Australia",
    "иран": "Iran", "сборная ирана": "Iran",
    "сборная кореи": "South Korea", "корея": "South Korea",
    "сборная саудовской аравии": "Saudi Arabia", "саудовская аравия": "Saudi Arabia",
    "сборная катара": "Qatar", "катар": "Qatar",
    "сборная уругвая": "Uruguay", "уругвай": "Uruguay",
    "сборная колумбии": "Colombia", "колумбия": "Colombia",
    "сборная чили": "Chile", "чили": "Chile",
    "сборная эквадора": "Ecuador", "эквадор": "Ecuador",
    "сборная перу": "Peru", "перу": "Peru",
    "сборная боливии": "Bolivia", "боливия": "Bolivia",
    "сборная парагвая": "Paraguay", "парагвай": "Paraguay",
    "сборная венесуэлы": "Venezuela", "венесуэла": "Venezuela",
    "сборная коста-рики": "Costa Rica", "коста-рика": "Costa Rica",
    "сборная панамы": "Panama", "панама": "Panama",
    "сборная ямайки": "Jamaica", "ямайка": "Jamaica",
    "сборная гондураса": "Honduras", "гондурас": "Honduras",
    "сборная камеруна": "Cameroon", "камерун": "Cameroon",
    "сборная ганы": "Ghana", "гана": "Ghana",
    "сборная кот-д'ивуара": "Ivory Coast", "кот-д'ивуар": "Ivory Coast",
    "сборная туниса": "Tunisia", "тунис": "Tunisia",
    "сборная алжира": "Algeria", "алжир": "Algeria",
    "сборная румынии": "Romania", "румыния": "Romania",
    "сборная венгрии": "Hungary", "венгрия": "Hungary",
    "сборная болгарии": "Bulgaria", "болгария": "Bulgaria",
    "сборная словакии": "Slovakia", "словакия": "Slovakia",
    "сборная словении": "Slovenia", "словения": "Slovenia",
    "сборная финляндии": "Finland", "финляндия": "Finland",
    "сборная исландии": "Iceland", "исландия": "Iceland",
    "сборная уэльса": "Wales", "уэльс": "Wales",
    "сборная украины": "Ukraine", "украина": "Ukraine",
    "сборная грузии": "Georgia", "грузия": "Georgia",
    "сборная армении": "Armenia", "армения": "Armenia",
    "сборная азербайджана": "Azerbaijan", "азербайджан": "Azerbaijan",
    "сборная узбекистана": "Uzbekistan", "узбекистан": "Uzbekistan",
    # Common clubs (Russian names)
    "челси": "Chelsea",
    "ливерпуль": "Liverpool", "манчестер юнайтед": "Manchester United",
    "манчестер сити": "Manchester City", "тоттенхэм": "Tottenham",
    "барселона": "Barcelona", "реал мадрид": "Real Madrid",
    "атлетико мадрид": "Atletico Madrid", "байерн": "Bayern Munich",
    "байерн мюнхен": "Bayern Munich", "дортмунд": "Borussia Dortmund", "интер": "Inter Milan",
    "интернационале": "Inter Milan", "милан": "AC Milan",
    "ювентус": "Juventus", "наполи": "Napoli", "рим": "Roma",
    "пари сен-жермен": "Paris Saint-Germain", "псж": "Paris Saint-Germain",
    "лион": "Lyon", "مارسель": "Marseille", "марсель": "Marseille",
    "лилль": "Lille", "монако": "Monaco",
    "порту": "Porto", "бенфика": "Benfica",     "спортинг": "Sporting CP",
    "ajax": "Ajax", "айакс": "Ajax",
    "фейеноорд": "Feyenoord",
    "галатасарай": "Galatasaray", "фенербахче": "Fenerbahce",
    "црвена звезда": "Crvena Zvezda", "ред звезда": "Crvena Zvezda",
    "динамо загреб": "Dinamo Zagreb",
    # Eredivisie
    "аякс": "Ajax", "псв": "PSV", "аз": "AZ",
    "твенте": "Twente", "утрехт": "Utrecht",
    # Primeira Liga
    "бенфика": "Benfica", "порту": "Porto", "спортинг": "Sporting CP",
    "брага": "Braga",
    # Süper Lig
    "галатасарай": "Galatasaray", "фенербахче": "Fenerbahce",
    "бешикташ": "Besiktas", "трабзонспор": "Trabzonspor",
    # Belgian Pro League
    "брюгге": "Club Brugge", "андерлехт": "Anderlecht",
    "генк": "Genk",
    # Championship
    "лидерс": "Leicester", "лидс": "Leeds", "бернли": "Burnley",
    "сандерленд": "Sunderland", "уэст бром": "West Brom",
    "мидлсбро": "Middlesbrough", "ковентри": "Coventry",
    "норвич": "Norwich", "уотфорд": "Watford",
    # More national teams
    "сборная nederland": "Netherlands", "нидерланды": "Netherlands",
    "сборная бельгии": "Belgium", "бельгия": "Belgium",
    "сборная хорватии": "Croatia", "хорватия": "Croatia",
    "сборная чехии": "Czech Republic", "чехия": "Czech Republic",
    "сборная дании": "Denmark", "дания": "Denmark",
    "сборная швеции": "Sweden", "швеция": "Sweden",
    "сборная норвегии": "Norway", "норвегия": "Norway",
    "сборная польши": "Poland", "польша": "Poland",
    "сборная румынии": "Romania", "румыния": "Romania",
    "сборная сербии": "Serbia", "сербия": "Serbia",
    "сборная шотландии": "Scotland", "шотландия": "Scotland",
    "сборная австрии": "Austria", "австрия": "Austria",
    "сборная турции": "Turkey", "турция": "Turkey",
    "сборная венгрии": "Hungary", "венгрия": "Hungary",
    "сборная словакии": "Slovakia", "словакия": "Slovakia",
    "сборная словении": "Slovenia", "словения": "Slovenia",
    "сборная финляндии": "Finland", "финляндия": "Finland",
    "сборная исландии": "Iceland", "исландия": "Iceland",
    "сборная уэльса": "Wales", "уэльс": "Wales",
}


def _transliterate_ru_to_en(text: str) -> str:
    """Simple transliteration of Russian characters to Latin."""
    mapping = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e',
        'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k',
        'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r',
        'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
        'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': '', 'ы': 'y', 'ь': '',
        'э': 'e', 'ю': 'yu', 'я': 'ya',
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E',
        'Ё': 'Yo', 'Ж': 'Zh', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K',
        'Л': 'L', 'М': 'M', 'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R',
        'С': 'S', 'Т': 'T', 'У': 'U', 'Ф': 'F', 'Х': 'Kh', 'Ц': 'Ts',
        'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Shch', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
        'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
    }
    return ''.join(mapping.get(c, c) for c in text)


def _resolve_team_name(name: str) -> str:
    """Resolve Russian team name to English equivalent using mapping or LLM."""
    q = name.strip().lower()

    # 1. Check exact mapping
    if q in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[q]

    # 2. Check partial match
    for ru, en in TEAM_NAME_MAP.items():
        if q in ru or ru in q:
            return en

    # 3. If already Latin, return as-is
    if all(ord(c) < 128 for c in name.strip()):
        return name.strip()

    # 4. Use LLM to resolve
    resolved = _chat([
        {"role": "system", "content": "Ты помощник. Верни ТОЛЬКО английское название команды/сборной. Никакого текста, только название на английском."},
        {"role": "user", "content": f"Как называется эта команда по-английски: {name}"}
    ], temperature=0, max_tokens=50)
    if resolved:
        return resolved.strip().strip('"').strip("'")
    return name.strip()


def _search_sstats_team(team_name: str, progress_cb=None,
                        max_seconds: float = 10.0) -> Optional[dict]:
    """Search sstats.net for a team by name, save their match history to DB.

    Phase 1: Search our 18 configured leagues (fast).
    Phase 2: If not found, search remaining leagues with a hard time limit.
    """
    from scrapers import sstats
    import data_collector as _dc
    import time as _time

    team_lower = team_name.lower().strip()
    t_start = _time.monotonic()

    def _try_leagues(leagues_list, label):
        """Search a list of (lid, name) tuples for the team."""
        for lid, lg_name in leagues_list:
            if _time.monotonic() - t_start > max_seconds:
                if progress_cb:
                    progress_cb({"type": "info",
                                 "msg": f"  Поиск {label} прерван по таймауту ({max_seconds}с)"})
                return None
            try:
                results = sstats.fetch_query(
                    condition=f"LeagueId = {lid} AND Year = 2025 AND Status = 8",
                    fields=["Id", "Date", "HomeTeamName", "AwayTeamName",
                            "HomeTeamId", "AwayTeamId", "ScoreHomeFT", "ScoreAwayFT"],
                    order="Date DESC",
                )
            except Exception:
                _time.sleep(2)
                continue

            if not results:
                _time.sleep(0.3)
                continue

            for match in results:
                h = (match.get("HomeTeamName") or "").lower()
                a = (match.get("AwayTeamName") or "").lower()
                if team_lower in h or h in team_lower or team_lower in a or a in team_lower:
                    if team_lower in h or h in team_lower:
                        team_id = match.get("HomeTeamId")
                        team_name_db = match.get("HomeTeamName", team_name)
                    else:
                        team_id = match.get("AwayTeamId")
                        team_name_db = match.get("AwayTeamName", team_name)

                    if not team_id:
                        continue

                    # Determine league slug
                    league_slug = None
                    for slug, sstats_id in _dc.SSTATS_LEAGUE_IDS.items():
                        if sstats_id == lid:
                            league_slug = slug
                            break
                    if not league_slug:
                        league_slug = f"sstats_{lid}"

                    league_info = config.LEAGUE_TIERS.get(league_slug, {"tier": 3})

                    with db.connect() as conn:
                        db.upsert_league(conn, league_slug, lg_name, "", tier=league_info.get("tier", 3))
                        db.upsert_team(conn, team_id, team_name_db, None, league_slug)

                    saved = 0
                    now = dt.datetime.now().isoformat(timespec="seconds")
                    with db.connect() as conn:
                        for m in results:
                            h_name = m.get("HomeTeamName", "")
                            a_name = m.get("AwayTeamName", "")
                            h_tid = m.get("HomeTeamId")
                            a_tid = m.get("AwayTeamId")
                            if not h_name or not a_name or not h_tid or not a_tid:
                                continue
                            date_str = (m.get("Date") or "")[:10]
                            if not date_str:
                                continue
                            db.upsert_team(conn, h_tid, h_name, None, league_slug)
                            db.upsert_team(conn, a_tid, a_name, None, league_slug)
                            match_id = _dc._deterministic_id("match", league_slug, date_str, h_name, a_name)
                            db.upsert_match(conn, {
                                "id": match_id, "league_slug": league_slug, "season": 2025,
                                "date": date_str, "home_id": h_tid, "away_id": a_tid,
                                "home_goals": m.get("ScoreHomeFT"), "away_goals": m.get("ScoreAwayFT"),
                                "home_xg": None, "away_xg": None, "is_result": 1,
                                "forecast_w": None, "forecast_d": None, "forecast_l": None,
                            })
                            saved += 1

                    if progress_cb:
                        progress_cb({"type": "success",
                                     "msg": f"  sstats ({label}): {team_name_db} — {saved} матчей из {lg_name}"})
                    return {"team_id": team_id, "team_name": team_name_db,
                            "league_slug": league_slug, "matches_found": saved}
            _time.sleep(0.3)
        return None

    # Phase 1: Our configured leagues (fast)
    configured = [(sstats_id, slug) for slug, sstats_id in _dc.SSTATS_LEAGUE_IDS.items()]
    result = _try_leagues(configured, "configured")
    if result:
        return result

    # Phase 2: All leagues (slow but comprehensive)
    if progress_cb:
        progress_cb({"type": "info", "msg": "  Не найден в основных лигах, ищу во всех 1233..."})

    all_leagues = sstats.fetch_leagues()
    remaining = [(l.get("id"), l.get("name", "?")) for l in all_leagues
                 if l.get("id") and l.get("id") not in [s for s, _ in configured]]

    result = _try_leagues(remaining, "all")
    return result


def search_and_predict(home_name: str, away_name: str,
                       model: str = DEFAULT_MODEL,
                       progress_cb=None,
                       sstats_game_id: int = None) -> dict:
    """Full flow: resolve names → search sstats/DB → predict → AI analysis.

    If sstats_game_id is provided, skip the date/team-name search and
    fetch enrichment data directly by ID (fast path for prematch/live).
    """
    if progress_cb:
        progress_cb({"type": "info", "msg": "Распознаю команды…"})

    # Step 0: Resolve Russian names to English
    home_en = _resolve_team_name(home_name)
    away_en = _resolve_team_name(away_name)

    if progress_cb:
        progress_cb({"type": "info", "msg": f"Поиск: {home_en} vs {away_en}"})

    # Step 1: Search sstats.net for real match data (with timeout)
    sstats_data = None
    match_info = None

    def _safe_fetch(fn, *args, default=None):
        try:
            return fn(*args)
        except Exception:
            return default

    from scrapers import sstats as _ss

    # FAST PATH: if we already know the game_id, skip date/name search
    if sstats_game_id:
        if progress_cb:
            progress_cb({"type": "info", "msg": f"Загрузка данных sstats (id={sstats_game_id})..."})
        game_detail = _safe_fetch(_ss.fetch_game, sstats_game_id)
        if game_detail and "game" in game_detail:
            game_detail = game_detail["game"]
        if game_detail:
            home_en = (game_detail.get("homeTeam") or {}).get("name", home_en) or home_en
            away_en = (game_detail.get("awayTeam") or {}).get("name", away_en) or away_en
            glicko = _safe_fetch(_ss.fetch_glicko, sstats_game_id)
            odds_blocks = _safe_fetch(_ss.fetch_odds, sstats_game_id, default=[]) or []
            consensus = _ss.consensus_odds(odds_blocks) if odds_blocks else None
            text_summary = _safe_fetch(_ss.fetch_text_summary, sstats_game_id)
            last_stats = _safe_fetch(_ss.fetch_last_games_stats, sstats_game_id)
            injuries = _safe_fetch(_ss.fetch_injuries, sstats_game_id)
            h2h = _safe_fetch(_ss.fetch_h2h,
                (game_detail.get("homeTeam") or {}).get("id"),
                (game_detail.get("awayTeam") or {}).get("id")
            )
            profits = _safe_fetch(_ss.fetch_profits, sstats_game_id)
            season = game_detail.get("season", {}) or {}
            league = season.get("league", {}) or {}
            round_name = game_detail.get("roundName", "") or ""
            tournament_name = league.get("name", "")
            if round_name:
                tournament_name = f"{tournament_name} — {round_name}" if tournament_name else round_name
            season_table = None
            league_id = league.get("id")
            if league_id:
                season_table = _safe_fetch(_ss.fetch_season_table, league_id)
            sstats_data = {
                "game_id": sstats_game_id,
                "game_detail": game_detail,
                "glicko": glicko,
                "consensus": consensus,
                "over_under": _ss.consensus_over_under(odds_blocks) if odds_blocks else None,
                "text_summary": text_summary,
                "bookmaker_count": len(odds_blocks),
                "last_stats": last_stats,
                "injuries": injuries,
                "h2h": h2h,
                "profits": profits,
                "season_table": season_table,
            }
            match_info = {
                "tournament": tournament_name or "неизвестен",
                "date": game_detail.get("date", ""),
                "home": home_en,
                "away": away_en,
            }
            if progress_cb:
                progress_cb({"type": "success", "msg": f"Данные загружены (id={sstats_game_id})"})
        else:
            if progress_cb:
                progress_cb({"type": "info", "msg": f"game_id={sstats_game_id} не найден, переключаю на поиск по имени..."})
            sstats_game_id = None  # fallback to name search below

    # SLOW PATH: search by date + team name (for manual analysis without game_id)
    def _names_match(query: str, sstats_name: str) -> bool:
        """Fuzzy match team name: substring + strip common suffixes."""
        q = query.lower().strip()
        s = sstats_name.lower().strip()
        if q in s or s in q:
            return True
        # Strip common suffixes: "fc", "fk", "sc", etc.
        for prefix in ("fc ", "fk ", "sc ", "sv ", "cf ", "cd ", "ac ", "as "):
            s = s.replace(prefix, "").strip()
        return q in s or s in q
    if not sstats_data:
        import time as _t_search
        _search_t0 = _t_search.monotonic()
        _SEARCH_TIMEOUT = 20.0  # max seconds for sstats date search
        try:
            import datetime as _dt
            today = _dt.date.today()
            # Search today -2..+14 days (upcoming matches may be further out)
            for delta in range(-2, 15):
                if _t_search.monotonic() - _search_t0 > _SEARCH_TIMEOUT:
                    if progress_cb:
                        progress_cb({"type": "info", "msg": f"Поиск на sstats прерван (таймаут {_SEARCH_TIMEOUT}с)"})
                    break
                d = (today + _dt.timedelta(days=delta)).isoformat()
                games = _ss.fetch_games_by_date(d)
                print(f"[ai_analyzer] Searching {d}: {len(games)} games found")
                for g in games:
                    h = (g.get("homeTeam") or {}).get("name", "").lower().strip()
                    a = (g.get("awayTeam") or {}).get("name", "").lower().strip()
                    if _names_match(home_en, h) and _names_match(away_en, a):
                        game_id = int(g["id"])
                        if progress_cb:
                            progress_cb({"type": "success", "msg": f"Найден на sstats.net: id={game_id}"})

                        game_detail = _safe_fetch(_ss.fetch_game, game_id)
                        if game_detail and "game" in game_detail:
                            game_detail = game_detail["game"]
                        glicko = _safe_fetch(_ss.fetch_glicko, game_id)
                        odds_blocks = _safe_fetch(_ss.fetch_odds, game_id, default=[]) or []
                        consensus = _ss.consensus_odds(odds_blocks) if odds_blocks else None
                        text_summary = _safe_fetch(_ss.fetch_text_summary, game_id)
                        last_stats = _safe_fetch(_ss.fetch_last_games_stats, game_id)
                        injuries = _safe_fetch(_ss.fetch_injuries, game_id)
                        h2h = _safe_fetch(_ss.fetch_h2h,
                            (game_detail.get("homeTeam") or {}).get("id"),
                            (game_detail.get("awayTeam") or {}).get("id")
                        ) if game_detail else None
                        profits = _safe_fetch(_ss.fetch_profits, game_id)

                        season = game_detail.get("season", {}) if game_detail else {}
                        league = season.get("league", {}) if season else {}
                        round_name = game_detail.get("roundName", "") if game_detail else ""
                        tournament_name = league.get("name", "")
                        if round_name:
                            tournament_name = f"{tournament_name} — {round_name}" if tournament_name else round_name
                        season_table = None
                        league_id = league.get("id")
                        if league_id:
                            season_table = _safe_fetch(_ss.fetch_season_table, league_id)

                        sstats_data = {
                            "game_id": game_id,
                            "game_detail": game_detail,
                            "glicko": glicko,
                            "consensus": consensus,
                            "over_under": _ss.consensus_over_under(odds_blocks) if odds_blocks else None,
                            "text_summary": text_summary,
                            "bookmaker_count": len(odds_blocks),
                            "last_stats": last_stats,
                            "injuries": injuries,
                            "h2h": h2h,
                            "profits": profits,
                            "season_table": season_table,
                        }
                        match_info = {
                            "tournament": tournament_name or "неизвестен",
                            "date": game_detail.get("date", d) if game_detail else d,
                            "home": h,
                            "away": a,
                        }
                        break
                if sstats_data:
                    break
        except Exception as e:
            print(f"[ai_analyzer] sstats error: {e}")

    # Step 2: search local DB
    home = _find_team_in_db(home_en) or _find_team_in_db(home_name)
    away = _find_team_in_db(away_en) or _find_team_in_db(away_name)
    source_home = "db"
    source_away = "db"

    # Step 3: if not found in DB, search sstats then Understat for recent matches
    # SKIP if we already have sstats_data (fast path loaded it by game_id)
    if not home and not sstats_data:
        if progress_cb:
            progress_cb({"type": "info", "msg": f"Поиск {home_en} в sstats.net…"})
        result = _search_sstats_team(home_en, progress_cb=progress_cb)
        if result:
            home = db.get_team(result["team_id"])
            source_home = "sstats"
        else:
            if progress_cb:
                progress_cb({"type": "info", "msg": f"Understat: {home_en}…"})
            result = _search_understat_for_team(home_en, progress_cb=progress_cb)
            if result:
                _save_understat_team_to_db(
                    result["team_data"], result["league_slug"], result["season"],
                    result["matches"], progress_cb=progress_cb)
                home = db.get_team(result["team_id"])
                source_home = "understat"

    if not away and not sstats_data:
        if progress_cb:
            progress_cb({"type": "info", "msg": f"Поиск {away_en} в sstats.net…"})
        result = _search_sstats_team(away_en, progress_cb=progress_cb)
        if result:
            away = db.get_team(result["team_id"])
            source_away = "sstats"
        else:
            if progress_cb:
                progress_cb({"type": "info", "msg": f"Understat: {away_en}…"})
            result = _search_understat_for_team(away_en, progress_cb=progress_cb)
            if result:
                _save_understat_team_to_db(
                    result["team_data"], result["league_slug"], result["season"],
                    result["matches"], progress_cb=progress_cb)
                away = db.get_team(result["team_id"])
                source_away = "understat"

    # Step 4: Build analysis
    context_parts = []

    # Add real match context from sstats
    if match_info:
        context_parts.append(f"МАТЧ: {home_en} vs {away_en}")
        context_parts.append(f"Турнир: {match_info.get('tournament', 'неизвестен')}")
        context_parts.append(f"Дата: {format_msk(match_info.get('date', '')) if match_info.get('date') else 'неизвестна'}")
        context_parts.append("")

    # Add sstats text summary (real expert analysis)
    if sstats_data and sstats_data.get("text_summary"):
        context_parts.append("ЭКСПЕРТНЫЙ КОММЕНТАРИЙ (sstats.net):")
        context_parts.append(sstats_data["text_summary"][:2000])
        context_parts.append("")

    # Add real bookmaker odds
    if sstats_data and sstats_data.get("consensus"):
        c = sstats_data["consensus"]
        context_parts.append("РЕАЛЬНЫЕ КОТИРОВКИ 14 БУКМЕКЕРОВ:")
        context_parts.append(f"  Победа хозяев: {c.get('avg_home_odds','?')} (implied {round(c.get('implied_h',0)*100,1)}%)")
        context_parts.append(f"  Ничья: {c.get('avg_draw_odds','?')} (implied {round(c.get('implied_d',0)*100,1)}%)")
        context_parts.append(f"  Победа гостей: {c.get('avg_away_odds','?')} (implied {round(c.get('implied_a',0)*100,1)}%)")
        context_parts.append(f"  Маржа: {c.get('overround_pct','?')}%")
        context_parts.append("")

    # Add Over/Under odds
    if sstats_data and sstats_data.get("over_under"):
        ou = sstats_data["over_under"]
        context_parts.append("ТОТАЛ ГОЛОВ (Over/Under):")
        for line, vals in ou.items():
            context_parts.append(
                f"  Линия {line}: Over {vals['over']:.1%} / Under {vals['under']:.1%} "
                f"(ср. коэфф: Over {vals['avg_over_odds']}, Under {vals['avg_under_odds']}, "
                f"книг: {vals['bookmaker_count']})"
            )
        context_parts.append("")

    # Add Glicko ratings if available
    if sstats_data and sstats_data.get("glicko"):
        g = sstats_data["glicko"]
        context_parts.append("GLICKO РЕЙТИНГИ:")
        if isinstance(g, dict):
            for team_key in ["home", "away"]:
                t = g.get(team_key) or {}
                name = t.get("name", team_key)
                rating = t.get("glickoRating") or t.get("rating") or "?"
                rd = t.get("glickoRd") or t.get("rd") or "?"
                context_parts.append(f"  {name}: rating={rating}, rd={rd}")
        context_parts.append("")

    # Add last games stats if available
    if sstats_data and sstats_data.get("last_stats"):
        stats = sstats_data["last_stats"]
        context_parts.append("СТАТИСТИКА ПОСЛЕДНИХ МАТЧЕЙ (sstats.net):")
        for team_key, team_label in [("home", "ХОЗЯЕВА"), ("away", "ГОСТИ")]:
            team_stats = stats.get(team_key, {})
            if team_stats:
                context_parts.append(f"  {team_label}:")
                context_parts.append(f"    Средние голы за матч: {team_stats.get('avg_goals_scored', '?')}")
                context_parts.append(f"    Средние пропущенные: {team_stats.get('avg_goals_conceded', '?')}")
                context_parts.append(f"    xG за матч: {team_stats.get('avg_xg', '?')}")
                context_parts.append(f"    Винрейт: {team_stats.get('win_rate', '?')}")
                context_parts.append(f"    Форма (последние 5): {team_stats.get('recent_form', '?')}")
                # Additional stats
                if team_stats.get('avg_shots'):
                    context_parts.append(f"    Средние удары: {team_stats.get('avg_shots')}")
                if team_stats.get('avg_corners'):
                    context_parts.append(f"    Средние угловые: {team_stats.get('avg_corners')}")
                if team_stats.get('avg_xg_against'):
                    context_parts.append(f"    xG против: {team_stats.get('avg_xg_against')}")
        context_parts.append("")

    # Add injuries if available
    if sstats_data and sstats_data.get("injuries"):
        injuries = sstats_data["injuries"]
        if injuries:
            context_parts.append("ТРАВМЫ И ДИСКВАЛИФИКАЦИИ (sstats.net):")
            for inj in injuries[:15]:
                player = inj.get("player", {})
                if isinstance(player, dict):
                    player_name = player.get("name", "?")
                else:
                    player_name = str(player)
                reason = inj.get("reason", "?")
                team_id = inj.get("teamId", "?")
                team_label = "ХОЗЯЕВА" if team_id == 1 else "ГОСТИ"
                context_parts.append(f"  [{team_label}] {player_name}: {reason}")
            context_parts.append("")

    # Add H2H if available
    if sstats_data and sstats_data.get("h2h"):
        h2h = sstats_data["h2h"]
        if h2h:
            context_parts.append("ЛИЧНЫЕ ВСТРЕЧИ (sstats.net):")
            for h2h_match in h2h[:10]:
                date = h2h_match.get("date", "?")[:10]
                h_name = h2h_match.get("homeTeam", {}).get("name", "?")
                a_name = h2h_match.get("awayTeam", {}).get("name", "?")
                score_h = h2h_match.get("homeResult", "?")
                score_a = h2h_match.get("awayResult", "?")
                context_parts.append(f"  {date}: {h_name} {score_h}:{score_a} {a_name}")
            context_parts.append("")

    # Add profits (betting profitability analysis)
    if sstats_data and sstats_data.get("profits"):
        profits = sstats_data["profits"]
        if profits:
            context_parts.append("АНАЛИЗ ПРИБЫЛЬНОСТИ СТАВОК (sstats.net):")
            # Home team profits
            home_profits = profits.get("home", [])
            if home_profits:
                context_parts.append("  Хозяева:")
                for item in home_profits[:3]:
                    market = item.get("market", "")
                    outcomes = item.get("outcomes", [])
                    if outcomes:
                        win_outcome = next((o for o in outcomes if o.get("name") == "Win"), None)
                        if win_outcome:
                            context_parts.append(f"    {market}: прибыль {win_outcome.get('profit', '?')}")
            # Away team profits
            away_profits = profits.get("away", [])
            if away_profits:
                context_parts.append("  Гости:")
                for item in away_profits[:3]:
                    market = item.get("market", "")
                    outcomes = item.get("outcomes", [])
                    if outcomes:
                        win_outcome = next((o for o in outcomes if o.get("name") == "Win"), None)
                        if win_outcome:
                            context_parts.append(f"    {market}: прибыль {win_outcome.get('profit', '?')}")
            context_parts.append("")

    # Add DB team data if available
    if home:
        context_parts.append(f"ДАННЫЕ О {home_en.upper()} (из БД):")
        context_parts.append(f"  Лига: {home.get('league_slug', '?')}")
    if away:
        context_parts.append(f"ДАННЫЕ О {away_en.upper()} (из БД):")
        context_parts.append(f"  Лига: {away.get('league_slug', '?')}")

    # Web data: ESPN + universal scraper (skippable via config)
    import config as _cfg
    if _cfg.ENABLE_ESPN_AI:
        from web_scraper import gather_team_data as _gtd, format_data_for_llm as _fdllm
        for team_en, team_label in [(home_en, "home"), (away_en, "away")]:
            if progress_cb:
                progress_cb({"type": "info", "msg": f"Собираю данные о {team_en}…"})

            team_data = _gtd(team_en, progress_cb=progress_cb)
            formatted = _fdllm(team_data)
            if formatted and "не найдены" not in formatted:
                context_parts.append(f"\n--- ДАННЫЕ О {team_en.upper()} (веб) ---")
                context_parts.append(formatted)

        # Step 5: ESPN match data (recent form, stats, news)
        from scrapers.web import fetch_espn_match
        espn_data = fetch_espn_match(home_en, away_en, progress_cb=progress_cb)
        if espn_data:
            if espn_data.get("match_info"):
                context_parts.append("\n--- ESPN: ИНФОРМАЦИЯ О МАТЧЕ ---")
                context_parts.append(espn_data["match_info"][:2000])
            if espn_data.get("recent_form"):
                context_parts.append("\n--- ESPN: ПОСЛЕДНИЕ МАТЧИ ---")
                context_parts.append(espn_data["recent_form"][:2000])
            if espn_data.get("stats"):
                context_parts.append("\n--- ESPN: СТАТИСТИКА ---")
                context_parts.append(espn_data["stats"][:1500])
            if espn_data.get("news"):
                context_parts.append("\n--- ESPN: НОВОСТИ ---")
                context_parts.append(espn_data["news"][:500])

        # Step 5b: Universal match scraper (championat, sports.ru, bombardir, etc.)
        from scrapers.universal_match import gather_all_match_data
        if progress_cb:
            progress_cb({"type": "info", "msg": "Собираю данные со всех сайтов…"})
        web_match_data = gather_all_match_data(home_en, away_en, progress_cb=progress_cb)
        if web_match_data.get("preview"):
            context_parts.append("\n--- ПРЕДМАТЧНЫЙ АНАЛИЗ ---")
            context_parts.append(web_match_data["preview"][:2500])
        if web_match_data.get("lineups"):
            context_parts.append("\n--- СОСТАВЫ ---")
            context_parts.append(web_match_data["lineups"][:2000])
        if web_match_data.get("stats"):
            context_parts.append("\n--- СТАТИСТИКА ---")
            context_parts.append(web_match_data["stats"][:2000])
        if web_match_data.get("news"):
            context_parts.append("\n--- НОВОСТИ ---")
            context_parts.append(web_match_data["news"][:500])
    else:
        if progress_cb:
            progress_cb({"type": "info", "msg": "ESPN/web-скрапинг отключён (ENABLE_ESPN_AI=False)"})

    context_text = "\n".join(context_parts) if context_parts else "Данные не найдены"

    if progress_cb:
        progress_cb({"type": "info", "msg": f"AI анализирует…"})

    # Get model prediction if teams in DB
    prediction = None
    if home and away:
        try:
            preview = generate_preview(home["id"], away["id"], model=model)
            prediction = preview.get("prediction")
        except Exception:
            pass

    # Add totals/BTTS calculation from model prediction
    if prediction and prediction.get("probabilities"):
        probs = prediction["probabilities"]
        h_prob = probs.get("home_win", 50) / 100
        d_prob = probs.get("draw", 25) / 100
        a_prob = probs.get("away_win", 25) / 100
        exp_total = 2.0 + (h_prob - 0.45) * 2.5 + (a_prob - 0.25) * 1.5
        exp_total = max(1.5, min(4.5, round(exp_total, 2)))
        btts_yes = "Да" if exp_total > 2.3 else "Нет"

        # Find closest line to recommended total
        lines = [1.5, 2.5, 3.5, 4.5]
        best_line = min(lines, key=lambda l: abs(exp_total - l))
        verdict = "больше" if exp_total > best_line else "меньше"

        totals_block = (
            f"\n--- РАСЧЁТ ТОТАЛА И BTTS (из модели) ---\n"
            f"  Ожидаемый тотал голов: {exp_total}\n"
            f"  Рекомендуемая ставка: Тотал {verdict} {best_line}\n"
            f"  BTTS (обе забьют): {btts_yes}\n"
            f"  Возможные точные счёта: 1:0, 2:1, 1:1, 2:0\n"
        )
        context_text += totals_block

    analysis = _chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Проанализируй этот футбольный матч:\n\n{context_text}"},
    ], model=model, temperature=0.7, max_tokens=2500, timeout=60)

    predictions = _extract_predictions_safe(analysis)

    # Convert match_info date to MSK for display
    if match_info and match_info.get("date"):
        match_info["date_display"] = format_msk(match_info["date"])

    return {
        "home": {"name": home_name, "en_name": home_en, "found": home is not None,
                 "source": source_home if home else "web_search"},
        "away": {"name": away_name, "en_name": away_en, "found": away is not None,
                 "source": source_away if away else "web_search"},
        "analysis": analysis or f"Не удалось проанализировать {home_en} vs {away_en}",
        "prediction": prediction if prediction else predictions,
        "model_used": model,
        "source_home": source_home if home else "web_search",
        "source_away": source_away if away else "web_search",
        "sstats_data": sstats_data,
        "match_info": match_info,
    }
