"""
telegram_bot.py — Telegram bot with Mini App for Football AI Predictor.

Setup:
  1. Create bot via @BotFather, get token
  2. Put token in Апи/telegram_token.txt (or set TELEGRAM_BOT_TOKEN env var)
  3. Set WEBAPP_URL env var to your HTTPS tunnel URL (e.g. https://xxxx.ngrok.io/mini-app)
  4. Run: python telegram_bot.py

Mini App opens via menu button or /app command.
Text predictions also work: "Team1 vs Team2"
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MenuButtonWebApp,
    Update,
    WebAppInfo,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

# ── Setup paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import db
import config
from ai_analyzer import search_and_predict, _resolve_team_name, analyze_match, _chat, SYSTEM_PROMPT

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("telegram_bot")


# ── Config ───────────────────────────────────────────────────────────────────

def _load_token() -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    if not token:
        token_path = PROJECT_ROOT / "Апи" / "telegram_token.txt"
        if token_path.exists():
            token = token_path.read_text().strip()
    return token


WEBAPP_URL = os.environ.get("WEBAPP_URL", "https://localhost:8000/mini-app")


# ── Handlers ─────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message."""
    user = update.effective_user
    text = (
        f"VisionPro — К вашим услугам, {user.first_name}!\n\n"
        f"Прогнозы футбольных матчей на основе ML (xG, Elo, форма команд, котировки).\n\n"
        f"Напиши название двух команд для прогноза:\n"
        f"  Ливерпуль vs Челси\n"
        f"  Реал Мадрид - Барселона"
    )

    await update.message.reply_text(text)


async def cmd_app(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show info about the web app."""
    await update.message.reply_text(
        f"Веб-приложение доступно по адресу:\n{WEBAPP_URL}\n\n"
        f"Откройте в браузере для полного функционала."
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Как пользоваться:\n\n"
        "Приложение:\n"
        "/app — открыть Mini App\n\n"
        "Текстовые прогнозы:\n"
        "  Ливерпуль vs Челси\n"
        "  Реал Мадрид - Барселона\n\n"
        "Команды:\n"
        "/start — начать\n"
        "/app — открыть приложение\n"
        "/help — помощь\n"
        "/trending — ближайшие матчи\n"
        "/prematch — матчи на сегодня"
    )


async def cmd_trending(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show upcoming matches from the DB."""
    try:
        upcoming = db.upcoming_matches(limit=20)
        if not upcoming:
            await update.message.reply_text("Нет запланированных матчей в базе.")
            return

        lines = ["Ближайшие матчи:\n"]
        for m in upcoming[:15]:
            date_str = (m.get("date") or "")[:10]
            home = m.get("home_name", "?")
            away = m.get("away_name", "?")
            lines.append(f"  {date_str}  {home} vs {away}")

        lines.append("\nНапиши название двух команд для прогноза!")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


async def cmd_prematch(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show today's prematch games from sstats.net."""
    try:
        from scrapers import sstats
        import datetime as _dt
        today = _dt.date.today().isoformat()
        games = sstats.fetch_games_by_date(today)
        if not games:
            await update.message.reply_text("Матчей на сегодня не найдено.")
            return

        lines = [f"Матчи на сегодня ({len(games)}):\n"]
        for g in games[:15]:
            home = (g.get("homeTeam") or {}).get("name", "?")
            away = (g.get("awayTeam") or {}).get("name", "?")
            league = (g.get("season") or {}).get("league", {}).get("name", "")
            lines.append(f"  {home} vs {away}")
            if league:
                lines.append(f"    {league}")

        lines.append("\nНапиши название двух команд для прогноза!")
        await update.message.reply_text("\n".join(lines))
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


def _quick_predict(home_name: str, away_name: str) -> Optional[str]:
    """Find match on sstats, call local API for full analysis."""
    from scrapers import sstats
    import requests as _req

    home_en = _resolve_team_name(home_name)
    away_en = _resolve_team_name(away_name)

    # Find game_id on sstats (today ±2 days)
    import datetime as _dt
    today = _dt.date.today()
    game_id = None
    for delta in range(-2, 3):
        d = (today + _dt.timedelta(days=delta)).isoformat()
        try:
            games = sstats.fetch_games_by_date(d)
            for g in games:
                h = (g.get("homeTeam") or {}).get("name", "").lower()
                a = (g.get("awayTeam") or {}).get("name", "").lower()
                if (home_en.lower() in h or h in home_en.lower()) and \
                   (away_en.lower() in a or a in away_en.lower()):
                    game_id = int(g["id"])
                    break
        except Exception:
            pass
        if game_id:
            break

    if not game_id:
        return None

    # Call local API — same endpoint as web version
    try:
        resp = _req.get(f"http://127.0.0.1:8000/api/prematch/{game_id}", timeout=120)
        data = resp.json()
    except Exception:
        return None

    analysis = data.get("ai_analysis", "")
    if not analysis:
        return None

    # Extract main bet
    prob_match = re.search(r'bet=([^:]+):confidence=([^\s]+)', analysis)
    main_bet = prob_match.group(1) if prob_match else ""
    confidence = prob_match.group(2) if prob_match else ""

    main_bet_ru = {"HOME": "Победа хозяев", "AWAY": "Победа гостей",
                   "DRAW": "Ничья"}.get(main_bet.upper(), main_bet)
    confidence_ru = {"high": "Высокая", "medium": "Средняя",
                     "low": "Низкая"}.get(confidence.lower(), confidence)

    clean = re.sub(r'PROB:home=[\d.]+:draw=[\d.]+:away=[\d.]+:bet=[^:]+:confidence=[^\s]+', '', analysis).strip()

    lines = [clean] if clean else []
    if main_bet_ru:
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━")
        lines.append("ГЛАВНЫЙ ПРОГНОЗ VISIONPRO")
        lines.append(f"  {main_bet_ru}")
        if confidence_ru:
            lines.append(f"  Уверенность: {confidence_ru}")
        lines.append("━━━━━━━━━━━━━━━━━━━━")

    return "\n".join(lines) or "Прогноз не удалось сформировать."


async def handle_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parse team names from message and generate prediction."""
    text = update.message.text.strip()

    # Parse "Team1 vs Team2" or "Team1 - Team2"
    teams = None
    for sep in ["vs", "VS", "Vs", " - ", " – ", "—", "/", "против"]:
        if sep in text:
            parts = text.split(sep, 1)
            teams = (parts[0].strip(), parts[1].strip())
            break

    if not teams or not teams[0] or not teams[1]:
        await update.message.reply_text(
            "Не могу распознать команды. Попробуй:\n"
            "  Ливерпуль vs Челси\n"
            "  Реал Мадрид - Барселона\n\n"
            "Или откройте приложение: /app"
        )
        return

    home_name, away_name = teams
    await update.message.reply_text(f"Анализирую: {home_name} vs {away_name}...")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _quick_predict(home_name, away_name),
        )
    except Exception as e:
        await update.message.reply_text(f"Ошибка при анализе: {e}")
        return

    if not result:
        await update.message.reply_text("Не удалось проанализировать матч.")
        return

    await update.message.reply_text(result)


async def handle_webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle data sent from Mini App via sendData()."""
    try:
        data = json.loads(update.effective_message.web_app_data.data)
        home_name = data.get("home", "")
        away_name = data.get("away", "")
        probabilities = data.get("probabilities", {})

        if not home_name or not away_name:
            await update.message.reply_text("Некорректные данные из приложения.")
            return

        # Build quick summary from data sent by Mini App
        hw = probabilities.get("home_win", 0)
        dr = probabilities.get("draw", 0)
        aw = probabilities.get("away_win", 0)

        text = (
            f"{home_name} vs {away_name}\n"
            f"Победа хозяев: {hw:.1f}%\n"
            f"Ничья: {dr:.1f}%\n"
            f"Победа гостей: {aw:.1f}%"
        )

        if data.get("analysis"):
            text += f"\n\n{data['analysis'][:500]}"

        await update.message.reply_text(text)
    except Exception as e:
        await update.message.reply_text(f"Ошибка обработки данных: {e}")


async def post_init(application: Application) -> None:
    """Set bot commands."""
    await application.bot.set_my_commands([
        BotCommand("start", "Начать"),
        BotCommand("help", "Помощь"),
        BotCommand("trending", "Ближайшие матчи"),
        BotCommand("prematch", "Матчи на сегодня"),
        BotCommand("app", "Открыть приложение"),
    ])


def main() -> None:
    token = _load_token()
    if not token:
        print("ERROR: Telegram bot token not found.")
        print("Put token in Апи/telegram_token.txt or set TELEGRAM_BOT_TOKEN env var.")
        print("Get token from @BotFather on Telegram.")
        return

    if not WEBAPP_URL.startswith("https://"):
        print(f"WARNING: WEBAPP_URL={WEBAPP_URL}")
        print("Telegram Mini App requires HTTPS. Set WEBAPP_URL to your tunnel URL.")
        print("Example: set WEBAPP_URL=https://your-id.ngrok.io/mini-app")

    # Initialize DB
    db.init_db()

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    async def error_handler(update, context):
        logger.error(f"Error: {context.error}")

    app.add_error_handler(error_handler)

    # Command handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("app", cmd_app))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("trending", cmd_trending))
    app.add_handler(CommandHandler("prematch", cmd_prematch))

    # Text prediction handler (fallback)
    app.add_handler(CallbackQueryHandler(handle_prediction, pattern="^example$"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_prediction))

    print(f"Telegram bot started.")
    print(f"WEBAPP_URL: {WEBAPP_URL}")
    print("Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
