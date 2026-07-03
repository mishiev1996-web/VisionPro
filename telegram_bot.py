"""
telegram_bot.py — Telegram bot with Mini App for Football AI Predictor.

Setup:
  1. Create bot via @BotFather, get token
  2. Put token in Апи/telegram_token.txt (or set TELEGRAM_BOT_TOKEN env var)
  3. Run: python telegram_bot.py

Mini App opens via /start or the "Прогноз" button.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from telegram import (
    BotCommand,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    MenuButtonCommands,
    Update,
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
from train import build_features, FEATURE_NAMES
from ai_analyzer import search_and_predict, _resolve_team_name

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


WEBAPP_URL = os.environ.get("WEBAPP_URL", "http://localhost:8000/mini-app")


# ── Handlers ─────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message."""
    user = update.effective_user
    text = (
        f"Football AI Predictor\n\n"
        f"Привет, {user.first_name}!\n\n"
        f"Я предсказываю результаты футбольных матчей с помощью ML.\n"
        f"Мои данные: xG, Elo, форма команд, котировки букмекеров.\n\n"
        f"Напиши названия двух команд, и я предскажу результат.\n\n"
        f"Пример: Ливерпуль vs Челси"
    )

    buttons = [[InlineKeyboardButton("Пример: Ливерпуль vs Ман Сити", callback_data="example")]]

    await update.message.reply_text(
        text,
        reply_markup=InlineKeyboardMarkup(buttons),
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Как пользоваться:\n\n"
        "1. Напиши: Команда1 vs Команда2\n"
        "   Пример: Ливерпуль vs Челси\n\n"
        "2. Или: Команда1 - Команда2\n"
        "   Пример: Реал Мадрид - Барселона\n\n"
        "Команды:\n"
        "/start — начать\n"
        "/help — помощь\n"
        "/trending — популярные матчи сегодня"
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
            "  Реал Мадрид - Барселона"
        )
        return

    home_name, away_name = teams
    await update.message.reply_text(f"Анализирую: {home_name} vs {away_name}...")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_and_predict(
                home_name, away_name,
                progress_cb=lambda e: None,
            ),
        )
    except Exception as e:
        await update.message.reply_text(f"Ошибка при анализе: {e}")
        return

    if not result:
        await update.message.reply_text("Не удалось найти команды в базе данных.")
        return

    # Build response text
    lines = []

    # Teams
    h_name = result.get("home", {}).get("name", home_name)
    a_name = result.get("away", {}).get("name", away_name)
    lines.append(f"{h_name} vs {a_name}")
    lines.append("")

    # Prediction from model
    pred = result.get("prediction")
    if pred and "probabilities" in pred:
        prob = pred["probabilities"]
        hw = prob.get("home_win", 0)
        dr = prob.get("draw", 0)
        aw = prob.get("away_win", 0)

        lines.append(f"Прогноз модели:")
        lines.append(f"  Победа хозяев: {hw:.1f}%")
        lines.append(f"  Ничья: {dr:.1f}%")
        lines.append(f"  Победа гостей: {aw:.1f}%")
        lines.append("")

        # Visual bar
        lines.append(_make_bar(hw, dr, aw))

        # Main prediction
        if hw > dr and hw > aw:
            lines.append(f"\nПрогноз: Победа {h_name} ({hw:.1f}%)")
        elif aw > dr:
            lines.append(f"\nПрогноз: Победа {a_name} ({aw:.1f}%)")
        else:
            lines.append(f"\nПрогноз: Ничья ({dr:.1f}%)")
    else:
        lines.append("Модель не смогла рассчитать прогноз.")

    # AI analysis (truncated)
    analysis = result.get("analysis")
    if analysis:
        lines.append("")
        lines.append("--- Аналитика ---")
        # Truncate to fit Telegram message limit (4096 chars)
        max_analysis = 2000
        if len(analysis) > max_analysis:
            analysis = analysis[:max_analysis] + "..."
        lines.append(analysis)

    response = "\n".join(lines)

    # Telegram has a 4096 char limit
    if len(response) > 4096:
        response = response[:4090] + "..."

    await update.message.reply_text(response)


def _make_bar(hw: float, dr: float, aw: float) -> str:
    """Create a text-based probability bar."""
    total = hw + dr + aw
    if total <= 0:
        return ""
    hw_r, dr_r, aw_r = hw / total * 100, dr / total * 100, aw / total * 100
    bar_len = 20
    h_len = max(1, round(hw_r / 100 * bar_len))
    d_len = max(1, round(dr_r / 100 * bar_len))
    a_len = bar_len - h_len - d_len
    if a_len < 1:
        a_len = 1
        d_len = bar_len - h_len - a_len
    return (
        f"{'█' * h_len}{'░' * d_len}{'▓' * a_len}\n"
        f"{hw_r:.0f}%   {dr_r:.0f}%   {aw_r:.0f}%"
    )


async def handle_webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle data sent from Mini App."""
    try:
        data = json.loads(update.effective_message.web_app_data.data)
        home_name = data.get("home", "")
        away_name = data.get("away", "")

        if not home_name or not away_name:
            await update.message.reply_text("Некорректные данные.")
            return

        await update.message.reply_text(f"Анализирую: {home_name} vs {away_name}...")

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: search_and_predict(home_name, away_name, progress_cb=lambda e: None),
        )

        if result and result.get("prediction"):
            pred = result["prediction"]
            prob = pred.get("probabilities", {})
            h_name = result.get("home", {}).get("name", home_name)
            a_name = result.get("away", {}).get("name", away_name)

            text = (
                f"{h_name} vs {a_name}\n"
                f"Хозяева: {prob.get('home_win', 0):.1f}%\n"
                f"Ничья: {prob.get('draw', 0):.1f}%\n"
                f"Гости: {prob.get('away_win', 0):.1f}%"
            )
            await update.message.reply_text(text)
        else:
            await update.message.reply_text("Не удалось рассчитать прогноз.")
    except Exception as e:
        await update.message.reply_text(f"Ошибка: {e}")


async def post_init(application: Application) -> None:
    """Set bot commands menu."""
    await application.bot.set_my_commands([
        BotCommand("start", "Начать"),
        BotCommand("help", "Помощь"),
        BotCommand("trending", "Ближайшие матчи"),
    ])

    # Set menu button to commands mode (web_app requires HTTPS)
    try:
        await application.bot.set_chat_menu_button(
            chat_id=None,
            menu_button=MenuButtonCommands(),
        )
    except Exception as e:
        logger.warning(f"Failed to set menu button: {e}")


def main() -> None:
    token = _load_token()
    if not token:
        print("ERROR: Telegram bot token not found.")
        print("Put token in Апи/telegram_token.txt or set TELEGRAM_BOT_TOKEN env var.")
        print("Get token from @BotFather on Telegram.")
        return

    # Initialize DB
    db.init_db()

    app = (
        Application.builder()
        .token(token)
        .post_init(post_init)
        .build()
    )

    # Error handler
    async def error_handler(update, context):
        logger.error(f"Error: {context.error}")

    app.add_error_handler(error_handler)

    # Handlers
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("trending", cmd_trending))
    app.add_handler(CallbackQueryHandler(handle_prediction, pattern="^example$"))
    app.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_prediction))

    print("Telegram bot started. Press Ctrl+C to stop.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
