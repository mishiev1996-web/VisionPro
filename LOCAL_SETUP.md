# Локальный запуск VisionPro (Football AI Predictor + Tennis)

## Необходимые компоненты

- Python 3.10+
- Telegram-бот (токен от @BotFather) — опционально
- HTTPS-туннель (ngrok, Cloudflare Tunnel) — для Telegram Mini App

## 1. Переменные окружения

Скопируйте `.env.example` в `.env` и заполните:

```
TELEGRAM_BOT_TOKEN=ваш_токен_бота
WEBAPP_URL=https://ваш-туннель.ngrok.io/mini-app
ALLOWED_ORIGINS=https://ваш-туннель.ngrok.io,http://localhost:8000
```

Или используйте файлы в папке `Апи/`:
- `Апи/telegram_token.txt` — токен бота
- `Апи/key.txt` — ключ Polza.ai API
- `Апи/sstats_key.txt` — ключ sstats.net API
- `Апи/tennis_key.txt` — ключ Tennis API (RapidAPI)

## 2. Установка и запуск

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервера
python -m uvicorn app:app --port 8000 --reload
```

Или через `start.bat` (Windows).

## 3. HTTPS-туннель (для Telegram Mini App)

Telegram Mini App требует HTTPS. Используйте один из вариантов:

### ngrok
```bash
ngrok http 8000
```
Скопируйте HTTPS-URL и обновите `WEBAPP_URL` и `ALLOWED_ORIGINS`.

### Cloudflare Tunnel
```bash
cloudflared tunnel --url http://localhost:8000
```

## 4. Telegram-бот (опционально)

```bash
python telegram_bot.py
```

Или через `start_bot.bat` (Windows).

## 5. Тесты

```bash
pytest tests/ -v
```

## Архитектура (после рефакторинга)

```
app.py (~150 строк)
    ├─ Создание FastAPI, middleware, lifespan
    ├─ Подключение роутеров (routers/)
    └─ Rate limiter, корневые эндпоинты (/ и /mini-app)

routers/
    ├─ football.py  — /api/leagues, /api/teams, /api/predict,
    │                  /api/prematch/*, /api/sstats/*, /api/fs/*
    ├─ tennis.py    — /api/tennis/*
    ├─ ai.py        — /api/ai/*, /api/bot/predict, /api/predictions/*
    └─ admin.py     — /api/collect/*, /api/train, /api/backtest/*,
                      /api/model-stats, /api/refresh-status

ai_core.py          — Общая LLM-инфраструктура (TLS, API key, chat, PROB-парсер)
helpers.py          — Общие константы (LEAGUE_NAME_MAP, TEAM_NAME_MAP) и prediction-хелперы
state.py            — Разделяемое состояние (JOB, MODEL, load_model)
config.py           — Конфигурация (пути, ключи, фичи, лиги)
```

## Переменные окружения

| Переменная | Описание | Обязательна |
|-----------|----------|-------------|
| `TELEGRAM_BOT_TOKEN` | Токен бота от @BotFather | Для бота |
| `WEBAPP_URL` | HTTPS-URL Mini App | Для Mini App |
| `ALLOWED_ORIGINS` | Разрешённые CORS-источники | Нет (default: localhost) |
| `POLZA_API_KEY` | Ключ Polza.ai (LLM-анализ) | Нет (из Апи/key.txt) |
| `SSTATS_API_KEY` | Ключ sstats.net | Нет (из Апи/sstats_key.txt) |
| `TENNIS_API_KEY` | Ключ Tennis API (RapidAPI) | Нет (из Апи/tennis_key.txt) |
