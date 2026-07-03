# Football AI Predictor

Локальный сайт для прогнозов футбольных матчей. Реальные данные с **xG** (expected goals),
букмекерские котировки, Elo-рейтинги и AI-анализ.

- **Источники данных:** Understat (xG), ESPN (матчи 40+ лиг), ClubElo, FBref (xG 25+ лиг),
  sstats.net (Glicko, букмекерские котировки), Open-Meteo (погода), Transfermarkt (травмы)
- **Хранилище:** SQLite (один файл `data/football.db`)
- **Модель:** Ансамбль XGBoost + LightGBM + Dixon-Coles (60+ фич, time-decay, expanding-window CV)
- **Бэкенд:** FastAPI + APScheduler (автообновление каждые 6 часов)
- **AI-анализ:** Polza.ai API (DeepSeek) с контекстом из sstats.net, ESPN, RSS

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Собрать данные (полный сбор ≈ 5-15 минут)
python data_collector.py --full

# 3. Обучить модель
python train.py

# 4. Запустить сервер
python start.bat
# или: uvicorn app:app --port 8000

# 5. Открыть http://localhost:8000
```

## Структура

```
football-ai/
├── app.py              # FastAPI: REST API + фронтенд + планировщик
├── train.py            # Обучение ансамбля (XGB+LGBM+DixonColes)
├── data_collector.py   # Оркестрация скрейперов → SQLite
├── db.py               # SQLite-схема и CRUD
├── config.py           # Централизованная конфигурация
├── backtest.py         # Walk-forward backtest
├── ai_analyzer.py      # LLM-анализ через Polza.ai
├── web_scraper.py      # ESPN: данные команда
├── news_scraper.py     # RSS: новости
├── models/
│   ├── ensemble.py     # Ансамбль моделей
│   └── dixon_coles.py  # Статистическая модель Dixon-Coles
├── scrapers/           # Все скрейперы данных
│   ├── understat.py    # Understat (xG, 6 лиг)
│   ├── espn.py         # ESPN (40+ лиг)
│   ├── fbref.py        # FBref (xG, 25+ лиг)
│   ├── clubelo.py      # ClubElo рейтинги
│   ├── sstats.py       # sstats.net (котировки, Glicko)
│   ├── openmeteo.py    # Погода
│   ├── transfermarkt.py# Травмы
│   ├── historical_odds.py # Исторические котировки
│   ├── web.py          # Jina Reader + DuckDuckGo
│   └── ...
├── frontend/           # SPA: index.html + PWA
├── requirements.txt
├── start.bat / stop.bat
└── data/
    └── football.db     # База данных (после data_collector.py)
```

## API эндпоинты

| Метод | URL | Описание |
|-------|-----|----------|
| GET | `/` | Веб-интерфейс |
| GET | `/api/leagues` | Список лиг |
| GET | `/api/teams?league=EPL` | Команды лиги |
| GET | `/api/search-teams?q=…` | Поиск команд |
| GET | `/api/standings?league=EPL` | Турнирная таблица с xG |
| GET | `/api/upcoming?league=EPL` | Ближайшие матчи + прогнозы |
| GET | `/api/results?league=EPL` | Сыгранные матчи |
| GET | `/api/predict?home_id=…&away_id=…` | Прогноз пары |
| GET | `/api/team-stats?team_id=…` | Статистика команды |
| GET | `/api/refresh-status` | Статус последнего обновления |
| GET | `/api/model-stats` | Метрики модели (accuracy, log-loss) |
| GET | `/api/backtest` | Walk-forward backtest |
| GET | `/api/injuries` | Травмы |
| GET | `/api/team-elo` | Elo-рейтинг |
| GET | `/api/weather` | Погода для матча |
| GET | `/api/fs/*` | FlashScore: страны, лиги, матчи, live |
| GET | `/api/sstats/*` | sstats.net: котировки, Glicko, анализ |
| GET | `/api/predictions/*` | Сохранённые прогнозы |
| GET | `/api/ai/analyze` | AI-анализ матча |
| GET | `/api/ai/models` | Доступные AI-модели |
| POST | `/api/collect/start` | Запуск сбора данных |
| POST | `/api/collect/stop` | Остановка сбора |
| POST | `/api/train` | Переобучение модели |
| POST | `/api/predictions/save` | Сохранить прогноз |

## Признаки модели (60+)

Ансамбль использует 60+ фичей по 16 категориям (Stage A-Q):
- Средние голы и xG за/против (скользящее окно 10 матчей)
- Винрейт общий, дома, в гостях
- Форма и H2H
- Elo-разница и тренд
- Дни отдыха, стрики, позиция в таблице
- xG over/under-performance
- Тренды голей, пропущенных, формы
- Моментум (short-term vs long-term)
- Книжные котировки (implied probabilities)
- Усталость, турнирный контекст
- Взаимодействия фичей

## CLI

```bash
python data_collector.py --full          # Полный сбор
python data_collector.py --current-only  # Только текущий сезон
python data_collector.py --live-only     # Только FlashScore live
python data_collector.py --health        # Health-check данных
python data_collector.py --export        # Экспорт в JSON

python train.py                          # Обучение модели
python backtest.py                       # Walk-forward backtest
python tune.py --trials 50               # Optuna-тюнинг
python ablation.py                       # Feature ablation
python backup.py                         # Снэпшот проекта
```
