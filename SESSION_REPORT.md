# Отчёт по сессии: проверка качества прогнозов VisionPro

**Дата:** 13 июля 2026  
**Длительность:** ~12 часов  
**Участники:** Залман + MiMoCode

---

## 1. Рефакторинг архитектуры

### Было → Стало

| Компонент | До | После |
|-----------|-----|-------|
| `app.py` | 2594 строк, 70+ эндпоинтов | ~150 строк (только FastAPI + middleware + роутеры) |
| `ai_analyzer.py` + `tennis_ai.py` | Дублирование `_get_api_key`, `_chat`, TLS | Общий модуль `ai_core.py` |
| `train.py` `build_dataset()` | O(N²) — 31K матчей >5 мин (таймаут) | O(N) — 31K матчей за 346 сек |

### Новые модули

| Файл | Назначение |
|------|-----------|
| `ai_core.py` | Общая LLM-инфраструктура: TLS-сессия, API key, chat с ретраями, PROB-парсер для футбола/тенниса |
| `calibration.py` | `OofCalibrator` — isotonic regression калибратор на OOF-предсказаниях |
| `state.py` | Разделяемое состояние: `JOB`, `MODEL`, `load_model()`, `load_tennis_model()` |
| `helpers.py` | Константы (`LEAGUE_NAME_MAP`, `TEAM_NAME_MAP`), prediction-хелперы (`predict_pair`, `with_prediction`) |
| `logging_config.py` | Ротация логов: `logs/server.log` (5MB/3 backups), `logs/errors.log` (2MB/5 backups) |

### Роутеры

| Файл | Эндпоинты |
|------|-----------|
| `routers/football.py` | `/api/leagues`, `/api/teams`, `/api/predict`, `/api/standings`, `/api/prematch/*`, `/api/sstats/*`, `/api/fs/*` |
| `routers/tennis.py` | `/api/tennis/*` (rankings, predict, analyze, live, prematch) |
| `routers/ai.py` | `/api/ai/*`, `/api/bot/predict`, `/api/predictions/*`, `/api/predictions/settle`, `/api/predictions/hitrate` |
| `routers/admin.py` | `/api/collect/*`, `/api/train`, `/api/backtest/*`, `/api/model-stats`, `/api/refresh-status` |

### Тесты

| Файл | Что проверяет |
|------|--------------|
| `tests/test_ai_core.py` | 11 тестов PROB-парсера (футбол + теннис) |
| `tests/test_prediction.py` | `build_features`, ансамбль inference |
| `tests/test_helpers.py` | Консенсус коэффициентов, name maps |
| `tests/conftest.py` | Фикстуры для тестовой БД |

---

## 2. Калибровка вероятностей

### Проблема
Модель систематически завышала вероятности: ECE 0.073, бакет 0.4-0.5 показывал gap +9.1% (pred=44%, actual=35%).

### Решение
Isotonic regression на OOF-предсказаниях с клиппингом (0.02-0.98).

### Результаты (честная nested CV)

| Метрика | Temperature-only | **Isotonic + clip** |
|---------|-----------------|---------------------|
| ECE | 0.0727 | **0.0096** |
| Log Loss | 0.6604 | **0.6489** |
| Brier Score | 0.2329 | **0.2276** |

### Ключевой баг
Isotonic без клиппинга выдавал вероятности ровно 0.0/1.0 на маленьких бакетах → log_loss = ∞. Клиппинг (0.02-0.98) исправил.

### Интеграция
- Калибратор обучается при `Ensemble.fit()` на OOF-данных
- Применяется в `predict_proba(apply_calibration=True)` по умолчанию
- Сохраняется/загружается вместе с model.pkl

---

## 3. Анализ эджа (Edge Analysis)

### Baselines

| Стратегия | Accuracy |
|-----------|----------|
| Всегда хозяева | 45.0% |
| Рынок (implied odds) | **50.5%** |
| Модель | **47.0%** |

**Вывод:** модель на +2% лучше "всегда хозяева", но на -3.5% хуже рынка.

### Walk-forward backtest (5 фолдов)

| Фолд | Accuracy | Log Loss | Brier |
|------|----------|----------|-------|
| 1 | 43.3% | 1.124 | 0.656 |
| 2 | 48.8% | 1.048 | 0.620 |
| 3 | 50.0% | 1.027 | 0.604 |
| 4 | 43.6% | 1.016 | 0.611 |
| 5 | 48.8% | 1.004 | 0.601 |
| **Итого** | **47.0%** | **1.042** | **0.618** |

Деградации со временем нет (43→49→50→44→49 — шум).

### Edge-анализ (11,304 прогнозов с коэффициентами)

| Edge | N | Win% | ROI | 95% CI |
|------|---|------|-----|--------|
| <0% | 2,816 | 62.1% | +0.1% | [-3.0%, +3.2%] |
| 0-2% | 452 | 55.8% | -4.5% | [-13.3%, +3.5%] |
| 2-5% | 710 | 54.1% | -0.1% | [-7.4%, +8.4%] |
| 5-10% | 2,100 | 42.6% | -0.1% | [-5.3%, +5.8%] |
| 10%+ | 5,226 | 39.2% | +1.4% | [-2.5%, +4.9%] |

**Ни один бакет не даёт статистически значимого положительного ROI.** Все CI включают 0.

### Распределение лиг в edge-анализе

| Тип | % |
|-----|---|
| Top-5 лиги (EPL, La Liga, Bundesliga, Serie A, Ligue 1) | 69.1% |
| Другие лиги (Championship, Eredivisie, Super_Lig и др.) | 30.9% |

---

## 4. Settlement (сверка прогнозов с результатом)

### Реализация
- Добавлено поле `settled_at` в таблицу `predictions` (миграция через ALTER TABLE)
- Функция `db.settle_predictions()` — ищет результат по `home_name`/`away_name`/`match_date` в `matches` и `sstats_matches`
- Эндпоинт `POST /api/predictions/settle` — ручной запуск
- Планищак `daily 03:00` через APScheduler — автоматическая сверка
- Статистика: `GET /api/predictions/hitrate` — hit-rate по confidence и по месяцам

### Текущее состояние
В БД 0 прогнозов (prophet ещё не накоплен) — инфраструктура готова.

---

## 5. Интеграция FlashScore live в теннис

### Новые файлы
- `tennis_live.py` — сбор live-данных из Tennis API + FlashScore
- `TENNIS_LIVE_PROMPT` — отдельный системный промпт для live-анализа тенниса

### Эндпоинты
- `GET /api/tennis/live` — текущие live-матчи
- `search_and_analyze()` автоматически проверяет live-статус и добавляет контекст (счёт, наборы, momentum)

---

## 6. O(N) фикс build_dataset()

### Проблема
```python
prior_same_league = [p for p in past_desc if p["league_slug"] == m["league_slug"]]
```
O(N) на каждый из 31K матчей → O(N²) суммарно → таймаут >5 мин.

### Решение
Инкрементальные индексы:
```python
_by_league: Dict[str, List[Dict]] = defaultdict(list)
_by_team: Dict[int, deque] = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
_by_h2h: Dict[frozenset, deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))
```
Каждый матч O(1) амортизированно.

### Результат
31K матчей: **>5 мин (таймаут) → 346 сек**

---

## 7. Разведка внешних источников

### sstats.net (17 функций)
| Функция | Статус |
|---------|--------|
| `fetch_profits` | **Независимый бенчмарк ROI** (flat stake на реальных коэффициентах) |
| `fetch_season_table` | **Не используется**, но содержит over/under статистику + form arrays — новые фичи |
| `fetch_last_games_stats` | Вернул None на test data — ненадёжный |
| `fetch_query` | Быстрее paginated `fetch_games_by_league` — кандидат на упрощение |
| `fetch_upcoming_by_team` | Работает, нигде не используется |

### BSD (Bzzoiro Sports Data)
- **Полезно:** тренерские фичи (pressing_intensity, defensive_line, formation), рефери (career stats), ML-прогноз (8 рынков), ai_preview
- **Дублирует:** коэффициенты (sstats лучше), погода (Open-Meteo)
- **Не полезно:** исторические данные (нет голов)

---

## 8. Проблема дублей команд в БД

Одна "Argentina" = 9 разных записей с разными ID (fifa.world, International, WC_2026). Приводит к тому, что разные вызовы `generate_preview()` возвращают разные прогнозы для "одного и того же" матча.

**Рекомендация:** добавить поле `source` в таблицу teams, при поиске показывать варианты пользователю.

---

## 9. Итоговый вывод по качеству модели

**Доказанного эджа против рынка нет.** Модель:
- Не обыгрывает trivial baseline "всегда хозяева" с существенным отрывом
- Проигрывает рыночным коэффициентам на -3.5%
- Ни один бакет edge (N≥100) не даёт статистически значимого положительного ROI
- Калибровка улучшена (ECE 0.07 → 0.01), но это не создаёт эдж

**Следующий шаг:** копить объём (больше матчей с коэффициентами), повторить через 3-6 месяцев когда N > 2000.
