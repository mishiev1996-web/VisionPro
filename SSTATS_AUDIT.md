# SSTATS_AUDIT.md — Аудит использования sstats.net в проекте Vision

Дата аудита: 2026-07-14

---

## 1. Инвентаризация API клиента

### Методы в `scrapers/sstats.py`

| # | Метод | Эндпоинт API | Описание |
|---|-------|-------------|----------|
| 1 | `account_info()` | `/Account/Info` | Проверка аккаунта |
| 2 | `fetch_leagues()` | `/Leagues` | Каталог всех лиг (~1400) |
| 3 | `fetch_games_by_date(date)` | `/Games/list?date=` | Матчи за день |
| 4 | `fetch_upcoming_by_team(team_id)` | `/Games/list?upcoming=true&team=` | Предстоящие матчи команды |
| 5 | `fetch_upcoming_all()` | `/Games/list?upcoming=true` | Все предстоящие |
| 6 | `fetch_live_matches()` | `/Games/list?live=true` | Лайв-матчи |
| 7 | `fetch_h2h(t1, t2)` | `/Games/list?ended=true&bothTeams=` | История очных встреч |
| 8 | `fetch_games_by_league(lid, page)` | `/Games/list?leagueId=` | История лиги (постранично) |
| 9 | `fetch_game(game_id)` | `/Games/{id}` | Полные данные матча |
| 10 | `fetch_glicko(game_id)` | `/Games/glicko/{id}` | Рейтинги Glicko-2 |
| 11 | `fetch_text_summary(game_id)` | `/Games/text-summary` | Текстовая сводка |
| 12 | `fetch_odds(game_id)` | `/Odds/{id}` | Котировки 8+ БК |
| 13 | `fetch_teams_in_league(lid)` | `/Teams/list?leagueId=` | Команды лиги |
| 14 | `fetch_last_games_stats(gid, ...)` | `/Games/last-games-stats` | Усреднённая статистика за N матчей |
| 15 | `fetch_injuries(game_id)` | `/Games/injuries` | Травмы/дисквалификации |
| 16 | `fetch_season_table(lid, year)` | `/Games/season-table` | Рейтинговая таблица + over/under |
| 17 | `fetch_profits(gid, ...)` | `/Games/profits` | ROI-бенчмарк по типам ставок |
| 18 | `fetch_query(cond, fields)` | `/Games/query` (POST) | SQL-подобный bulk-запрос |

**Хелперы (не API):**
- `consensus_odds(odds_blocks)` — усреднение котировок по БК
- `market_dispersion(odds_blocks)` — разброс котировок между БК

### Эндпоинты в API, которых НЕТ в клиенте

| Эндпоинт | Описание | Наша реализация |
|----------|----------|-----------------|
| `/Excel/Delux` | Комплексный запрос для Excel-таблицы (xG, статистика, коэффициенты за раз) | Нет |
| `/Excel/FootballCalc` | Аналог для другой Excel-таблицы | Нет |
| `/Excel/Results` | Быстрое извлечение результатов (Id, Score, Ended) | Нет |
| `/Ls/List` | Альтернативный список матчей (другой формат ID — Flashscore string IDs) | Нет |
| `/Seasons/standings` | Официальная турнирная таблица (vs вычисляемая season-table) | Нет |

---

## 2. Таблица использования: метод → вызывается? → сохраняется? → в фичах?

| Метод | Вызывается? | Где | Сохраняется в БД? | Таблица | Используется в фичах? | Комментарий |
|-------|------------|-----|-------------------|---------|----------------------|-------------|
| `fetch_leagues()` | ✅ | sstats_collector.py, ai_analyzer.py | ✅ | leagues (upsert) | ❌ | Только для каталога лиг |
| `fetch_games_by_date()` | ✅ | data_collector.py, telegram_bot.py | ✅ | sstats_matches | ❌ напрямую | Через sstats_matches → _team_sstats_games |
| `fetch_upcoming_by_team()` | ⚠️ | archive/scripts/test (тест) | ❌ | — | ❌ | Не используется в production |
| `fetch_upcoming_all()` | ✅ | routers/football.py (/api/sstats/upcoming) | ❌ | — | ❌ | Только API-эндпоинт |
| `fetch_live_matches()` | ✅ | routers/football.py (/api/sstats/live) | ❌ | — | ❌ | Только API-эндпоинт |
| `fetch_h2h()` | ✅ | routers/football.py (/api/sstats/h2h) | ❌ | — | ❌ | Только API-эндпоинт |
| `fetch_games_by_league()` | ✅ | sstats_collector.py | ✅ | sstats_matches | ❌ напрямую | Основной источник sstats_matches |
| `fetch_game()` | ✅ | data_collector.py, routers/football.py | ✅ | sstats_statistics, sstats_events | ✅ частично | xG, shots_on_target, goals_by_half |
| `fetch_glicko()` | ✅ | routers/football.py | ❌ | — | ❌ | Только API-эндпоинт |
| `fetch_text_summary()` | ✅ | routers/football.py | ❌ | — | ❌ | Только API-эндпоинт |
| `fetch_odds()` | ✅ | data_collector.py, sstats_collector.py | ✅ | sstats_odds | ❌ | Котировки лежат, но не читаются train.py |
| `fetch_teams_in_league()` | ✅ | sstats_collector.py | ✅ | teams (upsert) | ❌ | Только каталог команд |
| `fetch_last_games_stats()` | ⚠️ | routers/football.py, archive/scripts/test | ❌ | — | ❌ | API-эндпоинт, не сохраняется |
| `fetch_injuries()` | ✅ | routers/football.py | ❌ | — | ❌ | Данные не сохраняются! |
| `fetch_season_table()` | ⚠️ | archive/scripts/test (тест) | ❌ | — | ❌ | **Не используется** |
| `fetch_profits()` | ⚠️ | routers/football.py, archive/scripts/test | ❌ | — | ❌ | **Не используется для бенчмарка** |
| `fetch_query()` | ✅ | data_collector.py (sstats_bulk), ai_analyzer.py | ✅ | sstats_matches | ❌ | Bulk-запрос для массового сбора |
| `consensus_odds()` | ✅ | data_collector.py | ✅ | match_odds (source='sstats_consensus') | ✅ | Используется как market_implied_* |

### Ключевые находки

1. **`fetch_odds()` — 1M+ строк в БД, но НЕ читаются train.py**: Котировки sstats лежат в `sstats_odds`, но `build_features()` читает только `match_odds` (football-data.co.uk). Это огромный неиспользуемый массив данных.

2. **`fetch_injuries()` — данные не сохраняются**: Травмы запрашиваются для API-эндпоинта, но не попадают в БД. Transfermarkt — единственный источник injuries (49 записей).

3. **`fetch_season_table()` — не используется**: Содержит over/under статистику + форму команд. Потенциально ценные фичи.

4. **`fetch_profits()` — не используется как бенчмарк**: Может дать независимую оценку ROI для сверки с нашим edge-анализом.

5. **`fetch_last_games_stats()` — не сохраняется**: Усреднённая статистика за N матчей (xG, удары, форма) — уже рассчитывается нами вtrain.py, но API отдаёт готовый результат.

---

## 3. Покрытие sstats_odds по лигам

### sstats_odds vs match_odds

| Источник | Уникальных матчей | Покрытие |
|----------|-------------------|----------|
| **match_odds** (football-data.co.uk) | 101,536 | 15 лиг (EPL, La Liga, Bundesliga, Serie A, Ligue 1, Championship, Eredivisie, Primeira Liga, Super Lig, Belgian First, Greek Super, MLS, RFPL, Brasileirao) |
| **sstats_odds** | 276 | 27 лиг (те же 15 + World Cup, Copa Chile, Serie B/C/D, MLS Next Pro, и др.) |

### Дополнительные лиги в sstats_odds (нет в match_odds)

| Лига | Матчей | Потенциал |
|------|--------|-----------|
| World Cup | 6 | Высокий (ẖорошее покрытие) |
| Copa Chile | 4 | Средний |
| Serie B | 2 | Средний |
| MLS Next Pro | 6 | Низкий |
| Serie C/D, USL League Two, и др. | 1-3 | Низкий |

### Типы рынков в sstats_odds (top-10)

| Рынок | Записей |
|-------|---------|
| Exact Score | 119,720 |
| Goals Over/Under | 56,986 |
| Correct Score - First Half | 45,239 |
| Asian Handicap | 40,067 |
| Player Singles | 38,519 |
| Corners Over Under | 36,738 |
| Goals Over/Under First Half | 30,718 |
| HT/FT Double | 22,812 |
| Handicap Result | 22,555 |
| Result/Total Goals | 21,978 |

### Букмекеры в sstats_odds (top-5)

| БК | Матчей |
|----|--------|
| Marathonbet | 275 |
| Pinnacle | 273 |
| Bet365 | 255 |
| William Hill | 246 |
| Betano | 242 |

**Вывод**: sstats_odds покрывает только 276 матчей — это на 2 порядка меньше, чем match_odds (101K). **Расширение odds-выборки через sstats_odds невозможно** — данных слишком мало. Однако sstats_odds содержит ** больше типов рынков** (Asian Handicap, Corners, Player Shots), которые могут быть полезны для будущих фичей.

---

## 4. Поля в используемых таблицах

### sstats_statistics — какие колонки реально попадают в фичи

| stat_name | Записей | Используется в train.py? | Какая фича |
|-----------|---------|--------------------------|------------|
| xg | 204 | ✅ Да | `home_xg`, `away_xg`, `xg_diff` |
| shots_on_target | 316 | ❌ **НЕТ** | — |
| total_shots | 316 | ❌ **НЕТ** | — |
| possession | 316 | ❌ **НЕТ** | — |
| corners | 316 | ❌ **НЕТ** | — |
| yellow_cards | 309 | ❌ **НЕТ** | — |
| fouls | 302 | ❌ **НЕТ** | — |
| offsides | 299 | ❌ **НЕТ** | — |
| total_passes | 217 | ❌ **НЕТ** | — |
| passes_accurate | 217 | ❌ **НЕТ** | — |
| big_chances | 217 | ❌ **НЕТ** | — |
| red_cards | 136 | ❌ **НЕТ** | — |

**Вывод**: Из 12 типов статистики используется только **1 (xG)**. Остальные 11 лежат в БД, но не читаются.

### sstats_events — какие типы событий используются

| event_type | Записей | Используется? |
|-----------|---------|---------------|
| 3 (Goal) | 2,217 | ✅ Да → `avg_goals_1h` |
| 2 (Card) | 1,068 | ❌ Нет |
| 1 (Substitution) | 750 | ❌ Нет |
| 4 (Other) | 66 | ❌ Нет |

**Вывод**: Из 4 типов событий используется только **1 (Goal)**. Карточки и замены не используются.

### sstats_matches — какие поля читаются

| Поле | Используется? | Как |
|------|---------------|-----|
| game_id | ✅ | Связка с другими таблицами |
| home_id, away_id | ✅ | Маппинг команд |
| date | ✅ | Фильтрация по дате |
| home_result, away_result | ✅ | Результат матча |
| league_name | ✅ | Определение лиги |
| raw_json | ❌ | Лежит, не читается |

---

## 5. Детальный анализ неиспользуемых методов

### 5.1. fetch_season_table — over/under + форма команд

**Структура ответа** (из OpenAPI):
- `TeamId`, `TeamName` — идентификатор команды
- `TotalGames`, `Wins`, `Draws`, `Loss` — базовая статистика
- `GoalsScored`, `GoalsMissed`, `Points` — голы и очки
- **`Over25TotalGames`, `Under25TotalGames`** — процент матчей с тоталом >2.5 / <2.5
- **`Over25TotalHomeGames`, `Over25TotalAwayGames`** — дом/гость
- `HomeGames`, `AwayGames`, `HomeWins`, `AwayWins` — домашняя/выездная статистика
- `HomeScored`, `AwayScored`, `HomeMissed`, `AwayMissed` — голы по дом/гость
- `ScoreDiff`, `ScoreDiffHome`, `ScoreDiffAway` — разность голов
- `Rank`, `HomeRank` — позиция в таблице

**Потенциальные фичи**:
- `home_over25_rate` = Over25TotalHomeGames / HomeGames
- `away_over25_rate` = Over25TotalAwayGames / AwayGames
- `home_btts_rate` — (нет в API, но можно вычислить)
- `home_form_score` — на основе Wins/Draws/Loss
- `home_attack_strength` = HomeScored / HomeGames
- `home_defense_strength` = HomeMissed / HomeGames

**Приоритет**: 🔴 Высокий. Over/under — рыночно-релевантные фичи.

### 5.2. fetch_profits — ROI-бенчмарк

**Структура ответа**:
- Анализ прибыльности по типам ставок (1X2, тоталы, обе забьют)
- Показывает ROI flat-stake для каждого типа ставки
- Поддерживает фильтрацию: sameLeague, homeAway, sameGames (по xG)

**Назначение**: Сравнить наш edge-анализ с независимым бенчмарком. Если sstats показывает ROI +5% на home wins, а наша модель даёт +2% — это полезный контекст.

**Приоритет**: 🟡 Средний. Полезно для валидации, но не для фичей.

### 5.3. fetch_query — bulk-запросы

**Текущее использование**: Уже используется в `data_collector.py` (collect_sstats_bulk) и `ai_analyzer.py`.

**Сравнение с fetch_games_by_league**:
- `fetch_query` — POST-запрос, один вызов возвращает до 1000 матчей с фильтрацией
- `fetch_games_by_league` — GET-запрос, требует пагинации

**Рекомендация**: `fetch_query` быстрее для bulk-сбора. Уже используется. Миграция не требуется — оба метода работают.

---

## 6. Лимиты API и расход запросов

### Лимиты (из OpenAPI)

| Режим | Лимит |
|-------|-------|
| Без ключа | 300 запросов/мин на всех пользователей, 30/мин на 1 IP |
| С ключом | Не указано явно (предположительно выше) |

### Текущий расход (из data_log)

| Источник | Действий | Запросов | Период |
|----------|----------|----------|--------|
| sstats_bulk | collect | 6 | 2026-06-28 |
| sstats | collect | 2 | 2026-06-27 |

**Оценка расхода на 1 полный сбор**:
- `collect_sstats` (один день): ~50-100 запросов (fetch_games_by_date + fetch_odds + fetch_game на каждый матч)
- `collect_sstats_bulk` (один сезон): ~50-100 запросов fetch_query + ~50-100 fetch_odds + ~50-100 fetch_game
- **Итого**: ~200-300 запросов на сбор

**Запас для новых вызовов**:
- `fetch_season_table`: 1 вызов на лигу × 15 лиг = 15 запросов
- `fetch_profits`: 1 вызов на матч × N матчей = N запросов
- **Оценка**: Запас есть, но нужно добавить rate limiting (2 сек между запросами уже реализован).

---

## 7. Приоритизированный список находок

### 🔴 Высокий приоритет (добавить в первую очередь)

| # | Находка | Ожидаемый эффект | Трудозатраты |
|---|---------|------------------|--------------|
| 1 | **sstats_odds содержит Over/Under, Asian Handicap, Corners** — 1M+ строк с 20+ типами рынков | Новые фичи: total_over25_implied, asian_handicap_value, corners_implied | Средние (нужно распарсить и сохранить в новую таблицу) |
| 2 | **sstats_statistics содержит possession, corners, shots, fouls** — 316 матчей с 12 типами статистик | Новые фичи: possession_diff, corners_diff, shots_diff | Низкие (таблица уже есть, нужно читать в train.py) |
| 3 | **fetch_season_table** — over/under rates + форма команд | Новые фичи: home_over25_rate, away_over25_rate, form_score | Средние (новая таблица + sync + фичи) |

### 🟡 Средний приоритет (добавить позже)

| # | Находка | Ожидаемый эффект | Трудозатраты |
|---|---------|------------------|--------------|
| 4 | **fetch_profits** — независимый ROI-бенчмарк | Валидация edge-анализа | Низкие |
| 5 | **sstats_events содержит cards, substitutions** — 2768 записей | Новые фичи: cards_per_game, sub_timing | Низкие |
| 6 | **fetch_injuries() данные не сохраняются** | Дублирование с Transfermarkt | Низкие |

### 🟢 Низкий приоритет (опционально)

| # | Находка | Ожидаемый эффект | Трудозатраты |
|---|---------|------------------|--------------|
| 7 | **raw_json в sstats_matches** — содержит расширенные данные | Дополнительные поля | Средние |
| 8 | **`/Ls/List`** — альтернативный формат (Flashscore IDs) | Совместимость с Flashscore | Низкие |
| 9 | **`/Excel/Delux`** — комплексный запрос | Экономия вызовов | Низкие |

---

## 8. Рекомендации по реализации

### Шаг 1: Расширить использование sstats_statistics (фичи из БД)

Уже есть в БД, нужно только читать в `train.py`:
- `possession` → `home_possession_avg`, `away_possession_avg`, `possession_diff`
- `corners` → `home_corners_avg`, `away_corners_avg`
- `shots_on_target` → уже читается частично, но не все metrices
- `fouls`, `yellow_cards` → `home_fouls_avg`, `away_fouls_avg`

### Шаг 2: Добавить fetch_season_table → новая таблица + фичи

1. Новая таблица `sstats_season_tables` в db.py
2. `sync_season_tables()` в sstats_collector.py
3. Чтение в train.py → новые фичи

### Шаг 3: Распарсить sstats_odds для Over/Under фичей

1. Новая таблица `sstats_overunder` (game_id, market_name, home_value, away_value)
2. Извлечь Goals Over/Under из sstats_odds
3. Использовать как фичу: implied_over25_probability

### Шаг 4: fetch_profits как бенчмарк

1. Запросить для N матчей с котировками
2. Сравнить наш ROI с ROI sstats
3. Документировать в edge_analysis_v3
