# SSTATS_INTEGRATION_FIXES.md — Статус исправлений интеграции sstats.net

**Дата**: 2026-07-15
**Ветка**: текущая рабочая

---

## ✅ СДЕЛАНО

### Этап 1 — Баги в живых фичах (xG, avg_goals_1h/2h)

**1.1. train.py:658 — фильтр `"0"` значений**
- Было: `val = float(raw) if raw not in (None, "", "0") else None`
- Стало: `val = float(raw) if raw not in (None, "") else None`
- Эффект: валидные нулевые статистики (0 угловых, 0 фолов) больше не отбрасываются

**1.2. train.py:699 — распаковка кортежа is_home**
- Было: `relevant = [(gid, d) for gid, d, _ in games if d < match_date]`
- Стало: `relevant = [(gid, d, is_h) for gid, d, is_h in games if d < match_date]`
- Эффект: события (cards, subs) теперь считаются корректно по командам

**1.3. minute=None — убраны фичи avg_goals_1h/avg_goals_2h**
- API sstats НЕ возвращает поле `minute` для событий завершённых матчей (поле events = None или minute = None)
- Фичи `home_avg_goals_1h`, `away_avg_goals_1h` убраны из FEATURE_NAMES и return-вектора
- FEATURE_NAMES: 30 → 28 фич

**1.4. backtest — baseline зафиксирован в METRICS_LOG.md**

| Метрика | Значение |
|---------|----------|
| Dataset | 30,989 rows, 28 features |
| CV Mean Accuracy (3-fold) | 66.74% ± 23.53% |
| CV Mean Log-loss (3-fold) | 0.969 ± 0.387 |
| Test Accuracy | **51.60%** |
| Test Log-loss | **1.100** |
| ROI (edge ≥ 2%) | -8.28% |
| ROI (edge ≥ 5%) | -7.60% |
| ROI (edge ≥ 10%) | -9.78% |

### Этап 2 — Независимые исправления

**2.1. ESPN hardcoded `fifa.world` — ИСПРАВЛЕНО**
- `scrapers/web.py` — `find_espn_match_id()` теперь ищет по всем ~50 ESPN лигам
- Возвращает кортеж `(game_id, league_slug)` вместо строки
- `_fetch_espn_match_detail()` принимает параметр `league`
- `fetch_espn_match()` передаёт league через всю цепочку

**2.2. raw_json — ИСПРАВЛЕНО**
- `db.py:1294` — `str(game)` → `json.dumps(game, ensure_ascii=False, default=str)`

**2.3. ESPN team ID кеширование — ИСПРАВЛЕНО**
- `scrapers/web.py` — `_TEAM_ID_CACHE` сохраняется в `data/espn_team_cache.json`
- Загружается при импорте модуля, сохраняется при каждом новом найденном ID
- Переиспользует `_ESPN_LEAGUES` список (убран дубликат `_LEAGUES`)

### Этап 3 — Расширение использования данных (ЧАСТИЧНО)

**3.2. sstats_statistics → FEATURE_NAMES — ДОБАВЛЕНО (код)**
- FEATURE_NAMES: 28 → 52 фичи
- Добавлены: possession, corners, fouls, yellow_cards, shots_on_target, big_chances (home/away/diff)
- Return-вектор в `build_features()` обновлён

**3.3. cards_per_game, subs_per_game → FEATURE_NAMES — ДОБАВЛЕНО (код)**
- Добавлены: cards_per_game, subs_per_game (home/away/diff)

**3.4. sstats_injuries — ДОБАВЛЕНО (код)**
- `db.py` — новая таблица `sstats_injuries` в SCHEMA
- `db.py` — функция `save_sstats_injuries()` добавлена
- `data_collector.py` — вызов `save_sstats_injuries()` добавлен в `collect_sstats()` и `collect_sstats_bulk()`
- Протестировано: 12 injuries сохранены для game_id=1388308

### Этап 4 — Производительность

**4.1. Ранний exit для sstats-функций — ДОБАВЛЕНО (код)**
- `train.py:build_features()` — проверка `home_id in _team_sstats_games` перед вызовом 4 sstats-функций
- Экономит ~1.6M бесполезных lookup'ов для матчей без sstats-данных

### Дополнительно

- **METRICS_LOG.md** — обновлён с результатами Этапа 1 (28 фичей baseline)
- **SSTATS_AUDIT.md** — существующий аудит (не тронут)
- **SSTATS_INTEGRATION_AUDIT_V2.md** — полный аудит с новыми находками (см. ниже)

---

## ⏳ В ПРОЦЕССЕ / НЕ ЗАВЕРШЕНО

### Обучение модели с 52 фичами (Этап 3)

**Статус**: ✅ ЗАВЕРШЕНО

**Результаты**: Accuracy 51.98%, Log-loss 1.052, 5 новых sstats-фичей в топ-10. Детали в METRICS_LOG.md → "Этап 3".

### consensus_odds — добавить Over/Under (Этап 3.1)

**Статус**: НЕ СДЕЛАНО

**Что нужно сделать**:
- В `scrapers/sstats.py:consensus_odds()` добавить извлечение `marketName="Goals Over/Under"` → implied probability
- Сохранять как отдельное поле (НЕ в основные market_implied_h/d/a)
- Использовать ТОЛЬКО как контекст для ai_analyzer.py (276 матчей недостаточно для train.py)

### Расширение сбора sstats_odds

**Статус**: НЕ СДЕЛАНО (ограничение задачи)

**Что нужно сделать** (отдельная задача):
- Увеличить покрытие с 276 до 2000+ матчей
- Требует часов непрерывного сбора с rate limit 2 сек/запрос
- `python data_collector.py --sstats-bulk` с расширенным диапазоном сезонов

---

## 🔍 НОВЫЕ НАХОДКИ ИЗ АУДИТА (не в SSTATS_AUDIT.md)

### Баг: events НЕ в game, а на level data

**Проблема**: `sstats.fetch_game()` возвращает dict с ключами `game`, `events`, `statistics`, `lineups` — все на одном уровне. Но парсинг в `data_collector.py` обращается к `detail.get("events")` — это корректно. Проблема была в том, что при прямом fetch API возвращает `events=None` для большинства игр.

**Статус**: Данные в БД есть (4101 events для 268 игр), собраны ранее. Новые сборы могут не получить events.

### Баг: `_get_team_sstats_event_features` не различал home/away

**Проблема**: Кортеж `_team_sstats_games` содержал `(gid, date, is_home)`, но функция отбрасывала `is_home` при распаковке.
**Статус**: ИСПРАВЛЕНО (Этап 1.2)

### Производительность: ~1.6M бесполезных sstats lookup'ов

**Проблема**: Для каждого из 31K матчей вызывались 4 sstats-функции, даже если у команды нет sstats данных.
**Статус**: ИСПРАВЛЕНО (Этап 4.1) — ранний exit по проверке `_team_sstats_games`

### sstats_odds: 20+ рынков, используется только Match Winner

**Проблема**: `consensus_odds()` фильтрует только `marketName in ("Match Winner", "1X2")`. Over/Under, Asian Handicap, Corners, BTTS игнорируются.
**Статус**: НЕ ИСПРАВЛЕНО (Этап 3.1 — в очереди)

---

## ФАЙЛЫ, ИЗМЕНЁННЫЕ В ЭТОЙ СЕССИИ

| Файл | Изменения |
|------|-----------|
| `train.py` | FEATURE_NAMES 28→52, return-вектор, фильтр "0", is_home unpacking, sstats early exit |
| `db.py` | raw_json=json.dumps, таблица sstats_injuries, функция save_sstats_injuries |
| `scrapers/web.py` | ESPN all-leagues search, league param in detail, team ID file cache, import os |
| `data_collector.py` | save_sstats_injuries вызов в collect_sstats и collect_sstats_bulk |
| `METRICS_LOG.md` | Секция "Этап 1" с метриками baseline (28 фичей) |

## Уточнения перед Этапом 3

### 1. Фильтр "3 сезона" — СТАРЫЙ

**Статус**: Фильтр `min_year = current_year - 3` в `train.py:403-416` **присутствовал ДО текущей сессии sstats-fixes**.

**Доказательства**:
- В SSTATS_INTEGRATION_FIXES.md фильтр не упоминается среди изменений Этапов 1-4
- Комментарий в коде: "Filter to last 3 seasons for best balance of data volume vs recency"
- Датасет после фильтрации: 30,989 строк (Этап 1, METRICS_LOG.md:90)
- Датасет до фильтрации: ~31,857 строк (METRICS_LOG.md:19)
- Разница ~2% — фильтр применялся одинаково ко всем бэктестам

**Вывод**: Метрики Этапа 1 (30,989 строк) **сопоставимы** с историческими данными. Фильтр не менялся, данные одного масштаба.

### 2. Minute — БАГ СБОРЩИКА (не ограничение API)

**Статус**: **БАГ СБОРЩИКА** — API sstats возвращает поле `elapsed`, а старый collector искал `minute` → все `minute=None` в БД. Фикс применён в предыдущей сессии (не в этой).

**Доказательства** ( game_id=1351046 ):
1. API возвращает `elapsed: 62` (сырой JSON из шага 2.3):
```json
{
  "id": 6590026,
  "teamId": 123,
  "elapsed": 62,
  "extra": null,
  "type": 3,
  "name": "Substitution 2",
  "player": {"id": 292170, "name": "Lenny Lobato"},
  "assistPlayer": {"id": 191872, "name": "Carlos Alberto"}
}
```
2. В БД: `minute=None` для всех 14 событий (собрано 2026-06-28)
3. Текущий код `db.py:1409`: `ev.get("elapsed")` — **исправленный** (был применён между 2026-06-28 и сейчас, в предыдущей сессии, не в этой)
4. Доказательство что фикс НЕ был на месте при сборе: если бы код был `ev.get("elapsed")` 28 июня, `minute` был бы `62`, а не `None`
5. `data['events']` — правильный путь (14 событий), `data['game']['events']` — None

**Вывод**: Решение Этапа 1.3 (удаление `avg_goals_1h/2h`) было **временным**. После пересбора событий для 268 игр (с текущим исправленным collector) можно вернуть фичи `home_avg_goals_1h`, `away_avg_goals_1h` — `elapsed` позволяет split по таймам (≤45 = 1-й тайм, >45 = 2-й тайм).

**НЕ ЧИНИТЬ** data_collector.py в рамках этой задачи — только задокументировать. Починка + пересбор = отдельная задача.

---

## ФАЙЛЫ, СОЗДАННЫЕ В ЭТОЙ СЕССИИ

| Файл | Назначение |
|------|------------|
| `scripts/audit_sstats.py` | Аудит БД sstats таблиц |
| `scripts/audit_sstats2.py` | Аудит данных sstats |
| `scripts/check_minute.py` | Проверка minute=None в events |
| `scripts/check_events_api.py` | Проверка API events |
| `scripts/check_events_api2.py` | Проверка events для конкретных game_id |
| `scripts/check_events_api3.py` | Проверка events для live игр |
| `scripts/check_events_raw.py` | Проверка raw API response |
| `scripts/check_raw_api.py` | Проверка структуры ответа API |
| `scripts/test_injuries.py` | Тест сохранения injuries |
| `SSTATS_INTEGRATION_AUDIT_V2.md` | Полный аудит v2 |
| `SSTATS_INTEGRATION_FIXES.md` | Этот файл |
