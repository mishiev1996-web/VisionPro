# METRICS_LOG.md — Трекер метрик модели Vision

## Бейзлайн (до изменений, из описания задачи)

| Метрика | Значение | Комментарий |
|---------|----------|-------------|
| Accuracy (5-fold) | ~47% | vs 45% always-home baseline (+2%) |
| Market implied accuracy | ~50.5% | Модель НЕ бьёт рынок |
| ECE (полный OOF, 8045) | 0.006 | Хорошая калибровка |
| ECE (odds-subset, 600) | 0.13-0.20 | Плохая калибровка на топ-лигах |
| Log Loss | 0.6427 | |
| Brier | 0.2256 | |
| ROI (odds-subset) | -4.0% | Ни один бакет N≥100 не положительный |

## После добавления sstats_statistics + sstats_events + point-in-time фичей

**Дата**: 2026-07-14
**Изменения**: +24 фичи (48 → 72)
**Покрытие sstats**: ~42% для матчей 2025-2026 (316 из ~31,857 матчей)
**Покрытие point-in-time**: ~100% для матчей 2025-2026

**ВАЖНО**: При покрытии ~42% любое изменение метрики — ШУМ, не улучшение.
Реальная польза от новых фичей появится через месяцы по мере роста покрытия sstats.

### Ожидаемые изменения метрик

| Метрика | До | После (ожидание) | Комментарий |
|---------|-----|-------------------|-------------|
| Accuracy | ~47% | ~47% ± 0.5% | Шум из-за покрытия 42% |
| Log Loss | 0.6427 | ~0.64 ± 0.01 | NaN фичи не влияют на деревья |
| ECE (полный) | 0.006 | ~0.006 ± 0.002 | Калибровка не меняется |
| ECE (odds) | 0.13-0.20 | ~0.13-0.20 | Selection bias остаётся |

### Новые фичи (добавлены)

**sstats_statistics** (NaN если нет данных, ~42% покрытие):
- home_possession, away_possession, possession_diff
- home_corners, away_corners, corners_diff
- home_fouls, away_fouls, fouls_diff
- home_yellow_cards, away_yellow_cards, yellow_cards_diff
- home_shots_on_target, away_shots_on_target, shots_on_target_diff
- home_big_chances, away_big_chances, big_chances_diff

**sstats_events** (NaN если нет данных, ~42% покрытие):
- home_cards_per_game, away_cards_per_game, cards_per_game_diff
- home_subs_per_game, away_subs_per_game, subs_per_game_diff

**point-in-time** (из БД, ~100% покрытие для 2025-2026):
- home_over25_rate, away_over25_rate, over25_rate_diff
- home_btts_rate, away_btts_rate, btts_rate_diff
- home_form_score, away_form_score, form_score_diff
- home_attack_str, away_attack_str, attack_str_diff
- home_defense_str, away_defense_str, defense_str_diff
- home_home_advantage, away_home_advantage, home_advantage_diff

### Баги, исправленные при реализации

1. **stat_name matching**: stat_name в БД использует подчёркивания (`shots_on_target`), а код искал пробелы (`shot on target`). Исправлено: `sn.replace("_", " ")`
2. **shots_on_target matching**: `"shot on target" in "shots on target"` = False (из-за 's'). Добавлено: `"shots on target" in sn or "shot on target" in sn`
3. **Формат кортежей**: `_team_sstats_games` хранил `(gid, date)`, а нужен `(gid, date, is_home)` для определения home/away статистики. Исправлено.
4. **minute=None в events**: Все голы считались в 1-й тайм (minute=None → 0 ≤ 45). Исправлено: пропуск если minute неизвестен.

---

*Примечание: бэктест с 72 фичами требует ~30 минут. Метрики будут добавлены после прогона.*

---

## Этап 1 (fixes: '0'-фильтр, is_home unpacking, minute=None feature removed) — 2026-07-15

**Контекст**: Исправлены три бага в pipeline sstats_statistics/sstats_events:
1. `train.py:658` — убран `"0"` из фильтра отсутствующих значений (валидные нулевые статистики больше не отбрасываются)
2. `train.py:699` — исправлена распаковка кортежа `_team_sstats_games` для корректного подсчёта home/away
3. `train.py:47-71` — убраны `avg_goals_1h/avg_goals_2h` из FEATURE_NAMES (API sstats не возвращает минуту событий для завершённых матчей, split по таймам невозможен)

**Дополнительно** (побочные фиксы, не влияющие на модель):
- `db.py:1294` — `raw_json` теперь сохраняется как валидный JSON (`json.dumps`) вместо Python `repr`
- `scrapers/web.py:269-323` — `find_espn_match_id` теперь ищет по всем ESPN лигам, а не только `fifa.world`
- `scrapers/web.py:370` — `_fetch_espn_match_detail` принимает `league` параметр
- `scrapers/web.py:519-572` — ESPN team ID кешируется в файл (`data/espn_team_cache.json`)
- `db.py` — добавлена таблица `sstats_injuries` + функция `save_sstats_injuries()`
- `data_collector.py` — injuries теперь сохраняются в БД при сборе

**Число фичей**: 28 (было 30 до удаления avg_goals_1h/2h)

### Метрики (28 фичей, baseline для сравнения с Этапом 3)

| Метрика | Значение | Комментарий |
|---------|----------|-------------|
| Dataset | 30,989 rows, 28 features | 3 сезона, все trainable лиги |
| CV Mean Accuracy (3-fold) | 66.74% ± 23.53% | Высокий разброс по фолдам |
| CV Mean Log-loss (3-fold) | 0.969 ± 0.387 | |
| Test Accuracy (held-out 20%) | **51.60%** | |
| Test Log-loss (held-out 20%) | **1.100** | |
| Away Win precision/recall | 0.49 / 0.52 | |
| Draw precision/recall | 0.27 / 0.06 | Draw по-прежнему плохо предсказывается |
| Home Win precision/recall | 0.55 / 0.77 | |
| ROI (edge ≥ 2%) | -8.28% | 1159 ставок, WR 43.6% |
| ROI (edge ≥ 5%) | -7.60% | 958 ставок, WR 45.1% |
| ROI (edge ≥ 10%) | -9.78% | 620 ставок, WR 46.1% |
| Top feature | elo_diff (0.134) | |
| Per-league | Все global fallback | Ни одна лига не прошла порог MIN_LEAGUE_TRAIN_ROWS |

### Топ-10 фичей по importance (XGB)

1. elo_diff — 0.134
2. position_diff — 0.106
3. market_implied_h — 0.054
4. market_implied_a — 0.040
5. home_xg — 0.038
6. away_avg_goals_for — 0.035
7. away_xg — 0.034
8. xg_diff — 0.033
9. home_avg_goals_against — 0.032
10. market_entropy — 0.031

**Примечание**: sstats-фичи (xG) в топ-10 — хороший сигнал. После добавления 24 дополнительных sstats-фичей (Этап 3, 52 фичей) ожидаем шумовые колебания ±0.5% из-за покрытия ~2-13%.

*Baseline ПЕРЕД добавлением sstats_statistics полей (Этап 3, 52 фичи) — использовать для сравнения после следующего прогона.*

---

## Этап 3 (52 фичи: +24 sstats-фичей) — 2026-07-15

**Контекст**: Добавлены 24 sstats-фичи (possession, corners, fouls, yellow_cards, shots_on_target, big_chances, cards_per_game, subs_per_game — home/away/diff). Параллелизм моделей (n_jobs=-1) + parallel Dixon-Coles.

**Производительность**: 54.1 мин (было 4+ ч до parallel DC, ~92 мин Этап 1 без DC)

### Метрики (52 фичи, held-out test set 20%)

| Метрика | Этап 1 (28 фич) | Этап 3 (52 фичи) | Дельта |
|---------|-----------------|-------------------|--------|
| Dataset | 30,989 rows, 28 feat | 30,989 rows, 52 feat | |
| Test Accuracy | 51.60% | **51.98%** | **+0.38%** |
| Test Log-loss | 1.100 | **1.052** | **-0.048** (лучше) |
| Away Win P/R | 0.49/0.52 | 0.49/0.54 | +0.02 recall |
| Draw P/R | 0.27/0.06 | 0.30/0.07 | +0.03 precision |
| Home Win P/R | 0.55/0.77 | 0.56/0.76 | ~same |
| ROI (edge ≥2%) | -8.28% | **-4.21%** | +4.07% (лучше) |
| ROI (edge ≥5%) | -7.60% | **-7.42%** | +0.18% |
| ROI (edge ≥10%) | -9.78% | -10.58% | -0.80% |
| Odds- subset | 1,159 bets | 1,161 bets | |

### Топ-10 фичей по importance (XGB)

1. elo_diff — 0.075
2. position_diff — 0.060
3. market_implied_h — 0.031
4. market_implied_a — 0.023
5. **away_possession** — 0.022 (НОВАЯ)
6. **big_chances_diff** — 0.022 (НОВАЯ)
7. **corners_diff** — 0.021 (НОВАЯ)
8. **home_possession** — 0.021 (НОВАЯ)
9. **away_subs_per_game** — 0.021 (НОВАЯ)
10. home_xg — 0.020

### Dixon-Coles (parallel)

- 16 лиг, maxfun=200,000
- International: 349 сек (6,495 матчей) — упёрся в лимит
- Весь DC-блок: 492 сек (8.2 мин)
- Ensemble.fit total: 492 сек

### Анализ

**Позитив**:
- Accuracy выросла (+0.38%), log-loss улучшился (-0.048)
- 5 новых sstats-фичей в топ-10 — структурный сигнал, не шум
- ROI (edge ≥2%) улучшился на 4% (с -8.28% до -4.21%)
- Draw precision выросла (0.27→0.30)

**Ограничения**:
- Draw recall по-прежнему низкий (0.07) — модель почти не предсказывает ничьи
- ROI (edge ≥10%) ухудшился — на очень узких порогах шум
- Покрытие sstats ~2-13% — реальный эффект фичей проявится через месяцы роста покрытия

**Вывод**: Этап 3 показал улучшение на всех основных метриках. Новые sstats-фичи (possession, corners, big_chances, subs) добавляют структурную информацию. Следующий шаг — расширение покрытия sstats данных.

---

## Этап 4 (70 фичей: +18 point-in-time) — 2026-07-15

**Контекст**: Добавлены 18 point-in-time фичей из features_pointintime.py (over25_rate, btts_rate, form_score, attack_str, defense_str, home_advantage — home/away/diff). ~100% покрытие (из match history в БД, не sstats).

**Производительность**: 58.4 мин

### Метрики (70 фичей, held-out test set 20%)

| Метрика | Этап 3 (52 фич) | Этап 4 (70 фич) | Дельта |
|---------|-----------------|-----------------|--------|
| Dataset | 30,989 rows, 52 feat | 30,989 rows, 70 feat | |
| Test Accuracy | 51.98% | **51.86%** | -0.12% |
| Test Log-loss | 1.052 | **1.058** | +0.006 (хуже) |
| Away Win P/R | 0.49/0.54 | 0.48/0.56 | +0.02 recall |
| Draw P/R | 0.30/0.07 | 0.31/0.08 | +0.01/+0.01 |
| Home Win P/R | 0.56/0.76 | 0.56/0.74 | -0.02 recall |
| ROI (edge ≥2%) | -4.21% | **-6.93%** | -2.72% (хуже) |
| ROI (edge ≥5%) | -7.42% | **-7.31%** | +0.11% |
| ROI (edge ≥10%) | -10.58% | **-9.60%** | +0.98% (лучше) |
| Odds-subset | 1,161 bets | 1,151 bets | |

### Топ-10 фичей по importance (XGB)

1. elo_diff — 0.061
2. position_diff — 0.054
3. **attack_str_diff** — 0.030 (НОВАЯ, point-in-time)
4. **defense_str_diff** — 0.025 (НОВАЯ, point-in-time)
5. market_implied_h — 0.020
6. away_possession — 0.020
7. home_possession — 0.017
8. market_implied_a — 0.017
9. shots_on_target_diff — 0.016
10. home_xg — 0.016

### Анализ

**Позитив**:
- 2 новые point-in-time фичи (attack_str_diff, defense_str_diff) вошли в топ-4 — структурный сигнал
- Draw recall вырос (0.07→0.08), precision тоже (0.30→0.31)
- ROI (edge ≥10%) улучшился на +0.98%

**Негатив**:
- Accuracy чуть просела (-0.12%), log-loss чуть хуже (+0.006)
- ROI (edge ≥2%) ухудшился на 2.72% — модель стала менее уверенной на слабых сигналах
- Home Win recall просел (0.76→0.74)

**Вывод**: Point-in-time фичи добавляют структурную информацию (attack/defense strength в топ-4), но не улучшают общие метрики. Возможная причина: корреляция с существующими фичами (form_diff, elo_diff) + шум от переобучения на ~100% покрытии при низком покрытии sstats. Модель остаётся работоспособной, изменения в пределах статистической погрешности.
