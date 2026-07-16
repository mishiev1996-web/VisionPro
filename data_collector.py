"""
data_collector.py — Orchestrates all scrapers into SQLite.

v2:
  - Fixed hang: added timeouts to all network calls
  - Data logging: every collection run logged to data_log table
  - Quality report: automatic after each collection
  - Continue on error: one source failing doesn't block others
  - Better progress: percentage, ETA-like updates

CLI:
    python data_collector.py                   # full refresh
    python data_collector.py --current-only    # current season + live
    python data_collector.py --live-only       # FlashScore live only
    python data_collector.py --export          # export to JSON
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import threading
import time
from typing import Callable, List, Optional

import db
import config
from scrapers import (understat, clubelo, openmeteo, transfermarkt,
                       thesportsdb, espn, historical_odds, fbref)


ProgressCB = Callable[[dict], None]

SOURCE_TIMEOUT = 30  # seconds per individual source call
MAX_RETRIES = 2


def _retry(fn, *args, retries=MAX_RETRIES, delay=2.0, **kwargs):
    """Call fn with retries and exponential backoff."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(delay * attempt)
    raise last_exc


def _deterministic_id(*parts: str) -> int:
    """Generate a deterministic integer ID from string parts.

    Uses MD5 for deterministic behavior across sessions (unlike Python's hash()).
    Returns a 32-bit positive integer to avoid SQLite integer overflow issues.
    """
    combined = "|".join(str(p) for p in parts)
    digest = hashlib.md5(combined.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _resolve_team_id(conn, name: str, league_slug: str) -> int:
    """Resolve a team name to an existing DB team ID.

    Priority:
    1. Exact name match in the same league
    2. Fuzzy name match (substring / case-insensitive) in same league
    3. Fuzzy name match across all leagues
    4. Create a new deterministic ID
    """
    name_lower = name.strip().lower()

    # 1. Exact match in same league
    row = conn.execute(
        "SELECT id FROM teams WHERE LOWER(name)=? AND league_slug=?",
        (name_lower, league_slug),
    ).fetchone()
    if row:
        return row["id"]

    # 2. Fuzzy match in same league
    row = conn.execute(
        "SELECT id, name FROM teams WHERE league_slug=?",
        (league_slug,),
    ).fetchall()
    for r in row:
        if name_lower in r["name"].lower() or r["name"].lower() in name_lower:
            return r["id"]

    # 3. Fuzzy match across all leagues
    all_teams = conn.execute("SELECT id, name FROM teams").fetchall()
    for r in all_teams:
        tn = r["name"].lower()
        if name_lower == tn:
            return r["id"]
        if name_lower in tn or tn in name_lower:
            return r["id"]
        # Also check short_name
    all_teams_short = conn.execute(
        "SELECT id, short_name FROM teams WHERE short_name IS NOT NULL AND short_name != ''"
    ).fetchall()
    for r in all_teams_short:
        sn = r["short_name"].lower()
        if name_lower == sn or name_lower in sn or sn in name_lower:
            return r["id"]

    # 4. Create new
    return _deterministic_id("team", league_slug, name)


# ── ESPN match collection (all leagues) ──────────────────────────────────────

def collect_espn_all(progress_cb: Optional[ProgressCB] = None,
                     cancel_event: Optional[threading.Event] = None,
                     days_back: int = 60) -> dict:
    """Collect matches from ESPN across key leagues (60 days back + 7 days forward).

    Uses ESPN_TO_OURS mapping from config for league slug translation.
    Registers leagues with appropriate tiers from LEAGUE_TIERS.
    """
    from scrapers.espn import fetch_league_scoreboard

    # Build KEY_LEAGUES from ESPN's DEFAULT_LEAGUES, filtering to trainable + display
    espn_leagues = espn.DEFAULT_LEAGUES
    KEY_LEAGUES = [
        lg for lg in espn_leagues
        if lg["slug"] in config.ESPN_TO_OURS or lg["slug"].startswith("uefa.") or lg["slug"].startswith("fifa.")
    ]

    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "espn_all", "collect")

    saved = 0; skipped = 0; errors = 0
    now = dt.datetime.now().isoformat(timespec="seconds")
    today = dt.date.today()
    total_days = days_back + 7

    _emit(progress_cb, {"type": "info",
                        "msg": f"ESPN: {len(KEY_LEAGUES)} лиг × {total_days} дней (назад {days_back}д)"})

    with db.connect() as conn:
        for lg in KEY_LEAGUES:
            our_slug = config.ESPN_TO_OURS.get(lg["slug"], lg["slug"])
            tier_info = config.LEAGUE_TIERS.get(our_slug, {"tier": 3})
            db.upsert_league(conn, our_slug, lg["name"], lg["country"],
                             tier=tier_info["tier"])

        for offset in range(-days_back, 7):
            if cancel_event and cancel_event.is_set():
                break
            d = today + dt.timedelta(days=offset)
            day_matches = 0

            for lg in KEY_LEAGUES:
                if cancel_event and cancel_event.is_set():
                    break
                try:
                    matches = fetch_league_scoreboard(lg["slug"], d, lg["name"], lg["country"])
                except Exception:
                    errors += 1
                    continue

                for m in matches:
                    home_name = m.get("home", "")
                    away_name = m.get("away", "")
                    if not home_name or not away_name:
                        skipped += 1
                        continue

                    our_slug = config.ESPN_TO_OURS.get(lg["slug"], lg["slug"])

                    home_id = _resolve_team_id(conn, home_name, our_slug)
                    away_id = _resolve_team_id(conn, away_name, our_slug)

                    db.upsert_team(conn, home_id, home_name, None, our_slug)
                    db.upsert_team(conn, away_id, away_name, None, our_slug)

                    is_result = m.get("status") == "finished"
                    score_h = int(m["score_home"]) if m.get("score_home", "").isdigit() else None
                    score_a = int(m["score_away"]) if m.get("score_away", "").isdigit() else None

                    match_id = _deterministic_id("match", our_slug, d.isoformat(), home_name, away_name)
                    db.upsert_match(conn, {
                        "id": match_id,
                        "league_slug": our_slug,
                        "season": d.year if d.month >= 7 else d.year - 1,
                        "date": d.isoformat(),
                        "home_id": home_id,
                        "away_id": away_id,
                        "home_goals": score_h if is_result else None,
                        "away_goals": score_a if is_result else None,
                        "home_xg": None,
                        "away_xg": None,
                        "is_result": 1 if is_result else 0,
                        "forecast_w": None,
                        "forecast_d": None,
                        "forecast_l": None,
                    })
                    saved += 1
                    day_matches += 1

            day_pct = int((offset + days_back + 1) / total_days * 100)
            _emit(progress_cb, {"type": "info",
                                "msg": f"  ESPN · {d.isoformat()}: {day_matches} матчей ({day_pct}%)"})

        db.log_finish(conn, log_id, status="ok" if errors == 0 else "error",
                      rows_added=saved, rows_skipped=skipped, errors=errors,
                      details={"leagues": len(KEY_LEAGUES), "days": total_days})

    _emit(progress_cb, {"type": "success",
                        "msg": f"✓ ESPN: {saved} матчей, {errors} ошибок"})
    return {"saved": saved, "skipped": skipped, "errors": errors}


def _current_season_year() -> int:
    today = dt.date.today()
    return today.year if today.month >= 7 else today.year - 1


def _save_sstats_stats_dict(conn, game_id: int, stats_dict: dict, now: str) -> int:
    """Save sstats statistics from flat dict format to sstats_statistics table.

    Input format: {"shotsOnGoalHome": 5, "shotsOnGoalAway": 4, ...}
    Output: rows like (game_id, "shots_on_target", "5", "4", now)
    """
    FIELD_MAP = {
        "shotsOnGoal": "shots_on_target",
        "totalShots": "total_shots",
        "cornerKicks": "corners",
        "ballPossession": "possession",
        "fouls": "fouls",
        "offsides": "offsides",
        "yellowCards": "yellow_cards",
        "redCards": "red_cards",
        "expectedGoals": "xg",
        "expectedAssists": "xa",
        "bigChances": "big_chances",
        "xgOnTarget": "xg_on_target",
        "totalPasses": "total_passes",
        "passesAccurate": "passes_accurate",
        "dangerousAttacks": "dangerous_attacks",
        "attacks": "attacks",
    }
    saved = 0
    for prefix, stat_name in FIELD_MAP.items():
        home_val = stats_dict.get(f"{prefix}Home")
        away_val = stats_dict.get(f"{prefix}Away")
        if home_val is not None or away_val is not None:
            conn.execute(
                "INSERT INTO sstats_statistics(game_id, stat_name, home_value, away_value, collected_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (game_id, stat_name,
                 str(home_val) if home_val is not None else "",
                 str(away_val) if away_val is not None else "",
                 now),
            )
            saved += 1
    return saved


def _emit(cb: Optional[ProgressCB], event: dict) -> None:
    if cb:
        try: cb(event)
        except Exception: pass
    else:
        msg = str(event.get("msg") or event)
        try: print(f"  {msg}")
        except UnicodeEncodeError: print("  " + msg.encode("ascii", "replace").decode("ascii"))


def _to_int(v):
    try: return int(v) if v is not None else None
    except (TypeError, ValueError): return None


def _to_float(v):
    try: return float(v) if v is not None else None
    except (TypeError, ValueError): return None


def _timed_call(fn, *args, timeout=SOURCE_TIMEOUT, **kwargs):
    """Call fn with a real timeout using threading. Raises TimeoutError if exceeded."""
    result = [None]
    error = [None]
    def _target():
        try:
            result[0] = fn(*args, **kwargs)
        except Exception as e:
            error[0] = e
    t = threading.Thread(target=_target, daemon=True)
    start = time.monotonic()
    t.start()
    t.join(timeout=timeout)
    elapsed = time.monotonic() - start
    if t.is_alive():
        raise TimeoutError(f"{fn.__name__} timed out after {timeout}s")
    if error[0]:
        raise error[0]
    return result[0], elapsed


# ── Understat (parallel) ────────────────────────────────────────────────────

def collect_understat(seasons: List[int],
                      leagues: Optional[List[str]] = None,
                      progress_cb: Optional[ProgressCB] = None,
                      cancel_event: Optional[threading.Event] = None,
                      parallel: bool = True) -> dict:
    """Fetch Understat data for all leagues × seasons."""
    leagues = leagues or list(understat.LEAGUES.keys())
    tasks = [{"league_slug": s, "season": sz} for s in leagues for sz in seasons]
    total = len(tasks)
    summary = {"matches": 0, "teams_seen": 0, "errors": 0,
               "leagues_done": 0, "leagues_total": total}

    db.init_db()
    with db.connect() as conn:
        for slug, meta in understat.LEAGUES.items():
            tier_info = config.LEAGUE_TIERS.get(slug, {"tier": 1})
            db.upsert_league(conn, slug, meta["name"], meta["country"],
                             tier=tier_info["tier"])

    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "understat", "collect")

    _emit(progress_cb, {"type": "start",
                        "msg": f"Understat: {len(leagues)} лиг × {len(seasons)} сезонов "
                               f"({total} запросов, параллельно)" if parallel
                               else f"Understat: {total} запросов",
                        "summary": summary})

    try:
        if parallel and total > 4:
            _collect_understat_parallel(tasks, summary, progress_cb, cancel_event)
        else:
            _collect_understat_sequential(tasks, summary, progress_cb, cancel_event)
    except Exception as e:
        summary["errors"] += 1
        _emit(progress_cb, {"type": "error", "msg": f"Understat критическая ошибка: {e}"})

    with db.connect() as conn:
        db.log_finish(conn, log_id,
                      status="ok" if summary["errors"] == 0 else "error",
                      rows_added=summary["matches"],
                      errors=summary["errors"],
                      details={"leagues": len(leagues), "seasons": len(seasons)},
                      error_msg=str(e) if summary.get("errors") else "")

    return summary


def _collect_understat_parallel(tasks, summary, progress_cb, cancel_event):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading as _threading

    summary_lock = _threading.Lock()

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for t in tasks:
            if cancel_event and cancel_event.is_set():
                break
            fut = pool.submit(understat.fetch_understat_league,
                              t["league_slug"], t["season"])
            futures[fut] = t

        for fut in as_completed(futures):
            if cancel_event and cancel_event.is_set():
                break
            task = futures[fut]
            slug = task["league_slug"]
            season = task["season"]
            league_name = understat.LEAGUES[slug]["name"]

            try:
                payload = fut.result(timeout=SOURCE_TIMEOUT)
            except Exception as e:
                with summary_lock:
                    summary["errors"] += 1
                _emit(progress_cb, {"type": "error",
                                    "msg": f"✗ {league_name}/{season}: {e}",
                                    "summary": summary})
                with summary_lock:
                    summary["leagues_done"] += 1
                continue

            _save_understat_payload(payload, slug, season, summary, lock=summary_lock)
            with summary_lock:
                summary["leagues_done"] += 1

            _emit(progress_cb, {"type": "success",
                                "msg": f"✓ {league_name} {season}/{season+1}: "
                                       f"{len(payload['matches'])} матчей "
                                       f"({summary['leagues_done']}/{summary['leagues_total']})",
                                "summary": summary})

    if cancel_event and cancel_event.is_set():
        _emit(progress_cb, {"type": "stopped",
                            "msg": f"Остановлено. Готово {summary['leagues_done']}/{summary['leagues_total']}.",
                            "summary": summary})


def _collect_understat_sequential(tasks, summary, progress_cb, cancel_event):
    for task in tasks:
        if cancel_event and cancel_event.is_set():
            _emit(progress_cb, {"type": "stopped",
                                "msg": f"Остановлено. Готово {summary['leagues_done']}/{summary['leagues_total']}.",
                                "summary": summary})
            return
        slug = task["league_slug"]
        season = task["season"]
        league_name = understat.LEAGUES[slug]["name"]

        _emit(progress_cb, {"type": "info",
                            "msg": f"→ {league_name} {season}/{season+1} "
                                   f"({summary['leagues_done']+1}/{summary['leagues_total']})",
                            "summary": summary})
        try:
            payload = understat.fetch_understat_league(slug, season)
        except Exception as e:
            summary["errors"] += 1
            _emit(progress_cb, {"type": "error",
                                "msg": f"✗ {league_name}/{season}: {e}",
                                "summary": summary})
            summary["leagues_done"] += 1
            continue

        _save_understat_payload(payload, slug, season, summary)
        summary["leagues_done"] += 1

        _emit(progress_cb, {"type": "success",
                            "msg": f"✓ {league_name} {season}/{season+1}: "
                                   f"{len(payload['matches'])} матчей",
                            "summary": summary})


def _save_understat_payload(payload, league_slug, season, summary, lock=None):
    with db.connect() as conn:
        for tid_str, team_obj in payload["teams"].items():
            title = str(team_obj.get("title") or "").strip()
            if not title:
                continue
            db.upsert_team(conn, int(tid_str), title, None, league_slug)
            if lock:
                with lock: summary["teams_seen"] += 1
            else:
                summary["teams_seen"] += 1

        for m in payload["matches"]:
            home, away = m.get("h"), m.get("a")
            if not home or not away or not home.get("id") or not away.get("id"):
                continue
            # Validate match has datetime
            match_date = m.get("datetime")
            if not match_date:
                continue

            home_title = str(home.get("title") or "").strip()
            away_title = str(away.get("title") or "").strip()
            if not home_title or not away_title:
                continue

            db.upsert_team(conn, int(home["id"]), home_title,
                           str(home.get("short_title") or "") or None, league_slug)
            db.upsert_team(conn, int(away["id"]), away_title,
                           str(away.get("short_title") or "") or None, league_slug)
            is_result = bool(m.get("isResult"))
            goals = m.get("goals") or {}
            xg    = m.get("xG") or {}
            fc    = m.get("forecast") or {}

            # Validate goals are non-negative integers if present
            home_goals = _to_int(goals.get("h")) if is_result else None
            away_goals = _to_int(goals.get("a")) if is_result else None
            if is_result and (home_goals is None or away_goals is None):
                continue
            if home_goals is not None and home_goals < 0:
                home_goals = 0
            if away_goals is not None and away_goals < 0:
                away_goals = 0

            # Validate xG are non-negative floats if present
            home_xg = _to_float(xg.get("h")) if is_result else None
            away_xg = _to_float(xg.get("a")) if is_result else None
            if home_xg is not None and home_xg < 0:
                home_xg = 0.0
            if away_xg is not None and away_xg < 0:
                away_xg = 0.0

            db.upsert_match(conn, {
                "id": int(m["id"]), "league_slug": league_slug, "season": season,
                "date": str(match_date),
                "home_id": int(home["id"]), "away_id": int(away["id"]),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "home_xg":    home_xg,
                "away_xg":    away_xg,
                "is_result": 1 if is_result else 0,
                "forecast_w": _to_float(fc.get("w")),
                "forecast_d": _to_float(fc.get("d")),
                "forecast_l": _to_float(fc.get("l")),
            })
            if lock:
                with lock: summary["matches"] += 1
            else:
                summary["matches"] += 1

        db.set_meta(conn, "last_refresh",
                    dt.datetime.now().isoformat(timespec="seconds"))


# ── ClubElo ───────────────────────────────────────────────────────────────────

def collect_elo(progress_cb: Optional[ProgressCB] = None,
                cancel_event: Optional[threading.Event] = None,
                limit: int = 0) -> dict:
    import csv as _csv
    import io as _io
    import urllib.request

    # Get Elo for all teams in the DB (ClubElo covers most European clubs)
    teams = db.list_teams()
    if limit > 0:
        teams = teams[:limit]
    _emit(progress_cb, {"type": "info", "msg": f"ClubElo: {len(teams)} команд"})

    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "clubelo", "collect")

    saved = 0; skipped = 0; history_rows = 0
    now = dt.datetime.now().isoformat(timespec="seconds")

    with db.connect() as conn:
        for i, t in enumerate(teams):
            if cancel_event and cancel_event.is_set():
                break
            slug = clubelo.NAME_MAP.get(t["name"], t["name"].replace(" ", ""))
            try:
                req = urllib.request.Request(
                    f"{clubelo.CLUBELO_BASE}/{slug}",
                    headers={"User-Agent": "Mozilla/5.0 (Football-AI; learning)"}
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status != 200:
                        skipped += 1; continue
                    csv_text = resp.read().decode("utf-8", errors="replace")
            except Exception:
                skipped += 1; continue

            rows = []
            reader = _csv.DictReader(_io.StringIO(csv_text))
            for r in reader:
                try:
                    elo = float(r.get("Elo", "0") or 0)
                    date = r.get("From", "")
                    if elo > 0 and date:
                        rows.append((date, elo))
                except (TypeError, ValueError):
                    continue
            if not rows:
                skipped += 1; continue
            rows.sort(key=lambda x: x[0])
            latest = rows[-1][1]
            db.upsert_team_elo(conn, t["id"], t["name"], latest, now)
            db.save_elo_history(conn, t["id"], rows)
            history_rows += len(rows)
            saved += 1

            if (i + 1) % 20 == 0:
                _emit(progress_cb, {"type": "info",
                                    "msg": f"  ClubElo: {i+1}/{len(teams)} ({saved} ok, {skipped} skip)"})

        db.log_finish(conn, log_id, status="ok",
                      rows_added=saved, rows_skipped=skipped,
                      details={"history_rows": history_rows})

    _emit(progress_cb, {"type": "success",
                        "msg": f"ClubElo: {saved} команд, "
                               f"{history_rows} исторических записей"})
    return {"saved": saved, "skipped": skipped, "history_rows": history_rows}


# ── Open-Meteo weather ───────────────────────────────────────────────────────

def collect_weather(progress_cb: Optional[ProgressCB] = None,
                    cancel_event: Optional[threading.Event] = None,
                    limit: int = 80) -> dict:
    upcoming = db.upcoming_matches(limit=limit)
    _emit(progress_cb, {"type": "info",
                        "msg": f"Open-Meteo: погода для {len(upcoming)} матчей"})

    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "openmeteo", "collect")

    saved = 0; skipped = 0; errors = 0
    now = dt.datetime.now().isoformat(timespec="seconds")
    with db.connect() as conn:
        for m in upcoming:
            if cancel_event and cancel_event.is_set(): break
            try:
                w = openmeteo.fetch_match_weather(m.get("home_name") or "", m["date"])
                if w:
                    db.upsert_weather(conn, m["id"], w["temp_c"], w["rain_mm"], w["wind_ms"], now)
                    saved += 1
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                errors += 1
                if errors <= 3:
                    _emit(progress_cb, {"type": "error",
                                        "msg": f"  ✗ Open-Meteo: {m.get('home_name','?')} — {e}"})

        db.log_finish(conn, log_id,
                      status="ok" if errors == 0 else "error",
                      rows_added=saved, rows_skipped=skipped, errors=errors)

    _emit(progress_cb, {"type": "success",
                        "msg": f"✓ Open-Meteo: {saved} матчей, {errors} ошибок"})
    return {"saved": saved, "skipped": skipped, "errors": errors}


# ── Historical odds ──────────────────────────────────────────────────────────

def collect_odds(seasons: Optional[List[int]] = None,
                 progress_cb: Optional[ProgressCB] = None,
                 cancel_event: Optional[threading.Event] = None) -> dict:
    current = _current_season_year()
    seasons = seasons or list(range(current - 7, current + 1))
    leagues = list(historical_odds.LEAGUE_CODES.keys())

    _emit(progress_cb, {"type": "info",
                        "msg": f"Odds: {len(leagues)} лиг × {len(seasons)} сезонов"})

    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "historical_odds", "collect")

    matched = 0; unmatched = 0; fetched = 0
    now = dt.datetime.now().isoformat(timespec="seconds")

    with db.connect() as conn:
        for league in leagues:
            for season in seasons:
                if cancel_event and cancel_event.is_set():
                    db.log_finish(conn, log_id, status="ok",
                                  rows_added=matched, rows_skipped=unmatched)
                    _emit(progress_cb, {"type": "stopped",
                                        "msg": f"Odds остановлен ({matched} матчей)"})
                    return {"matched": matched, "unmatched": unmatched, "fetched": fetched}

                try:
                    rows = historical_odds.fetch_league_season_odds(league, season)
                except Exception:
                    continue
                fetched += len(rows)
                local_match = local_unmatch = 0
                for r in rows:
                    mid = db.find_match_by_date_teams(league, r["date"], r["home"], r["away"])
                    if not mid:
                        local_unmatch += 1; unmatched += 1; continue
                    p_h, p_d, p_a = historical_odds.odds_to_implied(
                        r["home_odds"], r["draw_odds"], r["away_odds"])
                    db.upsert_match_odds(conn, mid,
                                         r["home_odds"], r["draw_odds"], r["away_odds"],
                                         p_h, p_d, p_a, r["source"], now)
                    local_match += 1; matched += 1
                _emit(progress_cb, {"type": "info",
                                    "msg": f"  · {league}/{season}: "
                                           f"{local_match}✓ {local_unmatch}✗"})

        db.log_finish(conn, log_id, status="ok",
                      rows_added=matched, rows_skipped=unmatched,
                      details={"fetched": fetched})

    _emit(progress_cb, {"type": "success",
                        "msg": f"✓ Odds: {matched} матчей"})
    return {"matched": matched, "unmatched": unmatched, "fetched": fetched}


# ── Transfermarkt injuries ────────────────────────────────────────────────────

def collect_injuries(progress_cb: Optional[ProgressCB] = None,
                     cancel_event: Optional[threading.Event] = None) -> dict:
    teams = db.list_teams()
    known = [t for t in teams if t["name"] in transfermarkt.TEAM_URL_MAP]
    _emit(progress_cb, {"type": "info",
                        "msg": f"Transfermarkt: {len(known)} команд"})

    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "transfermarkt", "collect")

    saved_teams = 0; total_injuries = 0; errors = 0
    now = dt.datetime.now().isoformat(timespec="seconds")
    with db.connect() as conn:
        for t in known:
            if cancel_event and cancel_event.is_set():
                break
            try:
                injuries = transfermarkt.fetch_team_injuries(t["name"])
                db.replace_team_injuries(conn, t["id"], injuries, now)
                saved_teams += 1
                total_injuries += len(injuries)
                _emit(progress_cb, {"type": "info", "msg": f"  · {t['name']}: {len(injuries)}"})
            except Exception as e:
                errors += 1
                _emit(progress_cb, {"type": "error", "msg": f"  ✗ {t['name']}: {e}"})

        db.log_finish(conn, log_id,
                      status="ok" if errors == 0 else "error",
                      rows_added=total_injuries, errors=errors)

    _emit(progress_cb, {"type": "success",
                        "msg": f"✓ Transfermarkt: {total_injuries} травм, {errors} ошибок"})
    return {"teams": saved_teams, "injuries": total_injuries, "errors": errors}


# ── SStats enrichment ────────────────────────────────────────────────────────

def collect_sstats(progress_cb: Optional[ProgressCB] = None,
                   cancel_event: Optional[threading.Event] = None,
                   days_back: int = 3, days_forward: int = 3) -> dict:
    """Fetch match details, odds, statistics, and events from sstats.net API.

    Covers worldwide matches. Links to our matches table via date + team names.
    Provides multi-bookmaker odds (8+ bookmakers), match statistics, and events.
    """
    from scrapers import sstats
    import datetime as _dt

    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "sstats", "collect")

    saved_matches = 0; saved_odds = 0; saved_stats = 0; saved_events = 0
    errors = 0
    now = dt.datetime.now().isoformat(timespec="seconds")
    today = _dt.date.today()

    _emit(progress_cb, {"type": "info",
                        "msg": f"SStats: матчи ±{days_forward} дней"})

    with db.connect() as conn:
        for offset in range(-days_back, days_forward + 1):
            if cancel_event and cancel_event.is_set():
                break
            d = today + _dt.timedelta(days=offset)
            date_str = d.isoformat()

            try:
                games = sstats.fetch_games_by_date(date_str)
            except Exception as e:
                errors += 1
                _emit(progress_cb, {"type": "error",
                                    "msg": f"  ✗ SStats {date_str}: {e}"})
                continue

            if not games:
                continue

            day_saved = 0
            for game in games:
                if cancel_event and cancel_event.is_set():
                    break
                game_id = game.get("id")
                if not game_id:
                    continue

                try:
                    db.save_sstats_match(conn, game, now)
                    saved_matches += 1
                    day_saved += 1

                    # Fetch odds for this game
                    try:
                        odds_blocks = sstats.fetch_odds(game_id)
                        if odds_blocks:
                            n = db.save_sstats_odds(conn, game_id, odds_blocks, now)
                            saved_odds += n
                    except Exception:
                        pass

                    # Fetch full game detail (statistics + events)
                    try:
                        detail = sstats.fetch_game(game_id)
                        if detail and isinstance(detail, dict):
                            # Statistics: flat dict → save as key-value pairs
                            stats_dict = detail.get("statistics") or {}
                            if isinstance(stats_dict, dict) and stats_dict:
                                n = _save_sstats_stats_dict(conn, game_id, stats_dict, now)
                                saved_stats += n

                            # Events
                            event_list = detail.get("events") or []
                            if event_list:
                                n = db.save_sstats_events(conn, game_id, event_list, now)
                                saved_events += n

                            # Injuries
                            try:
                                injuries = sstats.fetch_injuries(game_id)
                                if injuries:
                                    db.save_sstats_injuries(conn, game_id, injuries, now)
                            except Exception:
                                pass
                    except Exception:
                        pass

                except Exception as e:
                    errors += 1

            if day_saved > 0:
                _emit(progress_cb, {"type": "info",
                                    "msg": f"  {date_str}: {day_saved} матчей"})

        db.log_finish(conn, log_id,
                      status="ok" if errors == 0 else "error",
                      rows_added=saved_matches, errors=errors,
                      details={"odds": saved_odds, "stats": saved_stats,
                               "events": saved_events})

    _emit(progress_cb, {"type": "success",
                        "msg": f"✓ SStats: {saved_matches} матчей, "
                               f"{saved_odds} odds, {saved_stats} stats, "
                               f"{saved_events} events, {errors} ошибок"})
    return {"matches": saved_matches, "odds": saved_odds,
            "stats": saved_stats, "events": saved_events, "errors": errors}


# ── SStats bulk extraction via /Games/query ──────────────────────────────────

SSTATS_LEAGUE_IDS = {
    "EPL": 39, "La_liga": 140, "Bundesliga": 78,
    "Serie_A": 135, "Ligue_1": 61, "RFPL": 235,
    "Eredivisie": 88, "Primeira_Liga": 94, "Super_Lig": 203,
    "Championship": 40, "Belgian_First": 144, "Greek_Super": 197,
    "MLS": 253, "Brasileirao": 71,
    "Belarus_PL": 116, "Belarus_1D": 117,
    "Kazakhstan_PL": 389, "Kazakhstan_1D": 388,
}


def collect_sstats_bulk(progress_cb: Optional[ProgressCB] = None,
                        cancel_event: Optional[threading.Event] = None,
                        seasons: Optional[List[int]] = None) -> dict:
    """Bulk-extract match data via /Games/query, then fetch odds + stats for each.

    Uses sstats.net's bulk query to get all match IDs for target leagues/seasons,
    then enriches each match with odds (14 bookmakers) and statistics.
    Much more efficient than date-by-date collection for historical data.
    """
    from scrapers import sstats

    current = _current_season_year()
    seasons = seasons or [current - 2, current - 1, current]
    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "sstats_bulk", "collect")

    saved_matches = 0; saved_odds = 0; saved_stats = 0; saved_events = 0
    errors = 0; skipped = 0
    now = dt.datetime.now().isoformat(timespec="seconds")

    leagues_to_query = {k: v for k, v in SSTATS_LEAGUE_IDS.items()
                        if k in config.LEAGUE_TIERS}

    _emit(progress_cb, {"type": "info",
                        "msg": f"SStats bulk: {len(leagues_to_query)} лиг × {len(seasons)} сезонов"})

    with db.connect() as conn:
        first_league = True
        for league_slug, sstats_id in leagues_to_query.items():
            if cancel_event and cancel_event.is_set():
                break

            # Rate limit: pause between leagues to avoid API throttling
            if not first_league:
                import time as _time
                _emit(progress_cb, {"type": "info",
                                    "msg": f"  ⏳ Пауза 30s (rate limit)..."})
                _time.sleep(30)
            first_league = False

            for season in seasons:
                if cancel_event and cancel_event.is_set():
                    break

                _emit(progress_cb, {"type": "info",
                                    "msg": f"  → {league_slug}/{season} (id={sstats_id})..."})

                try:
                    results = sstats.fetch_query(
                        condition=f"LeagueId = {sstats_id} AND Year = {season} AND Status = 8",
                        fields=["Id", "Date", "HomeTeamName", "AwayTeamName",
                                "ScoreHomeFT", "ScoreAwayFT"],
                        order="Date ASC",
                    )
                except Exception as e:
                    errors += 1
                    _emit(progress_cb, {"type": "error",
                                        "msg": f"  ✗ {league_slug}/{season}: {e}"})
                    continue

                if not results:
                    _emit(progress_cb, {"type": "info",
                                        "msg": f"  · {league_slug}/{season}: 0 матчей (пропускаю)"})
                    continue

                _emit(progress_cb, {"type": "info",
                                    "msg": f"  {league_slug}/{season}: {len(results)} матчей"})

                for match in results:
                    if cancel_event and cancel_event.is_set():
                        break

                    game_id = match.get("Id")
                    if not game_id:
                        continue

                    home_name = match.get("HomeTeamName", "")
                    away_name = match.get("AwayTeamName", "")
                    date_str = (match.get("Date") or "")[:10]
                    score_h = match.get("ScoreHomeFT")
                    score_a = match.get("ScoreAwayFT")

                    if not home_name or not away_name or not date_str:
                        skipped += 1
                        continue

                    # Resolve team IDs
                    home_id = _resolve_team_id(conn, home_name, league_slug)
                    away_id = _resolve_team_id(conn, away_name, league_slug)
                    db.upsert_team(conn, home_id, home_name, None, league_slug)
                    db.upsert_team(conn, away_id, away_name, None, league_slug)

                    # Create match in main matches table
                    match_id = _deterministic_id("match", league_slug, date_str, home_name, away_name)
                    db.upsert_match(conn, {
                        "id": match_id,
                        "league_slug": league_slug,
                        "season": season,
                        "date": date_str,
                        "home_id": home_id,
                        "away_id": away_id,
                        "home_goals": score_h,
                        "away_goals": score_a,
                        "home_xg": None,
                        "away_xg": None,
                        "is_result": 1,
                        "forecast_w": None,
                        "forecast_d": None,
                        "forecast_l": None,
                    })
                    saved_matches += 1

                    # Save sstats match record
                    db.save_sstats_match(conn, {
                        "id": game_id,
                        "date": match.get("Date", ""),
                        "homeTeam": {"name": home_name, "id": home_id},
                        "awayTeam": {"name": away_name, "id": away_id},
                        "homeResult": score_h,
                        "awayResult": score_a,
                        "statusName": "Finished",
                        "season": {"year": season, "league": {"id": sstats_id, "name": league_slug}},
                    }, now)

                    # Fetch odds
                    try:
                        odds_blocks = sstats.fetch_odds(game_id)
                        if odds_blocks:
                            n = db.save_sstats_odds(conn, game_id, odds_blocks, now)
                            saved_odds += n

                            # Also save consensus odds to match_odds
                            consensus = sstats.consensus_odds(odds_blocks)
                            if consensus:
                                db.upsert_match_odds(
                                    conn, match_id,
                                    consensus["avg_home_odds"],
                                    consensus["avg_draw_odds"],
                                    consensus["avg_away_odds"],
                                    consensus["implied_h"],
                                    consensus["implied_d"],
                                    consensus["implied_a"],
                                    "sstats_consensus", now,
                                )
                    except Exception:
                        pass

                    # Fetch game detail (statistics + events)
                    try:
                        detail = sstats.fetch_game(game_id)
                        if detail and isinstance(detail, dict):
                            stats_dict = detail.get("statistics") or {}
                            if isinstance(stats_dict, dict) and stats_dict:
                                n = _save_sstats_stats_dict(conn, game_id, stats_dict, now)
                                saved_stats += n

                            event_list = detail.get("events") or []
                            if event_list:
                                n = db.save_sstats_events(conn, game_id, event_list, now)
                                saved_events += n

                            # Injuries
                            try:
                                injuries = sstats.fetch_injuries(game_id)
                                if injuries:
                                    db.save_sstats_injuries(conn, game_id, injuries, now)
                            except Exception:
                                pass
                    except Exception:
                        pass

                _emit(progress_cb, {"type": "success",
                                    "msg": f"  ✓ {league_slug}/{season}: {len(results)} матчей"})

        db.log_finish(conn, log_id,
                      status="ok" if errors == 0 else "error",
                      rows_added=saved_matches, errors=errors,
                      details={"odds": saved_odds, "stats": saved_stats,
                               "events": saved_events, "skipped": skipped})

    _emit(progress_cb, {"type": "success",
                        "msg": f"✓ SStats bulk: {saved_matches} матчей, "
                               f"{saved_odds} odds, {saved_stats} stats, "
                               f"{saved_events} events, {errors} ошибок"})
    return {"matches": saved_matches, "odds": saved_odds,
            "stats": saved_stats, "events": saved_events,
            "errors": errors, "skipped": skipped}

def collect_fbref_xg(progress_cb: Optional[ProgressCB] = None,
                     cancel_event: Optional[threading.Event] = None) -> dict:
    """Collect xG data from FBref for all supported leagues.

    STATUS: Cloudflare Turnstile blocks ALL automated access.
    This function is kept for reference but will fail until FBref
    changes their protection or a paid API is used.

    For xG data, use Understat (6 Tier-1 leagues).
    """
    _emit(progress_cb, {"type": "info",
                        "msg": "FBref: ВНИМАНИЕ — Cloudflare блокирует доступ. "
                               "Пропускаю (xG доступен только через Understat для 6 лиг)"})
    with db.connect() as conn:
        log_id = db.log_start(conn, "fbref_xg", "collect")
        db.log_finish(conn, log_id, status="ok",
                      rows_added=0, rows_skipped=0,
                      details={"reason": "Cloudflare Turnstile blocks automated access"})
    return {"saved": 0, "skipped": 0, "errors": 0, "reason": "blocked_by_cloudflare"}


# ── FlashScore / ESPN / TheSportsDB ──────────────────────────────────────────

def collect_flashscore(progress_cb: Optional[ProgressCB] = None,
                       cancel_event: Optional[threading.Event] = None,
                       live_only: bool = False) -> dict:
    label = "live" if live_only else "все матчи ±неделя"
    _emit(progress_cb, {"type": "info", "msg": f"Загружаю {label}…"})

    log_id = None
    with db.connect() as conn:
        log_id = db.log_start(conn, "flashscore", "collect")

    bag: List[dict] = []

    try:
        _emit(progress_cb, {"type": "info", "msg": "→ ESPN…"})
        if live_only:
            esp = espn.fetch_live(progress_cb=progress_cb)
        else:
            esp = espn.fetch_week(progress_cb=progress_cb, cancel_event=cancel_event)
        bag.extend(esp)
        _emit(progress_cb, {"type": "success", "msg": f"✓ ESPN: {len(esp)}"})
    except Exception as e:
        _emit(progress_cb, {"type": "error", "msg": f"✗ ESPN: {e}"})

    if cancel_event and cancel_event.is_set():
        result = _save_live_bag(bag, progress_cb)
        with db.connect() as conn:
            db.log_finish(conn, log_id, status="ok", rows_added=result.get("matches", 0))
        return result

    try:
        _emit(progress_cb, {"type": "info", "msg": "→ TheSportsDB…"})
        if live_only:
            tsdb = thesportsdb.fetch_live_matches()
        else:
            tsdb = []
            for offset in range(-1, 7):
                if cancel_event and cancel_event.is_set(): break
                d = (dt.date.today() + dt.timedelta(days=offset)).isoformat()
                tsdb.extend(thesportsdb.fetch_events_for_date(d))
        bag.extend(tsdb)
        _emit(progress_cb, {"type": "success", "msg": f"✓ TheSportsDB: +{len(tsdb)}"})
    except Exception as e:
        _emit(progress_cb, {"type": "error", "msg": f"✗ TheSportsDB: {e}"})

    result = _save_live_bag(bag, progress_cb)
    with db.connect() as conn:
        db.log_finish(conn, log_id, status="ok", rows_added=result.get("matches", 0))
    return result


def _save_live_bag(matches: List[dict], progress_cb: Optional[ProgressCB] = None) -> dict:
    seen = set()
    unique: List[dict] = []
    for m in matches:
        key = (m.get("home", "").lower().strip(),
               m.get("away", "").lower().strip(),
               m.get("time", "")[:10])
        if key in seen: continue
        seen.add(key)
        unique.append(m)

    now = dt.datetime.now().isoformat(timespec="seconds")
    saved = 0
    with db.connect() as conn:
        for m in unique:
            try: db.upsert_fs_match(conn, m, now); saved += 1
            except Exception: pass

    by_status = {}
    for m in unique:
        s = m.get("status", "?")
        by_status[s] = by_status.get(s, 0) + 1

    _emit(progress_cb, {"type": "success",
                        "msg": f"✓ {saved} матчей · "
                               f"{by_status.get('live',0)} live · "
                               f"{by_status.get('finished',0)} завершено"})
    return {"matches": saved, "by_status": by_status}


# ── Top-level orchestrators ───────────────────────────────────────────────────

def collect_all(seasons: Optional[List[int]] = None,
                progress_cb: Optional[ProgressCB] = None,
                cancel_event: Optional[threading.Event] = None) -> dict:
    current = _current_season_year()
    seasons = seasons or list(range(current - 7, current + 1))
    out = {}

    # Register all configured leagues with proper tiers first
    _emit(progress_cb, {"type": "info", "msg": "Регистрирую лиги с тирами..."})
    try:
        with db.connect() as conn:
            for slug, info in config.LEAGUE_TIERS.items():
                db.upsert_league(conn, slug, info["name"], info["country"],
                                 tier=info["tier"])
    except Exception as e:
        _emit(progress_cb, {"type": "error", "msg": f"Ошибка регистрации лиг: {e}"})

    sources = [
        ("understat", collect_understat, {"seasons": seasons}),
        ("espn_all", collect_espn_all, {}),
        ("elo", collect_elo, {}),
        ("odds", collect_odds, {"seasons": seasons}),
        ("weather", collect_weather, {}),
        ("injuries", collect_injuries, {}),
        ("sstats", collect_sstats, {}),
        ("sstats_bulk", collect_sstats_bulk, {"seasons": [current]}),
        ("flashscore", collect_flashscore, {}),
    ]

    # FBref is broken (Cloudflare Turnstile blocks all automated access)
    # Only attempt if explicitly requested via --fbref flag
    # sources.append(("fbref_xg", collect_fbref_xg, {}))
    total_steps = len(sources)

    _emit(progress_cb, {"type": "start",
                        "msg": f"=== Полный сбор данных ({total_steps} этапов) ===",
                        "summary": {"leagues_done": 0, "leagues_total": total_steps}})

    for step_idx, (source_name, source_fn, kwargs) in enumerate(sources, 1):
        if cancel_event and cancel_event.is_set():
            break
        _emit(progress_cb, {"type": "info",
                            "msg": f"[{step_idx}/{total_steps}] {source_name}…",
                            "summary": {"leagues_done": step_idx - 1, "leagues_total": total_steps}})
        try:
            out[source_name] = source_fn(
                progress_cb=progress_cb, cancel_event=cancel_event, **kwargs)
        except Exception as e:
            _emit(progress_cb, {"type": "error",
                                "msg": f"✗ {source_name}: критическая ошибка — {e}"})
            out[source_name] = {"error": str(e)}

    # Compute quality report
    _emit(progress_cb, {"type": "info", "msg": "Считаю качество данных…"})
    try:
        with db.connect() as conn:
            quality = db.compute_quality(conn)
        out["quality"] = quality
        _emit(progress_cb, {"type": "success",
                            "msg": f"✓ Качество: {quality['total_matches']} матчей, "
                                   f"xG: {quality['coverage_xg_pct']}%, "
                                   f"odds: {quality['coverage_odds_pct']}%, "
                                   f"elo: {quality['coverage_elo_pct']}%"})
    except Exception as e:
        _emit(progress_cb, {"type": "error", "msg": f"✗ Quality report: {e}"})

    _emit(progress_cb, {"type": "done",
                        "msg": "=== Сбор завершён ===", "summary": out})
    return out


def collect(seasons, leagues=None, progress_cb=None, cancel_event=None):
    return collect_understat(seasons, leagues, progress_cb, cancel_event)


def refresh_current_season(progress_cb=None, cancel_event=None):
    return collect_understat([_current_season_year()],
                              progress_cb=progress_cb, cancel_event=cancel_event)


def refresh_live_only(progress_cb=None, cancel_event=None):
    return collect_flashscore(progress_cb=progress_cb, cancel_event=cancel_event,
                              live_only=True)


def export_json(output_path=None):
    """Export DB to JSON file."""
    path = db.export_json(output_path)
    return path


def health_check(progress_cb=None) -> dict:
    """Print a comprehensive data quality and health report."""
    import json as _json

    with db.connect() as conn:
        quality = db.compute_quality(conn)
        logs = db.get_data_logs(conn, 50)

    _emit(progress_cb, {"type": "info", "msg": "=== HEALTH CHECK ==="})

    # Source health: last successful run + error rate
    source_stats = {}
    for log in logs:
        src = log["source"]
        if src not in source_stats:
            source_stats[src] = {"last_ok": None, "last_run": None, "total": 0, "errors": 0}
        s = source_stats[src]
        s["total"] += 1
        s["errors"] += log["errors"]
        if log["status"] == "ok" and not s["last_ok"]:
            s["last_ok"] = log["started_at"]
        if not s["last_run"]:
            s["last_run"] = log["started_at"]

    _emit(progress_cb, {"type": "info", "msg": "\n--- Источники ---"})
    for src, stats in sorted(source_stats.items()):
        err_rate = stats["errors"] / max(stats["total"], 1) * 100
        status = "✓" if stats["last_ok"] else "✗"
        _emit(progress_cb, {"type": "info",
                            "msg": f"  {status} {src:20s} last_ok={str(stats['last_ok'] or 'never')[:16]} "
                                   f"err_rate={err_rate:.0f}% runs={stats['total']}"})

    # Per-league breakdown
    _emit(progress_cb, {"type": "info", "msg": "\n--- Покрытие по лигам ---"})
    with db.connect() as conn:
        leagues = conn.execute("""
            SELECT m.league_slug,
                   COUNT(*) as total,
                   SUM(CASE WHEN m.home_xg IS NOT NULL THEN 1 ELSE 0 END) as with_xg,
                   SUM(CASE WHEN m.is_result=1 THEN 1 ELSE 0 END) as finished,
                   MIN(m.date) as first_date,
                   MAX(m.date) as last_date
            FROM matches m
            GROUP BY m.league_slug
            ORDER BY total DESC
        """).fetchall()

        for row in leagues:
            slug = row[0]
            total = row[1]
            xg_pct = row[2] / total * 100 if total else 0
            finished = row[3]
            first = (row[4] or "")[:10]
            last = (row[5] or "")[:10]
            teams = conn.execute(
                "SELECT COUNT(*) FROM teams WHERE league_slug=?", (slug,)
            ).fetchone()[0]
            elo = conn.execute(
                "SELECT COUNT(DISTINCT t.id) FROM teams t "
                "JOIN team_elo te ON te.team_id = t.id "
                "WHERE t.league_slug=?", (slug,)
            ).fetchone()[0]
            tier_info = config.LEAGUE_TIERS.get(slug, {})
            tier = tier_info.get("tier", "?")
            _emit(progress_cb, {"type": "info",
                                "msg": f"  {slug:20s} T{tier} n={total:5d} xG={xg_pct:5.1f}% "
                                       f"teams={teams:3d} elo={elo:3d} {first}..{last}"})

    # Odds coverage
    _emit(progress_cb, {"type": "info", "msg": "\n--- Связность данных ---"})
    with db.connect() as conn:
        odds_match = conn.execute("SELECT COUNT(DISTINCT match_id) FROM match_odds").fetchone()[0]
        elo_teams = conn.execute("SELECT COUNT(*) FROM team_elo").fetchone()[0]
        elo_hist = conn.execute("SELECT COUNT(*) FROM team_elo_history").fetchone()[0]
        injuries = conn.execute("SELECT COUNT(*) FROM injuries").fetchone()[0]
        weather = conn.execute("SELECT COUNT(*) FROM weather").fetchone()[0]
        _emit(progress_cb, {"type": "info", "msg": f"  Odds: {odds_match} матчей"})
        _emit(progress_cb, {"type": "info", "msg": f"  Elo: {elo_teams} команд, {elo_hist} исторических записей"})
        _emit(progress_cb, {"type": "info", "msg": f"  Injuries: {injuries} записей"})
        _emit(progress_cb, {"type": "info", "msg": f"  Weather: {weather} записей"})

    return quality


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current-only", action="store_true")
    ap.add_argument("--live-only", action="store_true")
    ap.add_argument("--seasons", type=int, nargs="+")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--sequential", action="store_true")
    ap.add_argument("--export", action="store_true", help="Export DB to JSON")
    ap.add_argument("--health", action="store_true", help="Print data quality report")
    ap.add_argument("--fbref", action="store_true", help="Collect xG from FBref (25+ leagues)")
    ap.add_argument("--sstats", action="store_true", help="Collect SStats odds/statistics/events")
    ap.add_argument("--sstats-bulk", action="store_true", help="Bulk extract SStats data via /Games/query")
    args = ap.parse_args()

    if args.export:
        path = export_json()
        print(f"Exported to {path}")
    elif args.health:
        health_check()
    elif args.sstats_bulk:
        collect_sstats_bulk(seasons=args.seasons)
    elif args.sstats:
        collect_sstats()
    elif args.fbref:
        collect_fbref_xg()
    elif args.live_only:
        refresh_live_only()
    elif args.full:
        collect_all(args.seasons)
    elif args.current_only:
        refresh_current_season()
    else:
        current = _current_season_year()
        seasons = args.seasons or list(range(current - 3, current + 1))
        collect_understat(seasons, parallel=not args.sequential)
