"""
db.py — SQLite layer for football data.

Single-file DB at data/football.db. No external server required.
Stores leagues, teams, matches (with xG, odds, forecasts),
and lightweight metadata about when data was last refreshed.

New in v2:
  - data_log: history of every collection run
  - quality stats: automatic report after each collection
  - JSON export for external tools
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Tuple

DB_PATH = Path(__file__).parent / "data" / "football.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS leagues (
    slug      TEXT PRIMARY KEY,
    name      TEXT NOT NULL,
    country   TEXT NOT NULL,
    source_tier INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS teams (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    short_name  TEXT,
    league_slug TEXT NOT NULL,
    FOREIGN KEY (league_slug) REFERENCES leagues(slug)
);

CREATE TABLE IF NOT EXISTS matches (
    id            INTEGER PRIMARY KEY,
    league_slug   TEXT NOT NULL,
    season        INTEGER NOT NULL,
    date          TEXT NOT NULL,
    home_id       INTEGER NOT NULL,
    away_id       INTEGER NOT NULL,
    home_goals    INTEGER,
    away_goals    INTEGER,
    home_xg       REAL,
    away_xg       REAL,
    is_result     INTEGER NOT NULL DEFAULT 0,
    forecast_w    REAL,
    forecast_d    REAL,
    forecast_l    REAL,
    FOREIGN KEY (home_id)     REFERENCES teams(id),
    FOREIGN KEY (away_id)     REFERENCES teams(id),
    FOREIGN KEY (league_slug) REFERENCES leagues(slug)
);

CREATE INDEX IF NOT EXISTS idx_matches_date     ON matches(date);
CREATE INDEX IF NOT EXISTS idx_matches_league   ON matches(league_slug, season);
CREATE INDEX IF NOT EXISTS idx_matches_home     ON matches(home_id);
CREATE INDEX IF NOT EXISTS idx_matches_away     ON matches(away_id);
CREATE INDEX IF NOT EXISTS idx_matches_isresult ON matches(is_result);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

-- ClubElo rating per team (latest snapshot)
CREATE TABLE IF NOT EXISTS team_elo (
    team_id    INTEGER PRIMARY KEY,
    team_name  TEXT NOT NULL,
    elo        REAL NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id)
);

-- Full Elo history per team (time-aware training, no data leakage)
CREATE TABLE IF NOT EXISTS team_elo_history (
    team_id  INTEGER NOT NULL,
    date     TEXT NOT NULL,
    elo      REAL NOT NULL,
    PRIMARY KEY (team_id, date)
);
CREATE INDEX IF NOT EXISTS idx_elo_hist_team ON team_elo_history(team_id, date);

-- Weather forecast keyed by upcoming match
CREATE TABLE IF NOT EXISTS weather (
    match_id   INTEGER PRIMARY KEY,
    temp_c     REAL,
    rain_mm    REAL,
    wind_ms    REAL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

-- Active injuries / suspensions per team
CREATE TABLE IF NOT EXISTS injuries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id     INTEGER NOT NULL,
    player_name TEXT NOT NULL,
    reason      TEXT,
    since       TEXT,
    until       TEXT,
    updated_at  TEXT NOT NULL,
    FOREIGN KEY (team_id) REFERENCES teams(id)
);
CREATE INDEX IF NOT EXISTS idx_injuries_team ON injuries(team_id);

-- Bookmaker odds per match
CREATE TABLE IF NOT EXISTS match_odds (
    match_id    INTEGER PRIMARY KEY,
    home_odds   REAL NOT NULL,
    draw_odds   REAL NOT NULL,
    away_odds   REAL NOT NULL,
    implied_h   REAL NOT NULL,
    implied_d   REAL NOT NULL,
    implied_a   REAL NOT NULL,
    source      TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    FOREIGN KEY (match_id) REFERENCES matches(id)
);

-- FlashScore worldwide matches (broad coverage)
CREATE TABLE IF NOT EXISTS fs_matches (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    country    TEXT,
    league     TEXT,
    home       TEXT NOT NULL,
    away       TEXT NOT NULL,
    score_home TEXT,
    score_away TEXT,
    time       TEXT,
    status     TEXT,
    fetched_at TEXT NOT NULL,
    UNIQUE (country, league, home, away, time)
);
CREATE INDEX IF NOT EXISTS idx_fs_status ON fs_matches(status);
CREATE INDEX IF NOT EXISTS idx_fs_fetched ON fs_matches(fetched_at);
CREATE INDEX IF NOT EXISTS idx_fs_country_league ON fs_matches(country, league);

-- Saved predictions
CREATE TABLE IF NOT EXISTS predictions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    home_name   TEXT NOT NULL,
    away_name   TEXT NOT NULL,
    league      TEXT,
    match_date  TEXT,
    analysis    TEXT,
    main_bet    TEXT,
    confidence  TEXT,
    home_win    REAL,
    draw_prob   REAL,
    away_win    REAL,
    total_over  REAL,
    total_under REAL,
    btts_yes    REAL,
    btts_no     REAL,
    exact_score TEXT,
    model_used  TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_predictions_date ON predictions(created_at);

-- ── SStats detailed match data ─────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sstats_matches (
    game_id     INTEGER PRIMARY KEY,
    league_id   INTEGER,
    league_name TEXT,
    season      INTEGER,
    date        TEXT NOT NULL,
    home_team   TEXT,
    away_team   TEXT,
    home_id     INTEGER,
    away_id     INTEGER,
    status      TEXT,           -- 'Not Started' | 'First Half' | 'Finished' etc.
    home_result INTEGER,
    away_result INTEGER,
    home_ht     INTEGER,
    away_ht     INTEGER,
    round_name  TEXT,
    venue       TEXT,
    raw_json    TEXT,           -- full JSON response for future use
    collected_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sstats_date ON sstats_matches(date);
CREATE INDEX IF NOT EXISTS idx_sstats_status ON sstats_matches(status);

-- SStats odds (multiple bookmakers per match)
CREATE TABLE IF NOT EXISTS sstats_odds (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     INTEGER NOT NULL,
    bookmaker   TEXT NOT NULL,
    market      TEXT NOT NULL,
    name        TEXT NOT NULL,    -- 'Home' | 'Draw' | 'Away' | 'Over 2.5' etc.
    value       REAL,
    collected_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES sstats_matches(game_id)
);
CREATE INDEX IF NOT EXISTS idx_sstats_odds_game ON sstats_odds(game_id);

-- SStats game statistics (possession, shots, corners etc.)
CREATE TABLE IF NOT EXISTS sstats_statistics (
    game_id     INTEGER NOT NULL,
    stat_name   TEXT NOT NULL,
    home_value  TEXT,
    away_value  TEXT,
    collected_at TEXT NOT NULL,
    PRIMARY KEY (game_id, stat_name)
);

-- SStats events (goals, cards, substitutions)
CREATE TABLE IF NOT EXISTS sstats_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     INTEGER NOT NULL,
    minute      INTEGER,
    event_type  TEXT,
    player      TEXT,
    team        TEXT,
    detail      TEXT,
    collected_at TEXT NOT NULL,
    FOREIGN KEY (game_id) REFERENCES sstats_matches(game_id)
);
CREATE INDEX IF NOT EXISTS idx_sstats_events_game ON sstats_events(game_id);

-- ── NEW: Data collection log ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source      TEXT NOT NULL,        -- 'understat' | 'espn' | 'sstats' | 'elo' | ...
    action      TEXT NOT NULL,        -- 'collect' | 'refresh' | 'export'
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    status      TEXT NOT NULL DEFAULT 'running',  -- 'running' | 'ok' | 'error'
    rows_added  INTEGER DEFAULT 0,
    rows_updated INTEGER DEFAULT 0,
    rows_skipped INTEGER DEFAULT 0,
    errors      INTEGER DEFAULT 0,
    details     TEXT,                  -- JSON with extra info
    error_msg   TEXT                   -- error message if status='error'
);
CREATE INDEX IF NOT EXISTS idx_data_log_source ON data_log(source);
CREATE INDEX IF NOT EXISTS idx_data_log_time ON data_log(started_at);

-- ── NEW: Data quality report ──────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS data_quality (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_date TEXT NOT NULL,
    total_matches INTEGER DEFAULT 0,
    matches_with_xg INTEGER DEFAULT 0,
    matches_with_odds INTEGER DEFAULT 0,
    matches_with_forecast INTEGER DEFAULT 0,
    total_teams INTEGER DEFAULT 0,
    teams_with_elo INTEGER DEFAULT 0,
    teams_with_injuries INTEGER DEFAULT 0,
    total_leagues INTEGER DEFAULT 0,
    coverage_xg_pct REAL DEFAULT 0,
    coverage_odds_pct REAL DEFAULT 0,
    coverage_elo_pct REAL DEFAULT 0,
    details     TEXT                  -- JSON with per-league breakdown
);
"""


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(SCHEMA)
        # Migrations for existing databases
        _migrate(conn)


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply incremental schema migrations to existing databases."""
    # Check if source_tier column exists on leagues table
    cols = {r[1] for r in conn.execute("PRAGMA table_info(leagues)").fetchall()}
    if "source_tier" not in cols:
        conn.execute("ALTER TABLE leagues ADD COLUMN source_tier INTEGER NOT NULL DEFAULT 1")


def delete_db() -> None:
    """Delete the entire database file. Use before fresh rebuild."""
    if DB_PATH.exists():
        DB_PATH.unlink()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Upserts ───────────────────────────────────────────────────────────────────

def upsert_league(conn: sqlite3.Connection, slug: str, name: str, country: str,
                   tier: int = 1) -> None:
    conn.execute(
        "INSERT INTO leagues(slug, name, country, source_tier) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(slug) DO UPDATE SET name=excluded.name, country=excluded.country, "
        "source_tier=excluded.source_tier",
        (slug, name, country, tier),
    )


def upsert_team(conn: sqlite3.Connection, team_id: int, name: str,
                short_name: Optional[str], league_slug: str) -> None:
    conn.execute(
        "INSERT INTO teams(id, name, short_name, league_slug) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(id) DO UPDATE SET name=excluded.name, "
        "short_name=excluded.short_name, league_slug=excluded.league_slug",
        (team_id, name, short_name, league_slug),
    )


def upsert_match(conn: sqlite3.Connection, m: Dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO matches(id, league_slug, season, date, home_id, away_id,
                            home_goals, away_goals, home_xg, away_xg, is_result,
                            forecast_w, forecast_d, forecast_l)
        VALUES (:id, :league_slug, :season, :date, :home_id, :away_id,
                :home_goals, :away_goals, :home_xg, :away_xg, :is_result,
                :forecast_w, :forecast_d, :forecast_l)
        ON CONFLICT(id) DO UPDATE SET
            date        = excluded.date,
            home_goals  = excluded.home_goals,
            away_goals  = excluded.away_goals,
            home_xg     = excluded.home_xg,
            away_xg     = excluded.away_xg,
            is_result   = excluded.is_result,
            forecast_w  = excluded.forecast_w,
            forecast_d  = excluded.forecast_d,
            forecast_l  = excluded.forecast_l
        """,
        m,
    )


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


# ── Data log ──────────────────────────────────────────────────────────────────

def log_start(conn: sqlite3.Connection, source: str, action: str = "collect") -> int:
    """Log the start of a collection run. Returns the log id."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO data_log(source, action, started_at, status) VALUES (?, ?, ?, 'running')",
        (source, action, now),
    )
    return cur.lastrowid


def log_finish(conn: sqlite3.Connection, log_id: int, *,
               status: str = "ok", rows_added: int = 0, rows_updated: int = 0,
               rows_skipped: int = 0, errors: int = 0,
               details: Any = None, error_msg: str = "") -> None:
    """Update a data_log entry with results."""
    now = dt.datetime.now().isoformat(timespec="seconds")
    details_json = json.dumps(details, ensure_ascii=False) if details else None
    conn.execute(
        "UPDATE data_log SET finished_at=?, status=?, rows_added=?, rows_updated=?, "
        "rows_skipped=?, errors=?, details=?, error_msg=? WHERE id=?",
        (now, status, rows_added, rows_updated, rows_skipped, errors,
         details_json, error_msg, log_id),
    )


def get_data_logs(conn: sqlite3.Connection, limit: int = 20) -> List[Dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM data_log ORDER BY started_at DESC LIMIT ?", (limit,)
    ).fetchall()
    return [dict(r) for r in rows]


# ── Data quality ──────────────────────────────────────────────────────────────

def compute_quality(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Compute data quality report from current DB state."""
    total_matches = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    matches_with_xg = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE home_xg IS NOT NULL"
    ).fetchone()[0]
    matches_with_odds = conn.execute(
        "SELECT COUNT(*) FROM match_odds"
    ).fetchone()[0]
    matches_with_forecast = conn.execute(
        "SELECT COUNT(*) FROM matches WHERE forecast_w IS NOT NULL"
    ).fetchone()[0]
    total_teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
    teams_with_elo = conn.execute("SELECT COUNT(*) FROM team_elo").fetchone()[0]
    teams_with_injuries = conn.execute("SELECT COUNT(*) FROM injuries").fetchone()[0]
    total_leagues = conn.execute("SELECT COUNT(*) FROM leagues").fetchone()[0]

    xg_pct = round(matches_with_xg / total_matches * 100, 1) if total_matches else 0
    odds_pct = round(matches_with_odds / total_matches * 100, 1) if total_matches else 0
    elo_pct = round(teams_with_elo / total_teams * 100, 1) if total_teams else 0

    # Per-league breakdown
    league_stats = []
    for row in conn.execute(
        "SELECT league_slug, COUNT(*) as total, "
        "SUM(CASE WHEN home_xg IS NOT NULL THEN 1 ELSE 0 END) as with_xg "
        "FROM matches GROUP BY league_slug ORDER BY league_slug"
    ).fetchall():
        ls = dict(row)
        ls["xg_pct"] = round(ls["with_xg"] / ls["total"] * 100, 1) if ls["total"] else 0
        league_stats.append(ls)

    # Per-tier breakdown
    tier_stats = []
    for row in conn.execute(
        "SELECT l.source_tier, COUNT(DISTINCT l.slug) as leagues, "
        "COUNT(m.id) as matches, "
        "SUM(CASE WHEN m.home_xg IS NOT NULL THEN 1 ELSE 0 END) as with_xg "
        "FROM leagues l LEFT JOIN matches m ON m.league_slug = l.slug "
        "GROUP BY l.source_tier ORDER BY l.source_tier"
    ).fetchall():
        ts = dict(row)
        ts["xg_pct"] = round(ts["with_xg"] / ts["matches"] * 100, 1) if ts["matches"] else 0
        tier_stats.append(ts)

    report = {
        "report_date": dt.datetime.now().isoformat(timespec="seconds"),
        "total_matches": total_matches,
        "matches_with_xg": matches_with_xg,
        "matches_with_odds": matches_with_odds,
        "matches_with_forecast": matches_with_forecast,
        "total_teams": total_teams,
        "teams_with_elo": teams_with_elo,
        "teams_with_injuries": teams_with_injuries,
        "total_leagues": total_leagues,
        "coverage_xg_pct": xg_pct,
        "coverage_odds_pct": odds_pct,
        "coverage_elo_pct": elo_pct,
        "league_stats": league_stats,
        "tier_stats": tier_stats,
    }

    # Save to DB
    conn.execute(
        "INSERT INTO data_quality(report_date, total_matches, matches_with_xg, "
        "matches_with_odds, matches_with_forecast, total_teams, teams_with_elo, "
        "teams_with_injuries, total_leagues, coverage_xg_pct, coverage_odds_pct, "
        "coverage_elo_pct, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (report["report_date"], total_matches, matches_with_xg, matches_with_odds,
         matches_with_forecast, total_teams, teams_with_elo, teams_with_injuries,
         total_leagues, xg_pct, odds_pct, elo_pct,
         json.dumps(league_stats, ensure_ascii=False)),
    )

    return report


# ── JSON export ───────────────────────────────────────────────────────────────

def export_json(output_path: Optional[str] = None) -> str:
    """Export entire DB to JSON. Returns path to the file."""
    if output_path is None:
        output_path = str(DB_PATH.parent / "football_export.json")

    SAFE_TABLES = {"leagues", "teams", "matches", "team_elo", "weather",
                   "injuries", "match_odds", "fs_matches", "predictions",
                   "data_log", "data_quality", "sstats_matches", "sstats_odds",
                   "sstats_statistics", "sstats_events"}

    data = {}
    with connect() as conn:
        for table in SAFE_TABLES:
            rows = conn.execute(f"SELECT * FROM [{table}]").fetchall()
            data[table] = [dict(r) for r in rows]

        # Quality report
        data["quality_report"] = compute_quality(conn)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return output_path


# ── Reads ─────────────────────────────────────────────────────────────────────

def list_leagues() -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT slug, name, country, source_tier FROM leagues ORDER BY name").fetchall()
        return [dict(r) for r in rows]


def list_trainable_leagues() -> List[Dict[str, Any]]:
    """Leagues with tier <= 2 (usable for model training)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT slug, name, country, source_tier FROM leagues "
            "WHERE source_tier <= 2 ORDER BY source_tier, name"
        ).fetchall()
        return [dict(r) for r in rows]


def list_teams(league_slug: Optional[str] = None) -> List[Dict[str, Any]]:
    with connect() as conn:
        if league_slug:
            rows = conn.execute(
                "SELECT id, name, short_name, league_slug FROM teams "
                "WHERE league_slug=? ORDER BY name",
                (league_slug,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, name, short_name, league_slug FROM teams ORDER BY name"
            ).fetchall()
        return [dict(r) for r in rows]


def get_team(team_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT id, name, short_name, league_slug FROM teams WHERE id=?",
            (team_id,),
        ).fetchone()
        return dict(row) if row else None


def search_team_fuzzy(name: str, limit: int = 5) -> List[Dict[str, Any]]:
    q = name.strip().lower()
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, name, short_name, league_slug FROM teams "
            "WHERE LOWER(name) LIKE ? OR LOWER(short_name) LIKE ? "
            "ORDER BY "
            "  CASE WHEN LOWER(name)=? THEN 0 "
            "       WHEN LOWER(name) LIKE ? THEN 1 "
            "       WHEN LOWER(short_name)=? THEN 2 "
            "       ELSE 3 END, "
            "  name LIMIT ?",
            (f"%{q}%", f"%{q}%", q, f"{q}%", q, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]


def matches_played(team_id: int, before_date: Optional[str] = None,
                   limit: Optional[int] = None) -> List[Dict[str, Any]]:
    sql = ("SELECT * FROM matches WHERE is_result=1 AND (home_id=? OR away_id=?) ")
    args: List[Any] = [team_id, team_id]
    if before_date:
        sql += "AND date < ? "
        args.append(before_date)
    sql += "ORDER BY date DESC "
    if limit:
        sql += "LIMIT ? "
        args.append(int(limit))
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def head_to_head(home_id: int, away_id: int,
                 before_date: Optional[str] = None,
                 limit: int = 10) -> List[Dict[str, Any]]:
    sql = ("SELECT * FROM matches WHERE is_result=1 "
           "AND ((home_id=? AND away_id=?) OR (home_id=? AND away_id=?)) ")
    args: List[Any] = [home_id, away_id, away_id, home_id]
    if before_date:
        sql += "AND date < ? "
        args.append(before_date)
    sql += "ORDER BY date DESC LIMIT ?"
    args.append(int(limit))
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def upcoming_matches(league_slug: Optional[str] = None,
                     limit: int = 20) -> List[Dict[str, Any]]:
    sql = (
        "SELECT m.*, ht.name AS home_name, at.name AS away_name, "
        "       ht.short_name AS home_short, at.short_name AS away_short "
        "FROM matches m "
        "JOIN teams ht ON ht.id = m.home_id "
        "JOIN teams at ON at.id = m.away_id "
        "WHERE m.is_result=0 "
    )
    args: List[Any] = []
    if league_slug:
        sql += "AND m.league_slug=? "
        args.append(league_slug)
    sql += "ORDER BY m.date ASC LIMIT ?"
    args.append(int(limit))
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def recent_results(league_slug: Optional[str] = None,
                   limit: int = 20) -> List[Dict[str, Any]]:
    sql = (
        "SELECT m.*, ht.name AS home_name, at.name AS away_name, "
        "       ht.short_name AS home_short, at.short_name AS away_short "
        "FROM matches m "
        "JOIN teams ht ON ht.id = m.home_id "
        "JOIN teams at ON at.id = m.away_id "
        "WHERE m.is_result=1 "
    )
    args: List[Any] = []
    if league_slug:
        sql += "AND m.league_slug=? "
        args.append(league_slug)
    sql += "ORDER BY m.date DESC LIMIT ?"
    args.append(int(limit))
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def standings(league_slug: str, season: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT m.home_id, m.away_id, m.home_goals, m.away_goals, "
            "       m.home_xg, m.away_xg "
            "FROM matches m "
            "WHERE m.league_slug=? AND m.season=? AND m.is_result=1",
            (league_slug, season),
        ).fetchall()
        teams = {t["id"]: t for t in list_teams(league_slug)}

    table: Dict[int, Dict[str, Any]] = {
        tid: {
            "team_id": tid, "team": t["name"], "short": t["short_name"],
            "played": 0, "wins": 0, "draws": 0, "losses": 0,
            "gf": 0, "ga": 0, "gd": 0, "points": 0,
            "xg_for": 0.0, "xg_against": 0.0,
        }
        for tid, t in teams.items()
    }

    for r in rows:
        h, a = r["home_id"], r["away_id"]
        if h not in table or a not in table:
            continue
        hg, ag = r["home_goals"] or 0, r["away_goals"] or 0
        hx, ax = r["home_xg"] or 0.0, r["away_xg"] or 0.0
        table[h]["played"] += 1
        table[a]["played"] += 1
        table[h]["gf"] += hg; table[h]["ga"] += ag
        table[a]["gf"] += ag; table[a]["ga"] += hg
        table[h]["xg_for"] += hx; table[h]["xg_against"] += ax
        table[a]["xg_for"] += ax; table[a]["xg_against"] += hx
        if hg > ag:
            table[h]["wins"] += 1; table[h]["points"] += 3
            table[a]["losses"] += 1
        elif hg < ag:
            table[a]["wins"] += 1; table[a]["points"] += 3
            table[h]["losses"] += 1
        else:
            table[h]["draws"] += 1; table[h]["points"] += 1
            table[a]["draws"] += 1; table[a]["points"] += 1

    for t in table.values():
        t["gd"] = t["gf"] - t["ga"]
        t["xg_for"] = round(t["xg_for"], 2)
        t["xg_against"] = round(t["xg_against"], 2)
        t["xg_diff"] = round(t["xg_for"] - t["xg_against"], 2)

    rows_out = [t for t in table.values() if t["played"] > 0]
    rows_out.sort(key=lambda x: (-x["points"], -x["gd"], -x["gf"]))
    for i, t in enumerate(rows_out, 1):
        t["pos"] = i
    return rows_out


# ── New-source helpers ────────────────────────────────────────────────────────

def upsert_team_elo(conn: sqlite3.Connection, team_id: int, team_name: str,
                    elo: float, updated_at: str) -> None:
    conn.execute(
        "INSERT INTO team_elo(team_id, team_name, elo, updated_at) VALUES (?, ?, ?, ?) "
        "ON CONFLICT(team_id) DO UPDATE SET "
        "team_name=excluded.team_name, elo=excluded.elo, updated_at=excluded.updated_at",
        (team_id, team_name, elo, updated_at),
    )


def get_team_elo(team_id: int) -> Optional[float]:
    with connect() as conn:
        row = conn.execute("SELECT elo FROM team_elo WHERE team_id=?", (team_id,)).fetchone()
        return row["elo"] if row else None


def save_elo_history(conn: sqlite3.Connection, team_id: int,
                     history: List[Tuple[str, float]]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO team_elo_history(team_id, date, elo) VALUES (?, ?, ?)",
        [(team_id, date, elo) for date, elo in history],
    )


def get_elo_at_date(team_id: int, date_iso: str) -> Optional[float]:
    with connect() as conn:
        row = conn.execute(
            "SELECT elo FROM team_elo_history "
            "WHERE team_id=? AND date<=? ORDER BY date DESC LIMIT 1",
            (team_id, date_iso),
        ).fetchone()
        return row["elo"] if row else None


def upsert_match_odds(conn: sqlite3.Connection, match_id: int,
                      home_odds: float, draw_odds: float, away_odds: float,
                      implied_h: float, implied_d: float, implied_a: float,
                      source: str, updated_at: str) -> None:
    conn.execute(
        "INSERT INTO match_odds(match_id, home_odds, draw_odds, away_odds, "
        "implied_h, implied_d, implied_a, source, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(match_id) DO UPDATE SET "
        "home_odds=excluded.home_odds, draw_odds=excluded.draw_odds, "
        "away_odds=excluded.away_odds, implied_h=excluded.implied_h, "
        "implied_d=excluded.implied_d, implied_a=excluded.implied_a, "
        "source=excluded.source, updated_at=excluded.updated_at",
        (match_id, home_odds, draw_odds, away_odds,
         implied_h, implied_d, implied_a, source, updated_at),
    )


def get_match_odds(match_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM match_odds WHERE match_id=?", (match_id,)
        ).fetchone()
        return dict(row) if row else None


def find_match_by_date_teams(league_slug: str, date_iso: str,
                             home_name: str, away_name: str) -> Optional[int]:
    with connect() as conn:
        row = conn.execute(
            "SELECT m.id FROM matches m "
            "JOIN teams th ON th.id = m.home_id "
            "JOIN teams ta ON ta.id = m.away_id "
            "WHERE m.league_slug=? AND date(m.date)=? "
            "AND LOWER(th.name)=LOWER(?) AND LOWER(ta.name)=LOWER(?)",
            (league_slug, date_iso, home_name, away_name),
        ).fetchone()
        if row:
            return row["id"]
        row = conn.execute(
            "SELECT m.id FROM matches m "
            "JOIN teams th ON th.id = m.home_id "
            "JOIN teams ta ON ta.id = m.away_id "
            "WHERE m.league_slug=? AND date(m.date)=? "
            "AND (LOWER(th.name) LIKE LOWER(?) OR LOWER(?) LIKE '%'||LOWER(th.name)||'%') "
            "AND (LOWER(ta.name) LIKE LOWER(?) OR LOWER(?) LIKE '%'||LOWER(ta.name)||'%')",
            (league_slug, date_iso,
             f"%{home_name}%", home_name,
             f"%{away_name}%", away_name),
        ).fetchone()
        return row["id"] if row else None


def bulk_get_elo_at_date(team_dates: List[Tuple[int, str]]) -> Dict[Tuple[int, str], float]:
    out: Dict[Tuple[int, str], float] = {}
    with connect() as conn:
        for team_id, date in team_dates:
            row = conn.execute(
                "SELECT elo FROM team_elo_history "
                "WHERE team_id=? AND date<=? ORDER BY date DESC LIMIT 1",
                (team_id, date),
            ).fetchone()
            if row:
                out[(team_id, date)] = row["elo"]
    return out


def upsert_weather(conn: sqlite3.Connection, match_id: int, temp_c: Optional[float],
                   rain_mm: Optional[float], wind_ms: Optional[float],
                   updated_at: str) -> None:
    conn.execute(
        "INSERT INTO weather(match_id, temp_c, rain_mm, wind_ms, updated_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(match_id) DO UPDATE SET "
        "temp_c=excluded.temp_c, rain_mm=excluded.rain_mm, "
        "wind_ms=excluded.wind_ms, updated_at=excluded.updated_at",
        (match_id, temp_c, rain_mm, wind_ms, updated_at),
    )


def get_weather(match_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT temp_c, rain_mm, wind_ms FROM weather WHERE match_id=?",
            (match_id,),
        ).fetchone()
        return dict(row) if row else None


def replace_team_injuries(conn: sqlite3.Connection, team_id: int,
                          injuries: List[Dict[str, Any]], updated_at: str) -> None:
    conn.execute("DELETE FROM injuries WHERE team_id=?", (team_id,))
    for inj in injuries:
        conn.execute(
            "INSERT INTO injuries(team_id, player_name, reason, since, until, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (team_id, inj.get("player", ""), inj.get("reason", ""),
             inj.get("since", ""), inj.get("until", ""), updated_at),
        )


def list_injuries(team_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT player_name, reason, since, until, updated_at "
            "FROM injuries WHERE team_id=? ORDER BY player_name",
            (team_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def list_all_injuries(league_slug: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = (
        "SELECT i.*, t.name AS team_name, t.league_slug "
        "FROM injuries i JOIN teams t ON t.id = i.team_id "
    )
    args: List[Any] = []
    if league_slug:
        sql += "WHERE t.league_slug=? "
        args.append(league_slug)
    sql += "ORDER BY t.name, i.player_name"
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


def upsert_fs_match(conn: sqlite3.Connection, m: Dict[str, Any], fetched_at: str) -> None:
    conn.execute(
        """
        INSERT INTO fs_matches(country, league, home, away,
                               score_home, score_away, time, status, fetched_at)
        VALUES (:country, :league, :home, :away,
                :score_home, :score_away, :time, :status, :fetched_at)
        ON CONFLICT(country, league, home, away, time) DO UPDATE SET
            score_home=excluded.score_home,
            score_away=excluded.score_away,
            status=excluded.status,
            fetched_at=excluded.fetched_at
        """,
        {**m, "fetched_at": fetched_at},
    )


def fs_countries() -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT country, COUNT(DISTINCT league) AS leagues, "
            "       COUNT(*) AS matches "
            "FROM fs_matches WHERE country IS NOT NULL AND country!='' "
            "GROUP BY country ORDER BY country"
        ).fetchall()
        return [dict(r) for r in rows]


def fs_leagues_for_country(country: str) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT league, COUNT(*) AS matches "
            "FROM fs_matches WHERE country=? AND league IS NOT NULL AND league!='' "
            "GROUP BY league ORDER BY league",
            (country,),
        ).fetchall()
        return [dict(r) for r in rows]


def fs_matches_for(country: str, league: str, limit: int = 100) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM fs_matches WHERE country=? AND league=? "
            "ORDER BY status DESC, fetched_at DESC LIMIT ?",
            (country, league, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]


def fs_live_matches(limit: int = 100) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM fs_matches WHERE status='live' "
            "ORDER BY fetched_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]


def fs_stats() -> Dict[str, int]:
    with connect() as conn:
        r = conn.execute(
            "SELECT COUNT(*) AS total, "
            "       COUNT(DISTINCT country) AS countries, "
            "       COUNT(DISTINCT league) AS leagues, "
            "       SUM(CASE WHEN status='live' THEN 1 ELSE 0 END) AS live "
            "FROM fs_matches"
        ).fetchone()
        return dict(r)


def all_matches_for_training(league_slug: Optional[str] = None) -> List[Dict[str, Any]]:
    sql = "SELECT * FROM matches WHERE is_result=1 "
    args: List[Any] = []
    if league_slug:
        sql += "AND league_slug=? "
        args.append(league_slug)
    sql += "ORDER BY date ASC"
    with connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


# ── Saved predictions ─────────────────────────────────────────────────────

def save_prediction(p: Dict[str, Any]) -> int:
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO predictions "
            "(home_name, away_name, league, match_date, analysis, main_bet, "
            " confidence, home_win, draw_prob, away_win, total_over, total_under, "
            " btts_yes, btts_no, exact_score, model_used, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                p.get("home_name", ""),
                p.get("away_name", ""),
                p.get("league", ""),
                p.get("match_date", ""),
                p.get("analysis", ""),
                p.get("main_bet", ""),
                p.get("confidence", ""),
                p.get("home_win"),
                p.get("draw_prob"),
                p.get("away_win"),
                p.get("total_over"),
                p.get("total_under"),
                p.get("btts_yes"),
                p.get("btts_no"),
                p.get("exact_score", ""),
                p.get("model_used", ""),
                p.get("created_at", dt.datetime.now().isoformat(timespec="seconds")),
            ),
        )
        return cur.lastrowid


def list_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        ).fetchall()
        return [dict(r) for r in rows]


def get_prediction(pred_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM predictions WHERE id=?", (pred_id,)
        ).fetchone()
        return dict(row) if row else None


def delete_prediction(pred_id: int) -> bool:
    with connect() as conn:
        cur = conn.execute("DELETE FROM predictions WHERE id=?", (pred_id,))
        return cur.rowcount > 0


# ── SStats data helpers ──────────────────────────────────────────────────────

def save_sstats_match(conn: sqlite3.Connection, game: dict, collected_at: str) -> None:
    """Save a sstats match to DB."""
    ht = game.get("homeTeam") or {}
    at = game.get("awayTeam") or {}
    season_data = game.get("season") or {}
    league_data = season_data.get("league") or {}
    conn.execute(
        """INSERT INTO sstats_matches(game_id, league_id, league_name, season, date,
            home_team, away_team, home_id, away_id, status,
            home_result, away_result, home_ht, away_ht,
            round_name, venue, raw_json, collected_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_id) DO UPDATE SET
            status=excluded.status, home_result=excluded.home_result,
            away_result=excluded.away_result, home_ht=excluded.home_ht,
            away_ht=excluded.away_ht, raw_json=excluded.raw_json,
            collected_at=excluded.collected_at""",
        (
            game.get("id"),
            league_data.get("id"),
            league_data.get("name", ""),
            season_data.get("year"),
            game.get("date", ""),
            ht.get("name", ""),
            at.get("name", ""),
            ht.get("id"),
            at.get("id"),
            game.get("statusName", ""),
            game.get("homeResult"),
            game.get("awayResult"),
            game.get("homeHTResult"),
            game.get("awayHTResult"),
            game.get("roundName", ""),
            (game.get("venue") or {}).get("fullName", ""),
            str(game),
            collected_at,
        ),
    )


def save_sstats_odds(conn: sqlite3.Connection, game_id: int,
                     odds_blocks: list, collected_at: str) -> int:
    """Save odds from sstats to DB. Returns count of saved rows."""
    # Delete old odds for this game
    conn.execute("DELETE FROM sstats_odds WHERE game_id=?", (game_id,))
    saved = 0
    for bm in odds_blocks:
        bm_name = bm.get("bookmakerName", "unknown")
        for m in (bm.get("odds") or []):
            market = m.get("marketName", "")
            for o in (m.get("odds") or []):
                name = o.get("name", "")
                value = o.get("value")
                if value and float(value) > 1.0:
                    conn.execute(
                        "INSERT INTO sstats_odds(game_id, bookmaker, market, name, value, collected_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (game_id, bm_name, market, name, float(value), collected_at),
                    )
                    saved += 1
    return saved


def save_sstats_statistics(conn: sqlite3.Connection, game_id: int,
                           stats, collected_at: str) -> int:
    """Save match statistics from sstats to DB."""
    if not stats:
        return 0
    conn.execute("DELETE FROM sstats_statistics WHERE game_id=?", (game_id,))
    saved = 0
    if isinstance(stats, dict):
        stats = list(stats.values()) if stats else []
    for s in (stats if isinstance(stats, list) else []):
        if isinstance(s, dict):
            name = s.get("name", "")
            home_val = str((s.get("home") or {}).get("value", ""))
            away_val = str((s.get("away") or {}).get("value", ""))
            if name:
                conn.execute(
                    "INSERT INTO sstats_statistics(game_id, stat_name, home_value, away_value, collected_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (game_id, name, home_val, away_val, collected_at),
                )
                saved += 1
    return saved


def save_sstats_events(conn: sqlite3.Connection, game_id: int,
                       events, collected_at: str) -> int:
    """Save match events from sstats to DB."""
    if not events:
        return 0
    conn.execute("DELETE FROM sstats_events WHERE game_id=?", (game_id,))
    saved = 0
    for ev in (events if isinstance(events, list) else []):
        if isinstance(ev, dict):
            conn.execute(
                "INSERT INTO sstats_events(game_id, minute, event_type, player, team, detail, collected_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    game_id,
                    ev.get("minute"),
                    ev.get("type", ""),
                    (ev.get("player") or {}).get("name", ""),
                    ev.get("team", ""),
                    ev.get("detail", ""),
                    collected_at,
                ),
            )
            saved += 1
    return saved


def get_sstats_match(game_id: int) -> Optional[Dict[str, Any]]:
    with connect() as conn:
        row = conn.execute(
            "SELECT * FROM sstats_matches WHERE game_id=?", (game_id,)
        ).fetchone()
        return dict(row) if row else None


def get_sstats_odds(game_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sstats_odds WHERE game_id=? ORDER BY bookmaker, market",
            (game_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_sstats_statistics(game_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sstats_statistics WHERE game_id=?",
            (game_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def get_sstats_events(game_id: int) -> List[Dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM sstats_events WHERE game_id=? ORDER BY minute",
            (game_id,),
        ).fetchall()
        return [dict(r) for r in rows]
