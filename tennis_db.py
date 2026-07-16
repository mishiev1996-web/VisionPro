"""
tennis_db.py — SQLite layer for tennis data.

Separate database from football: data/tennis.db
Stores players, matches, rankings, H2H, and predictions.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Dict, Any, Optional, Tuple

from config import TENNIS_DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS tennis_players (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    short_name  TEXT,
    country     TEXT,
    ranking     INTEGER,
    ranking_points INTEGER,
    elo         REAL DEFAULT 1500,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS tennis_tournaments (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT,
    surface     TEXT,
    country     TEXT
);

CREATE TABLE IF NOT EXISTS tennis_matches (
    id              INTEGER PRIMARY KEY,
    api_event_id    INTEGER UNIQUE,
    tournament_id   INTEGER,
    tournament_name TEXT,
    round_name      TEXT,
    surface         TEXT,
    date            TEXT NOT NULL,
    player1_id      INTEGER,
    player2_id      INTEGER,
    player1_name    TEXT,
    player2_name    TEXT,
    winner_id       INTEGER,
    score           TEXT,
    status          TEXT DEFAULT 'scheduled',
    player1_elo     REAL,
    player2_elo     REAL,
    player1_ranking INTEGER,
    player2_ranking INTEGER,
    odds_player1    REAL,
    odds_player2    REAL,
    h2h_player1_wins INTEGER DEFAULT 0,
    h2h_player2_wins INTEGER DEFAULT 0,
    h2h_total       INTEGER DEFAULT 0,
    -- Serve statistics (winner)
    w_ace           INTEGER,
    w_df            INTEGER,
    w_svpt          INTEGER,
    w_1stIn         INTEGER,
    w_1stWon        INTEGER,
    w_2ndWon        INTEGER,
    w_SvGms         INTEGER,
    w_bpSaved       INTEGER,
    w_bpFaced       INTEGER,
    -- Serve statistics (loser)
    l_ace           INTEGER,
    l_df            INTEGER,
    l_svpt          INTEGER,
    l_1stIn         INTEGER,
    l_1stWon        INTEGER,
    l_2ndWon        INTEGER,
    l_SvGms         INTEGER,
    l_bpSaved       INTEGER,
    l_bpFaced       INTEGER,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (player1_id) REFERENCES tennis_players(id),
    FOREIGN KEY (player2_id) REFERENCES tennis_players(id)
);

CREATE INDEX IF NOT EXISTS idx_tmatches_date ON tennis_matches(date);
CREATE INDEX IF NOT EXISTS idx_tmatches_status ON tennis_matches(status);
CREATE INDEX IF NOT EXISTS idx_tmatches_p1 ON tennis_matches(player1_id);
CREATE INDEX IF NOT EXISTS idx_tmatches_p2 ON tennis_matches(player2_id);
CREATE INDEX IF NOT EXISTS idx_tmatches_api ON tennis_matches(api_event_id);

CREATE TABLE IF NOT EXISTS tennis_rankings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id       INTEGER,
    player_name     TEXT,
    ranking         INTEGER,
    ranking_points  INTEGER,
    tour            TEXT NOT NULL,
    fetched_at      TEXT NOT NULL,
    FOREIGN KEY (player_id) REFERENCES tennis_players(id)
);

CREATE TABLE IF NOT EXISTS tennis_predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    player1_name    TEXT NOT NULL,
    player2_name    TEXT NOT NULL,
    tournament      TEXT,
    match_date      TEXT,
    analysis        TEXT,
    main_bet        TEXT,
    confidence      TEXT,
    player1_win     REAL,
    player2_win     REAL,
    model_used      TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tennis_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@contextmanager
def connect():
    TENNIS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(TENNIS_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with connect() as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)


def _migrate(conn):
    """Apply incremental schema migrations."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tennis_matches)").fetchall()}
    # Add serve statistics columns if missing
    serve_cols = [
        ("w_ace", "INTEGER"), ("w_df", "INTEGER"), ("w_svpt", "INTEGER"),
        ("w_1stIn", "INTEGER"), ("w_1stWon", "INTEGER"), ("w_2ndWon", "INTEGER"),
        ("w_SvGms", "INTEGER"), ("w_bpSaved", "INTEGER"), ("w_bpFaced", "INTEGER"),
        ("l_ace", "INTEGER"), ("l_df", "INTEGER"), ("l_svpt", "INTEGER"),
        ("l_1stIn", "INTEGER"), ("l_1stWon", "INTEGER"), ("l_2ndWon", "INTEGER"),
        ("l_SvGms", "INTEGER"), ("l_bpSaved", "INTEGER"), ("l_bpFaced", "INTEGER"),
    ]
    for col_name, col_type in serve_cols:
        if col_name not in cols:
            conn.execute(f"ALTER TABLE tennis_matches ADD COLUMN {col_name} {col_type}")


def set_meta(conn, key: str, value: str):
    conn.execute(
        "INSERT OR REPLACE INTO tennis_meta (key, value) VALUES (?, ?)",
        (key, value),
    )


def get_meta(conn, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM tennis_meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


# ── Players ──────────────────────────────────────────────────────────────────

def upsert_player(conn, player: dict):
    conn.execute("""
        INSERT INTO tennis_players (id, name, short_name, country, ranking, ranking_points, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name=excluded.name, short_name=excluded.short_name,
            country=excluded.country, ranking=excluded.ranking,
            ranking_points=excluded.ranking_points, updated_at=excluded.updated_at
    """, (
        player["id"], player["name"], player.get("short_name"),
        player.get("country"), player.get("ranking"), player.get("ranking_points"),
        dt.datetime.now().isoformat(timespec="seconds"),
    ))


def get_player(player_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM tennis_players WHERE id=?", (player_id,)).fetchone()
        return dict(row) if row else None


def search_player(name: str, limit: int = 10) -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_players WHERE LOWER(name) LIKE ? ORDER BY ranking LIMIT ?",
            (f"%{name.lower()}%", limit),
        ).fetchall()
        return [dict(r) for r in rows]


def list_players(tour: str = "atp", limit: int = 50) -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_players WHERE ranking IS NOT NULL ORDER BY ranking LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Matches ──────────────────────────────────────────────────────────────────

def upsert_match(conn, match: dict):
    """Insert or update a match with all statistics."""
    sql = """
        INSERT INTO tennis_matches (
            api_event_id, tournament_id, tournament_name, round_name, surface,
            date, player1_id, player2_id, player1_name, player2_name,
            winner_id, score, status, player1_elo, player2_elo,
            player1_ranking, player2_ranking, odds_player1, odds_player2,
            h2h_player1_wins, h2h_player2_wins, h2h_total,
            w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced,
            l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced,
            created_at
        ) VALUES (
            :api_event_id, :tournament_id, :tournament_name, :round_name, :surface,
            :date, :player1_id, :player2_id, :player1_name, :player2_name,
            :winner_id, :score, :status, :player1_elo, :player2_elo,
            :player1_ranking, :player2_ranking, :odds_player1, :odds_player2,
            :h2h_player1_wins, :h2h_player2_wins, :h2h_total,
            :w_ace, :w_df, :w_svpt, :w_1stIn, :w_1stWon, :w_2ndWon, :w_SvGms, :w_bpSaved, :w_bpFaced,
            :l_ace, :l_df, :l_svpt, :l_1stIn, :l_1stWon, :l_2ndWon, :l_SvGms, :l_bpSaved, :l_bpFaced,
            :created_at
        )
        ON CONFLICT(api_event_id) DO UPDATE SET
            winner_id=excluded.winner_id, score=excluded.score, status=excluded.status,
            w_ace=excluded.w_ace, w_df=excluded.w_df, w_svpt=excluded.w_svpt,
            w_1stIn=excluded.w_1stIn, w_1stWon=excluded.w_1stWon, w_2ndWon=excluded.w_2ndWon,
            w_SvGms=excluded.w_SvGms, w_bpSaved=excluded.w_bpSaved, w_bpFaced=excluded.w_bpFaced,
            l_ace=excluded.l_ace, l_df=excluded.l_df, l_svpt=excluded.l_svpt,
            l_1stIn=excluded.l_1stIn, l_1stWon=excluded.l_1stWon, l_2ndWon=excluded.l_2ndWon,
            l_SvGms=excluded.l_SvGms, l_bpSaved=excluded.l_bpSaved, l_bpFaced=excluded.l_bpFaced
    """
    params = dict(
        api_event_id=match.get("api_event_id"),
        tournament_id=match.get("tournament_id"),
        tournament_name=match.get("tournament_name"),
        round_name=match.get("round_name"),
        surface=match.get("surface"),
        date=match.get("date"),
        player1_id=match.get("player1_id"),
        player2_id=match.get("player2_id"),
        player1_name=match.get("player1_name"),
        player2_name=match.get("player2_name"),
        winner_id=match.get("winner_id"),
        score=match.get("score"),
        status=match.get("status", "scheduled"),
        player1_elo=match.get("player1_elo"),
        player2_elo=match.get("player2_elo"),
        player1_ranking=match.get("player1_ranking"),
        player2_ranking=match.get("player2_ranking"),
        odds_player1=match.get("odds_player1"),
        odds_player2=match.get("odds_player2"),
        h2h_player1_wins=match.get("h2h_player1_wins", 0),
        h2h_player2_wins=match.get("h2h_player2_wins", 0),
        h2h_total=match.get("h2h_total", 0),
        w_ace=match.get("w_ace"), w_df=match.get("w_df"), w_svpt=match.get("w_svpt"),
        w_1stIn=match.get("w_1stIn"), w_1stWon=match.get("w_1stWon"), w_2ndWon=match.get("w_2ndWon"),
        w_SvGms=match.get("w_SvGms"), w_bpSaved=match.get("w_bpSaved"), w_bpFaced=match.get("w_bpFaced"),
        l_ace=match.get("l_ace"), l_df=match.get("l_df"), l_svpt=match.get("l_svpt"),
        l_1stIn=match.get("l_1stIn"), l_1stWon=match.get("l_1stWon"), l_2ndWon=match.get("l_2ndWon"),
        l_SvGms=match.get("l_SvGms"), l_bpSaved=match.get("l_bpSaved"), l_bpFaced=match.get("l_bpFaced"),
        created_at=dt.datetime.now().isoformat(timespec="seconds"),
    )
    conn.execute(sql, params)


def upcoming_matches(limit: int = 20) -> list:
    today = dt.date.today().isoformat()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_matches WHERE date >= ? AND status='scheduled' "
            "ORDER BY date LIMIT ?",
            (today, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def live_matches() -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_matches WHERE status NOT IN ('scheduled', 'finished') "
            "ORDER BY date DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def recent_results(limit: int = 20) -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_matches WHERE status='finished' "
            "ORDER BY date DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


def player_matches(player_id: int, limit: int = 10) -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_matches WHERE (player1_id=? OR player2_id=?) "
            "AND status='finished' ORDER BY date DESC LIMIT ?",
            (player_id, player_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


def head_to_head(p1_id: int, p2_id: int, limit: int = 10) -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_matches WHERE "
            "((player1_id=? AND player2_id=?) OR (player1_id=? AND player2_id=?)) "
            "AND status='finished' ORDER BY date DESC LIMIT ?",
            (p1_id, p2_id, p2_id, p1_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Rankings ─────────────────────────────────────────────────────────────────

def save_rankings(conn, rankings: list, tour: str):
    now = dt.datetime.now().isoformat(timespec="seconds")
    for r in rankings:
        conn.execute(
            "INSERT INTO tennis_rankings (player_id, player_name, ranking, ranking_points, tour, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (r.get("player_id"), r.get("player_name"), r.get("ranking"),
             r.get("ranking_points"), tour, now),
        )


# ── Predictions ──────────────────────────────────────────────────────────────

def save_prediction(pred: dict) -> int:
    with connect() as conn:
        cur = conn.execute("""
            INSERT INTO tennis_predictions (
                player1_name, player2_name, tournament, match_date,
                analysis, main_bet, confidence, player1_win, player2_win,
                model_used, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pred["player1_name"], pred["player2_name"], pred.get("tournament"),
            pred.get("match_date"), pred.get("analysis"), pred.get("main_bet"),
            pred.get("confidence"), pred.get("player1_win"), pred.get("player2_win"),
            pred.get("model_used"),
            dt.datetime.now().isoformat(timespec="seconds"),
        ))
        return cur.lastrowid


def list_predictions(limit: int = 50) -> list:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_predictions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ── Training data ─────────────────────────────────────────────────────────────

def all_finished_matches() -> list:
    """Get all finished matches sorted by date (oldest first) for training."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM tennis_matches WHERE status='finished' "
            "ORDER BY date ASC"
        ).fetchall()
        return [dict(r) for r in rows]


# ── Stats ────────────────────────────────────────────────────────────────────

def db_stats() -> dict:
    with connect() as conn:
        players = conn.execute("SELECT COUNT(*) AS c FROM tennis_players").fetchone()["c"]
        matches = conn.execute("SELECT COUNT(*) AS c FROM tennis_matches").fetchone()["c"]
        finished = conn.execute("SELECT COUNT(*) AS c FROM tennis_matches WHERE status='finished'").fetchone()["c"]
        live = conn.execute("SELECT COUNT(*) AS c FROM tennis_matches WHERE status NOT IN ('scheduled','finished')").fetchone()["c"]
        return {
            "players": players,
            "matches": matches,
            "finished": finished,
            "live": live,
        }
