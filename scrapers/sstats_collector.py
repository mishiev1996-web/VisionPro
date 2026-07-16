"""
scrapers/sstats_collector.py — Collect all data from sstats.net API → SQLite.

Free endpoints (no key required):
    /Leagues              — all leagues worldwide
    /Teams/list           — teams per league
    /Games/list           — match history per league
    /Odds/{id}            — bookmaker odds per match
    /Games/season-table   — league standings

Usage:
    python -m scrapers.sstats_collector --full        # full sync
    python -m scrapers.sstats_collector --leagues     # leagues only
    python -m scrapers.sstats_collector --league 195  # one league
    python -m scrapers.sstats_collector --today       # today's matches
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from typing import List, Optional, Dict, Any

import db
from scrapers import sstats


# ── League sync ──────────────────────────────────────────────────────────────

def sync_leagues(progress_cb=None) -> int:
    """Fetch all leagues from sstats and save to DB. Returns count."""
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    _emit("  Fetching leagues from sstats...")
    leagues = sstats.fetch_leagues()
    _emit(f"  Found {len(leagues)} leagues")

    saved = 0
    with db.connect() as conn:
        for lg in leagues:
            league_id = lg.get("id")
            name = lg.get("name", "")
            country_obj = lg.get("country") or {}
            country = country_obj.get("name", "") if isinstance(country_obj, dict) else str(country_obj)

            if not league_id or not name:
                continue

            # Create a slug from league ID
            slug = f"sstats_{league_id}"

            # Map to our tier system
            tier = _guess_tier(name, country)

            db.upsert_league(conn, slug, name, country, tier=tier)
            saved += 1

        conn.commit()

    _emit(f"  Saved {saved} leagues to DB")
    return saved


def _guess_tier(name: str, country: str) -> int:
    """Guess league tier based on name/country."""
    name_lower = name.lower()
    country_lower = country.lower()

    # Tier 1: Top 5 leagues + Champions League
    tier1_keywords = ["premier league", "la liga", "bundesliga", "serie a", "ligue 1",
                       "champions league", "europa league"]
    if any(k in name_lower for k in tier1_keywords):
        return 1

    # Tier 2: Well-known European leagues
    tier2_keywords = ["eredivisie", "primeira liga", "super lig", "championship",
                       "belgian", "greek", "russian", "ukrainian", "czech", "austrian",
                       "swiss", "danish", "norwegian", "swedish"]
    if any(k in name_lower for k in tier2_keywords):
        return 2

    # Tier 3: Everything else
    return 3


# ── Team sync ────────────────────────────────────────────────────────────────

def sync_teams_for_league(league_id: int, league_slug: str, progress_cb=None) -> int:
    """Fetch teams for a league and save to DB. Returns count."""
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    teams = sstats.fetch_teams_in_league(league_id)
    if not teams:
        return 0

    saved = 0
    with db.connect() as conn:
        for t in teams:
            team_id = t.get("id")
            name = t.get("name", "")
            if not team_id or not name:
                continue

            db.upsert_team(conn, team_id, name, None, league_slug)
            saved += 1

        conn.commit()

    _emit(f"    Teams: {saved}")
    return saved


# ── Match sync ───────────────────────────────────────────────────────────────

def sync_matches_for_league(league_id: int, league_slug: str,
                            pages: int = 5, progress_cb=None) -> int:
    """Fetch match history for a league (up to pages * 1000 matches)."""
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    total_saved = 0
    now = dt.datetime.now().isoformat()

    with db.connect() as conn:
        for page in range(pages):
            games = sstats.fetch_games_by_league(league_id, page=page)
            if not games:
                break

            for g in games:
                game_id = g.get("id")
                if not game_id:
                    continue

                # Save to sstats_matches table
                db.save_sstats_match(conn, g, now)
                total_saved += 1

                # Also save to main matches table if it has a result
                home = g.get("homeTeam") or {}
                away = g.get("awayTeam") or {}
                home_name = home.get("name", "")
                away_name = away.get("name", "")
                home_goals = g.get("homeResult")
                away_goals = g.get("awayResult")
                date = g.get("date", "")[:10]

                if date and home_goals is not None and away_goals is not None:
                    # Ensure teams exist
                    for name in [home_name, away_name]:
                        row = conn.execute("SELECT id FROM teams WHERE name=?", (name,)).fetchone()
                        if not row:
                            db.upsert_team(conn, hash(name) % (2**31 - 1), name, None, league_slug)

                    h_row = conn.execute("SELECT id FROM teams WHERE name=?", (home_name,)).fetchone()
                    a_row = conn.execute("SELECT id FROM teams WHERE name=?", (away_name,)).fetchone()

                    if h_row and a_row:
                        season = int(date[:4]) if date[:4].isdigit() else 2024
                        mid = hash(f"{date}_{home_name}_{away_name}") % (2**31 - 1)
                        conn.execute(
                            "INSERT OR IGNORE INTO matches(id,league_slug,season,date,"
                            "home_id,away_id,home_goals,away_goals,is_result) "
                            "VALUES (?,?,?,?,?,?,?,?,1)",
                            (mid, league_slug, season, date,
                             h_row[0], a_row[0], home_goals, away_goals)
                        )

            conn.commit()
            _emit(f"    Page {page + 1}: {len(games)} games")

            # Rate limiting
            time.sleep(0.5)

    _emit(f"  Total saved: {total_saved}")
    return total_saved


# ── Odds sync ────────────────────────────────────────────────────────────────

def sync_odds_for_recent(league_id: int, days: int = 7, limit: int = 100,
                         progress_cb=None) -> int:
    """Fetch odds for recent matches in a league."""
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    # Get recent finished matches
    games = sstats.fetch_games_by_league(league_id)
    if not games:
        return 0

    # Filter to finished matches from last N days
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()
    recent = [g for g in games
              if g.get("statusName") == "Finished"
              and g.get("date", "")[:10] >= cutoff][:limit]

    _emit(f"  Fetching odds for {len(recent)} recent matches...")

    saved = 0
    now = dt.datetime.now().isoformat()

    with db.connect() as conn:
        for g in recent:
            game_id = g.get("id")
            if not game_id:
                continue

            odds = sstats.fetch_odds(game_id)
            if odds:
                count = db.save_sstats_odds(conn, game_id, odds, now)
                saved += count

            time.sleep(2)  # Rate limit for odds endpoint

    _emit(f"  Saved {saved} odds records")
    return saved


# ── Game details sync (statistics + events) ──────────────────────────────────

def sync_game_details(game_ids: List[int], progress_cb=None) -> Dict[str, int]:
    """Fetch detailed statistics and events for specific games.

    Calls /Games/{id} which returns:
    - statistics: dict with 80+ metrics (shots, corners, possession, xG, etc.)
    - events: list of goals, cards, substitutions with minute timestamps

    Returns dict with counts: {"statistics": N, "events": N}
    """
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    _emit(f"  Fetching details for {len(game_ids)} games...")
    stats_count = 0
    events_count = 0
    now = dt.datetime.now().isoformat()

    with db.connect() as conn:
        for i, game_id in enumerate(game_ids):
            if i % 50 == 0 and i > 0:
                _emit(f"    Progress: {i}/{len(game_ids)}")

            detail = sstats.fetch_game(game_id)
            if not detail:
                continue

            # Save statistics
            statistics = detail.get("statistics")
            if statistics and isinstance(statistics, dict):
                count = db.save_sstats_statistics(conn, game_id, statistics, now)
                stats_count += count

            # Save events
            events = detail.get("events")
            if events and isinstance(events, list):
                count = db.save_sstats_events(conn, game_id, events, now)
                events_count += count

            # Rate limit: 2 seconds between requests
            time.sleep(2)

        conn.commit()

    _emit(f"  Saved: {stats_count} statistics, {events_count} events")
    return {"statistics": stats_count, "events": events_count}


def sync_details_for_league(league_id: int, limit: int = 100,
                            progress_cb=None) -> Dict[str, int]:
    """Fetch details for recent finished matches in a league."""
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    _emit(f"  Getting game IDs for league {league_id}...")

    # Get finished games from this league
    games = sstats.fetch_games_by_league(league_id)
    if not games:
        return {"statistics": 0, "events": 0}

    # Filter to finished games, take most recent
    finished = [g for g in games if g.get("statusName") == "Finished"]
    game_ids = [g["id"] for g in finished[:limit] if g.get("id")]

    _emit(f"  Found {len(game_ids)} finished games to fetch details for")

    return sync_game_details(game_ids, progress_cb)


def sync_details_for_recent(days: int = 30, limit: int = 200,
                            progress_cb=None) -> Dict[str, int]:
    """Fetch details for recent worldwide finished matches."""
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    _emit("  Fetching recent worldwide matches...")
    today = dt.date.today().isoformat()
    cutoff = (dt.date.today() - dt.timedelta(days=days)).isoformat()

    # Get matches from recent dates
    game_ids = []
    current = dt.date.today()
    while current.isoformat() >= cutoff and len(game_ids) < limit:
        games = sstats.fetch_games_by_date(current.isoformat())
        for g in (games or []):
            if g.get("statusName") == "Finished" and g.get("id"):
                game_ids.append(g["id"])
        current -= dt.timedelta(days=1)
        time.sleep(0.5)

    game_ids = game_ids[:limit]
    _emit(f"  Found {len(game_ids)} finished games from last {days} days")

    return sync_game_details(game_ids, progress_cb)


# ── Full sync ────────────────────────────────────────────────────────────────

def full_sync(league_ids: Optional[List[int]] = None,
              max_leagues: int = 50,
              pages_per_league: int = 3,
              details: bool = False,
              progress_cb=None) -> Dict[str, int]:
    """Full sync: leagues + teams + matches + details for all (or selected) leagues.

    Returns summary dict with counts.
    """
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    _emit("=== SStats Full Sync ===")
    start = time.time()

    # 1. Sync leagues
    _emit("\n[1/4] Syncing leagues...")
    league_count = sync_leagues(progress_cb)

    # 2. Get league list
    if league_ids:
        target_leagues = [(lid, f"sstats_{lid}") for lid in league_ids]
    else:
        # Use top leagues by tier
        with db.connect() as conn:
            rows = conn.execute(
                "SELECT slug FROM leagues WHERE slug LIKE 'sstats_%' "
                "ORDER BY source_tier ASC LIMIT ?",
                (max_leagues,)
            ).fetchall()
            target_leagues = [(int(r["slug"].replace("sstats_", "")), r["slug"])
                              for r in rows]

    _emit(f"\n[2/4] Syncing teams + matches for {len(target_leagues)} leagues...")

    stats = {"leagues": league_count, "teams": 0, "matches": 0, "odds": 0,
             "statistics": 0, "events": 0}

    for i, (league_id, league_slug) in enumerate(target_leagues):
        _emit(f"\n  [{i + 1}/{len(target_leagues)}] League {league_id}...")

        # Teams
        team_count = sync_teams_for_league(league_id, league_slug, progress_cb)
        stats["teams"] += team_count

        # Matches
        match_count = sync_matches_for_league(league_id, league_slug,
                                               pages=pages_per_league,
                                               progress_cb=progress_cb)
        stats["matches"] += match_count

    # 3. Sync odds for top leagues only (to avoid rate limits)
    _emit(f"\n[3/4] Syncing odds for top 10 leagues...")
    top_leagues = target_leagues[:10]
    for league_id, league_slug in top_leagues:
        odds_count = sync_odds_for_recent(league_id, days=7, limit=20, progress_cb=progress_cb)
        stats["odds"] += odds_count

    # 4. Sync game details (statistics + events) if requested
    if details:
        _emit(f"\n[4/4] Syncing game details for top 5 leagues...")
        detail_leagues = target_leagues[:5]
        for league_id, league_slug in detail_leagues:
            result = sync_details_for_league(league_id, limit=50, progress_cb=progress_cb)
            stats["statistics"] += result.get("statistics", 0)
            stats["events"] += result.get("events", 0)
    else:
        _emit(f"\n[4/4] Skipping game details (use --details to enable)")

    elapsed = time.time() - start
    _emit(f"\n=== Sync complete in {elapsed:.0f}s ===")
    _emit(f"  Leagues:    {stats['leagues']}")
    _emit(f"  Teams:      {stats['teams']}")
    _emit(f"  Matches:    {stats['matches']}")
    _emit(f"  Odds:       {stats['odds']}")
    _emit(f"  Statistics: {stats['statistics']}")
    _emit(f"  Events:     {stats['events']}")

    return stats


# ── Single league sync ──────────────────────────────────────────────────────

def sync_league(league_id: int, pages: int = 5, odds: bool = True,
                progress_cb=None) -> Dict[str, int]:
    """Sync one league: teams + matches + odds."""
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    slug = f"sstats_{league_id}"
    stats = {"teams": 0, "matches": 0, "odds": 0}

    _emit(f"Syncing league {league_id}...")

    # Teams
    stats["teams"] = sync_teams_for_league(league_id, slug, progress_cb)

    # Matches
    stats["matches"] = sync_matches_for_league(league_id, slug, pages, progress_cb)

    # Odds
    if odds:
        stats["odds"] = sync_odds_for_recent(league_id, days=30, limit=50, progress_cb=progress_cb)

    _emit(f"Done: {stats}")
    return stats


# ── Today's matches ─────────────────────────────────────────────────────────

def sync_today(progress_cb=None) -> int:
    """Fetch and save today's worldwide matches."""
    def _emit(msg):
        if progress_cb:
            progress_cb({"type": "info", "msg": msg})
        else:
            print(msg)

    today = dt.date.today().isoformat()
    _emit(f"Fetching matches for {today}...")
    games = sstats.fetch_games_by_date(today)
    _emit(f"Found {len(games)} matches")

    saved = 0
    now = dt.datetime.now().isoformat()

    with db.connect() as conn:
        for g in games:
            game_id = g.get("id")
            if not game_id:
                continue
            db.save_sstats_match(conn, g, now)
            saved += 1
        conn.commit()

    _emit(f"Saved {saved} matches")
    return saved


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SStats data collector")
    parser.add_argument("--full", action="store_true", help="Full sync (leagues + teams + matches)")
    parser.add_argument("--leagues", action="store_true", help="Sync leagues only")
    parser.add_argument("--league", type=int, help="Sync one league by ID")
    parser.add_argument("--today", action="store_true", help="Sync today's matches")
    parser.add_argument("--details", action="store_true", help="Fetch game details (statistics + events)")
    parser.add_argument("--detail-league", type=int, help="Fetch details for one league")
    parser.add_argument("--detail-recent", type=int, metavar="DAYS", help="Fetch details for recent N days")
    parser.add_argument("--max-leagues", type=int, default=50, help="Max leagues for full sync")
    parser.add_argument("--pages", type=int, default=3, help="Pages per league")
    args = parser.parse_args()

    if args.full:
        full_sync(max_leagues=args.max_leagues, pages_per_league=args.pages, details=args.details)
    elif args.leagues:
        sync_leagues()
    elif args.league:
        sync_league(args.league, pages=args.pages)
    elif args.detail_league:
        sync_details_for_league(args.detail_league, limit=100)
    elif args.detail_recent:
        sync_details_for_recent(days=args.detail_recent, limit=200)
    elif args.today:
        sync_today()
    else:
        parser.print_help()
