"""Try fetching events for a live or upcoming game."""
import sys
sys.path.insert(0, ".")
from scrapers import sstats
import json

# Try fetching a game that's currently live (status 3 = in progress)
# First check what live games exist
games = sstats.fetch_live_matches()
print(f"Live games: {len(games)}")
if games:
    gid = games[0]["id"]
    home = games[0].get("homeTeam", {}).get("name", "?")
    away = games[0].get("awayTeam", {}).get("name", "?")
    print(f"Fetching live game {gid}: {home} vs {away}")
    detail = sstats.fetch_game(gid)
    if detail:
        game = detail.get("game", detail)
        events = game.get("events", [])
        print(f"Events: {len(events)}")
        if events:
            for ev in events[:3]:
                print(json.dumps(ev, indent=2, default=str))
else:
    print("No live games. Trying a game from DB that might have events...")
    # Check a game from our bulk collection period
    import sqlite3
    db = sqlite3.connect("data/football.db")
    db.row_factory = sqlite3.Row
    # Try game_ids that were from sstats_bulk collection
    rows = db.execute(
        "SELECT game_id FROM sstats_events WHERE minute IS NOT NULL LIMIT 3"
    ).fetchall()
    db.close()
    if rows:
        print(f"Games with non-null minutes in DB: {[r['game_id'] for r in rows]}")
    else:
        print("No games with non-null minutes found in DB")
