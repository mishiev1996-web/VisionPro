"""Check actual events structure from API."""
import sys
sys.path.insert(0, ".")
from scrapers import sstats
import json

# Use a game from a major league that should have events
# EPL league_id=39, let's find a recent finished game
import sqlite3
db = sqlite3.connect("data/football.db")
db.row_factory = sqlite3.Row

# Find a game from EPL with events
rows = db.execute("""
    SELECT sm.game_id, sm.home_team, sm.away_team
    FROM sstats_matches sm
    WHERE sm.league_name LIKE '%Premier%'
    AND sm.game_id IN (SELECT game_id FROM sstats_events WHERE event_type='3')
    LIMIT 3
""").fetchall()
print("EPL games with goals:", [(r["game_id"], r["home_team"], r["away_team"]) for r in rows])

db.close()

# If none, try any game with events from the bulk collection
if not rows:
    import sqlite3
    db = sqlite3.connect("data/football.db")
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT sm.game_id, sm.home_team, sm.away_team
        FROM sstats_matches sm
        WHERE sm.game_id IN (SELECT game_id FROM sstats_events WHERE event_type='3')
        ORDER BY sm.collected_at DESC
        LIMIT 3
    """).fetchall()
    print("Any games with goals:", [(r["game_id"], r["home_team"], r["away_team"]) for r in rows])
    db.close()

if rows:
    gid = rows[0]["game_id"]
    print(f"\nFetching game {gid}...")
    raw = sstats._fetch_one(f"/Games/{gid}")
    if raw:
        data = raw.get("data", raw)
        events = data.get("events", [])
        print(f"Events at data level: {len(events) if events else 0}")
        if events:
            for ev in events[:2]:
                print(json.dumps(ev, indent=2, default=str)[:800])
        else:
            print("Events is None/empty")
            # Check statistics
            stats = data.get("statistics", {})
            print(f"Statistics type: {type(stats)}")
            if isinstance(stats, dict):
                print(f"Statistics keys: {list(stats.keys())[:10]}")
    else:
        print("API returned None")
