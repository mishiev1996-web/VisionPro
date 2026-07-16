"""Investigate minute=None in sstats_events."""
import sys
sys.path.insert(0, ".")
from scrapers import sstats
import json
import sqlite3

db = sqlite3.connect("data/football.db")
db.row_factory = sqlite3.Row

# Find a game with goal events
rows = db.execute("SELECT DISTINCT game_id FROM sstats_events WHERE event_type='3' LIMIT 5").fetchall()
game_ids = [r["game_id"] for r in rows]
print(f"Games with goals in DB: {game_ids}")

# Pick first one
if game_ids:
    gid = game_ids[0]
    print(f"\nFetching game {gid} from API...")
    detail = sstats.fetch_game(gid)
    if detail:
        # Check if detail has nested 'game' key
        game = detail.get("game", detail)
        events = game.get("events", [])
        print(f"Events count: {len(events)}")
        if events:
            print("\nFirst 3 events (FULL JSON):")
            for i, ev in enumerate(events[:3]):
                print(f"\n--- Event {i} ---")
                print(json.dumps(ev, indent=2, default=str))
        else:
            print("No events found in API response!")
            print("Top-level keys:", list(game.keys())[:20])
    else:
        print("API returned None")

# Also check: what's stored in DB for this game?
if game_ids:
    gid = game_ids[0]
    print(f"\n\n=== What's stored in DB for game {gid} ===")
    db_events = db.execute("SELECT * FROM sstats_events WHERE game_id=?", (gid,)).fetchall()
    for ev in db_events:
        print(f"  minute={ev['minute']}, type={ev['event_type']}, player={ev['player']}, team={ev['team']}")

db.close()
