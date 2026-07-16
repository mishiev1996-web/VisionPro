"""Check events from a specific game that HAS events in DB."""
import sys
sys.path.insert(0, ".")
from scrapers import sstats
import json
import sqlite3

db = sqlite3.connect("data/football.db")
db.row_factory = sqlite3.Row

# Find a game with goal events and check its API response
rows = db.execute(
    "SELECT game_id, COUNT(*) as n FROM sstats_events "
    "WHERE event_type='3' GROUP BY game_id ORDER BY n DESC LIMIT 3"
).fetchall()
print("Games with most goals in DB:")
for r in rows:
    print(f"  game_id={r['game_id']}: {r['n']} goals")

if rows:
    gid = rows[0]["game_id"]
    print(f"\nFetching game {gid} from API...")
    detail = sstats.fetch_game(gid)
    if detail:
        game = detail.get("game", detail)
        events = game.get("events", [])
        print(f"Events from API: {len(events)}")
        if events:
            for ev in events[:3]:
                print(json.dumps(ev, indent=2, default=str))
        else:
            print("No events from API. Top keys:", list(game.keys()))
    else:
        print("API returned None")

db.close()
