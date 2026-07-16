"""Check if any game returns events from API, and what fields they have."""
import sys
sys.path.insert(0, ".")
from scrapers import sstats
import json

# Try a recent finished game from today's list
import datetime
today = datetime.date.today().isoformat()
games = sstats.fetch_games_by_date(today)
finished = [g for g in games if g.get("status") == 8]
print(f"Finished games today: {len(finished)}")

# Pick a game with goals
for g in finished[:5]:
    gid = g["id"]
    home = g.get("homeTeam", {}).get("name", "?")
    away = g.get("awayTeam", {}).get("name", "?")
    hr = g.get("homeResult", "?")
    ar = g.get("awayResult", "?")
    print(f"\nGame {gid}: {home} {hr}-{ar} {away}")
    
    detail = sstats.fetch_game(gid)
    if detail:
        game = detail.get("game", detail)
        events = game.get("events", [])
        print(f"  Events: {len(events)}")
        if events:
            print(f"  First event FULL:")
            print(json.dumps(events[0], indent=2, default=str))
            # Check if 'elapsed' exists
            has_elapsed = any("elapsed" in str(ev) for ev in events[:3])
            has_minute = any("minute" in str(ev) for ev in events[:3])
            print(f"  Has 'elapsed' field: {has_elapsed}")
            print(f"  Has 'minute' field: {has_minute}")
            break
    else:
        print("  API returned None")
