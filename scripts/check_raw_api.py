"""Check raw API response for a game - full structure."""
import sys
sys.path.insert(0, ".")
from scrapers import sstats
import json

# Pick a game from a league that likely has events
games = sstats.fetch_games_by_date("2026-07-13")  # Yesterday - more likely finished
finished = [g for g in games if g.get("status") == 8][:5]
print(f"Yesterday's finished games: {len(finished)}")

for g in finished[:3]:
    gid = g["id"]
    home = g.get("homeTeam", {}).get("name", "?")
    away = g.get("awayTeam", {}).get("name", "?")
    print(f"\nGame {gid}: {home} vs {away}")
    
    detail = sstats.fetch_game(gid)
    if detail:
        # Print ALL top-level keys
        data = detail.get("data", detail)
        if isinstance(data, dict):
            print(f"  Keys in 'data': {list(data.keys())}")
            game_obj = data.get("game", data)
            if isinstance(game_obj, dict):
                print(f"  Keys in 'game': {list(game_obj.keys())}")
                events = game_obj.get("events")
                print(f"  events field: {type(events)} = {str(events)[:200] if events else 'None'}")
