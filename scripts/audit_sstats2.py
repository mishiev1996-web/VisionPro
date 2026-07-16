"""Audit sstats data quality and hidden bugs."""
import sqlite3

db = sqlite3.connect("data/football.db")
db.row_factory = sqlite3.Row

# 1. event_type values
print("=== event_type values in sstats_events ===")
rows = db.execute("SELECT event_type, COUNT(*) as c FROM sstats_events GROUP BY event_type ORDER BY c DESC LIMIT 20").fetchall()
for r in rows:
    print(f'  "{r["event_type"]}": {r["c"]}')

# 2. Sample events
print("\n=== Sample events ===")
rows = db.execute("SELECT * FROM sstats_events LIMIT 10").fetchall()
for r in rows:
    print(f'  game_id={r["game_id"]}, minute={r["minute"]}, event_type="{r["event_type"]}", player={r["player"]}, team={r["team"]}')

# 3. xG stat values
print("\n=== xG stat values ===")
rows = db.execute("SELECT stat_name, home_value, away_value FROM sstats_statistics WHERE stat_name='xg' LIMIT 5").fetchall()
for r in rows:
    print(f'  home="{r["home_value"]}", away="{r["away_value"]}"')

# 4. All stat_name values
print("\n=== All stat_name values ===")
rows = db.execute("SELECT DISTINCT stat_name FROM sstats_statistics ORDER BY stat_name").fetchall()
for r in rows:
    print(f'  {r["stat_name"]}')

# 5. raw_json format
print("\n=== raw_json format ===")
rows = db.execute("SELECT game_id, substr(raw_json, 1, 100) as rj FROM sstats_matches LIMIT 3").fetchall()
for r in rows:
    print(f'  game_id={r["game_id"]}: {r["rj"]}')

# 6. sstats_consensus odds
print("\n=== sstats_consensus odds ===")
rows = db.execute("SELECT * FROM match_odds WHERE source='sstats_consensus' LIMIT 5").fetchall()
for r in rows:
    print(f'  match_id={r["match_id"]}: odds={r["home_odds"]}/{r["draw_odds"]}/{r["away_odds"]}, implied={r["implied_h"]}/{r["implied_d"]}/{r["implied_a"]}')

# 7. Non-sstats odds
print("\n=== Non-sstats odds ===")
rows = db.execute("SELECT * FROM match_odds WHERE source != 'sstats_consensus' LIMIT 3").fetchall()
for r in rows:
    print(f'  match_id={r["match_id"]}: source={r["source"]}, odds={r["home_odds"]}/{r["draw_odds"]}/{r["away_odds"]}, implied={r["implied_h"]}/{r["implied_d"]}/{r["implied_a"]}')

# 8. How many sstats_matches have home_id/away_id = 0 or NULL?
print("\n=== sstats_matches with missing team IDs ===")
r = db.execute("SELECT COUNT(*) as c FROM sstats_matches WHERE home_id IS NULL OR away_id IS NULL OR home_id=0 OR away_id=0").fetchone()
print(f'  Missing team IDs: {r["c"]}')

# 9. Check sstats_odds: what markets are available beyond Match Winner?
print("\n=== sstats_odds market types (non-Match Winner) ===")
rows = db.execute("""
    SELECT market, COUNT(*) as c, COUNT(DISTINCT game_id) as games 
    FROM sstats_odds 
    WHERE market != 'Match Winner' 
    GROUP BY market 
    ORDER BY c DESC 
    LIMIT 15
""").fetchall()
for r in rows:
    print(f'  {r["market"]}: {r["c"]} odds, {r["games"]} games')

# 10. Check if sstats data links to main matches
print("\n=== sstats_matches linked to main matches ===")
r = db.execute("""
    SELECT COUNT(DISTINCT sm.game_id) as linked
    FROM sstats_matches sm
    WHERE sm.home_id IN (SELECT id FROM teams) AND sm.away_id IN (SELECT id FROM teams)
""").fetchone()
print(f'  sstats_matches with valid team IDs: {r["linked"]}')

db.close()
