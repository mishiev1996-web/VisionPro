"""Quick audit script for sstats integration."""
import sqlite3

db = sqlite3.connect("data/football.db")
db.row_factory = sqlite3.Row

print("=== sstats_matches ===")
r = db.execute("SELECT COUNT(*) as c FROM sstats_matches").fetchone()
print(f"Total rows: {r['c']}")
r = db.execute("SELECT COUNT(DISTINCT game_id) as c FROM sstats_matches").fetchone()
print(f"Unique game_ids: {r['c']}")
r = db.execute("SELECT MIN(date), MAX(date) FROM sstats_matches").fetchone()
print(f"Date range: {r[0]} .. {r[1]}")
rows = db.execute("SELECT league_name, COUNT(*) as c FROM sstats_matches GROUP BY league_name ORDER BY c DESC LIMIT 20").fetchall()
print("Top leagues:")
for row in rows:
    print(f"  {row['league_name']}: {row['c']}")

print("\n=== sstats_odds ===")
r = db.execute("SELECT COUNT(*) as c FROM sstats_odds").fetchone()
print(f"Total rows: {r['c']}")
r = db.execute("SELECT COUNT(DISTINCT game_id) as c FROM sstats_odds").fetchone()
print(f"Unique game_ids: {r['c']}")
r = db.execute("SELECT COUNT(DISTINCT bookmaker) as c FROM sstats_odds").fetchone()
print(f"Unique bookmakers: {r['c']}")
r = db.execute("SELECT COUNT(DISTINCT market) as c FROM sstats_odds").fetchone()
print(f"Unique markets: {r['c']}")
rows = db.execute("SELECT market, COUNT(*) as c FROM sstats_odds GROUP BY market ORDER BY c DESC LIMIT 15").fetchall()
print("Top markets:")
for row in rows:
    print(f"  {row['market']}: {row['c']}")
rows = db.execute("SELECT bookmaker, COUNT(DISTINCT game_id) as games, COUNT(*) as odds FROM sstats_odds GROUP BY bookmaker ORDER BY games DESC LIMIT 10").fetchall()
print("Top bookmakers:")
for row in rows:
    print(f"  {row['bookmaker']}: {row['games']} games, {row['odds']} odds")

print("\n=== sstats_statistics ===")
r = db.execute("SELECT COUNT(*) as c FROM sstats_statistics").fetchone()
print(f"Total rows: {r['c']}")
r = db.execute("SELECT COUNT(DISTINCT game_id) as c FROM sstats_statistics").fetchone()
print(f"Unique game_ids: {r['c']}")
rows = db.execute("SELECT stat_name, COUNT(*) as c, COUNT(DISTINCT game_id) as games FROM sstats_statistics GROUP BY stat_name ORDER BY c DESC").fetchall()
print("Stats breakdown:")
for row in rows:
    print(f"  {row['stat_name']}: {row['c']} records, {row['games']} games")

print("\n=== sstats_events ===")
r = db.execute("SELECT COUNT(*) as c FROM sstats_events").fetchone()
print(f"Total rows: {r['c']}")
r = db.execute("SELECT COUNT(DISTINCT game_id) as c FROM sstats_events").fetchone()
print(f"Unique game_ids: {r['c']}")
rows = db.execute("SELECT event_type, COUNT(*) as c, COUNT(DISTINCT game_id) as games FROM sstats_events GROUP BY event_type ORDER BY c DESC").fetchall()
print("Event types:")
for row in rows:
    print(f"  type={row['event_type']}: {row['c']} events, {row['games']} games")

print("\n=== match_odds sources ===")
rows = db.execute("SELECT source, COUNT(*) as c FROM match_odds GROUP BY source ORDER BY c DESC").fetchall()
for row in rows:
    print(f"  {row['source']}: {row['c']}")

print("\n=== data_log (sstats) ===")
rows = db.execute("SELECT source, status, started_at, rows_added, errors FROM data_log WHERE source LIKE '%sstats%' ORDER BY started_at DESC LIMIT 10").fetchall()
for row in rows:
    print(f"  {row['source']} | {row['status']} | {row['started_at']} | added={row['rows_added']} err={row['errors']}")

db.close()
