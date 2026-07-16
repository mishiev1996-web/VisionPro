"""Test: save injuries for one game_id, then verify in DB."""
import sys
sys.path.insert(0, ".")
from scrapers import sstats
import db

# Pick a game that likely has injuries (from a major league)
import sqlite3
conn_db = sqlite3.connect("data/football.db")
conn_db.row_factory = sqlite3.Row
row = conn_db.execute("""
    SELECT game_id FROM sstats_matches 
    WHERE league_name IN ('Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1')
    ORDER BY collected_at DESC LIMIT 1
""").fetchone()
conn_db.close()

if row:
    gid = row["game_id"]
    print(f"Testing injuries for game_id={gid}...")
    
    # Fetch injuries from API
    injuries = sstats.fetch_injuries(gid)
    print(f"API returned {len(injuries) if injuries else 0} injuries")
    if injuries:
        print(f"Sample: {injuries[0]}")
    
    # Save to DB
    import datetime as dt
    now = dt.datetime.now().isoformat(timespec="seconds")
    with db.connect() as conn:
        n = db.save_sstats_injuries(conn, gid, injuries, now)
        print(f"Saved {n} injuries to DB")
    
    # Verify
    conn_db = sqlite3.connect("data/football.db")
    r = conn_db.execute("SELECT COUNT(*) as c FROM sstats_injuries").fetchone()
    print(f"Total sstats_injuries rows: {r[0]}")
    rows = conn_db.execute("SELECT * FROM sstats_injuries LIMIT 5").fetchall()
    for r in rows:
        print(f"  game_id={r[1]}, player={r[2]}, team_id={r[3]}, reason={r[4]}")
    conn_db.close()
else:
    print("No game found")
