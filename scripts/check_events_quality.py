"""Check events data quality in DB."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

with db.connect() as conn:
    r = conn.execute("SELECT COUNT(*) FROM sstats_events").fetchone()
    print(f"Total events: {r[0]}")
    r2 = conn.execute("SELECT COUNT(DISTINCT game_id) FROM sstats_events").fetchone()
    print(f"Unique games: {r2[0]}")
    r3 = conn.execute("SELECT COUNT(*) FROM sstats_events WHERE minute IS NULL").fetchone()
    print(f"Events with minute=None: {r3[0]}")
    r4 = conn.execute("SELECT COUNT(*) FROM sstats_events WHERE minute IS NOT NULL").fetchone()
    print(f"Events with minute set: {r4[0]}")
    
    # Sample events with minute
    rows = conn.execute("SELECT game_id, minute, event_type, player FROM sstats_events WHERE minute IS NOT NULL LIMIT 5").fetchall()
    print(f"\nSample events WITH minute:")
    for r in rows:
        print(f"  game_id={r['game_id']}, minute={r['minute']}, type={r['event_type']}, player={r['player']}")
    
    # Sample events without minute
    rows = conn.execute("SELECT game_id, minute, event_type, player FROM sstats_events WHERE minute IS NULL LIMIT 5").fetchall()
    print(f"\nSample events WITHOUT minute:")
    for r in rows:
        print(f"  game_id={r['game_id']}, minute={r['minute']}, type={r['event_type']}, player={r['player']}")
