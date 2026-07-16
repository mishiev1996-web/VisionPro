"""Check match_odds coverage after sstats expansion."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

with db.connect() as conn:
    r = conn.execute("SELECT COUNT(*) FROM match_odds").fetchone()
    print(f"Total match_odds rows: {r[0]}")
    
    rows = conn.execute("""
        SELECT source, COUNT(*) as cnt, COUNT(DISTINCT match_id) as matches
        FROM match_odds
        GROUP BY source
        ORDER BY cnt DESC
    """).fetchall()
    print("\nPer-source breakdown:")
    for r in rows:
        print(f"  {r['source']:25s} {r['matches']:6d} matches, {r['cnt']:6d} rows")
    
    # Check if sstats_consensus odds are actually used in training
    r = conn.execute("""
        SELECT COUNT(DISTINCT m.id) 
        FROM matches m
        JOIN match_odds mo ON m.id = mo.match_id
        WHERE m.is_result = 1
    """).fetchone()
    print(f"\nFinished matches with ANY odds: {r[0]}")
    
    r2 = conn.execute("""
        SELECT COUNT(DISTINCT m.id)
        FROM matches m
        JOIN match_odds mo ON m.id = mo.match_id
        WHERE m.is_result = 1 AND mo.source = 'sstats_consensus'
    """).fetchone()
    print(f"Finished matches with sstats_consensus odds: {r2[0]}")
    
    r3 = conn.execute("""
        SELECT COUNT(DISTINCT m.id)
        FROM matches m
        WHERE m.is_result = 1
    """).fetchone()
    print(f"Total finished matches: {r3[0]}")
    print(f"\nOdds coverage: {r[0]}/{r3[0]} = {r[0]/r3[0]*100:.1f}%")
