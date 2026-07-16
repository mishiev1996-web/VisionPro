"""Count sstats_matches without odds, grouped by league, to estimate collection target."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

with db.connect() as conn:
    # Matches with odds
    with_odds = conn.execute("SELECT COUNT(DISTINCT game_id) FROM sstats_odds").fetchone()[0]
    
    # Matches without odds
    without_odds = conn.execute("""
        SELECT COUNT(DISTINCT m.game_id)
        FROM sstats_matches m
        WHERE m.game_id NOT IN (SELECT DISTINCT game_id FROM sstats_odds)
    """).fetchone()[0]
    
    print(f"With odds: {with_odds}")
    print(f"Without odds: {without_odds}")
    print(f"Total sstats_matches: {with_odds + without_odds}")
    
    # Per-league breakdown of matches without odds
    print("\n--- Matches WITHOUT odds by league ---")
    rows = conn.execute("""
        SELECT m.league_name, COUNT(DISTINCT m.game_id) as cnt
        FROM sstats_matches m
        WHERE m.game_id NOT IN (SELECT DISTINCT game_id FROM sstats_odds)
        GROUP BY m.league_name
        ORDER BY cnt DESC
    """).fetchall()
    total_without = 0
    for r in rows:
        print(f"  {r['league_name']:30s} {r['cnt']:5d}")
        total_without += r['cnt']
    print(f"  {'TOTAL':30s} {total_without:5d}")
    
    # Also check: how many matches have status=8 (finished) without odds
    print("\n--- Finished matches without odds (status filter) ---")
    rows = conn.execute("""
        SELECT m.league_name, COUNT(DISTINCT m.game_id) as cnt
        FROM sstats_matches m
        WHERE m.game_id NOT IN (SELECT DISTINCT game_id FROM sstats_odds)
          AND m.status = 'Finished'
        GROUP BY m.league_name
        ORDER BY cnt DESC
    """).fetchall()
    total_finished = 0
    for r in rows:
        print(f"  {r['league_name']:30s} {r['cnt']:5d}")
        total_finished += r['cnt']
    print(f"  {'TOTAL':30s} {total_finished:5d}")
    
    # Check what statuses exist
    print("\n--- Status distribution (all sstats_matches) ---")
    rows = conn.execute("""
        SELECT status, COUNT(*) as cnt
        FROM sstats_matches
        GROUP BY status
        ORDER BY cnt DESC
    """).fetchall()
    for r in rows:
        print(f"  {r['status']:20s} {r['cnt']:5d}")
