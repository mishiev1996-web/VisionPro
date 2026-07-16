"""Quantify exactly which sources were overwritten."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

with db.connect() as conn:
    # Current state
    rows = conn.execute("""
        SELECT source, COUNT(*) as cnt, COUNT(DISTINCT match_id) as matches
        FROM match_odds GROUP BY source ORDER BY cnt DESC
    """).fetchall()
    print("=== CURRENT match_odds per source ===")
    for r in rows:
        print(f"  {r['source']:25s} {r['matches']:6d} matches ({r['cnt']:6d} rows)")

    print(f"\n=== BEFORE vs AFTER ===")
    print(f"BEFORE backfill (from backfill log output):")
    print(f"  historical_odds:    88135 matches")
    print(f"  pinnacle_close:     12385 matches")
    print(f"  bet365:               776 matches")
    print(f"  sstats_consensus:    240 matches")
    print(f"  TOTAL:             101536 rows")
    print()
    print(f"AFTER backfill:")
    current = {}
    for r in rows:
        current[r['source']] = r['matches']
    for src in ['historical_odds', 'pinnacle_close', 'bet365', 'sstats_consensus']:
        before = {'historical_odds': 88135, 'pinnacle_close': 12385, 'bet365': 776, 'sstats_consensus': 240}[src]
        after = current.get(src, 0)
        delta = after - before
        flag = " <<< OVERWRITTEN" if delta < 0 else ""
        print(f"  {src:25s} {before:6d} -> {after:6d}  (delta {delta:+d}){flag}")
    print(f"  {'TOTAL':25s} {101536:6d} -> {sum(current.values()):6d}  (delta {sum(current.values())-101536:+d})")
    
    overwritten = 101536 + 4847 - sum(current.values())
    print(f"\n  OVERWRITTEN (lost): {overwritten} rows")
    print(f"  These had independent source (pinnacle_close/bet365)")
    print(f"  and were REPLACED by sstats_consensus")
    
    # How many match_ids that NOW have sstats_consensus ALSO previously had independent odds?
    r = conn.execute("""
        SELECT COUNT(DISTINCT match_id) FROM match_odds 
        WHERE source='sstats_consensus'
        AND match_id IN (
            SELECT match_id FROM match_odds 
            WHERE source IN ('historical_odds', 'pinnacle_close', 'bet365')
        )
    """).fetchone()
    print(f"\n  match_ids with BOTH sstats_consensus AND independent source: {r[0]}")
    print(f"  (these were NOT overwritten — they kept both)")
    
    r2 = conn.execute("""
        SELECT COUNT(DISTINCT match_id) FROM match_odds 
        WHERE source='sstats_consensus'
        AND match_id NOT IN (
            SELECT match_id FROM match_odds 
            WHERE source IN ('historical_odds', 'pinnacle_close', 'bet365')
        )
    """).fetchone()
    print(f"  match_ids with ONLY sstats_consensus (no independent): {r2[0]}")
