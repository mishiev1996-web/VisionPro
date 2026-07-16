"""Restore overwritten pinnacle_close/bet365 odds by re-running historical_odds collector."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db
import data_collector

# Run collect_odds which reads from football-data.co.uk CSVs
# This uses upsert_match_odds which will overwrite sstats_consensus
# for match_ids that originally had pinnacle_close/bet365
result = data_collector.collect_odds()
print(f"Result: {result}")

# Verify
with db.connect() as conn:
    rows = conn.execute("""
        SELECT source, COUNT(*) as cnt, COUNT(DISTINCT match_id) as matches
        FROM match_odds GROUP BY source ORDER BY cnt DESC
    """).fetchall()
    print("\n=== match_odds per source AFTER restore ===")
    for r in rows:
        print(f"  {r['source']:25s} {r['matches']:6d} matches")
