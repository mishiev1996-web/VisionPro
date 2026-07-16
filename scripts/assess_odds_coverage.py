"""Assess current sstats_odds coverage and theoretical capacity."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db
from data_collector import SSTATS_LEAGUE_IDS
import config

print("=" * 60)
print("CURRENT COVERAGE")
print("=" * 60)

with db.connect() as conn:
    r = conn.execute("SELECT COUNT(DISTINCT game_id) FROM sstats_odds").fetchone()
    print(f"Unique games with odds: {r[0]}")

    r = conn.execute("SELECT COUNT(*) FROM sstats_odds").fetchone()
    print(f"Total odds rows: {r[0]}")

    r = conn.execute("SELECT COUNT(DISTINCT game_id) FROM sstats_matches").fetchone()
    print(f"Total sstats_matches: {r[0]}")

    r = conn.execute("""
        SELECT COUNT(DISTINCT m.game_id)
        FROM sstats_matches m
        WHERE m.game_id NOT IN (SELECT DISTINCT game_id FROM sstats_odds)
    """).fetchone()
    print(f"Matches WITHOUT odds: {r[0]}")

    # Per-league breakdown of existing odds
    print("\n--- Per-league odds coverage ---")
    rows = conn.execute("""
        SELECT m.league_name, COUNT(DISTINCT o.game_id) as games
        FROM sstats_odds o
        JOIN sstats_matches m ON o.game_id = m.game_id
        GROUP BY m.league_name
        ORDER BY games DESC
    """).fetchall()
    for r in rows:
        print(f"  {r['league_name']:30s} {r['games']:5d} games")

    # Per-market breakdown
    print("\n--- Top markets ---")
    rows = conn.execute("""
        SELECT market, COUNT(*) as cnt, COUNT(DISTINCT game_id) as games
        FROM sstats_odds
        GROUP BY market
        ORDER BY games DESC
        LIMIT 10
    """).fetchall()
    for r in rows:
        print(f"  {r['market']:35s} {r['games']:5d} games, {r['cnt']:6d} rows")

print("\n" + "=" * 60)
print("THEORETICAL CAPACITY (leagues x seasons)")
print("=" * 60)

leagues_in_config = {k: v for k, v in SSTATS_LEAGUE_IDS.items() if k in config.LEAGUE_TIERS}
print(f"Leagues in config: {len(leagues_in_config)}")
for slug, sid in sorted(leagues_in_config.items(), key=lambda x: x[0]):
    print(f"  {slug:20s} sstats_id={sid}")

print(f"\nSSTATS_LEAGUE_IDS total: {len(SSTATS_LEAGUE_IDS)}")
print(f"LEAGUE_TIERS total: {len(config.LEAGUE_TIERS)}")
extra = set(SSTATS_LEAGUE_IDS.keys()) - set(config.LEAGUE_TIERS.keys())
if extra:
    print(f"Leagues in SSTATS but NOT in LEAGUE_TIERS: {extra}")
