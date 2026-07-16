"""Restore overwritten odds: for match_ids that now have sstats_consensus,
check if they originally had pinnacle_close or bet365 (from historical_odds.py).
Since we can't know the original data, we re-download the CSVs and re-insert."""
import sys, os, io, csv, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db
from scrapers import historical_odds

print("Restoring pinnacle_close/bet365 odds from football-data.co.uk CSVs...")

# Get all match_ids currently with sstats_consensus
with db.connect() as conn:
    rows = conn.execute("""
        SELECT match_id FROM match_odds WHERE source='sstats_consensus'
    """).fetchall()
    sstats_match_ids = set(r["match_id"] for r in rows)
print(f"sstats_consensus match_ids: {len(sstats_match_ids)}")

# Get match details for these match_ids
with db.connect() as conn:
    placeholders = ",".join("?" * len(sstats_match_ids))
    matches = conn.execute(f"""
        SELECT id, league_slug, season, date, home_id, away_id
        FROM matches WHERE id IN ({placeholders})
    """, list(sstats_match_ids)).fetchall()

# Build lookup: (league, date, home_name, away_name) → match_id
team_cache = {}
def team_name(tid):
    if tid not in team_cache:
        with db.connect() as c:
            t = c.execute("SELECT name FROM teams WHERE id=?", (tid,)).fetchone()
            team_cache[tid] = t["name"] if t else str(tid)
    return team_cache[tid_name]

# Collect odds from CSVs for affected leagues/seasons
leagues_needed = set()
seasons_needed = set()
for m in matches:
    leagues_needed.add(m["league_slug"])
    seasons_needed.add(m["season"])

print(f"Need odds for {len(leagues_needed)} leagues, seasons: {sorted(seasons_needed)}")

restored = 0
with db.connect() as conn:
    for league in sorted(leagues_needed):
        for season in sorted(seasons_needed):
            try:
                odds_list = historical_odds.fetch_league_season_odds(league, season)
                if not odds_list:
                    continue
                for entry in odds_list:
                    mid = historical_odds.find_match_id(conn, league, entry["date"],
                                                        entry["home"], entry["away"])
                    if mid is None:
                        continue
                    if mid not in sstats_match_ids:
                        continue
                    # This match_id currently has sstats_consensus — restore original source
                    p_h, p_d, p_a = historical_odds.odds_to_implied(
                        entry["home_odds"], entry["draw_odds"], entry["away_odds"])
                    db.upsert_match_odds(conn, mid,
                                         entry["home_odds"], entry["draw_odds"], entry["away_odds"],
                                         p_h, p_d, p_a,
                                         entry["source"], entry["date"])
                    restored += 1
            except Exception as e:
                print(f"  {league}/{season}: error - {e}")
        conn.commit()

print(f"\nRestored: {restored} rows")

# Verify
with db.connect() as conn:
    rows = conn.execute("""
        SELECT source, COUNT(*) as cnt, COUNT(DISTINCT match_id) as matches
        FROM match_odds GROUP BY source ORDER BY cnt DESC
    """).fetchall()
    print("\n=== match_odds per source AFTER restore ===")
    for r in rows:
        print(f"  {r['source']:25s} {r['matches']:6d} matches")
