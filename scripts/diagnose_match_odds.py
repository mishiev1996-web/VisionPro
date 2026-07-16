"""Diagnose why only 240 sstats_consensus entries are in match_odds."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import db

with db.connect() as conn:
    # How many sstats_odds games exist?
    r = conn.execute("SELECT COUNT(DISTINCT game_id) FROM sstats_odds").fetchone()
    print(f"sstats_odds unique games: {r[0]}")
    
    # How many sstats_matches exist?
    r = conn.execute("SELECT COUNT(DISTINCT game_id) FROM sstats_matches").fetchone()
    print(f"sstats_matches unique games: {r[0]}")
    
    # How many sstats_matches have home_id AND away_id (needed for match_id resolution)?
    r = conn.execute("""
        SELECT COUNT(*) FROM sstats_matches 
        WHERE home_id IS NOT NULL AND away_id IS NOT NULL
    """).fetchone()
    print(f"sstats_matches with team IDs: {r[0]}")
    
    # How many sstats_consensus in match_odds?
    r = conn.execute("""
        SELECT COUNT(*) FROM match_odds WHERE source='sstats_consensus'
    """).fetchone()
    print(f"match_odds with sstats_consensus: {r[0]}")
    
    # Check: do sstats game_ids map to match_ids in the matches table?
    r = conn.execute("""
        SELECT COUNT(DISTINCT sm.game_id)
        FROM sstats_matches sm
        WHERE sm.game_id IN (SELECT id FROM matches)
    """).fetchone()
    print(f"sstats_matches game_id exists in matches table: {r[0]}")
    
    # Check: does collect_sstats_bulk create matches in the matches table?
    r = conn.execute("""
        SELECT COUNT(DISTINCT sm.game_id)
        FROM sstats_matches sm
        WHERE sm.game_id NOT IN (SELECT id FROM matches)
    """).fetchone()
    print(f"sstats_matches game_id NOT in matches table: {r[0]}")
    
    # Sample: show a few sstats_matches with their team IDs
    rows = conn.execute("""
        SELECT game_id, home_id, away_id, date, league_name, home_team, away_team
        FROM sstats_matches
        WHERE home_id IS NOT NULL AND away_id IS NOT NULL
        LIMIT 5
    """).fetchall()
    print(f"\nSample sstats_matches with team IDs:")
    for r in rows:
        print(f"  game_id={r['game_id']}, home_id={r['home_id']}, away_id={r['away_id']}, {r['home_team']} vs {r['away_team']}, {r['league_name']}")
    
    # Check: are there matches in matches table for these teams?
    for r in rows[:2]:
        match = conn.execute("""
            SELECT id, league_slug, date FROM matches 
            WHERE home_id=? AND away_id=? AND date LIKE ?
        """, (r['home_id'], r['away_id'], (r['date'] or '')[:10] + '%')).fetchone()
        if match:
            print(f"  → Found match: id={match['id']}, {match['league_slug']}, {match['date']}")
        else:
            print(f"  → NO match found in matches table")
