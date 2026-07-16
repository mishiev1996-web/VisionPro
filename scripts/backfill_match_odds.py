"""Backfill match_odds with sstats consensus for all sstats_odds games.

Pipeline:
1. Get all game_ids from sstats_odds that don't have sstats_consensus in match_odds
2. For each game_id, compute consensus_odds() from raw odds
3. Find the corresponding match_id in matches table (by team IDs + date)
4. Save to match_odds with source='sstats_consensus'
"""
import sys, os, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
from scrapers import sstats
from data_collector import _deterministic_id

def get_odds_blocks_from_db(game_id):
    """Reconstruct odds_blocks format from sstats_odds table."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT bookmaker, market, name, value FROM sstats_odds WHERE game_id=?",
            (game_id,)
        ).fetchall()
    
    bookmakers = {}
    for r in rows:
        bm = r["bookmaker"]
        if bm not in bookmakers:
            bookmakers[bm] = {"bookmakerName": bm, "odds": []}
        market = None
        for m in bookmakers[bm]["odds"]:
            if m["marketName"] == r["market"]:
                market = m
                break
        if market is None:
            market = {"marketName": r["market"], "odds": []}
            bookmakers[bm]["odds"].append(market)
        market["odds"].append({"name": r["name"], "value": r["value"]})
    
    return list(bookmakers.values())


def main():
    print("=" * 60)
    print("Backfilling match_odds with sstats consensus")
    print("=" * 60)
    
    with db.connect() as conn:
        # Get game_ids with odds but no sstats_consensus in match_odds
        existing = conn.execute("""
            SELECT DISTINCT game_id FROM sstats_odds
        """).fetchall()
        existing_ids = [r["game_id"] for r in existing]
        
        already_done = conn.execute("""
            SELECT DISTINCT mo.match_id FROM match_odds mo
            WHERE mo.source = 'sstats_consensus'
        """).fetchall()
        done_match_ids = set(r["match_id"] for r in already_done)
    
    print(f"Total sstats_odds games: {len(existing_ids)}")
    print(f"Already have sstats_consensus in match_odds: {len(done_match_ids)}")
    
    # Get sstats_matches info for team ID mapping
    with db.connect() as conn:
        match_info = conn.execute("""
            SELECT game_id, home_id, away_id, date, league_name, home_team, away_team
            FROM sstats_matches
        """).fetchall()
    
    info_map = {}
    for r in match_info:
        info_map[r["game_id"]] = dict(r)
    
    saved = 0
    skipped = 0
    errors = 0
    
    with db.connect() as conn:
        for game_id in existing_ids:
            info = info_map.get(game_id)
            if not info or not info.get("home_id") or not info.get("away_id"):
                skipped += 1
                continue
            
            # Find match_id in matches table
            date_prefix = (info["date"] or "")[:10]
            match_row = conn.execute("""
                SELECT id FROM matches
                WHERE home_id=? AND away_id=? AND date LIKE ?
            """, (info["home_id"], info["away_id"], date_prefix + "%")).fetchone()
            
            if not match_row:
                skipped += 1
                continue
            
            match_id = match_row["id"]
            
            if match_id in done_match_ids:
                continue
            
            # Reconstruct odds_blocks and compute consensus
            try:
                odds_blocks = get_odds_blocks_from_db(game_id)
                if not odds_blocks:
                    skipped += 1
                    continue
                
                consensus = sstats.consensus_odds(odds_blocks)
                if not consensus:
                    skipped += 1
                    continue
                
                now = dt.datetime.now().isoformat(timespec="seconds")
                db.upsert_match_odds(
                    conn, match_id,
                    consensus["avg_home_odds"],
                    consensus["avg_draw_odds"],
                    consensus["avg_away_odds"],
                    consensus["implied_h"],
                    consensus["implied_d"],
                    consensus["implied_a"],
                    "sstats_consensus", now,
                )
                saved += 1
                done_match_ids.add(match_id)
                
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ERROR game_id={game_id}: {e}")
            
            if saved % 100 == 0 and saved > 0:
                conn.commit()
                print(f"  Progress: saved={saved}, skipped={skipped}, errors={errors}")
    
    # Final stats
    with db.connect() as conn:
        total_odds = conn.execute("SELECT COUNT(*) FROM match_odds").fetchone()[0]
        sstats_odds = conn.execute("""
            SELECT COUNT(*) FROM match_odds WHERE source='sstats_consensus'
        """).fetchone()[0]
        matches_with_odds = conn.execute("""
            SELECT COUNT(DISTINCT match_id) FROM match_odds
        """).fetchone()[0]
        total_matches = conn.execute("""
            SELECT COUNT(*) FROM matches WHERE is_result=1
        """).fetchone()[0]
    
    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Saved: {saved}")
    print(f"  Skipped: {skipped}")
    print(f"  Errors: {errors}")
    print(f"  match_odds total: {total_odds}")
    print(f"  sstats_consensus: {sstats_odds}")
    print(f"  Matches with ANY odds: {matches_with_odds}/{total_matches} ({matches_with_odds/total_matches*100:.1f}%)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
