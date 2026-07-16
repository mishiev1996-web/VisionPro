"""Backfill sstats_odds for matches that already exist in sstats_matches but lack odds.

Targets: sstats_matches.game_id NOT IN sstats_odds.
Rate limit: 2 sec per API call (sstats.net).
Resumable: skips matches that already have odds.
"""
import sys, os, time, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import db
from scrapers import sstats

LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "sstats_odds_backfill.log")

def log(msg):
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def get_targets(limit=None):
    """Get game_ids from sstats_matches that don't have odds yet."""
    with db.connect() as conn:
        sql = """
            SELECT m.game_id, m.league_name, m.date, m.home_team, m.away_team
            FROM sstats_matches m
            WHERE m.game_id NOT IN (SELECT DISTINCT game_id FROM sstats_odds)
            ORDER BY m.game_id
        """
        if limit:
            sql += f" LIMIT {limit}"
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]

def main():
    log("=" * 60)
    log("Starting sstats_odds backfill")
    log("=" * 60)

    # Current state
    with db.connect() as conn:
        before_count = conn.execute("SELECT COUNT(DISTINCT game_id) FROM sstats_odds").fetchone()[0]
    log(f"Current unique games with odds: {before_count}")

    targets = get_targets()
    log(f"Matches without odds: {len(targets)}")

    if not targets:
        log("Nothing to do!")
        return

    # League breakdown
    from collections import Counter
    league_counts = Counter(t["league_name"] for t in targets)
    log("\nTarget breakdown by league:")
    for lg, cnt in league_counts.most_common(20):
        log(f"  {lg:30s} {cnt:5d}")

    log(f"\nStarting collection ({len(targets)} matches)...")
    log(f"Estimated time: {len(targets) * 2 / 60:.0f} min (at 2 sec/match)")

    saved = 0
    errors = 0
    empty = 0
    t_start = time.time()

    now = dt.datetime.now().isoformat(timespec="seconds")

    with db.connect() as conn:
        for i, target in enumerate(targets):
            game_id = target["game_id"]
            league = target["league_name"]
            match_str = f"{target['home_team']} vs {target['away_team']}"

            try:
                odds_blocks = sstats.fetch_odds(game_id)
                if odds_blocks:
                    n = db.save_sstats_odds(conn, game_id, odds_blocks, now)
                    saved += n

                    # Also save consensus to match_odds
                    consensus = sstats.consensus_odds(odds_blocks)
                    if consensus:
                        # Find match_id in main matches table
                        match_row = conn.execute(
                            "SELECT id FROM matches WHERE home_id=? AND away_id=? AND date LIKE ?",
                            (target.get("home_id") or 0, target.get("away_id") or 0,
                             (target.get("date") or "")[:10] + "%")
                        ).fetchone()
                        if match_row:
                            db.upsert_match_odds(
                                conn, match_row["id"],
                                consensus["avg_home_odds"],
                                consensus["avg_draw_odds"],
                                consensus["avg_away_odds"],
                                consensus["implied_h"],
                                consensus["implied_d"],
                                consensus["implied_a"],
                                "sstats_consensus", now,
                            )
                else:
                    empty += 1
            except Exception as e:
                errors += 1
                if errors <= 10:
                    log(f"  ERROR game_id={game_id} ({match_str}): {e}")

            # Progress every 50 matches
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                remaining = (len(targets) - i - 1) / rate / 60 if rate > 0 else 0
                log(f"  Progress: {i+1}/{len(targets)} "
                    f"(saved_rows={saved}, empty={empty}, errors={errors}) "
                    f"rate={rate:.2f}/s ETA={remaining:.0f}min")

            # Commit every 100
            if (i + 1) % 100 == 0:
                conn.commit()

    # Final stats
    elapsed = time.time() - t_start
    with db.connect() as conn:
        after_count = conn.execute("SELECT COUNT(DISTINCT game_id) FROM sstats_odds").fetchone()[0]

    log("=" * 60)
    log(f"DONE in {elapsed/60:.1f} min")
    log(f"Processed: {len(targets)} matches")
    log(f"Saved odds rows: {saved}")
    log(f"Empty (no odds): {empty}")
    log(f"Errors: {errors}")
    log(f"Coverage: {before_count} -> {after_count} (+{after_count - before_count} new)")

    # Per-league final breakdown
    with db.connect() as conn:
        rows = conn.execute("""
            SELECT m.league_name, COUNT(DISTINCT o.game_id) as games
            FROM sstats_odds o
            JOIN sstats_matches m ON o.game_id = m.game_id
            GROUP BY m.league_name
            ORDER BY games DESC
        """).fetchall()
    log("\nFinal per-league coverage:")
    for r in rows:
        log(f"  {r['league_name']:30s} {r['games']:5d}")

if __name__ == "__main__":
    main()
