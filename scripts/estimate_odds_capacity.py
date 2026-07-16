"""Estimate total available matches per league via sstats query API.
Only counts, doesn't fetch odds. Runs fetch_query with limit=1 to get count."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scrapers import sstats
from data_collector import SSTATS_LEAGUE_IDS
import config

leagues = {k: v for k, v in SSTATS_LEAGUE_IDS.items() if k in config.LEAGUE_TIERS}
current_year = 2026  # approximate
seasons = [current_year - 2, current_year - 1, current_year]  # 2024, 2025, 2026

total_est = 0
print(f"Estimating match counts for {len(leagues)} leagues x {len(seasons)} seasons")
print(f"Seasons: {seasons}")
print()

for slug, sid in sorted(leagues.items(), key=lambda x: x[0]):
    league_total = 0
    for season in seasons:
        try:
            results = sstats.fetch_query(
                condition=f"LeagueId = {sid} AND Year = {season} AND Status = 8",
                fields=["Id"],
                order="Date ASC",
            )
            count = len(results) if results else 0
            league_total += count
            print(f"  {slug}/{season}: {count} matches")
        except Exception as e:
            print(f"  {slug}/{season}: ERROR - {e}")
    print(f"  {slug} TOTAL: {league_total}")
    total_est += league_total
    print()

print(f"\nGRAND TOTAL: {total_est} matches across all leagues/seasons")
print(f"Currently have odds for: 276 matches")
print(f"Potential new matches: {total_est - 276}")
print(f"\nTime estimate: ~{total_est * 2 / 3600:.1f} hours at 2 sec/match")
