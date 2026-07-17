"""
download_jeffsackmann.py — Download JeffSackmann ATP data (via fork sleepomeno/tennis_atp).

Data: 1968-2015, ATP main tour matches with serve stats.
Source: https://github.com/sleepomeno/tennis_atp (fork of JeffSackmann)
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import tennis.tennis_db as tennis_db

REPO = "sleepomeno/tennis_atp"
BASE_URL = f"https://raw.githubusercontent.com/{REPO}/master"
YEARS = list(range(1968, 2016))  # 1968-2015
DATA_DIR = Path(__file__).parent.parent / "data" / "jeffsackmann_cache"


def download_year(year: int) -> pd.DataFrame:
    """Download a single year CSV."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"atp_matches_{year}.csv"
    
    if cache.exists():
        return pd.read_csv(cache, low_memory=False)
    
    url = f"{BASE_URL}/atp_matches_{year}.csv"
    print(f"  Downloading {year}...")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            cache.write_bytes(r.content)
            return pd.read_csv(io.StringIO(r.content.decode('utf-8')))
    except Exception as e:
        print(f"    Error: {e}")
    
    return pd.DataFrame()


def import_to_db(df: pd.DataFrame, year: int):
    """Import JeffSackmann data into tennis_matches."""
    if df.empty:
        return 0
    
    imported = 0
    with tennis_db.connect() as conn:
        for _, row in df.iterrows():
            try:
                winner_id = int(row.get('winner_id', 0)) if pd.notna(row.get('winner_id')) else None
                loser_id = int(row.get('loser_id', 0)) if pd.notna(row.get('loser_id')) else None
                if not winner_id or not loser_id:
                    continue
                
                # Upsert players
                for pid, name, hand, ht, ioc, rank, rpts in [
                    (winner_id, row.get('winner_name'), row.get('winner_hand'),
                     row.get('winner_ht'), row.get('winner_ioc'),
                     row.get('winner_rank'), row.get('winner_rank_points')),
                    (loser_id, row.get('loser_name'), row.get('loser_hand'),
                     row.get('loser_ht'), row.get('loser_ioc'),
                     row.get('loser_rank'), row.get('loser_rank_points')),
                ]:
                    if pd.notna(name):
                        conn.execute("""
                            INSERT OR REPLACE INTO tennis_players 
                            (id, name, country, ranking, ranking_points)
                            VALUES (?, ?, ?, ?, ?)
                        """, (pid, str(name),
                              str(ioc) if pd.notna(ioc) else None,
                              int(rank) if pd.notna(rank) else None,
                              int(rpts) if pd.notna(rpts) else None))
                
                # Parse date
                tourney_date = str(row.get('tourney_date', ''))
                if len(tourney_date) == 8:
                    date_str = f"{tourney_date[:4]}-{tourney_date[4:6]}-{tourney_date[6:8]}"
                else:
                    date_str = tourney_date
                
                # best_of
                best_of = int(row.get('best_of', 3)) if pd.notna(row.get('best_of')) else 3
                
                conn.execute("""
                    INSERT OR IGNORE INTO tennis_matches 
                    (tournament_id, tournament_name, round_name,
                     surface, date, player1_id, player2_id, player1_name, player2_name,
                     winner_id, score, status,
                     w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms,
                     w_bpSaved, w_bpFaced,
                     l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms,
                     l_bpSaved, l_bpFaced, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'finished',
                            ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(row.get('tourney_id', '')),
                    str(row.get('tourney_name', '')),
                    str(row.get('round', '')),
                    str(row.get('surface', '')).lower(),
                    date_str,
                    winner_id, loser_id,
                    str(row.get('winner_name', '')),
                    str(row.get('loser_name', '')),
                    winner_id,
                    str(row.get('score', '')),
                    int(row.get('w_ace', 0)) if pd.notna(row.get('w_ace')) else None,
                    int(row.get('w_df', 0)) if pd.notna(row.get('w_df')) else None,
                    int(row.get('w_svpt', 0)) if pd.notna(row.get('w_svpt')) else None,
                    int(row.get('w_1stIn', 0)) if pd.notna(row.get('w_1stIn')) else None,
                    int(row.get('w_1stWon', 0)) if pd.notna(row.get('w_1stWon')) else None,
                    int(row.get('w_2ndWon', 0)) if pd.notna(row.get('w_2ndWon')) else None,
                    int(row.get('w_SvGms', 0)) if pd.notna(row.get('w_SvGms')) else None,
                    int(row.get('w_bpSaved', 0)) if pd.notna(row.get('w_bpSaved')) else None,
                    int(row.get('w_bpFaced', 0)) if pd.notna(row.get('w_bpFaced')) else None,
                    int(row.get('l_ace', 0)) if pd.notna(row.get('l_ace')) else None,
                    int(row.get('l_df', 0)) if pd.notna(row.get('l_df')) else None,
                    int(row.get('l_svpt', 0)) if pd.notna(row.get('l_svpt')) else None,
                    int(row.get('l_1stIn', 0)) if pd.notna(row.get('l_1stIn')) else None,
                    int(row.get('l_1stWon', 0)) if pd.notna(row.get('l_1stWon')) else None,
                    int(row.get('l_2ndWon', 0)) if pd.notna(row.get('l_2ndWon')) else None,
                    int(row.get('l_SvGms', 0)) if pd.notna(row.get('l_SvGms')) else None,
                    int(row.get('l_bpSaved', 0)) if pd.notna(row.get('l_bpSaved')) else None,
                    int(row.get('l_bpFaced', 0)) if pd.notna(row.get('l_bpFaced')) else None,
                    pd.Timestamp.now().isoformat(timespec='seconds'),
                ))
                imported += 1
            except Exception:
                continue
    
    return imported


def main():
    print("=" * 60)
    print("  JeffSackmann ATP Data Download & Import")
    print("=" * 60)
    
    tennis_db.init_db()
    
    total = 0
    for year in YEARS:
        df = download_year(year)
        if not df.empty:
            count = import_to_db(df, year)
            total += count
            print(f"  {year}: {count} matches")
    
    with tennis_db.connect() as conn:
        players = conn.execute("SELECT COUNT(*) FROM tennis_players").fetchone()[0]
        matches = conn.execute("SELECT COUNT(*) FROM tennis_matches").fetchone()[0]
    
    print(f"\n{'=' * 60}")
    print(f"  Imported: {total}")
    print(f"  Total: {players} players, {matches} matches")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
