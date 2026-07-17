"""
download_tml.py — Download TML-Database CSVs and import into tennis.db.

Uses name-based matching to map TML string IDs to integer ATP IDs.
"""
from __future__ import annotations

import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
import tennis.tennis_db as tennis_db

TML_BASE = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"
YEARS = list(range(2020, 2027))
DATA_DIR = Path(__file__).parent.parent / "data" / "tml_cache"


def string_to_int_id(s: str) -> int:
    """Convert TML string ID to a stable integer ID."""
    h = hashlib.md5(s.encode()).hexdigest()
    return int(h[:8], 16)


def download_csv(year: int) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / f"{year}.csv"
    if dest.exists():
        return dest
    url = f"{TML_BASE}/{year}.csv"
    print(f"  Downloading {year}.csv...")
    r = requests.get(url, timeout=30)
    if r.status_code == 200:
        dest.write_bytes(r.content)
    return dest


def import_to_db(csv_path: Path, name_to_id: Dict) -> int:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0
    
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  {csv_path.name}: {len(df)} rows")
    
    imported = 0
    now = datetime.now().isoformat(timespec="seconds")
    
    errors = []
    with tennis_db.connect() as conn:
        for idx, row in df.iterrows():
            try:
                # Get player IDs (TML string -> int)
                w_tml_id = str(row.get('winner_id', ''))
                l_tml_id = str(row.get('loser_id', ''))
                w_name = str(row.get('winner_name', ''))
                l_name = str(row.get('loser_name', ''))
                
                if not w_name or not l_name:
                    continue
                
                # Map or create integer IDs
                if w_name not in name_to_id:
                    name_to_id[w_name] = string_to_int_id(w_tml_id)
                if l_name not in name_to_id:
                    name_to_id[l_name] = string_to_int_id(l_tml_id)
                
                w_id = name_to_id[w_name]
                l_id = name_to_id[l_name]
                
                # Upsert players (only columns that exist in schema)
                for pid, name, ioc, rank, rpts in [
                    (w_id, w_name, row.get('winner_ioc'),
                     row.get('winner_rank'), row.get('winner_rank_points')),
                    (l_id, l_name, row.get('loser_ioc'),
                     row.get('loser_rank'), row.get('loser_rank_points')),
                ]:
                    conn.execute("""
                        INSERT OR REPLACE INTO tennis_players 
                        (id, name, country, ranking, ranking_points)
                        VALUES (?, ?, ?, ?, ?)
                    """, (pid, name,
                          str(ioc) if pd.notna(ioc) else None,
                          int(rank) if pd.notna(rank) else None,
                          int(rpts) if pd.notna(rpts) else None))
                
                # Parse date
                tourney_date = str(row.get('tourney_date', ''))
                if len(tourney_date) == 8:
                    date_str = f"{tourney_date[:4]}-{tourney_date[4:6]}-{tourney_date[6:8]}"
                else:
                    date_str = tourney_date
                
                # Insert match
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
                    string_to_int_id(str(row.get('tourney_id', ''))),
                    str(row.get('tourney_name', '')),
                    str(row.get('round', '')),
                    str(row.get('surface', '')).lower(),
                    date_str,
                    w_id, l_id, w_name, l_name,
                    w_id,
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
                    now,
                ))
                imported += 1
            except Exception as e:
                if len(errors) < 3:
                    errors.append(f"Row {idx}: {e}")
                continue
    
    if errors:
        for err in errors:
            print(f"  ERROR: {err}")
    
    return imported


def main():
    from typing import Dict
    print("=" * 60)
    print("  TML-Database Download & Import")
    print("=" * 60)
    
    tennis_db.init_db()
    name_to_id: Dict[str, int] = {}
    
    total = 0
    for year in YEARS:
        csv_path = download_csv(year)
        if csv_path.exists():
            count = import_to_db(csv_path, name_to_id)
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
