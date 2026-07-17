"""
tennis/features.py — Feature engineering for tennis predictions.

Computes: ELO (overall + surface), serve stats, surface form, H2H,
rank diff, fatigue, momentum. All from tennis.db historical data.

Usage:
    from tennis.features import TennisFeatureEngine
    engine = TennisFeatureEngine()
    engine.load_history()  # call once at startup
    features = engine.get_features(player1_id, player2_id, surface)
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
import tennis.tennis_db as tennis_db

# ── Constants ─────────────────────────────────────────────────────────────────
ELO_K = 32
ELO_INIT = 1500
ROLLING_WINDOW = 20
SURFACE_WINDOW = 20
SURFACES = ["hard", "clay", "grass", "carpet"]

FEATURE_NAMES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "form_diff",
    "surface_form_diff",
    "h2h_diff",
    "serve_1stWon_diff",
    "serve_2ndWon_diff",
    "serve_bpSaved_diff",
    "fatigue_diff",
    "streak_diff",
]


# ── ELO System ────────────────────────────────────────────────────────────────

class EloSystem:
    """ELO rating with surface-specific sub-ratings."""

    def __init__(self, k: int = ELO_K, init: float = ELO_INIT):
        self.k = k
        self.init = init
        self.ratings: Dict[int, float] = {}
        self.surface_ratings: Dict[str, Dict[int, float]] = {s: {} for s in SURFACES}

    def _expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))

    def get(self, player_id: int) -> float:
        return self.ratings.get(player_id, self.init)

    def get_surface(self, player_id: int, surface: str) -> float:
        s = surface.lower() if surface.lower() in SURFACES else "hard"
        return self.surface_ratings[s].get(player_id, self.init)

    def update(self, winner_id: int, loser_id: int, surface: str) -> Tuple[float, float, float, float]:
        """Update after match. Returns (w_elo, l_elo, w_surf, l_surf) BEFORE update."""
        w_elo = self.get(winner_id)
        l_elo = self.get(loser_id)
        exp = self._expected(w_elo, l_elo)
        delta = self.k * (1.0 - exp)
        self.ratings[winner_id] = w_elo + delta
        self.ratings[loser_id] = l_elo - delta

        s = surface.lower() if surface.lower() in SURFACES else "hard"
        w_s = self.get_surface(winner_id, s)
        l_s = self.get_surface(loser_id, s)
        exp_s = self._expected(w_s, l_s)
        delta_s = self.k * (1.0 - exp_s)
        self.surface_ratings[s][winner_id] = w_s + delta_s
        self.surface_ratings[s][loser_id] = l_s - delta_s

        return w_elo, l_elo, w_s, l_s


# ── Feature Engine ────────────────────────────────────────────────────────────

class TennisFeatureEngine:
    """Stateful feature engine for tennis match prediction."""

    def __init__(self):
        self.elo = EloSystem()
        self.h2h: Dict[Tuple[int, int], int] = defaultdict(int)
        self.recent: Dict[int, List[bool]] = defaultdict(list)  # last N results
        self.surface_recent: Dict[Tuple[int, str], List[bool]] = defaultdict(list)
        self.serve_stats: Dict[int, List[Dict]] = defaultdict(list)  # last N serve stats
        self.last_match_date: Dict[int, float] = {}  # player_id -> timestamp
        self.match_dates: Dict[int, List[float]] = defaultdict(list)
        self.streak: Dict[int, int] = defaultdict(int)
        self.player_names: Dict[int, str] = {}

    def load_history(self):
        """Load all finished matches from tennis.db and build feature state."""
        print("Loading tennis history for features...")
        with tennis_db.connect() as conn:
            matches = conn.execute("""
                SELECT player1_id, player2_id, winner_id, surface, date,
                       player1_name, player2_name,
                       w_1stIn, w_svpt, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced,
                       l_1stIn, l_svpt, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced
                FROM tennis_matches 
                WHERE status='finished' AND date IS NOT NULL
                ORDER BY date ASC
            """).fetchall()

        print(f"  Processing {len(matches)} matches...")
        for m in matches:
            p1_id, p2_id, winner_id = m[0], m[1], m[2]
            surface = (m[3] or "hard").lower()
            date_str = m[4] or ""
            p1_name, p2_name = m[5], m[6]

            if p1_id: self.player_names[p1_id] = p1_name
            if p2_id: self.player_names[p2_id] = p2_name

            # Parse date
            try:
                ts = float(date_str.replace("-", ""))
            except:
                ts = 0

            # Determine winner/loser IDs
            if winner_id == p1_id:
                w_id, l_id = p1_id, p2_id
            elif winner_id == p2_id:
                w_id, l_id = p2_id, p1_id
            else:
                continue

            # ELO
            self.elo.update(w_id, l_id, surface)

            # H2H
            self.h2h[(w_id, l_id)] += 1

            # Recent form
            self.recent[w_id].append(True)
            self.recent[l_id].append(False)
            if len(self.recent[w_id]) > ROLLING_WINDOW:
                self.recent[w_id] = self.recent[w_id][-ROLLING_WINDOW:]
            if len(self.recent[l_id]) > ROLLING_WINDOW:
                self.recent[l_id] = self.recent[l_id][-ROLLING_WINDOW:]

            # Surface form
            self.surface_recent[(w_id, surface)].append(True)
            self.surface_recent[(l_id, surface)].append(False)
            if len(self.surface_recent[(w_id, surface)]) > SURFACE_WINDOW:
                self.surface_recent[(w_id, surface)] = self.surface_recent[(w_id, surface)][-SURFACE_WINDOW:]
            if len(self.surface_recent[(l_id, surface)]) > SURFACE_WINDOW:
                self.surface_recent[(l_id, surface)] = self.surface_recent[(l_id, surface)][-SURFACE_WINDOW:]

            # Serve stats
            for pid, prefix in [(w_id, "w"), (l_id, "l")]:
                svpt = m[7+8] if prefix == "w" else m[14]  # w_svpt or l_svpt
                st_in = m[7+3] if prefix == "w" else m[14-3]  # w_1stIn or l_1stIn
                st_won = m[7+4] if prefix == "w" else m[14-2]  # w_1stWon or l_1stWon
                s2_won = m[7+5] if prefix == "w" else m[14-1]  # w_2ndWon or l_2ndWon
                bp_s = m[7+6] if prefix == "w" else m[14+1]   # w_bpSaved or l_bpSaved
                bp_f = m[7+7] if prefix == "w" else m[14+2]   # w_bpFaced or l_bpFaced

                stats = {}
                try:
                    svpt = float(svpt) if svpt else 0
                    if svpt > 0:
                        stats["1stIn"] = float(st_in) / svpt * 100 if st_in else 50
                        stats["1stWon"] = float(st_won) / float(st_in) * 100 if st_in and float(st_in) > 0 else 50
                        stats["2ndWon"] = float(s2_won) / (svpt - float(st_in)) * 100 if st_in and svpt > float(st_in) else 50
                        stats["bpSaved"] = float(bp_s) / float(bp_f) * 100 if bp_f and float(bp_f) > 0 else 50
                    else:
                        stats = {"1stIn": 50, "1stWon": 50, "2ndWon": 50, "bpSaved": 50}
                except:
                    stats = {"1stIn": 50, "1stWon": 50, "2ndWon": 50, "bpSaved": 50}

                self.serve_stats[pid].append(stats)
                if len(self.serve_stats[pid]) > ROLLING_WINDOW:
                    self.serve_stats[pid] = self.serve_stats[pid][-ROLLING_WINDOW:]

            # Fatigue
            self.last_match_date[w_id] = ts
            self.last_match_date[l_id] = ts
            self.match_dates[w_id].append(ts)
            self.match_dates[l_id].append(ts)

            # Streak
            self.streak[w_id] = max(self.streak[w_id], 0) + 1
            self.streak[l_id] = min(self.streak[l_id], 0) - 1

        print(f"  Loaded {len(self.player_names)} players")

    def _avg_serve(self, player_id: int, stat: str) -> float:
        stats = self.serve_stats.get(player_id, [])[-ROLLING_WINDOW:]
        if not stats:
            return 50.0
        vals = [s[stat] for s in stats if stat in s]
        return float(np.mean(vals)) if vals else 50.0

    def _form(self, player_id: int) -> float:
        recent = self.recent.get(player_id, [])[-ROLLING_WINDOW:]
        return sum(recent) / len(recent) if recent else 0.5

    def _surface_form(self, player_id: int, surface: str) -> float:
        s = surface.lower() if surface.lower() in SURFACES else "hard"
        recent = self.surface_recent.get((player_id, s), [])[-SURFACE_WINDOW:]
        return sum(recent) / len(recent) if recent else 0.5

    def _fatigue(self, player_id: int, current_ts: float) -> float:
        last = self.last_match_date.get(player_id)
        if last and current_ts > 0:
            return current_ts - last
        return 14.0  # default

    def _matches_14d(self, player_id: int, current_ts: float) -> int:
        dates = self.match_dates.get(player_id, [])
        cutoff = current_ts - 14
        return sum(1 for d in dates if d >= cutoff)

    def get_features(self, p1_id: int, p2_id: int, surface: str,
                     p1_rank: int = None, p2_rank: int = None,
                     current_date: str = None) -> Dict[str, float]:
        """Get feature vector for a match between p1 and p2."""
        # Parse current date
        try:
            current_ts = float((current_date or "").replace("-", ""))
        except:
            current_ts = 0

        elo1 = self.elo.get(p1_id)
        elo2 = self.elo.get(p2_id)
        s1 = self.elo.get_surface(p1_id, surface)
        s2 = self.elo.get_surface(p2_id, surface)

        h2h1 = self.h2h.get((p1_id, p2_id), 0)
        h2h2 = self.h2h.get((p2_id, p1_id), 0)

        f1 = self._form(p1_id)
        f2 = self._form(p2_id)
        sf1 = self._surface_form(p1_id, surface)
        sf2 = self._surface_form(p2_id, surface)

        rank1 = p1_rank if p1_rank else 100
        rank2 = p2_rank if p2_rank else 100

        return {
            "elo_diff": elo1 - elo2,
            "surface_elo_diff": s1 - s2,
            "rank_diff": rank2 - rank1,  # positive = p1 ranked higher
            "form_diff": f1 - f2,
            "surface_form_diff": sf1 - sf2,
            "h2h_diff": h2h1 - h2h2,
            "serve_1stWon_diff": self._avg_serve(p1_id, "1stWon") - self._avg_serve(p2_id, "1stWon"),
            "serve_2ndWon_diff": self._avg_serve(p1_id, "2ndWon") - self._avg_serve(p2_id, "2ndWon"),
            "serve_bpSaved_diff": self._avg_serve(p1_id, "bpSaved") - self._avg_serve(p2_id, "bpSaved"),
            "fatigue_diff": self._fatigue(p1_id, current_ts) - self._fatigue(p2_id, current_ts),
            "streak_diff": self.streak.get(p1_id, 0) - self.streak.get(p2_id, 0),
        }

    def get_player_name(self, player_id: int) -> str:
        return self.player_names.get(player_id, f"Player_{player_id}")


# ── Singleton ─────────────────────────────────────────────────────────────────

_engine: Optional[TennisFeatureEngine] = None

def get_engine() -> TennisFeatureEngine:
    global _engine
    if _engine is None:
        _engine = TennisFeatureEngine()
        _engine.load_history()
    return _engine
