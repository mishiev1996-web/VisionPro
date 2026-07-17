"""
tennis/features.py — Feature engineering for tennis predictions.

Computes: ELO (overall + surface), serve stats, surface form, H2H,
rank diff, fatigue, momentum, rolling totals.
"""
from __future__ import annotations

import math
import re
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

TOTAL_FEATURE_NAMES = [
    "elo_diff",
    "surface_elo_diff",
    "rank_diff",
    "form_diff",
    "surface_form_diff",
    "h2h_diff",
    "fatigue_diff",
    "streak_diff",
    "recent_total_diff",      # NEW: avg total games in recent matches
    "surface_total_diff",     # NEW: avg total games on this surface
    "is_bo5",                 # NEW: best_of indicator
]


def parse_total_games(score: str) -> Optional[int]:
    """Parse total games from score string like '6-4 7-5 6-3'."""
    if not score:
        return None
    if any(x in str(score).upper() for x in ['W/O', 'RET', 'DEF', 'ABN', 'UNF']):
        return None
    sets = re.findall(r'(\d+)-(\d+)', score)
    if not sets:
        return None
    return sum(int(a) + int(b) for a, b in sets)


def parse_sets_played(score: str) -> Optional[int]:
    """Parse number of sets from score."""
    if not score:
        return None
    sets = re.findall(r'(\d+)-(\d+)', score)
    return len(sets) if sets else None


# ── ELO System ────────────────────────────────────────────────────────────────

class EloSystem:
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
    def __init__(self):
        self.elo = EloSystem()
        self.h2h: Dict[Tuple[int, int], int] = defaultdict(int)
        self.recent: Dict[int, List[bool]] = defaultdict(list)
        self.surface_recent: Dict[Tuple[int, str], List[bool]] = defaultdict(list)
        self.serve_stats: Dict[int, List[Dict]] = defaultdict(list)
        self.last_match_date: Dict[int, float] = {}
        self.match_dates: Dict[int, List[float]] = defaultdict(list)
        self.streak: Dict[int, int] = defaultdict(int)
        self.player_names: Dict[int, str] = {}
        # NEW: rolling totals
        self.recent_totals: Dict[int, List[int]] = defaultdict(list)
        self.surface_totals: Dict[Tuple[int, str], List[int]] = defaultdict(list)

    def load_history(self):
        print("Loading tennis history for features...")
        with tennis_db.connect() as conn:
            matches = [tuple(r) for r in conn.execute("""
                SELECT player1_id, player2_id, winner_id, surface, date,
                       player1_name, player2_name,
                       w_1stIn, w_svpt, w_1stWon, w_2ndWon, w_bpSaved, w_bpFaced,
                       l_1stIn, l_svpt, l_1stWon, l_2ndWon, l_bpSaved, l_bpFaced,
                       score
                FROM tennis_matches 
                WHERE status='finished' AND date IS NOT NULL
                ORDER BY date ASC
            """).fetchall()]

        print(f"  Processing {len(matches)} matches...")
        for m in matches:
            p1_id, p2_id, winner_id = m[0], m[1], m[2]
            surface = (m[3] or "hard").lower()
            date_str = m[4] or ""
            p1_name, p2_name = m[5], m[6]
            score = m[20] if len(m) > 20 else None

            if p1_id: self.player_names[p1_id] = p1_name
            if p2_id: self.player_names[p2_id] = p2_name

            try:
                ts = float(date_str.replace("-", ""))
            except:
                ts = 0

            if winner_id == p1_id:
                w_id, l_id = p1_id, p2_id
            elif winner_id == p2_id:
                w_id, l_id = p2_id, p1_id
            else:
                continue

            # Update all state
            self.update(w_id, l_id, surface, date_str, score=score)

        print(f"  Loaded {len(self.player_names)} players")

    def update(self, winner_id: int, loser_id: int, surface: str,
               date: str = None, serve_stats: Dict = None, score: str = None):
        self.elo.update(winner_id, loser_id, surface)
        self.h2h[(winner_id, loser_id)] += 1

        self.recent[winner_id].append(True)
        self.recent[loser_id].append(False)
        if len(self.recent[winner_id]) > ROLLING_WINDOW:
            self.recent[winner_id] = self.recent[winner_id][-ROLLING_WINDOW:]
        if len(self.recent[loser_id]) > ROLLING_WINDOW:
            self.recent[loser_id] = self.recent[loser_id][-ROLLING_WINDOW:]

        s = surface.lower() if surface.lower() in SURFACES else "hard"
        self.surface_recent[(winner_id, s)].append(True)
        self.surface_recent[(loser_id, s)].append(False)
        if len(self.surface_recent[(winner_id, s)]) > SURFACE_WINDOW:
            self.surface_recent[(winner_id, s)] = self.surface_recent[(winner_id, s)][-SURFACE_WINDOW:]
        if len(self.surface_recent[(loser_id, s)]) > SURFACE_WINDOW:
            self.surface_recent[(loser_id, s)] = self.surface_recent[(loser_id, s)][-SURFACE_WINDOW:]

        if serve_stats:
            for pid, stats in serve_stats.items():
                self.serve_stats[pid].append(stats)
                if len(self.serve_stats[pid]) > ROLLING_WINDOW:
                    self.serve_stats[pid] = self.serve_stats[pid][-ROLLING_WINDOW:]

        try:
            ts = float((date or "").replace("-", ""))
        except:
            ts = 0
        if ts > 0:
            self.last_match_date[winner_id] = ts
            self.last_match_date[loser_id] = ts
            self.match_dates[winner_id].append(ts)
            self.match_dates[loser_id].append(ts)

        self.streak[winner_id] = max(self.streak[winner_id], 0) + 1
        self.streak[loser_id] = min(self.streak[loser_id], 0) - 1

        # Rolling totals
        total = parse_total_games(score)
        if total is not None:
            self.recent_totals[winner_id].append(total)
            self.recent_totals[loser_id].append(total)
            if len(self.recent_totals[winner_id]) > ROLLING_WINDOW:
                self.recent_totals[winner_id] = self.recent_totals[winner_id][-ROLLING_WINDOW:]
            if len(self.recent_totals[loser_id]) > ROLLING_WINDOW:
                self.recent_totals[loser_id] = self.recent_totals[loser_id][-ROLLING_WINDOW:]
            
            self.surface_totals[(winner_id, s)].append(total)
            self.surface_totals[(loser_id, s)].append(total)
            if len(self.surface_totals[(winner_id, s)]) > SURFACE_WINDOW:
                self.surface_totals[(winner_id, s)] = self.surface_totals[(winner_id, s)][-SURFACE_WINDOW:]
            if len(self.surface_totals[(loser_id, s)]) > SURFACE_WINDOW:
                self.surface_totals[(loser_id, s)] = self.surface_totals[(loser_id, s)][-SURFACE_WINDOW:]

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
        return 14.0

    def _recent_total(self, player_id: int) -> float:
        totals = self.recent_totals.get(player_id, [])[-ROLLING_WINDOW:]
        return float(np.mean(totals)) if totals else 25.0

    def _surface_total(self, player_id: int, surface: str) -> float:
        s = surface.lower() if surface.lower() in SURFACES else "hard"
        totals = self.surface_totals.get((player_id, s), [])[-SURFACE_WINDOW:]
        return float(np.mean(totals)) if totals else 25.0

    def get_features(self, p1_id: int, p2_id: int, surface: str,
                     p1_rank: int = None, p2_rank: int = None,
                     current_date: str = None) -> Dict[str, float]:
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
            "rank_diff": rank2 - rank1,
            "form_diff": f1 - f2,
            "surface_form_diff": sf1 - sf2,
            "h2h_diff": h2h1 - h2h2,
            "serve_1stWon_diff": self._avg_serve(p1_id, "1stWon") - self._avg_serve(p2_id, "1stWon"),
            "serve_2ndWon_diff": self._avg_serve(p1_id, "2ndWon") - self._avg_serve(p2_id, "2ndWon"),
            "serve_bpSaved_diff": self._avg_serve(p1_id, "bpSaved") - self._avg_serve(p2_id, "bpSaved"),
            "fatigue_diff": self._fatigue(p1_id, current_ts) - self._fatigue(p2_id, current_ts),
            "streak_diff": self.streak.get(p1_id, 0) - self.streak.get(p2_id, 0),
        }

    def get_total_features(self, p1_id: int, p2_id: int, surface: str,
                           p1_rank: int = None, p2_rank: int = None,
                           is_bo5: int = 0, current_date: str = None) -> Dict[str, float]:
        """Features for total games prediction."""
        base = self.get_features(p1_id, p2_id, surface, p1_rank, p2_rank, current_date)
        
        t1 = self._recent_total(p1_id)
        t2 = self._recent_total(p2_id)
        st1 = self._surface_total(p1_id, surface)
        st2 = self._surface_total(p2_id, surface)
        
        return {
            "elo_diff": base["elo_diff"],
            "surface_elo_diff": base["surface_elo_diff"],
            "rank_diff": base["rank_diff"],
            "form_diff": base["form_diff"],
            "surface_form_diff": base["surface_form_diff"],
            "h2h_diff": base["h2h_diff"],
            "fatigue_diff": base["fatigue_diff"],
            "streak_diff": base["streak_diff"],
            "recent_total_diff": t1 - t2,
            "surface_total_diff": st1 - st2,
            "is_bo5": is_bo5,
        }

    def get_player_name(self, player_id: int) -> str:
        return self.player_names.get(player_id, f"Player_{player_id}")


_engine: Optional[TennisFeatureEngine] = None

def get_engine() -> TennisFeatureEngine:
    global _engine
    if _engine is None:
        _engine = TennisFeatureEngine()
        _engine.load_history()
    return _engine
