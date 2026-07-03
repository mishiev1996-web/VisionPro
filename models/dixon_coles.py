"""
models/dixon_coles.py — classic football scoring model (Dixon & Coles, 1997).

Each team t has attack strength α_t and defence strength β_t. Home advantage γ
is a single scalar. Goals scored by home team are modelled as
    λ_h = exp(α_h + β_a + γ)
and by away team as
    λ_a = exp(α_a + β_h).
Joint goal counts are Poisson-distributed; outcome probabilities sum over the
score-grid up to MAX_GOALS.

Original DC adds a low-score correction τ(i, j, λ_h, λ_a) for {0-0, 0-1, 1-0, 1-1}
because plain independent Poisson under-predicts those specific scores.

Optional time-decay weight per match (ξ): observations from t days ago weigh
exp(-ξ × t). Standard literature value ξ ≈ 0.0065 (~halves every 100 days).

Fit by maximum likelihood. Pure numpy + scipy.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


MAX_GOALS = 8


def _dc_correction(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Low-score correction from Dixon & Coles."""
    if i == 0 and j == 0: return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1: return 1.0 + lam_h * rho
    if i == 1 and j == 0: return 1.0 + lam_a * rho
    if i == 1 and j == 1: return 1.0 - rho
    return 1.0


def _score_matrix(lam_h: float, lam_a: float, rho: float) -> np.ndarray:
    """Joint probability matrix P[i, j] = P(home=i, away=j) for i, j in 0..MAX_GOALS."""
    p_h = poisson.pmf(np.arange(MAX_GOALS + 1), lam_h)
    p_a = poisson.pmf(np.arange(MAX_GOALS + 1), lam_a)
    grid = np.outer(p_h, p_a)
    # Apply low-score correction
    grid[0, 0] *= _dc_correction(0, 0, lam_h, lam_a, rho)
    grid[0, 1] *= _dc_correction(0, 1, lam_h, lam_a, rho)
    grid[1, 0] *= _dc_correction(1, 0, lam_h, lam_a, rho)
    grid[1, 1] *= _dc_correction(1, 1, lam_h, lam_a, rho)
    return grid


class DixonColes:
    """Single-league Dixon-Coles model.

    Usage:
        m = DixonColes(); m.fit(matches); proba = m.predict_proba(home, away)
    """
    def __init__(self, decay_xi: float = 0.0065, max_iter: int = 200):
        self.decay_xi = decay_xi
        self.max_iter = max_iter
        self.teams_: List[str] = []
        self.team_idx_: Dict[str, int] = {}
        self.params_: Optional[np.ndarray] = None    # [α_1..α_n, β_1..β_n, γ, ρ]
        self.fitted = False

    def _decode(self, params: np.ndarray):
        n = len(self.teams_)
        att = params[:n]
        deff = params[n:2*n]
        gamma = params[2*n]
        rho = params[2*n + 1]
        return att, deff, gamma, rho

    def _nll(self, params: np.ndarray, matches: List[Dict]) -> float:
        att, deff, gamma, rho = self._decode(params)
        if rho < -0.5 or rho > 0.5: return 1e12

        n_matches = len(matches)
        home_idx = np.empty(n_matches, dtype=int)
        away_idx = np.empty(n_matches, dtype=int)
        home_goals = np.empty(n_matches, dtype=int)
        away_goals = np.empty(n_matches, dtype=float)
        weights = np.empty(n_matches, dtype=float)

        valid = 0
        for m in matches:
            h = self.team_idx_.get(m["home"])
            a = self.team_idx_.get(m["away"])
            if h is None or a is None: continue
            home_idx[valid] = h
            away_idx[valid] = a
            home_goals[valid] = int(m["home_goals"])
            away_goals[valid] = int(m["away_goals"])
            weights[valid] = float(m.get("weight", 1.0))
            valid += 1

        if valid == 0: return 1e12

        home_idx = home_idx[:valid]
        away_idx = away_idx[:valid]
        home_goals = home_goals[:valid]
        away_goals = away_goals[:valid].astype(int)
        weights = weights[:valid]

        lam_h = np.exp(att[home_idx] + deff[away_idx] + gamma)
        lam_a = np.exp(att[away_idx] + deff[home_idx])

        p_hg = poisson.pmf(home_goals, lam_h)
        p_ag = poisson.pmf(away_goals, lam_a)

        rho_clip = np.clip(rho, -0.49, 0.49)
        corr = np.ones(valid)
        m00 = (home_goals == 0) & (away_goals == 0)
        m01 = (home_goals == 0) & (away_goals == 1)
        m10 = (home_goals == 1) & (away_goals == 0)
        m11 = (home_goals == 1) & (away_goals == 1)
        corr[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho_clip
        corr[m01] = 1.0 + lam_h[m01] * rho_clip
        corr[m10] = 1.0 + lam_a[m10] * rho_clip
        corr[m11] = 1.0 - rho_clip

        joint = np.maximum(p_hg * p_ag * corr, 1e-12)
        ll = np.sum(weights * np.log(joint))
        constraint_penalty = (att.sum()) ** 2 * 100
        return -ll + constraint_penalty

    def fit(self, matches: List[Dict],
            reference_date: Optional[str] = None) -> "DixonColes":
        """matches: [{home, away, home_goals, away_goals, date(optional)}]"""
        # Discover teams
        seen = []
        seen_set = set()
        for m in matches:
            for k in ("home", "away"):
                if m[k] not in seen_set:
                    seen.append(m[k]); seen_set.add(m[k])
        self.teams_ = seen
        self.team_idx_ = {t: i for i, t in enumerate(seen)}
        n = len(seen)

        # Need at least 2 teams and some matches
        if n < 2 or len(matches) < 10:
            self.fitted = False
            return self

        # Time-decay weights
        if reference_date and self.decay_xi > 0:
            ref = _parse_date(reference_date)
            for m in matches:
                d = _parse_date(m.get("date", ""))
                if d and ref:
                    days = max(0, (ref - d).days)
                    m["weight"] = math.exp(-self.decay_xi * days)
                else:
                    m["weight"] = 1.0
        else:
            for m in matches:
                m.setdefault("weight", 1.0)

        # Initial: attack=0, defence=0 (means strength), gamma=0.3, rho=-0.1
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.3, -0.1]])
        try:
            res = minimize(
                self._nll, x0, args=(matches,),
                method="L-BFGS-B",
                options={"maxiter": self.max_iter, "disp": False},
            )
            # Check convergence
            if not res.success:
                print(f"  [DixonColes] optimizer warning: {res.message}")
            # Validate parameters are finite
            if not np.all(np.isfinite(res.x)):
                print(f"  [DixonColes] non-finite parameters, discarding fit")
                self.fitted = False
                return self
            self.params_ = res.x
            self.fitted = True
        except Exception as e:
            print(f"  [DixonColes] fit failed: {e}")
            self.fitted = False
        return self

    def predict_proba(self, home: str, away: str) -> Tuple[float, float, float]:
        """Returns (p_home_win, p_draw, p_away_win)."""
        if not self.fitted: return (0.45, 0.27, 0.28)   # mild defaults
        att, deff, gamma, rho = self._decode(self.params_)
        h = self.team_idx_.get(home)
        a = self.team_idx_.get(away)
        if h is None or a is None: return (0.45, 0.27, 0.28)
        lam_h = math.exp(att[h] + deff[a] + gamma)
        lam_a = math.exp(att[a] + deff[h])
        grid = _score_matrix(lam_h, lam_a, rho)
        p_home = float(np.tril(grid, -1).sum())
        p_draw = float(np.trace(grid))
        p_away = float(np.triu(grid, 1).sum())
        s = p_home + p_draw + p_away
        if s <= 0: return (0.45, 0.27, 0.28)
        return p_home / s, p_draw / s, p_away / s

    def expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        if not self.fitted: return (1.3, 1.1)
        att, deff, gamma, _ = self._decode(self.params_)
        h, a = self.team_idx_.get(home), self.team_idx_.get(away)
        if h is None or a is None: return (1.3, 1.1)
        return math.exp(att[h] + deff[a] + gamma), math.exp(att[a] + deff[h])


def _parse_date(s: str) -> Optional[dt.date]:
    try:
        return dt.datetime.fromisoformat(s.replace(" ", "T")).date()
    except Exception:
        return None
