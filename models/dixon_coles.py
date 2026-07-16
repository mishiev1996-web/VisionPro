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
Arrays are pre-vectorized once before optimize (NLL no longer re-parses dicts).
"""
from __future__ import annotations

import datetime as dt
import math
import time
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
    grid[0, 0] *= _dc_correction(0, 0, lam_h, lam_a, rho)
    grid[0, 1] *= _dc_correction(0, 1, lam_h, lam_a, rho)
    grid[1, 0] *= _dc_correction(1, 0, lam_h, lam_a, rho)
    grid[1, 1] *= _dc_correction(1, 1, lam_h, lam_a, rho)
    return grid


def _nll_arrays(
    params: np.ndarray,
    n_teams: int,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    weights: np.ndarray,
) -> float:
    """Negative log-likelihood on pre-built numpy arrays (hot path)."""
    att = params[:n_teams]
    deff = params[n_teams:2 * n_teams]
    gamma = params[2 * n_teams]
    rho = params[2 * n_teams + 1]
    if rho < -0.5 or rho > 0.5:
        return 1e12

    lam_h = np.exp(att[home_idx] + deff[away_idx] + gamma)
    lam_a = np.exp(att[away_idx] + deff[home_idx])

    p_hg = poisson.pmf(home_goals, lam_h)
    p_ag = poisson.pmf(away_goals, lam_a)

    rho_clip = float(np.clip(rho, -0.49, 0.49))
    corr = np.ones(len(home_goals), dtype=float)
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)
    corr[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho_clip
    corr[m01] = 1.0 + lam_h[m01] * rho_clip
    corr[m10] = 1.0 + lam_a[m10] * rho_clip
    corr[m11] = 1.0 - rho_clip

    joint = np.maximum(p_hg * p_ag * corr, 1e-12)
    ll = float(np.sum(weights * np.log(joint)))
    constraint_penalty = float(att.sum()) ** 2 * 100.0
    return -ll + constraint_penalty


class DixonColes:
    """Single-league Dixon-Coles model.

    Usage:
        m = DixonColes(); m.fit(matches); proba = m.predict_proba(home, away)
    """
    def __init__(self, decay_xi: float = 0.0065, max_iter: int = 500):
        self.decay_xi = decay_xi
        self.max_iter = max_iter
        self.teams_: List[str] = []
        self.team_idx_: Dict[str, int] = {}
        self.params_: Optional[np.ndarray] = None    # [α_1..α_n, β_1..β_n, γ, ρ]
        self.fitted = False

    def _decode(self, params: np.ndarray):
        n = len(self.teams_)
        att = params[:n]
        deff = params[n:2 * n]
        gamma = params[2 * n]
        rho = params[2 * n + 1]
        return att, deff, gamma, rho

    def fit(
        self,
        matches: List[Dict],
        reference_date: Optional[str] = None,
        progress_label: str = "DC",
    ) -> "DixonColes":
        """matches: [{home, away, home_goals, away_goals, date(optional)}]"""
        # Discover teams
        seen: List[str] = []
        seen_set = set()
        for m in matches:
            for k in ("home", "away"):
                if m[k] not in seen_set:
                    seen.append(m[k])
                    seen_set.add(m[k])
        self.teams_ = seen
        self.team_idx_ = {t: i for i, t in enumerate(seen)}
        n = len(seen)

        if n < 2 or len(matches) < 10:
            self.fitted = False
            return self

        # Time-decay weights + pre-vectorize once (critical for L-BFGS speed)
        ref = _parse_date(reference_date) if reference_date else None
        home_idx_l: List[int] = []
        away_idx_l: List[int] = []
        home_goals_l: List[int] = []
        away_goals_l: List[int] = []
        weights_l: List[float] = []

        for m in matches:
            h = self.team_idx_.get(m["home"])
            a = self.team_idx_.get(m["away"])
            if h is None or a is None:
                continue
            if ref and self.decay_xi > 0:
                d = _parse_date(m.get("date", ""))
                if d:
                    days = max(0, (ref - d).days)
                    w = math.exp(-self.decay_xi * days)
                else:
                    w = 1.0
            else:
                w = float(m.get("weight", 1.0))
            home_idx_l.append(h)
            away_idx_l.append(a)
            home_goals_l.append(int(m["home_goals"]))
            away_goals_l.append(int(m["away_goals"]))
            weights_l.append(w)

        if len(home_idx_l) < 10:
            self.fitted = False
            return self

        home_idx = np.asarray(home_idx_l, dtype=np.int32)
        away_idx = np.asarray(away_idx_l, dtype=np.int32)
        home_goals = np.asarray(home_goals_l, dtype=np.int32)
        away_goals = np.asarray(away_goals_l, dtype=np.int32)
        weights = np.asarray(weights_l, dtype=np.float64)

        n_matches = len(home_idx)
        n_params = 2 * n + 2
        label = progress_label or "DC"
        t0 = time.time()
        print(
            f"      [{label}] optimize: {n_matches} matches, {n} teams, "
            f"{n_params} params (maxiter={self.max_iter})",
            flush=True,
        )

        state = {"nfev": 0, "last_print": t0, "last_nfev": 0}

        def nll(params: np.ndarray) -> float:
            state["nfev"] += 1
            now = time.time()
            # Heartbeat every ~15s so long International fits never look "hung"
            if now - state["last_print"] >= 15.0:
                elapsed = now - t0
                nfev = state["nfev"]
                rate = (nfev - state["last_nfev"]) / max(now - state["last_print"], 1e-6)
                print(
                    f"      [{label}] still optimizing... "
                    f"nfev={nfev} elapsed={elapsed:.0f}s "
                    f"(~{rate:.1f} eval/s) — NOT hung",
                    flush=True,
                )
                state["last_print"] = now
                state["last_nfev"] = nfev
            return _nll_arrays(
                params, n, home_idx, away_idx, home_goals, away_goals, weights
            )

        def _cb(_xk):
            # Called once per L-BFGS iteration (not every nfev)
            elapsed = time.time() - t0
            print(
                f"      [{label}] L-BFGS step done, nfev={state['nfev']}, "
                f"elapsed={elapsed:.0f}s",
                flush=True,
            )

        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.3, -0.1]])
        try:
            res = minimize(
                nll,
                x0,
                method="L-BFGS-B",
                callback=_cb,
                options={
                    "maxiter": self.max_iter,
                    "maxfun": 200000,
                    "disp": False,
                    "ftol": 1e-8,
                },
            )
            elapsed = time.time() - t0
            if not res.success:
                print(
                    f"      [{label}] optimizer warning: {res.message} "
                    f"(nfev={state['nfev']}, {elapsed:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"      [{label}] converged nfev={state['nfev']} in {elapsed:.1f}s",
                    flush=True,
                )
            if not np.all(np.isfinite(res.x)):
                print(f"      [{label}] non-finite parameters, discarding fit", flush=True)
                self.fitted = False
                return self
            self.params_ = res.x
            self.fitted = True
        except Exception as e:
            print(f"      [{label}] fit failed: {e}", flush=True)
            self.fitted = False
        return self

    def predict_proba(self, home: str, away: str) -> Tuple[float, float, float]:
        """Returns (p_home_win, p_draw, p_away_win)."""
        if not self.fitted:
            return (0.45, 0.27, 0.28)
        att, deff, gamma, rho = self._decode(self.params_)
        h = self.team_idx_.get(home)
        a = self.team_idx_.get(away)
        if h is None or a is None:
            return (0.45, 0.27, 0.28)
        lam_h = math.exp(att[h] + deff[a] + gamma)
        lam_a = math.exp(att[a] + deff[h])
        grid = _score_matrix(lam_h, lam_a, rho)
        p_home = float(np.tril(grid, -1).sum())
        p_draw = float(np.trace(grid))
        p_away = float(np.triu(grid, 1).sum())
        s = p_home + p_draw + p_away
        if s <= 0:
            return (0.45, 0.27, 0.28)
        return p_home / s, p_draw / s, p_away / s

    def expected_goals(self, home: str, away: str) -> Tuple[float, float]:
        if not self.fitted:
            return (1.3, 1.1)
        att, deff, gamma, _ = self._decode(self.params_)
        h, a = self.team_idx_.get(home), self.team_idx_.get(away)
        if h is None or a is None:
            return (1.3, 1.1)
        return math.exp(att[h] + deff[a] + gamma), math.exp(att[a] + deff[h])


def _parse_date(s: str) -> Optional[dt.date]:
    try:
        return dt.datetime.fromisoformat(str(s).replace(" ", "T")).date()
    except Exception:
        return None
