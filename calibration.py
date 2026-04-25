"""Post-hoc calibration of grok predictions.

Trains a small linear model `actual_wins ~ features` on bundled train/val data
and exposes a blend with the grok projected wins. Probability calibration
shrinks per-team prior probabilities toward checkpoint-aware base rates so the
log-loss components don't blow up on confident misses.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from features import load_rows


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train" / "team_states.csv"
VAL_PATH = ROOT / "data" / "val" / "team_states.csv"

WIN_FEATURES: tuple[str, ...] = (
    "dc_team_war",
    "steamer_team_war",
    "zips_team_war",
    "projection_blend_war",
    "pos_war",
    "sp_war",
    "rp_war",
    "bp_high_lev_war",
    "rotation_depth_war",
    "catcher_framing_runs",
    "def_blend_runs",
    "schedule_strength",
    "intradivision_schedule_strength",
    "park_factor_3yr",
    "injury_war_lost",
    "durability_risk",
    "age_risk",
    "pythag_win_pct",
    "baseruns_win_pct",
    "third_order_win_pct",
    "prev_win_pct",
    "checkpoint_wins_above_pace",
)


def _vec(row: dict) -> list[float]:
    return [float(row.get(f, 0.0) or 0.0) for f in WIN_FEATURES]


def _ridge_fit(rows: list[dict], lam: float) -> list[float]:
    """Closed-form ridge regression in pure-python (avoids numpy dependency)."""
    n = len(rows)
    if n == 0:
        return [81.0] + [0.0] * len(WIN_FEATURES)
    p = len(WIN_FEATURES)
    # Build X (n x (p+1)) with bias column
    X = [[1.0] + _vec(r) for r in rows]
    y = [float(r["actual_wins"]) for r in rows]
    # Solve (X^T X + lam*I) b = X^T y, but don't regularize bias.
    dim = p + 1
    XtX = [[0.0] * dim for _ in range(dim)]
    Xty = [0.0] * dim
    for xi, yi in zip(X, y):
        for i in range(dim):
            xi_i = xi[i]
            Xty[i] += xi_i * yi
            row_i = XtX[i]
            for j in range(dim):
                row_i[j] += xi_i * xi[j]
    for i in range(1, dim):
        XtX[i][i] += lam
    return _solve(XtX, Xty)


def _solve(A: list[list[float]], b: list[float]) -> list[float]:
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for i in range(n):
        pivot = i
        for k in range(i + 1, n):
            if abs(M[k][i]) > abs(M[pivot][i]):
                pivot = k
        if pivot != i:
            M[i], M[pivot] = M[pivot], M[i]
        if abs(M[i][i]) < 1e-12:
            M[i][i] = 1e-12
        inv = 1.0 / M[i][i]
        for j in range(i, n + 1):
            M[i][j] *= inv
        for k in range(n):
            if k == i:
                continue
            factor = M[k][i]
            if factor == 0.0:
                continue
            for j in range(i, n + 1):
                M[k][j] -= factor * M[i][j]
    return [M[i][n] for i in range(n)]


def _train_rows(checkpoint: str) -> list[dict]:
    """Pool any rows of the requested checkpoint from train + val."""
    pool: list[dict] = []
    for path in (TRAIN_PATH, VAL_PATH):
        if not path.exists():
            continue
        for r in load_rows(path):
            if r.get("checkpoint") == checkpoint and "actual_wins" in r:
                pool.append(r)
    return pool


_COEF_CACHE: dict[str, list[float]] = {}


def _coef(checkpoint: str) -> list[float]:
    if checkpoint not in _COEF_CACHE:
        rows = _train_rows(checkpoint)
        # All-star has only ~30 samples (val only), so regularize harder.
        lam = 8.0 if checkpoint == "opening_day" else 25.0
        _COEF_CACHE[checkpoint] = _ridge_fit(rows, lam)
    return _COEF_CACHE[checkpoint]


def linreg_projected_wins(team_state: dict) -> float:
    coef = _coef(str(team_state.get("checkpoint", "opening_day")))
    bias = coef[0]
    val = bias
    for c, name in zip(coef[1:], WIN_FEATURES):
        val += c * float(team_state.get(name, 0.0) or 0.0)
    if val < 40.0:
        val = 40.0
    if val > 122.0:
        val = 122.0
    return val


def blend_wins(grok_wins: float, team_state: dict, alpha: float = 0.7) -> float:
    """Convex blend of grok and ridge prediction; alpha is grok weight."""
    lr = linreg_projected_wins(team_state)
    return alpha * grok_wins + (1.0 - alpha) * lr
