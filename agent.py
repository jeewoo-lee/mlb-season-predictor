"""Starter predictor for the MLB Season Predictor hive task.

This baseline is intentionally modest: it maps Depth Charts-style team WAR to
projected wins and playoff probability. It does not use a teacher LLM,
distillation, player-level Monte Carlo, Statcast, or roster tables.
"""

from __future__ import annotations

import math
from pathlib import Path

from features import baseline_projected_wins, clamp, load_rows, sigmoid


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train" / "team_states.csv"


def _fit_war_logistic() -> tuple[float, float]:
    """Fit a one-feature logistic curve on 2010-2022 training data.

    Tiny gradient descent keeps the starter independent of scikit-learn at eval
    time, even though agents may add sklearn pipelines later.
    """

    rows = load_rows(TRAIN_PATH)
    intercept = -0.45
    slope = 0.095
    lr = 0.015
    n = max(1, len(rows))
    for _ in range(900):
        gi = 0.0
        gs = 0.0
        for row in rows:
            x = float(row["dc_team_war"]) - 33.0
            y = float(row["made_playoffs"])
            p = sigmoid(intercept + slope * x)
            gi += p - y
            gs += (p - y) * x
        intercept -= lr * gi / n
        slope -= lr * gs / n
    return intercept, slope


_INTERCEPT: float | None = None
_SLOPE: float | None = None


def _coefficients() -> tuple[float, float]:
    global _INTERCEPT, _SLOPE
    if _INTERCEPT is None or _SLOPE is None:
        _INTERCEPT, _SLOPE = _fit_war_logistic()
    return _INTERCEPT, _SLOPE


def predict(team_state: dict) -> dict:
    intercept, slope = _coefficients()
    dc_war = float(team_state["dc_team_war"])
    projected_wins = baseline_projected_wins(team_state)

    logit = intercept + slope * (dc_war - 33.0)
    if team_state.get("checkpoint") == "all_star":
        logit += 0.10 * float(team_state.get("checkpoint_wins_above_pace", 0.0))

    # Deliberately underconfident starter calibration. This keeps the baseline
    # functional while leaving room for richer roster and simulation methods.
    playoff_prob = clamp(sigmoid(0.10 * logit), 0.03, 0.97)
    spread = 8.5 + 0.05 * abs(projected_wins - 81.0)
    return {
        "playoff_prob": playoff_prob,
        "projected_wins": projected_wins,
        "win_interval_80": [
            clamp(projected_wins - spread, 40.0, 116.0),
            clamp(projected_wins + spread, 46.0, 122.0),
        ],
    }
