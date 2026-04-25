"""Feature helpers for the starter MLB season predictor.

Agents are expected to replace or extend these helpers with richer baseball
features. Keep eval-time code self-contained: no network calls here.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean


NUMERIC_FIELDS = (
    "dc_team_war",
    "steamer_team_war",
    "zips_team_war",
    "pos_war",
    "sp_war",
    "rp_war",
    "bp_high_lev_war",
    "rotation_depth_war",
    "catcher_framing_runs",
    "def_blend_runs",
    "schedule_strength",
    "division_strength",
    "park_factor",
    "injury_war_lost",
    "age_risk",
    "baseruns_win_pct",
    "prev_win_pct",
    "actual_wins",
    "made_playoffs",
)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def coerce_row(row: dict) -> dict:
    out = dict(row)
    for field in NUMERIC_FIELDS:
        if field in out and out[field] != "":
            out[field] = float(out[field])
    if "season" in out:
        out["season"] = int(float(out["season"]))
    return out


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        return [coerce_row(row) for row in csv.DictReader(f)]


def projection_blend(row: dict) -> float:
    values = [
        float(row["dc_team_war"]),
        float(row["steamer_team_war"]),
        float(row["zips_team_war"]),
    ]
    return mean(values)


def baseline_projected_wins(row: dict) -> float:
    """Weak preseason-style wins estimate from projection WAR and context."""

    wins = 48.0 + float(row["dc_team_war"])
    wins += 5.0 * (float(row.get("prev_win_pct", 0.5)) - 0.5)
    wins += 0.9 * (float(row.get("baseruns_win_pct", 0.5)) - 0.5) * 162.0
    wins -= 0.45 * float(row.get("injury_war_lost", 0.0))
    wins -= 0.15 * float(row.get("schedule_strength", 0.0))
    if row.get("checkpoint") == "all_star":
        wins += 0.9 * float(row.get("checkpoint_wins_above_pace", 0.0))
    return clamp(wins, 48.0, 108.0)
