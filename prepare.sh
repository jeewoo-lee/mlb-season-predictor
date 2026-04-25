#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p data/train data/val eval/test_data

python - <<'PY'
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

teams = [
    "ARI","ATL","BAL","BOS","CHC","CHW","CIN","CLE","COL","DET",
    "HOU","KC","LAA","LAD","MIA","MIL","MIN","NYM","NYY","OAK",
    "PHI","PIT","SD","SEA","SF","STL","TB","TEX","TOR","WSH",
]

header = [
    "season","checkpoint","team_id","team_name","dc_team_war","steamer_team_war",
    "zips_team_war","pos_war","sp_war","rp_war","bp_high_lev_war",
    "rotation_depth_war","catcher_framing_runs","def_blend_runs",
    "schedule_strength","division_strength","park_factor","injury_war_lost",
    "age_risk","baseruns_win_pct","prev_win_pct","checkpoint_wins_above_pace",
    "actual_wins","made_playoffs",
]

strength = {
    team: 31.0 + 8.0 * math.sin((i + 1) * 1.73) + 4.5 * math.cos((i + 3) * 0.61)
    for i, team in enumerate(teams)
}
park = {team: 100.0 + 7.0 * math.sin(i * 0.9) for i, team in enumerate(teams)}


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def make_row(season: int, team: str, idx: int, checkpoint: str):
    rng = random.Random(season * 1000 + idx * 37 + (13 if checkpoint == "all_star" else 0))
    trend = 2.7 * math.sin((season - 2008) * 0.65 + idx * 0.37)
    base = strength[team] + trend + rng.gauss(0, 2.9)
    dc = base + rng.gauss(0, 2.4)
    steamer = base + rng.gauss(0, 2.2)
    zips = base + rng.gauss(0, 2.6)
    pos = 0.52 * base + rng.gauss(0, 1.7)
    sp = 0.31 * base + rng.gauss(0, 1.5)
    rp = 0.17 * base + rng.gauss(0, 0.9)
    schedule = rng.gauss(0, 2.0) + 0.8 * math.sin(idx)
    division = schedule + rng.gauss(0, 1.0)
    injury = max(0.0, rng.gauss(2.7, 1.6))
    age_risk = max(0.0, rng.gauss(1.0, 0.55))
    baseruns = 0.500 + (base - 31.0) / 250.0 + rng.gauss(0, 0.020)
    prev = 0.500 + (base - 31.0) / 310.0 + rng.gauss(0, 0.030)
    pace = 0.0
    if checkpoint == "all_star":
        pace = rng.gauss(0, 4.0) + (base - 32.0) * 0.10
        dc += 0.35 * pace
    true_wins = (
        48.0 + base + 5.5 * (baseruns - 0.5) * 162.0
        - 0.7 * injury - 0.25 * schedule + 0.55 * pace + rng.gauss(0, 6.2)
    )
    true_wins = max(50.0, min(110.0, true_wins))
    playoff_p = sigmoid(-1.02 + 0.145 * (true_wins - 84.0))
    made = int(rng.random() < playoff_p)
    return {
        "season": season,
        "checkpoint": checkpoint,
        "team_id": team,
        "team_name": f"{team} Baseball Club",
        "dc_team_war": dc,
        "steamer_team_war": steamer,
        "zips_team_war": zips,
        "pos_war": pos,
        "sp_war": sp,
        "rp_war": rp,
        "bp_high_lev_war": max(0.0, 0.42 * rp + rng.gauss(0, 0.35)),
        "rotation_depth_war": max(0.0, 0.72 * sp + rng.gauss(0, 0.8)),
        "catcher_framing_runs": rng.gauss(0, 5.5),
        "def_blend_runs": rng.gauss(0, 13.0),
        "schedule_strength": schedule,
        "division_strength": division,
        "park_factor": park[team],
        "injury_war_lost": injury,
        "age_risk": age_risk,
        "baseruns_win_pct": max(0.36, min(0.64, baseruns)),
        "prev_win_pct": max(0.35, min(0.65, prev)),
        "checkpoint_wins_above_pace": pace,
        "actual_wins": round(true_wins, 1),
        "made_playoffs": made,
    }


def write(path: str, seasons: list[int], checkpoints: list[str]):
    rows = []
    for season in seasons:
        for checkpoint in checkpoints:
            for idx, team in enumerate(teams):
                rows.append(make_row(season, team, idx, checkpoint))
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
    return len(rows)


train_n = write("data/train/team_states.csv", list(range(2010, 2023)), ["opening_day"])
val_n = write("data/val/team_states.csv", [2023], ["opening_day", "all_star"])
test_n = write("eval/test_data/mlb_frozen_2024_2025.csv", [2024, 2025], ["opening_day", "all_star"])

print(f"prepared train={train_n} val={val_n} frozen_test={test_n}")
PY
