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

player_header = [
    "season","checkpoint","team_id","player_id","player_name","age","throws_bats",
    "position","role","proj_pa","proj_ip","steamer_war","zips_war","dc_war",
    "xwoba","barrel_pct","hard_hit_pct","exit_velocity","chase_pct",
    "stuff_plus","location_plus","framing_runs","def_runs","injury_risk",
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


def make_players(season: int, team: str, idx: int, checkpoint: str, team_row: dict):
    rng = random.Random(season * 2000 + idx * 97 + (29 if checkpoint == "all_star" else 0))
    players = []
    roles = (
        [("C", "position_player"), ("1B", "position_player"), ("2B", "position_player"),
         ("3B", "position_player"), ("SS", "position_player"), ("LF", "position_player"),
         ("CF", "position_player"), ("RF", "position_player"), ("DH", "position_player")]
        + [("OF", "bench"), ("IF", "bench"), ("C", "bench"), ("UT", "bench")]
        + [(f"SP{i}", "starter") for i in range(1, 8)]
        + [(f"RP{i}", "reliever") for i in range(1, 10)]
        + [("CALL", "callup") for _ in range(4)]
    )
    pos_budget = max(5.0, float(team_row["pos_war"]))
    sp_budget = max(3.0, float(team_row["sp_war"]))
    rp_budget = max(1.0, float(team_row["rp_war"]))
    for slot, (position, role) in enumerate(roles, start=1):
        is_pitcher = role in {"starter", "reliever"}
        age = max(20.0, min(39.0, rng.gauss(28.2 if not is_pitcher else 28.8, 3.4)))
        if role == "starter":
            base_war = sp_budget * max(0.05, 1.25 - 0.13 * slot) / 4.9 + rng.gauss(0, 0.35)
            proj_ip = max(45.0, min(195.0, rng.gauss(145.0 - 8.0 * max(0, slot - 1), 18.0)))
            proj_pa = 0.0
        elif role == "reliever":
            base_war = rp_budget * max(0.03, 0.72 - 0.045 * (slot - 20)) / 3.8 + rng.gauss(0, 0.18)
            proj_ip = max(25.0, min(85.0, rng.gauss(58.0, 10.0)))
            proj_pa = 0.0
        elif role == "callup":
            base_war = rng.gauss(0.15, 0.25)
            proj_ip = max(0.0, rng.gauss(20.0, 20.0)) if rng.random() < 0.35 else 0.0
            proj_pa = 0.0 if proj_ip else max(40.0, rng.gauss(135.0, 45.0))
        else:
            base_war = pos_budget * max(0.04, 1.05 - 0.065 * slot) / 6.4 + rng.gauss(0, 0.30)
            proj_pa = max(120.0, min(690.0, rng.gauss(560.0 if role == "position_player" else 280.0, 85.0)))
            proj_ip = 0.0
        base_war = max(-0.8, base_war)
        player_id = f"{team}{season}{checkpoint[:1]}{slot:02d}"
        players.append({
            "season": season,
            "checkpoint": checkpoint,
            "team_id": team,
            "player_id": player_id,
            "player_name": f"{team} Player {slot:02d}",
            "age": age,
            "throws_bats": rng.choice(["R/R", "L/L", "R/L", "L/R", "S/R"]),
            "position": position,
            "role": role,
            "proj_pa": proj_pa,
            "proj_ip": proj_ip,
            "steamer_war": base_war + rng.gauss(0, 0.20),
            "zips_war": base_war + rng.gauss(0, 0.22),
            "dc_war": base_war + rng.gauss(0, 0.18),
            "xwoba": 0.0 if is_pitcher else max(0.250, min(0.430, rng.gauss(0.322 + base_war * 0.010, 0.025))),
            "barrel_pct": 0.0 if is_pitcher else max(1.0, min(20.0, rng.gauss(8.5 + base_war * 0.7, 2.8))),
            "hard_hit_pct": 0.0 if is_pitcher else max(25.0, min(60.0, rng.gauss(40.0 + base_war * 1.1, 5.0))),
            "exit_velocity": 0.0 if is_pitcher else max(84.0, min(96.0, rng.gauss(89.0 + base_war * 0.35, 1.8))),
            "chase_pct": 0.0 if is_pitcher else max(18.0, min(42.0, rng.gauss(29.0 - base_war * 0.25, 3.5))),
            "stuff_plus": 0.0 if not is_pitcher else max(78.0, min(128.0, rng.gauss(100.0 + base_war * 2.3, 8.5))),
            "location_plus": 0.0 if not is_pitcher else max(82.0, min(122.0, rng.gauss(100.0 + base_war * 1.1, 6.0))),
            "framing_runs": rng.gauss(float(team_row["catcher_framing_runs"]) / 2.0, 2.0) if position == "C" else 0.0,
            "def_runs": 0.0 if is_pitcher else rng.gauss(float(team_row["def_blend_runs"]) / 9.0, 3.5),
            "injury_risk": max(0.01, min(0.65, rng.gauss(float(team_row["injury_war_lost"]) / 20.0, 0.08))),
        })
    return players


def write(team_path: str, player_path: str, seasons: list[int], checkpoints: list[str]):
    rows = []
    player_rows = []
    for season in seasons:
        for checkpoint in checkpoints:
            for idx, team in enumerate(teams):
                row = make_row(season, team, idx, checkpoint)
                rows.append(row)
                player_rows.extend(make_players(season, team, idx, checkpoint, row))
    with Path(team_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
    with Path(player_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=player_header)
        writer.writeheader()
        for row in player_rows:
            writer.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in row.items()})
    return len(rows), len(player_rows)


train_n, train_p = write("data/train/team_states.csv", "data/train/player_states.csv", list(range(2010, 2023)), ["opening_day"])
val_n, val_p = write("data/val/team_states.csv", "data/val/player_states.csv", [2023], ["opening_day", "all_star"])
test_n, test_p = write("eval/test_data/mlb_frozen_2024_2025.csv", "eval/test_data/player_states_2024_2025.csv", [2024, 2025], ["opening_day", "all_star"])

print(f"prepared train={train_n}/{train_p} players val={val_n}/{val_p} players frozen_test={test_n}/{test_p} players")
PY
