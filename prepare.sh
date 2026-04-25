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

team_meta = {
    "ARI": ("NL", "NL West"), "ATL": ("NL", "NL East"), "BAL": ("AL", "AL East"),
    "BOS": ("AL", "AL East"), "CHC": ("NL", "NL Central"), "CHW": ("AL", "AL Central"),
    "CIN": ("NL", "NL Central"), "CLE": ("AL", "AL Central"), "COL": ("NL", "NL West"),
    "DET": ("AL", "AL Central"), "HOU": ("AL", "AL West"), "KC": ("AL", "AL Central"),
    "LAA": ("AL", "AL West"), "LAD": ("NL", "NL West"), "MIA": ("NL", "NL East"),
    "MIL": ("NL", "NL Central"), "MIN": ("AL", "AL Central"), "NYM": ("NL", "NL East"),
    "NYY": ("AL", "AL East"), "OAK": ("AL", "AL West"), "PHI": ("NL", "NL East"),
    "PIT": ("NL", "NL Central"), "SD": ("NL", "NL West"), "SEA": ("AL", "AL West"),
    "SF": ("NL", "NL West"), "STL": ("NL", "NL Central"), "TB": ("AL", "AL East"),
    "TEX": ("AL", "AL West"), "TOR": ("AL", "AL East"), "WSH": ("NL", "NL East"),
}

header = [
    "season","checkpoint","team_id","team_name","league","division","dc_team_war","steamer_team_war",
    "zips_team_war","projection_blend_war","pos_war","sp_war","rp_war",
    "sp1_war","sp2_war","sp3_war","sp4_war","sp5_war","sp6_war","sp7_war",
    "bp_high_lev_war","rotation_depth_war","catcher_framing_runs",
    "drs_runs","uzr_runs","oaa_runs","def_blend_runs",
    "schedule_strength","division_strength","intradivision_schedule_strength",
    "park_factor","park_factor_3yr","park_factor_1yr","injury_war_lost",
    "durability_risk","age_risk","pythag_win_pct","baseruns_win_pct",
    "third_order_win_pct","prev_win_pct","checkpoint_wins_above_pace",
    "actual_wins","overall_rank","league_rank","division_rank","made_playoffs",
    "won_division","league_champion","world_series_champion",
]

player_header = [
    "season","checkpoint","team_id","player_id","player_name","age","throws_bats",
    "position","role","proj_pa","proj_ip","prev_pa","prev_ip","steamer_war",
    "zips_war","dc_war","projection_blend_war","woba","xwoba","barrel_pct",
    "hard_hit_pct","exit_velocity","chase_pct","stuff_plus","location_plus",
    "projected_leverage_index","framing_runs","drs_runs","uzr_runs","oaa_runs",
    "def_runs","durability_score","injury_risk",
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
    sp_weights = [1.25, 1.08, 0.93, 0.78, 0.62, 0.45, 0.32]
    sp_parts = []
    for weight in sp_weights:
        sp_parts.append(max(0.0, sp * weight / sum(sp_weights) + rng.gauss(0, 0.22)))
    schedule = rng.gauss(0, 2.0) + 0.8 * math.sin(idx)
    division = schedule + rng.gauss(0, 1.0)
    intradivision = division + rng.gauss(0, 0.8)
    injury = max(0.0, rng.gauss(2.7, 1.6))
    durability = max(0.0, rng.gauss(1.0, 0.4) + injury / 8.0)
    age_risk = max(0.0, rng.gauss(1.0, 0.55))
    pythag = 0.500 + (base - 31.0) / 260.0 + rng.gauss(0, 0.018)
    baseruns = 0.500 + (base - 31.0) / 250.0 + rng.gauss(0, 0.020)
    third_order = 0.500 + (base - 31.0) / 245.0 + rng.gauss(0, 0.017)
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
    league, division_name = team_meta[team]
    drs = rng.gauss(0, 14.0)
    uzr = drs * 0.55 + rng.gauss(0, 10.0)
    oaa = drs * 0.45 + rng.gauss(0, 11.0)
    return {
        "season": season,
        "checkpoint": checkpoint,
        "team_id": team,
        "team_name": f"{team} Baseball Club",
        "league": league,
        "division": division_name,
        "dc_team_war": dc,
        "steamer_team_war": steamer,
        "zips_team_war": zips,
        "projection_blend_war": (dc + steamer + zips) / 3.0,
        "pos_war": pos,
        "sp_war": sp,
        "rp_war": rp,
        "sp1_war": sp_parts[0],
        "sp2_war": sp_parts[1],
        "sp3_war": sp_parts[2],
        "sp4_war": sp_parts[3],
        "sp5_war": sp_parts[4],
        "sp6_war": sp_parts[5],
        "sp7_war": sp_parts[6],
        "bp_high_lev_war": max(0.0, 0.42 * rp + rng.gauss(0, 0.35)),
        "rotation_depth_war": max(0.0, 0.72 * sp + rng.gauss(0, 0.8)),
        "catcher_framing_runs": rng.gauss(0, 5.5),
        "drs_runs": drs,
        "uzr_runs": uzr,
        "oaa_runs": oaa,
        "def_blend_runs": (drs + uzr + oaa) / 3.0,
        "schedule_strength": schedule,
        "division_strength": division,
        "intradivision_schedule_strength": intradivision,
        "park_factor": park[team],
        "park_factor_3yr": park[team],
        "park_factor_1yr": park[team] + rng.gauss(0, 4.0),
        "injury_war_lost": injury,
        "durability_risk": durability,
        "age_risk": age_risk,
        "pythag_win_pct": max(0.36, min(0.64, pythag)),
        "baseruns_win_pct": max(0.36, min(0.64, baseruns)),
        "third_order_win_pct": max(0.36, min(0.64, third_order)),
        "prev_win_pct": max(0.35, min(0.65, prev)),
        "checkpoint_wins_above_pace": pace,
        "actual_wins": round(true_wins, 1),
        "made_playoffs": made,
        "won_division": 0,
        "league_champion": 0,
        "world_series_champion": 0,
        "overall_rank": 0,
        "league_rank": 0,
        "division_rank": 0,
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
            prev_ip = max(0.0, min(220.0, proj_ip + rng.gauss(0, 32.0)))
            prev_pa = 0.0
        elif role == "reliever":
            base_war = rp_budget * max(0.03, 0.72 - 0.045 * (slot - 20)) / 3.8 + rng.gauss(0, 0.18)
            proj_ip = max(25.0, min(85.0, rng.gauss(58.0, 10.0)))
            proj_pa = 0.0
            prev_ip = max(0.0, min(95.0, proj_ip + rng.gauss(0, 16.0)))
            prev_pa = 0.0
        elif role == "callup":
            base_war = rng.gauss(0.15, 0.25)
            proj_ip = max(0.0, rng.gauss(20.0, 20.0)) if rng.random() < 0.35 else 0.0
            proj_pa = 0.0 if proj_ip else max(40.0, rng.gauss(135.0, 45.0))
            prev_ip = max(0.0, proj_ip + rng.gauss(0, 18.0)) if proj_ip else 0.0
            prev_pa = max(0.0, proj_pa + rng.gauss(0, 65.0)) if proj_pa else 0.0
        else:
            base_war = pos_budget * max(0.04, 1.05 - 0.065 * slot) / 6.4 + rng.gauss(0, 0.30)
            proj_pa = max(120.0, min(690.0, rng.gauss(560.0 if role == "position_player" else 280.0, 85.0)))
            proj_ip = 0.0
            prev_pa = max(0.0, min(720.0, proj_pa + rng.gauss(0, 90.0)))
            prev_ip = 0.0
        base_war = max(-0.8, base_war)
        steamer_war = base_war + rng.gauss(0, 0.20)
        zips_war = base_war + rng.gauss(0, 0.22)
        dc_war = base_war + rng.gauss(0, 0.18)
        drs = 0.0 if is_pitcher else rng.gauss(float(team_row["def_blend_runs"]) / 9.0, 4.0)
        uzr = 0.0 if is_pitcher else drs * 0.55 + rng.gauss(0, 2.6)
        oaa = 0.0 if is_pitcher else drs * 0.45 + rng.gauss(0, 2.8)
        durability_score = max(0.05, min(1.0, (prev_ip / 190.0 if is_pitcher else prev_pa / 650.0)))
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
            "prev_pa": prev_pa,
            "prev_ip": prev_ip,
            "steamer_war": steamer_war,
            "zips_war": zips_war,
            "dc_war": dc_war,
            "projection_blend_war": (steamer_war + zips_war + dc_war) / 3.0,
            "woba": 0.0 if is_pitcher else max(0.240, min(0.440, rng.gauss(0.318 + base_war * 0.009, 0.035))),
            "xwoba": 0.0 if is_pitcher else max(0.250, min(0.430, rng.gauss(0.322 + base_war * 0.010, 0.025))),
            "barrel_pct": 0.0 if is_pitcher else max(1.0, min(20.0, rng.gauss(8.5 + base_war * 0.7, 2.8))),
            "hard_hit_pct": 0.0 if is_pitcher else max(25.0, min(60.0, rng.gauss(40.0 + base_war * 1.1, 5.0))),
            "exit_velocity": 0.0 if is_pitcher else max(84.0, min(96.0, rng.gauss(89.0 + base_war * 0.35, 1.8))),
            "chase_pct": 0.0 if is_pitcher else max(18.0, min(42.0, rng.gauss(29.0 - base_war * 0.25, 3.5))),
            "stuff_plus": 0.0 if not is_pitcher else max(78.0, min(128.0, rng.gauss(100.0 + base_war * 2.3, 8.5))),
            "location_plus": 0.0 if not is_pitcher else max(82.0, min(122.0, rng.gauss(100.0 + base_war * 1.1, 6.0))),
            "projected_leverage_index": max(0.0, min(2.2, rng.gauss(1.15, 0.35))) if role == "reliever" else 0.0,
            "framing_runs": rng.gauss(float(team_row["catcher_framing_runs"]) / 2.0, 2.0) if position == "C" else 0.0,
            "drs_runs": drs,
            "uzr_runs": uzr,
            "oaa_runs": oaa,
            "def_runs": (drs + uzr + oaa) / 3.0,
            "durability_score": durability_score,
            "injury_risk": max(0.01, min(0.65, rng.gauss(float(team_row["injury_war_lost"]) / 20.0, 0.08))),
        })
    return players


def assign_outcomes(rows: list[dict]) -> None:
    groups: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((int(row["season"]), str(row["checkpoint"])), []).append(row)
    for group_rows in groups.values():
        overall = sorted(group_rows, key=lambda r: (-float(r["actual_wins"]), str(r["team_id"])))
        for rank, row in enumerate(overall, start=1):
            row["overall_rank"] = rank
            row["world_series_champion"] = int(rank == 1)
        for league in ("AL", "NL"):
            league_rows = [row for row in overall if row["league"] == league]
            for rank, row in enumerate(league_rows, start=1):
                row["league_rank"] = rank
                row["league_champion"] = int(rank == 1)
                row["made_playoffs"] = int(rank <= 6)
        divisions = sorted({row["division"] for row in group_rows})
        for division_name in divisions:
            division_rows = sorted(
                [row for row in group_rows if row["division"] == division_name],
                key=lambda r: (-float(r["actual_wins"]), str(r["team_id"])),
            )
            for rank, row in enumerate(division_rows, start=1):
                row["division_rank"] = rank
                row["won_division"] = int(rank == 1)


def write(team_path: str, player_path: str, seasons: list[int], checkpoints: list[str]):
    rows = []
    player_rows = []
    for season in seasons:
        for checkpoint in checkpoints:
            for idx, team in enumerate(teams):
                row = make_row(season, team, idx, checkpoint)
                rows.append(row)
                player_rows.extend(make_players(season, team, idx, checkpoint, row))
    assign_outcomes(rows)
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
