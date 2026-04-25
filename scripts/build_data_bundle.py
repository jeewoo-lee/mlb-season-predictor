"""Build the MLB Season Predictor data bundle (one-time, task author).

Pulls real MLB data 2010-2025 from the official MLB Stats API
(https://statsapi.mlb.com — public, no auth), derives the feature schema
agent.py / features.py / eval.py expect, anonymizes identifiers with a fixed
seed, writes CSVs into dist/mlb_season_data/, and zips them.

Why this script exists:
  - The published data ships as a release artifact; agents download it via
    prepare.sh. This script is the one place that ever scrapes/pulls.
  - Schema must match the original prepare.sh exactly so eval.py and the
    starter agent.py keep working without code changes.

Usage:
    python scripts/build_data_bundle.py                  # all years
    python scripts/build_data_bundle.py --years 2023 2024  # subset
    python scripts/build_data_bundle.py --skip-pull       # use cached raw

Output:
    dist/mlb_season_data/train/{team_states,player_states}.csv
    dist/mlb_season_data/val/{team_states,player_states}.csv
    dist/mlb_season_data/frozen/{frozen_test,frozen_test_players}.csv
    dist/mlb_season_data_v1.zip            <- upload this as a GitHub release
    eval/.identity_map.json                <- private; keep out of zip + git
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DIST = ROOT / "dist"
STAGING = DIST / "mlb_season_data"
ZIP_PATH = DIST / "mlb_season_data_v1.zip"
IDENTITY_MAP_PATH = ROOT / "eval" / ".identity_map.json"
ANON_SEED = 20260425

API = "https://statsapi.mlb.com/api/v1"
LEAGUE_AL, LEAGUE_NL = 103, 104

DIVISIONS = {
    200: "AL West", 201: "AL East", 202: "AL Central",
    203: "NL West", 204: "NL East", 205: "NL Central",
}

# Schema must match the original prepare.sh exactly.
TEAM_COLUMNS = [
    "season", "checkpoint", "team_id", "team_name", "league", "division",
    "dc_team_war", "steamer_team_war", "zips_team_war", "projection_blend_war",
    "pos_war", "sp_war", "rp_war",
    "sp1_war", "sp2_war", "sp3_war", "sp4_war", "sp5_war", "sp6_war", "sp7_war",
    "bp_high_lev_war", "rotation_depth_war", "catcher_framing_runs",
    "drs_runs", "uzr_runs", "oaa_runs", "def_blend_runs",
    "schedule_strength", "division_strength", "intradivision_schedule_strength",
    "park_factor", "park_factor_3yr", "park_factor_1yr",
    "injury_war_lost", "durability_risk", "age_risk",
    "pythag_win_pct", "baseruns_win_pct", "third_order_win_pct",
    "prev_win_pct", "checkpoint_wins_above_pace",
    "actual_wins", "overall_rank", "league_rank", "division_rank",
    "made_playoffs", "won_division", "league_champion", "world_series_champion",
]

PLAYER_COLUMNS = [
    "season", "checkpoint", "team_id", "player_id", "player_name", "age",
    "throws_bats", "position", "role",
    "proj_pa", "proj_ip", "prev_pa", "prev_ip",
    "steamer_war", "zips_war", "dc_war", "projection_blend_war",
    "woba", "xwoba", "barrel_pct", "hard_hit_pct", "exit_velocity",
    "chase_pct", "stuff_plus", "location_plus",
    "projected_leverage_index", "framing_runs",
    "drs_runs", "uzr_runs", "oaa_runs", "def_runs",
    "durability_score", "injury_risk",
]

# Park factors (rough, 3-year averages from public sources). Anonymization
# happens later, so real park identity is fine here.
PARK_FACTORS = {
    "ARI": 102, "ATL": 100, "BAL": 102, "BOS": 105, "CHC": 100, "CHW": 101,
    "CIN": 105, "CLE": 99, "COL": 116, "DET": 96, "HOU": 100, "KC": 103,
    "LAA": 99, "LAD": 99, "MIA": 96, "MIL": 102, "MIN": 100, "NYM": 96,
    "NYY": 102, "OAK": 96, "PHI": 102, "PIT": 99, "SD": 96, "SEA": 95,
    "SF": 96, "STL": 99, "TB": 96, "TEX": 105, "TOR": 102, "WSH": 100,
}

# Rough postseason results 2010-2025 used for league_champion / world_series_champion.
# League champion = pennant winner.
POSTSEASON = {
    2010: {"al": "TEX", "nl": "SF",  "ws": "SF"},
    2011: {"al": "TEX", "nl": "STL", "ws": "STL"},
    2012: {"al": "DET", "nl": "SF",  "ws": "SF"},
    2013: {"al": "BOS", "nl": "STL", "ws": "BOS"},
    2014: {"al": "KC",  "nl": "SF",  "ws": "SF"},
    2015: {"al": "KC",  "nl": "NYM", "ws": "KC"},
    2016: {"al": "CLE", "nl": "CHC", "ws": "CHC"},
    2017: {"al": "HOU", "nl": "LAD", "ws": "HOU"},  # HOU title later vacated; keep recorded outcome
    2018: {"al": "BOS", "nl": "LAD", "ws": "BOS"},
    2019: {"al": "HOU", "nl": "WSH", "ws": "WSH"},
    2020: {"al": "TB",  "nl": "LAD", "ws": "LAD"},
    2021: {"al": "HOU", "nl": "ATL", "ws": "ATL"},
    2022: {"al": "HOU", "nl": "PHI", "ws": "HOU"},
    2023: {"al": "TEX", "nl": "ARI", "ws": "TEX"},
    2024: {"al": "NYY", "nl": "LAD", "ws": "LAD"},
    2025: {"al": None,  "nl": None,  "ws": None},  # incomplete season as of build time
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def get(path: str, **params: Any) -> dict:
    """GET JSON from statsapi with retry/backoff."""
    url = f"{API}/{path}"
    last = None
    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError(f"GET {url} {params} failed: {last}")


def cache_path(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    return RAW / name


def cached_get(cache_name: str, path: str, **params: Any) -> dict:
    """GET with disk cache; raw pulls live in data/raw/ (gitignored)."""
    cp = cache_path(cache_name)
    if cp.exists():
        return json.loads(cp.read_text())
    js = get(path, **params)
    cp.write_text(json.dumps(js))
    return js


# ---------------------------------------------------------------------------
# Data pulls
# ---------------------------------------------------------------------------

def pull_standings(year: int) -> dict[str, dict]:
    """Return {team_abbrev: {wins, losses, RS, RA, division, league, ranks}}."""
    js = cached_get(f"standings_{year}.json", "standings",
                    leagueId=f"{LEAGUE_AL},{LEAGUE_NL}", season=year)
    out: dict[str, dict] = {}
    for div_record in js["records"]:
        league_id = div_record["league"]["id"]
        league = "AL" if league_id == LEAGUE_AL else "NL"
        division = DIVISIONS.get(div_record["division"]["id"], "?")
        for tr in div_record["teamRecords"]:
            abbrev = team_abbrev(tr["team"]["id"], tr["team"]["name"])
            wins = int(tr["wins"])
            losses = int(tr["losses"])
            rs = int(tr["runsScored"])
            ra = int(tr["runsAllowed"])
            out[abbrev] = {
                "team_mlb_id": tr["team"]["id"],
                "team_name_real": tr["team"]["name"],
                "league": league,
                "division": division,
                "wins": wins,
                "losses": losses,
                "runs_scored": rs,
                "runs_allowed": ra,
                "run_diff": rs - ra,
                "division_rank": int(tr["divisionRank"]),
                "league_rank": int(tr["leagueRank"]),
                "made_playoffs": _made_playoffs(tr, year),
                "won_division": int(int(tr["divisionRank"]) == 1),
                "pct": float(tr["winningPercentage"]),
            }
    return out


# Fixed mapping from MLB team IDs to traditional 2-3 letter abbreviations.
# Names sometimes change slightly across years (e.g. Cleveland Indians/Guardians);
# we key by stable MLB team id.
MLB_ID_TO_ABBREV = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC", 145: "CHW",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",
    108: "LAA", 119: "LAD", 146: "MIA", 158: "MIL", 142: "MIN", 121: "NYM",
    147: "NYY", 133: "OAK", 143: "PHI", 134: "PIT", 135: "SD", 136: "SEA",
    137: "SF", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH",
}


def team_abbrev(mlb_id: int, fallback_name: str = "") -> str:
    return MLB_ID_TO_ABBREV.get(mlb_id) or fallback_name[:3].upper()


def _made_playoffs(tr: dict, year: int) -> int:
    """Placeholder — _patch_playoff_flags is authoritative."""
    return 0


def _patch_playoff_flags(team_data: dict[str, dict], year: int) -> None:
    """Set made_playoffs = 1 for top-N-by-wins per league (authoritative)."""
    n = 12 if year >= 2022 else 10  # postseason expansion
    n_per_league = n // 2
    for t in team_data.values():
        t["made_playoffs"] = 0
    for league in ("AL", "NL"):
        teams = sorted(
            (t for t in team_data.values() if t["league"] == league),
            key=lambda t: -t["wins"],
        )[:n_per_league]
        for t in teams:
            t["made_playoffs"] = 1


def patch_postseason(team_data: dict[str, dict], year: int) -> None:
    """Set league_champion / world_series_champion from POSTSEASON table."""
    ps = POSTSEASON.get(year, {})
    for abbrev, t in team_data.items():
        t["league_champion"] = int(abbrev == ps.get("al" if t["league"] == "AL" else "nl"))
        t["world_series_champion"] = int(abbrev == ps.get("ws"))


def pull_roster(team_mlb_id: int, year: int, date: str) -> list[dict]:
    """Return active roster on a specific date."""
    cache = f"roster_{team_mlb_id}_{year}_{date}.json"
    js = cached_get(cache, f"teams/{team_mlb_id}/roster",
                    rosterType="Active", date=date)
    return js.get("roster", [])


def pull_player_stats_leaderboard(year: int, group: str) -> dict[int, dict]:
    """Pull all qualified players' season stats for a year. Returns {mlb_id: stat}."""
    cache = f"leaderboard_{group}_{year}.json"
    js = cached_get(cache, "stats",
                    stats="season", group=group, season=year, sportId=1, limit=2000)
    out: dict[int, dict] = {}
    for split in js.get("stats", [{}])[0].get("splits", []):
        pid = split.get("player", {}).get("id")
        if pid:
            out[pid] = split.get("stat", {})
    return out


def pull_team_stats(year: int, team_mlb_id: int) -> dict[str, dict]:
    """Pull season-aggregate hitting/pitching/fielding stats for one team."""
    cache = f"team_stats_{team_mlb_id}_{year}.json"
    js = cached_get(cache, f"teams/{team_mlb_id}/stats",
                    stats="season", season=year, group="hitting,pitching,fielding")
    out: dict[str, dict] = {}
    for s in js.get("stats", []):
        group = s["group"]["displayName"]
        splits = s.get("splits", [])
        if splits:
            out[group] = splits[0]["stat"]
    return out


def pull_half_season_stats(year: int, team_mlb_id: int, end_date: str) -> dict[str, dict]:
    """First-half stats up to All-Star break."""
    cache = f"team_stats_half_{team_mlb_id}_{year}.json"
    js = cached_get(cache, f"teams/{team_mlb_id}/stats",
                    stats="byDateRange", season=year,
                    startDate=f"{year}-03-15", endDate=end_date,
                    group="hitting,pitching,fielding")
    out: dict[str, dict] = {}
    for s in js.get("stats", []):
        group = s["group"]["displayName"]
        splits = s.get("splits", [])
        if splits:
            out[group] = splits[0]["stat"]
    return out


def pull_half_season_record(year: int, team_mlb_id: int, end_date: str) -> dict:
    """Half-season W-L record for checkpoint_wins_above_pace."""
    cache = f"team_record_half_{team_mlb_id}_{year}.json"
    js = cached_get(cache, "schedule",
                    sportId=1, season=year, teamId=team_mlb_id,
                    startDate=f"{year}-03-15", endDate=end_date,
                    gameType="R")
    wins = losses = 0
    for date in js.get("dates", []):
        for game in date.get("games", []):
            if game.get("status", {}).get("abstractGameState") != "Final":
                continue
            home = game["teams"]["home"]
            away = game["teams"]["away"]
            for side in (home, away):
                if side["team"]["id"] == team_mlb_id:
                    if side.get("isWinner"):
                        wins += 1
                    elif side.get("isWinner") is False:
                        losses += 1
    return {"wins": wins, "losses": losses, "games": wins + losses}


# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------

def pythag_win_pct(rs: float, ra: float, exp: float = 1.83) -> float:
    if rs <= 0 and ra <= 0:
        return 0.5
    return (rs ** exp) / (rs ** exp + ra ** exp)


def derive_team_war(team: dict) -> dict:
    """Synthesize WAR breakdowns from real RS/RA + roster size.

    These are PROXIES for the FanGraphs WAR breakdown the original schema
    expected. They preserve the relative ordering (good teams have higher
    derived WAR) but are not literal FanGraphs values.
    """
    pythag = pythag_win_pct(team["runs_scored"], team["runs_allowed"])
    expected_wins = pythag * (team["wins"] + team["losses"])  # uses games played
    # Replacement-level reference: 48 wins per 162-game season.
    games = team["wins"] + team["losses"]
    war_per_game = (expected_wins - 48 * games / 162) / games if games else 0
    team_war_total = max(5.0, war_per_game * 162)

    # Split by unit using rough public-domain heuristics:
    # ~52% position-player WAR, ~31% starter WAR, ~17% reliever WAR.
    pos_war = team_war_total * 0.52
    sp_war = team_war_total * 0.31
    rp_war = team_war_total * 0.17
    return {
        "team_war_total": team_war_total,
        "pos_war": pos_war,
        "sp_war": sp_war,
        "rp_war": rp_war,
    }


def make_team_row(year: int, checkpoint: str, abbrev: str, team: dict,
                  prev_team: dict | None,
                  league_strengths: dict[str, float],
                  division_strengths: dict[str, float],
                  half_record: dict | None = None) -> dict:
    """Produce one team_states row in the original schema."""
    rng = random.Random(year * 1000 + hash(abbrev) % 1000 + (13 if checkpoint == "all_star" else 0))
    war = derive_team_war(team)
    # Distribute SP WAR across SP1-SP7 with declining weights.
    sp_weights = [1.25, 1.08, 0.93, 0.78, 0.62, 0.45, 0.32]
    norm = sum(sp_weights)
    sp_parts = [war["sp_war"] * w / norm for w in sp_weights]

    # Projection systems = derived team WAR + system-specific noise.
    # Anchor on prior-year actual when available (real opening-day projections
    # are essentially regressed prior-year WAR).
    base_war = war["team_war_total"]
    if prev_team is not None and checkpoint == "opening_day":
        prev_war = derive_team_war(prev_team)["team_war_total"]
        base_war = 0.55 * prev_war + 0.45 * 30.0  # regress to league avg ~30
    dc = base_war + rng.gauss(0, 2.4)
    steamer = base_war + rng.gauss(0, 2.2)
    zips = base_war + rng.gauss(0, 2.6)

    pythag = pythag_win_pct(team["runs_scored"], team["runs_allowed"])
    if checkpoint == "all_star" and half_record and half_record["games"] > 0:
        pace = (half_record["wins"] - 0.5 * half_record["games"]) * 2  # wins above .500 pace, scaled to full season
    else:
        pace = 0.0

    park = PARK_FACTORS.get(abbrev, 100)
    schedule = league_strengths.get(team["league"], 0.0)
    div_str = division_strengths.get(team["division"], 0.0)

    # Real outcomes
    actual_wins = team["wins"] if checkpoint == "opening_day" else (
        # all_star: still predict full-season wins; the actual_wins label is the same final total
        team["wins"]
    )

    return {
        "season": year,
        "checkpoint": checkpoint,
        "team_id": abbrev,
        "team_name": f"{abbrev} Baseball Club",
        "league": team["league"],
        "division": team["division"],
        "dc_team_war": dc,
        "steamer_team_war": steamer,
        "zips_team_war": zips,
        "projection_blend_war": (dc + steamer + zips) / 3.0,
        "pos_war": war["pos_war"],
        "sp_war": war["sp_war"],
        "rp_war": war["rp_war"],
        "sp1_war": sp_parts[0], "sp2_war": sp_parts[1], "sp3_war": sp_parts[2],
        "sp4_war": sp_parts[3], "sp5_war": sp_parts[4], "sp6_war": sp_parts[5],
        "sp7_war": sp_parts[6],
        "bp_high_lev_war": war["rp_war"] * 0.42,
        "rotation_depth_war": war["sp_war"] * 0.72,
        "catcher_framing_runs": rng.gauss(0, 5.5),
        "drs_runs": rng.gauss(0, 14.0),
        "uzr_runs": rng.gauss(0, 11.0),
        "oaa_runs": rng.gauss(0, 12.0),
        "def_blend_runs": rng.gauss(0, 11.0),
        "schedule_strength": schedule + rng.gauss(0, 0.5),
        "division_strength": div_str,
        "intradivision_schedule_strength": div_str + rng.gauss(0, 0.4),
        "park_factor": park,
        "park_factor_3yr": park,
        "park_factor_1yr": park + rng.gauss(0, 3.0),
        "injury_war_lost": max(0.0, rng.gauss(2.7, 1.6)),
        "durability_risk": max(0.0, rng.gauss(1.0, 0.4)),
        "age_risk": max(0.0, rng.gauss(1.0, 0.55)),
        "pythag_win_pct": pythag,
        "baseruns_win_pct": pythag + rng.gauss(0, 0.012),
        "third_order_win_pct": pythag + rng.gauss(0, 0.014),
        "prev_win_pct": (prev_team["pct"] if prev_team else 0.5),
        "checkpoint_wins_above_pace": pace,
        "actual_wins": actual_wins,
        "overall_rank": 0,  # filled in assign_outcomes
        "league_rank": team["league_rank"],
        "division_rank": team["division_rank"],
        "made_playoffs": team.get("made_playoffs", 0),
        "won_division": team.get("won_division", 0),
        "league_champion": team.get("league_champion", 0),
        "world_series_champion": team.get("world_series_champion", 0),
    }


def assign_overall_ranks(rows: list[dict]) -> None:
    """Assign overall_rank within each (season, checkpoint) group by actual_wins."""
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["season"], row["checkpoint"])].append(row)
    for grp in groups.values():
        grp.sort(key=lambda r: (-r["actual_wins"], r["team_id"]))
        for rank, row in enumerate(grp, start=1):
            row["overall_rank"] = rank


def make_player_row(year: int, checkpoint: str, abbrev: str, slot: int,
                    roster_entry: dict, leaderboard_hit: dict, leaderboard_pit: dict,
                    team_war_breakdown: dict) -> dict:
    """Produce one player_states row from a roster entry + season leaderboards."""
    rng = random.Random(year * 2000 + hash(abbrev) % 2000 + slot * 17 + (29 if checkpoint == "all_star" else 0))
    person = roster_entry["person"]
    pid = person["id"]
    pos_data = roster_entry.get("position", {})
    pos = pos_data.get("abbreviation", "UT")
    pos_type = pos_data.get("type", "")
    is_pitcher = pos_type == "Pitcher"
    role = "starter" if pos == "SP" else ("reliever" if pos == "P" else "position_player")
    if not is_pitcher and slot > 9:
        role = "bench"

    hit = leaderboard_hit.get(pid, {})
    pit = leaderboard_pit.get(pid, {})
    age = _safe_int(hit.get("age") or pit.get("age")) or rng.randint(23, 36)

    # WAR proxy from leaderboard: scale OPS+ or ERA+ to a "WAR-like" value
    if is_pitcher:
        ip = _safe_float(pit.get("inningsPitched")) or 0.0
        era = _safe_float(pit.get("era"))
        # rough: each ERA point below 4.50 over 100 IP ≈ 1 WAR
        war = max(-0.5, ((4.50 - (era or 4.50)) * (ip / 100.0)) if era else 0.5)
        proj_ip = ip if checkpoint == "all_star" else 0.65 * ip + 0.35 * 100
        proj_pa = 0.0
        prev_ip = ip
        prev_pa = 0.0
    else:
        pa = _safe_int(hit.get("plateAppearances")) or 0
        ops_str = hit.get("ops") or ".700"
        ops = _safe_float(ops_str)
        # rough: each 0.100 OPS above .700 over 600 PA ≈ 2 WAR
        war = max(-0.5, ((ops - 0.700) * 20) * (pa / 600.0)) if pa else 0.0
        proj_pa = pa if checkpoint == "all_star" else 0.65 * pa + 0.35 * 400
        proj_ip = 0.0
        prev_pa = pa
        prev_ip = 0.0

    # Statcast aggregates: real ones for 2015+ would require extra pulls.
    # MVP: leave as 0.0 / midpoint. Agents can ignore these or fill from
    # Statcast in a future bundle revision.
    return {
        "season": year,
        "checkpoint": checkpoint,
        "team_id": abbrev,
        "player_id": f"{abbrev}{year}{checkpoint[:1]}{slot:02d}",
        "player_name": f"{abbrev} Player {slot:02d}",
        "age": age,
        "throws_bats": rng.choice(["R/R", "L/L", "R/L", "L/R", "S/R"]),
        "position": pos,
        "role": role,
        "proj_pa": proj_pa,
        "proj_ip": proj_ip,
        "prev_pa": prev_pa,
        "prev_ip": prev_ip,
        "steamer_war": war + rng.gauss(0, 0.20),
        "zips_war": war + rng.gauss(0, 0.22),
        "dc_war": war + rng.gauss(0, 0.18),
        "projection_blend_war": war,
        "woba": _safe_float(hit.get("ops")) * 0.45 if not is_pitcher else 0.0,
        "xwoba": 0.0,  # MVP: needs Statcast pull
        "barrel_pct": 0.0,
        "hard_hit_pct": 0.0,
        "exit_velocity": 0.0,
        "chase_pct": 0.0,
        "stuff_plus": 0.0,
        "location_plus": 0.0,
        "projected_leverage_index": 1.15 if role == "reliever" else 0.0,
        "framing_runs": 0.0,
        "drs_runs": 0.0,
        "uzr_runs": 0.0,
        "oaa_runs": 0.0,
        "def_runs": 0.0,
        "durability_score": min(1.0, max(0.05, (prev_ip / 190.0) if is_pitcher else (prev_pa / 650.0))),
        "injury_risk": rng.uniform(0.05, 0.25),
    }


def _safe_float(s: Any, default: float = 0.0) -> float:
    if s is None:
        return default
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _safe_int(s: Any) -> int | None:
    if s is None:
        return None
    try:
        return int(s)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_year(year: int) -> tuple[list[dict], list[dict]]:
    """Build all team_states + player_states rows for one season.

    Returns (team_rows, player_rows) where each year produces opening_day rows
    for all 30 teams, plus all_star rows where post-2010 (matches original
    schema: train is OD-only 2010-2022, val 2023 is OD+AS, frozen 2024-25 is
    OD+AS).
    """
    print(f"  building {year}...", flush=True)
    teams = pull_standings(year)
    _patch_playoff_flags(teams, year)
    patch_postseason(teams, year)

    # Per-league average wins (proxy for schedule strength)
    league_strengths = {}
    for league in ("AL", "NL"):
        league_teams = [t for t in teams.values() if t["league"] == league]
        league_strengths[league] = sum(t["wins"] for t in league_teams) / len(league_teams)
    division_strengths: dict[str, float] = {}
    for div_name in DIVISIONS.values():
        div_teams = [t for t in teams.values() if t["division"] == div_name]
        if div_teams:
            division_strengths[div_name] = sum(t["wins"] for t in div_teams) / len(div_teams)

    prev_year_data = pull_standings(year - 1) if year > 2010 else {}

    # Pull leaderboards once per (year, group)
    lb_hit = pull_player_stats_leaderboard(year, "hitting")
    lb_pit = pull_player_stats_leaderboard(year, "pitching")

    team_rows: list[dict] = []
    player_rows: list[dict] = []
    checkpoints = ["opening_day"]
    if year >= 2023:
        checkpoints.append("all_star")

    for checkpoint in checkpoints:
        if checkpoint == "opening_day":
            roster_date = f"{year}-04-01"
            half_records: dict[str, dict] = {}
        else:
            roster_date = f"{year}-04-01"  # use opening-day roster as the snapshot; pull half-season record below
            half_records = {abbrev: pull_half_season_record(year, t["team_mlb_id"], f"{year}-07-15")
                            for abbrev, t in teams.items()}

        for abbrev, team in teams.items():
            half = half_records.get(abbrev)
            row = make_team_row(year, checkpoint, abbrev, team,
                                prev_year_data.get(abbrev),
                                league_strengths, division_strengths, half)
            team_rows.append(row)

            roster = pull_roster(team["team_mlb_id"], year, roster_date)
            for slot, entry in enumerate(roster, start=1):
                war_breakdown = derive_team_war(team)
                prow = make_player_row(year, checkpoint, abbrev, slot, entry,
                                       lb_hit, lb_pit, war_breakdown)
                player_rows.append(prow)

    return team_rows, player_rows


# ---------------------------------------------------------------------------
# Anonymization
# ---------------------------------------------------------------------------

def build_anonymization_maps() -> tuple[dict, dict, dict, dict]:
    """Deterministic mapping from real → anonymous identifiers."""
    rng = random.Random(ANON_SEED)
    real_teams = list(MLB_ID_TO_ABBREV.values())
    fake_teams = [f"TEAM_{i:02d}" for i in range(1, 31)]
    rng.shuffle(fake_teams)
    team_map = dict(zip(real_teams, fake_teams))

    real_seasons = list(range(2010, 2026))
    # Use small integers (1..16) preserving chronological order. Numeric so
    # features.py / eval.py parse them with int(); not a real year so the
    # safety filter and any "what happened in year N" memorization shortcut
    # are both blocked.
    fake_seasons = list(range(1, len(real_seasons) + 1))
    season_map = dict(zip(real_seasons, fake_seasons))

    real_divs = ["AL East", "AL Central", "AL West", "NL East", "NL Central", "NL West"]
    fake_divs = [f"DIV_{i}" for i in range(1, 7)]
    rng.shuffle(fake_divs)
    div_map = dict(zip(real_divs, fake_divs))
    league_map = {"AL": "LG_A", "NL": "LG_B"}
    return team_map, season_map, div_map, league_map


def anonymize_team_row(row: dict, team_map, season_map, div_map, league_map) -> dict:
    new = dict(row)
    new["team_id"] = team_map[row["team_id"]]
    new["team_name"] = f"{new['team_id']} Baseball Club"
    new["season"] = season_map[row["season"]]
    new["division"] = div_map[row["division"]]
    new["league"] = league_map[row["league"]]
    return new


def anonymize_player_row(row: dict, team_map, season_map) -> dict:
    new = dict(row)
    new["team_id"] = team_map[row["team_id"]]
    new["season"] = season_map[row["season"]]
    slot = row["player_id"][-2:]
    if not slot.isdigit():
        slot = "00"
    new["player_id"] = f"{new['team_id']}_{new['season']}_{new['checkpoint'][:1]}_{slot}"
    new["player_name"] = f"{new['team_id']} Player {slot}"
    return new


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v
                        for k, v in row.items() if k in columns})


def split_by_year(rows: list[dict], train_years, val_years, frozen_years):
    train, val, frozen = [], [], []
    for r in rows:
        y = r["season"]
        if isinstance(y, str):
            # already anonymized; can't split this way — should be called pre-anonymization
            raise RuntimeError("split_by_year must run before anonymization")
        if y in train_years:
            train.append(r)
        elif y in val_years:
            val.append(r)
        elif y in frozen_years:
            frozen.append(r)
    return train, val, frozen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="*", help="Subset of years (default: 2010-2025)")
    ap.add_argument("--skip-pull", action="store_true", help="Use cached raw pulls only (no network)")
    args = ap.parse_args()

    years = args.years or list(range(2010, 2026))
    print(f"Building bundle for years {years[0]}-{years[-1]} ({len(years)} seasons)...")

    all_team_rows: list[dict] = []
    all_player_rows: list[dict] = []
    for year in years:
        try:
            t_rows, p_rows = build_year(year)
            all_team_rows.extend(t_rows)
            all_player_rows.extend(p_rows)
        except Exception as e:
            print(f"  FAIL {year}: {type(e).__name__}: {e}")
            raise

    assign_overall_ranks(all_team_rows)

    # Split BEFORE anonymization (year ints needed for split logic)
    train_years = list(range(2010, 2023))
    val_years = [2023]
    frozen_years = [2024, 2025]
    train_t, val_t, frozen_t = split_by_year(all_team_rows, train_years, val_years, frozen_years)
    train_p, val_p, frozen_p = split_by_year(all_player_rows, train_years, val_years, frozen_years)

    # Anonymize
    team_map, season_map, div_map, league_map = build_anonymization_maps()
    train_t = [anonymize_team_row(r, team_map, season_map, div_map, league_map) for r in train_t]
    val_t   = [anonymize_team_row(r, team_map, season_map, div_map, league_map) for r in val_t]
    frozen_t = [anonymize_team_row(r, team_map, season_map, div_map, league_map) for r in frozen_t]
    train_p = [anonymize_player_row(r, team_map, season_map) for r in train_p]
    val_p   = [anonymize_player_row(r, team_map, season_map) for r in val_p]
    frozen_p = [anonymize_player_row(r, team_map, season_map) for r in frozen_p]

    # Write CSVs
    if STAGING.exists():
        shutil.rmtree(STAGING)
    write_csv(train_t,  STAGING / "train" / "team_states.csv",       TEAM_COLUMNS)
    write_csv(train_p,  STAGING / "train" / "player_states.csv",     PLAYER_COLUMNS)
    write_csv(val_t,    STAGING / "val"   / "team_states.csv",       TEAM_COLUMNS)
    write_csv(val_p,    STAGING / "val"   / "player_states.csv",     PLAYER_COLUMNS)
    write_csv(frozen_t, STAGING / "frozen" / "frozen_test.csv",         TEAM_COLUMNS)
    write_csv(frozen_p, STAGING / "frozen" / "frozen_test_players.csv", PLAYER_COLUMNS)

    # Identity map (private)
    IDENTITY_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    IDENTITY_MAP_PATH.write_text(json.dumps({
        "teams":     {v: k for k, v in team_map.items()},
        "seasons":   {v: k for k, v in season_map.items()},
        "divisions": {v: k for k, v in div_map.items()},
        "leagues":   {v: k for k, v in league_map.items()},
    }, indent=2))

    # Zip
    DIST.mkdir(exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH).removesuffix(".zip"), "zip", STAGING)
    sha = hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()

    print()
    print(f"BUNDLE  : {ZIP_PATH}")
    print(f"  size  : {ZIP_PATH.stat().st_size:,} bytes")
    print(f"  sha256: {sha}")
    print(f"  train : {len(train_t)} team rows, {len(train_p)} player rows")
    print(f"  val   : {len(val_t)} team rows, {len(val_p)} player rows")
    print(f"  frozen: {len(frozen_t)} team rows, {len(frozen_p)} player rows")
    print(f"  identity map: {IDENTITY_MAP_PATH}")
    print()
    print(f"Next: gh release create mlb-data-v1 {ZIP_PATH}")


if __name__ == "__main__":
    main()
