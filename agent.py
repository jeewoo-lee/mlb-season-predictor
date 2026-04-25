"""API-first starter predictor for the MLB Season Predictor hive task.

When XAI_API_KEY is set, this file calls a Grok-compatible chat API and caches
strict JSON predictions. Without credentials, it falls back to a weak local
baseline so the task remains runnable for smoke tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from features import baseline_projected_wins, clamp, load_rows, sigmoid


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "train" / "team_states.csv"
CACHE_DIR = ROOT / ".cache" / "grok_predictions"
DEFAULT_MODEL = "grok-4-1-fast-reasoning"


def _fit_war_logistic() -> tuple[float, float]:
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


def _baseline_predict(team_state: dict) -> dict:
    intercept, slope = _coefficients()
    dc_war = float(team_state["dc_team_war"])
    projected_wins = baseline_projected_wins(team_state)

    logit = intercept + slope * (dc_war - 33.0)
    if team_state.get("checkpoint") == "all_star":
        logit += 0.10 * float(team_state.get("checkpoint_wins_above_pace", 0.0))

    playoff_prob = clamp(sigmoid(0.10 * logit), 0.03, 0.97)
    division_winner_prob = clamp(sigmoid((projected_wins - 88.0) / 7.0), 0.01, 0.80)
    league_champion_prob = clamp(sigmoid((projected_wins - 94.0) / 6.0) * 0.30, 0.005, 0.45)
    world_series_champion_prob = clamp(sigmoid((projected_wins - 97.0) / 6.0) * 0.16, 0.002, 0.30)
    spread = 8.5 + 0.05 * abs(projected_wins - 81.0)
    return {
        "playoff_prob": playoff_prob,
        "division_winner_prob": division_winner_prob,
        "league_champion_prob": league_champion_prob,
        "world_series_champion_prob": world_series_champion_prob,
        "projected_wins": projected_wins,
        "win_interval_80": [
            clamp(projected_wins - spread, 40.0, 116.0),
            clamp(projected_wins + spread, 46.0, 122.0),
        ],
    }


def _cache_key(team_state: dict, model: str) -> str:
    payload = {
        "model": model,
        "season": team_state["season"],
        "checkpoint": team_state["checkpoint"],
        "team_id": team_state["team_id"],
        "team": {k: v for k, v in team_state.items() if k != "roster"},
        "roster": team_state.get("roster", []),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


def _top_players(roster: list[dict], role: str | None = None, limit: int = 8) -> list[dict]:
    rows = [p for p in roster if role is None or p.get("role") == role]
    return sorted(rows, key=lambda p: float(p.get("projection_blend_war", 0.0)), reverse=True)[:limit]


def _summarize_player(player: dict) -> dict:
    keys = [
        "player_name",
        "age",
        "position",
        "role",
        "projection_blend_war",
        "proj_pa",
        "proj_ip",
        "xwoba",
        "barrel_pct",
        "hard_hit_pct",
        "stuff_plus",
        "location_plus",
        "framing_runs",
        "def_runs",
        "durability_score",
        "injury_risk",
    ]
    return {key: player.get(key) for key in keys if key in player}


def _prompt(team_state: dict, baseline: dict) -> str:
    roster = team_state.get("roster", [])
    hitters = [_summarize_player(p) for p in _top_players(roster, None, 10) if p.get("role") in {"position_player", "bench", "callup"}]
    starters = [_summarize_player(p) for p in _top_players(roster, "starter", 7)]
    relievers = [_summarize_player(p) for p in _top_players(roster, "reliever", 5)]
    team_keys = [
        "season",
        "checkpoint",
        "team_id",
        "team_name",
        "league",
        "division",
        "projection_blend_war",
        "pos_war",
        "sp_war",
        "rp_war",
        "sp1_war",
        "sp2_war",
        "sp3_war",
        "sp4_war",
        "sp5_war",
        "sp6_war",
        "sp7_war",
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
    ]
    team_summary = {key: team_state.get(key) for key in team_keys if key in team_state}
    payload = {
        "task": "Project MLB final standings outcomes from this checkpoint.",
        "team_state": team_summary,
        "baseline_prediction": baseline,
        "top_hitters": hitters,
        "starting_pitchers": starters,
        "high_leverage_relievers": relievers,
        "output_schema": {
            "projected_wins": "number",
            "playoff_prob": "0..1",
            "division_winner_prob": "0..1",
            "league_champion_prob": "0..1",
            "world_series_champion_prob": "0..1",
            "win_interval_80": ["low", "high"],
        },
    }
    return (
        "Return only compact JSON matching output_schema. Do not include markdown.\n"
        + json.dumps(payload, sort_keys=True)
    )


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def _normalize_prediction(raw: dict, fallback: dict) -> dict:
    projected_wins = clamp(float(raw.get("projected_wins", fallback["projected_wins"])), 40.0, 122.0)
    interval = raw.get("win_interval_80", fallback["win_interval_80"])
    if not isinstance(interval, list) or len(interval) != 2:
        interval = fallback["win_interval_80"]
    low = clamp(float(interval[0]), 35.0, projected_wins)
    high = clamp(float(interval[1]), projected_wins, 125.0)
    return {
        "playoff_prob": clamp(float(raw.get("playoff_prob", fallback["playoff_prob"])), 0.001, 0.999),
        "division_winner_prob": clamp(float(raw.get("division_winner_prob", fallback["division_winner_prob"])), 0.001, 0.999),
        "league_champion_prob": clamp(float(raw.get("league_champion_prob", fallback["league_champion_prob"])), 0.001, 0.999),
        "world_series_champion_prob": clamp(float(raw.get("world_series_champion_prob", fallback["world_series_champion_prob"])), 0.001, 0.999),
        "projected_wins": projected_wins,
        "win_interval_80": [low, high],
    }


def _call_grok(team_state: dict, fallback: dict) -> dict:
    api_key = os.getenv("XAI_API_KEY")
    if not api_key:
        return fallback

    try:
        from openai import OpenAI
    except Exception:
        return fallback

    model = os.getenv("XAI_MODEL", DEFAULT_MODEL)
    key = _cache_key(team_state, model)
    cache_path = CACHE_DIR / f"{key}.json"
    if cache_path.exists():
        return _normalize_prediction(json.loads(cache_path.read_text()), fallback)

    client = OpenAI(
        api_key=api_key,
        base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a calibrated MLB season forecasting model. "
                    "Use only the supplied team and roster state. Return strict JSON."
                ),
            },
            {"role": "user", "content": _prompt(team_state, fallback)},
        ],
    )
    content = response.choices[0].message.content or "{}"
    raw = _extract_json(content)
    prediction = _normalize_prediction(raw, fallback)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(prediction, sort_keys=True) + "\n")
    return prediction


def predict(team_state: dict) -> dict:
    fallback = _baseline_predict(team_state)
    return _call_grok(team_state, fallback)
