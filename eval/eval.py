from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "eval" / "test_data" / "mlb_frozen_2024_2025.csv"
ROSTER_PATH = ROOT / "eval" / "test_data" / "player_states_2024_2025.csv"
sys.path.insert(0, str(ROOT))

from features import clamp, load_rosters, load_rows, roster_key  # noqa: E402


def load_agent():
    spec = importlib.util.spec_from_file_location("agent", ROOT / "agent.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise RuntimeError("agent.py must define predict(team_state)")
    return module


def binary_loss(prob: float, label: int) -> float:
    p = clamp(float(prob), 1e-6, 1.0 - 1e-6)
    y = int(label)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def assign_predicted_ranks(scored_rows: list[dict]) -> None:
    groups: dict[tuple[int, str], list[dict]] = {}
    for item in scored_rows:
        row = item["row"]
        groups.setdefault((int(row["season"]), str(row["checkpoint"])), []).append(item)

    for group_items in groups.values():
        overall = sorted(
            group_items,
            key=lambda item: (-float(item["projected_wins"]), str(item["row"]["team_id"])),
        )
        for rank, item in enumerate(overall, start=1):
            item["pred_overall_rank"] = rank

        for league in ("AL", "NL"):
            league_items = [item for item in overall if item["row"]["league"] == league]
            for rank, item in enumerate(league_items, start=1):
                item["pred_league_rank"] = rank

        divisions = sorted({item["row"]["division"] for item in group_items})
        for division in divisions:
            division_items = [
                item for item in overall if item["row"]["division"] == division
            ]
            for rank, item in enumerate(division_items, start=1):
                item["pred_division_rank"] = rank


def main() -> None:
    if not TEST_PATH.exists():
        raise SystemExit("missing frozen test data; run bash prepare.sh first")

    agent = load_agent()
    rows = load_rows(TEST_PATH)
    rosters = load_rosters(ROSTER_PATH)
    scored_rows = []

    for row in rows:
        team_state = dict(row)
        team_state["roster"] = rosters.get(roster_key(row), [])
        pred = agent.predict(team_state)
        projected_wins = clamp(float(pred["projected_wins"]), 40.0, 122.0)
        scored_rows.append(
            {
                "row": row,
                "pred": pred,
                "projected_wins": projected_wins,
            }
        )

    assign_predicted_ranks(scored_rows)

    league_rank_abs_error = 0.0
    overall_rank_abs_error = 0.0
    division_rank_abs_error = 0.0
    win_abs_error = 0.0
    league_champion_loss = 0.0
    world_series_loss = 0.0
    playoff_loss = 0.0
    exact_league_rank = 0

    for item in scored_rows:
        row = item["row"]
        pred = item["pred"]
        projected_wins = item["projected_wins"]
        league_rank_abs_error += abs(
            int(item["pred_league_rank"]) - int(row["league_rank"])
        )
        overall_rank_abs_error += abs(
            int(item["pred_overall_rank"]) - int(row["overall_rank"])
        )
        division_rank_abs_error += abs(
            int(item["pred_division_rank"]) - int(row["division_rank"])
        )
        exact_league_rank += int(int(item["pred_league_rank"]) == int(row["league_rank"]))
        win_abs_error += abs(projected_wins - float(row["actual_wins"]))
        league_champion_loss += binary_loss(
            pred.get("league_champion_prob", 1.0 / 15.0), int(row["league_champion"])
        )
        world_series_loss += binary_loss(
            pred.get("world_series_champion_prob", 1.0 / 30.0),
            int(row["world_series_champion"]),
        )
        playoff_loss += binary_loss(pred.get("playoff_prob", 0.5), int(row["made_playoffs"]))

    total = len(scored_rows)
    print("---")
    print(f"rank_mae:         {league_rank_abs_error / total:.4f}")
    print(f"overall_rank_mae: {overall_rank_abs_error / total:.4f}")
    print(f"division_rank_mae:{division_rank_abs_error / total:.4f}")
    print(f"league_champ_log_loss: {league_champion_loss / total:.4f}")
    print(f"ws_champ_log_loss:{world_series_loss / total:.4f}")
    print(f"playoff_log_loss: {playoff_loss / total:.4f}")
    print(f"win_mae:          {win_abs_error / total:.1f}")
    print(f"correct:          {exact_league_rank}")
    print(f"total:            {total}")


if __name__ == "__main__":
    main()
