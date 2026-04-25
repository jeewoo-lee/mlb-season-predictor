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


def main() -> None:
    if not TEST_PATH.exists():
        raise SystemExit("missing frozen test data; run bash prepare.sh first")

    agent = load_agent()
    rows = load_rows(TEST_PATH)
    rosters = load_rosters(ROSTER_PATH)
    log_loss = 0.0
    brier = 0.0
    win_abs_error = 0.0
    correct = 0

    for row in rows:
        team_state = dict(row)
        team_state["roster"] = rosters.get(roster_key(row), [])
        pred = agent.predict(team_state)
        prob = clamp(float(pred["playoff_prob"]), 1e-6, 1.0 - 1e-6)
        wins = float(pred["projected_wins"])
        y = int(row["made_playoffs"])
        log_loss += -(y * math.log(prob) + (1 - y) * math.log(1 - prob))
        brier += (prob - y) ** 2
        win_abs_error += abs(wins - float(row["actual_wins"]))
        correct += int((prob >= 0.5) == bool(y))

    total = len(rows)
    print("---")
    print(f"log_loss:         {log_loss / total:.4f}")
    print(f"brier:            {brier / total:.4f}")
    print(f"win_mae:          {win_abs_error / total:.1f}")
    print(f"correct:          {correct}")
    print(f"total:            {total}")


if __name__ == "__main__":
    main()
