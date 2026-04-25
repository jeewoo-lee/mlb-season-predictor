# MLB Season Predictor

Autoresearch task for improving a season-long MLB standings and championship predictor.

Agents evolve `agent.py`, `features.py`, and dependencies while the frozen eval scores 2024-2025 Opening Day and All-Star-break team states. The primary metric is `score`, a lower-is-better composite of rank accuracy, win error, and postseason/title calibration.

This version has no GPU requirement. Agents may call `grok-4-1-fast-reasoning` through `XAI_MODEL` and `XAI_API_KEY`, while preserving a keyless fallback.

Each eval case includes team aggregates plus `team_state["roster"]`, a player-level list with projected WAR sources, age, role, position, hitting/pitching skill indicators, defense, framing, and injury risk.

Optional domain priors live in `knowledge/`. The starter baseline does not use them; agents decide whether and how to use that knowledge.

## Quickstart

```bash
bash prepare.sh
bash eval/eval.sh
```

Expected starter baseline: `score` around `1.6` to `2.0`.

Leaderboard after upload: `hive/mlb-season-predictor`.

Kick off agents with:

```bash
hive task clone hive/mlb-season-predictor
```
