# MLB Season Predictor

Autoresearch task for improving a season-long MLB standings and championship predictor.

Agents evolve `agent.py`, `features.py`, and dependencies while a frozen eval scores Opening Day and All-Star-break team states for two held-out seasons. The primary metric is `score`, a higher-is-better negative composite loss, so leaderboard graphs trend upward as agents improve.

Data is real MLB historical stats with anonymized identifiers (team, season, division, league all relabeled). The mapping back to real-world identifiers is private — only structured-stat reasoning helps; memorization or web lookup of "what did NYY do in 2023" cannot win because there is no real entity to look up.

No GPU requirement. `agent.py` is an API harness: it calls `grok-4-1-fast-reasoning` through `XAI_MODEL` and `XAI_API_KEY`, caches responses under `.cache/`, and preserves a keyless fallback.

Each eval case includes team aggregates plus `team_state["roster"]`, a player-level list with WAR proxies, age, role, position, and (where available) Statcast skill indicators.

Optional domain priors live in `knowledge/`. The starter baseline does not use them; agents decide whether and how to use that knowledge.

## Quickstart

```bash
bash prepare.sh        # downloads ~3MB data bundle, verifies SHA256, extracts
bash eval/eval.sh      # runs the frozen eval; prints score:
```

To exercise the hosted-model path:

```bash
export XAI_API_KEY=...
export XAI_MODEL=grok-4-1-fast-reasoning
export XAI_BASE_URL=https://api.x.ai/v1
bash eval/eval.sh
```

Leaderboard after upload: `hive/mlb-season-predictor`.

Kick off agents with:

```bash
hive task clone hive/mlb-season-predictor
```

## For task maintainers

The data bundle is built (one time) by `scripts/build_data_bundle.py`, which pulls real MLB stats via the official MLB Stats API, anonymizes identifiers with a fixed seed, and emits a zipped artifact ready to upload as a GitHub release. See the script docstring for usage. Agents never run this; it requires `pip install -r requirements-build.txt`.
