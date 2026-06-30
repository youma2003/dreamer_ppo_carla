# How to run & evaluate — quick runbook

This project trains **end-to-end with a single command per variant** — there is
nothing to run "part by part". The world model, traffic predictor, auxiliary
heads, PPO policy, curriculum, and dreaming are all set up and trained inside
one loop. The "Step 2–5" / "Tier 1–3" labels in the history are *build
milestones*, not a sequence you have to execute.

There are three variants (each writes its **own** CSV under `logs/`):

| Variant            | Command flag | What it is                                      |
|--------------------|--------------|-------------------------------------------------|
| PPO baseline       | `--baseline` | reference: no world model, no dreaming          |
| Dreamer-PPO        | *(none)*     | world model + greedy one-step dreaming          |
| SDBS Dreamer-PPO   | `--sdbs`     | full version: diverse beam search + curriculum  |

---

## 0. One-time check (no CARLA needed)

```bash
python tests/test_mock.py        # full pipeline smoke test, ~seconds
python tests/test_kpi.py         # KPI / comparison module
```

## 1. (Optional) mock smoke test — confirms it runs without a CARLA server

```bash
python -m training.dreamer_ppo --mock --baseline --episodes 3
python -m training.dreamer_ppo --mock            --episodes 3
python -m training.dreamer_ppo --mock --sdbs     --episodes 3
```
> In `--mock` mode the agents move as random noise, so the *numbers* are
> meaningless — this only proves the code runs. Real numbers need CARLA.

## 2. Real training (needs a running CARLA 0.9.15 server)

Run each variant once. Each writes its own log automatically:
`logs/baseline.csv`, `logs/dreamer.csv`, `logs/sdbs.csv`.

```bash
python -m training.dreamer_ppo --baseline      # -> logs/baseline.csv
python -m training.dreamer_ppo                 # -> logs/dreamer.csv
python -m training.dreamer_ppo --sdbs          # -> logs/sdbs.csv
```

Notes:
- `--sdbs` is compute-heavy (~18 ms per planning step). For a first run, add
  `--episodes N` to keep it short.
- Add `--device cuda` if a GPU is available.
- Override the filename with `--log-name myrun.csv` if you want.

## 3. Compare the variants — the KPIs (VRU safety first)

```bash
python -m scripts.compare_dreamers \
    baseline=logs/baseline.csv \
    dreamer=logs/dreamer.csv \
    sdbs=logs/sdbs.csv \
    --plot logs/plots/kpi.png
```

This prints a side-by-side table and saves a bar chart. Variants are **ranked
by VRU (pedestrian/cyclist) safety first**, then by the overall composite.

What the KPIs mean:
- **VRU safety (primary):** collisions/ep, collision rate, near-misses/ep,
  mean min time-to-collision, mean distance to VRU.
- **Vehicle safety (secondary):** collisions, near-misses, rear incidents.
- **Performance:** mean return, route completion, success rate.
- **Headline scores (0–100, higher = better):**
  - `vru_safety_score` — VRU safety only.
  - `composite_score` — safety-weighted overall (55% VRU, 25% progress,
    12% vehicle, 8% comfort), so a variant that drives further but hits a
    pedestrian always ranks **below** a safer one.

By default the comparison judges the **last 25% of episodes** (converged
behaviour). Use `--tail 1.0` to score the whole run.
