# Dreamer-PPO for CARLA (VRU-safe urban driving)

A Dreamer-style world model combined with a PPO policy for CARLA urban
driving, focused on the safety of Vulnerable Road Users (VRUs — pedestrians
and cyclists). The whole pipeline runs in a **mock mode** with no CARLA
installed, so every component can be developed and tested locally.

There are three variants, each trained end-to-end by a single command and
each writing its own CSV log under `logs/`:

| Variant           | Flag         | What it is                                     |
|-------------------|--------------|------------------------------------------------|
| PPO baseline      | `--baseline` | reference: no world model, no dreaming         |
| Dreamer-PPO       | *(none)*     | world model + greedy one-step dreaming         |
| SDBS Dreamer-PPO  | `--sdbs`     | full version: diverse beam search + curriculum |

See [RUNME.md](RUNME.md) for the exact commands and log-inspection steps.

## The state-vector contract

The policy input dimension is a single source of truth in
`env/carla_env.py`:

- `STATE_DIM = 55` — the full augmented state.
- `STATE_LAYOUT` — the index ranges of every field block.
- The base env emits `config.state_dim` (48) features: ego, lane, traffic,
  five vehicle blocks (ahead/behind/left/right/nearest), and two VRUs. The
  map-agnostic wrapper appends the final 7 features to reach 55 in the S-DBS
  path.

`validate_state_vector(state, expected_dim=STATE_DIM)` raises immediately on a
shape mismatch — it **never** silently pads, truncates, or reshapes. It is
called at the start of `CarlaEnv.reset()`/`step()`, `SDBSPlanner.plan()`, and
`ActorCritic.act()`/`forward()`, so a wrong dimension fails loudly at the
first model call instead of silently producing garbage.

## Integrating this checkpoint elsewhere

Before wiring a trained checkpoint into ANY external runtime/adapter
(e.g. a CARLA/SimLingo adapter):

1. **Run the dimension check.** Confirm the checkpoint's first-layer input
   dimension matches the state vector you intend to feed it:

   ```bash
   python -m utils.checkpoint_check checkpoints/sdbs_final_model.pt --state-dim 55
   ```

   A `FAIL` here means the checkpoint and your state vector disagree — it will
   produce meaningless output until fixed.

2. **Print the state shape in the adapter.** Add a state-vector shape print
   right before every model call and confirm it matches 55 (or your
   configured dim). A silent dimension/order mismatch between training and
   inference is the most common cause of a checkpoint "not working".

3. **Start S-DBS at horizon=1, groups=1.** Set
   `sdbs_force_fixed_params=True, sdbs_fixed_horizon=1, sdbs_fixed_groups=1`
   (behaviourally equivalent to greedy one-step dreaming) and confirm the
   result matches the plain dreamer variant before enabling full multi-step
   search. Then increase `sdbs_fixed_horizon` gradually and watch for where
   behaviour degrades — this isolates whether the problem is the search depth
   or something upstream.

4. **If route_completion stays at 0% across ALL variants** (baseline
   included), this is an env/reward wiring bug, not a policy-quality issue.
   The trainers run a `ProgressMonitor` that prints a warning when the
   `route_progress` signal never advances — check that output before
   concluding anything about which dreamer variant performs better.

## Tests (no CARLA needed)

```bash
python tests/test_mock.py                     # full mock pipeline
python tests/test_sdbs.py                      # S-DBS core logic
python tests/test_integration_diagnostics.py   # integration guardrails
python tests/test_tier1_safety.py
python tests/test_tier2_generalization.py
python tests/test_tier3_interpretability.py
```
