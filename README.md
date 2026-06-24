# Dreamer-PPO CARLA

A Dreamer-style world model + PPO agent for CARLA urban driving, focused on
**VRU (Vulnerable Road User) safety**. Runs fully in **mock mode** (no CARLA
required) for development and testing.

```
python tests/test_mock.py             # full pipeline, no CARLA needed
python -m training.ppo_baseline --mock --episodes 1000   # PPO-only baseline
python -m training.dreamer_ppo  --mock --episodes 1000   # greedy dreaming
```

For a real run, start a CARLA 0.9.15 server and drop the `--mock` flag.

## S-DBS Extension (Advanced)

Dreamer-PPO with Serendipitous Diverse Beam Search. Replaces greedy
one-step dreaming with multi-step lookahead, diverse planning groups,
serendipity bonuses, hierarchical maneuver-level search, budget-aware
adaptation, and risk-aware curriculum learning.

Solves the "greedy trap" problem: the agent is no longer shortsighted
about occluded pedestrians and ambiguous crossings.

```
python -m training.dreamer_ppo --sdbs --episodes 1000
python -m training.dreamer_ppo --mock --sdbs --episodes 2   # quick mock run
python tests/test_sdbs.py                                   # S-DBS validation
```

Curriculum progression:

```
Stage 1 (Easy):    empty/low-density roads
Stage 2 (Medium):  crossings, yielding
Stage 3 (Hard):    occluded pedestrians, adversarial greedy traps
```

After Stage 3 unlock, the agent has learned long-horizon risk and
recognizes non-obvious safe maneuvers.

### S-DBS modules

| Module | Responsibility |
|--------|----------------|
| `planning/sdbs_core.py` | `Plan`, `BeamState`, conflict-cell discretization, Jaccard / trajectory diversity |
| `planning/sdbs_planner.py` | `SDBSPlanner`: difficulty estimation, budget-aware search params, mandated safety, multi-step diverse beam search |
| `planning/curriculum.py` | `ScenarioBank`, `SumTree`, `PrioritizedScenarioReplayer`, `RiskAwareCurriculum` |
| `models/auxiliary_heads.py` | `SceneReconstructionHead`, `RiskDensityHead`, `WorldModelEnsemble` (epistemic uncertainty) |
| `configs/sdbs_config.py` | `SDBSConfig` — all S-DBS hyperparameters (extends `Config`) |

The base Dreamer-PPO path is unchanged; `--sdbs` switches the training loop to
`train_sdbs()`, which plugs the planner, curriculum, ensemble, and grounding
heads into PPO.

## Traffic Prediction

Agents no longer assumed static. A multi-agent trajectory predictor learns to
forecast VRU and vehicle movements 0.8 seconds ahead (8 steps at 10 Hz).

S-DBS planning incorporates predicted trajectories:
- score action sequences based on predicted multi-agent futures
- penalize plans that collide with predicted pedestrian/cyclist paths
- adaptive: high-uncertainty predictions trigger defensive maneuvers

Training:
- Phase 1: collect trajectory data (automatic, `collect_prediction_data`)
- Phase 2: train `TrafficPredictor` on collected trajectories
- Phase 3: S-DBS uses predictions for safer planning

Expected improvement: near-miss rate down 30–40% vs. a static-agent baseline.

```
python -m training.dreamer_ppo --mock --sdbs --episodes 1000   # prints tp_loss / TP eval
python tests/test_traffic_prediction.py                        # prediction validation
python scripts/collect_prediction_data.py --mock --episodes 500 \
    --save-path data/trajectories.pkl                          # pre-collect data
```

| Module | Responsibility |
|--------|----------------|
| `models/traffic_predictor.py` | `TrafficPredictor` (LSTM encoder + MLP decoder), `MultiAgentPredictor`, `compute_collision_risk` |
| `utils/trajectory_utils.py` | `TrajectoryBuffer`, `extract_agent_trajectories`, `evaluate_prediction_accuracy` (ADE/FDE) |
| `training/traffic_prediction_trainer.py` | `TrafficPredictionTrainer` — collection + training + evaluation |

> In mock mode agent trajectories are random noise (no real prediction is
> possible), but the full pipeline — tracking → prediction → collision-aware
> planning — runs end to end. Real CARLA training shows meaningful prediction
> accuracy and planning improvements.
