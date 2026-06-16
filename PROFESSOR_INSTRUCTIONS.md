# Running Dreamer-PPO in CARLA

## Requirements
- CARLA 0.9.15: https://github.com/carla-simulator/carla/releases/tag/0.9.15
- Python 3.8+
- pip install -r requirements.txt

## Setup
1. Start CARLA server:
   ```
   ./CarlaUE4.sh -RenderOffScreen   # Linux
   CarlaUE4.exe                      # Windows
   ```

2. Clone repo:
   ```
   git clone https://github.com/youma2003/dreamer_ppo_carla
   cd dreamer_ppo_carla
   pip install -r requirements.txt
   ```

## Running

### Verify installation (no CARLA needed):
```
python tests/test_mock.py
```

### Step 2 — PPO baseline only:
```
python -m training.ppo_baseline --episodes 1000
```

### Step 3-5 — Full Dreamer-PPO:
```
python -m training.dreamer_ppo --episodes 1000
```

### Monitor training:
```
python plot_results.py --log logs/training_log.csv
```

## What to expect

**Phase 1 (episodes 0-50): World model warming up, dreaming=OFF**
- PPO learns basic driving from random experience
- World model collecting transitions

**Phase 2 (episodes 50+): Dreaming activates**
- Every step: 5 candidate actions scored by world model
- Safest action selected before execution
- VRU collisions should decrease vs baseline

**Phase 3 (episodes 200+): Convergence**
- Route completion increasing
- VRU near-misses decreasing
- Eval return stabilizing

## Key metrics to monitor
- `eval_return`:           overall performance (higher = better)
- `eval_vru_collisions`:   pedestrian/cyclist hits (lower = better)
- `eval_near_misses`:      TTC < 2s events (lower = better)
- `eval_route_completion`: % of route completed (higher = better)
- `wm_state_err`:          world model accuracy (lower = better)

## Checkpoints
- Best model saved to: `checkpoints/best_model.pt`
- Periodic saves:      `checkpoints/episode_XXXX.pt`
