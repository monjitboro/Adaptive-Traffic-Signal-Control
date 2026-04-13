# Adaptive Traffic Signal Control (Single-Intersection RL)

This project trains and evaluates reinforcement learning traffic-light agents on CityFlow scenarios, focused on a single intersection. The implemented algorithms are DQN, PPO, and A2C using Stable-Baselines3.

## What Is Implemented

- Custom Gym-compatible environment for one real intersection: `sample-code/cityflow_env.py`
- Baseline simulation with fixed signal plan: `sample-code/run_default_plan.py`
- Training scripts:
	- `sample-code/train_dqn.py`
	- `sample-code/train_ppo.py`
	- `sample-code/train_a2c.py`
- Evaluation script for saved models: `sample-code/evaluate_rl.py`
- Multi-model evaluator and JSON exporter: `sample-code/evaluate_all_models.py`
- Continuation/tuning scripts:
	- `sample-code/train_ppo_tuned.py`
	- `sample-code/train_a2c_continue.py`
	- `sample-code/train_dqn_continue.py`
- Scenario inspection helper: `sample-code/inspect_scenario.py`
- Environment smoke test script: `sample-code/test_env.py`

## Repository Layout

- `sample-code/data/`: CityFlow scenarios and configs
- `sample-code/models/`: saved RL checkpoints (`.zip`)
- `sample-code/logs/`: TensorBoard training logs
- `Dockerfile.rl`: reproducible Python 3.10 + CityFlow + SB3 runtime

## Environment Notes

CityFlow installation can fail on newer Python versions. Use one of these paths:

- Preferred: Docker (stable and reproducible)
- Alternative: local Python 3.10 virtual environment

## Quick Start (Docker, Recommended)

Build image:

```bash
docker build -f Dockerfile.rl -t atsc-rl .
```

Run scripts from repo by mounting the project folder:

```bash
docker run --rm -it -v "$PWD":/workspace -w /workspace/sample-code atsc-rl bash
```

Inside container, examples:

```bash
python run_default_plan.py --scenario hangzhou_1x1_bc-tyc_18041607_1h
python train_ppo.py --scenario hangzhou_1x1_bc-tyc_18041607_1h --config data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json --total_timesteps 100000
python evaluate_rl.py --algo ppo --model_path models/ppo_hangzhou_1x1_bc-tyc_18041607_1h.zip --scenario hangzhou_1x1_bc-tyc_18041607_1h --config data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json
```

## Local Setup (Python 3.10)

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

If CityFlow fails locally, use Docker.

## Training and Evaluation Workflow

From `sample-code/`:

1. Run baseline fixed signal:

```bash
python run_default_plan.py --scenario hangzhou_1x1_bc-tyc_18041607_1h
```

2. Train one or more RL agents:

```bash
python train_dqn.py --scenario hangzhou_1x1_bc-tyc_18041607_1h --config data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json
python train_ppo.py --scenario hangzhou_1x1_bc-tyc_18041607_1h --config data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json
python train_a2c.py --scenario hangzhou_1x1_bc-tyc_18041607_1h --config data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json
```

3. Evaluate saved models:

```bash
python evaluate_rl.py --algo dqn --model_path models/dqn_hangzhou_1x1_bc-tyc_18041607_1h.zip --scenario hangzhou_1x1_bc-tyc_18041607_1h --config data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json
python evaluate_rl.py --algo ppo --model_path models/ppo_hangzhou_1x1_bc-tyc_18041607_1h.zip --scenario hangzhou_1x1_bc-tyc_18041607_1h --config data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json
python evaluate_rl.py --algo a2c --model_path models/a2c_hangzhou_1x1_bc-tyc_18041607_1h.zip --scenario hangzhou_1x1_bc-tyc_18041607_1h --config data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json
```

## Current Status Summary

- Single-intersection environment, reward, and policies are implemented.
- DQN/PPO/A2C checkpoints already exist in `sample-code/models/`.
- TensorBoard logs already exist in `sample-code/logs/`.
- Main remaining work is improving performance through hyperparameter tuning, reward shaping, and multi-seed evaluation.

## Project Management Plan

Our project follows a structured 6-week plan divided into 4 phases:

### Phase 1: Data Preprocessing (Week 1)
**Objective**: Convert raw traffic data to simulator-suitable format and set up environment.
- **Status**: ✅ COMPLETE
- Loaded and validated CityFlow traffic scenarios (Hangzhou, Atlanta, Manhattan, synthetic variants).
- Built custom Gym-compatible environment (`sample-code/cityflow_env.py`) wrapping CityFlow simulator.
- Defined observation space (incoming lane queue counts + one-hot phase encoding).
- Defined action space (discrete traffic light phases).
- Implemented baseline fixed-signal-plan simulation (`sample-code/run_default_plan.py`) for comparison.

### Phase 2: Define Control Policies & Reward Functions (Week 1)
**Objective**: Design control policies and reward functions for DQN, PPO, and A2C algorithms.
- **Status**: ✅ COMPLETE
- **Reward function**: Minimize total waiting vehicle count at the intersection.
- **Control policies**: Implemented for all three algorithms using Stable-Baselines3.
- **Hyperparameters**:
  - DQN: learning_rate=1e-4, buffer_size=50k, batch_size=64, gamma=0.99
  - PPO: learning_rate=3e-4, n_steps=512, gamma=0.99, gae_lambda=0.95
  - A2C: learning_rate=7e-4, gamma=0.99, gae_lambda=1.0
- Created evaluation pipeline measuring mean reward, queue length, and travel time.

### Phase 3: Training & Testing on Single Intersection (Weeks 3–4)
**Objective**: Train and test RL models on single-intersection scenarios; identify best performer.
- **Status**: ✅ COMPLETE
- **Training**:
  - Trained all three models (DQN, PPO, A2C) for 100k timesteps each.
  - Used Hangzhou 1x1 intersection scenario for primary training.
  - Saved trained checkpoints in `sample-code/models/`.
- **Testing & Evaluation**:
  - Evaluated all models over 3 deterministic episodes.
  - **Results**:
    - **A2C (BEST)**: Mean reward = -1.0083, Queue = 8.07, Travel time = 83.5s ⭐
    - **PPO**: Mean reward = -1.2071, Queue = 9.66, Travel time = 86.6s
    - **DQN**: Mean reward = -1.7026, Queue = 13.62, Travel time = 100.9s
- Generated TensorBoard logs tracking training progress for all models.
- Confirmed A2C as best baseline for single-intersection control.

### Phase 4: Training & Testing on Complex Intersections (Weeks 5–6)
**Objective**: Scale best model to multi-intersection scenarios and improve performance.
- **Status**: 🔄 IN PROGRESS / ⏳ PLANNED
- **Current work (Week 5)**:
  - Conducted improvement experiments:
    - PPO tuning (new hyperparams) → Performance degraded, not recommended.
    - A2C continuation → Performance degraded, not recommended.
    - DQN continuation → Performance degraded, not recommended.
  - Decision: Keep original A2C as baseline; focus on reward redesign rather than retraining.
  - Documented codebase thoroughly and cleaned workspace.
- **Planned work (Week 6)**:
  - **Test on complex intersections**: 2x2 (Hangzhou 4x4 scenarios available in `sample-code/data/`).
  - **Reward function improvements**: Add penalty terms for phase switches and vehicle delays.
  - **Hyperparameter tuning**: Grid search on A2C learning rate and entropy coefficient.
  - **Multi-seed evaluation**: Run 5+ seeds for statistically robust performance estimate.
  - **Generate final leaderboard**: Compare all models across single/multiple intersections.
  - **Export results**: JSON/CSV summary of all experiments and best configuration.
  - **Metrics focus**: Mean queue length, average travel time, convergence speed.

## Plain-Language Snapshot

- Goal: use RL to control one traffic signal better than a fixed signal plan.
- What works now: training, evaluation, and comparison for DQN, PPO, and A2C on one intersection.
- Best result so far: original A2C model.
- What did not improve: tuned PPO and continued-training variants in the last run.

## Measured Results (Hangzhou 1x1)

Using 3 deterministic evaluation episodes from `sample-code/evaluate_all_models.py`:

| Model | Mean Reward (higher better) | Mean Queue (lower better) | Final Avg Travel Time (lower better) |
|---|---:|---:|---:|
| DQN (original) | -1.7026 | 13.6208 | 100.9892 |
| PPO (original) | -1.2071 | 9.6569 | 86.6261 |
| A2C (original) | -1.0083 | 8.0667 | 83.5011 |
| PPO tuned (new) | -9.0469 | 72.3750 | 296.6677 |

Current best checkpoint is still A2C original:

- `sample-code/models/a2c_hangzhou_1x1_bc-tyc_18041607_1h.zip`

### Notes on additional experiments

- PPO tuned checkpoint and best-callback checkpoint were both significantly worse than baseline PPO.
- A2C continuation and DQN continuation were also worse than their original checkpoints.
- Recommendation: keep original A2C as production baseline and run further tuning with reward shaping + multi-seed validation.

## Suggested Next Improvements

- Add a single script to evaluate all models and write a JSON/CSV leaderboard.
- Add tuned variants (especially PPO) with larger rollout horizon and longer training.
- Add repeated evaluation over multiple seeds for stable comparison.
- Add reward ablations (queue-only vs queue+delay+phase-change penalties).

## Codebase Health Check (March 30, 2026)

- Workspace diagnostics report no editor/lint problems.
- Python syntax compile pass succeeds for `sample-code/`.
- A duplicate typo file `sample-code/evalulate_rl.py` was removed to avoid confusion with `sample-code/evaluate_rl.py`.
- `requirements.txt` is currently empty, so local setup instructions that use `pip install -r requirements.txt` will not install dependencies; Docker workflow remains the reliable path.
