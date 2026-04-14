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

## Report-Ready Experiment Plan

For the final report, the project now supports two clean experiment phases:

### Phase 1: Fair Algorithm Comparison

- Train **DQN, PPO, and A2C** with the same queue-only reward.
- This isolates the effect of the RL algorithm.
- Use this phase to answer: **which algorithm performs best under the same objective?**

### Phase 2: Reward Function Comparison

- Keep the best baseline algorithm (**A2C**) fixed.
- Retrain A2C with multiple reward functions:
  - `queue`: queue-only baseline
  - `queue_delay`: queue + delay proxy penalty
  - `queue_switch`: queue + phase-switch penalty
  - `hybrid`: queue + delay proxy + phase-switch penalty
- This isolates the effect of reward design.
- Use this phase to answer: **which reward function gives the best traffic behavior?**

The reward logic is now configurable in `sample-code/cityflow_env.py`, and the report runner is:

```bash
cd sample-code
python run_report_experiments.py --phase phase1
python run_report_experiments.py --phase phase2
```

To actually run the experiments instead of only printing commands:

```bash
python run_report_experiments.py --phase all --execute
```

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
- Added reward-ablation support for report experiments (`queue`, `queue_delay`, `queue_switch`, `hybrid`).

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
    - **A2C (BEST)**: Mean reward = -1.2087, Queue = 9.67, Travel time = 89.0s ⭐
    - **PPO**: Mean reward = -1.2510, Queue = 10.01, Travel time = 89.6s
    - **DQN**: Mean reward = -1.7799, Queue = 14.24, Travel time = 103.1s
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
- What works now: Phase 1 algorithm comparison and Phase 2 reward-ablation experiments on one intersection.
- Best algorithm under the same queue-only reward: A2C.
- Best reward design on A2C so far: hybrid reward (queue + delay proxy + phase-switch penalty).
- Fixed-signal baseline comparison is now available and shows a large improvement from RL on this scenario.
- What still needs to be added for a stronger claim: multi-seed validation.

## Measured Results (Hangzhou 1x1, Completed Docker Run)

### Phase 1: Algorithm Comparison Under Queue-Only Reward

Using 3 deterministic evaluation episodes from `sample-code/logs/phase1_hangzhou_1x1_bc-tyc_18041607_1h_summary.json`:

| Model | Mean Reward (higher better) | Mean Queue (lower better) | Final Avg Travel Time (lower better) |
|---|---:|---:|---:|
| DQN | -1.7799 | 14.2389 | 103.1061 |
| PPO | -1.2510 | 10.0083 | 89.6347 |
| A2C | -1.2087 | 9.6694 | 89.0352 |

Phase 1 conclusion:

- A2C was the best algorithm when all three models used the same queue-only reward.
- PPO was close, but A2C had both lower queue and lower travel time.
- DQN was clearly worse on this scenario.

### Phase 2: Reward Ablation on A2C

Using `sample-code/logs/phase2_a2c_hangzhou_1x1_bc-tyc_18041607_1h_reward_summary.json`:

| Reward Mode | Mean Reward | Mean Queue (lower better) | Final Avg Travel Time (lower better) |
|---|---:|---:|---:|
| queue | -1.2944 | 10.3556 | 90.0352 |
| queue_delay | -1.5055 | 10.8556 | 90.4026 |
| queue_switch | -1.1715 | 8.7472 | 85.3994 |
| hybrid | -1.1822 | 7.8431 | 83.2976 |

Phase 2 conclusion:

- The `hybrid` reward gave the best traffic outcome by queue length and travel time.
- `queue_switch` was second best and also improved over queue-only reward.
- `queue_delay` performed worse than queue-only on this setup.

### Fixed-Signal Baseline Comparison

Using `sample-code/logs/fixed_signal_baseline_hangzhou_1x1_bc-tyc_18041607_1h.json`:

| Controller | Comparable Mean Reward (queue-only) | Mean Queue (lower better) | Final Avg Travel Time (lower better) |
|---|---:|---:|---:|
| Fixed signal plan | -6.6407 | 53.1256 | 385.1585 |
| A2C with queue reward | -1.2087 | 9.6694 | 89.0352 |
| A2C with hybrid reward | -1.1822* | 7.8431 | 83.2976 |

\* The hybrid reward uses a different formula, so its reward value is not directly comparable to the fixed-signal queue-only reward. Queue and travel time are the reliable comparison metrics here.

Baseline comparison conclusion:

- The fixed signal plan performed much worse than the RL controllers on this scenario.
- Compared with the fixed plan, A2C under the queue-only setup reduced mean queue by about **81.8%** and reduced average travel time by about **76.9%**.
- Compared with the fixed plan, A2C with the hybrid reward reduced mean queue by about **85.2%** and reduced average travel time by about **78.4%**.
- This closes the earlier reporting gap: on this scenario, the RL controller did improve over the original fixed traffic-signal plan.

Important note:

- Mean reward values are only directly comparable within the same reward definition.
- Across different reward modes, the more reliable comparison is queue length and travel time.
- So the honest takeaway is:
  - best algorithm = A2C
  - best reward design = hybrid
  - RL beats the fixed-signal baseline on this scenario

## Suggested Next Improvements

- Repeat the fixed-signal baseline and RL evaluation over multiple seeds or repeated runs for stronger statistical support.
- Add repeated evaluation over multiple seeds for stable comparison.
- Tune the hybrid-reward A2C setup now that it is the best current configuration.
- Extend the same experiment framework to larger intersections.

## Reward Modes

The single-intersection environment supports these reward modes:

- `queue`: `-(average incoming-lane queue)`
- `queue_delay`: `-(queue + 0.5 * delay_proxy)`
- `queue_switch`: `-(queue + 0.1 * phase_change)`
- `hybrid`: `-(queue + 0.5 * delay_proxy + 0.1 * phase_change)`

Notes:

- `delay_proxy` is the ratio of queued vehicles to vehicles present on incoming lanes.
- `phase_change` is `1` when the agent changes the signal phase at a decision step, otherwise `0`.
- Default coefficients are `reward_delay_coef=0.5` and `reward_switch_coef=0.1`, both configurable from the CLI.

## Codebase Health Check (March 30, 2026)

- Workspace diagnostics report no editor/lint problems.
- Python syntax compile pass succeeds for `sample-code/`.
- A duplicate typo file `sample-code/evalulate_rl.py` was removed to avoid confusion with `sample-code/evaluate_rl.py`.
- `requirements.txt` is currently empty, so local setup instructions that use `pip install -r requirements.txt` will not install dependencies; Docker workflow remains the reliable path.
