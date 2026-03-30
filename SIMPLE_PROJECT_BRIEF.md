# Simple Project Brief

## 1) What was the problem?

We wanted to reduce traffic congestion at one intersection.

The fixed traffic-light plan is not smart enough for changing traffic.
So we tried Reinforcement Learning (RL) to choose better traffic-light phases.

## 2) What did we do?

- Built a custom CityFlow environment for one real intersection.
- Trained 3 RL models:
  - DQN
  - PPO
  - A2C
- Added scripts to evaluate all models together.
- Tried improvement experiments:
  - Tuned PPO training
  - Continued training from old A2C model
  - Continued training from old DQN model
- Documented full setup and workflow in README.

## 3) What are the results?

From current evaluated checkpoints:

- Best model: original A2C
- Second: original PPO
- Third: original DQN

Recent improvement attempts did not beat the original A2C.
So right now we should keep original A2C as the main baseline.

The A2C metrics tell you how well the traffic light control model performed:

Mean reward = -1.0083: The reward function penalizes waiting vehicles (negative queue count). A value of -1.0083 is good because it means the model kept the queue small. (Negative rewards are intentional here—the goal is to minimize waiting, so the reward reflects that.)

Queue = 8.07: On average, 8 vehicles were waiting at the intersection during testing. This is the key metric for traffic efficiency—fewer waiting vehicles = better traffic flow.

Travel time = 83.5s: Vehicles spent an average of 83.5 seconds from arrival to departure at the intersection. This is how long it takes a car to completely pass through or leave the intersection area.

In practical terms: The A2C model got traffic moving reasonably well—only ~8 cars stacking up and taking ~84 seconds per vehicle on average. This beat the other algorithms (PPO had 9.67 cars waiting, DQN had 13.6), making A2C the best performer for this single intersection.

## 4) What do we do next?

1. Keep original A2C as the reference model.
2. Run multi-seed evaluation (not just one run) for fair comparison.
3. Improve reward design (queue + delay + phase-switch penalty).
4. Tune one algorithm at a time with clear experiment tracking.
5. Export a final leaderboard CSV/JSON for report-ready results.

## 5) Codebase health check (quick)

- No syntax errors found in Python files.
- No editor problem diagnostics found.
- Removed duplicate typo file (`evalulate_rl.py`) to avoid confusion.
- Note: `requirements.txt` is empty right now, so local pip setup is incomplete unless Docker is used.
