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
