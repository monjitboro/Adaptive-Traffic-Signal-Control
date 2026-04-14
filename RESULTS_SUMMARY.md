# Results Summary

This file summarizes the completed experiment run from the generated outputs in `sample-code/logs/`.

## Scenario

- Scenario: `hangzhou_1x1_bc-tyc_18041607_1h`
- Evaluation: 3 deterministic episodes per model
- Execution path: Docker run using `sample-code/run_report_experiments.py --phase all --execute`

## Phase 1: Algorithm Comparison

Source:

- `sample-code/logs/phase1_hangzhou_1x1_bc-tyc_18041607_1h_summary.json`
- `sample-code/logs/phase1_hangzhou_1x1_bc-tyc_18041607_1h_summary.csv`

All three algorithms were trained and evaluated using the same queue-only reward.

| Model | Mean Reward | Mean Queue | Mean Vehicles | Final Avg Travel Time |
|---|---:|---:|---:|---:|
| DQN | -1.7799 | 14.2389 | 36.3069 | 103.1061 |
| PPO | -1.2510 | 10.0083 | 29.7556 | 89.6347 |
| A2C | -1.2087 | 9.6694 | 29.6514 | 89.0352 |

Phase 1 conclusion:

- A2C was the best algorithm on this scenario under the same queue-based reward.
- PPO was the second-best algorithm.
- DQN had the worst queue and travel-time performance.

## Phase 2: Reward Function Comparison on A2C

Source:

- `sample-code/logs/phase2_a2c_hangzhou_1x1_bc-tyc_18041607_1h_reward_summary.json`
- `sample-code/logs/phase2_a2c_hangzhou_1x1_bc-tyc_18041607_1h_reward_summary.csv`

A2C was retrained using multiple reward modes.

| Reward Mode | Mean Reward | Mean Queue | Mean Vehicles | Final Avg Travel Time |
|---|---:|---:|---:|---:|
| queue | -1.2944 | 10.3556 | 30.2597 | 90.0352 |
| queue_delay | -1.5055 | 10.8556 | 30.4694 | 90.4026 |
| queue_switch | -1.1715 | 8.7472 | 27.9181 | 85.3994 |
| hybrid | -1.1822 | 7.8431 | 26.8528 | 83.2976 |

Phase 2 conclusion:

- The `hybrid` reward produced the best traffic result overall.
- `queue_switch` also improved over queue-only reward.
- `queue_delay` was worse than queue-only in this experiment.

## Honest Interpretation

- The project now shows two clear results:
  - best algorithm under the same reward: `A2C`
  - best reward design on A2C: `hybrid`
- Mean reward values should not be compared across different reward modes as if they were the same metric, because each reward mode uses a different formula.
- For reward-ablation conclusions, queue length and travel time are the safer metrics.

## Fixed-Signal Baseline

Source:

- `sample-code/logs/fixed_signal_baseline_hangzhou_1x1_bc-tyc_18041607_1h.json`

The fixed traffic-light plan was evaluated with comparable incoming-lane queue and travel-time metrics.

| Controller | Comparable Mean Reward (queue-only) | Mean Queue | Mean Vehicles | Final Avg Travel Time |
|---|---:|---:|---:|---:|
| Fixed signal plan | -6.6407 | 53.1256 | 71.3964 | 385.1585 |
| A2C with queue reward | -1.2087 | 9.6694 | 29.6514 | 89.0352 |
| A2C with hybrid reward | -1.1822* | 7.8431 | 26.8528 | 83.2976 |

\* The hybrid reward uses a different formula, so queue and travel time are the fair comparison metrics.

Baseline comparison conclusion:

- RL clearly outperformed the original fixed traffic-signal plan on this scenario.
- A2C with queue reward reduced mean queue by about **81.8%** and reduced travel time by about **76.9%** relative to the fixed plan.
- A2C with hybrid reward reduced mean queue by about **85.2%** and reduced travel time by about **78.4%** relative to the fixed plan.

## Final Takeaway

- Best algorithm under a shared queue-only reward: `A2C`
- Best reward design on A2C: `hybrid`
- Compared with the fixed signal baseline, the RL controller substantially improved queue length and travel time on the evaluated Hangzhou 1x1 scenario.
