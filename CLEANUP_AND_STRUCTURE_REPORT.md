
## Final Directory Structure

```
Adaptive-Traffic-Signal-Control/
├── README.md                          # Main documentation (with iterations 1-5)
├── SIMPLE_PROJECT_BRIEF.md            # Quick 1-page summary
├── .gitignore                         # Updated ignore patterns
├── Dockerfile.rl                      # Docker runtime (Python 3.10 + CityFlow)
├── LICENSE                            # Project license
├── requirements.txt                   # (Currently empty) Dependencies
├── Final project proposal template-Spring26.pdf  # Project brief
│
└── sample-code/                       # Main codebase
    ├── SOURCE CODE (fixed)
    │   ├── cityflow_env.py            # Custom gym environment
    │   ├── run_default_plan.py        # Baseline fixed-signal simulation
    │   ├── test_env.py                # Smoke test
    │   ├── inspect_scenario.py        # Scenario inspector
    │   │
    │   ├── TRAINING SCRIPTS
    │   ├── train_dqn.py               # DQN training (100k steps)
    │   ├── train_ppo.py               # PPO training (100k steps)
    │   ├── train_a2c.py               # A2C training (100k steps)
    │   │
    │   ├── EVALUATION SCRIPTS
    │   ├── evaluate_rl.py             # Single-model evaluator
    │   └── evaluate_all_models.py     # Multi-model JSON exporter
    │   
    │   ├── EXPERIMENTS (for reference, not recommended)
    │   ├── train_ppo_tuned.py         # PPO tuned (regressed)
    │   ├── train_a2c_continue.py      # A2C continuation (regressed)
    │   └── train_dqn_continue.py      # DQN continuation (regressed)
    │
    ├── data/                          # CityFlow scenarios (20+ variants)
    │   ├── hangzhou_1x1_bc-tyc_18041607_1h/  # ✅ Recommended default
    │   ├── hangzhou_4x4_gudang_18041610_1h/  # Larger intersection
    │   ├── atlanta_1x5/
    │   ├── manhattan_*/ and synth_*/ (various)
    │   └── [each has: roadnet.json, flow.json, config.json]
    │
    ├── models/ (1.5MB total)          # ✅ All original checkpoints
    │   ├── a2c_hangzhou_1x1_bc-tyc_18041607_1h.zip        # ⭐ BEST
    │   ├── ppo_hangzhou_1x1_bc-tyc_18041607_1h.zip        # 2nd
    │   ├── dqn_hangzhou_1x1_bc-tyc_18041607_1h.zip        # 3rd
    │   ├── a2c_continued_hangzhou_1x1_bc-tyc_18041607_1h.zip   # (worse)
    │   ├── dqn_continued_hangzhou_1x1_bc-tyc_18041607_1h.zip   # (worse)
    │   ├── ppo_tuned_hangzhou_1x1_bc-tyc_18041607_1h.zip       # (worse)
    │   └── ppo_tuned_best_hangzhou_1x1_bc-tyc_18041607_1h/     # (worse)
    │
    └── logs/ (800KB total)            # ✅ Training & eval logs
        ├── dqn/                       # DQN training logs
        ├── ppo/                       # PPO training logs
        ├── a2c/                       # A2C training logs
        ├── a2c_continue/              # Continuation experiment logs
        ├── ppo_tuned/                 # Tuned PPO logs
        ├── ppo_tuned_eval_*/          # Tuned PPO eval logs
        └── eval_summary_*.json        # Evaluation results
```

