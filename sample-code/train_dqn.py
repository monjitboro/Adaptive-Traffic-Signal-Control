import os
import argparse

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from cityflow_env import CityFlowSingleIntersectionEnv

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
parser.add_argument("--config", type=str, default="data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json")
parser.add_argument("--total_timesteps", type=int, default=100000)
parser.add_argument("--num_step", type=int, default=3600)
parser.add_argument("--decision_interval", type=int, default=5)
args = parser.parse_args()

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

env = CityFlowSingleIntersectionEnv(
    config_path=args.config,
    scenario_name=args.scenario,
    num_step=args.num_step,
    decision_interval=args.decision_interval
)
env = Monitor(env)

model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="logs/dqn/",
    learning_rate=1e-4,
    buffer_size=50000,
    learning_starts=2000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000
)

model.learn(total_timesteps=args.total_timesteps)
model.save(f"models/dqn_{args.scenario}")
print("Saved:", f"models/dqn_{args.scenario}")