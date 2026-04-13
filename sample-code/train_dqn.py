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
parser.add_argument(
    "--reward_mode",
    type=str,
    default="queue",
    choices=CityFlowSingleIntersectionEnv.SUPPORTED_REWARD_MODES,
)
parser.add_argument("--reward_delay_coef", type=float, default=0.5)
parser.add_argument("--reward_switch_coef", type=float, default=0.1)
parser.add_argument("--experiment_tag", type=str, default="")
args = parser.parse_args()

os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

name_parts = [args.scenario]
if args.reward_mode != "queue":
    name_parts.append(args.reward_mode)
if args.experiment_tag:
    name_parts.append(args.experiment_tag)
experiment_name = "_".join(name_parts)

env = CityFlowSingleIntersectionEnv(
    config_path=args.config,
    scenario_name=args.scenario,
    num_step=args.num_step,
    decision_interval=args.decision_interval,
    reward_mode=args.reward_mode,
    reward_delay_coef=args.reward_delay_coef,
    reward_switch_coef=args.reward_switch_coef,
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

model.learn(total_timesteps=args.total_timesteps, tb_log_name=experiment_name)
model_path = f"models/dqn_{experiment_name}"
model.save(model_path)
print("Reward mode:", args.reward_mode)
print("Saved:", model_path)
