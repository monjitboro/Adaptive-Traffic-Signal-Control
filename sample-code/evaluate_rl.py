import argparse
import json
import os
import numpy as np

from stable_baselines3 import PPO, DQN, A2C
from cityflow_env import CityFlowSingleIntersectionEnv

parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, choices=["ppo", "dqn", "a2c"], required=True)
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--scenario", type=str, required=True)
parser.add_argument("--config", type=str, required=True)
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
parser.add_argument("--out", type=str, default="")
args = parser.parse_args()

env = CityFlowSingleIntersectionEnv(
    config_path=args.config,
    scenario_name=args.scenario,
    num_step=args.num_step,
    decision_interval=args.decision_interval,
    reward_mode=args.reward_mode,
    reward_delay_coef=args.reward_delay_coef,
    reward_switch_coef=args.reward_switch_coef,
)

if args.algo == "ppo":
    model = PPO.load(args.model_path)
elif args.algo == "dqn":
    model = DQN.load(args.model_path)
else:
    model = A2C.load(args.model_path)

obs, info = env.reset()

done = False
rewards = []
queues = []
vehicles = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    rewards.append(float(reward))
    queues.append(float(info["queue"]))
    vehicles.append(float(info["vehicles"]))

results = {
    "scenario": args.scenario,
    "algorithm": args.algo,
    "reward_mode": args.reward_mode,
    "reward_delay_coef": args.reward_delay_coef,
    "reward_switch_coef": args.reward_switch_coef,
    "mean_reward": float(np.mean(rewards)),
    "mean_queue": float(np.mean(queues)),
    "mean_vehicles": float(np.mean(vehicles)),
    "final_avg_travel_time": float(info["avg_travel_time"]),
}

if args.out:
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print("Scenario              :", args.scenario)
print("Algorithm             :", args.algo)
print("Reward mode           :", args.reward_mode)
print("Mean reward           :", results["mean_reward"])
print("Mean queue            :", results["mean_queue"])
print("Mean vehicles         :", results["mean_vehicles"])
print("Final avg travel time :", results["final_avg_travel_time"])
if args.out:
    print("Saved JSON            :", args.out)
print("=" * 60)
