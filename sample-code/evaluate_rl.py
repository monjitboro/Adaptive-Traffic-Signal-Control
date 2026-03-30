import argparse
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
args = parser.parse_args()

env = CityFlowSingleIntersectionEnv(
    config_path=args.config,
    scenario_name=args.scenario,
    num_step=args.num_step,
    decision_interval=args.decision_interval
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
    rewards.append(reward)
    queues.append(info["queue"])
    vehicles.append(info["vehicles"])

print("=" * 60)
print("EVALUATION RESULTS")
print("=" * 60)
print("Scenario              :", args.scenario)
print("Algorithm             :", args.algo)
print("Mean reward           :", float(np.mean(rewards)))
print("Mean queue            :", float(np.mean(queues)))
print("Mean vehicles         :", float(np.mean(vehicles)))
print("Final avg travel time :", float(info["avg_travel_time"]))
print("=" * 60)