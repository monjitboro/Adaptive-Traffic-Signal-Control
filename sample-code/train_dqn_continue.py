import argparse
import os

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor

from cityflow_env import CityFlowSingleIntersectionEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
    parser.add_argument("--config", type=str, default="data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json")
    parser.add_argument("--base_model", type=str, default="models/dqn_hangzhou_1x1_bc-tyc_18041607_1h.zip")
    parser.add_argument("--extra_timesteps", type=int, default=120000)
    parser.add_argument("--num_step", type=int, default=3600)
    parser.add_argument("--decision_interval", type=int, default=5)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    env = CityFlowSingleIntersectionEnv(
        config_path=args.config,
        scenario_name=args.scenario,
        num_step=args.num_step,
        decision_interval=args.decision_interval,
    )
    env = Monitor(env)

    model = DQN.load(args.base_model, env=env)

    model.learn(total_timesteps=args.extra_timesteps, reset_num_timesteps=False)

    out_path = f"models/dqn_continued_{args.scenario}"
    model.save(out_path)
    print("Saved continued model:", out_path)


if __name__ == "__main__":
    main()
