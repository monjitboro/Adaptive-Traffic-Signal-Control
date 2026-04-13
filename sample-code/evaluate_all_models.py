import argparse
import json
import os
from statistics import mean

from stable_baselines3 import PPO, DQN, A2C

from cityflow_env import CityFlowSingleIntersectionEnv


def evaluate_one(model, env):
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

    return {
        "mean_reward": mean(rewards),
        "mean_queue": mean(queues),
        "mean_vehicles": mean(vehicles),
        "final_avg_travel_time": float(info["avg_travel_time"]),
    }


def run_eval(
    algo,
    model_path,
    config,
    scenario,
    num_step,
    decision_interval,
    episodes,
    reward_mode,
    reward_delay_coef,
    reward_switch_coef,
):
    env = CityFlowSingleIntersectionEnv(
        config_path=config,
        scenario_name=scenario,
        num_step=num_step,
        decision_interval=decision_interval,
        reward_mode=reward_mode,
        reward_delay_coef=reward_delay_coef,
        reward_switch_coef=reward_switch_coef,
    )

    if algo == "ppo":
        model = PPO.load(model_path)
    elif algo == "dqn":
        model = DQN.load(model_path)
    else:
        model = A2C.load(model_path)

    episode_results = [evaluate_one(model, env) for _ in range(episodes)]

    return {
        "episodes": episodes,
        "mean_reward": mean(r["mean_reward"] for r in episode_results),
        "mean_queue": mean(r["mean_queue"] for r in episode_results),
        "mean_vehicles": mean(r["mean_vehicles"] for r in episode_results),
        "final_avg_travel_time": mean(r["final_avg_travel_time"] for r in episode_results),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
    parser.add_argument("--config", type=str, default="data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num_step", type=int, default=3600)
    parser.add_argument("--decision_interval", type=int, default=5)
    parser.add_argument("--out", type=str, default="logs/eval_summary.json")
    parser.add_argument("--extra_ppo_model", type=str, default="")
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

    name_parts = [args.scenario]
    if args.reward_mode != "queue":
        name_parts.append(args.reward_mode)
    if args.experiment_tag:
        name_parts.append(args.experiment_tag)
    experiment_name = "_".join(name_parts)

    model_specs = {
        "dqn": os.path.join(args.models_dir, f"dqn_{experiment_name}.zip"),
        "ppo": os.path.join(args.models_dir, f"ppo_{experiment_name}.zip"),
        "a2c": os.path.join(args.models_dir, f"a2c_{experiment_name}.zip"),
    }

    results = {
        "_meta": {
            "scenario": args.scenario,
            "reward_mode": args.reward_mode,
            "reward_delay_coef": args.reward_delay_coef,
            "reward_switch_coef": args.reward_switch_coef,
            "experiment_name": experiment_name,
        }
    }
    for algo, model_path in model_specs.items():
        if not os.path.exists(model_path):
            results[algo] = {"error": f"missing model: {model_path}"}
            continue

        results[algo] = run_eval(
            algo=algo,
            model_path=model_path,
            config=args.config,
            scenario=args.scenario,
            num_step=args.num_step,
            decision_interval=args.decision_interval,
            episodes=args.episodes,
            reward_mode=args.reward_mode,
            reward_delay_coef=args.reward_delay_coef,
            reward_switch_coef=args.reward_switch_coef,
        )

    if args.extra_ppo_model:
        if os.path.exists(args.extra_ppo_model):
            results["ppo_tuned"] = run_eval(
                algo="ppo",
                model_path=args.extra_ppo_model,
                config=args.config,
                scenario=args.scenario,
                num_step=args.num_step,
                decision_interval=args.decision_interval,
                episodes=args.episodes,
                reward_mode=args.reward_mode,
                reward_delay_coef=args.reward_delay_coef,
                reward_switch_coef=args.reward_switch_coef,
            )
        else:
            results["ppo_tuned"] = {"error": f"missing model: {args.extra_ppo_model}"}

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("MULTI-MODEL EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    print("Saved:", args.out)


if __name__ == "__main__":
    main()
