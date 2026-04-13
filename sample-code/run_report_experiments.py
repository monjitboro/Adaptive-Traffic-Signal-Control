import argparse
import csv
import json
import os
import shlex
import subprocess
import sys


PHASE1_ALGOS = ("dqn", "ppo", "a2c")
PHASE2_REWARD_MODES = ("queue", "queue_delay", "queue_switch", "hybrid")


def quote_cmd(cmd):
    return " ".join(shlex.quote(part) for part in cmd)


def run_or_print(cmd, execute):
    print(quote_cmd(cmd))
    if execute:
        subprocess.run(cmd, check=True)


def write_phase1_csv(json_path, csv_path):
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    rows = []
    for algo in PHASE1_ALGOS:
        algo_result = results.get(algo, {})
        if "error" in algo_result:
            rows.append({
                "algorithm": algo,
                "reward_mode": results["_meta"]["reward_mode"],
                "mean_reward": "",
                "mean_queue": "",
                "mean_vehicles": "",
                "final_avg_travel_time": "",
                "error": algo_result["error"],
            })
            continue

        rows.append({
            "algorithm": algo,
            "reward_mode": results["_meta"]["reward_mode"],
            "mean_reward": algo_result["mean_reward"],
            "mean_queue": algo_result["mean_queue"],
            "mean_vehicles": algo_result["mean_vehicles"],
            "final_avg_travel_time": algo_result["final_avg_travel_time"],
            "error": "",
        })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "algorithm",
                "reward_mode",
                "mean_reward",
                "mean_queue",
                "mean_vehicles",
                "final_avg_travel_time",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_phase2_csv(json_path, csv_path):
    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "reward_mode",
                "mean_reward",
                "mean_queue",
                "mean_vehicles",
                "final_avg_travel_time",
                "error",
            ],
        )
        writer.writeheader()
        for reward_mode, reward_result in results.items():
            if "error" in reward_result:
                writer.writerow({
                    "reward_mode": reward_mode,
                    "mean_reward": "",
                    "mean_queue": "",
                    "mean_vehicles": "",
                    "final_avg_travel_time": "",
                    "error": reward_result["error"],
                })
                continue

            writer.writerow({
                "reward_mode": reward_mode,
                "mean_reward": reward_result["mean_reward"],
                "mean_queue": reward_result["mean_queue"],
                "mean_vehicles": reward_result["mean_vehicles"],
                "final_avg_travel_time": reward_result["final_avg_travel_time"],
                "error": "",
            })


def build_common_args(args, reward_mode):
    cmd = [
        "--scenario", args.scenario,
        "--config", args.config,
        "--num_step", str(args.num_step),
        "--decision_interval", str(args.decision_interval),
        "--reward_mode", reward_mode,
        "--reward_delay_coef", str(args.reward_delay_coef),
        "--reward_switch_coef", str(args.reward_switch_coef),
    ]
    if args.experiment_tag:
        cmd.extend(["--experiment_tag", args.experiment_tag])
    return cmd


def phase1(args):
    print("Phase 1: Compare DQN, PPO, and A2C under the same queue-only reward.")

    common_args = build_common_args(args, "queue")
    for algo in PHASE1_ALGOS:
        train_cmd = [
            sys.executable,
            f"train_{algo}.py",
            "--total_timesteps", str(args.total_timesteps),
            *common_args,
        ]
        run_or_print(train_cmd, args.execute)

    out_json = os.path.join("logs", f"phase1_{args.scenario}_summary.json")
    out_csv = os.path.join("logs", f"phase1_{args.scenario}_summary.csv")
    eval_cmd = [
        sys.executable,
        "evaluate_all_models.py",
        "--episodes", str(args.episodes),
        "--out", out_json,
        *common_args,
    ]
    run_or_print(eval_cmd, args.execute)

    if args.execute and os.path.exists(out_json):
        write_phase1_csv(out_json, out_csv)
        print("Saved CSV:", out_csv)


def phase2(args):
    print("Phase 2: Compare multiple reward functions on the best baseline algorithm (A2C).")

    summary = {}
    for reward_mode in PHASE2_REWARD_MODES:
        common_args = build_common_args(args, reward_mode)
        train_cmd = [
            sys.executable,
            "train_a2c.py",
            "--total_timesteps", str(args.total_timesteps),
            *common_args,
        ]
        run_or_print(train_cmd, args.execute)

        experiment_name_parts = [args.scenario]
        if reward_mode != "queue":
            experiment_name_parts.append(reward_mode)
        if args.experiment_tag:
            experiment_name_parts.append(args.experiment_tag)
        experiment_name = "_".join(experiment_name_parts)
        model_path = os.path.join("models", f"a2c_{experiment_name}.zip")
        out_path = os.path.join("logs", f"phase2_a2c_{experiment_name}.json")

        eval_cmd = [
            sys.executable,
            "evaluate_rl.py",
            "--algo", "a2c",
            "--model_path", model_path,
            "--out", out_path,
            *common_args,
        ]
        run_or_print(eval_cmd, args.execute)

        if args.execute and os.path.exists(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                summary[reward_mode] = json.load(f)
        else:
            summary[reward_mode] = {
                "error": "Run with --execute to generate results."
            }

    summary_json = os.path.join("logs", f"phase2_a2c_{args.scenario}_reward_summary.json")
    summary_csv = os.path.join("logs", f"phase2_a2c_{args.scenario}_reward_summary.csv")

    if args.execute:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        write_phase2_csv(summary_json, summary_csv)
        print("Saved JSON:", summary_json)
        print("Saved CSV:", summary_csv)


def main():
    parser = argparse.ArgumentParser(
        description="Run report-ready experiment phases for single-intersection RL."
    )
    parser.add_argument("--phase", type=str, choices=["phase1", "phase2", "all"], default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
    parser.add_argument("--config", type=str, default="data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json")
    parser.add_argument("--total_timesteps", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--num_step", type=int, default=3600)
    parser.add_argument("--decision_interval", type=int, default=5)
    parser.add_argument("--reward_delay_coef", type=float, default=0.5)
    parser.add_argument("--reward_switch_coef", type=float, default=0.1)
    parser.add_argument("--experiment_tag", type=str, default="")
    args = parser.parse_args()

    if args.phase in ("phase1", "all"):
        phase1(args)
    if args.phase in ("phase2", "all"):
        phase2(args)


if __name__ == "__main__":
    main()
