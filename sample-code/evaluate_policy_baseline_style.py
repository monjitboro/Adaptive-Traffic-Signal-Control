import argparse
import json
import os

import cityflow
import numpy as np
from stable_baselines3 import A2C, DQN, PPO


DEFAULT_SCENARIO = "hangzhou_1x1_bc-tyc_18041607_1h"
DEFAULT_CONFIG = f"data/{DEFAULT_SCENARIO}/config_rl.json"
DEFAULT_MODEL_PATH = f"models/a2c_{DEFAULT_SCENARIO}_hybrid"
DEFAULT_OUT = (
    f"logs/model_metrics_{DEFAULT_SCENARIO}_a2c_hybrid_baseline_style.json"
)


def normalize_model_path(model_path):
    # Stable-Baselines3 accepts the path without ".zip" and may append it internally.
    if model_path.endswith(".zip"):
        return model_path[:-4]
    return model_path


def load_model(algo, model_path):
    model_path = normalize_model_path(model_path)
    if algo == "ppo":
        return PPO.load(model_path)
    if algo == "dqn":
        return DQN.load(model_path)
    return A2C.load(model_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ppo", "dqn", "a2c"], default="a2c")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scenario", type=str, default=DEFAULT_SCENARIO)
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--num_step", type=int, default=3600)
    parser.add_argument("--decision_interval", type=int, default=5)
    parser.add_argument("--label", type=str, default="a2c_hybrid_7am")
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    roadnet_path = os.path.join("data", args.scenario, "roadnet.json")
    with open(roadnet_path, "r", encoding="utf-8") as f:
        roadnet = json.load(f)

    real_intersection = next(
        inter for inter in roadnet["intersections"]
        if not inter.get("virtual", False)
    )
    intersection_id = real_intersection["id"]

    incoming_roads = [
        road for road in roadnet["roads"]
        if road["endIntersection"] == intersection_id
    ]
    incoming_lanes = []
    for road in incoming_roads:
        for lane_idx in range(len(road.get("lanes", []))):
            incoming_lanes.append(f"{road['id']}_{lane_idx}")

    phases = real_intersection.get("trafficLight", {}).get("lightphases", [])
    valid_phases = [
        idx for idx, ph in enumerate(phases)
        if len(ph.get("availableRoadLinks", [])) > 0
    ]
    if not valid_phases:
        valid_phases = list(range(len(phases)))

    eng = cityflow.Engine(args.config, thread_num=1)
    model = load_model(args.algo, args.model_path)

    current_phase_pos = 0
    eng.set_tl_phase(intersection_id, valid_phases[current_phase_pos])

    def get_obs():
        lane_waiting = eng.get_lane_waiting_vehicle_count()
        queue_feats = [float(lane_waiting.get(lane_id, 0)) for lane_id in incoming_lanes]
        phase_one_hot = np.zeros(len(valid_phases), dtype=np.float32)
        phase_one_hot[current_phase_pos] = 1.0
        return np.concatenate([
            np.array(queue_feats, dtype=np.float32),
            phase_one_hot,
        ]).astype(np.float32)

    all_waiting = []
    all_vehicles = []
    incoming_queue_history = []
    peak_waiting = 0
    peak_vehicles = 0
    current_step = 0
    obs = get_obs()

    while current_step < args.num_step:
        action, _ = model.predict(obs, deterministic=True)
        current_phase_pos = int(action)
        eng.set_tl_phase(intersection_id, valid_phases[current_phase_pos])

        for _ in range(args.decision_interval):
            if current_step >= args.num_step:
                break
            eng.next_step()
            current_step += 1
            lane_waiting = eng.get_lane_waiting_vehicle_count()
            lane_vehicles = eng.get_lane_vehicle_count()
            total_waiting_now = float(sum(lane_waiting.values()))
            total_vehicles_now = float(sum(lane_vehicles.values()))
            incoming_queue_now = float(sum(lane_waiting.get(lane_id, 0) for lane_id in incoming_lanes))
            all_waiting.append(total_waiting_now)
            all_vehicles.append(total_vehicles_now)
            incoming_queue_history.append(incoming_queue_now)
            peak_waiting = max(peak_waiting, int(total_waiting_now))
            peak_vehicles = max(peak_vehicles, int(total_vehicles_now))

        obs = get_obs()

    avg_waiting = sum(all_waiting) / len(all_waiting)
    avg_vehicles = sum(all_vehicles) / len(all_vehicles)
    mean_incoming_queue = sum(incoming_queue_history) / len(incoming_queue_history)
    congestion_rate = (avg_waiting / avg_vehicles * 100.0) if avg_vehicles > 0 else 0.0
    final_avg_travel_time = float(eng.get_average_travel_time())
    queue_reward_equivalent = -mean_incoming_queue / max(1, len(incoming_lanes))

    results = {
        "scenario": args.scenario,
        "controller": args.label or f"{args.algo}_model",
        "model_path": normalize_model_path(args.model_path),
        "incoming_lanes": len(incoming_lanes),
        "avg_waiting_all_lanes": avg_waiting,
        "avg_vehicles_all_lanes": avg_vehicles,
        "congestion_rate_percent": congestion_rate,
        "peak_waiting_all_lanes": peak_waiting,
        "peak_vehicles_all_lanes": peak_vehicles,
        "mean_queue": mean_incoming_queue,
        "mean_reward_queue_only_equivalent": queue_reward_equivalent,
        "final_avg_travel_time": final_avg_travel_time,
    }

    if args.out:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    if args.out:
        print("Saved:", args.out)


if __name__ == "__main__":
    main()
