import cityflow as engine
import argparse
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
parser.add_argument("--num_step", type=int, default=3600)
parser.add_argument("--out", type=str, default="")
args = parser.parse_args()

config_file = "data/{}/config.json".format(args.scenario)
eng = engine.Engine(config_file, thread_num=1)

scenario_dir = "data/{}".format(args.scenario)
roadnet_path = os.path.join(scenario_dir, "roadnet.json")
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
incoming_lane_count = max(1, len(incoming_lanes))

# Tracking metrics
total_waiting = []
total_vehicles = []
incoming_queue_history = []
incoming_vehicle_history = []

print("=" * 60)
print("CITYFLOW SIMULATION - BASELINE (Fixed Signal Plan)")
print("Scenario: {}".format(args.scenario))
print("=" * 60)

for step in range(args.num_step):
    eng.next_step()

    lane_waiting = eng.get_lane_waiting_vehicle_count()
    lane_vehicles = eng.get_lane_vehicle_count()
    vehicle_speed = eng.get_vehicle_speed()

    total_waiting_now = sum(lane_waiting.values())
    total_vehicles_now = sum(lane_vehicles.values())
    incoming_queue_now = sum(lane_waiting.get(lane_id, 0) for lane_id in incoming_lanes)
    incoming_vehicles_now = sum(lane_vehicles.get(lane_id, 0) for lane_id in incoming_lanes)

    # vehicles that have speed ~0 are waiting
    waiting_vehicles = sum(1 for s in vehicle_speed.values() if s < 0.1)

    total_waiting.append(total_waiting_now)
    total_vehicles.append(total_vehicles_now)
    incoming_queue_history.append(incoming_queue_now)
    incoming_vehicle_history.append(incoming_vehicles_now)

    if step % 300 == 0 and step > 0:
        avg_waiting = sum(total_waiting) / len(total_waiting)
        avg_vehicles = sum(total_vehicles) / len(total_vehicles)
        avg_incoming_queue = sum(incoming_queue_history) / len(incoming_queue_history)
        print("\n--- Step {} / {} ---".format(step, args.num_step))
        print("  Vehicles on road      : {}".format(total_vehicles_now))
        print("  Waiting vehicles      : {}".format(waiting_vehicles))
        print("  Avg waiting (so far)  : {:.2f} vehicles".format(avg_waiting))
        print("  Avg vehicles (so far) : {:.2f}".format(avg_vehicles))
        print("  Avg incoming queue    : {:.2f}".format(avg_incoming_queue))

# Final summary
avg_waiting_total = sum(total_waiting) / len(total_waiting)
avg_vehicles_total = sum(total_vehicles) / len(total_vehicles)
peak_vehicles = max(total_vehicles)
peak_waiting = max(total_waiting)
mean_incoming_queue = sum(incoming_queue_history) / len(incoming_queue_history)
mean_incoming_vehicles = sum(incoming_vehicle_history) / len(incoming_vehicle_history)
final_avg_travel_time = float(eng.get_average_travel_time())
queue_only_reward_equivalent = -float(mean_incoming_queue) / incoming_lane_count

results = {
    "scenario": args.scenario,
    "controller": "fixed_signal_plan",
    "incoming_lanes": incoming_lane_count,
    "mean_reward_queue_only_equivalent": queue_only_reward_equivalent,
    "mean_queue": float(mean_incoming_queue),
    "mean_vehicles": float(mean_incoming_vehicles),
    "final_avg_travel_time": final_avg_travel_time,
    "avg_waiting_all_lanes": float(avg_waiting_total),
    "avg_vehicles_all_lanes": float(avg_vehicles_total),
    "peak_waiting_all_lanes": int(peak_waiting),
    "peak_vehicles_all_lanes": int(peak_vehicles),
}

print("\n" + "=" * 60)
print("SIMULATION COMPLETE - FINAL STATS")
print("=" * 60)
print("  Total simulation time     : {} seconds".format(args.num_step))
print("  Avg vehicles on road      : {:.2f}".format(avg_vehicles_total))
print("  Avg waiting vehicles      : {:.2f}".format(avg_waiting_total))
print("  Peak vehicles on road     : {}".format(peak_vehicles))
print("  Peak waiting vehicles     : {}".format(peak_waiting))
print("  Congestion rate           : {:.1f}%".format(
    (avg_waiting_total / avg_vehicles_total * 100) if avg_vehicles_total > 0 else 0))
print("  Comparable mean queue     : {:.4f}".format(mean_incoming_queue))
print("  Comparable mean reward    : {:.4f}".format(queue_only_reward_equivalent))
print("  Final avg travel time     : {:.4f}".format(final_avg_travel_time))
print("=" * 60)
print("\nThis is your BASELINE to beat with RL!")

if args.out:
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved baseline JSON       : {}".format(args.out))
