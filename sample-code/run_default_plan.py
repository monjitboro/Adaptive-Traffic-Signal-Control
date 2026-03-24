import cityflow as engine
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
parser.add_argument("--num_step", type=int, default=3600)
args = parser.parse_args()

config_file = "data/{}/config.json".format(args.scenario)
eng = engine.Engine(config_file, thread_num=1)

# Tracking metrics
total_waiting = []
total_vehicles = []
throughput = 0
all_travel_times = []

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

    # vehicles that have speed ~0 are waiting
    waiting_vehicles = sum(1 for s in vehicle_speed.values() if s < 0.1)

    total_waiting.append(total_waiting_now)
    total_vehicles.append(total_vehicles_now)

    if step % 300 == 0 and step > 0:
        avg_waiting = sum(total_waiting) / len(total_waiting)
        avg_vehicles = sum(total_vehicles) / len(total_vehicles)
        print("\n--- Step {} / {} ---".format(step, args.num_step))
        print("  Vehicles on road      : {}".format(total_vehicles_now))
        print("  Waiting vehicles      : {}".format(waiting_vehicles))
        print("  Avg waiting (so far)  : {:.2f} vehicles".format(avg_waiting))
        print("  Avg vehicles (so far) : {:.2f}".format(avg_vehicles))

# Final summary
avg_waiting_total = sum(total_waiting) / len(total_waiting)
avg_vehicles_total = sum(total_vehicles) / len(total_vehicles)
peak_vehicles = max(total_vehicles)
peak_waiting = max(total_waiting)

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
print("=" * 60)
print("\nThis is your BASELINE to beat with RL!")
