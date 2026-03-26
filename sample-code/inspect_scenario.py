import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
args = parser.parse_args()

scenario_dir = os.path.join("data", args.scenario)
roadnet_path = os.path.join(scenario_dir, "roadnet.json")

with open(roadnet_path, "r") as f:
    roadnet = json.load(f)

print("=" * 60)
print("SCENARIO INSPECTION")
print("=" * 60)

intersections = roadnet["intersections"]
roads = roadnet["roads"]

real_intersections = [i for i in intersections if not i.get("virtual", False)]
print(f"Total intersections      : {len(intersections)}")
print(f"Real intersections       : {len(real_intersections)}")
print(f"Total roads              : {len(roads)}")

for inter in real_intersections:
    print("\nIntersection ID:", inter["id"])
    tl = inter.get("trafficLight", {})
    phases = tl.get("lightphases", [])
    print("Number of lightphases:", len(phases))

    for idx, ph in enumerate(phases):
        available = ph.get("availableRoadLinks", [])
        print(f"  Phase {idx}: availableRoadLinks={available}")

    incoming_roads = [r for r in roads if r["endIntersection"] == inter["id"]]
    print("Incoming roads:")
    for road in incoming_roads:
        lane_count = len(road.get("lanes", []))
        print(f"  {road['id']}  lanes={lane_count}")