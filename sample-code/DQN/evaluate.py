import sys
sys.path.append('/sample-code/DQN')

import json
import numpy as np
from environment import TrafficEnvironment
from dqn_agent import DQNAgent

TEST_CONFIG  = "data/hangzhou_1x1_bc-tyc_18041607_1h/config.json"
MODEL_PATH   = "DQN/models/best_model.pth"
NUM_STEPS    = 1800

test_config = {
    "interval": 1.0,
    "seed": 0,
    "dir": "data/hangzhou_1x1_bc-tyc_18041607_1h/",
    "roadnetFile": "roadnet.json",
    "flowFile": "flow.json",
    "rlTrafficLight": True,
    "saveReplay": True,
    "roadnetLogFile": "roadnet_log.json",
    "replayLogFile": "replay_test.txt"
}
with open(TEST_CONFIG, 'w') as f:
    json.dump(test_config, f, indent=4)

print("=" * 60)
print("DQN EVALUATION - 7AM TEST DATASET")
print("=" * 60)
print("Model    : {}".format(MODEL_PATH))
print("Scenario : hangzhou_1x1_bc-tyc_18041607_1h (7am)")
print("=" * 60)

env   = TrafficEnvironment(TEST_CONFIG, NUM_STEPS)
agent = DQNAgent(env.state_size, env.action_size)
agent.load(MODEL_PATH)
agent.epsilon = 0.0

state        = env.reset()
total_reward = 0
all_waiting  = []
all_vehicles = []
all_congestion = []
switches     = 0
prev_phase   = 0

print("\nRunning evaluation...\n")

for step in range(NUM_STEPS):
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    metrics = env.get_metrics()

    if env.current_phase != prev_phase:
        switches += 1
    prev_phase = env.current_phase

    total_reward += reward
    all_waiting.append(metrics["waiting"])
    all_vehicles.append(metrics["total_vehicles"])
    all_congestion.append(metrics["congestion_rate"])

    if step % 300 == 0 and step > 0:
        print("Step {:4d}/{} | Waiting: {:5.1f} | Vehicles: {:4d} | Congestion: {:5.1f}%".format(
            step, NUM_STEPS,
            metrics["waiting"],
            metrics["total_vehicles"],
            metrics["congestion_rate"]
        ))

    state = next_state
    if done:
        break

avg_waiting    = np.mean(all_waiting)
avg_vehicles   = np.mean(all_vehicles)
avg_congestion = np.mean(all_congestion)
peak_waiting   = max(all_waiting)

BASELINE_WAITING    = 53.13
BASELINE_CONGESTION = 64.5
improvement = ((BASELINE_WAITING - avg_waiting) / BASELINE_WAITING) * 100

print("\n" + "=" * 60)
print("EVALUATION RESULTS - 7AM TEST DATASET")
print("=" * 60)
print("\n  METRIC               BASELINE    DQN MODEL   CHANGE")
print("  " + "-" * 55)
print("  Avg waiting          {:8.2f}    {:8.2f}    {:+.1f}%".format(
    BASELINE_WAITING, avg_waiting, improvement))
print("  Avg congestion       {:7.1f}%    {:7.1f}%    {:+.1f}pp".format(
    BASELINE_CONGESTION, avg_congestion, BASELINE_CONGESTION - avg_congestion))
print("  Peak waiting         {:8.1f}    {:8.1f}".format(104, peak_waiting))
print("  Avg vehicles         {:8.2f}    {:8.2f}".format(82.34, avg_vehicles))
print("  Phase switches       {:8s}    {:8d}".format("N/A", switches))
print("  " + "-" * 55)

if avg_waiting < BASELINE_WAITING:
    print("\n  DQN BEATS BASELINE by {:.1f}%".format(improvement))
else:
    print("\n  DQN below baseline by {:.1f}%".format(abs(improvement)))

print("\n  Replay saved to: data/hangzhou_1x1_bc-tyc_18041607_1h/replay_test.txt")
print("=" * 60)
