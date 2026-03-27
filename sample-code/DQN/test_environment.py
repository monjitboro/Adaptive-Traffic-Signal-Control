import sys
sys.path.append('/sample-code/DQN')
import cityflow
from environment import TrafficEnvironment

# Test with training dataset
config_file = "data/hangzhou_1x1_bc-tyc_18041610_1h/config.json"

print("=" * 60)
print("TESTING ENVIRONMENT")
print("=" * 60)

# Initialize environment
env = TrafficEnvironment(config_file, num_steps=3600)

# Reset and get initial state
state = env.reset()
print("\nInitial state shape : {}".format(state.shape))
print("Initial state       : {}".format(state))

# Run a few steps manually
print("\n--- Running 5 test steps ---")
for i in range(5):
    action = i % 2  # alternate keep/switch
    next_state, reward, done = env.step(action)
    metrics = env.get_metrics()
    print("\nStep {}:".format(i+1))
    print("  Action          : {}".format("SWITCH" if action == 1 else "KEEP"))
    print("  Reward          : {:.4f}".format(reward))
    print("  Waiting         : {}".format(metrics["waiting"]))
    print("  Total vehicles  : {}".format(metrics["total_vehicles"]))
    print("  Congestion      : {}%".format(metrics["congestion_rate"]))

print("\n" + "=" * 60)
print("ENVIRONMENT TEST PASSED")
print("=" * 60)
