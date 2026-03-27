import sys
sys.path.append('/sample-code/DQN')

import os
import json
import numpy as np
from environment import TrafficEnvironment
from dqn_agent import DQNAgent

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
TRAIN_CONFIG    = "data/hangzhou_1x1_bc-tyc_18041610_1h/config.json"
VAL_CONFIG      = "data/hangzhou_1x1_bc-tyc_18041608_1h/config.json"
NUM_EPISODES    = 1000
NUM_STEPS       = 3600
VALIDATE_EVERY  = 100
SAVE_DIR        = "DQN/models"
LOG_DIR         = "DQN/logs"
REPLAY_DIR      = "DQN/replays"

# Save replay at these episodes to visualize improvement
SAVE_REPLAY_AT  = [1, 100, 250, 500, 1000]

# Create folders
os.makedirs(SAVE_DIR,   exist_ok=True)
os.makedirs(LOG_DIR,    exist_ok=True)
os.makedirs(REPLAY_DIR, exist_ok=True)

# ─────────────────────────────────────────
# HELPER - Create config with replay enabled
# ─────────────────────────────────────────
def make_config(scenario, save_replay=False, replay_filename="replay.txt"):
    return {
        "interval"       : 1.0,
        "seed"           : 0,
        "dir"            : "data/{}/".format(scenario),
        "roadnetFile"    : "roadnet.json",
        "flowFile"       : "flow.json",
        "rlTrafficLight" : True,
        "saveReplay"     : save_replay,
        "roadnetLogFile" : "roadnet_log.json",
        "replayLogFile"  : replay_filename
    }

# Write base configs
TRAIN_SCENARIO = "hangzhou_1x1_bc-tyc_18041610_1h"
VAL_SCENARIO   = "hangzhou_1x1_bc-tyc_18041608_1h"

with open(TRAIN_CONFIG, 'w') as f:
    json.dump(make_config(TRAIN_SCENARIO, False), f, indent=4)

with open(VAL_CONFIG, 'w') as f:
    json.dump(make_config(VAL_SCENARIO, False), f, indent=4)

# ─────────────────────────────────────────
# INITIALIZE
# ─────────────────────────────────────────
print("=" * 60)
print("DQN TRAINING - ADAPTIVE TRAFFIC SIGNAL CONTROL")
print("=" * 60)
print("Train    : {}".format(TRAIN_CONFIG))
print("Validate : {}".format(VAL_CONFIG))
print("Episodes : {}".format(NUM_EPISODES))
print("Replays  : episodes {}".format(SAVE_REPLAY_AT))
print("=" * 60)

train_env = TrafficEnvironment(TRAIN_CONFIG, NUM_STEPS)
val_env   = TrafficEnvironment(VAL_CONFIG,   NUM_STEPS)
agent     = DQNAgent(train_env.state_size, train_env.action_size)

# ─────────────────────────────────────────
# TRACKING
# ─────────────────────────────────────────
train_rewards    = []
train_waiting    = []
train_congestion = []
val_rewards      = []
val_waiting      = []
best_val_waiting = float('inf')

# ─────────────────────────────────────────
# SAVE REPLAY EPISODE
# ─────────────────────────────────────────
def save_replay_episode(agent, episode):
    """Run episode with replay saving enabled"""
    print("  >> Saving replay for episode {}...".format(episode))

    # Create config with replay enabled
    replay_file    = "replay_ep{}.txt".format(episode)
    replay_config  = make_config(TRAIN_SCENARIO, True, replay_file)
    replay_config_path = "data/{}/config_replay.json".format(TRAIN_SCENARIO)

    with open(replay_config_path, 'w') as f:
        json.dump(replay_config, f, indent=4)

    # Create temporary environment with replay
    replay_env = TrafficEnvironment(replay_config_path, NUM_STEPS)
    state      = replay_env.reset()

    old_epsilon   = agent.epsilon
    agent.epsilon = 0.0  # greedy during replay saving

    for step in range(NUM_STEPS):
        action = agent.act(state)
        next_state, reward, done = replay_env.step(action)
        state = next_state
        if done:
            break

    agent.epsilon = old_epsilon

    # Copy replay file to replays folder
    src = "data/{}/{}".format(TRAIN_SCENARIO, replay_file)
    dst = "{}/{}".format(REPLAY_DIR, replay_file)
    os.rename(src, dst)

    # Copy roadnet log for visualizer
    roadnet_src = "data/{}/roadnet_log.json".format(TRAIN_SCENARIO)
    roadnet_dst = "{}/roadnet_log.json".format(REPLAY_DIR)
    if os.path.exists(roadnet_src) and not os.path.exists(roadnet_dst):
        import shutil
        shutil.copy(roadnet_src, roadnet_dst)

    print("  >> Replay saved: {}/{}".format(REPLAY_DIR, replay_file))

# ─────────────────────────────────────────
# VALIDATION FUNCTION
# ─────────────────────────────────────────
def validate(env, agent):
    state        = env.reset()
    total_reward = 0
    all_waiting  = []
    all_congestion = []

    for step in range(NUM_STEPS):
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        metrics = env.get_metrics()

        total_reward += reward
        all_waiting.append(metrics["waiting"])
        all_congestion.append(metrics["congestion_rate"])
        state = next_state

        if done:
            break

    return {
        "reward"        : total_reward,
        "avg_waiting"   : np.mean(all_waiting),
        "avg_congestion": np.mean(all_congestion)
    }

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
print("\nStarting training...\n")

for episode in range(1, NUM_EPISODES + 1):
    state        = train_env.reset()
    total_reward = 0
    total_loss   = 0
    all_waiting  = []
    all_congestion = []
    loss_count   = 0

    for step in range(NUM_STEPS):
        action = agent.act(state)
        next_state, reward, done = train_env.step(action)
        metrics = train_env.get_metrics()

        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()

        if loss > 0:
            total_loss += loss
            loss_count += 1

        total_reward += reward
        all_waiting.append(metrics["waiting"])
        all_congestion.append(metrics["congestion_rate"])
        state = next_state

        if done:
            break

    agent.decay_epsilon()

    avg_waiting    = np.mean(all_waiting)
    avg_congestion = np.mean(all_congestion)
    avg_loss       = total_loss / loss_count if loss_count > 0 else 0

    train_rewards.append(total_reward)
    train_waiting.append(avg_waiting)
    train_congestion.append(avg_congestion)

    # Print every 10 episodes
    if episode % 10 == 0:
        print("Episode {:3d}/{} | Reward: {:8.2f} | Waiting: {:5.2f} | "
              "Congestion: {:5.1f}% | Epsilon: {:.3f} | Loss: {:.4f}".format(
              episode, NUM_EPISODES,
              total_reward, avg_waiting,
              avg_congestion, agent.epsilon, avg_loss))

    # Save replay at specific episodes
    if episode in SAVE_REPLAY_AT:
        save_replay_episode(agent, episode)

    # Validate every 50 episodes
    if episode % VALIDATE_EVERY == 0:
        print("\n  >> Validating on 8am dataset...")
        old_epsilon   = agent.epsilon
        agent.epsilon = 0.0
        val_metrics   = validate(val_env, agent)
        agent.epsilon = old_epsilon

        val_rewards.append(val_metrics["reward"])
        val_waiting.append(val_metrics["avg_waiting"])

        print("  >> Val Reward: {:.2f} | Val Waiting: {:.2f} | "
              "Val Congestion: {:.1f}%".format(
              val_metrics["reward"],
              val_metrics["avg_waiting"],
              val_metrics["avg_congestion"]))

        if val_metrics["avg_waiting"] < best_val_waiting:
            best_val_waiting = val_metrics["avg_waiting"]
            agent.save("{}/best_model.pth".format(SAVE_DIR))
            print("  >> New best model saved! "
                  "Avg waiting: {:.2f}".format(best_val_waiting))
        print()

# ─────────────────────────────────────────
# SAVE LOGS
# ─────────────────────────────────────────
logs = {
    "train_rewards"    : train_rewards,
    "train_waiting"    : train_waiting,
    "train_congestion" : train_congestion,
    "val_rewards"      : val_rewards,
    "val_waiting"      : val_waiting,
    "best_val_waiting" : best_val_waiting
}
with open("{}/training_logs.json".format(LOG_DIR), 'w') as f:
    json.dump(logs, f, indent=4)

# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print("  Best validation waiting : {:.2f}".format(best_val_waiting))
print("  Baseline waiting        : 53.13")
print("  Improvement             : {:.1f}%".format(
    (53.13 - best_val_waiting) / 53.13 * 100))
print("  Model saved to          : {}/best_model.pth".format(SAVE_DIR))
print("  Logs saved to           : {}/training_logs.json".format(LOG_DIR))
print("\nReplays saved in DQN/replays/:")
for ep in SAVE_REPLAY_AT:
    print("  - replay_ep{}.txt  → episode {}".format(ep, ep))
print("\nTo visualize:")
print("  1. Open http://localhost:8080")
print("  2. Load DQN/replays/roadnet_log.json")
print("  3. Load DQN/replays/replay_ep1.txt   (untrained agent)")
print("  4. Load DQN/replays/replay_ep500.txt (trained agent)")
print("  5. Compare the difference!")
print("=" * 60)