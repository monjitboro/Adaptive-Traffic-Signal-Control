from cityflow_env import CityFlowSingleIntersectionEnv
import numpy as np

env = CityFlowSingleIntersectionEnv(
    config_path="data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json",
    scenario_name="hangzhou_1x1_bc-tyc_18041607_1h",
    num_step=3600,
    decision_interval=5
)

obs, info = env.reset()
print("Initial obs shape:", obs.shape)
print("Initial info:", info)

done = False
ep_reward = 0.0
step_idx = 0

while not done:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    ep_reward += reward
    step_idx += 1

    if step_idx % 20 == 0:
        print(f"step={step_idx}, reward={reward:.3f}, queue={info['queue']}, avg_tt={info['avg_travel_time']:.2f}")

print("Episode reward:", ep_reward)