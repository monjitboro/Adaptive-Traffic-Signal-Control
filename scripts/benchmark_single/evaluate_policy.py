import os
import csv
import time
import numpy as np 
import sumo_rl
from sumo_rl import SumoEnvironment

BASE = os.path.dirname(sumo_rl.__file__)
NET = os.path.join(BASE, "nets", "single-intersection", "single-intersection.net.xml")
ROU = os.path.join(BASE, "nets", "single-intersection", "single-intersection.rou.xml")


def make_env(seed: int, use_gui: bool = False, num_seconds: int = 2000):
    return SumoEnvironment(
        net_file=NET,
        route_file=ROU,
        use_gui=use_gui,
        num_seconds=num_seconds,
        delta_time=5,
        yellow_time=2,
        min_green=5,
        max_green=50,
        single_agent=True,
        reward_fn="diff-waiting-time",
        sumo_seed=seed,
        sumo_warnings=False,
        additional_sumo_cmd="--no-step-log true",
    )


def eval_policy(name: str, policy_fn, episodes: int = 5,
                seed0: int = 0,
                out_csv="logs/single_benchmark.csv"):

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = []

    for ep in range(episodes):
        seed = seed0 + ep
        env = make_env(seed=seed, use_gui=False, num_seconds=2000)

        try:
            # Compatible reset (Gym vs Gymnasium)
            r = env.reset()
            obs = r[0] if isinstance(r, tuple) else r

            ep_reward = 0.0
            steps = 0
            t0 = time.time()

            while True:
                action = policy_fn(obs)
                out = env.step(action)

                # Handle both APIs
                if len(out) == 5:
                    obs, reward, terminated, truncated, info = out
                    done = bool(terminated or truncated)
                else:
                    obs, reward, dones, info = out
                    done = bool(dones["__all__"])

                ep_reward += float(reward)
                steps += 1

                if done:
                    break

        finally:
            env.close()

        rows.append({
            "model": name,
            "episode": ep,
            "seed": seed,
            "ep_reward": ep_reward,
            "steps": steps,
            "wall_sec": time.time() - t0
        })

    write_header = not os.path.exists(out_csv)

    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            w.writeheader()
        w.writerows(rows)

    print(f"[OK] Wrote {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    # Baseline 1: Always action 0
    eval_policy("always_0", lambda obs: 0)

    # Baseline 2: Random (reproducible)
    rng = np.random.default_rng(123)
    eval_policy("random_01", lambda obs: int(rng.integers(0, 2)))
