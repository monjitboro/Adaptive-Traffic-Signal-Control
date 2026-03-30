import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from cityflow_env import CityFlowSingleIntersectionEnv


def make_env(config, scenario, num_step, decision_interval):
    env = CityFlowSingleIntersectionEnv(
        config_path=config,
        scenario_name=scenario,
        num_step=num_step,
        decision_interval=decision_interval,
    )
    return Monitor(env)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="hangzhou_1x1_bc-tyc_18041607_1h")
    parser.add_argument("--config", type=str, default="data/hangzhou_1x1_bc-tyc_18041607_1h/config_rl.json")
    parser.add_argument("--total_timesteps", type=int, default=300000)
    parser.add_argument("--num_step", type=int, default=3600)
    parser.add_argument("--decision_interval", type=int, default=5)
    args = parser.parse_args()

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    train_env = make_env(args.config, args.scenario, args.num_step, args.decision_interval)
    eval_env = make_env(args.config, args.scenario, args.num_step, args.decision_interval)

    # Tuned for more stable on-policy learning than the baseline PPO script.
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="logs/ppo_tuned/",
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=128,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"models/ppo_tuned_best_{args.scenario}",
        log_path=f"logs/ppo_tuned_eval_{args.scenario}",
        eval_freq=5000,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)

    final_path = f"models/ppo_tuned_{args.scenario}"
    model.save(final_path)
    print("Saved final model:", final_path)


if __name__ == "__main__":
    main()
