"""
PPO training for MuJoCo continuous control.
Usage:
    python train.py --env HalfCheetah-v4 --run_id 1
    python train.py --env Hopper-v4      --run_id 1 --resume
"""

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import os
import time
from datetime import datetime

from ppo_agent import PPOAgent
from rollout_buffer import RolloutBuffer


# ── Hyperparameters ────────────────────────────────────────────────────────────

HORIZON        = 4096       # T: steps collected per PPO iteration (2x paper, better for 10M budget)
N_EPOCHS       = 10         # K: gradient epochs per iteration
MINIBATCH_SIZE = 128        # M: scaled with horizon to keep T/M ratio the same (4096/128 = 32)
GAMMA          = 0.99       # discount factor
LAM            = 0.95       # GAE lambda
LR             = 3e-4       # initial Adam learning rate (linearly annealed to 0)
CLIP_EPS       = 0.2        # PPO clipping parameter epsilon
VF_COEF        = 0.5        # c1: value function loss coefficient
ENTROPY_COEF   = 0.01       # c2: entropy bonus coefficient
MAX_GRAD_NORM  = 0.5        # gradient clipping threshold

TOTAL_STEPS    = 10_000_000  # training budget per run
EVAL_FREQ      = 10_000     # evaluate every this many env steps
EVAL_EPISODES  = 10         # episodes per evaluation


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_logger(log_path):
    log_file = open(log_path, "a", buffering=1)

    def log(msg=""):
        print(msg)
        log_file.write(str(msg) + "\n")

    return log, log_file


def make_env(env_name):
    return gym.make(env_name)


@torch.no_grad()
def evaluate(agent, env_name, n_episodes):
    """
    Run n_episodes with the deterministic policy (mean action, no sampling).
    Returns mean total reward across episodes.
    """
    env = make_env(env_name)
    rewards = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total  = 0.0
        done   = False

        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            # Deterministic: use mean of Gaussian, no sampling
            action  = agent.actor(state_t).squeeze(0).cpu().numpy()
            # Clip to valid action range
            action  = np.clip(action, env.action_space.low, env.action_space.high)
            obs, r, terminated, truncated, _ = env.step(action)
            total  += r
            done    = terminated or truncated

        rewards.append(total)

    env.close()
    return float(np.mean(rewards))


def save_checkpoint(path, agent, step, eval_rewards, eval_steps):
    torch.save({
        "actor":        agent.actor.state_dict(),
        "critic":       agent.critic.state_dict(),
        "optimizer":    agent.optimizer.state_dict(),
        "step":         step,
        "eval_rewards": eval_rewards,
        "eval_steps":   eval_steps,
    }, path)


def load_checkpoint(path, agent):
    ckpt = torch.load(path, weights_only=False)
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["step"], ckpt["eval_rewards"], ckpt["eval_steps"]


# ── Training loop ──────────────────────────────────────────────────────────────

def train(env_name, run_id, total_steps, save_dir, resume):
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = f"{save_dir}/checkpoint_run{run_id}.pt"
    log_path  = f"{save_dir}/train_run{run_id}.log"

    log, log_file = make_logger(log_path)
    log("=" * 65)
    log(f"PPO  |  env={env_name}  |  run={run_id}  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    log("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Device: {device}")

    env = make_env(env_name)
    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    log(f"obs_dim={obs_dim}  action_dim={action_dim}")

    agent = PPOAgent(
        obs_dim      = obs_dim,
        action_dim   = action_dim,
        device       = device,
        lr           = LR,
        clip_eps     = CLIP_EPS,
        vf_coef      = VF_COEF,
        entropy_coef = ENTROPY_COEF,
        max_grad_norm= MAX_GRAD_NORM,
    )

    buffer = RolloutBuffer(
        T          = HORIZON,
        obs_dim    = obs_dim,
        action_dim = action_dim,
        gamma      = GAMMA,
        lam        = LAM,
        device     = device,
    )

    # ── Resume or fresh start ──────────────────────────────────────────────────
    if resume and os.path.exists(ckpt_path):
        global_step, eval_rewards, eval_steps = load_checkpoint(ckpt_path, agent)
        log(f"Resumed from step {global_step}")
    else:
        global_step  = 0
        eval_rewards = []
        eval_steps   = []

    log(f"\nHyperparameters:")
    log(f"  total_steps={total_steps}, horizon={HORIZON}, epochs={N_EPOCHS}, minibatch={MINIBATCH_SIZE}")
    log(f"  lr={LR} (linearly annealed to 0), gamma={GAMMA}, lambda={LAM}")
    log(f"  clip_eps={CLIP_EPS}, vf_coef={VF_COEF}, entropy_coef={ENTROPY_COEF}, grad_clip={MAX_GRAD_NORM}")
    log(f"  eval_freq={EVAL_FREQ}, eval_episodes={EVAL_EPISODES}")
    log("-" * 65)

    # ── Collect initial observation ────────────────────────────────────────────
    obs, _ = env.reset()
    next_eval_at = (global_step // EVAL_FREQ + 1) * EVAL_FREQ

    t_start = time.time()

    while global_step < total_steps:

        # ── Linear LR annealing ────────────────────────────────────────────────
        frac = 1.0 - global_step / total_steps
        for pg in agent.optimizer.param_groups:
            pg["lr"] = LR * frac

        # ── Collect HORIZON steps ──────────────────────────────────────────────
        buffer.reset()

        for _ in range(HORIZON):
            action, log_prob, value = agent.select_action(obs)

            # Clip action to env bounds before stepping
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
            next_obs, reward, terminated, truncated, _ = env.step(clipped_action)
            done = terminated or truncated

            buffer.store(obs, action, reward, log_prob, value, float(done))
            global_step += 1
            obs = next_obs

            if done:
                obs, _ = env.reset()

            # ── Evaluate if we crossed a 10k checkpoint ────────────────────────
            if global_step >= next_eval_at:
                eval_reward = evaluate(agent, env_name, EVAL_EPISODES)
                eval_rewards.append(eval_reward)
                eval_steps.append(global_step)

                elapsed = time.time() - t_start
                sps     = global_step / elapsed
                lr_now  = agent.optimizer.param_groups[0]["lr"]
                log(
                    f"step {global_step:>8,} | "
                    f"eval_reward={eval_reward:>9.2f} | "
                    f"lr={lr_now:.2e} | "
                    f"{sps:>6.0f} steps/s"
                )

                np.save(f"{save_dir}/eval_rewards_run{run_id}.npy", eval_rewards)
                np.save(f"{save_dir}/eval_steps_run{run_id}.npy",   eval_steps)
                save_checkpoint(ckpt_path, agent, global_step, eval_rewards, eval_steps)

                next_eval_at += EVAL_FREQ

        # ── Compute GAE and update ─────────────────────────────────────────────
        last_value = agent.get_value(obs)
        last_done  = 0.0  # obs is a fresh state (either continuing or just reset)
        buffer.compute_returns_and_advantages(last_value, last_done)

        losses = agent.update(buffer, N_EPOCHS, MINIBATCH_SIZE)

    # ── Final eval and save ────────────────────────────────────────────────────
    env.close()

    final_reward = evaluate(agent, env_name, 100)
    log(f"\nFinal reward (100 episodes): {final_reward:.2f}")
    np.save(f"{save_dir}/final_reward_run{run_id}.npy", np.array([final_reward]))
    agent.save(f"{save_dir}/model_run{run_id}.pt")

    # ── Plot ───────────────────────────────────────────────────────────────────
    plt.figure(figsize=(10, 4))
    plt.plot(eval_steps, eval_rewards)
    plt.xlabel("Training Steps")
    plt.ylabel("Average Reward")
    plt.title(f"PPO — {env_name} — Run {run_id}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/learning_curve_run{run_id}.png", dpi=150)
    plt.close()

    log(f"Saved results to {save_dir}/")
    log_file.close()
    return final_reward


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        type=str, default="Hopper-v4")
    parser.add_argument("--run_id",     type=int, default=1)
    parser.add_argument("--steps",      type=int, default=TOTAL_STEPS)
    parser.add_argument("--save_dir",   type=str, default="results")
    parser.add_argument("--resume",     action="store_true")
    args = parser.parse_args()

    save_dir = f"{args.save_dir}/{args.env}"

    reward = train(
        env_name    = args.env,
        run_id      = args.run_id,
        total_steps = args.steps,
        save_dir    = save_dir,
        resume      = args.resume,
    )

    print(f"\n{'='*50}")
    print(f"Run {args.run_id} complete  |  Final reward: {reward:.2f}")
    print(f"{'='*50}")
