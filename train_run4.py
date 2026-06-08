"""
PPO training — Run 4 (improved baseline).

Changes vs train.py:
  1. N_ENVS=8 parallel environments  — more diverse experience per update
  2. ENTROPY_COEF=0.03 (was 0.01)    — stronger exploration early on
  3. LR warm-up for first 200k steps  — avoids large updates on noisy early data

Usage (Kaggle):
    from train_run4 import train
    reward = train(env_name="HalfCheetah-v4", run_id=4, total_steps=10_000_000,
                   save_dir="/kaggle/working/results/HalfCheetah-v4", resume=False)
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

N_ENVS         = 8           # parallel environments  ← NEW
HORIZON        = 4096        # total transitions per PPO iteration (same as before)
T_PER_ENV      = HORIZON // N_ENVS   # vectorized steps per iteration = 512
N_EPOCHS       = 10
MINIBATCH_SIZE = 128
GAMMA          = 0.99
LAM            = 0.95
LR             = 3e-4
WARMUP_STEPS   = 200_000     # ← NEW: LR ramps 0→LR over first 200k steps
CLIP_EPS       = 0.2
VF_COEF        = 0.5
ENTROPY_COEF   = 0.03        # ← changed from 0.01
MAX_GRAD_NORM  = 0.5

TOTAL_STEPS    = 10_000_000
EVAL_FREQ      = 10_000
EVAL_EPISODES  = 10


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_logger(log_path):
    log_file = open(log_path, "a", buffering=1)

    def log(msg=""):
        print(msg)
        log_file.write(str(msg) + "\n")

    return log, log_file


def lr_schedule(global_step, total_steps):
    """Warm-up then linear anneal."""
    if global_step < WARMUP_STEPS:
        return LR * (global_step / max(WARMUP_STEPS, 1))
    frac = 1.0 - (global_step - WARMUP_STEPS) / (total_steps - WARMUP_STEPS)
    return LR * max(frac, 0.0)


@torch.no_grad()
def evaluate(agent, env_name, n_episodes):
    """Deterministic evaluation using a single env (unchanged from train.py)."""
    env = gym.make(env_name)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            state_t = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
            action  = agent.actor(state_t).squeeze(0).cpu().numpy()
            action  = np.clip(action, env.action_space.low, env.action_space.high)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
            done   = terminated or truncated
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


def compute_gae_per_env(env_rewards, env_values, env_dones, last_values):
    """
    GAE computed independently per environment.

    Args:
        env_rewards : (N_ENVS, T_per_env)
        env_values  : (N_ENVS, T_per_env)
        env_dones   : (N_ENVS, T_per_env)
        last_values : (N_ENVS,)  — V(s) of the final observation for each env

    Returns:
        advantages  : (N_ENVS, T_per_env)
        returns     : (N_ENVS, T_per_env)
    """
    n, T = env_rewards.shape
    advantages = np.zeros((n, T), dtype=np.float32)

    for i in range(n):
        gae = 0.0
        for t in reversed(range(T)):
            if t == T - 1:
                next_v = last_values[i]
                next_d = 0.0  # vector env auto-resets, so final obs is always fresh
            else:
                next_v = env_values[i, t + 1]
                next_d = env_dones[i, t + 1]
            delta = env_rewards[i, t] + GAMMA * next_v * (1.0 - next_d) - env_values[i, t]
            gae   = delta + GAMMA * LAM * (1.0 - next_d) * gae
            advantages[i, t] = gae

    returns = advantages + env_values
    return advantages, returns


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

    # Single env just to read dims; training uses the vector env below
    _tmp       = gym.make(env_name)
    obs_dim    = _tmp.observation_space.shape[0]
    action_dim = _tmp.action_space.shape[0]
    act_low    = _tmp.action_space.low
    act_high   = _tmp.action_space.high
    _tmp.close()
    log(f"obs_dim={obs_dim}  action_dim={action_dim}  N_ENVS={N_ENVS}")

    agent = PPOAgent(
        obs_dim       = obs_dim,
        action_dim    = action_dim,
        device        = device,
        lr            = LR,
        clip_eps      = CLIP_EPS,
        vf_coef       = VF_COEF,
        entropy_coef  = ENTROPY_COEF,
        max_grad_norm = MAX_GRAD_NORM,
    )

    # Buffer sized for HORIZON total transitions (same as original)
    buffer = RolloutBuffer(
        T          = HORIZON,
        obs_dim    = obs_dim,
        action_dim = action_dim,
        gamma      = GAMMA,
        lam        = LAM,
        device     = device,
    )

    # Per-env arrays for one iteration (reused every iteration)
    env_states    = np.zeros((N_ENVS, T_PER_ENV, obs_dim),    dtype=np.float32)
    env_actions   = np.zeros((N_ENVS, T_PER_ENV, action_dim), dtype=np.float32)
    env_rewards   = np.zeros((N_ENVS, T_PER_ENV),             dtype=np.float32)
    env_log_probs = np.zeros((N_ENVS, T_PER_ENV),             dtype=np.float32)
    env_values    = np.zeros((N_ENVS, T_PER_ENV),             dtype=np.float32)
    env_dones     = np.zeros((N_ENVS, T_PER_ENV),             dtype=np.float32)

    # ── Resume or fresh start ──────────────────────────────────────────────────
    if resume and os.path.exists(ckpt_path):
        global_step, eval_rewards, eval_steps = load_checkpoint(ckpt_path, agent)
        log(f"Resumed from step {global_step}")
    else:
        global_step  = 0
        eval_rewards = []
        eval_steps   = []

    log(f"\nHyperparameters:")
    log(f"  total_steps={total_steps}, horizon={HORIZON} ({N_ENVS} envs x {T_PER_ENV} steps)")
    log(f"  epochs={N_EPOCHS}, minibatch={MINIBATCH_SIZE}")
    log(f"  lr={LR} (warm-up {WARMUP_STEPS:,} steps then linear anneal to 0)")
    log(f"  gamma={GAMMA}, lambda={LAM}")
    log(f"  clip_eps={CLIP_EPS}, vf_coef={VF_COEF}, entropy_coef={ENTROPY_COEF}, grad_clip={MAX_GRAD_NORM}")
    log(f"  eval_freq={EVAL_FREQ}, eval_episodes={EVAL_EPISODES}")
    log("-" * 65)

    vec_env = gym.vector.SyncVectorEnv([lambda: gym.make(env_name) for _ in range(N_ENVS)])
    obs, _  = vec_env.reset()            # obs: (N_ENVS, obs_dim)

    next_eval_at = (global_step // EVAL_FREQ + 1) * EVAL_FREQ
    t_start = time.time()

    while global_step < total_steps:

        # ── LR schedule: warm-up then anneal ──────────────────────────────────
        current_lr = lr_schedule(global_step, total_steps)
        for pg in agent.optimizer.param_groups:
            pg["lr"] = current_lr

        # ── Collect T_PER_ENV vectorized steps ────────────────────────────────
        for t in range(T_PER_ENV):
            # Sample action for each env independently
            actions = np.zeros((N_ENVS, action_dim), dtype=np.float32)
            for i in range(N_ENVS):
                a, lp, v = agent.select_action(obs[i])
                env_states[i, t]    = obs[i]
                env_actions[i, t]   = a
                env_log_probs[i, t] = lp
                env_values[i, t]    = v
                actions[i]          = a

            clipped = np.clip(actions, act_low, act_high)
            next_obs, rewards, terminated, truncated, _ = vec_env.step(clipped)
            dones = (terminated | truncated).astype(np.float32)

            env_rewards[: , t] = rewards
            env_dones[:, t]    = dones

            global_step += N_ENVS
            obs = next_obs

            # ── Evaluate if we crossed a 10k checkpoint ────────────────────────
            if global_step >= next_eval_at:
                eval_reward = evaluate(agent, env_name, EVAL_EPISODES)
                eval_rewards.append(eval_reward)
                eval_steps.append(global_step)

                elapsed = time.time() - t_start
                sps     = global_step / elapsed
                log(
                    f"step {global_step:>8,} | "
                    f"eval_reward={eval_reward:>9.2f} | "
                    f"lr={current_lr:.2e} | "
                    f"{sps:>6.0f} steps/s"
                )

                np.save(f"{save_dir}/eval_rewards_run{run_id}.npy", eval_rewards)
                np.save(f"{save_dir}/eval_steps_run{run_id}.npy",   eval_steps)
                save_checkpoint(ckpt_path, agent, global_step, eval_rewards, eval_steps)

                next_eval_at += EVAL_FREQ

        # ── GAE per env, then merge into buffer ───────────────────────────────
        last_values = np.array([agent.get_value(obs[i]) for i in range(N_ENVS)],
                               dtype=np.float32)

        advantages, returns = compute_gae_per_env(
            env_rewards, env_values, env_dones, last_values
        )

        # Normalize advantages across all envs jointly
        adv_flat = advantages.reshape(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # Load merged data directly into the buffer (bypasses store() loop)
        buffer.states    = env_states.reshape(HORIZON, obs_dim)
        buffer.actions   = env_actions.reshape(HORIZON, action_dim)
        buffer.log_probs = env_log_probs.reshape(HORIZON)
        buffer.values    = env_values.reshape(HORIZON)
        buffer.advantages = adv_flat
        buffer.returns   = returns.reshape(HORIZON)
        buffer.ptr       = HORIZON   # mark as full for get_minibatches()

        agent.update(buffer, N_EPOCHS, MINIBATCH_SIZE)

    # ── Final eval and save ────────────────────────────────────────────────────
    vec_env.close()

    final_reward = evaluate(agent, env_name, 100)
    log(f"\nFinal reward (100 episodes): {final_reward:.2f}")
    np.save(f"{save_dir}/final_reward_run{run_id}.npy", np.array([final_reward]))
    agent.save(f"{save_dir}/model_run{run_id}.pt")

    plt.figure(figsize=(10, 4))
    plt.plot(eval_steps, eval_rewards)
    plt.xlabel("Training Steps")
    plt.ylabel("Average Reward")
    plt.title(f"PPO — {env_name} — Run {run_id} (8 envs, c₂=0.03, LR warm-up)")
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
    parser.add_argument("--env",      type=str, default="HalfCheetah-v4")
    parser.add_argument("--run_id",   type=int, default=4)
    parser.add_argument("--steps",    type=int, default=TOTAL_STEPS)
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--resume",   action="store_true")
    args = parser.parse_args()

    save_dir = f"{args.save_dir}/{args.env}"
    reward   = train(
        env_name    = args.env,
        run_id      = args.run_id,
        total_steps = args.steps,
        save_dir    = save_dir,
        resume      = args.resume,
    )
    print(f"\n{'='*50}")
    print(f"Run {args.run_id} done  |  Final reward: {reward:.2f}")
    print(f"{'='*50}")
