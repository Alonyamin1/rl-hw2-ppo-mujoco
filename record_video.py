"""
Record a video of the best trained PPO agent.
Usage:
    python record_video.py --env HalfCheetah-v4 --model results/HalfCheetah-v4/model_run1.pt
    python record_video.py --env Hopper-v4      --model results/Hopper-v4/model_run2.pt

Requires: pip install imageio imageio-ffmpeg
MuJoCo rgb_array rendering is offscreen — no display needed (works on Kaggle).
"""

import gymnasium as gym
import numpy as np
import torch
import imageio
import argparse
import os

from networks import Actor


def find_best_run(save_dir, n_runs=3):
    """Return the run_id with the highest final reward."""
    best_run    = 1
    best_reward = -float("inf")

    for run_id in range(1, n_runs + 1):
        path = f"{save_dir}/final_reward_run{run_id}.npy"
        if os.path.exists(path):
            reward = np.load(path)[0]
            print(f"  Run {run_id}: final reward = {reward:.2f}")
            if reward > best_reward:
                best_reward = reward
                best_run    = run_id

    print(f"Best run: {best_run} (reward={best_reward:.2f})")
    return best_run


def load_actor(model_path, obs_dim, action_dim, device):
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    actor = Actor(obs_dim, action_dim).to(device)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


@torch.no_grad()
def record(env_name, model_path, output_path, n_episodes=3):
    """
    Record n_episodes using the deterministic policy (mean action).
    Saves a single video file containing all episodes back-to-back.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # rgb_array mode renders frames to a numpy array — no display needed
    env = gym.make(env_name, render_mode="rgb_array")

    obs_dim    = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = load_actor(model_path, obs_dim, action_dim, device)

    frames       = []
    all_rewards  = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        total  = 0.0
        done   = False
        ep_frames = 0

        while not done:
            frame = env.render()           # returns (H, W, 3) uint8 array
            frames.append(frame)
            ep_frames += 1

            state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action  = actor(state_t).squeeze(0).cpu().numpy()
            action  = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done   = terminated or truncated

        all_rewards.append(total)
        print(f"  Episode {ep + 1}: reward={total:.2f}, frames={ep_frames}")

    env.close()

    # Write frames to video at 30 fps
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    imageio.mimwrite(output_path, frames, fps=30, quality=8)

    print(f"\nVideo saved to: {output_path}")
    print(f"Total frames: {len(frames)}")
    print(f"Mean reward over {n_episodes} episodes: {np.mean(all_rewards):.2f}")

    return np.mean(all_rewards)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",        type=str, required=True,
                        help="e.g. HalfCheetah-v4 or Hopper-v4")
    parser.add_argument("--model",      type=str, default=None,
                        help="Path to model .pt file. If omitted, picks best run automatically.")
    parser.add_argument("--save_dir",   type=str, default="results",
                        help="Results directory (used to find best run if --model not given)")
    parser.add_argument("--output",     type=str, default=None,
                        help="Output video path. Default: results/<env>/best_agent.mp4")
    parser.add_argument("--episodes",   type=int, default=3,
                        help="Number of episodes to record")
    args = parser.parse_args()

    # Auto-find best model if not specified
    if args.model is None:
        env_save_dir = f"{args.save_dir}/{args.env}"
        print(f"Finding best run in {env_save_dir} ...")
        best_run   = find_best_run(env_save_dir)
        model_path = f"{env_save_dir}/model_run{best_run}.pt"
    else:
        model_path = args.model

    output_path = args.output or f"{args.save_dir}/{args.env}/best_agent.mp4"

    print(f"\nRecording {args.episodes} episodes")
    print(f"Model:  {model_path}")
    print(f"Output: {output_path}\n")

    record(
        env_name    = args.env,
        model_path  = model_path,
        output_path = output_path,
        n_episodes  = args.episodes,
    )
