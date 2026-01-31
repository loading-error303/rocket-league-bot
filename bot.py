#!/usr/bin/env python3
"""Run inference with a trained Rocket League agent."""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from src.env import make_env


def play(model_path: Path, num_episodes: int = 5) -> None:
    """Run the trained agent in inference mode.

    Args:
        model_path: Path to the saved model.
        num_episodes: Number of episodes to play.
    """
    print(f"Loading model from {model_path}...")
    model = PPO.load(str(model_path))

    env = make_env(spawn_opponents=True, team_size=1)

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f"  Steps: {steps}, Total reward: {total_reward:.2f}")

    env.close()
    print("\nDone!")


def main() -> None:
    """Parse arguments and run inference."""
    parser = argparse.ArgumentParser(description="Run a trained Rocket League agent")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("./models/rl_agent_final.zip"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )
    args = parser.parse_args()

    if not args.model.exists():
        # Try without .zip extension
        model_path = args.model.with_suffix("")
        if not model_path.exists():
            print(f"Error: Model not found at {args.model}")
            print("Train a model first with: python train.py")
            return
        args.model = model_path

    play(args.model, args.episodes)


if __name__ == "__main__":
    main()
