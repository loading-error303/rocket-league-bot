#!/usr/bin/env python3
"""Train a Rocket League agent using PPO and RocketSim."""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from src.config import TrainingConfig
from src.env import make_env


def train(config: TrainingConfig) -> None:
    """Train the agent using PPO.

    Args:
        config: Training configuration and hyperparameters.
    """
    print("Creating environment...")
    env = make_env(
        spawn_opponents=config.spawn_opponents,
        team_size=config.team_size,
        action_repeat=config.action_repeat,
        no_touch_timeout=config.no_touch_timeout,
        game_timeout=config.game_timeout,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        tensorboard_log=str(config.tensorboard_dir),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=str(config.checkpoint_dir),
        name_prefix="rl_agent",
    )

    print(f"Starting training for {config.total_timesteps:,} timesteps...")
    print(f"Checkpoints: {config.checkpoint_dir}")
    print(f"TensorBoard: {config.tensorboard_dir}")

    model.learn(
        total_timesteps=config.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True,
    )

    model.save(str(config.model_save_path))
    print(f"Training complete! Model saved to {config.model_save_path}")


def main() -> None:
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train a Rocket League RL agent")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=10_000_000,
        help="Total training timesteps (default: 10M)",
    )
    parser.add_argument(
        "--team-size",
        type=int,
        default=1,
        help="Players per team (default: 1)",
    )
    parser.add_argument(
        "--no-opponents",
        action="store_true",
        help="Train without opponents",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        total_timesteps=args.timesteps,
        team_size=args.team_size,
        spawn_opponents=not args.no_opponents,
    )

    train(config)


if __name__ == "__main__":
    main()
