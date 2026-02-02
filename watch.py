#!/usr/bin/env python3
"""Watch a trained agent play with RLViser visualization."""

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import (
    AnyCondition,
    GoalCondition,
    NoTouchTimeoutCondition,
    TimeoutCondition,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)
from rlgym.rocket_league import common_values
from rlgym.rocket_league.rlviser import RLViserRenderer


def make_env_with_renderer():
    """Create environment with RLViser renderer."""
    action_parser = RepeatAction(LookupTableAction(), repeats=8)

    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=30),
        TimeoutCondition(timeout_seconds=300),
    )

    reward_fn = CombinedReward(
        (TouchReward(), 0.1),
        (GoalReward(), 10.0),
    )

    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([
            1 / common_values.SIDE_WALL_X,
            1 / common_values.BACK_NET_Y,
            1 / common_values.CEILING_Z,
        ]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator(),
    )

    # Create RLViser renderer for visualization
    renderer = RLViserRenderer()

    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        renderer=renderer,
    )

    return env


def watch(model_path: Path, num_episodes: int = 3, speed: float = 1.0):
    """Watch the trained agent play.

    Args:
        model_path: Path to the trained model.
        num_episodes: Number of episodes to watch.
        speed: Playback speed (1.0 = realtime, 2.0 = 2x speed).
    """
    print(f"Loading models from {model_path}...")
    model_blue = PPO.load(str(model_path))
    model_orange = PPO.load(str(model_path))

    print("Creating environment with RLViser...")
    env = make_env_with_renderer()

    # Time per step for realtime playback
    # 8 action repeat * (1/120) seconds per tick = 0.0667 seconds per step
    step_time = (8 / 120) / speed

    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        
        obs_dict = env.reset()
        agents = list(obs_dict.keys())
        
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            step_start = time.time()
            
            # Get actions for all agents (separate model instances with same weights)
            actions = {}
            for i, agent in enumerate(agents):
                obs = obs_dict[agent].astype(np.float32)
                if i == 0:
                    action, _ = model_blue.predict(obs, deterministic=True)
                else:
                    # First move is random for orange to vary kickoffs
                    if steps == 0:
                        action = np.random.randint(0, 90)
                    else:
                        action, _ = model_orange.predict(obs, deterministic=True)
                actions[agent] = np.array([action], dtype=np.int64)

            # Step environment
            obs_dict, reward_dict, terminated_dict, truncated_dict = env.step(actions)
            
            # Render
            env.render()

            total_reward += reward_dict[agents[0]]
            steps += 1
            
            # Check if done
            done = terminated_dict[agents[0]] or truncated_dict[agents[0]]

            # Maintain realtime playback
            elapsed = time.time() - step_start
            sleep_time = step_time - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"Episode finished: {steps} steps, reward: {total_reward:.2f}")

    print("\nDone watching!")


def main():
    parser = argparse.ArgumentParser(description="Watch trained agent play")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("./models/rl_agent_final.zip"),
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to watch",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed (1.0 = realtime)",
    )
    args = parser.parse_args()

    # Find model
    model_path = args.model
    if not model_path.exists():
        model_path = model_path.with_suffix(".zip")
    if not model_path.exists():
        # Try checkpoint
        checkpoints = list(Path("./checkpoints").glob("*.zip"))
        if checkpoints:
            model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Using latest checkpoint: {model_path}")
        else:
            print(f"Error: No model found at {args.model}")
            return

    watch(model_path, args.episodes, args.speed)


if __name__ == "__main__":
    main()
