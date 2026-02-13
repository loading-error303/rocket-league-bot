#!/usr/bin/env python3
"""Watch a trained agent play with RLViser visualization.

Supports both Stable Baselines 3 (.zip) and rlgym-ppo (.pt) checkpoints.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

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

from src.rewards import OwnGoalPenalty, WiffOrWeakShotPenalty


def make_env(render: bool = True):
    """Create environment, optionally with RLViser renderer."""
    action_parser = RepeatAction(LookupTableAction(), repeats=8)

    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=30),
        TimeoutCondition(timeout_seconds=300),
    )

    reward_fn = CombinedReward(
        (TouchReward(), 0.1),
        (GoalReward(), 10.0),
        (OwnGoalPenalty(), 1.0),
        (WiffOrWeakShotPenalty(), 1.0),
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

    renderer = RLViserRenderer() if render else None

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


# ============================================================================
# Model Loading - supports both SB3 and rlgym-ppo formats
# ============================================================================

class RLGymPPOPolicy(torch.nn.Module):
    """Policy network matching rlgym-ppo's DiscreteFF architecture."""
    
    def __init__(self, input_dim: int, output_dim: int, layer_sizes: list):
        super().__init__()
        layers = []
        prev_size = input_dim
        for size in layer_sizes:
            layers.append(torch.nn.Linear(prev_size, size))
            layers.append(torch.nn.ReLU())
            prev_size = size
        layers.append(torch.nn.Linear(prev_size, output_dim))
        self.model = torch.nn.Sequential(*layers)
    
    def predict(self, obs: np.ndarray, deterministic: bool = True):
        """Get action from observation (matches SB3 interface)."""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)
            logits = self.model(obs_tensor)
            action = torch.argmax(logits, dim=-1).numpy() if deterministic else \
                     torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1).numpy()
        return action, None


def load_rlgym_ppo_policy(checkpoint_path: Path) -> RLGymPPOPolicy:
    """Load policy from rlgym-ppo checkpoint directory."""
    policy_path = checkpoint_path / "PPO_POLICY.pt"
    state_dict = torch.load(policy_path, map_location="cpu", weights_only=True)
    
    # Infer architecture from state dict
    layer_sizes, input_dim, output_dim = [], None, None
    for key, value in state_dict.items():
        if "weight" in key:
            if input_dim is None:
                input_dim = value.shape[1]
            idx = int(key.split(".")[1])
            if f"model.{idx + 2}.weight" in state_dict:
                layer_sizes.append(value.shape[0])
            else:
                output_dim = value.shape[0]
    
    print(f"rlgym-ppo model: input={input_dim}, layers={layer_sizes}, output={output_dim}")
    policy = RLGymPPOPolicy(input_dim, output_dim, layer_sizes)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy


def find_latest_rlgym_ppo_checkpoint() -> Path | None:
    """Find the latest rlgym-ppo checkpoint directory."""
    # Check the standard checkpoints folder first
    checkpoints_dir = Path("./checkpoints")
    if checkpoints_dir.exists():
        step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and (d / "PPO_POLICY.pt").exists()]
        if step_dirs:
            return max(step_dirs, key=lambda p: int(p.name))
    
    # Fall back to timestamped folders (legacy)
    checkpoint_dirs = list(Path(".").glob("checkpoints_ppo-*")) + list(Path(".").glob("checkpoints-*"))
    if not checkpoint_dirs:
        return None
    latest_dir = max(checkpoint_dirs, key=lambda p: int(p.name.split("-")[1]))
    step_dirs = [d for d in latest_dir.iterdir() if d.is_dir()]
    return max(step_dirs, key=lambda p: int(p.name)) if step_dirs else None


def load_model(model_path: Path):
    """Auto-detect model format and load appropriately."""
    # rlgym-ppo checkpoint directory
    if model_path.is_dir() and (model_path / "PPO_POLICY.pt").exists():
        print(f"Loading rlgym-ppo checkpoint: {model_path}")
        return load_rlgym_ppo_policy(model_path)
    
    # SB3 .zip file
    from stable_baselines3 import PPO
    zip_path = model_path if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
    print(f"Loading SB3 checkpoint: {zip_path}")
    return PPO.load(str(zip_path))


def watch(
    model_path: Path,
    num_episodes: int = 3,
    speed: float = 1.0,
    hours: float | None = None,
    headless: bool = False,
):
    """Watch the trained agent play.

    Args:
        model_path: Path to the trained model.
        num_episodes: Number of episodes to watch.
        speed: Playback speed (1.0 = realtime, 2.0 = 2x speed).
        hours: Optional wall-clock time limit in hours.
        headless: If True, disable visualization.
    """
    print(f"Loading models from {model_path}...")
    model_blue = load_model(model_path)
    model_orange = load_model(model_path)

    if headless:
        print("Creating environment (headless)...")
    else:
        print("Creating environment with RLViser...")
    env = make_env(render=not headless)

    # Time per step for realtime playback
    # 8 action repeat * (1/120) seconds per tick = 0.0667 seconds per step
    step_time = (8 / 120) / speed

    end_time = time.time() + (hours * 3600) if hours is not None else None

    episode = 0
    while True:
        if end_time is not None and time.time() >= end_time:
            break
        if num_episodes is not None and num_episodes > 0 and episode >= num_episodes:
            break

        episode += 1
        episode_label = f"{episode}/{num_episodes}" if num_episodes and num_episodes > 0 else f"{episode}/∞"
        print(f"\n=== Episode {episode_label} ===")
        
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
            if not headless:
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
        default=None,
        help="Path to trained model (if omitted, use latest checkpoint)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to watch (0 = unlimited)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed (1.0 = realtime)",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Run for a wall-clock time limit in hours",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable visualization",
    )
    args = parser.parse_args()

    # Find model - prefer rlgym-ppo checkpoints when no explicit model is provided
    if args.model is None:
        # First try rlgym-ppo checkpoints (our current training format)
        rlgym_ppo_ckpt = find_latest_rlgym_ppo_checkpoint()
        if rlgym_ppo_ckpt:
            model_path = rlgym_ppo_ckpt
            print(f"Using latest checkpoint: {model_path}")
        else:
            # Fall back to SB3 checkpoints
            checkpoints = list(Path("./checkpoints").glob("*.zip"))
            if checkpoints:
                model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
                print(f"Using latest SB3 checkpoint: {model_path}")
            else:
                print(f"Error: No model found")
                return
    else:
        model_path = args.model
        if not model_path.exists():
            model_path = model_path.with_suffix(".zip")
            if not model_path.exists():
                print(f"Error: Model not found at {model_path}")
                return

    watch(model_path, args.episodes, args.speed, args.hours, args.headless)


if __name__ == "__main__":
    main()
