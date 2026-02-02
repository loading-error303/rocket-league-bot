#!/usr/bin/env python3
"""Train a Rocket League agent using PPO and rlgym-ppo.

Uses shared memory vectorization for ~25k steps/sec (4x faster than SB3).
"""

import argparse
import os

# Thread limits for efficient CPU usage (set before importing numpy/torch)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import torch


def build_rlgym_env():
    """Build rlgym v2 environment for training."""
    from rlgym.api import RLGym
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import GoalCondition, NoTouchTimeoutCondition
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward, TouchReward
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator, KickoffMutator
    from rlgym.rocket_league import common_values
    from rlgym_ppo.util import RLGymV2GymWrapper
    
    # Import our custom rewards
    from src.rewards import (
        SpeedReward, SpeedTowardBallReward, BallSpeedReward,
        BoostPickupReward, BoostUsageReward, BoostPadDirectionReward,
        AirReward, FlipPenalty
    )
    
    # Environment config
    tick_skip = 8  # 15 Hz decision rate
    timeout_seconds = 30
    
    # Action parser with frame skip
    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    
    # Done conditions
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout_seconds=timeout_seconds)
    
    # Reward functions with weights
    rewards_and_weights = (
        # Core rewards
        (GoalReward(), 10.0),
        (TouchReward(), 0.5),
        
        # Speed rewards - encourage fast play
        (SpeedReward(), 0.1),
        (SpeedTowardBallReward(), 0.3),
        (BallSpeedReward(), 0.2),
        
        # Boost management
        (BoostPickupReward(), 0.05),
        (BoostUsageReward(), 0.02),
        (BoostPadDirectionReward(), 0.05),
        
        # Movement
        (AirReward(), 0.01),
        (FlipPenalty(), 0.1),
    )
    
    reward_fn = CombinedReward(*rewards_and_weights)
    
    # Observation builder with normalization
    obs_builder = DefaultObs(
        zero_padding=None,
        pos_coef=np.asarray([
            1 / common_values.SIDE_WALL_X,
            1 / common_values.BACK_NET_Y,
            1 / common_values.CEILING_Z
        ]),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL
    )
    
    # State mutator for team setup
    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=1, orange_size=1),
        KickoffMutator()
    )
    
    # Build the rlgym environment
    env = RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
    )
    
    # Wrap for rlgym-ppo compatibility
    return RLGymV2GymWrapper(env)


def train(timesteps: int, n_proc: int, checkpoint_freq: int) -> None:
    """Train the agent using PPO with rlgym-ppo.

    Args:
        timesteps: Total training timesteps.
        n_proc: Number of parallel environment processes.
        checkpoint_freq: Save checkpoint every N timesteps.
    """
    from rlgym_ppo import Learner
    
    # GPU optimizations
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.enable_flash_sdp(True)
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
    
    # Minimum batch size for inference
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    
    # PPO hyperparameters optimized for speed
    ppo_batch_size = 200_000
    ts_per_iteration = 200_000
    exp_buffer_size = 600_000
    ppo_epochs = 2
    ppo_minibatch_size = 100_000
    
    # Network architecture
    policy_layer_sizes = (512, 512, 256)
    critic_layer_sizes = (512, 512, 256)
    
    print(f"Starting training: {timesteps:,} timesteps, {n_proc} processes")
    print(f"PPO batch size: {ppo_batch_size:,}")
    print(f"Checkpoints every: {checkpoint_freq:,} steps")
    
    learner = Learner(
        env_create_function=build_rlgym_env,
        n_proc=n_proc,
        min_inference_size=min_inference_size,
        
        # PPO config
        ppo_batch_size=ppo_batch_size,
        ts_per_iteration=ts_per_iteration,
        exp_buffer_size=exp_buffer_size,
        ppo_minibatch_size=ppo_minibatch_size,
        ppo_epochs=ppo_epochs,
        ppo_ent_coef=0.01,
        
        # Network architecture
        policy_layer_sizes=policy_layer_sizes,
        critic_layer_sizes=critic_layer_sizes,
        
        # Learning rates
        policy_lr=3e-4,
        critic_lr=3e-4,
        
        # Normalization
        standardize_returns=True,
        standardize_obs=False,
        
        # Checkpointing
        checkpoints_save_folder="checkpoints",
        add_unix_timestamp=False,  # Use plain "checkpoints" folder
        save_every_ts=checkpoint_freq,
        timestep_limit=timesteps,
        
        # Logging
        log_to_wandb=False,
        
        # Device
        device="auto",
    )
    
    learner.learn()
    print("Training complete!")


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
        "--n-envs",
        type=int,
        default=48,
        help="Number of parallel environments (default: 48)",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1_000_000,
        help="Checkpoint frequency in timesteps (default: 1M)",
    )
    args = parser.parse_args()

    train(args.timesteps, args.n_envs, args.checkpoint_freq)


if __name__ == "__main__":
    main()
