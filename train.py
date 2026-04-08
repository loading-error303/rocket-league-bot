#!/usr/bin/env python3
"""Train a Rocket League agent using PPO and rlgym-ppo.

Uses shared memory vectorization for ~25k steps/sec (4x faster than SB3).
"""

import argparse
import os
import sys
from io import StringIO
from pathlib import Path
import contextlib

# Check if CPU-only mode is requested (before importing torch)
if "--cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Hide all CUDA devices

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
    from src.rewards import (SpeedReward, TurnLeftReward, ForwardReward, DriveToOpponentGoalReward,
                              JumpReward, BoostPickupReward, SpeedTowardBallReward,
                              TouchBallReward, BallSpeedAfterTouchReward, GoalScoredReward)
    
    # Environment config
    tick_skip = 12  # 10 Hz decision rate (faster training)
    timeout_seconds = 15
    
    # Action parser with frame skip
    action_parser = RepeatAction(LookupTableAction(), repeats=tick_skip)
    
    # Done conditions
    termination_condition = GoalCondition()
    truncation_condition = NoTouchTimeoutCondition(timeout_seconds=timeout_seconds)
    
    # Reward functions with weights
    rewards_and_weights = (
        (SpeedTowardBallReward(), 0.25),        # Continuous: close the gap to ball
        (TouchBallReward(), 5.0),              # One-shot: reward each ball touch
        (BallSpeedAfterTouchReward(), 3.0),    # One-shot: reward powerful hits
        (GoalScoredReward(), 20.0)            # Big reward for scoring, penalty for conceding
        # (JumpReward(), 3.0),                   # Continuous: reward jumping
        # (BoostPickupReward(), 1.0),            # One-shot: collect boost pads
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


def train(timesteps: int, n_proc: int, checkpoint_freq: int, force_cpu: bool = False) -> None:
    """Train the agent using PPO with rlgym-ppo.

    Args:
        timesteps: Additional training timesteps to run.
        n_proc: Number of parallel environment processes.
        checkpoint_freq: Save checkpoint every N timesteps.
        force_cpu: Force CPU-only training (avoids CUDA issues).
    """
    import json
    from rlgym_ppo import Learner
    
    # Use GPU if available and not forcing CPU
    if force_cpu:
        device = "cpu"
        print("Forcing CPU-only training (GPU disabled)")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Find the current cumulative timesteps from the latest checkpoint
    # so --timesteps means "train N more steps" not "train until N total"
    checkpoints_dir = Path("checkpoints")
    cumulative_so_far = 0
    if checkpoints_dir.exists():
        step_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and (d / "BOOK_KEEPING_VARS.json").exists()]
        if step_dirs:
            latest = max(step_dirs, key=lambda p: int(p.name))
            with open(latest / "BOOK_KEEPING_VARS.json") as f:
                cumulative_so_far = json.load(f).get("cumulative_timesteps", 0)
            print(f"Found checkpoint at {cumulative_so_far:,} cumulative steps")
    
    total_limit = cumulative_so_far + timesteps
    print(f"Will train {timesteps:,} additional steps (limit: {total_limit:,})")
    
    # Minimum batch size for inference
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    
    # PPO hyperparameters optimized for speed
    ppo_batch_size = 200_000
    ts_per_iteration = 200_000
    exp_buffer_size = 600_000
    ppo_epochs = 2
    ppo_minibatch_size = 100_000
    
    # Network architecture
    policy_layer_sizes = (256, 256, 128)
    critic_layer_sizes = (256, 256, 128)
    
    print(f"Starting training: {timesteps:,} additional timesteps, {n_proc} processes")
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
        timestep_limit=total_limit,
        
        # Logging
        log_to_wandb=False,
        
        # Device
        device=device,
    )
    
    # Set up TensorBoard logging
    from torch.utils.tensorboard import SummaryWriter
    tb_log_dir = Path("./tensorboard_logs")
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=str(tb_log_dir))

    # Patch rlgym-ppo reporting to be more concise + log to TensorBoard
    import rlgym_ppo.util.reporting as reporting
    original_report = reporting.report_metrics
    iteration_count = [0]
    
    def concise_report(loggable_metrics, debug_metrics, wandb_run):
        iteration_count[0] += 1
        steps = loggable_metrics.get("Cumulative Timesteps", 0)

        # Log all metrics to TensorBoard
        for key, value in loggable_metrics.items():
            if isinstance(value, (int, float)):
                tb_writer.add_scalar(key, value, steps)
        tb_writer.flush()

        if iteration_count[0] % 5 == 0:  # Print every 5 iterations
            reward = loggable_metrics.get("Policy Reward", 0)
            sps = loggable_metrics.get("Overall Steps per Second", 0)
            print(f"[{steps:,} steps] Reward: {reward:.2f} | Speed: {sps:,.0f} steps/sec")
        original_report(loggable_metrics, debug_metrics, wandb_run)
    
    reporting.report_metrics = concise_report
    
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
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU-only training (avoids CUDA/GPU issues)",
    )
    args = parser.parse_args()

    train(args.timesteps, args.n_envs, args.checkpoint_freq, args.cpu)


if __name__ == "__main__":
    main()
