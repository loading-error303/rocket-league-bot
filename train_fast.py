"""Fast training using rlgym-ppo with shared memory vectorization.

This achieves 60k+ steps/sec instead of ~6k with SubprocVecEnv by using:
- Shared memory for IPC (no serialization overhead)
- UDP sockets for fast signaling
- Batched inference across all environments
"""

import numpy as np


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


if __name__ == "__main__":
    import os
    
    # =====================================
    # PERFORMANCE OPTIMIZATIONS
    # =====================================
    
    # Set thread limits for efficient CPU usage
    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["MKL_NUM_THREADS"] = "4"
    os.environ["OPENBLAS_NUM_THREADS"] = "4"
    
    import torch
    
    # GPU optimizations
    if torch.cuda.is_available():
        # Enable TF32 for faster matrix math on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Auto-tune convolutions for best performance
        torch.backends.cudnn.benchmark = True
        
        # Enable flash attention if available (faster transformers)
        torch.backends.cuda.enable_flash_sdp(True)
        
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    
    from rlgym_ppo import Learner
    
    # =====================================
    # TRAINING CONFIGURATION
    # =====================================
    
    # Process count - more processes = faster data collection
    # RocketSim is very lightweight (~114k tps single thread in pure C++)
    # Python overhead is the bottleneck, so spawn many processes
    # i7-12700K has 8P + 4E = 12 physical cores, 20 threads
    # NOTE: 80 processes caused "paging file too small" - reduced to 48
    n_proc = 48  # Balance between speed and memory usage
    
    # Minimum batch size for inference - higher = more efficient GPU batching
    # But too high means processes wait longer for inference
    min_inference_size = max(1, int(round(n_proc * 0.9)))
    
    # PPO hyperparameters - larger batches for faster GPU utilization
    # Larger batches = more efficient GPU compute, fewer Python loops
    ppo_batch_size = 200_000  # Doubled from 100k
    ts_per_iteration = 200_000  # Match batch size
    exp_buffer_size = 600_000  # 3x batch size
    ppo_epochs = 2
    ppo_minibatch_size = 100_000  # 50% of batch for gradient updates
    
    # Network architecture - smaller = faster inference
    # Balance between speed and capability
    policy_layer_sizes = (512, 512, 256)
    critic_layer_sizes = (512, 512, 256)
    
    # Learning rates
    policy_lr = 3e-4
    critic_lr = 3e-4
    
    print(f"Starting rlgym-ppo training with {n_proc} processes")
    print(f"PPO batch size: {ppo_batch_size:,}")
    print(f"Target: 30k+ steps/sec with optimized rewards")
    
    # Create the learner
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
        policy_lr=policy_lr,
        critic_lr=critic_lr,
        
        # Normalization
        standardize_returns=True,
        standardize_obs=False,
        
        # Checkpointing
        checkpoints_save_folder="checkpoints_ppo",
        save_every_ts=1_000_000,
        timestep_limit=1_000_000_000,
        
        # Logging
        log_to_wandb=False,
        
        # Device
        device="auto",  # Will use GPU if available
    )
    
    # Start training
    learner.learn()
