#!/usr/bin/env python3
"""Train a Rocket League agent using PPO and RocketSim."""

import argparse
import os
import sys

# =============================================================================
# CPU OPTIMIZATION: Disable E-cores, use only P-cores (i7-12700K)
# =============================================================================
# P-cores: 8 cores with HT = 16 threads (logical processors 0-15)
# E-cores: 4 cores = 4 threads (logical processors 16-19)
# We want to use ONLY P-cores for maximum single-thread performance

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    
    # Set affinity to P-cores only (0-15) = 0xFFFF
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    handle = kernel32.GetCurrentProcess()
    # Affinity mask: bits 0-15 = P-core threads
    P_CORE_MASK = 0xFFFF  # 16 P-core threads
    kernel32.SetProcessAffinityMask(handle, P_CORE_MASK)
    print(f"CPU Affinity: P-cores only (mask=0x{P_CORE_MASK:X})")

# Thread limits for math libraries (use 4 threads for BLAS ops)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

# Force NumPy to use P-cores by setting affinity for OpenMP threads
os.environ["GOMP_CPU_AFFINITY"] = "0-15"      # GNU OpenMP (gcc-compiled libs)
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"  # Intel OpenMP
os.environ["KMP_HW_SUBSET"] = "16c,1t"        # Intel: 16 cores, 1 thread each

import torch
torch.set_num_threads(4)

# Also limit NumPy at runtime (more reliable than env vars)
try:
    import threadpoolctl
    threadpoolctl.threadpool_limits(limits=4, user_api='blas')
    threadpoolctl.threadpool_limits(limits=4, user_api='openmp')
    print("ThreadPoolCtl: BLAS/OpenMP limited to 4 threads")
except ImportError:
    pass  # threadpoolctl not installed, env vars will work

# GPU optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for faster matmul
    torch.backends.cudnn.allow_tf32 = True
    
    # Compile model for faster inference (PyTorch 2.0+)
    torch._dynamo.config.suppress_errors = True
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"PyTorch compile: enabled")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.config import TrainingConfig
from src.env import make_env


def train(config: TrainingConfig) -> None:
    """Train the agent using PPO.

    Args:
        config: Training configuration and hyperparameters.
    """
    print(f"Creating {config.n_envs} parallel environments...")
    
    def _make_env():
        return make_env(
            spawn_opponents=config.spawn_opponents,
            team_size=config.team_size,
            action_repeat=config.action_repeat,
            no_touch_timeout=config.no_touch_timeout,
            game_timeout=config.game_timeout,
        )
    
    # Use SubprocVecEnv for parallel simulation (big speedup)
    env = SubprocVecEnv([_make_env for _ in range(config.n_envs)])
    env = VecMonitor(env)  # Adds episode stats tracking

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Batch size: {config.batch_size}, Buffer: {config.n_envs * config.n_steps}")

    # Custom policy with larger network for better GPU utilization
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),  # Bigger network
    )

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
        device="cuda",  # Force GPU
        policy_kwargs=policy_kwargs,
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
    parser.add_argument(
        "--n-envs",
        type=int,
        default=24,
        help="Number of parallel environments (default: 24 for i7-12700K)",
    )
    args = parser.parse_args()

    config = TrainingConfig(
        total_timesteps=args.timesteps,
        team_size=args.team_size,
        spawn_opponents=not args.no_opponents,
        n_envs=args.n_envs,
    )

    train(config)


if __name__ == "__main__":
    main()
