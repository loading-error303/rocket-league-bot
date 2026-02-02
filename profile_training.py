#!/usr/bin/env python3
"""Profile training to find bottlenecks."""

import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

# CPU affinity
if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    handle = kernel32.GetCurrentProcess()
    kernel32.SetProcessAffinityMask(handle, 0xFFFF)

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import numpy as np
import torch

torch.set_num_threads(4)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback
from src.env import make_env


# =============================================================================
# PROFILING CALLBACK
# =============================================================================

class ProfilingCallback(BaseCallback):
    """Callback that profiles each component of training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.times = defaultdict(list)
        self.step_times = []
        self.rollout_times = []
        self.train_times = []
        self._rollout_start = None
        self._train_start = None
        self._step_start = None
    
    def _on_rollout_start(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._rollout_start = time.perf_counter()
    
    def _on_step(self):
        return True
    
    def _on_rollout_end(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if self._rollout_start:
            elapsed = time.perf_counter() - self._rollout_start
            self.rollout_times.append(elapsed)
        self._train_start = time.perf_counter()
    
    def _on_training_end(self):
        pass
    
    def on_training_start(self, locals_, globals_):
        pass


def profile_training(n_envs=24, n_steps=4096, batch_size=8192, profile_updates=5):
    """Run training with detailed profiling."""
    
    print("=" * 70)
    print("TRAINING PROFILER")
    print("=" * 70)
    print(f"n_envs: {n_envs}")
    print(f"n_steps: {n_steps}")
    print(f"batch_size: {batch_size}")
    print(f"buffer_size: {n_envs * n_steps:,}")
    print(f"profile_updates: {profile_updates}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU: Not available, using CPU")
    
    # ==========================================================================
    # Environment creation
    # ==========================================================================
    print("\n--- Creating environments ---")
    env_start = time.perf_counter()
    
    def _make_env():
        return make_env(
            spawn_opponents=True,
            team_size=1,
            action_repeat=8,
            no_touch_timeout=30.0,
            game_timeout=300.0,
        )
    env = SubprocVecEnv([_make_env for _ in range(n_envs)])
    env = VecMonitor(env)
    
    env_time = time.perf_counter() - env_start
    print(f"  Created {n_envs} environments in {env_time:.2f}s")
    
    # ==========================================================================
    # Model creation
    # ==========================================================================
    print("\n--- Creating model ---")
    policy_kwargs = dict(
        net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256]),
    )
    
    model_start = time.perf_counter()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        device=device,
        policy_kwargs=policy_kwargs,
    )
    model_time = time.perf_counter() - model_start
    print(f"  Model created in {model_time:.2f}s (device: {model.device})")
    
    # ==========================================================================
    # Warmup
    # ==========================================================================
    print("\n--- Warmup (1 rollout) ---")
    warmup_start = time.perf_counter()
    model.learn(total_timesteps=n_envs * n_steps, progress_bar=False)
    warmup_time = time.perf_counter() - warmup_start
    print(f"  Warmup done in {warmup_time:.2f}s")
    
    # ==========================================================================
    # Profile with manual timing
    # ==========================================================================
    print(f"\n--- Profiling {profile_updates} updates ---")
    
    timesteps_per_update = n_envs * n_steps
    total_timesteps = profile_updates * timesteps_per_update
    
    # Manually time rollout vs training
    rollout_times = []
    train_times = []
    
    original_collect = model.collect_rollouts
    original_train = model.train
    
    def timed_collect(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_collect(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rollout_times.append(time.perf_counter() - start)
        return result
    
    def timed_train(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = original_train(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_times.append(time.perf_counter() - start)
        return result
    
    model.collect_rollouts = timed_collect
    model.train = timed_train
    
    profile_start = time.perf_counter()
    model.learn(total_timesteps=total_timesteps, progress_bar=True, reset_num_timesteps=False)
    profile_total = time.perf_counter() - profile_start
    
    # ==========================================================================
    # Additional profiling: pure env step speed
    # ==========================================================================
    print("\n--- Pure environment step speed ---")
    
    obs = env.reset()
    env_step_times = []
    
    for _ in range(100):
        actions = np.array([env.action_space.sample() for _ in range(n_envs)])
        start = time.perf_counter()
        obs, rewards, dones, infos = env.step(actions)
        env_step_times.append(time.perf_counter() - start)
    
    avg_env_step = np.mean(env_step_times) * 1000
    steps_per_sec_env = n_envs / np.mean(env_step_times)
    
    print(f"  Avg step time: {avg_env_step:.2f}ms ({steps_per_sec_env:,.0f} steps/sec)")
    
    # ==========================================================================
    # Additional profiling: pure inference speed
    # ==========================================================================
    print("\n--- Pure inference speed ---")
    
    obs_tensor = torch.as_tensor(obs, device=model.device, dtype=torch.float32)
    inference_times = []
    
    for _ in range(100):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            actions, values, log_probs = model.policy(obs_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_times.append(time.perf_counter() - start)
    
    avg_inference = np.mean(inference_times) * 1000
    inferences_per_sec = 1 / np.mean(inference_times)
    
    print(f"  Avg inference time: {avg_inference:.2f}ms ({inferences_per_sec:,.0f} batches/sec)")
    
    # ==========================================================================
    # Report
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PROFILING RESULTS")
    print("=" * 70)
    
    avg_rollout = np.mean(rollout_times) if rollout_times else 0
    avg_train = np.mean(train_times) if train_times else 0
    total_measured = avg_rollout + avg_train
    
    print(f"\n{'Component':<25} {'Time (s)':<12} {'% of Update':<15}")
    print("-" * 52)
    print(f"{'Rollout (env + inference)':<25} {avg_rollout:<12.3f} {avg_rollout/total_measured*100:<15.1f}")
    print(f"{'Training (GPU gradient)':<25} {avg_train:<12.3f} {avg_train/total_measured*100:<15.1f}")
    print("-" * 52)
    print(f"{'Total per update':<25} {total_measured:<12.3f}")
    
    steps_per_sec = total_timesteps / profile_total
    print(f"\n{'Overall throughput':<25} {steps_per_sec:,.0f} steps/sec")
    print(f"{'Timesteps per update':<25} {timesteps_per_update:,}")
    print(f"{'Updates profiled':<25} {profile_updates}")
    
    # Identify bottleneck
    print("\n" + "=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)
    
    if avg_rollout > avg_train:
        rollout_pct = avg_rollout / total_measured * 100
        print(f"\n🔥 BOTTLENECK: Rollout collection ({rollout_pct:.1f}% of time)")
        print("\n   This includes:")
        print(f"   • Environment stepping: {avg_env_step:.2f}ms per step")
        print(f"   • Model inference: {avg_inference:.2f}ms per batch")
        
        # Estimate breakdown within rollout
        total_steps = n_steps  # steps in one rollout
        estimated_env_time = total_steps * np.mean(env_step_times)
        estimated_inference_time = total_steps * np.mean(inference_times)
        other_time = avg_rollout - estimated_env_time - estimated_inference_time
        
        print(f"\n   Estimated breakdown per rollout ({avg_rollout:.2f}s):")
        print(f"   • Env steps:  ~{estimated_env_time:.2f}s ({estimated_env_time/avg_rollout*100:.1f}%)")
        print(f"   • Inference:  ~{estimated_inference_time:.2f}s ({estimated_inference_time/avg_rollout*100:.1f}%)")
        print(f"   • Overhead:   ~{max(0,other_time):.2f}s ({max(0,other_time)/avg_rollout*100:.1f}%)")
        
        print("\n   💡 Suggestions:")
        print("   • Increase n_envs to collect more data in parallel")
        print("   • Increase action_repeat to reduce steps needed")
        print("   • Use larger n_steps to amortize env reset overhead")
    else:
        train_pct = avg_train / total_measured * 100
        print(f"\n🔥 BOTTLENECK: GPU training ({train_pct:.1f}% of time)")
        print("\n   💡 Suggestions:")
        print("   • Increase batch_size for better GPU utilization")
        print("   • Decrease n_epochs (fewer gradient steps)")
        print("   • Use a smaller network architecture")
        print("   • Ensure GPU is being used (not CPU)")
    
    # Time estimates
    print("\n" + "=" * 70)
    print("TIME ESTIMATES")
    print("=" * 70)
    for target in [10_000_000, 100_000_000, 320_000_000, 1_000_000_000]:
        hours = target / steps_per_sec / 3600
        print(f"  {target/1e6:.0f}M steps: ~{hours:.1f} hours")
    
    env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-envs", type=int, default=24)
    parser.add_argument("--n-steps", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--profile-updates", type=int, default=5)
    args = parser.parse_args()
    
    profile_training(
        n_envs=args.n_envs,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        profile_updates=args.profile_updates,
    )
