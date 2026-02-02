#!/usr/bin/env python3
"""Benchmark different training configurations to find the fastest combo."""

import os
import sys
import time
import itertools

# =============================================================================
# CPU OPTIMIZATION: Disable E-cores, use only P-cores (i7-12700K)
# =============================================================================
if sys.platform == "win32":
    import ctypes
    kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
    handle = kernel32.GetCurrentProcess()
    P_CORE_MASK = 0xFFFF  # P-cores only (threads 0-15)
    kernel32.SetProcessAffinityMask(handle, P_CORE_MASK)
    print(f"CPU Affinity: P-cores only (mask=0x{P_CORE_MASK:X})")

# Thread limits - must be before imports
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.env import make_env


# =============================================================================
# CONFIGURATIONS TO TEST
# =============================================================================

# Number of parallel environments (more = more CPU usage)
N_ENVS_OPTIONS = [16, 24, 32, 48, 64]

# Steps per environment before update (higher = less sync overhead)
N_STEPS_OPTIONS = [2048, 4096, 8192]

# Batch size for gradient updates (higher = better GPU utilization)
# With 8GB VRAM and MLP policy, we can go HUGE
BATCH_SIZE_OPTIONS = [8192, 16384, 32768, 65536, 98304, 131072]

# Network architectures
NET_ARCH_OPTIONS = [
    {"name": "medium", "arch": dict(pi=[512, 512], vf=[512, 512])},
    {"name": "large", "arch": dict(pi=[512, 512, 256], vf=[512, 512, 256])},
    {"name": "xlarge", "arch": dict(pi=[1024, 512, 256], vf=[1024, 512, 256])},
]

# Benchmark duration (timesteps per config)
BENCHMARK_TIMESTEPS = 100_000


def benchmark_config(n_envs: int, n_steps: int, batch_size: int, net_arch: dict) -> dict:
    """Run a single benchmark configuration and return steps/sec."""
    
    # Skip invalid combinations
    buffer_size = n_envs * n_steps
    if batch_size > buffer_size:
        return None
    if buffer_size % batch_size != 0:
        return None
    
    try:
        # Create environments
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
        
        # Create model
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
            device="cuda" if torch.cuda.is_available() else "cpu",
            policy_kwargs={"net_arch": net_arch},
        )
        
        # Warmup (1 full rollout)
        model.learn(total_timesteps=n_envs * n_steps, progress_bar=False)
        
        # Benchmark
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.perf_counter()
        
        model.learn(total_timesteps=BENCHMARK_TIMESTEPS, progress_bar=False, reset_num_timesteps=False)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.perf_counter() - start_time
        
        steps_per_sec = BENCHMARK_TIMESTEPS / elapsed
        
        # Cleanup
        env.close()
        del model
        del env
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return {
            "n_envs": n_envs,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "buffer_size": buffer_size,
            "steps_per_sec": steps_per_sec,
            "elapsed": elapsed,
        }
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    print("=" * 70)
    print("BENCHMARK: Finding optimal training configuration")
    print("=" * 70)
    print(f"CPU: i7-12700K (8P + 4E cores)")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Timesteps per config: {BENCHMARK_TIMESTEPS:,}")
    print()
    
    results = []
    total_configs = len(N_ENVS_OPTIONS) * len(N_STEPS_OPTIONS) * len(BATCH_SIZE_OPTIONS) * len(NET_ARCH_OPTIONS)
    valid_configs = 0
    
    print(f"Testing up to {total_configs} configurations...\n")
    
    for net_info in NET_ARCH_OPTIONS:
        net_name = net_info["name"]
        net_arch = net_info["arch"]
        
        print(f"\n--- Network: {net_name} ---")
        
        for n_envs in N_ENVS_OPTIONS:
            for n_steps in N_STEPS_OPTIONS:
                for batch_size in BATCH_SIZE_OPTIONS:
                    buffer_size = n_envs * n_steps
                    
                    # Skip invalid
                    if batch_size > buffer_size or buffer_size % batch_size != 0:
                        continue
                    
                    valid_configs += 1
                    config_str = f"envs={n_envs}, steps={n_steps}, batch={batch_size}"
                    print(f"  [{valid_configs}] Testing {config_str}...", end=" ", flush=True)
                    
                    result = benchmark_config(n_envs, n_steps, batch_size, net_arch)
                    
                    if result:
                        result["net_arch"] = net_name
                        results.append(result)
                        print(f"{result['steps_per_sec']:,.0f} steps/sec")
                    else:
                        print("SKIPPED")
    
    print("\n" + "=" * 70)
    print("RESULTS (sorted by speed)")
    print("=" * 70)
    
    # Sort by speed
    results.sort(key=lambda x: x["steps_per_sec"], reverse=True)
    
    # Print top 10
    print(f"\n{'Rank':<5} {'Envs':<6} {'Steps':<7} {'Batch':<7} {'Net':<8} {'Steps/sec':<12} {'Buffer':<10}")
    print("-" * 70)
    
    for i, r in enumerate(results[:15], 1):
        print(f"{i:<5} {r['n_envs']:<6} {r['n_steps']:<7} {r['batch_size']:<7} {r['net_arch']:<8} {r['steps_per_sec']:>10,.0f}  {r['buffer_size']:>10,}")
    
    # Best config
    if results:
        best = results[0]
        print("\n" + "=" * 70)
        print("BEST CONFIGURATION")
        print("=" * 70)
        print(f"  n_envs:     {best['n_envs']}")
        print(f"  n_steps:    {best['n_steps']}")
        print(f"  batch_size: {best['batch_size']}")
        print(f"  net_arch:   {best['net_arch']}")
        print(f"  Speed:      {best['steps_per_sec']:,.0f} steps/sec")
        print(f"  Buffer:     {best['buffer_size']:,} samples/update")
        
        # Estimate training time
        for target in [10_000_000, 100_000_000, 320_000_000, 1_000_000_000]:
            hours = target / best['steps_per_sec'] / 3600
            print(f"  {target/1e6:.0f}M steps: ~{hours:.1f} hours")


if __name__ == "__main__":
    main()
