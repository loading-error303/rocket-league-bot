"""Optimized vectorized environment using threading instead of subprocess.

RocketSim releases the GIL during physics steps, so we can use threads
instead of subprocesses to avoid IPC serialization overhead.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Callable
import gymnasium as gym
from gymnasium import spaces


class ThreadedVecEnv:
    """Vectorized environment using ThreadPoolExecutor.
    
    Much faster than SubprocVecEnv because:
    1. No IPC serialization overhead
    2. No subprocess spawn overhead
    3. RocketSim releases GIL during C++ physics
    """
    
    def __init__(self, env_fns: List[Callable[[], gym.Env]], n_threads: Optional[int] = None):
        """
        Args:
            env_fns: List of functions that create environments
            n_threads: Number of threads (default: len(env_fns))
        """
        self.n_envs = len(env_fns)
        self.n_threads = n_threads or self.n_envs
        
        # Create all environments in main thread
        self.envs = [fn() for fn in env_fns]
        
        # Get spaces from first env
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        
        # Pre-allocate arrays
        obs_shape = self.observation_space.shape
        self._obs = np.zeros((self.n_envs,) + obs_shape, dtype=np.float32)
        self._rewards = np.zeros(self.n_envs, dtype=np.float32)
        self._dones = np.zeros(self.n_envs, dtype=bool)
        self._infos = [{} for _ in range(self.n_envs)]
        
        # Thread pool
        self._executor = ThreadPoolExecutor(max_workers=self.n_threads)
        
    def reset(self):
        """Reset all environments."""
        def _reset(i):
            obs, info = self.envs[i].reset()
            self._obs[i] = obs
            self._infos[i] = info
        
        # Reset in parallel
        futures = [self._executor.submit(_reset, i) for i in range(self.n_envs)]
        for f in futures:
            f.result()
        
        return self._obs.copy()
    
    def step(self, actions: np.ndarray):
        """Step all environments with given actions."""
        
        def _step(i):
            obs, reward, terminated, truncated, info = self.envs[i].step(actions[i])
            
            # Auto-reset on done
            if terminated or truncated:
                obs, info = self.envs[i].reset()
                info['terminal_observation'] = obs
            
            self._obs[i] = obs
            self._rewards[i] = reward
            self._dones[i] = terminated or truncated
            self._infos[i] = info
        
        # Step in parallel
        futures = [self._executor.submit(_step, i) for i in range(self.n_envs)]
        for f in futures:
            f.result()
        
        return self._obs.copy(), self._rewards.copy(), self._dones.copy(), self._infos.copy()
    
    def close(self):
        """Clean up."""
        self._executor.shutdown(wait=True)
        for env in self.envs:
            env.close()


class BatchedRocketSimEnv(gym.Env):
    """Single environment that internally runs multiple RocketSim instances.
    
    This is even faster than ThreadedVecEnv because we can batch
    observations and actions as numpy arrays without any Python overhead.
    """
    
    def __init__(
        self,
        make_env_fn: Callable[[], gym.Env],
        n_instances: int = 32,
    ):
        """
        Args:
            make_env_fn: Function to create a single environment
            n_instances: Number of parallel game instances
        """
        self.n_instances = n_instances
        
        # Create all instances
        self.envs = [make_env_fn() for _ in range(n_instances)]
        
        # Combine observation/action spaces
        single_obs_space = self.envs[0].observation_space
        single_act_space = self.envs[0].action_space
        
        self.observation_space = spaces.Box(
            low=np.tile(single_obs_space.low, (n_instances, 1)),
            high=np.tile(single_obs_space.high, (n_instances, 1)),
            dtype=np.float32,
        )
        
        if isinstance(single_act_space, spaces.Discrete):
            self.action_space = spaces.MultiDiscrete([single_act_space.n] * n_instances)
        else:
            self.action_space = spaces.Box(
                low=np.tile(single_act_space.low, (n_instances, 1)),
                high=np.tile(single_act_space.high, (n_instances, 1)),
                dtype=single_act_space.dtype,
            )
        
        # Pre-allocate
        self._obs = np.zeros((n_instances,) + single_obs_space.shape, dtype=np.float32)
        self._executor = ThreadPoolExecutor(max_workers=n_instances)
    
    def reset(self, seed=None, options=None):
        def _reset(i):
            obs, _ = self.envs[i].reset()
            self._obs[i] = obs
        
        futures = [self._executor.submit(_reset, i) for i in range(self.n_instances)]
        for f in futures:
            f.result()
        
        return self._obs.flatten(), {}
    
    def step(self, actions):
        rewards = np.zeros(self.n_instances, dtype=np.float32)
        dones = np.zeros(self.n_instances, dtype=bool)
        
        def _step(i):
            obs, reward, term, trunc, _ = self.envs[i].step(actions[i])
            if term or trunc:
                obs, _ = self.envs[i].reset()
                dones[i] = True
            self._obs[i] = obs
            rewards[i] = reward
        
        futures = [self._executor.submit(_step, i) for i in range(self.n_instances)]
        for f in futures:
            f.result()
        
        return self._obs.flatten(), rewards.sum(), dones.all(), False, {}
    
    def close(self):
        self._executor.shutdown(wait=True)
        for env in self.envs:
            env.close()


def benchmark_threading():
    """Quick benchmark comparing SubprocVecEnv vs ThreadedVecEnv."""
    import time
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from src.env import make_env
    
    n_envs = 24
    n_steps = 1000
    
    def _make_env():
        return make_env(
            spawn_opponents=True,
            team_size=1,
            action_repeat=8,
            no_touch_timeout=30.0,
            game_timeout=300.0,
        )
    
    print("=" * 60)
    print("VECTORIZED ENV BENCHMARK")
    print("=" * 60)
    print(f"n_envs: {n_envs}")
    print(f"n_steps: {n_steps}")
    
    # Test SubprocVecEnv
    print("\n--- SubprocVecEnv (baseline) ---")
    env = SubprocVecEnv([_make_env for _ in range(n_envs)])
    obs = env.reset()
    
    start = time.perf_counter()
    for _ in range(n_steps):
        actions = np.array([env.action_space.sample() for _ in range(n_envs)])
        obs, rewards, dones, infos = env.step(actions)
    subproc_time = time.perf_counter() - start
    subproc_sps = (n_envs * n_steps) / subproc_time
    print(f"  Time: {subproc_time:.2f}s")
    print(f"  Steps/sec: {subproc_sps:,.0f}")
    env.close()
    
    # Test ThreadedVecEnv
    print("\n--- ThreadedVecEnv (optimized) ---")
    env = ThreadedVecEnv([_make_env for _ in range(n_envs)])
    obs = env.reset()
    
    start = time.perf_counter()
    for _ in range(n_steps):
        actions = np.array([env.action_space.sample() for _ in range(n_envs)])
        obs, rewards, dones, infos = env.step(actions)
    threaded_time = time.perf_counter() - start
    threaded_sps = (n_envs * n_steps) / threaded_time
    print(f"  Time: {threaded_time:.2f}s")
    print(f"  Steps/sec: {threaded_sps:,.0f}")
    env.close()
    
    print("\n--- Results ---")
    print(f"Speedup: {threaded_sps / subproc_sps:.2f}x")


if __name__ == "__main__":
    benchmark_threading()
