"""Training configuration and hyperparameters."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    """PPO training hyperparameters."""

    # Environment
    spawn_opponents: bool = True
    team_size: int = 1
    action_repeat: int = 8
    no_touch_timeout: float = 30.0
    game_timeout: float = 300.0
    
    # ==========================================================================
    # MAXIMUM SPEED CONFIG for i7-12700K + RTX 5060
    # ==========================================================================
    # 24 parallel envs - saturates CPU (RocketSim is fast, more envs = more data)
    n_envs: int = 24

    # PPO hyperparameters
    learning_rate: float = 3e-4  # Higher LR with larger batches
    # More steps per update = more GPU work per iteration
    n_steps: int = 4096
    # 24 envs * 4096 steps = 98304 total samples per update
    # Large batch = better GPU utilization (RTX 5060 has 8GB VRAM)
    batch_size: int = 8192
    n_epochs: int = 4  # Fewer epochs with larger batches (same total gradient steps)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01

    # Training
    total_timesteps: int = 10_000_000
    checkpoint_freq: int = 100_000

    # Paths
    checkpoint_dir: Path = Path("./checkpoints")
    tensorboard_dir: Path = Path("./tensorboard_logs")
    model_save_path: Path = Path("./models/rl_agent_final")

    def __post_init__(self):
        """Ensure directories exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
