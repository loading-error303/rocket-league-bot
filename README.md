# 🚀 Rocket League RL Bot

Train a Rocket League agent using **RLGym** + **RocketSim** (headless simulator) and **Stable Baselines3** (PPO).

## 📦 Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- [just](https://github.com/casey/just) command runner (optional, for automation)

## 🛠️ Setup

### Quick Start (with just)

```powershell
# Install just (Windows)
winget install Casey.Just

# Setup and train
just setup
just train
```

### Manual Setup

```powershell
# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

pip install uv
uv sync
```

## 🎮 Training

```powershell
# Train with default settings (10M timesteps, 16 parallel envs)
just train

# Or manually:
.\.venv\Scripts\python.exe train.py --timesteps 10000000 --n-envs 16

# Customize training
.\.venv\Scripts\python.exe train.py --timesteps 1000000 --team-size 2

# Train without opponents
.\.venv\Scripts\python.exe train.py --no-opponents
```

### Monitor with TensorBoard

```powershell
just tensorboard
# Or: .\.venv\Scripts\python.exe -m tensorboard --logdir=./tensorboard_logs/
```

Open http://localhost:6006 to view training metrics.

## 👁️ Visualization (RLViser)

Watch your trained bot play in a 3D visualizer!

### Download RLViser

1. Download [RLViser v0.8.0](https://github.com/VirxEC/rlviser/releases/download/v0.8.0/rlviser-win64.exe) for Windows
2. Rename to `rlviser.exe` and place in project root

Or use just:
```powershell
just download-rlviser
```

### Watch the Bot

```powershell
# Watch trained agent (default: 5 episodes at 1x speed)
just watch

# Or manually with options:
.\.venv\Scripts\python.exe watch.py --model ./models/rl_agent_final.zip --episodes 10 --speed 2.0
```

**Controls:**
- The visualizer runs at configurable speed (1.0 = real-time, 2.0 = 2x speed)
- Close the RLViser window to stop

## 🤖 Inference (Headless)

```powershell
# Run trained agent without visualization
.\.venv\Scripts\python.exe bot.py --model ./models/rl_agent_final.zip --episodes 5
```

## 📁 Project Structure

```
.
├── src/
│   ├── __init__.py      # Package init
│   ├── config.py        # Training configuration
│   └── env.py           # RLGym environment wrapper + custom rewards
├── train.py             # Training script (parallel envs, GPU support)
├── bot.py               # Inference script (headless)
├── watch.py             # Visualization script (RLViser)
├── justfile             # Task automation
├── pyproject.toml       # Project dependencies
└── README.md
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

- **Environment**: Team size, timeouts, action repeat
- **PPO Hyperparameters**: Learning rate, batch size, etc.
- **Training**: Timesteps, checkpoint frequency
- **Parallel Envs**: Number of parallel environments (default: 8)

## 🎯 Custom Rewards

The bot uses multiple reward functions to learn:

| Reward | Weight | Purpose |
|--------|--------|---------|
| `TouchReward` | 0.1 | Encourage ball contact |
| `GoalReward` | 10.0 | Score goals, penalize own goals |
| `SpeedReward` | 0.01 | Reward high car speed |
| `SpeedTowardBallReward` | 0.05 | Move toward the ball |
| `AirReward` | 0.002 | Encourage aerial play |
| `BallSpeedReward` | 0.02 | Hit ball with power |

Edit `src/env.py` to customize rewards.

## 📚 Resources

- [RLGym Documentation](https://rlgym.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [RocketSim](https://github.com/ZealanL/RocketSim)
- [RLViser](https://github.com/VirxEC/rlviser)

## 📄 License

MIT
