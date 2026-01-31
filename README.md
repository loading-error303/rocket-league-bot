# 🚀 Rocket League RL Bot

Train a Rocket League agent using **RLGym** + **RocketSim** (headless simulator) and **Stable Baselines3** (PPO).

## 📦 Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

## 🛠️ Setup

```bash
# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate   # Linux/Mac

pip install uv
uv sync
```

## 🎮 Training

```bash
# Train with default settings (10M timesteps, 1v1)
uv run python train.py

# Customize training
uv run python train.py --timesteps 1000000 --team-size 2

# Train without opponents
uv run python train.py --no-opponents
```

### Monitor with TensorBoard

```bash
uv run tensorboard --logdir=./tensorboard_logs/
```

Open http://localhost:6006 to view training metrics.

## 🤖 Inference

```bash
# Run trained agent
uv run python bot.py --model ./models/rl_agent_final.zip --episodes 5
```

## 📁 Project Structure

```
.
├── src/
│   ├── __init__.py      # Package init
│   ├── config.py        # Training configuration
│   └── env.py           # RLGym environment wrapper
├── train.py             # Training script
├── bot.py               # Inference script
├── pyproject.toml       # Project dependencies
└── README.md
```

## ⚙️ Configuration

Edit `src/config.py` to customize:

- **Environment**: Team size, timeouts, action repeat
- **PPO Hyperparameters**: Learning rate, batch size, etc.
- **Training**: Timesteps, checkpoint frequency

## 🎯 Customization

### Custom Rewards

Create custom reward functions in `src/rewards.py`:

```python
from rlgym.api import RewardFunction, AgentID
from rlgym.rocket_league.api import GameState

class MyReward(RewardFunction[AgentID, GameState, float]):
    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        return {agent: 0.0 for agent in agents}
```

## 📚 Resources

- [RLGym Documentation](https://rlgym.org/)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)
- [RocketSim](https://github.com/ZealanL/RocketSim)

## 📄 License

MIT
