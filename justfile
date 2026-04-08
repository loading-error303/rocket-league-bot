# Rocket League RL Bot - Task Runner
# Install just: winget install Casey.Just

# Use PowerShell on Windows
set shell := ["powershell.exe", "-NoLogo", "-Command"]

# Default recipe
default:
    @just --list

# Setup the project (create venv and install deps)
setup:
    uv venv -p 3.12
    uv sync

# Train the model (default: 10M timesteps, 48 parallel envs)
train timesteps="10000000" envs="48":
    uv run python train.py --timesteps {{timesteps}} --n-envs {{envs}}

# Quick training run (1M timesteps, ~1 minute)
train-quick:
    uv run python train.py --timesteps 1000000

# Train and then watch the results
train-watch timesteps="5000000" episodes="5":
    uv run python train.py --timesteps {{timesteps}}
    uv run python watch.py --episodes {{episodes}} --speed 1.0

# Watch the trained bot with RLViser
watch episodes="5" speed="1.0":
    uv run python watch.py --episodes {{episodes}} --speed {{speed}}

# Run inference without visualization
run episodes="5":
    uv run python bot.py --episodes {{episodes}}

# Start TensorBoard for monitoring
tensorboard:
    uv run tensorboard --logdir=./tensorboard_logs/

# Download RLViser executable
download-rlviser:
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/VirxEC/rlviser/releases/download/v0.8.0/rlviser-win64.exe' -OutFile 'rlviser.exe'"
    @echo "Downloaded rlviser.exe"

# Clean training outputs
clean:
    if (Test-Path checkpoints) { Remove-Item -Recurse -Force checkpoints }; if (Test-Path tensorboard_logs) { Remove-Item -Recurse -Force tensorboard_logs }; if (Test-Path models) { Remove-Item -Recurse -Force models }
    @echo "Cleaned training outputs"

# Clean everything including venv
clean-all: clean
    powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .venv"
    @echo "Cleaned all (including .venv)"

# Overnight training (~2 hours for 180M timesteps at 25k/sec)
overnight:
    @echo "Starting overnight training (180M timesteps, ~2 hours)..."
    uv run python train.py --timesteps 180000000
    @echo "Training complete! Launching visualization..."
    uv run python watch.py --episodes 10 --speed 1.0

# Show GPU status
gpu-check:
    uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Install PyTorch with CUDA support
install-cuda:
    uv sync
