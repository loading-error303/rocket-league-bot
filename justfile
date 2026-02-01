# Rocket League RL Bot - Task Runner
# Install just: winget install Casey.Just

# Default recipe
default:
    @just --list

# Setup the project (create venv and install deps)
setup:
    python -m venv .venv
    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install uv
    .\.venv\Scripts\python.exe -m uv sync

# Train the model (10M timesteps, 16 parallel envs)
train timesteps="10000000" envs="16":
    .\.venv\Scripts\python.exe train.py --timesteps {{timesteps}} --n-envs {{envs}}

# Quick training run (1M timesteps)
train-quick:
    .\.venv\Scripts\python.exe train.py --timesteps 1000000 --n-envs 16

# Watch the trained bot with RLViser
watch episodes="5" speed="1.0":
    .\.venv\Scripts\python.exe watch.py --episodes {{episodes}} --speed {{speed}}

# Run inference without visualization
run episodes="5":
    .\.venv\Scripts\python.exe bot.py --episodes {{episodes}}

# Start TensorBoard for monitoring
tensorboard:
    .\.venv\Scripts\python.exe -m tensorboard --logdir=./tensorboard_logs/

# Download RLViser executable
download-rlviser:
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/VirxEC/rlviser/releases/download/v0.8.0/rlviser-win64.exe' -OutFile 'rlviser.exe'"
    @echo "Downloaded rlviser.exe"

# Clean training outputs
clean:
    powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue checkpoints, tensorboard_logs, models"
    @echo "Cleaned training outputs"

# Clean everything including venv
clean-all: clean
    powershell -Command "Remove-Item -Recurse -Force -ErrorAction SilentlyContinue .venv"
    @echo "Cleaned all (including .venv)"

# Show GPU status
gpu-check:
    .\.venv\Scripts\python.exe -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Install PyTorch with CUDA support
install-cuda:
    .\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu130
