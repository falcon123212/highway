#!/usr/bin/env bash
set -e

echo "=== Starting WSL2 Environment Setup ==="

# 1. Install Miniconda if not present
if [ ! -d "$HOME/miniconda3" ]; then
    echo "Miniconda not found. Downloading and installing..."
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    wget -q "$MINICONDA_URL" -O miniconda.sh
    bash miniconda.sh -b -p "$HOME/miniconda3"
    rm miniconda.sh
    echo "Miniconda installed successfully."
else
    echo "Miniconda already installed at $HOME/miniconda3."
fi

# Initialize conda for bash
echo "Initializing conda..."
"$HOME/miniconda3/bin/conda" init bash || true

# Source conda definitions to use conda commands directly in this shell
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# 2. Create Conda environment
ENV_NAME="poseidon_wsl"
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating Conda environment '$ENV_NAME' with Python 3.11..."
    conda create -y -n "$ENV_NAME" python=3.11
    echo "Conda environment created."
fi

# 3. Install packages inside the environment
echo "Installing pip dependencies in '$ENV_NAME'..."
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install vllm transformers sentence-transformers pandas numpy rank_bm25 tqdm openai

echo "=== WSL2 Environment Setup Complete ==="

