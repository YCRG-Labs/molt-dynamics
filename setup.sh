#!/usr/bin/env bash
set -euo pipefail

echo "=== Molt Dynamics — VM Setup ==="

# 1. System deps (assumes Ubuntu/Debian)
echo "[1/6] Installing system packages..."
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3-pip git git-lfs

# 2. Clone the repo (skip if already present)
if [ ! -d "molt-dynamics" ]; then
    echo "[2/6] Cloning repository..."
    git clone https://github.com/ycrg-labs/molt-dynamics.git
    cd molt-dynamics
else
    echo "[2/6] Repository already exists, pulling latest..."
    cd molt-dynamics
    git pull
fi

# 3. Create venv + install deps
echo "[3/6] Setting up Python environment..."
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -e . -q

# 4. Download the Observatory Archive (uses git-lfs for parquet files)
if [ ! -d "moltbook-observatory-archive" ]; then
    echo "[4/6] Downloading Observatory Archive (~2 GB)..."
    git lfs install
    git clone https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive
else
    echo "[4/6] Observatory Archive already present, skipping download."
fi

# 5. Create output directory
echo "[5/6] Creating output directory..."
mkdir -p output

# 6. Run the pipeline
echo "[6/6] Running analysis pipeline..."
python -m molt_dynamics \
    --config config/default.yaml \
    --dataset-path moltbook-observatory-archive \
    --rq all

echo ""
echo "=== Done! Results are in output/ ==="
echo ""
echo "To re-run without reloading data:"
echo "  source .venv/bin/activate"
echo "  python -m molt_dynamics --skip-loading --rq all"
