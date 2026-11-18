#!/usr/bin/env bash
# Optional conda setup. Install the PyTorch build matching your GPU first (https://pytorch.org).
# Usage: bash setup_env.sh
set -euo pipefail
REPO=$(cd "$(dirname "$0")" && pwd)
conda create -n anti-persona python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate anti-persona
python -m pip install -r "$REPO/requirements.txt"
echo "Activate with: conda activate anti-persona"
echo "Then: python scripts/download_weights.py && python scripts/download_coco100.py"
