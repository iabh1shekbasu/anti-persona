#!/usr/bin/env bash
# Environment setup for Anti-Persona (the protection attack).
# Usage:  bash setup_env.sh
set -euo pipefail

ENV_NAME="anti-persona"

conda create -n "$ENV_NAME" python=3.10 -y
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

pip install -r requirements.txt
# OpenAI CLIP (imported as `clip`) is not on PyPI under that name:
pip install git+https://github.com/openai/CLIP.git

echo
echo "[anti-persona] environment ready. Activate it with: conda activate ${ENV_NAME}"
echo "[anti-persona] For the victim evaluations, also install Yo'LLaVA and MyVLM"
echo "               into this environment (see README, 'Reproducing the tables')."
