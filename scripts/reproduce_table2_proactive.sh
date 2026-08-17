#!/usr/bin/env bash
# =============================================================================
# Table 2 -- Proactive anti-personalization  (Ours, Yo'LLaVA)
# -----------------------------------------------------------------------------
# Reference images are PROTECTED before personalization; the victim is trained on
# them and evaluated on CLEAN test images. See docs/REPRODUCE.md.
# TEMPLATE -- fill the <...> paths.
# =============================================================================
set -euo pipefail

ANTI=$(cd "$(dirname "$0")/.." && pwd)
LLAVA_MODEL=<path-to-llava-v1.5-13b>
CKPT=<checkpoints-out-root>
DATA=<yollava-data-root>
SKS=<identity>                           # repeat over all 10
GPU=${1:-0}

# 1) protect the TRAIN / reference images with Ours (attack env):
bash "$ANTI/scripts/protect.sh" "$DATA/train/$SKS" "$ANTI/protected_train/$SKS"

cd "$ANTI/victims/yollava"

# 2) personalize on the PROTECTED references:
CUDA_VISIBLE_DEVICES=$GPU python personalize.py \
    --model_path "$LLAVA_MODEL" \
    --data_root "$ANTI/protected_train" --sks_name "$SKS" \
    --checkpoint_path "$CKPT" --exp_name ours --prefix_token 16 --seed 42

# 3) evaluate recognition on the CLEAN held-out test images:
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --model_path "$LLAVA_MODEL" --checkpoint_path "$CKPT" \
    --sks_name "$SKS" --exp_name ours --epoch best --prefix_token 16 \
    --data_root "$DATA/test" --save_txt
