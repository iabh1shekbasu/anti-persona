#!/usr/bin/env bash
# =============================================================================
# Table 1 -- Reactive image protection  (Ours)
# -----------------------------------------------------------------------------
# Victim personalized on CLEAN references; the held-out TEST image is protected
# and the already-personalized victim is queried. Protection rate = fraction of
# test images the victim FAILS to recognize. See docs/REPRODUCE.md.
#
# Verified: `thao` with the authors' `best` checkpoint gives Ours = 1.000 (10/10).
# TEMPLATE -- fill the <...> paths for your setup.
# =============================================================================
set -euo pipefail

ANTI=$(cd "$(dirname "$0")/.." && pwd)   # this repo
LLAVA_MODEL=<path-to-llava-v1.5-13b>
CKPT=<path-to-pretrained_concepts/checkpoints>
DATA=<yollava-data-root>
SKS=<identity>                           # e.g. thao  (repeat over all 10)
GPU=${1:-0}

# 1) protect the identity's TEST images with Ours (attack env):
bash "$ANTI/scripts/protect.sh" "$DATA/test/$SKS" "$ANTI/repro/test/relaxed_smooth_eps4"

# 2) Yo'LLaVA recognition with the CLEAN (authors') checkpoint:
cd "$ANTI/victims/yollava"
CUDA_VISIBLE_DEVICES=$GPU python evaluate.py \
    --model_path "$LLAVA_MODEL" --checkpoint_path "$CKPT" \
    --sks_name "$SKS" --epoch best --exp_name "" --prefix_token 16 \
    --data_root "$ANTI/repro/test" --save_txt
# -> the "relaxed_smooth_eps4: x.xxx (c/t)" line is the protection rate.

# 3) MyVLM column (CVLface AdaFace FR head):
#   cd "$ANTI/victims/myvlm" && python run_paper_eval.py --concepts "$SKS" --variants clean ours
