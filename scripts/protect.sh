#!/usr/bin/env bash
# Generate protected images with Anti-Persona (paper "Ours": eps=4/255, sigma=0.8, dct_keep=5).
# Usage: bash scripts/protect.sh <input_dir> <output_dir>
set -euo pipefail

IN=${1:?usage: bash scripts/protect.sh <input_dir> <output_dir>}
OUT=${2:?usage: bash scripts/protect.sh <input_dir> <output_dir>}

python anti_persona/protect.py \
    --input_dir  "$IN" \
    --output_dir "$OUT" \
    --model openai/clip-vit-large-patch14-336 \
    --epsilon 4 --num_iters 500 --lr 0.5 --seed 42 \
    --averageloss_lambda 2.0 \
    --gaussian_sigma 0.8 --dct_keep 5
