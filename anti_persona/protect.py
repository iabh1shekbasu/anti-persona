#!/usr/bin/env python
"""
Average-prototype K ablation for IJCB 2026.

This script imports the normal attack implementation:
    attack.py

It does not modify the normal attack code.

K definition:
    For each attacked image, K is the number of OTHER identity images used to
    build the average prototype. The current attacked image is always excluded.

Default hyperparameters match relaxed_smooth_eps4_cleaner:
    epsilon=4
    num_iters=500
    lr=0.5
    seed=42
    resize_to_clip=True
    dynamic_scale=False
    gd_sign=True
    averageloss_lambda=2.0
    gaussian_smooth=True
    gaussian_kernel=3
    gaussian_sigma=0.8
    dct_lowpass=True
    dct_keep=5
    dct_mask_shape=triangular
    smooth_every=3
    model=openai/clip-vit-large-patch14-336
"""

import argparse
import importlib.util
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def import_base_attack_module(base_script_path=None):
    candidates = []

    if base_script_path is not None:
        candidates.append(Path(base_script_path).expanduser())

    here = Path(__file__).resolve().parent
    candidates.extend([
        here / "attack.py",
        here.parent / "attack.py",
    ])

    base_path = None
    for p in candidates:
        if p.exists():
            base_path = p
            break

    if base_path is None:
        msg = "Could not find attack.py. Tried:\n"
        msg += "\n".join(f"  {p}" for p in candidates)
        raise FileNotFoundError(msg)

    print(f"[INFO] Importing base attack code from: {base_path}")

    spec = importlib.util.spec_from_file_location("base_prm_attack_faceshield", str(base_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["base_prm_attack_faceshield"] = module
    spec.loader.exec_module(module)
    return module


def list_images(folder):
    paths = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and os.path.splitext(name)[1].lower() in VALID_EXTS:
            paths.append(path)
    return paths


def count_pngs(folder):
    if not os.path.isdir(folder):
        return 0
    return len([
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f))
        and os.path.splitext(f)[1].lower() == ".png"
    ])


def get_subset_leave_one_out_averageloss(
    per_image_features,
    feature_sum,
    exclude_idx,
    subset_size=-1,
    subset_seed=1234,
):
    """
    Build average-loss features from K other identity images.

    subset_size <= 0 or subset_size >= N-1:
        use all other images, matching original leave-one-out.

    subset_size = K:
        use exactly K other images.

    The attacked image exclude_idx is never included.
    """
    n = len(per_image_features)
    if n < 2:
        raise ValueError(f"Need at least 2 images for averageloss; got {n}")

    other_indices = [j for j in range(n) if j != exclude_idx]
    max_k = len(other_indices)

    if subset_size is None:
        subset_size = -1
    subset_size = int(subset_size)

    if subset_size <= 0 or subset_size >= max_k:
        excluded_features = per_image_features[exclude_idx]
        avg = []
        for layer_idx in range(len(feature_sum)):
            avg.append((feature_sum[layer_idx] - excluded_features[layer_idx]) / (n - 1))
        print(f"  Averageloss subset: using all K={max_k}/{max_k} other images")
        return avg, max_k, other_indices

    if subset_size < 1:
        raise ValueError(f"--averageloss_k must be -1/all or >=1, got {subset_size}")
    if subset_size > max_k:
        raise ValueError(f"Requested K={subset_size}, but max K={max_k}")

    rng = random.Random(int(subset_seed) + int(exclude_idx) * 1000003)
    selected = sorted(rng.sample(other_indices, subset_size))

    avg = []
    for layer_idx in range(len(per_image_features[0])):
        layer_sum = None
        for j in selected:
            feat = per_image_features[j][layer_idx]
            layer_sum = feat.clone() if layer_sum is None else layer_sum + feat
        avg.append(layer_sum / subset_size)

    print(f"  Averageloss subset: using K={subset_size}/{max_k} other images; indices={selected}")
    return avg, subset_size, selected


def main():
    parser = argparse.ArgumentParser("Average-loss K ablation wrapper")

    parser.add_argument("--base_script_path", type=str, default=None)

    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--target_path", type=str, default=None)
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model", type=str, default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--num_iters", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict_determinism", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--averageloss_k", type=int, default=-1)
    parser.add_argument("--averageloss_subset_seed", type=int, default=1234)
    parser.add_argument("--averageloss_lambda", type=float, default=2.0)

    # Defaults match relaxed_smooth_eps4_cleaner.
    parser.add_argument("--no_resize_to_clip", action="store_true")
    parser.add_argument("--resize_size", type=int, default=None)
    parser.add_argument("--dynamic_scale", action="store_true")
    parser.add_argument("--normalized_gd", action="store_true")

    parser.add_argument("--no_gaussian_smooth", action="store_true")
    parser.add_argument("--gaussian_kernel", type=int, default=3)
    parser.add_argument("--gaussian_sigma", type=float, default=0.8)
    parser.add_argument("--gaussian_mask_dilation", type=int, default=9)

    parser.add_argument("--no_dct_lowpass", action="store_true")
    parser.add_argument("--dct_block_size", type=int, default=8)
    parser.add_argument("--dct_keep", type=int, default=5)
    parser.add_argument("--dct_mask_shape", type=str, default="triangular", choices=["triangular", "square"])
    parser.add_argument("--smooth_every", type=int, default=3)

    parser.add_argument("--skip_if_complete", action="store_true")
    parser.add_argument("--overwrite_partial", action="store_true")

    args = parser.parse_args()

    if args.input_path is None and args.input_dir is None:
        raise ValueError("Provide --input_path or --input_dir")
    if args.target_path is None and args.target_dir is None:
        # Default neutral anchor: the bundled gray image next to this script.
        default_target = Path(__file__).resolve().parent / "assets" / "gray_image.png"
        if default_target.exists():
            args.target_path = str(default_target)
            print(f"[INFO] No target given; using bundled neutral anchor: {default_target}")
        else:
            raise ValueError("Provide --target_path or --target_dir")

    base = import_base_attack_module(args.base_script_path)

    resize_to_clip = not args.no_resize_to_clip
    dynamic_scale = args.dynamic_scale
    gd_sign = not args.normalized_gd
    gaussian_smooth = not args.no_gaussian_smooth
    dct_lowpass = not args.no_dct_lowpass

    print("=" * 70)
    print("Average-loss K ablation attack")
    print("=" * 70)
    print(f"Model:                  {args.model}")
    print(f"Epsilon:                {args.epsilon}")
    print(f"Iterations:             {args.num_iters}")
    print(f"LR:                     {args.lr}")
    print(f"Seed:                   {args.seed}")
    print(f"Resize to CLIP:          {resize_to_clip}")
    print(f"Dynamic scale:           {dynamic_scale}")
    print(f"GD sign:                 {gd_sign}")
    print(f"Averageloss lambda:      {args.averageloss_lambda}")
    print(f"Averageloss K:           {args.averageloss_k}")
    print(f"Gaussian smooth:         {gaussian_smooth}")
    print(f"Gaussian kernel/sigma:   {args.gaussian_kernel}/{args.gaussian_sigma}")
    print(f"DCT low-pass:            {dct_lowpass}")
    print(f"DCT keep/mask:           {args.dct_keep}/{args.dct_mask_shape}")
    print(f"Smooth every:            {args.smooth_every}")
    print(f"Output dir:              {args.output_dir}")
    print("=" * 70)

    base.set_global_seeds(args.seed, strict_determinism=args.strict_determinism)

    if args.input_path is not None:
        source_paths = [args.input_path]
    else:
        source_paths = list_images(args.input_dir)

    if args.target_path is not None:
        target_paths = [args.target_path]
    else:
        target_paths = list_images(args.target_dir)

    if len(source_paths) == 0:
        raise ValueError("No source images found")
    if len(target_paths) == 0:
        raise ValueError("No target images found")
    if len(source_paths) < 2:
        raise ValueError("K ablation requires at least 2 source images")

    max_k = len(source_paths) - 1
    if args.averageloss_k > max_k:
        raise ValueError(f"Requested K={args.averageloss_k}, but max K={max_k}")
    if args.averageloss_k != -1 and args.averageloss_k < 1:
        raise ValueError("--averageloss_k must be -1/all or >=1")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.skip_if_complete:
        existing = count_pngs(args.output_dir)
        if existing == len(source_paths):
            print(f"[SKIP] Output complete: {existing}/{len(source_paths)}")
            return
        if existing > 0 and existing < len(source_paths):
            if args.overwrite_partial:
                print(f"[CLEAN] Partial output: {existing}/{len(source_paths)}. Deleting PNGs.")
                for f in os.listdir(args.output_dir):
                    if f.lower().endswith(".png"):
                        os.remove(os.path.join(args.output_dir, f))
            else:
                raise RuntimeError("Partial output found. Pass --overwrite_partial to rerun.")

    print(f"Source images: {len(source_paths)}")
    print(f"Target images: {len(target_paths)}")
    print(f"Max K for this class: {max_k}")
    print()

    model_components = base.load_model(args.model, args.device)

    clip_resize = None
    if args.resize_size is not None:
        clip_resize = args.resize_size
        print(f"  Resize override: {clip_resize}x{clip_resize}")
    elif resize_to_clip:
        clip_resize = model_components.get("native_size", None)
        if clip_resize is None:
            clip_resize = 224
            print(f"  WARNING: No native_size found, defaulting to {clip_resize}")
        print(f"  Resizing all images to {clip_resize}x{clip_resize} before attack")
    print()

    print("=" * 70)
    print("Pre-computing identity features for K-prototype ablation")
    print("=" * 70)
    per_image_features, feature_sum = base.compute_averageloss_features(
        source_paths,
        model_components,
        args.device,
        clip_resize=clip_resize,
    )
    print()

    for i, src_path in enumerate(source_paths):
        fname = os.path.basename(src_path)
        print("=" * 70)
        print(f"[{i + 1}/{len(source_paths)}] Attacking: {fname}")
        print("=" * 70)

        base.seed_for_image(args.seed, i)
        print(f"  RNG seed for this image: {args.seed + i}")

        image_tensor = base.load_image(src_path)
        print(f"  Loaded source: {list(image_tensor.shape)}, range: [{image_tensor.min():.1f}, {image_tensor.max():.1f}]")

        if clip_resize is not None:
            image_tensor = F.interpolate(
                image_tensor.unsqueeze(0),
                (clip_resize, clip_resize),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 255)
            print(f"  Resized source to: {list(image_tensor.shape)}")

        tgt_path = target_paths[i % len(target_paths)]
        target_tensor = base.load_image(tgt_path)
        if clip_resize is not None:
            target_tensor = F.interpolate(
                target_tensor.unsqueeze(0),
                (clip_resize, clip_resize),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 255)

        print(f"  Target: {os.path.basename(tgt_path)} ({list(target_tensor.shape)})")

        avg_features, used_k, selected = get_subset_leave_one_out_averageloss(
            per_image_features=per_image_features,
            feature_sum=feature_sum,
            exclude_idx=i,
            subset_size=args.averageloss_k,
            subset_seed=args.averageloss_subset_seed,
        )

        print(f"  Using average prototype from K={used_k} other image(s), lambda={args.averageloss_lambda}")

        adv_image, delta = base.prm_attack(
            image_tensor=image_tensor,
            model_components=model_components,
            target_tensor=target_tensor,
            averageloss_features=avg_features,
            averageloss_lambda=args.averageloss_lambda,
            epsilon=args.epsilon,
            num_iters=args.num_iters,
            lr=args.lr,
            dynamic_scale=dynamic_scale,
            device=args.device,
            gd_sign=gd_sign,
            gaussian_smooth=gaussian_smooth,
            gaussian_kernel=args.gaussian_kernel,
            gaussian_sigma=args.gaussian_sigma,
            gaussian_mask_dilation=args.gaussian_mask_dilation,
            dct_lowpass=dct_lowpass,
            dct_block_size=args.dct_block_size,
            dct_keep=args.dct_keep,
            dct_mask_shape=args.dct_mask_shape,
            smooth_every=args.smooth_every,
        )

        out_path = os.path.join(args.output_dir, fname)
        saved_path = base.save_image(adv_image, out_path)

        print(f"  Saved: {saved_path}")
        print(f"  Output image size: {list(adv_image.shape)}")
        print(f"  Delta L_inf: {delta.abs().max().item():.2f}, L2: {delta.norm().item():.2f}")
        print()

    print("=" * 70)
    print("Done.")
    print(f"Adversarial images saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
