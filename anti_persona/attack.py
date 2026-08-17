"""
PRM / FaceShield-style attack script for IJCB 2026 YoLLaVA experiments.

Purpose:
    This is the main attack-generation script. It creates protected /
    adversarial images by optimizing CLIP vision features.

    The attack is targeted: each source image is pushed away from its clean
    identity features and toward a target image's features.

Main options:
    --epsilon
        L-infinity perturbation budget in pixel scale.
        Example: --epsilon 4 means 4/255.

    --num_iters
        Number of optimization iterations per image.

    --lr
        Attack step size.

    --model
        Surrogate CLIP vision model.
        Examples:
            ViT-B/16
            ViT-L/14
            openai/clip-vit-large-patch14

    --gd_sign
        Use sign gradient descent, FaceShield-style.
        If not passed, uses mean-normalized gradient descent.

    --gaussian_smooth
        Apply FaceShield-style masked Gaussian smoothing to the update step.

    --dct_lowpass
        Apply 8x8 block-DCT low-pass projection to the perturbation.

    --dct_mask_shape triangular
        Uses JPEG-style triangular low-pass mask.
        This is the default and recommended setting.

    --smooth_every
        Apply Gaussian/DCT smoothing every K iterations.
        Default: 1, meaning every iteration.

    --use_averageloss
        Uses leave-one-out average identity features from the source identity.
        Requires at least 2 images in --input_dir.

Expected input:
    Single image mode:
        --input_path /path/to/source.png
        --target_path /path/to/target.png

    Folder mode:
        --input_dir /path/to/source_class_images
        --target_dir /path/to/random_or_target_images

Important:
    --input_dir is NOT recursive. It only reads image files directly inside
    the given folder.

Expected output:
    output_dir/
      0.png
      1.png
      ...

Dependencies:
    pip install torch torchvision pillow numpy

    For OpenAI CLIP models such as ViT-B/16:
        pip install git+https://github.com/openai/CLIP.git

    For HuggingFace CLIP models:
        pip install transformers

Example: single class, epsilon 4, FaceShield-style smoothing + DCT:
    python ijcb2026/attacks/prm_attack_faceshield_gaussian_v2.py \
      --input_dir /path/to/YoLLaVA/ijcb2026/input_train/train/ciin \
      --target_dir /path/to/YoLLaVA/ijcb2026/random-images \
      --output_dir /path/to/YoLLaVA/ijcb2026/output_paper/ciin/prm_faceshield_eps4 \
      --model ViT-B/16 \
      --epsilon 4 \
      --num_iters 250 \
      --lr 0.5 \
      --gd_sign \
      --gaussian_smooth \
      --gaussian_kernel 9 \
      --gaussian_sigma 5 \
      --gaussian_mask_dilation 9 \
      --dct_lowpass \
      --dct_block_size 8 \
      --dct_keep 5 \
      --dct_mask_shape triangular \
      --smooth_every 1 \
      --device cuda:0 \
      --seed 42

Example: epsilon 8:
    python ijcb2026/attacks/prm_attack_faceshield_gaussian_v2.py \
      --input_dir /path/to/YoLLaVA/ijcb2026/input_train/train/ciin \
      --target_dir /path/to/YoLLaVA/ijcb2026/random-images \
      --output_dir /path/to/YoLLaVA/ijcb2026/output_paper/ciin/prm_faceshield_eps8 \
      --model ViT-B/16 \
      --epsilon 8 \
      --num_iters 250 \
      --lr 0.5 \
      --gd_sign \
      --gaussian_smooth \
      --dct_lowpass \
      --dct_keep 5 \
      --dct_mask_shape triangular \
      --device cuda:0 \
      --seed 42

Example: use average-loss identity consistency:
    python ijcb2026/attacks/prm_attack_faceshield_gaussian_v2.py \
      --input_dir /path/to/YoLLaVA/ijcb2026/input_train/train/ciin \
      --target_dir /path/to/YoLLaVA/ijcb2026/random-images \
      --output_dir /path/to/YoLLaVA/ijcb2026/output_paper/ciin/prm_faceshield_avg_eps4 \
      --model ViT-B/16 \
      --epsilon 4 \
      --num_iters 250 \
      --lr 0.5 \
      --use_averageloss \
      --averageloss_lambda 0.1 \
      --gd_sign \
      --gaussian_smooth \
      --dct_lowpass \
      --device cuda:0 \
      --seed 42

Example: loop over all IJCB classes:
    CLASSES=(
      ciin
      denisdang
      khanhvy
      oong
      phuc-map
      thao
      thuytien
      viruss
      willinvietnam
      yuheng
    )

    GPU=0
    METHOD=prm_faceshield_eps4

    for CLS in "${CLASSES[@]}"; do
      python ijcb2026/attacks/prm_attack_faceshield_gaussian_v2.py \
        --input_dir /path/to/YoLLaVA/ijcb2026/input_train/train/${CLS} \
        --target_dir /path/to/YoLLaVA/ijcb2026/random-images \
        --output_dir /path/to/YoLLaVA/ijcb2026/output_paper/${CLS}/${METHOD} \
        --model ViT-B/16 \
        --epsilon 4 \
        --num_iters 250 \
        --lr 0.5 \
        --gd_sign \
        --gaussian_smooth \
        --dct_lowpass \
        --dct_keep 5 \
        --dct_mask_shape triangular \
        --device cuda:${GPU} \
        --seed 42
    done

Notes:
    - This script generates images only. It does not evaluate YoLLaVA accuracy.
    - After generating images, evaluate them using eval_yollava_all_classes_modular.sh.
    - For PSNR / SSIM / FID, use ijcb2026/utils/calc_all_metrics_nested_used_in_ijcb.py.
"""


"""
PRM attack — Version A (fixed): signed/normalized GD + FaceShield-style
masked Gaussian step smoothing + 8x8 block-DCT low-pass projection.

Fixes vs the earlier "Version A" file:

  CRITICAL BUG FIX — DCT was the identity transform.
    The original einsums contracted the same axis twice:
      tmp   = einsum('ij,bchwmj->bchwmi', D, blocks)    # contracts j
      coeff = einsum('bchwmj,ij->bchwmi', tmp, Dt)       # contracts j AGAIN
    The first einsum makes tmp's last axis be FREQUENCY (no longer spatial),
    but the second einsum reuses the letter 'j' for what's now the frequency
    axis and contracts it with D.T. Net result: coeff = blocks @ D.T @ D
    = blocks @ I = blocks. The "DCT" did nothing; the mask in between then
    just zeroed a structured pattern of spatial pixels in each 8x8 block.

    A self-test runs at module import time and aborts if reintroduced.

  TRIANGULAR DCT MASK is now the default (--dct_mask_shape triangular).
    Keeps coefficients where i+j <= dct_keep. This matches FaceShield's
    JPEG-style cutoff and is closer to an isotropic low-pass — the right
    choice when the downstream pipeline does bicubic resampling. Square
    mask still available via --dct_mask_shape square.

  --smooth_every knob added so the per-iteration projection can be
    relaxed for ablation studies. Default 1 (every iter, FaceShield default).

  DCT basis matrix cached per (N, device, dtype) instead of rebuilt
    inside dct_lowpass_project on every iteration.

Other behavior — signed/normalized GD update rule, FaceShield-style masked
Gaussian smoothing of the step (not the accumulated perturbation),
reflect-padding before DCT — is preserved unchanged.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ==============================================================================
# Seed Management
# ==============================================================================

def set_global_seeds(seed, strict_determinism=False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict_determinism:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        print(f"  Seed: {seed} (STRICT deterministic mode - may be slower)")
    else:
        print(f"  Seed: {seed}")


def seed_for_image(base_seed, image_idx):
    image_seed = base_seed + image_idx
    random.seed(image_seed)
    np.random.seed(image_seed)
    torch.manual_seed(image_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(image_seed)
        torch.cuda.manual_seed_all(image_seed)


# ==============================================================================
# Feature Extraction - OpenAI CLIP
# ==============================================================================

def grab_clip_features_openai_vit(x, clip_model):
    grid_size = round((clip_model.visual.positional_embedding.shape[0] - 1) ** 0.5)
    all_features = []

    x = clip_model.visual.conv1(x)
    b, _, gh, gw = x.size()
    x = x.reshape(x.shape[0], x.shape[1], -1)
    x = x.permute(0, 2, 1)

    cls_token = clip_model.visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)

    if x.shape[1] != clip_model.visual.positional_embedding.shape[0]:
        pos_embed = clip_model.visual.positional_embedding
        cls_pos = pos_embed[0:1, :]
        patch_pos = F.interpolate(
            pos_embed[1:, :].permute(1, 0).view(1, -1, grid_size, grid_size),
            size=(gh, gw),
            mode="bicubic",
            align_corners=False,
        ).reshape(-1, gh * gw).permute(1, 0)
        pos_embed = torch.cat([cls_pos, patch_pos])
        x = x + pos_embed.to(x.dtype)
    else:
        x = x + clip_model.visual.positional_embedding.to(x.dtype)

    x = clip_model.visual.ln_pre(x)
    x = x.permute(1, 0, 2)
    all_features.append(x)

    for resblock in clip_model.visual.transformer.resblocks:
        x = resblock(x)
        all_features.append(x)

    return all_features


# ==============================================================================
# Feature Extraction - HuggingFace CLIP
# ==============================================================================

def grab_clip_features_hf_vit(pixel_values, vision_model):
    embeddings = vision_model.embeddings

    x = embeddings.patch_embedding(pixel_values)
    B, D, gh, gw = x.shape
    x = x.reshape(B, D, gh * gw).permute(0, 2, 1)

    cls = embeddings.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1)

    pos = embeddings.position_embedding.weight
    n_pos = pos.shape[0]
    native_grid = round((n_pos - 1) ** 0.5)

    if x.shape[1] != n_pos:
        cls_pos = pos[:1]
        patch_pos = pos[1:]
        patch_pos = F.interpolate(
            patch_pos.T.reshape(1, D, native_grid, native_grid),
            size=(gh, gw),
            mode="bicubic",
            align_corners=False,
        ).reshape(D, gh * gw).T
        pos_current = torch.cat([cls_pos, patch_pos], dim=0)
    else:
        pos_current = pos

    x = x + pos_current.to(x.dtype)
    x = vision_model.pre_layrnorm(x)

    all_features = [x.permute(1, 0, 2).contiguous()]

    hidden_states = x
    for layer in vision_model.encoder.layers:
        try:
            layer_output = layer(
                hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
            )
        except TypeError:
            layer_output = layer(hidden_states, attention_mask=None)

        hidden_states = layer_output[0] if isinstance(layer_output, tuple) else layer_output
        all_features.append(hidden_states.permute(1, 0, 2).contiguous())

    return all_features


# ==============================================================================
# Averageloss token resizing helper
# ==============================================================================

def _resize_averageloss_tokens(avg_feat, target_num_tokens, has_cls=True):
    Tp = avg_feat.shape[0]
    if Tp == target_num_tokens:
        return avg_feat

    D = avg_feat.shape[-1]

    if has_cls:
        cls = avg_feat[:1]
        patches = avg_feat[1:]
        num_patches_src = Tp - 1
        num_patches_tgt = target_num_tokens - 1
        gp = int(round(num_patches_src ** 0.5))
        ga = int(round(num_patches_tgt ** 0.5))

        if gp * gp != num_patches_src or ga * ga != num_patches_tgt:
            vec = patches.squeeze(1).T.unsqueeze(0)
            vec_t = F.interpolate(vec, size=num_patches_tgt, mode="linear", align_corners=False)
            patches_t = vec_t.squeeze(0).T.unsqueeze(1)
            return torch.cat([cls, patches_t], dim=0)

        grid = patches.squeeze(1).T.reshape(1, D, gp, gp)
        grid_t = F.interpolate(grid, size=(ga, ga), mode="bicubic", align_corners=False)
        patches_t = grid_t.reshape(D, ga * ga).T.unsqueeze(1)
        return torch.cat([cls, patches_t], dim=0)

    gp = int(round(Tp ** 0.5))
    ga = int(round(target_num_tokens ** 0.5))
    if gp * gp != Tp or ga * ga != target_num_tokens:
        vec = avg_feat.squeeze(1).T.unsqueeze(0)
        vec_t = F.interpolate(vec, size=target_num_tokens, mode="linear", align_corners=False)
        return vec_t.squeeze(0).T.unsqueeze(1)

    grid = avg_feat.squeeze(1).T.reshape(1, D, gp, gp)
    grid_t = F.interpolate(grid, size=(ga, ga), mode="bicubic", align_corners=False)
    return grid_t.reshape(D, ga * ga).T.unsqueeze(1)


# ==============================================================================
# Loss
# ==============================================================================

def compute_prm_loss_targeted(
    adv_features,
    clean_features,
    target_features,
    averageloss_features=None,
    averageloss_lambda=0.1,
):
    loss = 0.0
    for layer_idx, (adv_feat, clean_feat, tgt_feat) in enumerate(
        zip(adv_features, clean_features, target_features)
    ):
        _, _, D = adv_feat.shape
        adv_norm = F.normalize(adv_feat.reshape(-1, D), dim=-1)
        clean_norm = F.normalize(clean_feat.reshape(-1, D), dim=-1)
        tgt_norm = F.normalize(tgt_feat.reshape(-1, D), dim=-1)

        cos_to_source = F.cosine_similarity(adv_norm, clean_norm.detach()).mean()
        cos_to_target = F.cosine_similarity(adv_norm, tgt_norm.detach()).mean()
        loss += cos_to_source - cos_to_target  

        if averageloss_features is not None:
            avg_feat = averageloss_features[layer_idx]
            avg_feat = _resize_averageloss_tokens(avg_feat, adv_feat.shape[0], has_cls=True)

            adv_slice = adv_feat.reshape(-1, D)
            avg_slice = avg_feat.reshape(-1, D)
            if adv_feat.shape[1] > 1:
                avg_slice = avg_slice.repeat(adv_feat.shape[1], 1)

            adv_avg_norm = F.normalize(adv_slice, dim=-1)
            avg_norm = F.normalize(avg_slice, dim=-1)
            cos_to_avg = F.cosine_similarity(adv_avg_norm, avg_norm.detach()).mean()
            loss += averageloss_lambda * cos_to_avg

    return loss


# ==============================================================================
# Averageloss feature computation
# ==============================================================================

def compute_averageloss_features(image_paths, model_components, device, clip_resize=None):
    clip_mean = model_components["mean"].to(device)
    clip_std = model_components["std"].to(device)
    grab_features = model_components["grab_features"]
    model = model_components["model"]
    clip_image_size = model_components.get("clip_image_size", None)

    print(f"  Computing averageloss features for {len(image_paths)} images...")

    per_image_features = []
    feature_sum = None

    with torch.no_grad():
        for idx, path in enumerate(image_paths):
            img_tensor = load_image(path).to(device)
            if clip_resize is not None:
                img_tensor = F.interpolate(
                    img_tensor.unsqueeze(0),
                    (clip_resize, clip_resize),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(0).clamp(0, 255)

            img_batch = img_tensor.unsqueeze(0)
            img_norm = (img_batch - clip_mean) / clip_std

            if clip_image_size is not None:
                img_norm = F.interpolate(
                    img_norm,
                    (clip_image_size, clip_image_size),
                    mode="bicubic",
                    align_corners=False,
                )

            features = grab_features(img_norm, model)
            features = [f.detach() for f in features]
            per_image_features.append(features)

            if feature_sum is None:
                feature_sum = [f.clone() for f in features]
            else:
                for layer_idx in range(len(feature_sum)):
                    feature_sum[layer_idx] = feature_sum[layer_idx] + features[layer_idx]

            print(f"    [{idx+1}/{len(image_paths)}] Cached features for {os.path.basename(path)}")

    sample_layer = per_image_features[0][0]
    n_layers = len(per_image_features[0])
    L, B, D = sample_layer.shape
    total_bytes = len(per_image_features) * n_layers * L * B * D * 4
    print(
        f"  Cached {len(per_image_features)} x {n_layers} layers, "
        f"each [{L}, {B}, {D}], total ~{total_bytes / (1024**2):.0f} MB"
    )

    print("  -- Identity consistency diagnostic --")
    with torch.no_grad():
        final_cls = torch.stack(
            [per_image_features[i][-1][0, 0] for i in range(len(per_image_features))],
            dim=0,
        )
        final_cls_norm = F.normalize(final_cls, dim=-1)
        pairwise_sim = final_cls_norm @ final_cls_norm.T

        n = len(per_image_features)
        mask = ~torch.eye(n, dtype=torch.bool, device=pairwise_sim.device)
        off_diag = pairwise_sim[mask]
        print(
            f"    Pairwise CLS cos sim (final layer): mean={off_diag.mean().item():.3f}, "
            f"min={off_diag.min().item():.3f}, max={off_diag.max().item():.3f}"
        )

        per_image_avg = (pairwise_sim.sum(dim=1) - 1.0) / (n - 1)
        worst_idx = per_image_avg.argmin().item()
        worst_sim = per_image_avg[worst_idx].item()
        if worst_sim < 0.5:
            print(
                f"    WARNING: Image {worst_idx} ({os.path.basename(image_paths[worst_idx])}) "
                f"has low avg similarity ({worst_sim:.3f}) to other identity images"
            )

    return per_image_features, feature_sum


def get_leave_one_out_averageloss(per_image_features, feature_sum, exclude_idx):
    n = len(per_image_features)
    if n < 2:
        raise ValueError(
            f"Cannot compute leave-one-out averageloss with only {n} image(s). Need at least 2."
        )

    excluded_features = per_image_features[exclude_idx]
    averageloss = []
    for layer_idx in range(len(feature_sum)):
        layer_mean = (feature_sum[layer_idx] - excluded_features[layer_idx]) / (n - 1)
        averageloss.append(layer_mean)
    return averageloss


# ==============================================================================
# FaceShield-style masked Gaussian smoothing of the step
# Applied to the update step, not to the accumulated perturbation.
# ==============================================================================

def gaussian_kernel2d(kernel_size=9, sigma=5.0, channels=3, device="cuda", dtype=torch.float32):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
    return kernel


def scale_tensor(tensor, eps=1e-12):
    """Min-max normalize a tensor for mask construction only."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 2:
        tensor = tensor[None, None, :, :]

    min_val = tensor.amin(dim=(-2, -1), keepdim=True)
    max_val = tensor.amax(dim=(-2, -1), keepdim=True)
    return (tensor - min_val) / (max_val - min_val + eps)


def calculate_gradients(image):
    """Sobel gradients used by FaceShield's line-mask construction."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    elif image.dim() == 2:
        image = image[None, None, :, :]

    B, C, _, _ = image.shape
    sobel_x = torch.tensor(
        [[1, 0, -1],
         [2, 0, -2],
         [1, 0, -1]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3).repeat(C, 1, 1, 1)
    sobel_y = torch.tensor(
        [[1, 2, 1],
         [0, 0, 0],
         [-1, -2, -1]],
        device=image.device,
        dtype=image.dtype,
    ).view(1, 1, 3, 3).repeat(C, 1, 1, 1)

    grad_x = F.conv2d(image, sobel_x, padding=1, groups=C)
    grad_y = F.conv2d(image, sobel_y, padding=1, groups=C)
    return grad_x, grad_y


def create_line_mask(image, dilation_kernel=9):
    """
    FaceShield-style line mask: pick strongest Sobel-response pixels,
    then dilate them so Gaussian blur is applied around line-like artifacts.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    elif image.dim() == 2:
        image = image[None, None, :, :]

    B, C, _, _ = image.shape
    grad_x, grad_y = calculate_gradients(image)
    grad_x_abs = grad_x.abs()
    grad_y_abs = grad_y.abs()

    # FaceShield used the global max; this per-image/channel max is equivalent
    # for batch size 1 and safer if batch size is ever increased.
    max_x = grad_x_abs.amax(dim=(-2, -1), keepdim=True)
    max_y = grad_y_abs.amax(dim=(-2, -1), keepdim=True)
    mask = ((grad_x_abs == max_x) | (grad_y_abs == max_y)).to(image.dtype)

    kernel = torch.ones((C, 1, dilation_kernel, dilation_kernel), device=image.device, dtype=image.dtype)
    dilated_mask = F.conv2d(mask, kernel, padding=dilation_kernel // 2, groups=C)
    return dilated_mask.clamp(0, 1)


def apply_gaussian_smoothing_step(step, kernel_size=9, sigma=5.0, mask_dilation=9):
    """
    FaceShield-style masked Gaussian smoothing.

    1. Build a line mask from the normalized update step.
    2. Gaussian-blur the update step.
    3. Replace only masked regions with the blurred version.

    This mirrors FaceShield's:
        d_rgb = scale_tensor(new_delta)
        mask = create_line_mask(..., d_rgb)
        new_delta = apply_gaussian(..., new_delta, mask, 9, 5)
    while avoiding hard-coded CUDA/device assumptions.
    """
    if step.dim() == 3:
        step = step.unsqueeze(0)
    elif step.dim() == 2:
        step = step[None, None, :, :]

    B, C, _, _ = step.shape
    mask_input = scale_tensor(step)
    mask = create_line_mask(mask_input, dilation_kernel=mask_dilation)

    kernel = gaussian_kernel2d(
        kernel_size=kernel_size,
        sigma=sigma,
        channels=C,
        device=step.device,
        dtype=step.dtype,
    )
    blurred_step = F.conv2d(step, kernel, padding=kernel_size // 2, groups=C)
    return blurred_step * mask + step * (1 - mask)


# ==============================================================================
# 8x8 block-DCT low-pass projection
# ==============================================================================

def dct_matrix(N, device, dtype):
    """Standard orthonormal DCT-II basis. mat[k, n] = alpha_k * cos((2n+1)*k*pi/(2N))."""
    mat = torch.empty((N, N), device=device, dtype=dtype)
    factor = math.pi / (2 * N)
    scale0 = math.sqrt(1.0 / N)
    scale = math.sqrt(2.0 / N)
    for k in range(N):
        alpha = scale0 if k == 0 else scale
        for n in range(N):
            mat[k, n] = alpha * math.cos((2 * n + 1) * k * factor)
    return mat


_DCT_BASIS_CACHE = {}


def _get_dct_basis(N, device, dtype):
    key = (N, str(device), str(dtype))
    if key not in _DCT_BASIS_CACHE:
        _DCT_BASIS_CACHE[key] = dct_matrix(N, device, dtype)
    return _DCT_BASIS_CACHE[key]


def _build_lowpass_mask(N, keep, shape, device, dtype):
    """
    keep:
      - shape='triangular' -> keep coefficients where (i + j) <= keep.
        For N=8, keep=5 -> 21 coefs; keep=6 -> 28 coefs.
      - shape='square'    -> keep top-left [keep, keep] block.
        For N=8, keep=5 -> 25 coefs.
    """
    if shape == "triangular":
        ii = torch.arange(N, device=device).view(N, 1)
        jj = torch.arange(N, device=device).view(1, N)
        return ((ii + jj) <= keep).to(dtype)
    if shape == "square":
        keep = min(max(int(keep), 0), N)
        m = torch.zeros((N, N), device=device, dtype=dtype)
        m[:keep, :keep] = 1.0
        return m
    raise ValueError(f"Unknown DCT mask shape: {shape!r} (use 'triangular' or 'square')")


def dct_lowpass_project(delta, block_size=8, keep=5, mask_shape="triangular"):
    """
    8x8 block DCT-II low-pass projection. Forward: coeff = D @ X @ D.T per block.
    Inverse: X = D.T @ coeff @ D per block. Mask zeros high-frequency coefficients
    between forward and inverse.
    """
    B, C, H, W = delta.shape
    N = block_size

    pad_h = (N - H % N) % N
    pad_w = (N - W % N) % N
    x = F.pad(delta, (0, pad_w, 0, pad_h), mode="reflect")

    Hp, Wp = x.shape[-2:]
    hb = Hp // N
    wb = Wp // N

    # Layout: [B, C, hb, wb, N (row-in-block), N (col-in-block)]
    blocks = x.view(B, C, hb, N, wb, N).permute(0, 1, 2, 4, 3, 5).contiguous()

    D = _get_dct_basis(N, x.device, x.dtype)

    # Forward 2D DCT-II per block: coeff = D @ blocks @ D.T
    #   step 1 (DCT along rows, contracts 'k'):   tmp = D @ blocks
    #     letters: D='ik', blocks='bchwkj', out='bchwij'   (k summed)
    #   step 2 (DCT along cols, contracts 'j'):   coeff = tmp @ D.T
    #     letters: tmp='bchwij', D='lj', out='bchwil'      (j summed; D[l,j] = D.T[j,l])
    tmp = torch.einsum("ik,bchwkj->bchwij", D, blocks)
    coeff = torch.einsum("bchwij,lj->bchwil", tmp, D)

    mask = _build_lowpass_mask(N, keep, mask_shape, x.device, x.dtype)
    coeff = coeff * mask.view(1, 1, 1, 1, N, N)

    # Inverse 2D DCT-II per block: recon = D.T @ coeff @ D
    #   step 1 (contracts 'i'):   tmp = D.T @ coeff
    #     letters: D='ik' so D.T[k,i] = D[i,k];  coeff='bchwij', out='bchwkj'  (i summed)
    #   step 2 (contracts 'j'):   recon = tmp @ D
    #     letters: tmp='bchwkj', D='jl', out='bchwkl'      (j summed)
    tmp = torch.einsum("ik,bchwij->bchwkj", D, coeff)
    recon = torch.einsum("bchwkj,jl->bchwkl", tmp, D)

    out = recon.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, Hp, Wp)
    return out[:, :, :H, :W]


def _dct_self_test():
    """Verify the DCT round-trip is identity and that forward DCT matches the
    textbook 2D DCT-II for a known input. Aborts if either check fails."""
    N = 8
    device = torch.device("cpu")
    dtype = torch.float64
    D = dct_matrix(N, device, dtype)

    # Orthogonality
    err_orth = (D @ D.T - torch.eye(N, dtype=dtype)).abs().max().item()
    assert err_orth < 1e-10, f"DCT basis not orthonormal: |D@D.T - I|_max = {err_orth}"

    # Forward DCT vs einsum, on a non-trivial block in a fake [B,C,hb,wb,N,N] tensor
    torch.manual_seed(0)
    X = torch.randn(N, N, dtype=dtype)
    expected = D @ X @ D.T
    blocks = X.view(1, 1, 1, 1, N, N)
    tmp = torch.einsum("ik,bchwkj->bchwij", D, blocks)
    coeff = torch.einsum("bchwij,lj->bchwil", tmp, D)
    err_fwd = (coeff[0, 0, 0, 0] - expected).abs().max().item()
    assert err_fwd < 1e-10, f"Forward DCT mismatch vs D @ X @ D.T: max err = {err_fwd}"

    # Round-trip (no mask) reconstructs X
    delta = X.view(1, 1, N, N).to(torch.float32)
    no_mask_keep = 2 * (N - 1)  # i+j <= 14 keeps everything for N=8
    rec = dct_lowpass_project(delta, block_size=N, keep=no_mask_keep, mask_shape="triangular")
    err_rt = (rec - delta).abs().max().item()
    assert err_rt < 1e-4, f"DCT round-trip not identity: max err = {err_rt}"


_dct_self_test()


# ==============================================================================
# Attack
# ==============================================================================

def prm_attack(
    image_tensor,
    model_components,
    target_tensor,
    averageloss_features=None,
    averageloss_lambda=0.1,
    epsilon=8.0,
    num_iters=250,
    lr=0.5,
    dynamic_scale=True,
    scale_range=(0.5, 1.0),
    device="cuda",
    gd_sign=False,
    gaussian_smooth=False,
    gaussian_kernel=9,
    gaussian_sigma=5.0,
    gaussian_mask_dilation=9,
    dct_lowpass=False,
    dct_block_size=8,
    dct_keep=5,
    dct_mask_shape="triangular",
    smooth_every=1,
):
    if target_tensor is None:
        raise ValueError("target_tensor is required")

    clip_mean = model_components["mean"].to(device)
    clip_std = model_components["std"].to(device)
    grab_features = model_components["grab_features"]
    model = model_components["model"]

    use_averageloss = averageloss_features is not None
    mode_str = (
        f"TARGETED + AVERAGELOSS (lambda={averageloss_lambda})"
        if use_averageloss else "TARGETED"
    )

    clean_image = image_tensor.unsqueeze(0).to(device)
    perturbation = torch.zeros_like(clean_image, requires_grad=True)

    target_image = target_tensor.unsqueeze(0).to(device)
    if target_image.shape != clean_image.shape:
        target_image = F.interpolate(
            target_image,
            clean_image.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

    update_mode = "sign-GD" if gd_sign else "normalized-GD"
    print(f"  Mode: {mode_str}")
    print(f"  Image shape: {list(clean_image.shape)}, device: {clean_image.device}")
    print(
        f"  Epsilon: {epsilon}, Step size: {lr}, Iters: {num_iters}, "
        f"Dynamic scale: {dynamic_scale}, Update: {update_mode}"
    )
    print(
        f"  Gaussian smooth: {gaussian_smooth} "
        f"(FaceShield masked, k={gaussian_kernel}, sigma={gaussian_sigma}, "
        f"mask_dilation={gaussian_mask_dilation})"
    )
    print(
        f"  DCT low-pass: {dct_lowpass} (block={dct_block_size}, keep={dct_keep}, "
        f"mask_shape={dct_mask_shape})"
    )
    print(f"  Smoothing applied every {smooth_every} iter(s)")

    first_ratios_log = []
    first_grids_log = []
    loss_history = []
    pert_linf_raw_history = []

    for iteration in range(num_iters):
        adv_image = (clean_image + perturbation.clamp(-epsilon, epsilon)).clamp(0.0, 255.0)

        if dynamic_scale:
            ratio = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
            h, w = adv_image.shape[2], adv_image.shape[3]
            target_h, target_w = int(ratio * h), int(ratio * w)
            adv_scaled = F.interpolate(adv_image, (target_h, target_w), mode="bicubic", align_corners=False)
            clean_scaled = F.interpolate(clean_image, (target_h, target_w), mode="bicubic", align_corners=False)
            tgt_scaled = F.interpolate(target_image, (target_h, target_w), mode="bicubic", align_corners=False)
        else:
            ratio = 1.0
            adv_scaled = adv_image
            clean_scaled = clean_image
            tgt_scaled = target_image

        if iteration < 3:
            first_ratios_log.append(ratio)

        adv_input = (adv_scaled - clip_mean) / clip_std
        clean_input = (clean_scaled - clip_mean) / clip_std
        tgt_input = (tgt_scaled - clip_mean) / clip_std

        if iteration < 3:
            first_grids_log.append(tuple(adv_input.shape[2:]))

        if iteration == 0:
            print(f"  Encoder input shape: {list(adv_input.shape)}")
            h_in, w_in = adv_input.shape[2], adv_input.shape[3]
            model_type = model_components.get("type", "unknown")
            if model_type == "openai":
                ps = model.visual.conv1.kernel_size[0]
                grid_h, grid_w = h_in // ps, w_in // ps
                native_grid = round((model.visual.positional_embedding.shape[0] - 1) ** 0.5)
                print(f"  Patch grid: {grid_h}x{grid_w} = {grid_h * grid_w} patches + 1 CLS = {grid_h * grid_w + 1} tokens")
                if grid_h != native_grid or grid_w != native_grid:
                    print(f"  Positional embedding INTERPOLATION active: {native_grid}x{native_grid} -> {grid_h}x{grid_w}")
                else:
                    print("  Using native positional embeddings (no interpolation)")
            elif model_type == "hf":
                n_pos = model.embeddings.position_embedding.weight.shape[0]
                native_grid = round((n_pos - 1) ** 0.5)
                ps = model.embeddings.patch_embedding.kernel_size[0]
                grid_h, grid_w = h_in // ps, w_in // ps
                print(f"  Patch grid: {grid_h}x{grid_w} = {grid_h * grid_w} patches + 1 CLS = {grid_h * grid_w + 1} tokens")
                if grid_h != native_grid or grid_w != native_grid:
                    print(f"  Positional embedding INTERPOLATION active: {native_grid}x{native_grid} -> {grid_h}x{grid_w}")
                else:
                    print("  Using native positional embeddings (no interpolation)")

        adv_features = grab_features(adv_input, model)
        with torch.no_grad():
            clean_features = grab_features(clean_input, model)
            tgt_features = grab_features(tgt_input, model)

        if iteration == 0:
            print(f"  Number of feature layers: {len(adv_features)}")
            print(f"  Feature shapes (first/last): {list(adv_features[0].shape)} / {list(adv_features[-1].shape)}")
            num_tokens = adv_features[0].shape[0]
            embed_dim = adv_features[0].shape[2]
            print(f"  Tokens per layer: {num_tokens} (1 CLS + {num_tokens - 1} patches), Embed dim: {embed_dim}")
            if use_averageloss:
                avg_shape = list(averageloss_features[0].shape)
                adv_shape = list(adv_features[0].shape)
                if avg_shape[2] != adv_shape[2]:
                    raise RuntimeError(
                        f"Averageloss/adv embed dim mismatch: averageloss={avg_shape}, adv={adv_shape}"
                    )
                if avg_shape[0] != adv_shape[0]:
                    print(f"  Averageloss shape: {avg_shape}, adv shape: {adv_shape} - will grid-resample averageloss")
                else:
                    print(f"  Averageloss shape OK: {avg_shape}")

        loss = compute_prm_loss_targeted(
            adv_features,
            clean_features,
            tgt_features,
            averageloss_features=averageloss_features if use_averageloss else None,
            averageloss_lambda=averageloss_lambda,
        )
        loss_history.append(loss.item())

        if perturbation.grad is not None:
            perturbation.grad.zero_()
        loss.backward()

        if iteration == 0:
            if perturbation.grad is not None:
                grad_norm = perturbation.grad.norm().item()
                print(f"  Perturbation grad norm (iter 0): {grad_norm:.6f}")
                if grad_norm == 0:
                    print("  WARNING: Zero gradient! Attack may not work.")
            else:
                print("  WARNING: No gradient on perturbation!")

        with torch.no_grad():
            grad = perturbation.grad
            if gd_sign:
                step = lr * grad.sign()
            else:
                step = lr * grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-12)

            apply_smoothing_this_iter = ((iteration + 1) % smooth_every == 0)

            if gaussian_smooth and apply_smoothing_this_iter:
                step = apply_gaussian_smoothing_step(
                    step,
                    kernel_size=gaussian_kernel,
                    sigma=gaussian_sigma,
                    mask_dilation=gaussian_mask_dilation,
                )

            perturbation.data = perturbation.data - step

            if dct_lowpass and apply_smoothing_this_iter:
                perturbation.data = dct_lowpass_project(
                    perturbation.data,
                    block_size=dct_block_size,
                    keep=dct_keep,
                    mask_shape=dct_mask_shape,
                )

            perturbation.data = perturbation.data.clamp(-epsilon, epsilon)

        pert_linf_raw_history.append(perturbation.data.abs().max().item())

        if iteration == 0 or (iteration + 1) % 50 == 0:
            pert_linf = perturbation.data.clamp(-epsilon, epsilon).abs().max().item()
            current_lr = lr
            with torch.no_grad():
                cos_to_tgt = sum(
                    F.cosine_similarity(
                        F.normalize(a.reshape(-1, a.shape[2]), dim=-1),
                        F.normalize(t.reshape(-1, t.shape[2]), dim=-1),
                    ).mean().item()
                    for a, t in zip(adv_features, tgt_features)
                ) / len(adv_features)
                cos_to_src = sum(
                    F.cosine_similarity(
                        F.normalize(a.reshape(-1, a.shape[2]), dim=-1),
                        F.normalize(c.reshape(-1, c.shape[2]), dim=-1),
                    ).mean().item()
                    for a, c in zip(adv_features, clean_features)
                ) / len(adv_features)

            if use_averageloss:
                with torch.no_grad():
                    cos_to_avg = 0.0
                    for a, p in zip(adv_features, averageloss_features):
                        T_ = a.shape[0]
                        D_ = a.shape[2]
                        p_resized = _resize_averageloss_tokens(p, T_, has_cls=True)
                        a_slice = a.reshape(-1, D_)
                        p_slice = p_resized.reshape(-1, D_)
                        cos_to_avg += F.cosine_similarity(
                            F.normalize(a_slice, dim=-1),
                            F.normalize(p_slice, dim=-1),
                        ).mean().item()
                    cos_to_avg /= len(adv_features)

                print(
                    f"  Iter {iteration+1}/{num_iters}, loss: {loss.item():.4f}, "
                    f"cos->src: {cos_to_src:.4f}, cos->tgt: {cos_to_tgt:.4f}, "
                    f"cos->avg: {cos_to_avg:.4f}, lr: {current_lr:.4f}, "
                    f"pert L_inf: {pert_linf:.2f}"
                )
            else:
                print(
                    f"  Iter {iteration+1}/{num_iters}, loss: {loss.item():.4f}, "
                    f"cos->src: {cos_to_src:.4f}, cos->tgt: {cos_to_tgt:.4f}, "
                    f"lr: {current_lr:.4f}, pert L_inf: {pert_linf:.2f}"
                )

    with torch.no_grad():
        final_adv = (clean_image + perturbation.clamp(-epsilon, epsilon)).clamp(0.0, 255.0)
        final_delta = perturbation.clamp(-epsilon, epsilon)

    loss_min = min(loss_history)
    loss_max = max(loss_history)
    loss_final = loss_history[-1]
    times_loss_increased = sum(1 for a, b in zip(loss_history, loss_history[1:]) if b > a)
    raw_linf_final = pert_linf_raw_history[-1]
    raw_linf_max = max(pert_linf_raw_history)

    print("  -- Attack summary --")
    print(f"    First 3 dyn_scale ratios: [{', '.join(f'{r:.4f}' for r in first_ratios_log)}]")
    print(f"    First 3 encoder-input grids: [{', '.join(f'{g[0]}x{g[1]}' for g in first_grids_log)}]")
    print(
        f"    Loss: min={loss_min:.4f}, max={loss_max:.4f}, final={loss_final:.4f}, "
        f"Delta(final-min)={loss_final - loss_min:+.4f}"
    )
    print(
        f"    Loss went UP in {times_loss_increased}/{num_iters-1} transitions "
        f"({100 * times_loss_increased / (num_iters - 1):.1f}%)"
    )
    print(
        f"    Pert L_inf: final={final_delta.abs().max().item():.2f} (clamped), "
        f"raw={raw_linf_final:.2f}, raw_max={raw_linf_max:.2f}, ratio_raw/eps={raw_linf_final / epsilon:.2f}x"
    )

    return final_adv.squeeze(0), final_delta.squeeze(0)


# ==============================================================================
# Model loading
# ==============================================================================

def load_model(model_name, device):
    is_hf = (
        "/" in model_name
        and not model_name.startswith("ViT-")
        and not model_name.startswith("RN")
    )

    if is_hf:
        print(f"Loading HuggingFace CLIP model: {model_name}")
        from transformers import CLIPModel

        hf_model = CLIPModel.from_pretrained(model_name).to(device)
        hf_model = hf_model.float()
        hf_model.eval()
        hf_model.requires_grad_(False)

        vision_model = hf_model.vision_model

        hf_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1) * 255.0
        hf_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1) * 255.0

        image_size = vision_model.config.image_size
        num_layers = len(vision_model.encoder.layers)
        print(f"  Vision encoder: {num_layers} layers, hidden_size={vision_model.config.hidden_size}")
        print(f"  Native image size: {image_size}, patch size: {vision_model.config.patch_size}")
        print("  pos-embed interpolation enabled -> variable input size supported")

        return {
            "type": "hf",
            "model": vision_model,
            "grab_features": grab_clip_features_hf_vit,
            "mean": hf_mean,
            "std": hf_std,
            "clip_image_size": None,
            "native_size": image_size,
        }

    print(f"Loading OpenAI CLIP model: {model_name}")
    import clip

    clip_model, _ = clip.load(model_name, device=device)
    clip_model = clip_model.float()
    clip_model.eval()
    clip_model.requires_grad_(False)

    openai_mean = torch.tensor([122.7709383, 116.7460125, 104.09373615]).view(1, 3, 1, 1)
    openai_std = torch.tensor([68.5005327, 66.6321579, 70.3231630]).view(1, 3, 1, 1)

    num_layers = len(clip_model.visual.transformer.resblocks)
    native_pos_tokens = clip_model.visual.positional_embedding.shape[0]
    native_grid = round((native_pos_tokens - 1) ** 0.5)
    native_size = native_grid * clip_model.visual.conv1.kernel_size[0]
    patch_size = clip_model.visual.conv1.kernel_size[0]
    hidden_dim = clip_model.visual.ln_pre.normalized_shape[0]

    print(f"  Vision encoder: {num_layers} layers, hidden_size={hidden_dim}")
    print(f"  Native image size: {native_size}, patch size: {patch_size}")

    return {
        "type": "openai",
        "model": clip_model,
        "grab_features": lambda x, m: grab_clip_features_openai_vit(x, m),
        "mean": openai_mean,
        "std": openai_std,
        "clip_image_size": None,
        "native_size": native_size,
    }


# ==============================================================================
# Image I/O
# ==============================================================================

def load_image(path):
    img = Image.open(path).convert("RGB")
    tensor = transforms.ToTensor()(img) * 255.0
    return tensor


def save_image(tensor, path):
    img = transforms.ToPILImage()(tensor.byte().cpu())
    save_path = os.path.splitext(path)[0] + ".png"
    img.save(save_path, format="PNG")
    return save_path


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PRM Attack - targeted, with optional averageloss + Gaussian/DCT smoothing"
    )

    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--target_path", type=str, default=None)
    parser.add_argument("--target_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./adv_images")

    parser.add_argument("--model", type=str, default="ViT-B/16")
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--num_iters", type=int, default=250)
    parser.add_argument("--lr", type=float, default=0.5)

    parser.add_argument("--no_dynamic_scale", action="store_true")
    parser.add_argument("--resize_to_clip", action="store_true")
    parser.add_argument("--resize_size", type=int, default=None)

    parser.add_argument("--use_averageloss", action="store_true")
    parser.add_argument("--averageloss_lambda", type=float, default=0.1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict_determinism", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # Update rule
    parser.add_argument("--gd_sign", action="store_true",
                        help="Use signed gradient descent (FaceShield-style). Default: mean-normalized GD.")

    # Gaussian smoothing of the step
    parser.add_argument("--gaussian_smooth", action="store_true",
                        help="Apply FaceShield-style masked Gaussian smoothing to the gradient step, not the perturbation.")
    parser.add_argument("--gaussian_kernel", type=int, default=9,
                        help="Gaussian kernel size for masked step smoothing. FaceShield uses 9.")
    parser.add_argument("--gaussian_sigma", type=float, default=5.0,
                        help="Gaussian sigma for masked step smoothing. FaceShield uses 5.")
    parser.add_argument("--gaussian_mask_dilation", type=int, default=9,
                        help="Dilation kernel size for the FaceShield line mask. FaceShield uses 9.")

    # DCT low-pass projection of the perturbation
    parser.add_argument("--dct_lowpass", action="store_true",
                        help="Apply 8x8 block-DCT low-pass projection to perturbation each smoothing iter.")
    parser.add_argument("--dct_block_size", type=int, default=8)
    parser.add_argument("--dct_keep", type=int, default=5,
                        help="Triangular: keep where i+j<=keep. Square: keep top-left [keep,keep] block.")
    parser.add_argument("--dct_mask_shape", type=str, default="triangular",
                        choices=["triangular", "square"],
                        help="DCT low-pass mask shape. Triangular ~= JPEG zigzag cutoff (default).")

    # How often the smoothing fires
    parser.add_argument("--smooth_every", type=int, default=1,
                        help="Apply Gaussian + DCT smoothing every K iterations (default 1).")

    args = parser.parse_args()

    if args.target_path is None and args.target_dir is None:
        raise ValueError("A target is required. Provide --target_path or --target_dir.")

    print("=" * 60)
    mode_label = "TARGETED"
    if args.use_averageloss:
        mode_label += f" + AVERAGELOSS (lambda={args.averageloss_lambda})"
    print(f"PRM Attack - {mode_label}")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Epsilon: {args.epsilon} ({args.epsilon / 255:.4f} in [0,1] scale)")
    print(f"Iterations: {args.num_iters}")
    print(f"Dynamic scale: {not args.no_dynamic_scale}")
    print(f"Resize to CLIP: {args.resize_to_clip}")
    if args.resize_size:
        print(f"Resize override: {args.resize_size}x{args.resize_size}")

    set_global_seeds(args.seed, strict_determinism=args.strict_determinism)
    print()

    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}

    if args.input_path:
        source_paths = [args.input_path]
    elif args.input_dir:
        source_paths = sorted(
            [
                os.path.join(args.input_dir, f)
                for f in os.listdir(args.input_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
        )
    else:
        raise ValueError("Provide either --input_path or --input_dir")

    if args.target_path:
        target_paths = [args.target_path]
    else:
        target_paths = sorted(
            [
                os.path.join(args.target_dir, f)
                for f in os.listdir(args.target_dir)
                if os.path.splitext(f)[1].lower() in valid_exts
            ]
        )

    if len(source_paths) == 0:
        raise ValueError("No source images found")
    if len(target_paths) == 0:
        raise ValueError("No target images found")
    if args.use_averageloss and len(source_paths) < 2:
        raise ValueError("--use_averageloss requires at least 2 source images in --input_dir")

    print(f"Source images: {len(source_paths)}")
    print(f"Target images: {len(target_paths)} (will cycle if fewer than sources)")
    os.makedirs(args.output_dir, exist_ok=True)

    model_components = load_model(args.model, args.device)

    clip_resize = None
    if args.resize_size is not None:
        clip_resize = args.resize_size
        print(f"  Resize override: {clip_resize}x{clip_resize}")
    elif args.resize_to_clip:
        clip_resize = model_components.get("native_size", None)
        if clip_resize is None:
            clip_resize = 224
            print(f"  WARNING: No native_size found, defaulting to {clip_resize}")
        print(f"  Resizing all images to {clip_resize}x{clip_resize} before attack")
    print()

    per_image_features = None
    feature_sum = None
    if args.use_averageloss:
        print("=" * 60)
        print("Pre-computing averageloss features (leave-one-out means)")
        print("=" * 60)
        per_image_features, feature_sum = compute_averageloss_features(
            source_paths,
            model_components,
            args.device,
            clip_resize=clip_resize,
        )
        print()

    for i, src_path in enumerate(source_paths):
        fname = os.path.basename(src_path)
        print(f"[{i+1}/{len(source_paths)}] Attacking: {fname}")

        per_image_seed = args.seed + i
        seed_for_image(args.seed, i)
        print(f"  RNG seed for this image: {per_image_seed}  (= args.seed {args.seed} + image_idx {i})")

        image_tensor = load_image(src_path)
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
        target_tensor = load_image(tgt_path)
        if clip_resize is not None:
            target_tensor = F.interpolate(
                target_tensor.unsqueeze(0),
                (clip_resize, clip_resize),
                mode="bicubic",
                align_corners=False,
            ).squeeze(0).clamp(0, 255)
        print(f"  Target: {os.path.basename(tgt_path)} ({list(target_tensor.shape)})")

        averageloss_for_this_image = None
        if args.use_averageloss:
            averageloss_for_this_image = get_leave_one_out_averageloss(
                per_image_features,
                feature_sum,
                exclude_idx=i,
            )
            print(f"  Using averageloss: mean of {len(source_paths) - 1} other images (lambda={args.averageloss_lambda})")

        adv_image, delta = prm_attack(
            image_tensor,
            model_components,
            target_tensor=target_tensor,
            averageloss_features=averageloss_for_this_image,
            averageloss_lambda=args.averageloss_lambda,
            epsilon=args.epsilon,
            num_iters=args.num_iters,
            lr=args.lr,
            dynamic_scale=not args.no_dynamic_scale,
            device=args.device,
            gd_sign=args.gd_sign,
            gaussian_smooth=args.gaussian_smooth,
            gaussian_kernel=args.gaussian_kernel,
            gaussian_sigma=args.gaussian_sigma,
            gaussian_mask_dilation=args.gaussian_mask_dilation,
            dct_lowpass=args.dct_lowpass,
            dct_block_size=args.dct_block_size,
            dct_keep=args.dct_keep,
            dct_mask_shape=args.dct_mask_shape,
            smooth_every=args.smooth_every,
        )

        out_path = os.path.join(args.output_dir, fname)
        saved_path = save_image(adv_image, out_path)

        print(f"  Saved: {saved_path}")
        print(f"  Output image size: {list(adv_image.shape)}, range: [{adv_image.min():.1f}, {adv_image.max():.1f}]")
        print(f"  Delta L_inf: {delta.abs().max().item():.2f}, L2: {delta.norm().item():.2f}")
        print()

    print("=" * 60)
    print("Done! Adversarial images saved to:", args.output_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()