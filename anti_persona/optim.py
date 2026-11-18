"""The optimisation loop: signed gradient descent with JPEG in the loop, gradient smoothing and bounded rounding.

Per step: x' = clip(x + clamp(delta, -eps, eps)); with probability 0.7 x' goes through a real JPEG (quality drawn
uniformly from 60..95, straight-through gradient); L is evaluated; every third step the gradient is smoothed with a
3 x 3 Gaussian (sigma 3); delta <- clamp(delta - 0.5 * sign(grad), -4, 4). The result is rounded to whole numbers
that stay within 4 of both the clean and the rounded clean image.
"""
import io
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import EPS, JPEG_CLEAN_P, JPEG_Q, SMOOTH_EVERY, SMOOTH_SIGMA, STEP, STEPS
from .losses import loss_terms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if torch.version.hip:
        torch.backends.cudnn.enabled = False


def jpeg_view(image, quality):
    """A real JPEG (4:4:4) of the rounded image in the forward pass; the gradient passes straight through."""
    rgb = image[0].detach().round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    decoded = torch.from_numpy(np.array(Image.open(buffer).convert("RGB"))).permute(2, 0, 1)[None].to(image)
    return image + (decoded - image).detach()


def smooth_gradient(gradient):
    """3 x 3 Gaussian with sigma 3 (exp(-(x^2 + y^2) / (2 sigma^2)), normalised), per channel, zero padding."""
    axis = torch.arange(3, device=gradient.device, dtype=gradient.dtype) - 1
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * SMOOTH_SIGMA**2))
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return F.conv2d(gradient, kernel, padding=1, groups=3)


def quantize_bounded(protected, clean, epsilon=EPS):
    """Whole-number pixels that stay within epsilon of the clean image AND of the rounded clean image."""
    rounded = clean.round().clamp(0, 255)
    lower = torch.maximum((clean - epsilon).ceil(), (rounded - epsilon).ceil()).clamp(0, 255)
    upper = torch.minimum((clean + epsilon).floor(), (rounded + epsilon).floor()).clamp(0, 255)
    return torch.maximum(torch.minimum(protected.round(), upper), lower).byte()


def protect(clean, encoder, prototype, bank, face, seed, steps=STEPS, callback=None):
    """Protect one photo. ``seed`` = run seed + the photo's position in its album (it only drives the JPEG draws).
    ``callback(step, delta)``, if given, is called after every update (e.g. to save snapshots).
    Returns (the protected image as uint8 [1, 3, 336, 336], one dict of logged values per step)."""
    seed_everything(seed)
    delta = torch.zeros_like(clean, requires_grad=True)                       # zero start
    with torch.no_grad():
        clean_features = encoder(clean)
    trace = []
    for step in range(steps):
        image = (clean + delta.clamp(-EPS, EPS)).clamp(0, 255)
        quality = random.randint(*JPEG_Q) if random.random() >= JPEG_CLEAN_P else None      # JPEG with probability 0.7
        view = jpeg_view(image, quality) if quality is not None else image
        loss, terms = loss_terms(view, encoder, clean_features, prototype, bank, face)
        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()
        with torch.no_grad():
            gradient = smooth_gradient(delta.grad) if (step + 1) % SMOOTH_EVERY == 0 else delta.grad
            delta.copy_((delta - STEP * gradient.sign()).clamp(-EPS, EPS))    # descend: the loss is minimised
        trace.append(dict(step=step + 1, jpeg_quality=quality, total_loss=float(loss.detach()),
                          **{k: float(v.detach()) if torch.is_tensor(v) else v for k, v in terms.items()},
                          mean_abs_delta=float(delta.detach().abs().mean()), max_abs_delta=float(delta.detach().abs().max())))
        if callback is not None:
            callback(step + 1, delta.detach())
    protected = (clean + delta.detach().clamp(-EPS, EPS)).clamp(0, 255)
    return quantize_bounded(protected, clean), trace
