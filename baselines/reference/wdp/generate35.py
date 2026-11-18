# REFERENCE COPY (anti-persona release): the generator that produced the paper's WDP + face images, kept for its exact
# procedure and settings. Only absolute paths were replaced by environment variables; the logic is unchanged.
# It runs inside the research harness it was written for (a queue PLAN.json, the legacy baseline module and the
# stock wrapper); see baselines/README.md, section "DiffAttack and WDP".
# Instance face term: the lines marked PBFACE add the baseline face term of ../pbface.py to the self-term generator.
#!/usr/bin/env python
"""WDP re-run on OUR loss plus the instance face term for ONE photo, at the PAPER's 35 adversarial latent iterations --
eps- or PSNR-BOUNDED. HERE = this code folder; R = the legacy baselines folder (LEGACY_BASELINES_DIR: code_snapshot,
baselines.py, generate.py).

Image seed: the per-photo image seed = PLAN run_seed (0 or 202) + the photo's album_index (Ours' rule) REPLACES the
authors' two hard-coded seeds, both recorded in the manifest (seed_override):
  1. main.initialize_seed(seed=10) -> initialize_seed(seed=image_seed) (the stock function, called by this generator as
     before, only the argument changes);
  2. Adversarial_Opt.image2latent's VAE posterior generator torch.Generator().manual_seed(8888) -> manual_seed(image_seed):
     the stock method compiled by AST from the pinned stock file with only that constant swapped (adversarial_optimization
     module globals), bound on the ONE opt instance before the loss swap is installed (the class is never edited). This
     sample is the only random draw of the protection, so without (2) the two run seeds would give identical images.

PSNR bound: --psnr P (exclusive with --eps): the in-loop projection of the full pasted-back 336 photo onto the PSNR-P
L2 ball around the rounded clean photo (L2Ball), the saved PNG (stock paste-back) projected + rounded to PSNR >= P over
the whole photo (asserted); outside the face quad and no-face photos stay clean. The paper uses P = 30.

L-inf bound: ONE opt-in flag, --eps E (integer pixel levels, E/255). Without --eps every statement that runs is the unbounded generator's (same outputs; the
manifest/CSV carry no extra field). With --eps E the bound is applied INSIDE the optimisation: at every protection
step the full 336 photo our loss sees (rounded clean outside the face quad, differentiable paste of the decoded face
inside, clamp 0..255) is projected onto the L-inf ball of radius E around the rounded clean photo c:
    x_proj = (c + (x - c).clamp(-E, E)).clamp(0, 255)
and our loss is computed on x_proj (plain autograd: gradient 0 on clipped pixels, no straight-through). The delivered
PNG gets the same projection: the stock 256 face PNG -> stock paste_back (unchanged) -> clip to [c - E, c + E] in
integers; max |x_adv - c| <= E is asserted on the re-read PNG. Pixels outside the face quad and no-face photos stay
the rounded clean (as unbounded WDP). Logged: per-step self term, fraction of raw values outside the ball, fraction
at the bound (whole photo and face region), final loss, PSNR, and the loss of the raw (unprojected) pasted photo.

Protection steps: WDP uses the paper's 35 protection steps, not the code default prot_steps=30. The only change for this
is opt.prot_steps = 35 on the Adversarial_Opt instance after construction, recorded in the PLAN method record and every manifest rule record
(prot_steps_run 35, prot_steps_reason). Attack weight 0.003 and every other default unchanged (no tuning).

WDP = Salar et al., "Enhancing Facial Privacy Protection via Weakening Diffusion Purification", CVPR 2025, official code
@ bd0a8e4, imported UNEDITED from the stock clone and run once at the repo's own main.parse_args() defaults (no tuning).
The I/O is the stock wrapper's (gen/wdp/gen_wdp.py under the legacy baselines' STOCK folder; reference/wdp/gen_wdp.py
here; imported as a module, its helpers called as they are): dlib + FFHQ alignment of the photo to a 256 face, the stock Adversarial_Opt.run() on it, and
the wrapper's paste_back (Lanczos-4 inverse of the alignment, hard quad mask) of the protected face into the photo.

The ONLY change to the method is its face-recognition (impersonation) loss. Stock WDP minimises, over the DDIM latent,
    adv_optim_weight * sum_k (1 - cos(FR_k(MTCNN crop of the decoded face), FR_k(target)))  +  self-attention MSE.
Here the FR term is replaced by Ours' CLIP self term (loss preset 'self': anti_persona.attack.clip_objective with
weights self=1, every other weight 0, zero placeholder prototype, through baselines.SelfObjective) on Yo'LLaVA's own
openai/clip-vit-large-patch14-336 (attack.ClipFeatures, hard-checked), computed on the FULL 336x336 photo after the
decoded face is pasted back, because that photo is what Yo'LLaVA sees. The paste is PasteBack below: the wrapper's
paste_back made differentiable (same inverse maps, same mask, cv2's own Lanczos-4 sub-pixel table and taps, border
replicate), so the gradient of the self term reaches WDP's optimisation variable (the DDIM latent) through the VAE
decoder and the three DDIM steps exactly as the FR gradient did. The clean reference is ClipFeatures(attack.read_image(
<original photo>)). adv_optim_weight (0.003), the self-attention term, AdamW, the step counts and every other default
stay as published. Unbounded as published: the integer L-inf and PSNR vs the rounded clean are recorded.

How the loss is swapped without editing WDP: on the one Adversarial_Opt instance (never the class), get_FR_embeddings
returns the decoded 256 face itself, cosine_loss returns the self term of the pasted-back photo (WDP then multiplies it
by adv_optim_weight and adds its self-attention loss, in its own attacker()), and MTCNN_cropping is False because that
flag only crops the FR-loss input (and the FR target image); the MTCNN detection itself still runs, as the stock
wrapper's no-face gate. The stock attacker()/run() code executes as is.

WDP runs on the rounded clean photo round(attack.read_image(source, 'cpu')) (336x336 uint8; pixel-identical to the input
the stock WDP run used and to the evaluators' clean reference, R/cpu_test test 9; rounded on the CPU because GPU bicubic
can round a pixel differently). No dlib face, or no MTCNN face on the aligned crop: the output is that clean PNG
unchanged, recorded protected=false (the stock rule). L-inf and PSNR are measured against it.

  python -B generate35.py --root <queue root> --setting <s> --album <a> --photo <p>

Writes <root>/chunks/<setting>/<album>/<photo>/{GROUP.json, wdp/{<output_name>, manifest.json[, <photo>_loss.csv]}} via a
private, uniquely named <root>/work/<setting>/<album>/<photo>/<photo>.partial.<random> folder renamed into place (same file
system); refuses if the final folder exists. Stock work files (clean PNG, aligned face, stock output face, stock stdout)
go to <root>/work/<setting>/<album>/<photo>/attempt.<random>/. The photo entry, its source hash, the encoder record and
the method record come from <root>/PLAN.json.
"""
import sys

sys.dont_write_bytecode = True

import argparse              # noqa: E402
import ast                   # noqa: E402
import types                 # noqa: E402
import contextlib            # noqa: E402
import csv                   # noqa: E402
import functools             # noqa: E402
import importlib             # noqa: E402
import importlib.metadata    # noqa: E402
import importlib.util        # noqa: E402
import json                  # noqa: E402
import math                  # noqa: E402
import os                    # noqa: E402
from pathlib import Path     # noqa: E402
import platform              # noqa: E402
import re                    # noqa: E402
import shutil                # noqa: E402
import socket                # noqa: E402
import subprocess            # noqa: E402
import tempfile              # noqa: E402
import time                  # noqa: E402

HERE = Path(__file__).resolve().parent                    # this code folder
R = Path(os.environ['LEGACY_BASELINES_DIR'])   # research harness: legacy baselines.py + code_snapshot/anti_persona
sys.path.insert(0, str(R / 'code_snapshot'))
sys.path.insert(0, str(R))

import numpy as np           # noqa: E402
import torch                 # noqa: E402
from PIL import Image        # noqa: E402

from anti_persona import attack   # noqa: E402
import baselines                  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # PBFACE: pbface.py is in the parent folder
import pbface as PB               # noqa: E402  PBFACE

STOCK_WRAPPER = baselines.STOCK / 'gen/wdp/gen_wdp.py'
ARM, METHOD, ARM_DIR = 'wdp_ourloss', 'wdp', 'wdp'
PROT_STEPS, PROT_STEPS_REASON, PROT_STEPS_CODE_DEFAULT = 35, 'paper', 30
EXCLUDED = ('-193', '-212', '-214', '-207', '-218')
# Instance attributes the loss swap changes; everything else on the Adversarial_Opt instance is the stock object's.
SWAPPED = ['MTCNN_cropping', 'cosine_loss', 'get_FR_embeddings']
TARGET_NOT_USED = ('stock impersonation-target embeddings: not part of our loss',)


def load_file(name, path):
    """Import a file under its own module name (our folder holds a generate.py/build.py of its own)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RGEN = load_file('ourloss_generate', R / 'generate.py')    # photo_entry, stat_binding, put
gen_wdp = load_file('gen_wdp', STOCK_WRAPPER)               # the stock wrapper: constants + I/O helpers, unmodified
assert Path(attack.__file__).resolve() == R / 'code_snapshot/anti_persona/attack.py', attack.__file__
assert Path(baselines.__file__).resolve() == R / 'baselines.py', baselines.__file__
sha256_file, ffhq_quad, stock_frame, paste_back, psnr = (gen_wdp.sha256_file, gen_wdp.ffhq_quad, gen_wdp.stock_frame,
                                                         gen_wdp.paste_back, gen_wdp.psnr)
REPO, IMG_SIZE, STOCK_SEED, RUNTIME = gen_wdp.REPO, gen_wdp.IMG_SIZE, gen_wdp.STOCK_SEED, gen_wdp.RUNTIME


RUN_SEEDS = (0, 202)                      # the paper's two run seeds
STOCK_VAE_SEED = 8888


def image_seed(run_seed, album_index):
    """Ours' per-photo rule: run seed + the photo's index in the person's album."""
    assert run_seed in RUN_SEEDS and album_index >= 0
    return run_seed + album_index


def seeded_image2latent(module, seed):
    """Adversarial_Opt.image2latent compiled from the stock file with ONLY manual_seed(8888) -> manual_seed(seed)."""
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(), filename=str(path))
    (cls,) = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Adversarial_Opt']
    (fn,) = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'image2latent']
    assert 'torch.Generator().manual_seed(8888)' in ast.unparse(fn), ast.unparse(fn)
    hits = [n for n in ast.walk(fn) if isinstance(n, ast.Constant) and type(n.value) is int and n.value == STOCK_VAE_SEED]
    assert len(hits) == 1, len(hits)
    hits[0].value = seed
    scratch = dict(vars(module))          # a copy of the stock module's globals (torch)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scratch)
    return scratch['image2latent']


# ------------------------------------------------------------------------------------------- differentiable paste-back
INTER_BITS, INTER_TAB_SIZE = 5, 32          # cv2 imgproc: remap's sub-pixel phases per axis


def inverse_maps(quad, scale, H, W, n):
    """The geometry of the stock wrapper's paste_back, statement for statement (gen_wdp.py paste_back, first half):
    photo pixel -> (s, t) in the aligned square -> cv2.remap source maps, and the hard quad-footprint mask."""
    nw, sw, se, ne = quad
    A = np.stack([ne - nw, sw - nw], axis=1)  # P' - nw = A @ [s, t]
    Ainv = np.linalg.inv(A)
    xx, yy = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xs = (xx + 0.5) * scale[0] - 0.5
    ys = (yy + 0.5) * scale[1] - 0.5
    P = np.stack([xs - nw[0], ys - nw[1]], axis=-1)
    st = P @ Ainv.T
    s, t = st[..., 0], st[..., 1]
    inside = (s >= 0) & (s <= 1) & (t >= 0) & (t <= 1)
    mapx = (s * n - 0.5).astype(np.float32)
    mapy = (t * n - 0.5).astype(np.float32)
    return mapx, mapy, inside


def lanczos4_table():
    """cv2 initInterTab1D(INTER_LANCZOS4, tab, INTER_TAB_SIZE) -> interpolateLanczos4(i / 32), same float/double steps:
    8 taps at offsets -3..+4 for each of the 32 sub-pixel phases, normalised to sum 1."""
    s45 = 0.70710678118654752440084436210485
    cs = [(1, 0), (-s45, -s45), (0, 1), (s45, -s45), (-1, 0), (s45, s45), (0, -1), (-s45, s45)]
    scale = np.float32(1.) / np.float32(INTER_TAB_SIZE)
    tab = np.zeros((INTER_TAB_SIZE, 8), np.float32)
    for i in range(INTER_TAB_SIZE):
        x3 = np.float32(np.float32(np.float32(i) * scale) + np.float32(3))
        y0 = -float(x3) * math.pi * 0.25
        s0, c0 = math.sin(y0), math.cos(y0)
        coeffs, total = np.zeros(8, np.float32), np.float32(0)
        for k in range(8):
            y0_ = np.float32(x3 - np.float32(k))
            if abs(float(y0_)) >= 1e-6:
                y = -float(y0_) * math.pi * 0.25
                coeffs[k] = np.float32((cs[k][0] * s0 + cs[k][1] * c0) / (y * y))
            else:
                coeffs[k] = np.float32(1e30)
            total = np.float32(total + coeffs[k])
        tab[i] = coeffs * np.float32(np.float32(1) / total)
    return tab


class PasteBack(torch.nn.Module):
    """The stock paste_back as a differentiable map: 256 face [1, 3, n, n] (pixel units) -> photo [1, 3, H, W].
    Outside the quad footprint: the rounded clean photo (constant). Inside: cv2.remap(face, mapx, mapy, INTER_LANCZOS4,
    BORDER_REPLICATE) evaluated as a fixed linear combination of 8x8 face pixels per photo pixel, with cv2's own
    coordinate rounding (cvRound(map * 32): integer tap origin + 5-bit phase) and its 2-D weight table
    (float32 products of the 1-D table). clamp(0, 255) = the uint8 saturation of the delivered PNG."""

    def __init__(self, clean, quad, scale, n=256, device='cpu', dtype=torch.float32):
        super().__init__()
        H, W = clean.shape[:2]
        mapx, mapy, inside = inverse_maps(quad, scale, H, W, n)
        sel = np.flatnonzero(inside.ravel())
        sx = np.rint(mapx.ravel()[sel] * np.float32(INTER_TAB_SIZE)).astype(np.int64)
        sy = np.rint(mapy.ravel()[sel] * np.float32(INTER_TAB_SIZE)).astype(np.int64)
        taps = np.arange(8) - 3
        xs = np.clip((sx >> INTER_BITS)[:, None] + taps, 0, n - 1)       # BORDER_REPLICATE
        ys = np.clip((sy >> INTER_BITS)[:, None] + taps, 0, n - 1)
        tab = lanczos4_table()
        weight = tab[sy & (INTER_TAB_SIZE - 1)][:, :, None] * tab[sx & (INTER_TAB_SIZE - 1)][:, None, :]
        self.register_buffer('index', torch.from_numpy((ys[:, :, None] * n + xs[:, None, :]).reshape(-1, 64)).to(device))
        self.register_buffer('weight', torch.from_numpy(weight.reshape(-1, 64).astype(np.float32)).to(device, dtype))
        self.register_buffer('sel', torch.from_numpy(sel).to(device))
        self.register_buffer('base', torch.from_numpy(np.ascontiguousarray(clean.transpose(2, 0, 1)))
                             .reshape(3, H * W).to(device, dtype))
        self.mask, self.maps, self.n, self.size = inside, (mapx, mapy), n, (H, W)

    def warp(self, face):
        """[1, 3, n, n] -> [3, N]: the Lanczos-4 samples at the N photo pixels inside the footprint."""
        assert tuple(face.shape) == (1, 3, self.n, self.n), tuple(face.shape)
        return (face.reshape(3, self.n * self.n)[:, self.index] * self.weight).sum(-1)

    def forward(self, face, clamp=True):
        full = self.base.index_copy(1, self.sel, self.warp(face)).reshape(1, 3, *self.size)
        return full.clamp(0, 255) if clamp else full


def psnr_value(clean, img):
    """The paper tables' PSNR of an image vs the clean PNG: 10 log10(255^2 / MSE) over ALL values (float64); None when
    identical (infinite PSNR)."""
    mse = float(((np.asarray(img).astype(np.float64) - np.asarray(clean).astype(np.float64)) ** 2).mean())
    return None if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


class L2Ball:
    """PSNR ball: {x : PSNR(x, c) >= P} = the L2 ball of radius r = sqrt(N * 255^2 / 10^(P/10)) around c (pixel units),
    N = number of values. project(x): d = x - c; d *= min(1, r / ||d||_2); (c + d).clamp(0, 255) -- plain autograd."""
    STAT_KEYS = ('raw_psnr', 'proj_psnr', 'scale', 'raw_l2')

    def __init__(self, clean_hwc_uint8, psnr_db, device):
        assert 20.0 <= float(psnr_db) <= 60.0, psnr_db
        self.psnr_db = float(psnr_db)
        self.center_np = np.asarray(clean_hwc_uint8, dtype=np.uint8)
        self.center = torch.from_numpy(self.center_np.copy()).permute(2, 0, 1)[None].float().to(device)
        self.n = int(self.center.numel())
        assert self.n == self.center_np.size, (self.n, self.center_np.size)
        self.radius = float(np.sqrt(self.n * 255.0 ** 2 / 10.0 ** (self.psnr_db / 10.0)))

    def __call__(self, x):
        assert tuple(x.shape) == tuple(self.center.shape), (tuple(x.shape), tuple(self.center.shape))
        d = x - self.center
        scale = torch.clamp(self.radius / (d.flatten().norm() + 1e-12), max=1.0)
        return (self.center + d * scale).clamp(0, 255)

    def stats(self, raw, proj):
        """Diagnostics (no grad): PSNR of the raw view and of its projection, the scale min(1, r/||d||), raw L2."""
        with torch.no_grad():
            dr = raw.detach() - self.center
            dp = proj.detach() - self.center
            norm = float(dr.flatten().norm())
            to_psnr = lambda m: None if m == 0 else float(10 * np.log10(255.0 ** 2 / m))   # noqa: E731
            return dict(raw_psnr=to_psnr(float((dr ** 2).mean())), proj_psnr=to_psnr(float((dp ** 2).mean())),
                        scale=1.0 if norm <= self.radius else self.radius / norm, raw_l2=norm)

    def project_uint8(self, img, steps=60):
        """The same projection on the saved image, then integers. d = img - c; s0 = min(1, r/||d||). The scale s in
        [s0, 1] is the largest whose CLAMPED offset t = clip(s d, -c, 255 - c) still has ||t||^2 <= r^2 (= s0 unless the
        clamp to [0, 255] removed part of the budget; bisection, never beyond the raw output). Budget-filling rounding of
        t: every value goes to one of its two nearest integers -- towards zero (|t| floored) first, then away from zero in
        order of decreasing fractional part while the sum of squares stays <= r^2 (1e-9 relative margin) -- so the saved
        PNG has PSNR >= P and as close to P as integer steps allow (plain rint + shrink overshot by up to ~0.15 dB because
        rounding moves whole groups of values at the same scale). Returns (x, record)."""
        c = self.center_np.astype(np.float64)
        d = np.asarray(img).astype(np.float64) - c
        norm = float(np.sqrt((d ** 2).sum()))
        budget = self.radius ** 2 * (1 - 1e-9)
        if norm ** 2 <= budget:                          # raw output already inside the ball: unchanged (integers)
            x = np.asarray(img).astype(np.uint8)
            p = psnr_value(self.center_np, x)
            return x, dict(scale_ball=1.0, scale_final=1.0, bisection_steps=0, upgraded_frac=0.0, raw_l2=norm,
                           radius=self.radius, n_values=self.n, psnr=p)
        s0 = self.radius / norm
        sse = lambda s: float((np.clip(s * d, -c, 255 - c) ** 2).sum())                 # noqa: E731
        lo, hi, n = s0, 1.0, 0
        if sse(lo) > budget:
            lo = s0 * (1 - 1e-9)
        while hi - lo > 1e-12 and n < steps:
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sse(mid) <= budget else (lo, mid)
            n += 1
        t = np.clip(lo * d, -c, 255 - c)
        mag, sign = np.abs(t), np.sign(t)
        base = np.floor(mag)
        frac = mag - base
        used = float((base ** 2).sum())
        cand = np.flatnonzero(frac.ravel() > 0)
        order = cand[np.argsort(-frac.ravel()[cand], kind='stable')]
        cost = 2 * base.ravel()[order] + 1
        k = int(np.searchsorted(np.cumsum(cost), budget - used, side='right'))
        up = np.zeros(mag.size, bool)
        up[order[:k]] = True
        x = (c + sign * (base + up.reshape(mag.shape))).astype(np.int64)
        assert x.min() >= 0 and x.max() <= 255, (x.min(), x.max())
        x = x.astype(np.uint8)
        p = psnr_value(self.center_np, x)
        assert p is None or p >= self.psnr_db, ('PSNR below the budget after rounding', p, self.psnr_db)
        return x, dict(scale_ball=s0, scale_final=lo, bisection_steps=n, upgraded_frac=float(k / mag.size), raw_l2=norm,
                       radius=self.radius, n_values=self.n, psnr=p)


class LinfBall:
    """Projection onto {x : |x - c|_inf <= eps} intersected with [0, 255], c = the rounded clean 336 photo (pixel units).
    project(x) = (c + (x - c).clamp(-eps, eps)).clamp(0, 255): plain autograd (gradient 0 where clipped)."""

    STAT_KEYS = ('frac_outside', 'frac_at_bound', 'frac_outside_face', 'frac_at_bound_face', 'raw_linf')

    def __init__(self, clean_hwc_uint8, eps, device, region=None):
        assert isinstance(eps, int) and 1 <= eps <= 255, eps
        self.eps = eps
        self.center_np = np.asarray(clean_hwc_uint8, dtype=np.uint8)
        self.center = torch.from_numpy(self.center_np.copy()).permute(2, 0, 1)[None].float().to(device)
        self.region = None if region is None else torch.from_numpy(np.asarray(region, bool)).to(device)

    def __call__(self, x):
        assert tuple(x.shape) == tuple(self.center.shape), (tuple(x.shape), tuple(self.center.shape))
        return (self.center + (x - self.center).clamp(-self.eps, self.eps)).clamp(0, 255)

    def stats(self, raw, proj):
        """Diagnostics (no grad): fraction of raw values outside the ball (gradient 0 there), fraction of projected values
        at the bound (|x - c| >= eps - 1e-3), both over the whole photo and over the face quad; raw L-inf."""
        with torch.no_grad():
            out_raw = (raw.detach() - self.center).abs() > self.eps
            at = (proj.detach() - self.center).abs() >= self.eps - 1e-3
            row = dict(frac_outside=float(out_raw.float().mean()), frac_at_bound=float(at.float().mean()),
                       raw_linf=float((raw.detach() - self.center).abs().max()))
            if self.region is not None and bool(self.region.any()):
                m = self.region[None, None].expand_as(at)
                row.update(frac_outside_face=float(out_raw[m].float().mean()), frac_at_bound_face=float(at[m].float().mean()))
            return row

    def project_uint8(self, img):
        """The same projection on an integer HxWx3 image (c, eps integers: exact, no rounding)."""
        c = self.center_np.astype(np.int16)
        return np.clip(np.clip(np.asarray(img).astype(np.int16), c - self.eps, c + self.eps), 0, 255).astype(np.uint8)


def bound_metrics(clean, adv, eps, region=None):
    """At-bound fractions of a delivered uint8 image vs the rounded clean (whole photo, and the face quad if given)."""
    d = np.abs(adv.astype(np.int16) - clean.astype(np.int16))
    out = dict(linf=int(d.max()), frac_values_at_bound=float((d == eps).mean()),
               frac_pixels_any_channel_at_bound=float((d == eps).any(-1).mean()), frac_values_changed=float((d > 0).mean()))
    if region is not None and region.any():
        out.update(face_frac_values_at_bound=float((d[region] == eps).mean()), face_frac_values_changed=float((d[region] > 0).mean()))
    return out


class SelfTermLoss:
    """Takes the place of WDP's FR impersonation loss on ONE Adversarial_Opt instance (see the module docstring)."""

    def __init__(self, objective, paste, ball=None):
        self.objective, self.paste, self.ball = objective, paste, ball
        self.trace, self.target_calls = [], 0

    def install(self, opt):
        stock = dict(MTCNN_cropping=opt.MTCNN_cropping, is_obfuscation=opt.is_obfuscation, is_makeup=opt.is_makeup,
                     comparison_null_text=opt.comparison_null_text)
        assert stock == dict(MTCNN_cropping=True, is_obfuscation=False, is_makeup=False, comparison_null_text=False), stock
        before = dict(vars(opt))
        opt.MTCNN_cropping = False          # FR-loss input crop only; our loss reads the whole decoded face
        opt.get_FR_embeddings = self.embeddings
        opt.cosine_loss = self.loss
        changed = sorted(k for k, v in vars(opt).items() if k not in before or before[k] is not v)
        assert changed == SWAPPED, changed
        return changed

    def embeddings(self, image):
        """Stock call sites: run() -> target_embeddings (under no_grad, the 1024 target photo); attacker() -> the decoded
        256 face with grad. The first becomes a marker our loss refuses to use, the second passes through untouched."""
        if not torch.is_grad_enabled() and tuple(image.shape[-2:]) != (256, 256):
            self.target_calls += 1
            return TARGET_NOT_USED
        if not torch.is_grad_enabled() or tuple(image.shape) != (1, 3, 256, 256):
            raise RuntimeError(f'unexpected FR-embedding call: shape {tuple(image.shape)}, grad {torch.is_grad_enabled()}')
        return [image]

    def loss(self, protected_feature, target_feature, source_feature):
        """Stock call: adv_loss = self.cosine_loss(output_embeddings, target_embeddings, source_embeddings) * 0.003."""
        if target_feature is not TARGET_NOT_USED or source_feature is not None or len(protected_feature) != 1:
            raise RuntimeError('the loss swap saw an unexpected stock call')
        face = protected_feature[0]
        face255 = (face / 2 + 0.5).clamp(0, 1) * 255          # the stock latent2image mapping, before its uint8 cast
        photo = self.paste(face255)
        assert tuple(photo.shape) == (1, 3, IMG_SIZE, IMG_SIZE), tuple(photo.shape)
        extra = {}
        if self.ball is not None:          # --eps: our loss sees the projection onto the L-inf ball
            raw = photo
            photo = self.ball(raw)
            extra = self.ball.stats(raw, photo)
        value = self.objective(photo)
        row = dict(step=len(self.trace) + 1, self_term=float(value.detach()), face_grad_abs_mean=None, **extra)
        self.trace.append(row)
        if face.requires_grad:
            face.register_hook(lambda g, row=row: row.__setitem__('face_grad_abs_mean', float(g.abs().mean())))
        return value


def rounded_clean(src):
    """round(attack.read_image(src)) computed on the CPU, as the evaluators' clean reference and the stock WDP input are
    (R/cpu_test test 9; DiffAttack's stage_clean): bicubic on a GPU can round a pixel differently. HxWx3 uint8."""
    return attack.read_image(src, 'cpu').round().clamp(0, 255).byte()[0].permute(1, 2, 0).numpy()


# ------------------------------------------------------------------------------------------------ checks (read-only)
def check_repo():
    """gen_wdp.check_repo with --no-optional-locks, so git never refreshes the clone's index (nothing written there)."""
    git = ['git', '--no-optional-locks', '-C', REPO]
    head = subprocess.check_output(git + ['rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(git + ['status', '--porcelain', '--untracked-files=all'], text=True).strip()
    if not head.startswith(gen_wdp.REPO_COMMIT):
        raise RuntimeError(f'WDP clone at {head}, expected {gen_wdp.REPO_COMMIT}')
    if dirty:
        raise RuntimeError(f'WDP clone is not clean:\n{dirty}')
    return head


def check_runtime():
    """Read-only replacement of gen_wdp.setup_runtime: every link it would create must already exist, as it made it."""
    want = {os.path.join(RUNTIME, 'assets', sub): os.path.join(REPO, 'assets', sub)
            for sub in ['target_images', 'test_images', 'obfs_target_images']}
    want[os.path.join(RUNTIME, 'assets', 'face_recognition_models')] = gen_wdp.FR_DIR
    want[os.path.join(RUNTIME, 'temp', 'shape_predictor_68_face_landmarks.dat')] = gen_wdp.DLIB_DAT
    for rel in gen_wdp.SD_LINKED:
        want[os.path.join(gen_wdp.SD_LOCAL, rel)] = os.path.realpath(os.path.join(gen_wdp.SD_SNAPSHOT, rel))
    bad = [d for d, s in want.items() if not (os.path.islink(d) and os.readlink(d) == s)]
    if bad:
        raise RuntimeError(f'stock runtime links missing or changed: {bad[:5]}')


def parse_log(text):
    """Every per-step value the stock attacker prints (gen_wdp.parse_losses keeps only first/last)."""
    get = lambda key: [float(v) for v in re.findall(rf'^{key}:\s+([-\d.eE+]+)', text, re.M)]
    return dict(adv_loss=get('adv_loss'), self_attn_loss=get('self_attn_loss'), loss=get('loss'))


def method_record(defaults):
    """What the PLAN and every manifest record about the method (build.py writes it, the generator re-derives it)."""
    return dict(
        name=METHOD, arm=ARM,
        paper='Salar et al., Enhancing Facial Privacy Protection via Weakening Diffusion Purification, CVPR 2025',
        repo=REPO, url=gen_wdp.REPO_URL, commit=gen_wdp.REPO_COMMIT,
        stock_wrapper=dict(path=str(STOCK_WRAPPER), sha256=sha256_file(STOCK_WRAPPER)),
        defaults=defaults, seed=STOCK_SEED, prot_steps=PROT_STEPS, prot_steps_reason=PROT_STEPS_REASON,
        prot_steps_code_default=defaults['prot_steps'],
        procedure='stock: dlib + FFHQ align -> 256 face; DDIM inversion (20 steps) + null-text optimisation (20 per '
                  'step); AdamW lr 1e-2 on the DDIM latent at start_step 17 for prot_steps 35 (paper; code default 30); loss = adv_optim_weight '
                  '* adv term + self-attention MSE; stock paste_back (Lanczos-4, hard quad mask). No pixel budget.',
        adv_term='Ours\' CLIP self term on the full 336 photo after a differentiable paste-back (replaces the FR '
                 'impersonation cosine loss; adv_optim_weight 0.003 kept)',
        instance_overrides={'get_FR_embeddings': 'returns the decoded 256 face itself (target call -> unused marker)',
                            'cosine_loss': 'returns Ours\' self term of the pasted-back 336 photo',
                            'MTCNN_cropping': 'False: the flag only crops the FR-loss input and the FR target; the MTCNN '
                                              'detection still runs as the stock no-face gate'},
        no_face='dlib no detection, or MTCNN no detection on the aligned crop -> clean PNG unchanged, protected=false')


# ------------------------------------------------------------------------------------------------------------ one photo
def run(root, setting, album, photo, device='cuda', test=None, eps=None, p_db=None):
    """One photo. `test` is for the CPU tests only (e.g. dict(prot_steps=1) under a CPU_TEST root)."""
    t0 = time.time()
    root = Path(root).resolve()
    assert (test is None) == (device != 'cpu'), 'CPU runs (with test overrides) are for the CPU tests only'
    if test is not None:
        assert set(test) <= {'prot_steps'} and str(root).startswith(str(HERE / 'CPU_TEST35') + os.sep), (test, root)
    plan = json.loads((root / 'PLAN.json').read_text())
    assert Path(plan['code_root']) == HERE, (plan['code_root'], HERE)
    assert plan.get('eps_levels') == eps, ('--eps differs from the PLAN', eps, plan.get('eps_levels'))
    assert plan.get('psnr_db') == p_db, ('--psnr differs from the PLAN', p_db, plan.get('psnr_db'))
    assert eps is None or p_db is None, '--eps and --psnr are exclusive'
    arm = (ARM if eps is None and p_db is None else f'{ARM}_eps{eps}' if eps is not None else f'{ARM}_psnr{p_db:g}')
    assert plan.get('arm', ARM) == arm, (plan.get('arm'), arm)
    entry = RGEN.photo_entry(plan, setting, album, photo)
    run_seed = plan['run_seed']
    seed = image_seed(run_seed, entry['album_index'])
    src = entry['source']
    src_sha = sha256_file(src)
    assert src_sha == entry['source_sha256'], (src, 'source photo changed since the build')
    assert entry['output_name'].endswith('.png'), entry
    name = entry['output_name']
    wrapper_sha = sha256_file(STOCK_WRAPPER)
    assert wrapper_sha == plan['method']['stock_wrapper']['sha256'], 'stock wrapper changed since the build'
    out = root / 'chunks' / setting / album / photo
    assert not os.path.lexists(out), f'{out} exists: a finished photo is never regenerated'

    # ---- preflight: before any write ----
    head = check_repo()
    check_runtime()
    out.parent.mkdir(parents=True, exist_ok=True)
    wroot = root / 'work' / setting / album / photo
    wroot.mkdir(parents=True, exist_ok=True)
    # Unique per attempt, on the same file system as chunks/ (atomic rename), and kept OUT of chunks/ so that folder only
    # ever holds finished photos. A failed attempt's folders stay in work/ as evidence and are never read.
    partial = Path(tempfile.mkdtemp(prefix=f'{photo}.partial.', dir=wroot))
    work = Path(tempfile.mkdtemp(prefix='attempt.', dir=wroot))
    dst = partial / ARM_DIR / name
    dst.parent.mkdir()

    # ---- our loss: Yo'LLaVA's encoder through Ours' code, hard-gated; loaded BEFORE the stock seed is set ----
    encoder, encoder_record = baselines.load_encoder(device)
    want = plan['encoder']
    for key in ('model_id', 'snapshot', 'config_sha256', 'weights_file'):
        assert encoder_record[key] == want[key], (key, encoder_record[key], want[key])
    assert RGEN.stat_binding(want['weights_file']) == want['weights_stat'], 'CLIP weights file changed since the build'
    encoder_record.update(weights_sha256_at_build=want['weights_sha256'], weights_stat=want['weights_stat'])
    # PBFACE: instance face term (pbface.py): target = THIS photo's clean buffalo_l embedding; weight asserted
    wtag = plan['weight_tag']
    face_weight = PB.check_weight(wtag, plan['face_weight'])
    face_matrix = PB.cached_matrix(plan, album, photo, src, entry['source_sha256'])
    if True:
        face_model, _face_bl = PB.load_recogniser(device)
        pbf_face = PB.InstanceFace(src, face_matrix, face_model, device)
        face_self_cos = pbf_face.self_check()
    # SHARE-MATCHED (the face weight is scaled to each method's own loss size, so the face term has the same share for
    # every baseline): the backend returns self + w * face and the method multiplies it by its
    # own adversarial weight a, so the total is a * self + (a * w) * face (+ its other terms): face weight scaled to the
    # method's own loss weight. w = 0.125 or 1/28 exactly as for the noise baselines (where a = 1).
    face_outer = 0.003
    assert face_outer == 0.003, face_outer
    objective = PB.FaceSelf(baselines.SelfObjective(encoder, encoder_record), pbf_face, face_weight, wtag)
    face_total_weight = face_outer * objective.weight
    assert objective.face_weight == objective.weight == face_weight == PB.WEIGHTS[wtag] and objective.outer == 1.0
    assert abs(face_total_weight - dict(w125=0.003 * 0.125, w28=0.003 * (1 / 28))[wtag]) < 1e-15, (face_total_weight, wtag)
    assert objective.face.source == str(src)
    clean_float = attack.read_image(src, device)                       # Ours' clean start point, float 336 (loss only)
    objective.begin(clean_float)                                       # clean reference features (no grad)
    clean = rounded_clean(src)               # WDP's input, the no-face output and the L-inf/PSNR reference (CPU rounding)
    rounding_device_vs_cpu = int((clean_float.round().clamp(0, 255).byte()[0].permute(1, 2, 0).cpu().numpy()
                                  != clean).any(-1).sum())             # diagnostic only
    H, W = clean.shape[:2]
    assert (H, W) == (IMG_SIZE, IMG_SIZE), (H, W)
    clean_png = work / 'clean_336.png'                                 # dlib and image_align read a file path
    Image.fromarray(clean).save(clean_png, format='PNG')
    assert np.array_equal(np.asarray(Image.open(clean_png).convert('RGB')), clean)
    measured = gen_wdp.verify_assets()
    cwd0 = os.getcwd()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.chdir(RUNTIME)
    sys.path.insert(0, RUNTIME)  # for 'assets.face_recognition_models' (namespace package)
    sys.path.insert(0, REPO)     # stock code

    import cv2
    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms
    import clip  # vendored in the repo
    # I/O only: point clip's downloader at our cache dir. It still checks the URL's sha256.
    clip.load = functools.partial(clip.load, download_root=gen_wdp.CLIP_ROOT)
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    wdp_main = importlib.import_module('main')          # stock main.py (guarded by __main__)
    import utils as wdp_utils                            # stock utils (builds the MTCNN at import)
    from dataset import ImageDataset
    from adversarial_optimization import Adversarial_Opt
    from align import LandmarksDetector, image_align
    for mod in (wdp_main, wdp_utils, sys.modules['dataset'], sys.modules['adversarial_optimization'],
                sys.modules['align'], sys.modules['attention_control'], sys.modules['criteria.cosine_loss'], clip):
        if not gen_wdp.inside_dir(mod.__file__, REPO):
            raise RuntimeError(f'{mod.__name__} imported from {mod.__file__}, not the clone')
    for m in ('irse', 'ir152', 'facenet'):
        f = sys.modules['assets.face_recognition_models.' + m].__file__
        if not gen_wdp.inside_dir(f, gen_wdp.FR_DIR):
            raise RuntimeError(f'FR model code {m} imported from {f}, not {gen_wdp.FR_DIR}')

    # ---- stock settings: the repo's own argparse defaults, untouched ----
    saved_argv = sys.argv
    sys.argv = ['main.py']
    args = wdp_main.parse_args()
    sys.argv = saved_argv
    stock_args = {k: v for k, v in vars(args).items()}
    assert json.loads(json.dumps(stock_args)) == plan['method']['defaults'], 'stock defaults differ from the PLAN'
    assert stock_args['prot_steps'] == PROT_STEPS_CODE_DEFAULT and plan['method']['prot_steps'] == PROT_STEPS \
        and plan['method']['prot_steps_reason'] == PROT_STEPS_REASON, 'prot_steps record differs from the PLAN'
    assert STOCK_SEED == 10, STOCK_SEED
    wdp_main.initialize_seed(seed=seed)                    # image seed: was initialize_seed(seed=STOCK_SEED = 10)
    args.device = torch.device(device)
    if device != 'cpu' and not torch.cuda.is_available():
        raise RuntimeError('no GPU visible')

    diff_model = StableDiffusionPipeline.from_pretrained(gen_wdp.SD_LOCAL, local_files_only=True).to(args.device)
    diff_model.scheduler = DDIMScheduler.from_config(diff_model.scheduler.config)
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    args.source_dir = ''
    args.dataloader = None
    opt = Adversarial_Opt(args, diff_model)
    assert opt.prot_steps == args.prot_steps == PROT_STEPS_CODE_DEFAULT, opt.prot_steps
    opt.prot_steps = PROT_STEPS          # the paper's 35: the ONLY change to the stock settings
    # image seed: the VAE posterior sample seed 8888 -> image seed, on this instance only (before the loss swap snapshot)
    opt.image2latent = types.MethodType(seeded_image2latent(sys.modules['adversarial_optimization'], seed), opt)
    if test is not None:
        opt.prot_steps = test.get('prot_steps', opt.prot_steps)
    landmarks_detector = LandmarksDetector(os.path.join(RUNTIME, 'temp', 'shape_predictor_68_face_landmarks.dat'))
    import diffusers, transformers, dlib
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'hip': torch.version.hip,
           'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0) if device != 'cpu' else 'cpu',
           'diffusers': diffusers.__version__, 'transformers': transformers.__version__,
           'dlib': dlib.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
           'facenet_pytorch': importlib.metadata.version('facenet-pytorch'),
           'python_executable': sys.executable, 'host': platform.node(),
           'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES'),
           'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'slurm_job': os.environ.get('SLURM_JOB_ID')}

    safe = re.sub(r'[^A-Za-z0-9_-]', '_', os.path.splitext(name)[0])
    iw = work / safe  # stock run() rebuilds '<source_dir>/<name>.png' from split('.')[0]
    al_dir = iw / 'aligned'
    al_dir.mkdir(parents=True)
    face = {}
    setup = time.time() - t0

    faces = list(landmarks_detector.get_landmarks(str(clean_png)))
    face['dlib_n_faces'] = len(faces)
    status, trace, losses = None, [], None
    ball, raw_img, bound, search = None, None, None, None
    if not faces:
        status = 'no_face_dlib'
    else:
        lm = faces[0]  # stock work_landmark keeps only the first detection (later ones hit 'exists')
        al_path = al_dir / (safe + '.png')
        image_align(str(clean_png), str(al_path), lm, output_size=256)
        aligned_clean = np.array(Image.open(al_path).convert('RGB'))
        if aligned_clean.shape != (256, 256, 3):
            raise RuntimeError(f'{name}: stock alignment gave {aligned_clean.shape}')
        quad, qsize = ffhq_quad(lm)
        fquad, scale, shrink = stock_frame(quad, qsize, W, H)
        face.update({'quad_nw_sw_se_ne': quad.round(3).tolist(), 'qsize': round(qsize, 3), 'stock_shrink': shrink})
        rt, inside = paste_back(clean, aligned_clean, fquad, scale)
        face['roundtrip_psnr_region'] = psnr(rt, clean, inside)
        try:  # the stock run() calls utils.alignment(Image.open(path)) -> boxes[0]; None would crash it
            bb = wdp_utils.alignment(Image.open(al_path))
            face['mtcnn_box_aligned'] = [round(float(v), 2) for v in bb]
        except TypeError:
            status = 'no_face_mtcnn_aligned'
    if status is None:
        if eps is not None:
            ball = LinfBall(clean, eps, args.device, region=inside)
        elif p_db is not None:
            ball = L2Ball(clean, p_db, args.device)
        hook = SelfTermLoss(objective, PasteBack(clean, fquad, scale, n=256, device=args.device), ball)
        assert np.array_equal(hook.paste.mask, inside), 'differentiable paste footprint != stock paste footprint'
        face['instance_attributes_swapped'] = hook.install(opt)
        stock_out = iw / 'stock_out'
        prot_path = stock_out / args.test_model_name[0] / args.target_choice / (safe + '.png')
        log_path = iw / 'stock_stdout.log'
        opt.source_dir = str(al_dir)
        opt.protected_image_dir = str(stock_out)
        opt.dataloader = DataLoader(ImageDataset(str(al_dir), transform), batch_size=1, shuffle=False)
        rng_before, cpu_before = gen_wdp.rng_fingerprint(), torch.get_rng_state()
        t1 = time.time()
        with open(log_path, 'w') as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
            opt.run()
        face['stock_run_seconds'] = round(time.time() - t1, 2)
        # Expected CPU state = before + only the DataLoader iterator's base_seed draw (unused, num_workers=0)
        cpu_after = torch.get_rng_state()
        torch.set_rng_state(cpu_before)
        iter(DataLoader(ImageDataset(str(al_dir), transform), batch_size=1, shuffle=False))
        cpu_expect = torch.get_rng_state()
        torch.set_rng_state(cpu_after)
        face['global_rng_untouched_by_protection'] = bool(gen_wdp.rng_fingerprint() == rng_before
                                                          and torch.equal(cpu_after, cpu_expect))
        text = log_path.read_text()
        face.update(gen_wdp.parse_losses(text))
        losses = parse_log(text)
        trace = hook.trace
        n = opt.prot_steps
        assert hook.target_calls == 1 and len(trace) == n, (hook.target_calls, len(trace), n)
        assert args.adv_optim_weight == face_outer == 0.003, (args.adv_optim_weight, face_outer)   # PBFACE
        face_residual = PB.check_calls(objective.calls, objective.weight, pbf_face.active, expect_steps=n)   # PBFACE
        assert len(losses['adv_loss']) == len(losses['self_attn_loss']) == len(losses['loss']) == n, \
            {k: len(v) for k, v in losses.items()}
        for row, adv in zip(trace, losses['adv_loss']):   # WDP's adv_loss IS adv_optim_weight * our self term
            assert math.isclose(adv, args.adv_optim_weight * row['self_term'], rel_tol=1e-5, abs_tol=1e-9), (adv, row)
            assert row['face_grad_abs_mean'] is not None and math.isfinite(row['face_grad_abs_mean']), row
        raw = cv2.imread(str(prot_path)) if prot_path.is_file() else None
        if raw is None or raw.shape != (256, 256, 3):
            raise RuntimeError(f'{name}: stock run produced no valid 256x256 face at {prot_path}')
        prot = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        out_img, inside = paste_back(clean, prot, fquad, scale)
        if ball is not None:               # the same projection as in the loop, on the stock pasted-back photo
            raw_img = out_img
            if p_db is None:
                out_img = ball.project_uint8(raw_img)
            else:
                out_img, search = ball.project_uint8(raw_img)
        Image.fromarray(out_img).save(dst, format='PNG')
        diff = np.abs(out_img.astype(np.int16) - clean.astype(np.int16))
        face.update({'status': 'protected', 'face_found': True,
                     'region_frac': round(float(inside.mean()), 5),
                     'changed_frac': round(float((diff.max(-1) > 0).mean()), 5),
                     'psnr_region': psnr(out_img, clean, inside),
                     'aligned_psnr_vs_clean_aligned': psnr(prot, aligned_clean)})
    else:
        shutil.copyfile(clean_png, dst)
        face.update({'status': status, 'face_found': status != 'no_face_dlib', 'kept_clean': True,
                     'region_frac': 0.0, 'changed_frac': 0.0})
    with Image.open(dst) as im:
        assert im.format == 'PNG' and im.mode == 'RGB' and im.size == (IMG_SIZE, IMG_SIZE), (dst, im.format, im.size)
        saved = np.asarray(im)
    protected = face['status'] == 'protected'
    if not protected:
        assert np.array_equal(saved, clean) and sha256_file(dst) == sha256_file(clean_png), 'no-face output != clean'
    linf = int(np.abs(saved.astype(np.int16) - clean.astype(np.int16)).max())
    if eps is not None:
        assert linf <= eps, ('L-inf bound violated on the delivered PNG', linf, eps)
    psnr_check = psnr_value(clean, saved)
    if p_db is not None:
        assert (psnr_check is None) == (not protected), ('no-face photo must be identical to clean, protected must differ', psnr_check)
        assert psnr_check is None or (psnr_check >= p_db and psnr_check == search['psnr']), (
            'PSNR bound violated on the delivered PNG', psnr_check, p_db, search)
    with torch.no_grad():
        delivered = float(objective(torch.from_numpy(saved).permute(2, 0, 1)[None].float().to(args.device)))
    if p_db is not None:
        bound = dict(psnr_db=p_db, center='rounded clean 336 photo (round(attack.read_image(source, "cpu")))',
                     projection='d = x - c on the full pasted-back 336 photo; d *= min(1, r / ||d||_2), r = sqrt(N 255^2 / '
                                '10^(P/10)); (c + d).clamp(0, 255); plain autograd; saved: same projection + budget-'
                                'filling rounding to whole-photo PSNR >= P (L2Ball.project_uint8)',
                     delivered=dict(search or {}, psnr=psnr_check, linf=linf, self_loss=delivered,
                                    frac_values_changed=float((saved != clean).mean())),
                     psnr_assert=f'PSNR >= {p_db} (whole 336 photo vs the rounded clean) asserted on the re-read PNG')
        if protected:
            with torch.no_grad():
                raw_loss = float(objective(torch.from_numpy(raw_img).permute(2, 0, 1)[None].float().to(args.device)))
            bound.update(raw_unprojected=dict(linf=int(np.abs(raw_img.astype(np.int16) - clean.astype(np.int16)).max()),
                                              psnr=psnr_value(clean, raw_img), self_loss=raw_loss),
                         loop_first=trace[0], loop_last=trace[-1], loop_min_self_term=min(r['self_term'] for r in trace))
    if eps is not None:
        region = inside if protected else None
        bound = dict(eps_levels=eps, eps=f'{eps}/255', center='rounded clean 336 photo (round(attack.read_image(source, "cpu")))',
                     projection='x_proj = (c + (x - c).clamp(-eps, eps)).clamp(0, 255) on the full pasted-back 336 photo; '
                                'plain autograd', delivered=dict(bound_metrics(clean, saved, eps, region),
                                                                 psnr=psnr(saved, clean), self_loss=delivered),
                     linf_assert=f'max |x_adv - c| <= {eps} asserted on the re-read PNG')
        if protected:
            with torch.no_grad():
                raw_loss = float(objective(torch.from_numpy(raw_img).permute(2, 0, 1)[None].float().to(args.device)))
            bound.update(raw_unprojected=dict(linf=int(np.abs(raw_img.astype(np.int16) - clean.astype(np.int16)).max()),
                                              psnr=psnr(raw_img, clean), self_loss=raw_loss),
                         loop_first=trace[0], loop_last=trace[-1], loop_min_self_term=min(r['self_term'] for r in trace))
    if protected:
        with (dst.parent / f'{photo}_loss.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            bkeys = [] if ball is None else list(ball.STAT_KEYS)
            writer.writerow(['step', 'self_term', 'adv_loss', 'self_attn_loss', 'loss', 'face_grad_abs_mean'] + bkeys)
            for row, a, s, l in zip(trace, losses['adv_loss'], losses['self_attn_loss'], losses['loss']):
                writer.writerow([row['step'], row['self_term'], a, s, l, row['face_grad_abs_mean']] + [row.get(k) for k in bkeys])
    check_repo()  # raises if anything (e.g. __pycache__) was written into the clone during the run
    code = {str(p): sha256_file(p) for p in (Path(__file__).resolve(), R / 'baselines.py', R / 'generate.py',
                                             R / 'code_snapshot/anti_persona/attack.py', STOCK_WRAPPER)}
    for m in list(sys.modules.values()):
        f = getattr(m, '__file__', None)
        if f and (gen_wdp.inside_dir(f, REPO) or gen_wdp.inside_dir(f, gen_wdp.FR_DIR)):
            code[os.path.realpath(f)] = sha256_file(f)
    node = socket.gethostname().split('.')[0]
    rule = dict(plan['method'], git_head=head, worktree_clean_before_and_after=True, seed_used=seed,
                test_overrides=test, prot_steps_run=opt.prot_steps, prot_steps_reason=PROT_STEPS_REASON,
                assets=dict(measured_sha256={k: measured[k] for k in ('sd', 'fr', 'mtcnn', 'dlib', 'clip')},
                            sd_used=f'{gen_wdp.SD_MIRROR_ID}@{gen_wdp.SD_MIRROR_REV}'))
    loss = dict(objective.record(), applied_as='WDP adv_loss = adv_optim_weight (0.003) * self term; WDP loss = adv_loss '
                                               '+ self_attn_loss (stock attacker)',
                image='full 336x336 photo: rounded clean outside the stock quad mask, differentiable Lanczos-4 '
                      'paste of (decoded face / 2 + 0.5).clamp(0, 1) * 255 inside, clamp(0, 255)',
                self_term_first=trace[0]['self_term'] if trace else None,
                self_term_last_step=trace[-1]['self_term'] if trace else None,
                self_term_delivered_png=delivered,
                adv_loss_first=losses['adv_loss'][0] if losses else None,
                adv_loss_last=losses['adv_loss'][-1] if losses else None,
                self_attn_loss_first=losses['self_attn_loss'][0] if losses else None,
                self_attn_loss_last=losses['self_attn_loss'][-1] if losses else None)
    manifest = dict(
        status='complete', arm=ARM, method=METHOD, queue=plan['queue'], setting=setting, album=album, photo=photo,
        source=src, source_sha256=src_sha, ordinal=entry['ordinal'], inputs=[src],
        clean='round(attack.read_image(source, "cpu")) (336x336 uint8: what WDP aligns and pastes into, the no-face '
              'output and the L-inf/PSNR reference); loss reference = ClipFeatures(attack.read_image(source, device))',
        rounded_clean_pixels_differing_device_vs_cpu=rounding_device_vs_cpu,
        output=str(out / ARM_DIR / name), output_sha256=sha256_file(dst), protected=protected, face=face,
        rule=rule, loss=loss, encoder=encoder_record, image_seed=seed, run_seed=run_seed, album_index=entry['album_index'],
        image_seed_rule='run_seed + album_index (Ours\' rule)',
        seed_override=dict(initialize_seed=dict(stock=STOCK_SEED, used=seed, how='main.initialize_seed(seed=image_seed)'),
                           vae_sample_generator=dict(stock=STOCK_VAE_SEED, used=seed,
                                                     how='AST: Adversarial_Opt.image2latent manual_seed(8888) constant '
                                                         'swapped; bound on the one opt instance')),
        code_sha256=code, linf_int_vs_rounded_clean=linf, psnr_vs_rounded_clean=psnr(saved, clean),
        psnr_note='null = identical to the rounded clean (infinite PSNR); no pixel budget (unbounded, as published)',
        env=env, hostname=node, gpu=os.environ.get('HIP_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES')),
        slot=os.environ.get('SLURM_JOB_ID'), device=str(args.device), work=str(work),
        gpu_max_memory_allocated_gb=(round(torch.cuda.max_memory_allocated() / 2 ** 30, 3) if device != 'cpu' else None),
        gpu_max_memory_reserved_gb=(round(torch.cuda.max_memory_reserved() / 2 ** 30, 3) if device != 'cpu' else None),
        setup_seconds=round(setup, 2), seconds=round(time.time() - t0, 2))
    if p_db is not None:               # --psnr only
        manifest.update(arm=arm, rule=dict(rule, psnr_db=p_db), bound=bound,
                        psnr_note='null = identical to the rounded clean (infinite PSNR); PSNR >= ' + f'{p_db:g} dB inside the optimisation')
    if eps is not None:                # --eps only: the unbounded manifest stays exactly the unbounded generator's
        manifest.update(arm=arm, rule=dict(rule, eps_levels=eps), bound=bound,
                        psnr_note='null = identical to the rounded clean (infinite PSNR); L-inf bound ' + f'{eps}/255 inside the optimisation')
    manifest.update(weight_tag=wtag, face_weight=face_weight, face_active=bool(pbf_face.active), face_matrix=face_matrix,   # PBFACE
                    face_target='clean embedding of the SAME photo', face_target_self_cosine=face_self_cos,
                    face_calls=len(objective.calls),
                    face_raw_first=objective.calls[0]['face_raw'] if objective.calls else None,
                    face_raw_last=objective.calls[-1]['face_raw'] if objective.calls else None,
                    terms_residual_max=face_residual if protected else None, face_rows=plan['face_rows'],
                    face_rows_sha256=plan['face_rows_sha256'],
                    face_outer_multiplier=face_outer, face_internal_coefficient=objective.weight, face_total_weight=face_total_weight,
                    face_weight_rule='share-matched: face weight scaled to the method own loss weight',
                    loss_note='self term = self + face_weight * face; x adv_optim_weight 0.003 as published -> 0.003 self + (0.003 face_weight) face')
    RGEN.put(dst.parent / 'manifest.json', manifest)
    RGEN.put(partial / 'GROUP.json', dict(status='complete', setting=setting, album=album, photo=photo, arms=[ARM_DIR],
                                          protected=protected, hostname=node, seconds=round(time.time() - t0, 2)))
    os.chdir(cwd0)
    os.rename(partial, out)
    summary = dict(photo=f'{setting}/{album}/{photo}', status=face['status'], linf=linf,
                   psnr=psnr(saved, clean), self_first=loss['self_term_first'],
                   self_last=loss['self_term_last_step'], delivered=delivered,
                   seconds=round(time.time() - t0, 1))
    if p_db is not None:
        summary.update(psnr_db=p_db, psnr_delivered=psnr_check, scale_final=(search or {}).get('scale_final'),
                       raw_loss=bound.get('raw_unprojected', {}).get('self_loss'))
    if eps is not None:
        summary.update(eps=eps, at_bound=bound['delivered']['frac_values_at_bound'],
                       at_bound_face=bound['delivered'].get('face_frac_values_at_bound'),
                       raw_loss=bound.get('raw_unprojected', {}).get('self_loss'))
    print(json.dumps(summary), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--setting', required=True)
    p.add_argument('--album', required=True)
    p.add_argument('--photo', required=True)
    p.add_argument('--eps', type=int, default=None, help='L-inf bound in pixel levels (E/255), applied inside the '
                   'optimisation; omitted = the unbounded generator, unchanged')
    p.add_argument('--psnr', type=float, default=None, help='PSNR bound in dB (L2 ball), applied inside the '
                   'optimisation; exclusive with --eps')
    a = p.parse_args()
    node = socket.gethostname().split('.')[0]
    assert not node.endswith(EXCLUDED), node
    assert os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline HF env'
    out = run(a.root, a.setting, a.album, a.photo, eps=a.eps, p_db=a.psnr)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
