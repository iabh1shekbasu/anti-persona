# REFERENCE COPY (anti-persona release): the generator that produced the paper's DiffAttack + face images, kept for its exact
# procedure and settings. Only absolute paths were replaced by environment variables; the logic is unchanged.
# It runs inside the research harness it was written for (a queue PLAN.json, the legacy baseline module and the
# stock wrapper); see baselines/README.md, section "DiffAttack and WDP".
# Instance face term: the lines marked PBFACE add the baseline face term of ../pbface.py to the self-term generator.
"""DiffAttack (Chen et al., TPAMI 2024) re-run with OUR loss plus the instance face term, for ONE photo -- eps- or
PSNR-BOUNDED. HERE = this code folder; R = the legacy baselines folder (LEGACY_BASELINES_DIR: code_snapshot,
baselines.py, build.py).

Image seed: the per-photo image seed = PLAN run_seed (0 or 202) + the photo's album_index (Ours' rule) REPLACES the
authors' two hard-coded seeds, both recorded in the manifest (seed_override):
  1. main.py's module-level seed_torch(42) -> seed_torch(image_seed): main.py is still run unedited from disk, but its
     source is parsed and that ONE call's constant swapped (AST) before it is executed as __main__ (runpy's own
     _run_module_code, i.e. what runpy.run_path does after compiling the file);
  2. diff_latent_attack.encoder's VAE posterior generator torch.Generator().manual_seed(8888) -> manual_seed(image_seed)
     (compiled by AST from the pinned stock file in the stock module's globals; the module attribute is replaced, so
     ddim_reverse_sample, the only caller, uses it). This sample is the only random draw of the procedure, so without
     (2) the two run seeds would give identical images.

PSNR bound: --psnr P (exclusive with --eps) applies an in-loop projection onto the PSNR-P L2 ball around the rounded
clean 336 photo (see L2Ball); the saved PNG is projected + rounded to PSNR >= P (asserted). The paper uses P = 30.

L-inf bound: ONE opt-in flag, --eps E (integer pixel levels, E/255). Without --eps every statement that runs is the unbounded generator's (same
outputs; the manifest/CSV carry no extra field). With --eps E the bound is applied INSIDE the optimisation: at every
latent iteration the decoded image's 336 view v (the stock decode -> (x/2+0.5).clamp(0,1)*255 -> antialiased bicubic
336, the image our loss sees) is projected onto the L-inf ball of radius E around the rounded clean 336 photo c (the
image every metric is measured against):  x_proj = (c + (v - c).clamp(-E, E)).clamp(0, 255),  and our loss is computed
on x_proj; the gradient reaches the latent through the projection by plain autograd (zero on clipped pixels, no
straight-through). The delivered PNG gets the same projection: the stock 352 adv PNG -> PIL bicubic 336 (unchanged
stock wrapper step) -> clip to [c - E, c + E] in integers (= the formula above; c and E integers, so no rounding) and the
generator asserts max |x_adv - c| <= E on the PNG it re-reads. Iterations, LR, weights, seed, surrogate, preprocessing:
unchanged. Logged per photo: per-iteration loss + fraction of the raw view outside the ball + fraction at the bound
(<photo>_loss.csv), final loss, PSNR, at-bound fraction, and the loss of the raw (unprojected) delivered image.

Why: stock DiffAttack attacks its authors' ImageNet Inception-v3 (cross-entropy on an ImageNet label). In the paper
every baseline optimises OUR loss on the encoder Ours attacks and otherwise keeps its own published procedure, run once
at its defaults (no tuning). DiffAttack's procedure is kept by running the stock wrapper's steps
(gen/diffattack/gen_diffattack.py under the legacy baselines' STOCK folder; reference/diffattack/gen_diffattack.py here;
its helpers are imported from that file, not copied) and the authors' main.py VERBATIM (runpy, in this process). The ONE change to the
method: in the authors' diff_latent_attack.diffattack() the classifier attack loss (two statements of the latent loop)

    if args.dataset_name != 'imagenet_compatible':
        pred = classifier(out_image) / 10
    else:
        pred = classifier(out_image)
    attack_loss = -cross_entro(pred, label) * args.attack_loss_weight

is replaced by

    attack_loss = OURLOSS(init_out_image) * args.attack_loss_weight

compiled from the pinned stock file by AST (every other statement of the function is the stock one; the module
attribute diff_latent_attack.diffattack, which main.run_diffusion_attack calls, is set to the compiled function).
OURLOSS = Ours' CLIP self term (loss preset 'self': anti_persona.attack.clip_objective with LossWeights(self=1, gray=0,
prototype=0, evidence=0, face=0) and a zero prototype placeholder = R/baselines.SelfObjective) on Yo'LLaVA's own vision
tower openai/clip-vit-large-patch14-336 through anti_persona.attack.ClipFeatures (hard-gated by baselines.load_encoder;
built with eager attention, as in Ours' llava env, so its features are bit-identical there), against the clean features ClipFeatures(attack.read_image(<original photo>)). It sees the decoded image's 336 view:
init_out_image ([1, 3, 352, 352] in [-1, 1]; init_mask = ones, so it is the VAE decode) -> (x / 2 + 0.5).clamp(0, 1)
(the stock code's own mapping) * 255 -> antialiased bicubic resize to 336 (torch's PIL-style kernel; the delivered PNG
is the PIL-bicubic 336 resize of the stock 352 output) -> clamp(0, 255). Ours minimises the self term (cosine to the
clean features) and the stock loss is minimised, so it enters with +; its weight is the stock attack_loss_weight (10).
Every CLIP computation (clean reference, the loss forward AND backward, the delivered-PNG diagnostic) runs under Ours'
convolution backend (attack.seed_everything's cudnn/MIOpen flags: disabled on ROCm); SD and Inception keep the stock
seed_torch(42) flags. Nothing else of Ours runs: no prototype, face term, JPEG EOT, gradient smoothing or person mask.

Unchanged stock procedure: main.py's own argparse defaults with only the stock wrapper's four arguments (--res 352
--images_root --label_path --save_dir): DDIM 20 steps, attack from step 15, 30 latent iterations (AdamW 1e-2),
guidance 2.5, null-text (uncond embedding) optimisation, cross-attention variance x10000, self-attention structure
x100, no mask, Manojb/stable-diffusion-2-base, seed_torch(42). The ImageNet label that sets the SD prompt and the
cross-attention token is the stock wrapper's rule (compute_labels: Inception-v3 top-1 on the clean image at 352,
imported from the wrapper); the stock clean check must agree with it. 352 -> 336 I/O as the stock wrapper: the 336
input (round(attack.read_image(source)) = the stock run's staged input) goes in, the stock Lanczos 336 -> 352 runs
inside main.py, and the stock 352 *_adv_image.png is PIL-bicubic resized to the delivered 336x336 PNG. The stock
post-generation transfer/FID evaluation (other_attacks.model_transfer) is never entered: a stub raises where the stock
wrapper terminated main.py (right after the 'Adv acc:' line). UNBOUNDED as published; integer L-inf and PSNR vs the
rounded clean are recorded.

  python -B generate.py --root <queue root> --setting <s> --album <a> --photo <p> [--device cuda]
  (diffattack env python; cwd = the stock repo, as the stock wrapper ran main.py)

Writes <root>/chunks/<setting>/<album>/<photo>/{GROUP.json, diffattack/{<output_name>, manifest.json,
<photo>_loss.csv}} by renaming a private mkdtemp folder <root>/partial/<setting>/<album>/<photo>.<random> into place only
after every check passed; refuses if the final folder exists. The stock side files (staged input, labels.txt, the stock
352 outputs, the stock log) stay in <root>/work/<setting>/<album>/<photo>.<random>/. The photo entry, its source hash,
the encoder snapshot, the stock commit/wrapper hash, the Inception weights and the SD-2-base snapshot binding come from
<root>/PLAN.json and are checked before any model loads.
"""
import sys

sys.dont_write_bytecode = True

import argparse              # noqa: E402
import ast                   # noqa: E402
import contextlib            # noqa: E402
import csv                   # noqa: E402
import glob                  # noqa: E402
import importlib.util        # noqa: E402
import io                    # noqa: E402
import json                  # noqa: E402
import os                    # noqa: E402
from pathlib import Path     # noqa: E402
import runpy                 # noqa: E402
import socket                # noqa: E402
import tempfile              # noqa: E402
import time                  # noqa: E402

HERE = Path(__file__).resolve().parent
R = Path(os.environ['LEGACY_BASELINES_DIR'])   # research harness: legacy baselines.py + code_snapshot/anti_persona
sys.path.insert(0, str(R / 'code_snapshot'))
sys.path.insert(0, str(R))

import numpy as np           # noqa: E402
import torch                 # noqa: E402
import torch.nn.functional as F   # noqa: E402
from PIL import Image        # noqa: E402

from anti_persona import attack   # noqa: E402
import baselines                  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # PBFACE: pbface.py is in the parent folder
import pbface as PB               # noqa: E402  PBFACE

assert Path(attack.__file__).resolve() == R / 'code_snapshot/anti_persona/attack.py', attack.__file__
assert Path(baselines.__file__).resolve() == R / 'baselines.py', baselines.__file__

WRAPPER_FILE = baselines.STOCK / 'gen/diffattack/gen_diffattack.py'


def load_stock_wrapper(path=WRAPPER_FILE):
    """The stock wrapper as a module (its main() is never called; module level only defines constants)."""
    spec = importlib.util.spec_from_file_location('stock_gen_diffattack', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


W = load_stock_wrapper()
REPO = Path(W.REPO)
sys.path.insert(0, str(REPO))     # as `python main.py` (the script folder first): the stock modules are the repo's
EXCLUDED = ('-193', '-212', '-214', '-207', '-218')
ARM, METHOD = 'diffattack_ourloss', 'diffattack'
ARM_SEED = 42                     # = the stock hard-coded seed_torch(42); recorded only (see image_seed)
STOCK_MODULES = ('main.py', 'diff_latent_attack.py', 'attentionControl.py', 'utils.py', 'distances.py', 'other_attacks.py')
# The two statements of the stock latent loop that ARE the classifier attack loss (ast.unparse form) and the replacement.
STOCK_PRED = ("if args.dataset_name != 'imagenet_compatible':\n    pred = classifier(out_image) / 10\n"
              "else:\n    pred = classifier(out_image)")
STOCK_LOSS = 'attack_loss = -cross_entro(pred, label) * args.attack_loss_weight'
OUR_LOSS = 'attack_loss = OURLOSS(init_out_image) * args.attack_loss_weight'
MAIN_EXTRA_ARGS = ()              # production: none (the CPU test alone sets ('--iterations', '1'))
VIEW = 336


def put(path, value):
    path.write_text(json.dumps(value, indent=1, default=str) + '\n')


def stat_binding(path):
    s = Path(path).stat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, ino=s.st_ino)


def photo_entry(plan, setting, album, photo):
    rows = [r for r in plan['settings'][setting]['photos'] if r['album'] == album and r['photo'] == photo]
    assert len(rows) == 1, (setting, album, photo, len(rows))
    return rows[0]


RUN_SEEDS = (0, 202)              # the paper's two run seeds
STOCK_VAE_SEED = 8888


def image_seed(run_seed, album_index):
    """Ours' per-photo rule: run seed + the photo's index in the person's album."""
    assert run_seed in RUN_SEEDS and album_index >= 0
    return run_seed + album_index


def _swap_constant(node, old, new):
    """Replace the single int Constant `old` inside `node` by `new`; asserts exactly one occurrence."""
    hits = [n for n in ast.walk(node) if isinstance(n, ast.Constant) and type(n.value) is int and n.value == old]
    assert len(hits) == 1, (old, len(hits))
    hits[0].value = new


def seeded_main_code(path, seed):
    """main.py compiled with ONLY its module-level `seed_torch(42)` call changed to `seed_torch(seed)`."""
    tree = ast.parse(Path(path).read_text(), filename=str(path))
    calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and isinstance(n.value.func, ast.Name) and n.value.func.id == 'seed_torch']
    assert len(calls) == 1 and ast.unparse(calls[0]) == f'seed_torch({ARM_SEED})', [ast.unparse(c) for c in calls]
    _swap_constant(calls[0], ARM_SEED, seed)
    assert ast.unparse(calls[0]) == f'seed_torch({seed})'
    return compile(tree, str(path), 'exec')


def seeded_vae_encoder(module, seed):
    """diff_latent_attack.encoder compiled from the stock file with ONLY manual_seed(8888) -> manual_seed(seed)."""
    path = Path(module.__file__)
    tree = ast.parse(path.read_text(), filename=str(path))
    (fn,) = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'encoder']
    assert 'torch.Generator().manual_seed(8888)' in ast.unparse(fn), ast.unparse(fn)
    _swap_constant(fn, STOCK_VAE_SEED, seed)
    scratch = dict(vars(module))      # a copy of the stock module's globals (it reads torch / preprocess only)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), scratch)
    return scratch['encoder']


@contextlib.contextmanager
def eager_clip_attention():
    """CLIPModel.from_pretrained(..., attn_implementation='eager') while Ours' ClipFeatures builds the tower: the
    attention Ours runs in its own env (transformers 4.37: eager); this env's transformers 4.46 would pick SDPA (CPU
    parity: eager is bit-identical to the llava env, SDPA is not). Scoped: CLIPModel is restored before SD loads."""
    from transformers import CLIPModel
    assert 'from_pretrained' not in vars(CLIPModel), 'CLIPModel.from_pretrained already overridden'
    inherited = CLIPModel.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        assert 'attn_implementation' not in kwargs, kwargs
        return inherited(cls, *args, attn_implementation='eager', **kwargs)
    CLIPModel.from_pretrained = classmethod(from_pretrained)
    try:
        yield
    finally:
        del CLIPModel.from_pretrained
    assert 'from_pretrained' not in vars(CLIPModel)


def load_encoder_eager(device):
    """baselines.load_encoder (the hard gate: snapshot listing, model id, vision config, every vision tensor ==
    pytorch_model.bin) with eager attention; the backend is asserted on every layer and recorded."""
    with eager_clip_attention():
        encoder, record = baselines.load_encoder(device)
    kinds = sorted({type(layer.self_attn).__name__ for layer in encoder.vision.encoder.layers})
    assert kinds == ['CLIPAttention'] and encoder.vision.config._attn_implementation == 'eager', (
        kinds, encoder.vision.config._attn_implementation)
    record.update(attn_implementation='eager', attention_class='CLIPAttention')
    return encoder, record


def sd_binding_problems(want):
    """[] unless StableDiffusionPipeline.from_pretrained(<stock repo id>) resolves exactly the snapshot bound at the
    build: hub cache folder, refs/main, the snapshot's file listing, every small file's sha256 and every large weight
    file's (size, mtime_ns, inode) (its sha256 was taken at the build)."""
    from huggingface_hub.constants import HF_HUB_CACHE
    d = Path(HF_HUB_CACHE) / ('models--' + want['repo_id'].replace('/', '--'))
    if str(d) != want['cache']:
        return [f'hub cache {d} != {want["cache"]}']
    snap = (d / 'refs/main').read_text().strip()
    if snap != want['snapshot']:
        return [f'refs/main {snap} != {want["snapshot"]}']
    root = d / 'snapshots' / snap
    listing = sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())
    if listing != sorted(want['files']):
        return [f'snapshot listing {listing} != {sorted(want["files"])}']
    out = []
    for name, rec in want['files'].items():
        if 'stat' in rec:
            if stat_binding((root / name).resolve()) != rec['stat']:
                out.append(f'{name}: weights file changed since the build')
        elif baselines.sha(root / name) != rec['sha256']:
            out.append(f'{name}: sha256 differs from the build')
    return out


def inception_problems(want):
    """[] unless the Inception-v3 checkpoint the stock label rule and clean check load (torchvision, TORCH_HOME = the
    stock torch_home) exists with the build's stat and sha256; checked BEFORE the loader could try to download."""
    path = Path(W.TORCH_HOME) / 'hub/checkpoints/inception_v3_google-0cc3c7bd.pth'
    if str(path) != want['file'] or not path.is_file():
        return [f'{path} missing or not the built path {want["file"]}']
    if stat_binding(path) != want['stat'] or baselines.sha(path) != want['sha256']:
        return [f'{path} changed since the build']
    return []


def stage_clean(source):
    """The 336 input DiffAttack gets: round(attack.read_image(source)) on CPU, i.e. the stock run's staged input
    (R cpu_test 9: byte-equal for all 335 photos) and the rounded clean every metric here is measured against."""
    clean = attack.read_image(source, 'cpu')
    return clean.round().clamp(0, 255)[0].permute(1, 2, 0).to(torch.uint8).numpy()


# Ours' convolution backend for every CLIP computation = what attack.seed_everything sets before Ours' loop (on ROCm it
# disables MIOpen: cudnn.enabled False; benchmark False, deterministic True). The stock code (SD, Inception) keeps the
# stock seed_torch(42) flags (enabled True, benchmark False, deterministic True). cpu_test checks both.
CLIP_BACKEND = dict(enabled=not torch.version.hip, benchmark=False, deterministic=True)
STOCK_BACKEND = dict(enabled=True, benchmark=False, deterministic=True)


def backend_now():
    c = torch.backends.cudnn
    return dict(enabled=c.enabled, benchmark=c.benchmark, deterministic=c.deterministic)


@contextlib.contextmanager
def clip_backend():
    c = torch.backends.cudnn
    with c.flags(enabled=CLIP_BACKEND['enabled'], benchmark=CLIP_BACKEND['benchmark'], benchmark_limit=c.benchmark_limit,
                 deterministic=CLIP_BACKEND['deterministic'], allow_tf32=c.allow_tf32):
        assert backend_now() == CLIP_BACKEND
        yield


class _UnderClipBackend(torch.autograd.Function):
    """fn(x) inside the stock graph, its forward AND backward under Ours' backend (the convolution backend is chosen
    when each op runs, so the backward of a plain call would run under the stock flags of loss.backward())."""

    @staticmethod
    def forward(ctx, x, fn):
        with torch.enable_grad(), clip_backend():
            leaf = x.detach().requires_grad_(True)
            out = fn(leaf)
        ctx.leaf, ctx.out = leaf, out
        return out.detach()

    @staticmethod
    def backward(ctx, grad):
        with clip_backend():
            (g,) = torch.autograd.grad(ctx.out, ctx.leaf, grad)
        ctx.leaf = ctx.out = None
        return g, None


def begin_clean(objective, clean):
    """objective.begin (clean features, no grad) under Ours' backend."""
    with clip_backend():
        objective.begin(clean)


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
    project(v) = (c + (v - c).clamp(-eps, eps)).clamp(0, 255): plain autograd (gradient 0 where clipped)."""

    STAT_KEYS = ('frac_outside', 'frac_at_bound', 'raw_linf')

    def __init__(self, clean_hwc_uint8, eps, device):
        assert isinstance(eps, int) and 1 <= eps <= 255, eps
        self.eps = eps
        self.center_np = np.asarray(clean_hwc_uint8, dtype=np.uint8)
        self.center = torch.from_numpy(self.center_np.copy()).permute(2, 0, 1)[None].float().to(device)

    def __call__(self, v):
        assert tuple(v.shape) == tuple(self.center.shape), (tuple(v.shape), tuple(self.center.shape))
        return (self.center + (v - self.center).clamp(-self.eps, self.eps)).clamp(0, 255)

    def stats(self, raw, proj):
        """Diagnostics (no grad): fraction of raw values outside the ball (their gradient is 0), fraction of projected
        values sitting at the bound (|x - c| >= eps - 1e-3), raw L-inf."""
        with torch.no_grad():
            d_raw = (raw.detach() - self.center).abs()
            d_proj = (proj.detach() - self.center).abs()
            return dict(frac_outside=float((d_raw > self.eps).float().mean()),
                        frac_at_bound=float((d_proj >= self.eps - 1e-3).float().mean()),
                        raw_linf=float(d_raw.max()))

    def project_uint8(self, img):
        """The same projection on an integer HxWx3 image (c, eps integers: exact, no rounding)."""
        c = self.center_np.astype(np.int16)
        out = np.clip(np.clip(np.asarray(img).astype(np.int16), c - self.eps, c + self.eps), 0, 255).astype(np.uint8)
        return out


def bound_metrics(clean, adv, eps):
    """At-bound fractions of a delivered uint8 image vs the rounded clean (whole photo)."""
    d = np.abs(adv.astype(np.int16) - clean.astype(np.int16))
    return dict(linf=int(d.max()), frac_values_at_bound=float((d == eps).mean()),
                frac_pixels_any_channel_at_bound=float((d == eps).any(-1).mean()),
                frac_values_changed=float((d > 0).mean()))


class OurLoss:
    """The replacement of the classifier attack loss: Ours' self term on the 336 view of the decoded image
    (with --eps: on its projection onto the L-inf ball, see LinfBall)."""

    def __init__(self, objective, ball=None):
        self.objective = objective
        self.ball = ball
        self.trace = []
        self.outer_backend = None

    @staticmethod
    def view(decoded):
        image = (decoded / 2 + 0.5).clamp(0, 1) * 255
        return F.interpolate(image, (VIEW, VIEW), mode='bicubic', align_corners=False, antialias=True).clamp(0, 255)

    def clip(self, image):
        """Ours' self term of a [1, 3, 336, 336] pixel image, forward and backward under Ours' backend."""
        return _UnderClipBackend.apply(image, self.objective)

    def __call__(self, decoded):
        assert tuple(decoded.shape) == (1, 3, W.RUN_RES, W.RUN_RES), tuple(decoded.shape)
        outer = backend_now()
        assert self.outer_backend in (None, outer), (self.outer_backend, outer)
        self.outer_backend = outer
        if self.ball is None:
            loss = self.clip(self.view(decoded))
            assert backend_now() == outer, 'backend flags leaked'
            self.trace.append(dict(call=len(self.trace) + 1, self_loss=float(loss.detach())))
            return loss
        raw = self.view(decoded)
        proj = self.ball(raw)
        loss = self.clip(proj)
        assert backend_now() == outer, 'backend flags leaked'
        self.trace.append(dict(call=len(self.trace) + 1, self_loss=float(loss.detach()), **self.ball.stats(raw, proj)))
        return loss


def swapped_ast(path):
    """The stock diffattack() FunctionDef parsed from its file, with exactly the classifier-loss statements swapped.
    Returns (FunctionDef, swap record)."""
    path = Path(path)
    tree = ast.parse(path.read_text(), filename=str(path))
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'diffattack']
    assert len(fns) == 1, len(fns)
    fn = fns[0]
    loops = [n for n in fn.body if isinstance(n, ast.For) and ast.unparse(n.iter) == 'enumerate(pbar)']
    assert len(loops) == 1, len(loops)
    body = loops[0].body
    i_pred = [i for i, s in enumerate(body) if ast.unparse(s) == STOCK_PRED]
    i_loss = [i for i, s in enumerate(body) if ast.unparse(s) == STOCK_LOSS]
    assert len(i_pred) == 1 and len(i_loss) == 1 and i_loss[0] == i_pred[0] + 1, (i_pred, i_loss)
    removed = [(body[i].lineno, body[i].end_lineno, ast.unparse(body[i])) for i in (i_pred[0], i_loss[0])]
    new = ast.parse(OUR_LOSS).body[0]
    ast.copy_location(new, body[i_loss[0]])
    body[i_pred[0]:i_loss[0] + 1] = [new]
    ast.fix_missing_locations(fn)
    swap = dict(file=str(path), function='diffattack', removed=[dict(lines=f'{a}-{b}', source=s) for a, b, s in removed],
                inserted=OUR_LOSS, decorator=[ast.unparse(d) for d in fn.decorator_list])
    return fn, swap


def ourloss_diffattack(module, ourloss):
    """Compile the swapped diffattack() (see the module docstring) in a copy of the stock module's globals. Returns
    (function, swap record). Every other statement, the decorator and the stock module globals (helpers, imports) are
    the stock ones; OURLOSS is the only added name."""
    path = Path(module.__file__)
    fn, swap = swapped_ast(path)
    namespace = dict(vars(module))
    namespace['OURLOSS'] = ourloss
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['diffattack'], swap


class GenerationDone(Exception):
    """Raised by the model_transfer stub: main.py finished generation ('Adv acc:' printed) and would now start the
    stock transfer/FID evaluation, where the stock wrapper terminated it."""


def stop_before_transfer_eval(*args, **kwargs):
    raise GenerationDone()


class Tee(io.TextIOBase):
    """stdout -> the task log AND a line list (the stock wrapper read main.py's output through a pipe)."""

    def __init__(self, *streams):
        self.streams, self.lines, self._tail = streams, [], ''

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
        self._tail += s
        *done, self._tail = self._tail.split('\n')
        self.lines.extend(done)
        return len(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def run(root, setting, album, photo, device='cuda', eps=None, psnr=None):
    started = time.monotonic()
    plan = json.loads((root / 'PLAN.json').read_text())
    assert Path(plan['code_root']) == HERE, (plan['code_root'], HERE)
    assert plan.get('eps_levels') == eps, ('--eps differs from the PLAN', eps, plan.get('eps_levels'))
    assert plan.get('psnr_db') == psnr, ('--psnr differs from the PLAN', psnr, plan.get('psnr_db'))
    assert eps is None or psnr is None, '--eps and --psnr are exclusive'
    arm = (ARM if eps is None and psnr is None else f'{ARM}_eps{eps}' if eps is not None else f'{ARM}_psnr{psnr:g}')
    assert plan.get('arm') == arm, (plan.get('arm'), arm)
    assert Path.cwd().resolve() == REPO, ('run with cwd = the stock repo (as the stock wrapper ran main.py)', Path.cwd())
    entry = photo_entry(plan, setting, album, photo)
    source = Path(entry['source'])
    assert baselines.sha(source) == entry['source_sha256'], (source, 'source photo changed since the build')
    run_seed = plan['run_seed']
    seed = image_seed(run_seed, entry['album_index'])
    assert entry['output_name'].endswith('.png'), entry
    final = root / 'chunks' / setting / album / photo
    assert not os.path.lexists(final), f'{final} exists: a finished photo is never regenerated'

    # ---- provenance checks (the stock wrapper's) ----
    method = plan['method']
    commit = W.git('rev-parse', 'HEAD')
    assert commit == method['commit'] and commit.startswith(W.EXPECTED_COMMIT_PREFIX), (commit, method['commit'])
    dirty = W.git('status', '--porcelain')
    assert not dirty, f'stock repo has local changes, refusing to run:\n{dirty}'
    defaults = W.stock_defaults()
    bad = {k: (defaults.get(k), v) for k, v in W.EXPECTED_STOCK_DEFAULTS.items() if defaults.get(k) != v}
    assert not bad, f'stock main.py defaults differ from expected: {bad}'
    assert defaults == method['stock_defaults'], 'stock defaults differ from the build'
    assert baselines.sha(WRAPPER_FILE) == method['stock_wrapper']['sha256'], 'stock wrapper changed since the build'
    problems = inception_problems(method['inception_weights']) + sd_binding_problems(method['sd_weights'])
    assert not problems, ('stock model weights differ from the build', problems)

    # ---- private folders: the delivered tree (renamed into place at the end) and the stock side files ----
    final.parent.mkdir(parents=True, exist_ok=True)
    for d in ('partial', 'work'):
        (root / d / setting / album).mkdir(parents=True, exist_ok=True)
    partial = Path(tempfile.mkdtemp(prefix=f'{photo}.', dir=root / 'partial' / setting / album))
    work = Path(tempfile.mkdtemp(prefix=f'{photo}.', dir=root / 'work' / setting / album))
    assert not any(c in str(work) for c in '*?['), f'glob metacharacters in path (stock main.py globs it): {work}'
    stage_imgs, label_path, save_dir, log_path = work / 'images', work / 'labels.txt', work / 'stock_out', work / 'stock_main.log'
    stage_imgs.mkdir()
    save_dir.mkdir()

    # ---- stage the input (the rounded clean 336 PNG = the stock run's input) ----
    rounded = stage_clean(source)
    staged = stage_imgs / entry['output_name']
    Image.fromarray(rounded).save(staged, format='PNG')
    src = np.array(Image.open(staged).convert('RGB'))
    assert src.shape == (W.TARGET_RES, W.TARGET_RES, 3) and np.array_equal(src, rounded), staged
    from natsort import natsorted, ns
    ordered = natsorted(glob.glob(os.path.join(stage_imgs, '*')), alg=ns.PATH)     # exact ordering code of main.py
    assert ordered == [str(staged)], ordered

    # ---- label: the stock wrapper's rule, at the run resolution ----
    os.environ['TORCH_HOME'] = W.TORCH_HOME
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    t0 = time.monotonic()
    label_rows, surrogate_weights = W.compute_labels(ordered, W.RUN_RES)
    label_secs = time.monotonic() - t0
    assert surrogate_weights['sha256'] == method['inception_weights']['sha256'], surrogate_weights
    label_path.write_text(''.join(f"{r['label_file_value']}\n" for r in label_rows))
    label = label_rows[0]['label_0based']

    # ---- our loss: Yo'LLaVA's vision tower (hard gate) and the photo's clean features ----
    t0 = time.monotonic()
    encoder, encoder_record = load_encoder_eager(device)
    want = plan['encoder']
    for key in ('model_id', 'snapshot', 'config_sha256', 'weights_file'):
        assert encoder_record[key] == want[key], (key, encoder_record[key], want[key])
    assert stat_binding(want['weights_file']) == want['weights_stat'], 'CLIP weights file changed since the build'
    encoder_record.update(weights_sha256_at_build=want['weights_sha256'], weights_stat=want['weights_stat'])
    # PBFACE: instance face term (pbface.py): target = THIS photo's clean buffalo_l embedding; weight asserted
    wtag = plan['weight_tag']
    face_weight = PB.check_weight(wtag, plan['face_weight'])
    face_matrix = PB.cached_matrix(plan, album, photo, source, entry['source_sha256'])
    with clip_backend():
        face_model, _face_bl = PB.load_recogniser(device)
        pbf_face = PB.InstanceFace(source, face_matrix, face_model, device)
        face_self_cos = pbf_face.self_check()
    # SHARE-MATCHED (the face weight is scaled to each method's own loss size, so the face term has the same share for
    # every baseline): the backend returns self + w * face and the method multiplies it by its
    # own adversarial weight a, so the total is a * self + (a * w) * face (+ its other terms): face weight scaled to the
    # method's own loss weight. w = 0.125 or 1/28 exactly as for the noise baselines (where a = 1).
    face_outer = float(defaults['attack_loss_weight'])
    assert face_outer == 10.0, face_outer
    objective = PB.FaceSelf(baselines.SelfObjective(encoder, encoder_record), pbf_face, face_weight, wtag)
    face_total_weight = face_outer * objective.weight
    assert objective.face_weight == objective.weight == face_weight == PB.WEIGHTS[wtag] and objective.outer == 1.0
    assert abs(face_total_weight - dict(w125=1.25, w28=10.0 * (1 / 28))[wtag]) < 1e-15, (face_total_weight, wtag)
    assert objective.face.source == str(source)
    begin_clean(objective, attack.read_image(source, device))
    ball = (LinfBall(rounded, eps, device) if eps is not None else L2Ball(rounded, psnr, device) if psnr is not None
            else None)
    ourloss = OurLoss(objective, ball)
    encoder_secs = time.monotonic() - t0

    # ---- the stock modules, with exactly the classifier attack loss swapped ----
    import diff_latent_attack     # noqa: E402  (the repo's: REPO is sys.path[0])
    import other_attacks          # noqa: E402
    for m in (diff_latent_attack, other_attacks):
        assert Path(m.__file__).resolve().parent == REPO, m.__file__
    patched, swap = ourloss_diffattack(diff_latent_attack, ourloss)
    diff_latent_attack.diffattack = patched
    diff_latent_attack.encoder = seeded_vae_encoder(diff_latent_attack, seed)        # image seed: VAE sample seed
    main_code = seeded_main_code(REPO / 'main.py', seed)                             # image seed: seed_torch(seed)
    other_attacks.model_transfer = stop_before_transfer_eval
    argv = [str(REPO / 'main.py'), '--res', str(W.RUN_RES), '--images_root', str(stage_imgs),
            '--label_path', str(label_path), '--save_dir', str(save_dir), *MAIN_EXTRA_ARGS]
    overrides = dict(zip(MAIN_EXTRA_ARGS[::2], MAIN_EXTRA_ARGS[1::2]))
    iterations = int(overrides.get('--iterations', defaults['iterations']))

    # ---- run the authors' main.py, verbatim ----
    print('RUN:', ' '.join(argv), flush=True)
    t_start = time.monotonic()
    gen_done = False
    old_argv = sys.argv
    with open(log_path, 'w') as log:
        tee = Tee(sys.stdout, log)
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(tee):
                runpy._run_module_code(main_code, None, '__main__', pkg_name='', script_name=str(REPO / 'main.py'))   # = run_path's call
        except GenerationDone:
            gen_done = True
        finally:
            sys.argv = old_argv
            tee.flush()
    stock_secs = time.monotonic() - t_start
    done_at = [i for i, line in enumerate(tee.lines) if W.GEN_DONE_PREFIX in line]
    assert gen_done and len(done_at) == 1, ('main.py did not finish generation', gen_done, done_at)
    gen_lines = tee.lines[:done_at[0] + 1]
    clean_chk, adv_acc, after_pred, stock_dist = W.parse_stock_log(gen_lines)
    assert len(clean_chk) == len(adv_acc) == len(after_pred) == len(stock_dist) == 1, (
        len(clean_chk), len(adv_acc), len(after_pred), len(stock_dist))
    assert clean_chk[0]['pred'] == label and clean_chk[0]['gt'] == label, ('stock clean check disagrees with the label',
                                                                            clean_chk[0], label)
    assert len(ourloss.trace) == iterations, ('OURLOSS calls != latent iterations', len(ourloss.trace), iterations)
    face_residual = PB.check_calls(objective.calls, objective.weight, pbf_face.active, expect_steps=iterations)   # PBFACE
    assert ourloss.outer_backend == STOCK_BACKEND, ('the stock loop did not run under the stock flags', ourloss.outer_backend)

    # ---- collect: the stock wrapper's 352 -> 336 output step ----
    origin = save_dir / '0000_originImage.png'
    adv_p = save_dir / '0000_adv_image.png'
    assert np.array_equal(np.array(Image.open(origin).convert('RGB')), src), 'order mismatch at 0'
    adv = Image.open(adv_p).convert('RGB')
    assert adv.size == (W.RUN_RES, W.RUN_RES), adv.size
    out_img = adv.resize((W.TARGET_RES, W.TARGET_RES), resample=Image.BICUBIC)
    raw_out, search = None, None
    if ball is not None:                  # the same projection as in the loop, on the stock delivered 336 image
        raw_out = np.array(out_img)
        assert np.array_equal(ball.center_np, src), 'ball centre != the rounded clean'
        if psnr is None:
            out_img = Image.fromarray(ball.project_uint8(raw_out))
        else:
            projected, search = ball.project_uint8(raw_out)
            out_img = Image.fromarray(projected)
    folder = partial / METHOD
    folder.mkdir()
    dst = folder / entry['output_name']
    out_img.save(dst, format='PNG')
    with Image.open(dst) as im:
        assert im.format == 'PNG' and im.mode == 'RGB' and im.size == (VIEW, VIEW), (dst, im.format, im.mode, im.size)
        out = np.array(im)
    m = W.metrics(src, out)
    if eps is not None:
        linf_check = int(np.abs(out.astype(np.int16) - src.astype(np.int16)).max())
        assert linf_check <= eps and m['linf'] <= eps, ('L-inf bound violated on the delivered PNG', linf_check, m['linf'], eps)
    clean_up = Image.fromarray(src).resize((W.RUN_RES, W.RUN_RES), resample=Image.LANCZOS)
    native = W.metrics(np.array(clean_up), np.array(adv))
    floor = W.metrics(src, np.array(clean_up.resize((W.TARGET_RES, W.TARGET_RES), resample=Image.BICUBIC)))
    if psnr is not None:
        psnr_check = psnr_value(src, out)
        assert psnr_check is not None and psnr_check >= psnr and m['psnr'] >= psnr and psnr_check == search['psnr'], (
            'PSNR bound violated on the delivered PNG', psnr_check, m['psnr'], psnr, search)
    with torch.no_grad(), clip_backend():
        delivered = float(objective(torch.from_numpy(out).permute(2, 0, 1)[None].float().to(device)))
    bound = None
    if ball is not None:
        with torch.no_grad(), clip_backend():
            raw_loss = float(objective(torch.from_numpy(raw_out).permute(2, 0, 1)[None].float().to(device)))
        raw_m = W.metrics(src, raw_out)
        bound = dict(
            psnr_db=psnr, center='rounded clean 336 photo (round(attack.read_image(source)))', radius=ball.radius,
            n_values=ball.n, space='336 delivered pixel space; in-loop on the antialiased-bicubic 336 view of the decode, '
                                   'delivered on the stock PIL-bicubic 336 resize of the stock 352 PNG',
            projection='d = x - c; d *= min(1, r / ||d||_2), r = sqrt(N 255^2 / 10^(P/10)); (c + d).clamp(0, 255); plain '
                       'autograd; saved: same projection + budget-filling rounding to PSNR >= P (L2Ball.project_uint8)',
            delivered=dict(search, psnr=psnr_check, linf=m['linf'], self_loss=delivered,
                           frac_values_changed=float((out != src).mean())),
            raw_unprojected=dict(linf=raw_m['linf'], psnr=psnr_value(src, raw_out), self_loss=raw_loss),
            loop_first=ourloss.trace[0], loop_last=ourloss.trace[-1],
            loop_min_self_loss=min(r['self_loss'] for r in ourloss.trace),
            psnr_assert=f'PSNR >= {psnr} (whole 336 photo vs the rounded clean) asserted on the re-read PNG') if psnr is not None else dict(eps_levels=eps, eps=f'{eps}/255', center='rounded clean 336 photo (round(attack.read_image(source)))',
                     space='336 delivered pixel space; in-loop on the antialiased-bicubic 336 view of the decode, '
                           'delivered on the stock PIL-bicubic 336 resize of the stock 352 PNG',
                     projection='x_proj = (c + (x - c).clamp(-eps, eps)).clamp(0, 255); plain autograd',
                     delivered=dict(bound_metrics(src, out, eps), psnr=m['psnr'], self_loss=delivered),
                     raw_unprojected=dict(linf=raw_m['linf'], psnr=raw_m['psnr'], self_loss=raw_loss),
                     loop_first=ourloss.trace[0], loop_last=ourloss.trace[-1],
                     loop_min_self_loss=min(r['self_loss'] for r in ourloss.trace),
                     linf_assert=f'max |x_adv - c| <= {eps} asserted on the re-read PNG')
    with (folder / f'{photo}_loss.csv').open('w', newline='') as stream:
        weight = defaults['attack_loss_weight']
        names = ['call', 'self_loss', 'attack_loss'] + ([] if ball is None else list(ball.STAT_KEYS))
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(dict(r, attack_loss=r['self_loss'] * weight) for r in ourloss.trace)

    node = socket.gethostname().split('.')[0]
    code = {str(p): baselines.sha(p) for p in (HERE / 'generate.py', HERE.parent / 'pbface.py', R / 'baselines.py',
                                                R / 'code_snapshot/anti_persona/attack.py', WRAPPER_FILE)}
    code.update({str(REPO / f): baselines.sha(REPO / f) for f in STOCK_MODULES})
    seconds = time.monotonic() - started
    rule = dict(method['rule'], main_argv=argv[1:], main_extra_args=list(MAIN_EXTRA_ARGS), iterations=iterations,
                loss_swap=swap)
    manifest = dict(
        status='complete', arm=ARM, method=METHOD, queue=plan['queue'], setting=setting, album=album, photo=photo,
        source=str(source), source_sha256=entry['source_sha256'], ordinal=entry['ordinal'], inputs=[str(source)],
        clean='round(attack.read_image(source)) 336x336 PNG = the input DiffAttack got (the stock run\'s staged input); '
              'loss reference = ClipFeatures(attack.read_image(source)) (float)',
        output=str(final / METHOD / entry['output_name']), output_sha256=baselines.sha(dst),
        protected=bool(m['linf'] > 0), rule=rule,
        loss=dict(objective.record(), function='anti_persona.attack.clip_objective (self term only; R/baselines.SelfObjective)',
                  replaces=swap['removed'], multiplier=dict(name='args.attack_loss_weight', value=weight),
                  view='init_out_image [1,3,352,352] in [-1,1] -> (x/2+0.5).clamp(0,1)*255 -> '
                       'F.interpolate(336, bicubic, align_corners=False, antialias=True) -> clamp(0,255)',
                  clean_reference='ClipFeatures(attack.read_image(source))', calls=len(ourloss.trace),
                  backend=dict(clip_forward_and_backward=CLIP_BACKEND, stock_loop=ourloss.outer_backend,
                               note='cudnn/MIOpen flags: CLIP as attack.seed_everything sets them for Ours; SD and '
                                    'Inception keep the stock seed_torch(42) flags')),
        encoder=encoder_record,
        image_seed=seed, run_seed=run_seed, album_index=entry['album_index'],
        image_seed_rule='run_seed + album_index (Ours\' rule)',
        seed_override=dict(main_seed_torch=dict(stock=ARM_SEED, used=seed, how='AST: main.py module-level seed_torch(42) '
                                                 'constant swapped, then executed as __main__ via runpy._run_module_code'),
                           vae_sample_generator=dict(stock=STOCK_VAE_SEED, used=seed, how='AST: diff_latent_attack.encoder '
                                                     'torch.Generator().manual_seed(8888) constant swapped; module attribute replaced')),
        label=dict(label_rows[0], rule=method['rule']['label_rule'], seconds=label_secs), surrogate_weights=surrogate_weights,
        stock_clean_check=clean_chk[0], stock_native_float_adv_pred=after_pred[0],
        stock_native_float_surrogate_fooled=after_pred[0] != label, stock_printed_distances_352=stock_dist[0],
        stock_adv_png=str(adv_p), stock_adv_sha256=baselines.sha(adv_p), stock_log=str(log_path), work=str(work),
        loss_first=ourloss.trace[0]['self_loss'], loss_last_iteration=ourloss.trace[-1]['self_loss'],
        loss_delivered_png=delivered,
        linf_int_vs_rounded_clean=m['linf'], psnr_vs_rounded_clean=m['psnr'],
        native_352_vs_lanczos_clean=native, resize_only_roundtrip=floor,
        env=W.env_info(), sd_weights=dict(W.sd_snapshot(defaults['pretrained_diffusion_path']),
                                          bound_at_build=method['sd_weights']['snapshot'], binding_checked=True),
        code_sha256=code, hostname=node, gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        hip_visible_devices=os.environ.get('HIP_VISIBLE_DEVICES'), slot=os.environ.get('SLURM_JOB_ID'), device=device,
        seconds=seconds, stock_main_seconds=stock_secs, encoder_setup_seconds=encoder_secs,
        field_notes=dict(seconds='whole generator wall time for this photo (monotonic), incl. model loads',
                         stock_main_seconds='runpy of main.py: SD load, DDIM inversion, null-text optimisation, 30 latent '
                                            'iterations, decode, stock saves',
                         loss_first='self term (cosine to the clean features, mean over 25 entries) at the first latent '
                                    'iteration; attack_loss = 10 x self_loss',
                         loss_delivered_png='self term of the delivered 336 PNG',
                         protected='True when the delivered PNG differs from the rounded clean (L-inf > 0)'),
    )
    if bound is not None:              # --eps only: the unbounded manifest stays exactly the unbounded generator's
        manifest.update(arm=arm, rule=dict(rule, **({'eps_levels': eps} if eps is not None else {'psnr_db': psnr})), bound=bound)
    manifest.update(weight_tag=wtag, face_weight=face_weight, face_active=bool(pbf_face.active), face_matrix=face_matrix,   # PBFACE
                    face_target='clean embedding of the SAME photo', face_target_self_cosine=face_self_cos,
                    face_raw_first=objective.calls[0]['face_raw'], face_raw_last=objective.calls[-1]['face_raw'],
                    self_raw_first=objective.calls[0]['self_raw'], self_raw_last=objective.calls[-1]['self_raw'],
                    terms_residual_max=face_residual, face_rows=plan['face_rows'], face_rows_sha256=plan['face_rows_sha256'],
                    face_outer_multiplier=face_outer, face_internal_coefficient=objective.weight, face_total_weight=face_total_weight,
                    face_weight_rule='share-matched: face weight scaled to the method own loss weight',
                    loss_note='OURLOSS = self + face_weight * face; x attack_loss_weight 10 as published -> 10 self + (10 face_weight) face')
    put(folder / 'manifest.json', manifest)
    put(partial / 'GROUP.json', dict(status='complete', setting=setting, album=album, photo=photo, arms=[METHOD],
                                     hostname=node, seconds=time.monotonic() - started))
    os.rename(partial, final)
    summary = dict(photo=f'{setting}/{album}/{photo}', label=label_rows[0]['class_name'], linf=m['linf'],
                   psnr=m['psnr'], loss_first=ourloss.trace[0]['self_loss'],
                   loss_last=ourloss.trace[-1]['self_loss'], delivered=delivered, seconds=round(seconds, 1))
    if bound is not None and psnr is not None:
        summary.update(psnr_db=psnr, psnr_delivered=bound['delivered']['psnr'], scale_final=bound['delivered']['scale_final'],
                       raw_loss=bound['raw_unprojected']['self_loss'], raw_psnr=bound['raw_unprojected']['psnr'])
    elif bound is not None:
        summary.update(eps=eps, at_bound=bound['delivered']['frac_values_at_bound'], raw_loss=bound['raw_unprojected']['self_loss'],
                       raw_linf=bound['raw_unprojected']['linf'], loop_frac_outside_last=ourloss.trace[-1]['frac_outside'])
    print(json.dumps(summary), flush=True)
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--setting', required=True)
    p.add_argument('--album', required=True)
    p.add_argument('--photo', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--eps', type=int, default=None, help='L-inf bound in pixel levels (E/255), applied inside the '
                   'optimisation; omitted = the unbounded generator, unchanged')
    p.add_argument('--psnr', type=float, default=None, help='PSNR bound in dB (L2 ball), applied inside the '
                   'optimisation; exclusive with --eps')
    a = p.parse_args()
    node = socket.gethostname().split('.')[0]
    assert not node.endswith(EXCLUDED), node
    assert os.environ.get('HF_HUB_OFFLINE') == '1' and os.environ.get('TRANSFORMERS_OFFLINE') == '1', 'offline HF env'
    out = run(a.root.resolve(), a.setting, a.album, a.photo, a.device, a.eps, a.psnr)
    print('DONE', out, flush=True)


if __name__ == '__main__':
    main()
