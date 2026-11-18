# REFERENCE COPY (anti-persona release): the generator that produced the paper's WDP (stock wrapper) images, kept for its exact
# procedure and settings. Only absolute paths were replaced by environment variables; the logic is unchanged.
# It runs inside the research harness it was written for (a queue PLAN.json, the legacy baseline module and the
# stock wrapper); see baselines/README.md, section "DiffAttack and WDP".
#!/usr/bin/env python
"""Stock WDP baseline generator for 336x336 photos.

WDP = "Enhancing Facial Privacy Protection via Weakening Diffusion Purification" (Salar et al., CVPR 2025),
official code https://github.com/parham1998/Facial-Privacy-Protection @ bd0a8e4, used UNEDITED and imported
from the fresh clone. Every protection setting is taken from the repo's own main.parse_args() defaults.

The only adaptation is image I/O. For each 336x336 photo:
  1. Detect and align the face exactly as the repo's align.py does (dlib frontal detector + 68 landmarks, the
     first detection, image_align(..., output_size=256) with its defaults). The repo's image_align function is
     called as-is.
  2. Protect the 256x256 aligned face with the repo's Adversarial_Opt.run(), using the stock defaults.
  3. Map the protected face back into the photo by inverting the alignment transform. The aligned square is a
     similarity transform of the FFHQ "quad", which we recompute with the same arithmetic. Pixels inside the
     quad footprint come from the protected face; all other pixels stay clean.
If dlib finds no face, or the stock MTCNN finds no face on the aligned crop (the stock code would crash), the
photo is copied through unchanged and the reason is recorded.

Usage: python gen_wdp.py --input-dir DIR --output-dir DIR [--seed 0] [--work-dir DIR]
"""
import argparse
import contextlib
import functools
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True  # never drop __pycache__ into the clean clone or the asset dirs

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get('WDP_DIR', os.path.join(HERE, '..', '..', 'third_party', 'WDP'))   # parham1998/Facial-Privacy-Protection
REPO_URL = 'https://github.com/parham1998/Facial-Privacy-Protection'
REPO_COMMIT = 'bd0a8e4'
# AMT-GAN face-recognition assets: byte-identical (CRC32) to assets/models/ in AMT-GAN's asset.zip
FPP_LOCAL = os.environ.get('FPP_ASSETS_DIR', REPO)   # a checkout holding assets/face_recognition_models + temp/dlib .dat
FR_DIR = FPP_LOCAL + '/assets/face_recognition_models'
FR_FILES = ['ir152.pth', 'irse50.pth', 'facenet.pth', 'mobile_face.pth', 'ir152.py', 'irse.py', 'facenet.py']
# align.py downloads dlib.net's shape_predictor_68_face_landmarks.dat into ./temp/ (dlib.net is blocked here)
DLIB_DAT = FPP_LOCAL + '/temp/shape_predictor_68_face_landmarks.dat'
DLIB_DAT_SHA256 = 'fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f'
# Stock model id 'stabilityai/stable-diffusion-2-base' is gone from HF (HTTP 401). The Manojb mirror at this
# revision has unet/vae/text_encoder sha256s identical to two other independent mirrors (PeggyWang, luongphamit).
SD_STOCK_ID = 'stabilityai/stable-diffusion-2-base'
SD_MIRROR_ID = 'Manojb/stable-diffusion-2-base'
SD_MIRROR_REV = '64bf7b4f10eee35494b38d55c06c0c78cf8b44d0'
SD_SNAPSHOT = os.path.expanduser(
    '~/.cache/huggingface/hub/models--Manojb--stable-diffusion-2-base/snapshots/' + SD_MIRROR_REV)
# diffusers 0.9.0 (stock pin) cannot read .safetensors, so the exact upstream .bin files are fetched to here
SD_LOCAL = os.path.join(HERE, 'cache', 'sd2base_bin')
SD_EXPECTED_SHA256 = {  # HF LFS oids of the mirror revision above
    'unet/diffusion_pytorch_model.bin': '048b92a2df00b2c57e8499d0cb871c100406fd966f16e02dd305895c9c879ac7',
    'vae/diffusion_pytorch_model.bin': 'a4302e1efa25f3a47ceb7536bc335715ad9d1f203e90c2d25507600d74006e89',
    'text_encoder/model.safetensors': 'cce6febb0b6d876ee5eb24af35e27e764eb4f9b1d0b7c026c8c3333d4cfc916c',
}
SD_LINKED = ['model_index.json', 'README.md', 'scheduler/scheduler_config.json', 'tokenizer/merges.txt',
             'tokenizer/special_tokens_map.json', 'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json',
             'feature_extractor/preprocessor_config.json', 'text_encoder/config.json',
             'text_encoder/model.safetensors', 'unet/config.json', 'vae/config.json']
# Stock NCELoss loads CLIP ViT-B/32 at init (it is only used when is_makeup=True; the default is False). The
# exact OpenAI file is kept here; clip's own sha256 check still runs.
CLIP_ROOT = os.path.join(HERE, 'cache', 'clip')
CLIP_SHA256 = '40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af'
RUNTIME = os.path.join(HERE, 'runtime')  # cwd for the stock code (it uses relative 'assets/...' paths)
STOCK_SEED = 10  # main.py hard-codes initialize_seed(seed=10)
IMG_SIZE = 336
ENV_KEYS = ['python', 'torch', 'hip', 'cuda', 'device', 'diffusers', 'transformers', 'dlib', 'numpy', 'opencv',
            'facenet_pytorch', 'python_executable']  # must match for resume


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(chunk), b''):
            h.update(b)
    return h.hexdigest()


def symlink(src, dst):
    if os.path.islink(dst) and os.readlink(dst) == src:
        return
    if os.path.lexists(dst):
        raise RuntimeError(f'{dst} exists and is not a link to {src}')
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    os.symlink(src, dst)


def inside_dir(path, root):
    p, r = os.path.realpath(path), os.path.realpath(root)
    return p == r or p.startswith(r + os.sep)


def validate_paths(in_dir, out_dir, work):
    if not os.path.isdir(in_dir):
        raise RuntimeError(f'input dir {in_dir} not found')
    for p in (out_dir, work):
        if inside_dir(p, in_dir):
            raise RuntimeError(f'{p} is (inside) the input dir {in_dir}')
        for prot in (REPO, FPP_LOCAL):
            if inside_dir(p, prot):
                raise RuntimeError(f'{p} is inside the protected repo {prot}')
    if inside_dir(in_dir, work):
        raise RuntimeError(f'input dir {in_dir} is inside the work dir {work}')
    GUARD_IN_DIR[0] = in_dir


GUARD_IN_DIR = [None]  # set by validate_paths


def check_dest(path, src=None):
    """Refuse to write through a link, onto the source, or (after resolving every symlinked parent) into the
    input dir or either protected repo. Called on every path the wrapper or the stock code writes."""
    if os.path.islink(path):
        raise RuntimeError(f'{path} is a symlink; refusing to write through it')
    if os.path.isfile(path) and os.stat(path).st_nlink > 1:
        raise RuntimeError(f'{path} is hard-linked elsewhere; refusing to overwrite it')
    if src is not None and os.path.exists(path) and os.path.samefile(path, src):
        raise RuntimeError(f'{path} is the same file as the input {src}')
    for root in [r for r in (GUARD_IN_DIR[0], REPO, FPP_LOCAL) if r]:
        if inside_dir(path, root):
            raise RuntimeError(f'{path} resolves inside {root}')


def check_repo():
    head = subprocess.check_output(['git', '-C', REPO, 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', REPO, 'status', '--porcelain', '--untracked-files=all'],
                                    text=True).strip()
    if not head.startswith(REPO_COMMIT):
        raise RuntimeError(f'WDP clone at {head}, expected {REPO_COMMIT}')
    if dirty:
        raise RuntimeError(f'WDP clone is not clean:\n{dirty}')
    return head


def setup_runtime():
    """Build the run directory from symlinks only. Nothing is copied into or changed in the repo."""
    for sub in ['target_images', 'test_images', 'obfs_target_images']:
        symlink(os.path.join(REPO, 'assets', sub), os.path.join(RUNTIME, 'assets', sub))
    symlink(FR_DIR, os.path.join(RUNTIME, 'assets', 'face_recognition_models'))
    symlink(DLIB_DAT, os.path.join(RUNTIME, 'temp', 'shape_predictor_68_face_landmarks.dat'))
    for rel in SD_LINKED:
        symlink(os.path.realpath(os.path.join(SD_SNAPSHOT, rel)), os.path.join(SD_LOCAL, rel))


def verify_assets():
    """Measure the sha256 of every model/config/code asset that gets loaded; fail on any expected-hash mismatch."""
    measured = {'sd': {}, 'fr': {}, 'dlib': None, 'clip': None}
    for rel in SD_LINKED + ['unet/diffusion_pytorch_model.bin', 'vae/diffusion_pytorch_model.bin']:
        p = os.path.join(SD_LOCAL, rel)
        if not os.path.isfile(p):
            raise RuntimeError(f'missing {p}')
        measured['sd'][rel] = sha256_file(p)
    for rel, sha in SD_EXPECTED_SHA256.items():
        if measured['sd'][rel] != sha:
            raise RuntimeError(f'{rel}: sha256 {measured["sd"][rel]} != expected {sha}')
    for f in FR_FILES:
        measured['fr'][f] = sha256_file(os.path.join(FR_DIR, f))
    import facenet_pytorch
    mt_dir = os.path.join(os.path.dirname(facenet_pytorch.__file__), 'data')
    measured['mtcnn'] = {f: sha256_file(os.path.join(mt_dir, f)) for f in ('pnet.pt', 'rnet.pt', 'onet.pt')}
    measured['mtcnn_dir'] = mt_dir
    measured['dlib'] = sha256_file(DLIB_DAT)
    if measured['dlib'] != DLIB_DAT_SHA256:
        raise RuntimeError('dlib landmark model sha256 mismatch')
    measured['clip'] = sha256_file(os.path.join(CLIP_ROOT, 'ViT-B-32.pt'))
    if measured['clip'] != CLIP_SHA256:
        raise RuntimeError('CLIP ViT-B-32.pt sha256 mismatch')
    return measured


def ffhq_quad(face_landmarks):
    """The 'Choose oriented crop rectangle' block of the stock align.image_align, copied verbatim. Returns
    the quad (NW, SW, SE, NE) in original-photo pixel coordinates, and qsize."""
    import numpy as np
    lm = np.array(face_landmarks)
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    return quad, float(qsize)


def stock_frame(quad, qsize, W, H, output_size=256):
    """Mirror the stock 'Shrink' step: when shrink > 1 the photo is resized to rint(size / shrink) and the quad is
    divided by shrink. Returns the quad in that frame plus the per-axis scale from original to that frame.
    Crop and pad are integer translations applied to both, so they cancel."""
    import numpy as np
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rw, rh = int(np.rint(float(W) / shrink)), int(np.rint(float(H) / shrink))
        return quad / shrink, (rw / W, rh / H), shrink
    return quad, (1.0, 1.0), shrink


def paste_back(clean, face, quad, scale=(1.0, 1.0)):
    """Inverse of the stock alignment. PIL QUAD (data = quad + 0.5) followed by the LANCZOS resize puts aligned
    pixel k at fraction s = (k + 0.5) / N along the quad edges, so a point P' in the alignment frame maps to
    P' = NW + s (NE - NW) + t (SW - NW). Photo pixel P maps to P' = (P + 0.5) * scale - 0.5 (the identity unless
    the stock shrink step ran). That relation is inverted here and the face is sampled with Lanczos-4 (best
    clean round trip among linear/cubic/lanczos/area+linear on the smoke photos). The mask is the quad
    footprint inside the photo, a hard edge with no blending."""
    import cv2
    import numpy as np
    H, W = clean.shape[:2]
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
    n = face.shape[0]
    mapx = (s * n - 0.5).astype(np.float32)
    mapy = (t * n - 0.5).astype(np.float32)
    warped = cv2.remap(face, mapx, mapy, interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
    out = clean.copy()
    out[inside] = warped[inside]
    return out, inside


def psnr(a, b, mask=None):
    """RGB PSNR on uint8 (peak 255). None when the images are identical (infinite PSNR) or the mask is empty."""
    import numpy as np
    d = a.astype(np.float64) - b.astype(np.float64)
    if mask is not None:
        d = d[mask]
    if d.size == 0:
        return None
    mse = float(np.mean(d ** 2))
    return None if mse == 0 else round(float(10 * np.log10(255.0 ** 2 / mse)), 3)


def parse_losses(log_text):
    adv = [float(v) for v in re.findall(r'^adv_loss:\s+([-\d.eE+]+)', log_text, re.M)]
    sa = [float(v) for v in re.findall(r'^self_attn_loss:\s+([-\d.eE+]+)', log_text, re.M)]
    return {'n_prot_steps_logged': len(adv),
            'adv_loss_first': adv[0] if adv else None, 'adv_loss_last': adv[-1] if adv else None,
            'self_attn_loss_first': sa[0] if sa else None, 'self_attn_loss_last': sa[-1] if sa else None}


def rng_fingerprint():
    """Hash of the global RNG states other than torch CPU (all GPUs, numpy, python). The torch CPU state is
    checked separately because the stock DataLoader iterator draws its base_seed from it."""
    import numpy as np
    import torch
    h = hashlib.sha256()
    for s in torch.cuda.get_rng_state_all():
        h.update(s.cpu().numpy().tobytes())
    st = np.random.get_state()
    h.update(st[1].tobytes() + repr(st[2:]).encode())
    h.update(repr(random.getstate()).encode())
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + '.tmp'
    check_dest(path)
    check_dest(tmp)
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir', required=True, help='336x336 clean PNGs')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--seed', type=int, default=0,
                    help=f'passed to the stock initialize_seed (main.py hard-codes {STOCK_SEED}). Each image records '
                         'whether any global RNG was drawn during protection')
    ap.add_argument('--work-dir', default=None,
                    help='aligned and protected 256 faces plus stock logs (default <output-dir>/_wdp_work)')
    cli = ap.parse_args()
    in_dir, out_dir = os.path.abspath(cli.input_dir), os.path.abspath(cli.output_dir)
    work = os.path.abspath(cli.work_dir or os.path.join(out_dir, '_wdp_work'))

    # ---- preflight: before any write ----
    validate_paths(in_dir, out_dir, work)
    head = check_repo()
    from PIL import Image
    names = sorted(f for f in os.listdir(in_dir) if f.lower().endswith('.png'))
    if not names:
        raise RuntimeError(f'no PNGs in {in_dir}')
    bad = [(n, Image.open(os.path.join(in_dir, n)).size) for n in names]
    bad = [(n, s) for n, s in bad if s != (IMG_SIZE, IMG_SIZE)]
    if bad:
        raise RuntimeError(f'inputs must be {IMG_SIZE}x{IMG_SIZE}: {bad[:5]}')
    safes = [re.sub(r'[^A-Za-z0-9_-]', '_', os.path.splitext(n)[0]) for n in names]
    if len(set(safes)) != len(safes):
        raise RuntimeError('basenames collide after sanitising for the stock loader')
    wrapper_sha = sha256_file(os.path.abspath(__file__))

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(work, exist_ok=True)
    setup_runtime()
    measured = verify_assets()
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    os.chdir(RUNTIME)
    sys.path.insert(0, RUNTIME)  # for 'assets.face_recognition_models' (namespace package)
    sys.path.insert(0, REPO)     # stock code

    import numpy as np
    import cv2
    import torch
    from torch.utils.data import DataLoader
    from torchvision.transforms import transforms
    import clip  # vendored in the repo
    # I/O only: point clip's downloader at our cache dir. It still checks the URL's sha256.
    clip.load = functools.partial(clip.load, download_root=CLIP_ROOT)
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    wdp_main = importlib.import_module('main')          # stock main.py (guarded by __main__)
    import utils as wdp_utils                            # stock utils (builds the MTCNN at import)
    from dataset import ImageDataset
    from adversarial_optimization import Adversarial_Opt
    from align import LandmarksDetector, image_align
    for mod in (wdp_main, wdp_utils, sys.modules['dataset'], sys.modules['adversarial_optimization'],
                sys.modules['align'], sys.modules['attention_control'], sys.modules['criteria.cosine_loss'], clip):
        if not inside_dir(mod.__file__, REPO):
            raise RuntimeError(f'{mod.__name__} imported from {mod.__file__}, not the clone')
    for m in ('irse', 'ir152', 'facenet'):
        f = sys.modules['assets.face_recognition_models.' + m].__file__
        if not inside_dir(f, FR_DIR):
            raise RuntimeError(f'FR model code {m} imported from {f}, not {FR_DIR}')

    # ---- stock settings: the repo's own argparse defaults, untouched ----
    saved_argv = sys.argv
    sys.argv = ['main.py']
    args = wdp_main.parse_args()
    sys.argv = saved_argv
    stock_args = {k: v for k, v in vars(args).items()}
    wdp_main.initialize_seed(seed=cli.seed)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device.type != 'cuda':
        raise RuntimeError('no GPU visible')

    diff_model = StableDiffusionPipeline.from_pretrained(SD_LOCAL, local_files_only=True).to(args.device)
    diff_model.scheduler = DDIMScheduler.from_config(diff_model.scheduler.config)
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    args.source_dir = ''
    args.dataloader = None
    opt = Adversarial_Opt(args, diff_model)
    landmarks_detector = LandmarksDetector(os.path.join(RUNTIME, 'temp', 'shape_predictor_68_face_landmarks.dat'))

    target_rel = {'1': 'assets/target_images/005869.jpg', '2': 'assets/target_images/085807.jpg',
                  '3': 'assets/target_images/116481.jpg', '4': 'assets/target_images/169284.jpg',
                  '5': 'assets/obfs_target_images/0808002.png'}[args.target_choice]
    target_crop = {'1': '[168:912, 205:765]', '2': '[187:891, 244:764]', '3': '[214:955, 188:773]',
                   '4': '[173:925, 233:792]'}.get(args.target_choice) if args.MTCNN_cropping else None
    import diffusers, transformers, dlib
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'hip': torch.version.hip,
           'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(0),
           'diffusers': diffusers.__version__, 'transformers': transformers.__version__,
           'dlib': dlib.__version__, 'numpy': np.__version__, 'opencv': cv2.__version__,
           'facenet_pytorch': importlib.metadata.version('facenet-pytorch'),
           'python_executable': sys.executable, 'host': platform.node(),
           'HIP_VISIBLE_DEVICES': os.environ.get('HIP_VISIBLE_DEVICES'),
           'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'slurm_job': os.environ.get('SLURM_JOB_ID')}
    manifest = {
        'method': 'WDP',
        'paper': 'Salar et al., Enhancing Facial Privacy Protection via Weakening Diffusion Purification, CVPR 2025',
        'repo': {'path': REPO, 'url': REPO_URL, 'commit': head, 'worktree_clean_before_run': True,
                 'code_imported_unmodified': ['main.parse_args', 'main.initialize_seed', 'dataset.ImageDataset',
                                              'adversarial_optimization.Adversarial_Opt (run)',
                                              'align.LandmarksDetector', 'align.image_align', 'utils (MTCNN, FR loaders)']},
        'wrapper': {'path': os.path.abspath(__file__), 'sha256': wrapper_sha},
        'stock_settings': {
            'argparse_defaults': stock_args,
            'mode': 'impersonation' if not args.is_obfuscation else 'obfuscation',
            'surrogate_FR_ensemble': args.surrogate_model_names,
            'held_out_test_model': args.test_model_name,
            'diffusion_model': SD_STOCK_ID,
            'scheduler': 'DDIMScheduler.from_config(pipeline scheduler config), eta=0',
            'latent_optimizer': 'AdamW lr=1e-2 on the DDIM-inverted latent at start_step',
            'null_text_optimizer': 'AdamW lr=1e-1, MSE to inversion trajectory',
            'loss': 'adv_optim_weight * sum_k (1 - cos(FR_k(crop(x_adv)), FR_k(target))) + MSE(self-attn adv, '
                    'self-attn clean) [weight 1]; makeup/CLIP loss off',
            'FR_input': 'MTCNN box of the 256 aligned image (detected on the clean aligned PNG), bilinear resize to '
                        '112 (ir152, irse50) / 160 (facenet)',
            'vae_latent_sampling': 'latent_dist.sample(generator seeded 8888)',
            'pixel_budget': 'none (the method has no epsilon)',
            'stock_seed_in_main_py': STOCK_SEED,
            'stock_alignment': 'align.py: dlib frontal detector (upsample 1) + 68-landmark predictor, first '
                               'detection kept, image_align(output_size=256, transform_size=4096, enable_padding=True)',
        },
        'seed_used': cli.seed,
        'target_image': {'choice': args.target_choice, 'path': os.path.realpath(target_rel),
                         'sha256': sha256_file(target_rel), 'stock_crop_rows_cols': target_crop,
                         'origin': 'AMT-GAN target identity (CelebA-HQ), shipped in the WDP repo'},
        'assets': {
            'sd_weights': {'stock_id': SD_STOCK_ID, 'stock_id_status': 'removed from HF (HTTP 401, 2026-11-06)',
                           'used': f'{SD_MIRROR_ID}@{SD_MIRROR_REV}', 'local_dir': SD_LOCAL,
                           'measured_sha256': measured['sd'],
                           'note': 'sha256 of unet/vae/text_encoder identical across Manojb, PeggyWang and '
                                   'luongphamit mirrors; .bin used because diffusers 0.9.0 cannot read safetensors; '
                                   'text_encoder loaded from model.safetensors by transformers'},
            'face_recognition_models': {'dir': FR_DIR, 'note': 'byte-identical (CRC32) to AMT-GAN asset.zip '
                                        'assets/models/*', 'measured_sha256': measured['fr']},
            'dlib_landmarks': {'path': DLIB_DAT, 'measured_sha256': measured['dlib']},
            'mtcnn': {'dir': measured['mtcnn_dir'], 'measured_sha256': measured['mtcnn'],
                      'note': 'facenet-pytorch bundled weights used by the stock utils.MTCNN'},
            'clip_vit_b32': {'path': os.path.join(CLIP_ROOT, 'ViT-B-32.pt'), 'measured_sha256': measured['clip'],
                             'note': 'exact OpenAI file; loaded by stock NCELoss init; unused when is_makeup=False'},
        },
        'io_adaptation': {
            'input': '336x336 photo -> stock align.py alignment -> 256x256 aligned face (stock image_size)',
            'output': 'protected 256 face mapped back by the inverse similarity transform of the FFHQ quad; pixels '
                      'inside the quad footprint are replaced (hard mask, no blending); outside pixels stay clean',
            'resampling_back': 'cv2.remap INTER_LANCZOS4 at the exact inverse coordinates (no pre-resize)',
            'no_face': 'dlib no detection, or stock MTCNN no detection on the aligned crop -> photo copied clean',
            'not_run': 'stock main.py also runs its mobile_face PSR evaluation (tests.attack_local_models); that is '
                       'evaluation, not generation, so it is skipped',
            'roundtrip_psnr_region': 'PSNR inside the footprint of clean photo vs (clean aligned face pasted back): '
                                     'error from the alignment/paste resampling alone, without protection',
            'psnr_null': 'psnr is null when output == input (infinite PSNR)',
        },
        'env': env,
        'input_dir': in_dir, 'output_dir': out_dir, 'work_dir': work,
        'images': {}, 'status': 'running',
    }
    man_path = os.path.join(out_dir, 'manifest.json')
    old = {}
    if os.path.isfile(man_path):  # resume only from an identical wrapper / commit / seed / asset set
        try:
            prev = json.load(open(man_path))
            if prev.get('wrapper', {}).get('sha256') == wrapper_sha and prev.get('repo', {}).get('commit') == head \
                    and prev.get('seed_used') == cli.seed and prev.get('assets') == json.loads(json.dumps(manifest['assets'])) \
                    and {k: prev.get('env', {}).get(k) for k in ENV_KEYS} == {k: env.get(k) for k in ENV_KEYS}:
                old = prev.get('images', {})
        except Exception:
            old = {}
    write_json(man_path, manifest)

    print(f'[gen_wdp] {len(names)} images, seed {cli.seed}, device {env["device"]}', flush=True)
    for k, (name, safe) in enumerate(zip(names, safes)):
        src = os.path.join(in_dir, name)
        dst = os.path.join(out_dir, name)
        src_sha = sha256_file(src)
        e_old = old.get(name)
        if e_old and e_old.get('source_sha256') == src_sha and os.path.isfile(dst) \
                and sha256_file(dst) == e_old.get('output_sha256'):
            manifest['images'][name] = e_old  # keeps its own host/device/time fields
            print(f'[gen_wdp] {k + 1}/{len(names)} {name}: reused', flush=True)
            continue
        t0 = time.time()
        clean = np.array(Image.open(src).convert('RGB'))
        H, W = clean.shape[:2]
        iw = os.path.join(work, safe)  # stock run() rebuilds '<source_dir>/<name>.png' from split('.')[0]
        al_dir = os.path.join(iw, 'aligned')
        check_dest(dst, src)
        for d in (iw, al_dir):
            check_dest(d)
        os.makedirs(al_dir, exist_ok=True)
        entry = {'source_sha256': src_sha, 'env': env,
                 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

        faces = list(landmarks_detector.get_landmarks(src))
        entry['dlib_n_faces'] = len(faces)
        status = None
        if not faces:
            status = 'no_face_dlib'
        else:
            lm = faces[0]  # stock work_landmark keeps only the first detection (later ones hit 'exists')
            al_path = os.path.join(al_dir, safe + '.png')
            check_dest(al_path)
            if os.path.exists(al_path):
                os.remove(al_path)
            image_align(src, al_path, lm, output_size=256)
            aligned_clean = np.array(Image.open(al_path).convert('RGB'))
            if aligned_clean.shape != (256, 256, 3):
                raise RuntimeError(f'{name}: stock alignment gave {aligned_clean.shape}')
            quad, qsize = ffhq_quad(lm)
            fquad, scale, shrink = stock_frame(quad, qsize, W, H)
            entry.update({'quad_nw_sw_se_ne': quad.round(3).tolist(), 'qsize': round(qsize, 3),
                          'stock_shrink': shrink})
            rt, inside = paste_back(clean, aligned_clean, fquad, scale)
            entry['roundtrip_psnr_region'] = psnr(rt, clean, inside)
            try:  # the stock run() calls utils.alignment(Image.open(path)) -> boxes[0]; None would crash it
                bb = wdp_utils.alignment(Image.open(al_path))
                entry['mtcnn_box_aligned'] = [round(float(v), 2) for v in bb]
            except TypeError:
                status = 'no_face_mtcnn_aligned'
        if status is None:
            stock_out = os.path.join(iw, 'stock_out')
            prot_path = os.path.join(stock_out, args.test_model_name[0], args.target_choice, safe + '.png')
            log_path = os.path.join(iw, 'stock_stdout.log')
            for d in (stock_out, os.path.dirname(os.path.dirname(prot_path)), os.path.dirname(prot_path),
                      prot_path, os.path.join(os.path.dirname(prot_path), 'time.txt'), log_path):
                check_dest(d)
            if os.path.exists(prot_path):
                os.remove(prot_path)
            opt.source_dir = al_dir
            opt.protected_image_dir = stock_out
            opt.dataloader = DataLoader(ImageDataset(al_dir, transform), batch_size=1, shuffle=False)
            rng_before, cpu_before = rng_fingerprint(), torch.get_rng_state()
            t1 = time.time()
            with open(log_path, 'w') as lf, contextlib.redirect_stdout(lf), contextlib.redirect_stderr(lf):
                opt.run()
            entry['stock_run_seconds'] = round(time.time() - t1, 2)
            # Expected CPU state = before + only the DataLoader iterator's base_seed draw (unused, num_workers=0)
            cpu_after = torch.get_rng_state()
            torch.set_rng_state(cpu_before)
            iter(DataLoader(ImageDataset(al_dir, transform), batch_size=1, shuffle=False))
            cpu_expect = torch.get_rng_state()
            torch.set_rng_state(cpu_after)
            entry['global_rng_untouched_by_protection'] = bool(rng_fingerprint() == rng_before
                                                               and torch.equal(cpu_after, cpu_expect))
            entry.update(parse_losses(open(log_path).read()))
            raw = cv2.imread(prot_path) if os.path.isfile(prot_path) else None
            if raw is None or raw.shape != (256, 256, 3):
                raise RuntimeError(f'{name}: stock run produced no valid 256x256 face at {prot_path}')
            prot = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            out, inside = paste_back(clean, prot, fquad, scale)
            Image.fromarray(out).save(dst)
            diff = np.abs(out.astype(np.int16) - clean.astype(np.int16))
            entry.update({'status': 'protected', 'face_found': True,
                          'region_frac': round(float(inside.mean()), 5),
                          'changed_frac': round(float((diff.max(-1) > 0).mean()), 5),
                          'linf': int(diff.max()), 'psnr': psnr(out, clean),
                          'psnr_region': psnr(out, clean, inside),
                          'aligned_psnr_vs_clean_aligned': psnr(prot, aligned_clean)})
        else:
            shutil.copyfile(src, dst)
            entry.update({'status': status, 'face_found': status != 'no_face_dlib', 'kept_clean': True,
                          'region_frac': 0.0, 'changed_frac': 0.0, 'linf': 0, 'psnr': None})
        chk = np.array(Image.open(dst).convert('RGB'))
        if chk.shape != (IMG_SIZE, IMG_SIZE, 3):
            raise RuntimeError(f'{name}: output has shape {chk.shape}')
        entry['output_sha256'] = sha256_file(dst)
        entry['seconds'] = round(time.time() - t0, 2)
        manifest['images'][name] = entry
        write_json(man_path, manifest)
        print(f'[gen_wdp] {k + 1}/{len(names)} {name}: {entry["status"]} linf={entry["linf"]} '
              f'psnr={entry["psnr"]} changed={entry["changed_frac"]} {entry["seconds"]}s', flush=True)

    ims = list(manifest['images'].values())
    prot = [e for e in ims if e['status'] == 'protected']
    mean = lambda v: round(float(np.mean(v)), 5) if v else None
    manifest['summary'] = {
        'n_images': len(ims), 'n_protected': len(prot),
        'n_kept_clean': len(ims) - len(prot),
        'kept_clean': {n: e['status'] for n, e in manifest['images'].items() if e['status'] != 'protected'},
        'mean_changed_frac_all': mean([e['changed_frac'] for e in ims]),
        'mean_changed_frac_protected': mean([e['changed_frac'] for e in prot]),
        'mean_psnr_protected': mean([e['psnr'] for e in prot if e['psnr'] is not None]),
        'max_linf': max([e['linf'] for e in ims], default=None),
        'mean_seconds_protected': mean([e['seconds'] for e in prot]),
        'all_global_rng_untouched': all(e.get('global_rng_untouched_by_protection', True) for e in prot),
    }
    check_repo()  # raises if anything (e.g. __pycache__) was written into the clone during the run
    manifest['repo']['worktree_clean_after_run'] = True
    manifest['status'] = 'complete'
    write_json(man_path, manifest)
    print('[gen_wdp] complete', json.dumps(manifest['summary']), flush=True)


if __name__ == '__main__':
    main()
