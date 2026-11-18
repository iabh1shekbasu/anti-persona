# REFERENCE COPY (anti-persona release): the generator that produced the paper's DiffAttack (stock wrapper) images, kept for its exact
# procedure and settings. Only absolute paths were replaced by environment variables; the logic is unchanged.
# It runs inside the research harness it was written for (a queue PLAN.json, the legacy baseline module and the
# stock wrapper); see baselines/README.md, section "DiffAttack and WDP".
#!/usr/bin/env python
"""Stock DiffAttack baseline generator (Chen et al., "Diffusion Models for Imperceptible
and Transferable Adversarial Attack", TPAMI).

Runs the authors' released main.py UNMODIFIED (fresh clone, commit e7094d0) as a
subprocess with its own defaults. The only things this wrapper supplies are:
  (1) image I/O at our 336x336 resolution:
        stock main.py asserts `res % 32 == 0`, so 336 cannot run (336 % 32 = 16; the
        attention aggregation at res//32 would also mismatch the 11x11 UNet maps).
        The two nearest supported values tie (320, 352); we use 352 so the stock code
        only ever UP-samples our input (its own Lanczos resize, 336->352) and never
        discards input pixels before the attack. The stock 352x352 adversarial PNG is
        then PIL-bicubic resized to 336x336.
  (2) the ImageNet label the method needs (face photos have none):
        label = top-1 of the stock surrogate (other_attacks.model_selection("inception"),
        i.e. torchvision inception_v3(pretrained=True)) on the clean image, preprocessed
        exactly as diff_latent_attack.diffattack() does its clean check (Lanczos resize
        to --res, ImageNet mean/std). Written 1-indexed into the label file main.py reads.
  (3) no pixel budget: DiffAttack is unrestricted, kept as published.

Everything else (loss, weights, diffusion_steps, start_step, iterations, guidance,
surrogate, SD checkpoint, masks) is the stock default and is not passed on the command
line. The stock post-generation transfer/FID evaluation (other_attacks.model_transfer,
evaluation only, needs tf2torch .npy weights the release does not ship) is stopped once
main.py prints its end-of-generation "Adv acc:" line; adversarial PNGs are already on disk
by then and are not affected.

Usage:
  python gen_diffattack.py --input-dir DIR --output-dir DIR [--seed INT]
Run with the `diffattack` env python, one visible GPU (HIP_VISIBLE_DEVICES=k).
"""
import argparse
import ast
import glob
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.realpath(os.environ.get("DIFFATTACK_DIR", os.path.join(HERE, "..", "..", "third_party", "DiffAttack")))
EXPECTED_COMMIT_PREFIX = "e7094d0"
TARGET_RES = 336
RUN_RES = 352
TORCH_HOME = os.path.join(HERE, "torch_home")  # torchvision Inception-v3 weights cached here
GEN_DONE_PREFIX = "Adv acc:"  # printed by stock main.py right after the generation loop

# Stock defaults we rely on (verified against main.py's argparse via ast before running).
EXPECTED_STOCK_DEFAULTS = {
    "diffusion_steps": 20, "start_step": 15, "iterations": 30, "res": 224,
    "model_name": "inception", "dataset_name": "imagenet_compatible",
    "guidance": 2.5, "attack_loss_weight": 10, "cross_attn_loss_weight": 10000,
    "self_attn_loss_weight": 100, "is_apply_mask": False, "is_hard_mask": False,
    "is_test": False, "pretrained_diffusion_path": "Manojb/stable-diffusion-2-base",
}
OVERRIDDEN = ("res", "images_root", "label_path", "save_dir")
_STATE = {"manifest": None, "path": None, "done": False}  # so an unexpected crash marks the manifest "failed"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def stock_defaults():
    """Read every add_argument default from the stock main.py without executing it."""
    tree = ast.parse(open(os.path.join(REPO, "main.py")).read())
    out = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument" and node.args):
            name = ast.literal_eval(node.args[0]).lstrip("-")
            for kw in node.keywords:
                if kw.arg == "default":
                    out[name] = ast.literal_eval(kw.value)
    return out


def git(*args):
    return subprocess.check_output(["git", "--no-optional-locks", "-C", REPO, *args], text=True).strip()


def compute_labels(staged, res):
    """Stock-surrogate top-1 on each clean image, preprocessed exactly as diffattack()."""
    sys.dont_write_bytecode = True
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    import numpy as np
    import torch
    from PIL import Image
    import other_attacks  # stock module (unmodified)
    from dataset_caption import imagenet_label  # stock module (unmodified)
    import random

    # ---- verbatim body of stock main.py seed_torch(42): same backend flags as the stock clean check
    # (this ROCm torch build defaults to cudnn.benchmark=True / deterministic=False) ----
    seed = 42
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    classifier = other_attacks.model_selection("inception").eval()
    classifier.requires_grad_(False)
    rows = []
    with torch.no_grad():
        for p in staged:
            image = Image.open(p).convert("RGB")  # as main.py
            # ---- verbatim from diff_latent_attack.diffattack() clean check ----
            test_image = image.resize((res, res), resample=Image.LANCZOS)
            test_image = np.float32(test_image) / 255.0
            test_image = test_image[:, :, :3]
            test_image[:, :, ] -= (np.float32(0.485), np.float32(0.456), np.float32(0.406))
            test_image[:, :, ] /= (np.float32(0.229), np.float32(0.224), np.float32(0.225))
            test_image = test_image.transpose((2, 0, 1))
            test_image = torch.from_numpy(test_image).unsqueeze(0)
            pred = classifier(test_image.cuda())
            # -------------------------------------------------------------------
            prob = torch.softmax(pred, 1)[0]
            top = int(torch.argmax(pred, 1).item())
            rows.append({"label_0based": top, "label_file_value": top + 1,
                         "class_name": imagenet_label.refined_Label[top],
                         "full_imagenet_name": imagenet_label.Label[top],
                         "surrogate_clean_prob": float(prob[top].item())})
    del classifier
    torch.cuda.empty_cache()
    wpath = os.path.join(torch.hub.get_dir(), "checkpoints", "inception_v3_google-0cc3c7bd.pth")
    weights = {"file": wpath, "sha256": sha256_file(wpath) if os.path.exists(wpath) else None}
    return rows, weights


def env_info():
    import diffusers, numpy, PIL, torch, torchvision, transformers
    return {"python": sys.executable, "python_version": sys.version.split()[0],
            "torch": torch.__version__, "hip": torch.version.hip, "torchvision": torchvision.__version__,
            "diffusers": diffusers.__version__, "transformers": transformers.__version__,
            "numpy": numpy.__version__, "pillow": PIL.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "ROCR_VISIBLE_DEVICES": os.environ.get("ROCR_VISIBLE_DEVICES"),
            "host": os.uname().nodename}


def sd_snapshot(repo_id):
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        d = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"))
        return {"repo_id": repo_id, "snapshot": open(os.path.join(d, "refs", "main")).read().strip()}
    except Exception as e:  # informational only
        return {"repo_id": repo_id, "snapshot": None, "error": repr(e)}


def metrics(a, b):
    import numpy as np
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    # identical images -> psnr reported as None (JSON has no inf); excluded from mean_psnr
    return {"linf": int(np.abs(a - b).max()),
            "psnr": float(10.0 * np.log10(255.0 ** 2 / mse)) if mse > 0 else None}


def parse_stock_log(lines):
    """Per-image stock prints, in image order."""
    import re
    clean, adv_acc, after, dist = [], [], [], []
    for ln in lines:
        m = re.search(r"gt_label:\s*(\d+)\s+pred_label:\s*(\d+)\s+pred_clean_logit\s+([0-9.eE+-]+)", ln)
        if m:
            clean.append({"gt": int(m.group(1)), "pred": int(m.group(2)), "prob": float(m.group(3))})
        m = re.search(r"Accuracy on adversarial examples:\s*([0-9.]+)%", ln)
        if m:
            adv_acc.append(float(m.group(1)))
        m = re.search(r"after_pred:\s*tensor\(\[(\d+)\]", ln)
        if m:
            after.append(int(m.group(1)))
        num = r"\[?\s*([0-9.eE+-]+)\s*\]?"  # stock prints 1-element numpy arrays, [0,1] units
        m = re.search(r"L1:\s*" + num + r"\s+L2:\s*" + num + r"\s+Linf:\s*" + num, ln)
        if m:
            dist.append({"L1": float(m.group(1)), "L2": float(m.group(2)), "Linf": float(m.group(3))})
    return clean, adv_acc, after, dist


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input-dir", required=True, help="336x336 clean PNGs (lossless)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=0,
                    help="recorded only: stock main.py hard-codes seed_torch(42) and the VAE "
                         "posterior sample uses a fixed generator (8888); there is no stock seed flag")
    args = ap.parse_args()
    t_wrapper = time.time()

    from natsort import natsorted, ns
    from PIL import Image
    import numpy as np

    in_dir = os.path.realpath(args.input_dir)
    out_dir = os.path.realpath(args.output_dir)
    for d in (in_dir, out_dir):
        if any(c in d for c in "*?["):
            sys.exit(f"glob metacharacters in path (stock main.py globs it): {d}")
    if out_dir == in_dir:
        sys.exit("--output-dir must differ from --input-dir")
    if out_dir == REPO or out_dir.startswith(REPO + os.sep):
        sys.exit("--output-dir must not be inside the stock repo")
    if os.path.exists(out_dir) and os.listdir(out_dir):
        sys.exit(f"--output-dir must be new or empty (no stale outputs/manifest): {out_dir}")
    work = os.path.join(out_dir, "_work")
    stage_imgs = os.path.join(work, "images")
    label_path = os.path.join(work, "labels.txt")
    save_dir = os.path.join(work, "stock_out")
    log_path = os.path.join(work, "stock_main.log")
    manifest_path = os.path.join(out_dir, "manifest.json")
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(work):
        shutil.rmtree(work)
    os.makedirs(stage_imgs)
    os.makedirs(save_dir)

    # ---- provenance checks ----
    commit = git("rev-parse", "HEAD")
    if not commit.startswith(EXPECTED_COMMIT_PREFIX):
        sys.exit(f"stock repo HEAD {commit} != expected {EXPECTED_COMMIT_PREFIX}")
    dirty = git("status", "--porcelain")
    if dirty:
        sys.exit(f"stock repo has local changes, refusing to run:\n{dirty}")
    defaults = stock_defaults()
    bad = {k: (defaults.get(k), v) for k, v in EXPECTED_STOCK_DEFAULTS.items() if defaults.get(k) != v}
    if bad:
        sys.exit(f"stock main.py defaults differ from expected: {bad}")

    # ---- stage inputs (only PNGs; stock main.py globs images_root/*) ----
    all_in = sorted(os.listdir(in_dir))
    pngs = [f for f in all_in if f.lower().endswith(".png") and os.path.isfile(os.path.join(in_dir, f))]
    ignored = [f for f in all_in if f not in pngs]
    if not pngs:
        sys.exit(f"no PNGs in {in_dir}")
    hidden = [f for f in pngs if f.startswith(".")]
    if hidden:  # stock glob("*") skips dot-files
        sys.exit(f"hidden input files not supported (stock glob skips them): {hidden}")
    src_sha = {}
    for f in pngs:
        im = Image.open(os.path.join(in_dir, f))
        if im.size != (TARGET_RES, TARGET_RES):
            sys.exit(f"{f}: size {im.size}, expected {TARGET_RES}x{TARGET_RES}")
        shutil.copyfile(os.path.join(in_dir, f), os.path.join(stage_imgs, f))
        src_sha[f] = sha256_file(os.path.join(in_dir, f))
        assert sha256_file(os.path.join(stage_imgs, f)) == src_sha[f]
    # exact ordering code of stock main.py
    ordered = natsorted(glob.glob(os.path.join(stage_imgs, "*")), alg=ns.PATH)
    order = [os.path.basename(p) for p in ordered]
    assert sorted(order) == sorted(pngs)

    # ---- labels: stock surrogate top-1 at the run resolution ----
    os.environ["TORCH_HOME"] = TORCH_HOME
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    t0 = time.time()
    label_rows, surrogate_weights = compute_labels(ordered, RUN_RES)
    label_secs = time.time() - t0
    with open(label_path, "w") as f:
        f.write("".join(f"{r['label_file_value']}\n" for r in label_rows))

    cmd = [sys.executable, os.path.join(REPO, "main.py"),
           "--res", str(RUN_RES), "--images_root", stage_imgs,
           "--label_path", label_path, "--save_dir", save_dir]
    sub_env = dict(os.environ)
    sub_env.update({"HF_HUB_OFFLINE": "1", "TORCH_HOME": TORCH_HOME, "PYTHONUNBUFFERED": "1",
                    "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1"})
    env_keys = ["HF_HUB_OFFLINE", "TORCH_HOME", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
                "PYTHONNOUSERSITE", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"]

    manifest = {
        "method": "DiffAttack (Chen et al., TPAMI 2024) -- stock released code",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper": {"path": os.path.abspath(__file__), "sha256": sha256_file(os.path.abspath(__file__))},
        "repo": {"path": REPO, "commit": commit, "local_changes": False, "entry": "main.py",
                 "url": "https://github.com/WindVChen/DiffAttack"},
        "env": env_info(),
        "command": cmd, "command_str": shlex.join(cmd), "cwd": REPO,
        "command_env": {k: sub_env.get(k) for k in env_keys},
        "stock_defaults_read_from_main_py": defaults,
        "overridden_args": {"res": RUN_RES, "images_root": stage_imgs,
                            "label_path": label_path, "save_dir": save_dir},
        "resolution": {
            "target": TARGET_RES, "run_res": RUN_RES,
            "why": "stock main.py asserts res % 32 == 0 and res >= 96; 336 % 32 = 16 so --res 336 is "
                   "rejected (also aggregate_attention uses res//32=10 vs the 11x11 UNet maps of a 42x42 "
                   "latent). 320 and 352 tie as nearest; 352 chosen so the stock code only up-samples "
                   "our input (its own Lanczos 336->352) and no input pixels are discarded before the attack.",
            "input_to_stock": "336x336 PNG as-is; stock main.py/diffattack() Lanczos-resize it to 352",
            "postprocess": f"stock {RUN_RES}x{RUN_RES} *_adv_image.png -> PIL Image.BICUBIC -> "
                           f"{TARGET_RES}x{TARGET_RES} PNG",
        },
        "pixel_budget": "none (DiffAttack is unrestricted, as published)",
        "seed": {"requested": args.seed, "used": False,
                 "note": "stock main.py hard-codes seed_torch(42); the VAE posterior sample uses a fixed "
                         "torch.Generator seed 8888; no stock seed flag, so --seed cannot be forwarded "
                         "without editing the code. Generation is deterministic given inputs."},
        "label_rule": "label = top-1 of the stock surrogate (other_attacks.model_selection('inception') = "
                      "torchvision inception_v3(pretrained=True), eval) on the clean image, preprocessed "
                      "exactly as diffattack()'s clean check (PIL Lanczos resize to run_res, ImageNet "
                      "mean/std). labels.txt holds label+1 (stock reads 1-indexed). The stock code also "
                      "uses imagenet_label.refined_Label[label] as the SD text prompt and as the token "
                      "whose cross-attention variance it minimises.",
        "surrogate_weights": surrogate_weights,
        "sd_weights": sd_snapshot(defaults["pretrained_diffusion_path"]),
        "label_seconds": label_secs,
        "labels": [dict(index=i, image=order[i], **r) for i, r in enumerate(label_rows)],
        "ignored_input_files": ignored,
        "stock_post_generation_eval": "other_attacks.model_transfer (15-classifier transfer test + FID) "
                                      "is evaluation, not generation, and needs 4 tf2torch .npy weights "
                                      "the release does not ship; the wrapper terminates main.py once it "
                                      f"prints '{GEN_DONE_PREFIX}' (end of the generation loop).",
        "external_state": "reads: HF hub cache (Manojb/stable-diffusion-2-base, HF_HUB_OFFLINE=1); "
                          "writes outside the output dir are confined to TORCH_HOME (gen/diffattack/torch_home) "
                          "except two accepted runtime side effects: ART's import-time ~/.art/config.json "
                          "(created only if absent; it already exists on this account) and the ROCm/MIOpen "
                          "kernel/find caches shared by every GPU job.",
        "images": {},
    }
    write_json(manifest_path, manifest)
    _STATE.update(manifest=manifest, path=manifest_path)

    # ---- run stock main.py ----
    print("RUN:", shlex.join(cmd), flush=True)
    t_start = time.time()
    gen_done = False
    gen_lines = []  # stock output up to and including the end-of-generation line
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=REPO, env=sub_env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                log.write(line); log.flush()
                gen_lines.append(line.rstrip("\n"))
                if "%|" not in line:
                    print("[stock]", line.rstrip(), flush=True)
                if GEN_DONE_PREFIX in line:
                    gen_done = True
                    break
            if gen_done:
                proc.terminate()  # our own child, by PID: stops the stock transfer/FID evaluation
                try:
                    rest, _ = proc.communicate(timeout=120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rest, _ = proc.communicate()
                log.write(rest or "")
            else:
                proc.wait()
        finally:  # never leave the stock child running if the wrapper itself fails
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    rc = proc.returncode
    t_total = time.time() - t_start
    manifest["stock_returncode"] = rc
    manifest["stock_wall_seconds"] = t_total
    manifest["stock_log"] = log_path
    if not gen_done:
        manifest["status"] = "failed"
        manifest["error"] = f"stock main.py exited rc={rc} before finishing generation; see {log_path}"
        write_json(manifest_path, manifest)
        sys.exit(manifest["error"])

    # ---- collect outputs ----
    clean_chk, adv_acc, after_pred, stock_dist = parse_stock_log(gen_lines)
    n = len(order)
    if not (len(clean_chk) == len(adv_acc) == len(after_pred) == len(stock_dist) == n):
        manifest["status"] = "failed"
        manifest["error"] = (f"log parse mismatch: n={n} clean={len(clean_chk)} adv_acc={len(adv_acc)} "
                             f"after={len(after_pred)} dist={len(stock_dist)}")
        write_json(manifest_path, manifest)
        sys.exit(manifest["error"])

    psnrs, linfs, secs = [], [], []
    for i, name in enumerate(order):
        origin = os.path.join(save_dir, f"{i:04d}_originImage.png")
        adv_p = os.path.join(save_dir, f"{i:04d}_adv_image.png")
        src = np.array(Image.open(os.path.join(in_dir, name)).convert("RGB"))
        # the stock code re-saves each input it reads: confirms index -> basename mapping
        assert np.array_equal(np.array(Image.open(origin).convert("RGB")), src), f"order mismatch at {i}"
        adv = Image.open(adv_p).convert("RGB")
        assert adv.size == (RUN_RES, RUN_RES), adv.size
        out_img = adv.resize((TARGET_RES, TARGET_RES), resample=Image.BICUBIC)
        dst = os.path.join(out_dir, name)
        out_img.save(dst, format="PNG")
        out = np.array(Image.open(dst).convert("RGB"))
        m = metrics(src, out)
        clean_up = Image.fromarray(src).resize((RUN_RES, RUN_RES), resample=Image.LANCZOS)
        native = metrics(np.array(clean_up), np.array(adv))
        floor = metrics(src, np.array(clean_up.resize((TARGET_RES, TARGET_RES), resample=Image.BICUBIC)))
        sec = os.stat(adv_p).st_mtime - os.stat(origin).st_mtime
        lab = label_rows[i]["label_0based"]
        manifest["images"][name] = {
            "stock_index": i, "source_sha256": src_sha[name], "output_sha256": sha256_file(dst),
            "stock_adv_png": adv_p, "stock_adv_sha256": sha256_file(adv_p),
            "linf": m["linf"], "psnr": m["psnr"], "seconds": sec,
            "label_0based": lab, "class_name": label_rows[i]["class_name"],
            "stock_clean_pred": clean_chk[i]["pred"], "stock_clean_matches_label": clean_chk[i]["pred"] == lab,
            "stock_gt_label": clean_chk[i]["gt"],
            "stock_native_float_adv_pred": after_pred[i],
            "stock_native_float_surrogate_fooled": after_pred[i] != lab,
            "native_352_vs_lanczos_clean": native,
            "stock_printed_distances_352": stock_dist[i],
            "resize_only_roundtrip": floor,
        }
        psnrs.append(m["psnr"]); linfs.append(m["linf"]); secs.append(sec)
        print(f"{name}: label={lab}({label_rows[i]['class_name']}) stock_native_adv_pred={after_pred[i]} "
              f"linf={m['linf']} psnr={m['psnr']} sec={sec:.1f}", flush=True)

    finite = [p for p in psnrs if p is not None]
    manifest["summary"] = {
        "n_images": n, "mean_psnr": float(np.mean(finite)) if finite else None,
        "n_identical_outputs": len(psnrs) - len(finite),
        "mean_linf": float(np.mean(linfs)), "max_linf": int(np.max(linfs)),
        "mean_seconds": float(np.mean(secs)),
        "stock_native_float_surrogate_fooled": int(sum(
            v["stock_native_float_surrogate_fooled"] for v in manifest["images"].values())),
        "stock_clean_matches_label": int(sum(v["stock_clean_matches_label"] for v in manifest["images"].values())),
    }
    manifest["field_notes"] = {
        "linf/psnr": "delivered 336x336 uint8 PNG vs source 336x336 uint8 PNG, RGB, 0-255",
        "seconds": "mtime(stock NNNN_adv_image.png) - mtime(stock NNNN_originImage.png): stock per-image "
                   "work (surrogate reload, DDIM inversion, uncond-embedding optimisation, 30 latent "
                   "iterations, decode); excludes one-time SD pipeline load, label pass, our resize. "
                   "stock_wall_seconds = main.py subprocess lifetime; wrapper_wall_seconds = whole wrapper.",
        "stock_native_float_*": "stock's own printout on the float 352x352 decode before PNG quantisation "
                                "and before our 336 resize; not an evaluation of the delivered PNG",
        "native_352_vs_lanczos_clean": "stock 352 adv PNG vs Lanczos-352 clean (the stock code's own reference)",
        "resize_only_roundtrip": "source -> Lanczos 352 -> bicubic 336 vs source: resize-only floor",
    }
    mism = [k for k, v in manifest["images"].items()
            if not (v["stock_clean_matches_label"] and v["stock_gt_label"] == v["label_0based"])]
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    if mism:
        manifest["status"] = "failed"
        manifest["error"] = f"stock clean check disagrees with supplied label for {mism}"
        write_json(manifest_path, manifest)
        sys.exit(manifest["error"])
    manifest["wrapper_wall_seconds"] = time.time() - t_wrapper
    final = dict(manifest, status="complete")
    write_json(manifest_path, final)
    _STATE["done"] = True
    manifest["status"] = "complete"
    print("DONE", json.dumps(manifest["summary"]), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        m = _STATE["manifest"]
        if m is not None and not _STATE["done"] and m.get("status") != "failed":
            m["status"] = "failed"
            m["error"] = f"{type(e).__name__}: {e}"
            write_json(_STATE["path"], m)
        raise
