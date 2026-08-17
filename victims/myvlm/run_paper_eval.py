####IJCB 

"""
run_paper_eval.py — recognition-probe eval for the IJCB-2026 paper outputs.

Adapted from run_probe_eval.py. Differences vs. the original:

  1. SWEEP_ROOT points at /path/to/YoLLaVA/ijcb2026/output_paper/, not the
     transfer-attack TESTS/sweep_eval tree.

  2. Variants are DISCOVERED per identity via glob, not hardcoded as
     (eps × recipe). The output_paper directory tree uses variant names
     like 'baseline_eps4', 'baseline_avg_eps8', 'relaxed_smooth_eps4',
     'relaxed_smooth_eps4_cleaner', 'relaxed_smooth_eps8_cleaner', which
     don't fit the EPS_LEVELS × RECIPES × concept layout.

  3. Each variant directory IS the leaf — there's no nested
     <concept>_<vname>/ subdir. Path is output_paper/<concept>/<variant>/.

  4. CSV outputs go to /path/to/MyVLM/captioning_outputs/_paper_results/
     so they don't collide with the existing _probe_results.

Everything else (captioning checkpoints under captioning_outputs/, probe
prompt, score_probe_json in 'protection' mode, final-iteration auto-detect,
resume support, per-run YAML write, subprocess inference) is identical to
run_probe_eval.py.

Usage:
  cd /path/to/MyVLM
  conda activate myvlmllava
  PYTHONPATH=. python run_paper_eval.py --gpu 0

  # Subset:
  PYTHONPATH=. python run_paper_eval.py --gpu 0 --concepts denisdang
  PYTHONPATH=. python run_paper_eval.py --gpu 0 --variants baseline_eps8

  # Force re-score (ignore cached details):
  PYTHONPATH=. python run_paper_eval.py --gpu 0 --force

  # Skip inference, just re-score existing JSONs:
  PYTHONPATH=. python run_paper_eval.py --gpu 0 --skip_inference
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_scoring import score_probe_json  # noqa: E402


# ----------------- CONFIG -----------------
CONCEPTS = [
    "willinvietnam", "khanhvy", "oong", "viruss", "thao",
    "denisdang", "thuytien", "yuheng", "ciin", "phuc-map",
]

REPO_ROOT   = Path(__file__).resolve().parent            # the vendored MyVLM dir
# Root holding the protected images, as <SWEEP_ROOT>/<concept>/<variant>/*.png.
# Override with the SWEEP_ROOT env var; defaults to <repo>/output_paper.
SWEEP_ROOT  = Path(os.environ.get("SWEEP_ROOT", REPO_ROOT / "output_paper"))
RESULTS_DIR = REPO_ROOT / "captioning_outputs/_paper_results"
LOGS_DIR    = RESULTS_DIR / "logs"
DETAIL_DIR  = RESULTS_DIR / "details"
CFG_OUT_DIR = REPO_ROOT / "example_configs/inference/_paper_generated"
SEED        = 42

# MyVLM's prompt.format() replaces {concept} with cfg.concept_identifier.
PROBE_PROMPT = ("Can you see if {concept} is in this photo? "
                "Answer with a single word or phrase.")

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# Subdirs under each <concept>/ that are NOT variant dirs and should be ignored
# during glob-based variant discovery. 'logs' shows up in the IJCB tree.
NON_VARIANT_DIRS = {"logs"}
# ------------------------------------------


# ----- Variant parsing ---------------------------------------------------
# A variant name like 'relaxed_smooth_eps4_cleaner' encodes (recipe, eps,
# suffix). Everything before _eps{N} is the recipe; everything after the
# eps token (if anything) is a suffix that's part of the recipe identity.
# We keep recipe+suffix together as a single 'recipe' string for CSV
# pivots, and pull eps out for per-eps grouping.
_EPS_RE = re.compile(r"_eps(\d+)(.*)$")


def parse_variant(vname: str):
    """
    Parse a variant directory name into (recipe, eps).

    Examples:
      'baseline_eps4'                -> ('baseline',                4)
      'baseline_avg_eps8'            -> ('baseline_avg',            8)
      'relaxed_smooth_eps4'          -> ('relaxed_smooth',          4)
      'relaxed_smooth_eps4_cleaner'  -> ('relaxed_smooth_cleaner',  4)
      'relaxed_smooth_eps8_cleaner'  -> ('relaxed_smooth_cleaner',  8)

    Returns (None, None) if the name doesn't contain an _eps{N} token —
    such directories are skipped at discovery time.
    """
    m = _EPS_RE.search(vname)
    if not m:
        return None, None
    eps = int(m.group(1))
    suffix = m.group(2).lstrip("_")           # 'cleaner' or ''
    prefix = vname[: m.start()].rstrip("_")   # 'relaxed_smooth' / 'baseline'
    if suffix:
        recipe = f"{prefix}_{suffix}"
    else:
        recipe = prefix
    return recipe, eps


def discover_variants(concept: str):
    """
    Return list of variant directory names under SWEEP_ROOT/<concept>/.
    Filters out non-variant dirs (logs, etc.) and dirs that lack _eps{N}.
    """
    base = SWEEP_ROOT / concept
    if not base.is_dir():
        return []
    vnames = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in NON_VARIANT_DIRS:
            continue
        recipe, eps = parse_variant(entry.name)
        if recipe is None:
            continue
        vnames.append(entry.name)
    return vnames


# ----- YAML / inference / paths ------------------------------------------

def build_yaml(concept: str, img_dir: Path) -> Path:
    CFG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = REPO_ROOT / f"captioning_outputs/{concept}/{concept}/seed_{SEED}"
    cfg_path = CFG_OUT_DIR / f"{concept}__{img_dir.name}.yaml"

    imgs = sorted(
        str(p) for p in img_dir.glob("*")
        if p.suffix.lower() in VALID_EXTS
    )
    if not imgs:
        raise RuntimeError(f"No images found in {img_dir}")
    images_yaml = "\n".join(f"  - {p}" for p in imgs)

    yaml_body = f"""concept_name: {concept}
concept_identifier: {concept}
concept_type: PERSON
vlm_type: LLAVA
personalization_task: CAPTIONING
image_paths:
{images_yaml}
checkpoint_path: {ckpt}
concept_head_path: ./models
seed: {SEED}
prompts:
  - "{PROBE_PROMPT}"
"""
    cfg_path.write_text(yaml_body)
    return cfg_path


def run_inference(cfg_path: Path, gpu: int, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "inference/run_myvlm_inference.py"),
        "--config_path", str(cfg_path),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"# cmd: {' '.join(cmd)}\n")
        f.flush()
        p = subprocess.run(cmd, cwd=REPO_ROOT, env=env,
                           stdout=f, stderr=subprocess.STDOUT)
    return p.returncode


def find_inference_json(concept: str) -> Path:
    return (REPO_ROOT /
            f"outputs/{concept}/inference_outputs/seed_{SEED}/"
            f"inference_outputs_llava_captioning.json")


# ----- main --------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--concepts", nargs="*", default=None,
                    help="Subset of concepts to evaluate. Default: all 10.")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="Filter variant directory names (e.g. baseline_eps8). "
                         "Default: all variants discovered per concept.")
    ap.add_argument("--skip_inference", action="store_true",
                    help="Reuse existing inference JSONs; only score+aggregate.")
    ap.add_argument("--force", action="store_true",
                    help="Don't skip cells that already have details on disk.")
    args = ap.parse_args()

    concepts = args.concepts or CONCEPTS
    variant_filter = set(args.variants) if args.variants else None

    for d in (RESULTS_DIR, LOGS_DIR, DETAIL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── Build the full job list up front so we can print the total count
    # ── and so missing-checkpoint / missing-image cells still appear in
    # ── the summary CSV.
    jobs = []   # list of (concept, vname, recipe, eps)
    for concept in concepts:
        for vname in discover_variants(concept):
            if variant_filter and vname not in variant_filter:
                continue
            recipe, eps = parse_variant(vname)
            jobs.append((concept, vname, recipe, eps))

    n_total = len(jobs)
    print(f"Paper eval: {n_total} runs on GPU {args.gpu}")
    print(f"Using CAPTIONING checkpoints under {REPO_ROOT}/captioning_outputs/")
    print(f"Sweep root:  {SWEEP_ROOT}")
    print(f"Logs:        {LOGS_DIR}")
    print(f"Details:     {DETAIL_DIR}")
    print(f"Results CSV: {RESULTS_DIR}")
    print()

    summary_rows = []
    run_start = time.time()
    done = 0

    for concept, vname, recipe, eps in jobs:
        done += 1
        tag = f"{concept}__{vname}"
        img_dir = SWEEP_ROOT / concept / vname
        detail_path = DETAIL_DIR / f"{tag}.json"

        print("=" * 92)
        print(f"[{done}/{n_total}] {tag}  images={img_dir}")
        print("=" * 92)

        # Resume — skip if details already on disk
        if detail_path.exists() and not args.force:
            try:
                cached = json.loads(detail_path.read_text())
                summary_rows.append({
                    "concept": concept, "eps": eps, "recipe": recipe,
                    "variant": vname,
                    "positives": cached["positives"],
                    "total":     cached["total"],
                    "score":     f"{cached['score']:.4f}",
                    "status":    "cached",
                })
                print(f"  [CACHED] score={cached['score']:.4f}  "
                      f"({cached['positives']}/{cached['total']})")
                continue
            except Exception:
                pass

        # Sanity: checkpoint must exist
        ckpt_dir = REPO_ROOT / f"captioning_outputs/{concept}/{concept}/seed_{SEED}"
        has_ckpt = ckpt_dir.is_dir() and any(ckpt_dir.glob("*.pt"))
        if not has_ckpt:
            print(f"  [SKIP] no captioning checkpoint at {ckpt_dir}")
            summary_rows.append({
                "concept": concept, "eps": eps, "recipe": recipe,
                "variant": vname, "positives": "", "total": "",
                "score": "", "status": "missing_checkpoint",
            })
            continue

        if not img_dir.is_dir():
            print(f"  [SKIP] missing image dir")
            summary_rows.append({
                "concept": concept, "eps": eps, "recipe": recipe,
                "variant": vname, "positives": "", "total": "",
                "score": "", "status": "missing_images",
            })
            continue

        # 1) YAML
        try:
            cfg = build_yaml(concept, img_dir)
        except Exception as e:
            print(f"  [FAIL] YAML build: {e}")
            summary_rows.append({
                "concept": concept, "eps": eps, "recipe": recipe,
                "variant": vname, "positives": "", "total": "",
                "score": "", "status": f"yaml_error:{e}",
            })
            continue

        # 2) Inference
        if not args.skip_inference:
            log = LOGS_DIR / f"{tag}.log"
            t0 = time.time()
            rc = run_inference(cfg, args.gpu, log)
            dt = time.time() - t0
            if rc != 0:
                print(f"  [FAIL] inference rc={rc} in {dt:.1f}s  log={log}")
                summary_rows.append({
                    "concept": concept, "eps": eps, "recipe": recipe,
                    "variant": vname, "positives": "", "total": "",
                    "score": "", "status": f"inference_rc_{rc}",
                })
                continue
            print(f"  [OK]   inference {dt:.1f}s")

        # 3) Score
        jpath = find_inference_json(concept)
        if not jpath.exists():
            print(f"  [FAIL] no inference JSON at {jpath}")
            summary_rows.append({
                "concept": concept, "eps": eps, "recipe": recipe,
                "variant": vname, "positives": "", "total": "",
                "score": "", "status": "json_missing",
            })
            continue

        try:
            # Auto-detect the final iteration key (captioning = 200 steps =
            # 'iteration_199'; this future-proofs against changes).
            with open(jpath) as f:
                _data = json.load(f)
            iter_keys = [k for k in _data if k.startswith("iteration_")]
            if not iter_keys:
                raise RuntimeError(f"No iteration_* keys in {jpath}")
            last_key = max(iter_keys, key=lambda k: int(k.split("_")[1]))
            result = score_probe_json(str(jpath), concept,
                                      mode="protection",
                                      iteration_key=last_key)
        except Exception as e:
            print(f"  [FAIL] scoring: {e}")
            summary_rows.append({
                "concept": concept, "eps": eps, "recipe": recipe,
                "variant": vname, "positives": "", "total": "",
                "score": "", "status": f"score_error:{e}",
            })
            continue

        print(f"  [SCORE] {result['positives']}/{result['total']} = "
              f"{result['score']:.4f}  (protection, {last_key})")

        with open(detail_path, "w") as f:
            json.dump({
                "concept": concept, "eps": eps, "recipe": recipe,
                "variant": vname, "mode": "protection",
                "iteration": last_key,
                "image_dir": str(img_dir),
                "total": result["total"],
                "positives": result["positives"],
                "score": result["score"],
                "details": result["details"],
            }, f, indent=2)

        summary_rows.append({
            "concept": concept, "eps": eps, "recipe": recipe,
            "variant": vname,
            "positives": result["positives"], "total": result["total"],
            "score": f"{result['score']:.4f}", "status": "ok",
        })

    # ----- CSV outputs -----
    long_csv = RESULTS_DIR / "summary.csv"
    with open(long_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "concept", "eps", "recipe", "variant",
            "positives", "total", "score", "status",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    # Build pivot index across concepts × variants discovered.
    # All variants ever seen — sorted (eps asc, recipe alpha) for stable cols.
    all_variants = sorted(
        {(r["eps"], r["recipe"], r["variant"]) for r in summary_rows
         if r.get("eps") not in (None, "") and r.get("recipe")},
        key=lambda t: (t[0], t[1]),
    )
    variant_cols = [v for _, _, v in all_variants]

    scores_by = {(r["concept"], r["variant"]): r["score"]
                 for r in summary_rows if r.get("score")}

    # Per-eps pivots
    eps_groups = sorted({e for e, _, _ in all_variants})
    for eps in eps_groups:
        eps_variants = [v for (e, _, v) in all_variants if e == eps]
        eps_recipes  = [r for (e, r, _) in all_variants if e == eps]
        p = RESULTS_DIR / f"summary_eps{eps}.csv"
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["concept"] + eps_recipes)
            for concept in concepts:
                w.writerow([concept] + [
                    scores_by.get((concept, v), "") for v in eps_variants
                ])

    # Wide pivot (all variants in column order)
    wide_csv = RESULTS_DIR / "summary_pivot.csv"
    with open(wide_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept"] + variant_cols)
        for concept in concepts:
            row = [concept] + [scores_by.get((concept, v), "")
                               for v in variant_cols]
            w.writerow(row)

    # ----- Console pivot -----
    print()
    print("#" * 92)
    print(f"#  PAPER-EVAL SUMMARY   (elapsed: {time.time() - run_start:.1f}s)")
    print("#" * 92)
    col = 14
    for eps in eps_groups:
        eps_variants = [(r, v) for (e, r, v) in all_variants if e == eps]
        print(f"\n--- eps={eps} ---")
        print(f"{'concept':<16}" + "".join(f"{r:<{col}}" for r, _ in eps_variants))
        print("-" * (16 + col * len(eps_variants)))
        for concept in concepts:
            cells = []
            for _, v in eps_variants:
                val = scores_by.get((concept, v), "")
                cells.append(f"{val:<{col}}" if val else f"{'-':<{col}}")
            print(f"{concept:<16}" + "".join(cells))

    print()
    print(f"Long CSV : {long_csv}")
    print(f"Wide CSV : {wide_csv}")
    if eps_groups:
        print(f"Per-eps  : {RESULTS_DIR}/summary_eps{{{','.join(map(str, eps_groups))}}}.csv")
    print(f"Details  : {DETAIL_DIR}")


if __name__ == "__main__":
    main()