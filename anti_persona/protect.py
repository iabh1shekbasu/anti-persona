"""Protect every photo of one person's album, leave-one-out (the final Anti-Persona method).

    python -m anti_persona.protect --album <dir> --out <dir> [--seed 0]

For an album of n >= 2 photos of the same person the CLI
  1. encodes every photo with CLIP ViT-L/14-336 (25 hidden states) -> token prototype sums,
  2. segments every photo with DeepLabV3 -> person patches -> CES anchors; calibrates each anchor's threshold on the
     COCO-100 photos (90th percentile) -> the person's evidence bank (cached in <out>/_cache/bank.pt),
  3. detects and aligns the largest face of every photo (SCRFD det_10g) for the buffalo_l face prototype,
  4. protects each photo i with the references built from the OTHER n - 1 photos (prototype, anchors and face
     prototype all leave photo i out), image seed = seed + i, and writes <out>/<stem>.png and <out>/<stem>_loss.csv.
Outputs are 336 x 336 PNGs (the resolution the victims' CLIP encoder sees), within 4/255 of the squashed clean photo.
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
REPO = Path(__file__).resolve().parents[1]


def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", Path(path).name)]


def album_paths(album, order=None):
    """Album order = the file names listed in ``order`` (one per line) or, by default, natural sort of the names."""
    album = Path(album)
    if order:
        names = [line.strip() for line in Path(order).read_text().splitlines() if line.strip()]
        paths = [album / name for name in names]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"listed in --order but missing: {missing}")
        return paths
    return sorted((p for p in album.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key)


def coco_paths(coco_dir, ids_json=REPO / "configs" / "coco100.json"):
    ids = json.loads(Path(ids_json).read_text())["image_ids"]
    paths = [Path(coco_dir) / f"{i:012d}.jpg" for i in ids]
    missing = [p.name for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} COCO-100 calibration photos missing in {coco_dir} "
                                f"(run scripts/download_coco100.py): {missing[:3]} ...")
    return paths


def protect_album(paths, out, seed, encoder, person_tokens, public_images, device, face_model=None, face_rows=None,
                  steps=None, only=None, bank=None, log=print):
    """The whole album pipeline with already-loaded models (also used by the CPU equivalence test).

    paths          album photos in album order (n >= 2)
    person_tokens  list of 576-bool tensors (DeepLabV3 person patches), one per photo; ignored when ``bank`` is given
    public_images  calibration photos (paths or 336 x 336 tensors); ignored when ``bank`` is given
    face_model     a loaded BuffaloL (or None: face term absent); face_rows = [(matrix or None, clean)] per photo
    only           optional set of indices to protect (default: all)
    Returns a list of per-photo records."""
    import torch
    from PIL import Image

    from .bank import album_features, album_sums, bank_without, build_ces_bank, prototype_from_sums
    from .config import STEPS
    from .encoder import read_image
    from .face import FaceTerm
    from .optim import protect

    steps = STEPS if steps is None else steps
    if len(paths) < 2:
        raise ValueError("an album needs at least two photos of the same person (leave-one-out references)")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    album = [read_image(p, device) for p in paths]
    features = album_features(encoder, album)
    sums = album_sums(features)
    if bank is None:
        bank = build_ces_bank(encoder, album, person_tokens, public_images, features=features)
    records = []
    for index, path in enumerate(paths):
        if only is not None and index not in only:
            continue
        prototype = prototype_from_sums(sums, features[index], len(album))
        query_bank = bank_without(bank, index)
        face = FaceTerm(None, face_rows, index, device, model=face_model) if face_model is not None else None
        began = time.monotonic()
        png, trace = protect(album[index], encoder, prototype, query_bank, face, seed + index, steps)
        stem = Path(path).stem
        Image.fromarray(png[0].permute(1, 2, 0).cpu().numpy()).save(out / f"{stem}.png")
        with (out / f"{stem}_loss.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
            writer.writeheader()
            writer.writerows(trace)
        linf = float((png.float() - album[index].round().clamp(0, 255)).abs().max())
        record = dict(photo=Path(path).name, index=index, image_seed=seed + index, steps=steps,
                      face_active=bool(face is not None and face.active),
                      anchors_used=int(query_bank[0][0].shape[0]), linf_vs_rounded_clean=linf,
                      seconds=round(time.monotonic() - began, 1), final=trace[-1])
        records.append(record)
        log(json.dumps(record))
        del png, trace
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return records


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m anti_persona.protect",
        description="Anti-Persona (final method): protect every photo of one person's album, leave-one-out.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--album", type=Path, required=True, help="folder with >= 2 photos of ONE person")
    p.add_argument("--out", type=Path, required=True, help="output folder (PNGs, loss tables, album.json)")
    p.add_argument("--seed", type=int, default=0, help="run seed; photo i uses image seed = seed + i (the paper: 0 and 202)")
    p.add_argument("--order", type=Path, default=None,
                   help="text file listing the album file names in album order (default: natural sort of names)")
    p.add_argument("--only", nargs="+", default=None, help="protect only these file names (references still use the whole album)")
    p.add_argument("--weights", type=Path, default=REPO / "weights",
                   help="folder filled by scripts/download_weights.py (DeepLabV3, buffalo_l torch weights, det_10g.onnx)")
    p.add_argument("--coco-dir", type=Path, default=REPO / "data" / "coco100",
                   help="the 100 COCO val2017 calibration photos of configs/coco100.json (scripts/download_coco100.py)")
    p.add_argument("--clip", default=None,
                   help="CLIP ViT-L/14-336 name or local folder (default: openai/clip-vit-large-patch14-336 from the HF hub)")
    p.add_argument("--bank", type=Path, default=None, help="reuse a stored evidence bank (.pt) built for exactly this album order")
    p.add_argument("--face-cache", type=Path, default=None,
                   help="recorded face alignments JSON (rows: selected_matrix, clean_png) instead of running the detector")
    p.add_argument("--steps", type=int, default=501, help="optimisation steps (501 in the paper)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--threads", type=int, default=2, help="CPU threads (JPEG round trips, face alignment)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import torch

    from .bank import save_ces_bank
    from .config import CLIP_NAME
    from .encoder import ClipEncoder
    from .face import face_rows_detect, face_rows_from_cache, load_buffalo_l
    from .optim import seed_everything
    from .person import DEEPLAB_FILE, load_deeplab, person_tokens, save_tokens

    torch.set_num_threads(args.threads)
    paths = album_paths(args.album, args.order)
    if len(paths) < 2:
        sys.exit(f"{args.album}: an album needs at least two photos of the same person (found {len(paths)})")
    only = None
    if args.only:
        names = [p.name for p in paths]
        only = {names.index(n) for n in args.only}
    cache = args.out / "_cache"
    cache.mkdir(parents=True, exist_ok=True)
    w = args.weights
    face_weights, detector = w / "buffalo_l_w600k_r50_torch.pt", w / "buffalo_l" / "det_10g.onnx"
    for needed in [face_weights] + ([] if args.face_cache else [detector]):
        if not needed.is_file():
            sys.exit(f"missing {needed}: run scripts/download_weights.py first")

    seed_everything(args.seed)
    bank = None
    tokens, public = None, None
    names = [p.name for p in paths]
    cached = (cache / "bank.pt").is_file() and (cache / "bank.json").is_file() and \
        json.loads((cache / "bank.json").read_text()).get("album") == names
    if args.bank is not None or cached:
        bank_path = args.bank or cache / "bank.pt"
        bank = torch.load(bank_path, map_location=args.device, weights_only=True)
        print(f"evidence bank: {bank_path}")
    else:
        deeplab = load_deeplab(w / DEEPLAB_FILE, args.device)
        tokens = []
        for p in paths:
            t, mask336 = person_tokens(deeplab, p, args.device)
            save_tokens(cache / "person_tokens" / f"{p.stem}.npz", t, mask336)
            if int(t.sum()) == 0:
                print(f"warning: no person patch found in {p.name}; it contributes no CES anchor")
            tokens.append(t)
        del deeplab
        public = coco_paths(args.coco_dir)

    local_clip = w / "clip-vit-large-patch14-336"
    encoder = ClipEncoder(args.device, args.clip or (str(local_clip) if (local_clip / "config.json").is_file() else CLIP_NAME))
    if bank is None:
        from .bank import album_features, build_ces_bank
        from .encoder import read_image
        album = [read_image(p, args.device) for p in paths]
        bank = build_ces_bank(encoder, album, tokens, public, features=album_features(encoder, album))
        save_ces_bank(bank, cache / "bank.pt")
        (cache / "bank.json").write_text(json.dumps(dict(
            album=names, anchors_per_photo=[int((bank[0]["reference_index"] == i).sum()) for i in range(len(paths))],
            coco_ids="configs/coco100.json", quantile=0.9, temperature=0.05), indent=1) + "\n")
        del album
    rows = face_rows_from_cache(args.face_cache, args.device) if args.face_cache else face_rows_detect(paths, args.device, detector)
    face_model = load_buffalo_l(face_weights, args.device)
    records = protect_album(paths, args.out, args.seed, encoder, None, None, args.device, face_model=face_model,
                            face_rows=rows, steps=args.steps, only=only, bank=bank)
    (args.out / "album.json").write_text(json.dumps(dict(
        album=[p.name for p in paths], seed=args.seed, steps=args.steps,
        faces=[m is not None for m, _ in rows], records=records), indent=1) + "\n")


if __name__ == "__main__":
    main()
