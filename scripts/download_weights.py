"""Download (and convert) every model the final method needs into ./weights and verify the checksums.

    python scripts/download_weights.py [--weights weights]

  weights/clip-vit-large-patch14-336/        CLIP ViT-L/14-336 (Hugging Face, pinned revision)
  weights/deeplabv3_resnet101_coco-586e9e4e.pth   DeepLabV3-ResNet101 (torchvision COCO_WITH_VOC_LABELS_V1)
  weights/buffalo_l/det_10g.onnx, w600k_r50.onnx  insightface buffalo_l (SCRFD detector + ArcFace recogniser)
  weights/buffalo_l_w600k_r50_torch.pt        the recogniser as a torch node list (scripts/convert_buffalo_l.py)

Model weights are NOT part of this repository; each is governed by its own licence (see licenses/).
The insightface models are released for non-commercial research use only.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLIP_REPO = "openai/clip-vit-large-patch14-336"
CLIP_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
CLIP_FILES = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
DEEPLAB_URL = "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def expected_sums():
    sums = {}
    for line in (REPO / "scripts" / "checksums" / "weights.sha256").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            sums[name] = digest
    return sums


def fetch_clip(dst):
    from huggingface_hub import hf_hub_download
    dst.mkdir(parents=True, exist_ok=True)
    for name in CLIP_FILES:
        if not (dst / name).is_file():
            got = hf_hub_download(CLIP_REPO, name, revision=CLIP_REVISION)
            shutil.copyfile(got, dst / name)


def fetch_deeplab(dst):
    if not dst.is_file():
        from torch.hub import download_url_to_file
        download_url_to_file(DEEPLAB_URL, str(dst), hash_prefix="586e9e4e")


def fetch_buffalo(dst):
    dst.mkdir(parents=True, exist_ok=True)
    if (dst / "det_10g.onnx").is_file() and (dst / "w600k_r50.onnx").is_file():
        return
    from torch.hub import download_url_to_file
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "buffalo_l.zip"
        download_url_to_file(BUFFALO_URL, str(archive))
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                name = Path(member).name
                if name in ("det_10g.onnx", "w600k_r50.onnx"):
                    with z.open(member) as src, open(dst / name, "wb") as out:
                        shutil.copyfileobj(src, out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", type=Path, default=REPO / "weights")
    p.add_argument("--skip-clip", action="store_true", help="let transformers fetch CLIP into its own cache instead")
    args = p.parse_args()
    w = args.weights
    w.mkdir(parents=True, exist_ok=True)
    if not args.skip_clip:
        fetch_clip(w / "clip-vit-large-patch14-336")
    fetch_deeplab(w / "deeplabv3_resnet101_coco-586e9e4e.pth")
    fetch_buffalo(w / "buffalo_l")
    if not (w / "buffalo_l_w600k_r50_torch.pt").is_file():
        subprocess.run([sys.executable, str(REPO / "scripts" / "convert_buffalo_l.py"),
                        "--onnx", str(w / "buffalo_l" / "w600k_r50.onnx"),
                        "--out", str(w / "buffalo_l_w600k_r50_torch.pt")], check=True)
    bad = []
    for name, digest in expected_sums().items():
        path = w / name
        if not path.is_file():
            print(f"skipped  {name} (not downloaded)")
            continue
        ok = sha256(path) == digest
        print(f"{'ok      ' if ok else 'MISMATCH'} {name}")
        if not ok and name != "buffalo_l_w600k_r50_torch.pt":   # torch.save bytes can differ across torch versions
            bad.append(name)
    if bad:
        sys.exit(f"checksum mismatch: {bad}")


if __name__ == "__main__":
    main()
