"""Fetch the 100 COCO val2017 photos that calibrate the CES anchor thresholds (configs/coco100.json) and verify them.

    python scripts/download_coco100.py [--out data/coco100] [--from-local /path/to/coco/val2017]

The photos are fixed, unrelated public scenes containing people (sampling seed 42). They are not redistributed here;
they are downloaded from the official COCO server (CC BY 4.0 annotations; images under their Flickr licences).
"""
import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
URL = "http://images.cocodataset.org/val2017/{name}"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=REPO / "data" / "coco100")
    p.add_argument("--from-local", type=Path, default=None, help="copy from an existing COCO val2017 folder")
    args = p.parse_args()
    ids = json.loads((REPO / "configs" / "coco100.json").read_text())["image_ids"]
    expected = dict(reversed(line.split()) for line in (REPO / "scripts" / "checksums" / "coco100.sha256").read_text().splitlines())
    args.out.mkdir(parents=True, exist_ok=True)
    bad = []
    for i in ids:
        name = f"{i:012d}.jpg"
        dst = args.out / name
        if not dst.is_file():
            if args.from_local:
                shutil.copyfile(args.from_local / name, dst)
            else:
                urllib.request.urlretrieve(URL.format(name=name), dst)
        if sha256(dst) != expected[name]:
            bad.append(name)
    print(f"{len(ids) - len(bad)} / {len(ids)} COCO-100 photos verified in {args.out}")
    if bad:
        sys.exit(f"checksum mismatch: {bad}")


if __name__ == "__main__":
    main()
