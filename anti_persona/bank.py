"""Album-level references: the leave-one-out token prototype and the CES evidence bank.

Bank building (done once per album, before any photo is protected):
  1. DeepLabV3 person mask of every album photo -> 576 person/not-person patch flags   (anti_persona.person)
  2. up to 32 person patches per photo, evenly spaced, as unit-length anchors, in each of the 25 entries
  3. threshold of each anchor = 90th percentile of its soft-max response over the 100 COCO val2017 photos of
     configs/coco100.json (unrelated public scenes)
At protection time the anchors of the photo being protected are removed (leave-one-out).
"""
import torch
import torch.nn.functional as F

from .config import ANCHORS_PER_PHOTO, NUM_ENTRIES, QUANTILE
from .losses import patch_response


# ------------------------------------------------------------------ token prototype
@torch.no_grad()
def album_features(encoder, album):
    """Clean features of every album photo (list over photos of 25 entries)."""
    return [encoder(image) for image in album]


def album_sums(features):
    """Per entry, the sum over all album photos, accumulated in album order."""
    sums = None
    for photo in features:
        sums = [f.clone() for f in photo] if sums is None else [s + f for s, f in zip(sums, photo)]
    return sums


def prototype_from_sums(sums, own, count):
    """Per entry: (album sum - the query photo) / (number of photos - 1)."""
    return [(s - f) / (count - 1) for s, f in zip(sums, own)]


@torch.no_grad()
def leave_one_out_prototype(encoder, album, index):
    """Leave-one-out token prototype of photo ``index`` (unmasked) and that photo's clean features."""
    features = album_features(encoder, album)
    return prototype_from_sums(album_sums(features), features[index], len(album)), features[index]


# ------------------------------------------------------------------ CES evidence bank
def anchor_indices(tokens, per_photo=ANCHORS_PER_PHOTO):
    """Up to ``per_photo`` person patches, evenly spaced over the photo's person patches (row-major order)."""
    idx = torch.nonzero(tokens).flatten()
    if len(idx) > per_photo:
        idx = idx[torch.linspace(0, len(idx) - 1, per_photo).round().long()]
    return idx


@torch.no_grad()
def build_ces_bank(encoder, album, person_tokens, public_images, features=None):
    """The person's evidence bank. ``person_tokens[i]`` = 576 booleans of album photo i (DeepLabV3 person patches);
    ``public_images`` = the calibration photos (336 x 336 tensors, or paths read lazily with ``read_image``).
    Returns ``{entry: dict(anchors [N, D], thresholds [N], reference_index [N])}``."""
    picks = [anchor_indices(tokens) for tokens in person_tokens]
    feats = features if features is not None else [encoder(image) for image in album]
    anchors = [torch.cat([F.normalize(f[l][1:, 0, :].float(), dim=-1)[s.to(f[l].device)] for f, s in zip(feats, picks)])
               for l in range(NUM_ENTRIES)]
    reference_index = torch.cat([torch.full((len(s),), i, dtype=torch.long) for i, s in enumerate(picks)])
    responses = [torch.empty(len(public_images), len(reference_index)) for _ in range(NUM_ENTRIES)]
    for k, image in enumerate(public_images):
        if not torch.is_tensor(image):
            from .encoder import read_image
            image = read_image(image, anchors[0].device)
        public = encoder(image)
        for l in range(NUM_ENTRIES):
            responses[l][k] = patch_response(anchors[l], public[l][1:, 0, :].float()).cpu()
    thresholds = [torch.quantile(responses[l].to(anchors[l].device), QUANTILE, dim=0) for l in range(NUM_ENTRIES)]
    return {l: dict(anchors=anchors[l], thresholds=thresholds[l], reference_index=reference_index) for l in range(NUM_ENTRIES)}


def bank_without(saved, index):
    """Leave-one-out: drop the anchors of the photo being protected. Returns a list of (anchors, thresholds)."""
    bank = []
    for entry in range(NUM_ENTRIES):
        anchors, thresholds = saved[entry]["anchors"], saved[entry]["thresholds"]
        keep = (saved[entry]["reference_index"] != index).to(anchors.device)
        bank.append((anchors[keep], thresholds[keep.to(thresholds.device)]))
    return bank


def save_ces_bank(bank, path):
    torch.save({l: {k: v.cpu() for k, v in entry.items()} for l, entry in bank.items()}, path)


def load_ces_bank(path, index, device):
    """A stored bank (as written by ``save_ces_bank`` or the paper's bank builder), minus the query's anchors."""
    saved = torch.load(path, map_location=device, weights_only=True)
    return bank_without(saved, index)
