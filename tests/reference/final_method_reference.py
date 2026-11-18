"""The final Anti-Persona method ("Ours"), restated in ONE file of plain PyTorch.

THIS FILE IS A READING AID. The numbers in the paper were produced by the research code (a generator script and the
modules it imports; not part of this release). This file repeats that arithmetic so that it can be read top to bottom;
where the two could ever disagree, the research code is right. How it was tested against that code: README.md here.

The loss that is MINIMISED over the perturbation delta (one photo, the other album photos of the person as references):

    L = 1 * L_self + 0.5 * L_proto + 2 * L_ces + 0.125 * L_face

  L_self   mean token cosine between the photo's CLIP features and the CLEAN photo's features.
  L_proto  the same cosine against the album average of the OTHER photos, position by position, over the summary token
           and ALL 576 patches (no segmentation mask anywhere in this term).
  L_ces    "evidence" term: for each stored anchor (a person patch of another album photo) a soft maximum of its
           cosine over the photo's 576 patches, penalised above a threshold calibrated on 100 public COCO photos;
           the strongest quarter of the anchors is averaged. NOT position-matched.
  L_face   cosine between the buffalo_l face embedding of the photo's aligned face and the mean embedding of the other
           album faces. The recogniser is fed B,G,R (the "colour fix").
Every CLIP term is a plain mean over the 25 feature entries (the tokens before the first block and the output of each
of the 24 blocks of CLIP ViT-L/14-336); no entry has its own weight.

Optimiser: 501 steps, delta starts at zero, delta <- clamp(delta - 0.5 * sign(gradient), -4, 4) in pixel units of
[0, 255] (0.5/255 per step, budget 4/255). At each step the photo is passed through a real JPEG with probability 0.7
(quality uniform in 60..95, gradient passed straight through); every third step the gradient is smoothed with a 3x3
Gaussian (sigma 3). Segmentation: only DeepLabV3 person masks, and only to choose the anchors of L_ces. No SAM 3.

Line references (X:a-b = lines a to b of module X of the research code, not part of this release):
  A  = anti_persona/attack.py     P  = anti_persona/person.py     FA = anti_persona/face.py
  E  = evidence-term module       G  = album generator            W  = final-method generator
  BL = buffalo_l replay           BO = face-term module           M  = person-mask module
"""
import argparse
import csv
import io
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

W_SELF, W_PROTO, W_CES, W_FACE = 1.0, 0.5, 2.0, 0.125   # E:33-35 with the arm entry W:104,218 (w_prototype 0.5)
EPS, STEP, STEPS = 4.0, 0.5, 501                        # pixel units of [0, 255]            E:50,69  G:131
JPEG_CLEAN_P, JPEG_Q = 0.3, (60, 95)                    # no JPEG with probability 0.3       E:51
SMOOTH_EVERY = 3                                        # E:65
TAU, MARGIN, TOP_SHARE = 0.05, 0.05, 0.25               # CES temperature, margin, top share A:135-137,160-161
ANCHORS_PER_PHOTO, QUANTILE = 32, 0.9                   # bank calibration                   P:47-51,68


# ------------------------------------------------------------------ image and encoder
def read_image(path, device):
    """A:66-71. RGB, squashed to 336 x 336 (bicubic), values in [0, 255]."""
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255).mul(255)
    return F.interpolate(x[None].to(device), (336, 336), mode="bicubic", align_corners=False).clamp(0, 255)


class ClipEncoder(torch.nn.Module):
    """A:86-117. CLIP ViT-L/14-336 vision tower. Returns 25 entries, each [577, 1, 1024] (token 0 = summary token):
    entry 0 = the tokens after the pre-transformer layer norm, entries 1..24 = the output of each block."""

    def __init__(self, device, name="openai/clip-vit-large-patch14-336"):
        super().__init__()
        from transformers import CLIPModel
        self.vision = CLIPModel.from_pretrained(name).to(device).float().eval().vision_model.requires_grad_(False)
        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1) * 255)

    def forward(self, image):
        emb = self.vision.embeddings
        x = emb.patch_embedding((image - self.mean) / self.std)
        batch, channels, height, width = x.shape
        x = x.reshape(batch, channels, height * width).permute(0, 2, 1)
        cls = emb.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(batch, -1, -1)
        x = self.vision.pre_layrnorm(torch.cat((cls, x), dim=1) + emb.position_embedding.weight.to(x.dtype))
        features = [x.permute(1, 0, 2).contiguous()]
        for block in self.vision.encoder.layers:
            result = block(x, attention_mask=None, causal_attention_mask=None)
            x = result[0] if isinstance(result, tuple) else result
            features.append(x.permute(1, 0, 2).contiguous())
        return features


# ------------------------------------------------------------------ L_self and L_proto (one function, two references)
def mean_token_cosine(feature, reference):
    """A:120-125 (= A:309-312). Cosine between token t of the photo and token t of the reference (same position),
    averaged over the summary token and all 576 patches of one entry. The reference carries no gradient."""
    dim = feature.shape[-1]
    query = F.normalize(feature.reshape(-1, dim), dim=-1)
    target = F.normalize(reference.reshape(-1, dim), dim=-1).detach()
    return F.cosine_similarity(query, target).mean()


def self_term(features, clean_features):
    """A:307-318. Reference = the clean photo itself. Plain mean over the 25 entries."""
    return torch.stack([mean_token_cosine(f, c) for f, c in zip(features, clean_features)]).mean()


def prototype_term(features, prototype):
    """P:88-97 with every one of the 576 positions kept (W:265-275), which is A:128-132: reference = the album average
    of the OTHER photos at the same grid position. Plain mean over the 25 entries. No mask."""
    return torch.stack([mean_token_cosine(f, p) for f, p in zip(features, prototype)]).mean()


@torch.no_grad()
def leave_one_out_prototype(encoder, album, index):
    """G:163-168. Per entry: (sum over all album photos - the query photo) / (number of photos - 1), unmasked.
    `album` = the 336 x 336 images in album order. Also returns the clean features of the query photo."""
    sums, own = None, None
    for i, image in enumerate(album):
        features = encoder(image)
        sums = [f.clone() for f in features] if sums is None else [s + f for s, f in zip(sums, features)]
        own = features if i == index else own
    return [(s - f) / (len(album) - 1) for s, f in zip(sums, own)], own


# ------------------------------------------------------------------ L_ces (the evidence term)
def patch_response(anchors, patches, temperature=TAU):
    """A:135-137. Soft maximum over the photo's patches of cos(anchor, patch): tau * (logsumexp(cos / tau) - log 576)."""
    similarity = F.normalize(anchors, dim=-1) @ F.normalize(patches, dim=-1).T
    return temperature * (torch.logsumexp(similarity / temperature, dim=-1) - math.log(len(patches)))


def ces_term(features, bank):
    """A:153-162. Per entry: penalty = 0.05 * softplus((response - threshold + 0.05) / 0.05) per anchor, mean of the
    strongest 25% of the anchors; then a plain mean over the 25 entries. bank[l] = (anchors [N, 1024], thresholds [N])."""
    terms = []
    for feature, (anchors, thresholds) in zip(features, bank):
        response = patch_response(anchors, feature[1:, 0, :])                 # patches only, no summary token
        penalty = TAU * F.softplus((response - thresholds + MARGIN) / TAU)    # smooth max(0, response - threshold + margin)
        terms.append(penalty.topk(max(1, math.ceil(len(penalty) * TOP_SHARE))).values.mean())
    return torch.stack(terms).mean()


def load_ces_bank(path, index, device):
    """G:113-115,170 and A:146-150. The stored bank of the person, minus the anchors of the query photo itself."""
    saved = torch.load(path, map_location=device, weights_only=True)
    bank = []
    for entry in range(25):
        keep = saved[entry]["reference_index"] != index
        bank.append((saved[entry]["anchors"][keep], saved[entry]["thresholds"][keep]))
    return bank


@torch.no_grad()
def build_ces_bank(encoder, album, person_tokens, public_images):
    """P:47-70 (called by the bank-preparation and calibration scripts). How the stored bank was made.
    person_tokens[i] = 576 yes/no values of album photo i from its DeepLabV3 person mask (M:38-58,88-96: class
    "person", probability > 0.5, dilated by 2% of the long side, squashed to 336 x 336, patch counts when at least 25%
    of it is person). Anchors: up to 32 person patches per album photo, evenly spaced, L2-normalised. Threshold of
    each anchor: the 90th percentile of its patch_response over the 100 public COCO photos."""
    picks = []
    for tokens in person_tokens:
        idx = torch.nonzero(tokens).flatten()
        if len(idx) > ANCHORS_PER_PHOTO:
            idx = idx[torch.linspace(0, len(idx) - 1, ANCHORS_PER_PHOTO).round().long()]
        picks.append(idx)
    feats = [encoder(image) for image in album]
    anchors = [torch.cat([F.normalize(f[l][1:, 0, :].float(), dim=-1)[s.to(f[l].device)] for f, s in zip(feats, picks)]) for l in range(25)]
    reference_index = torch.cat([torch.full((len(s),), i, dtype=torch.long) for i, s in enumerate(picks)])
    responses = [torch.empty(len(public_images), len(reference_index)) for _ in range(25)]
    for k, image in enumerate(public_images):
        features = encoder(image)
        for l in range(25):
            responses[l][k] = patch_response(anchors[l], features[l][1:, 0, :].float()).cpu()
    thresholds = [torch.quantile(responses[l].to(anchors[l].device), QUANTILE, dim=0) for l in range(25)]
    return {l: dict(anchors=anchors[l], thresholds=thresholds[l], reference_index=reference_index) for l in range(25)}


# ------------------------------------------------------------------ L_face
class BuffaloL(torch.nn.Module):
    """BL:39-78. insightface buffalo_l w600k_r50 (the recogniser of MyVLM's face gate) as a differentiable replay of
    its ONNX node list; the weights file was converted from the ONNX file (convert_bl.py) and is frozen."""

    def __init__(self, saved):
        super().__init__()
        self.nodes, self.input_name, self.output_name, self.buffer_of = saved["nodes"], saved["input"], saved["output"], {}
        for i, (name, value) in enumerate(saved["tensors"].items()):
            self.register_buffer(f"t{i}", value.clone().float())
            self.buffer_of[name] = f"t{i}"

    def forward(self, x):
        env = {self.input_name: x}
        for node in self.nodes:
            ins = [env[n] if n in env else getattr(self, self.buffer_of[n]) for n in node["inputs"]]
            op, a = node["op"], node["attrs"]
            if op == "Conv":
                y = F.conv2d(ins[0], ins[1], ins[2] if len(ins) > 2 else None, stride=a["strides"],
                             padding=(a["pads"][0], a["pads"][1]), dilation=a["dilations"], groups=a["group"])
            elif op == "PRelu":
                slope = ins[1].reshape(1, -1, 1, 1) if ins[1].numel() == ins[0].shape[1] else ins[1]
                y = torch.where(ins[0] >= 0, ins[0], ins[0] * slope)
            elif op == "BatchNormalization":
                y = F.batch_norm(ins[0], ins[3], ins[4], ins[1], ins[2], training=False, eps=a["epsilon"])
            elif op == "Add":
                y = ins[0] + ins[1]
            elif op == "Flatten":
                y = ins[0].flatten(1)
            elif op == "Gemm":
                y = F.linear(ins[0], ins[1], ins[2])
            else:
                raise ValueError(op)
            env[node["outputs"][0]] = y
        return env[self.output_name]


def aligned_crop(image, matrix):
    """FA:62-90. The 112 x 112 aligned face. Forward value = OpenCV's warp of the rounded image (what a real face
    pipeline sees); gradient = that of the bilinear sampling of the same transform."""
    import cv2
    height, width = image.shape[-2:]
    affine = torch.eye(3, dtype=torch.float64)
    affine[:2] = torch.as_tensor(matrix, dtype=torch.float64)
    yy, xx = torch.meshgrid(torch.arange(112, dtype=torch.float64), torch.arange(112, dtype=torch.float64), indexing="ij")
    xy = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1) @ torch.linalg.inv(affine).T
    grid = torch.stack((2 * xy[..., 0] / (width - 1) - 1, 2 * xy[..., 1] / (height - 1) - 1), dim=-1)[None]
    soft = F.grid_sample(image, grid.to(device=image.device, dtype=image.dtype), mode="bilinear", padding_mode="zeros", align_corners=True)
    rgb = image[0].detach().round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    crop = cv2.warpAffine(rgb, np.asarray(matrix, dtype=np.float64), (112, 112), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    exact = torch.from_numpy(crop.copy()).permute(2, 0, 1)[None].to(image)
    return soft + (exact - soft).detach()


class FaceTerm:
    """BL:87-100, BO:12-18, FA:163-180, with the colour fix W:121-133. `rows` = per album photo (matrix or None,
    clean 336 x 336 image): the 2x3 alignment matrix of its largest face (SCRFD det_10g landmarks, FA:106-128) or None
    when no face was found. Inactive (term absent) when the query photo or every other photo has no face."""

    def __init__(self, weights_path, rows, index, device):
        self.model = BuffaloL(torch.load(weights_path, map_location="cpu", weights_only=True)).eval().requires_grad_(False).to(device)
        self.matrix = rows[index][0]
        with torch.no_grad():
            others = [self.embed(aligned_crop(clean, m)) for i, (m, clean) in enumerate(rows) if i != index and m is not None]
        self.active = self.matrix is not None and bool(others)                          # FA:163-168
        self.prototype = torch.stack([e[0] for e in others]).mean(0).reshape(1, 1, -1) if others else None   # FA:177-178

    def embed(self, crop):
        """FA:93-103: pixels to [-1, 1]; W:133: channels reversed to B,G,R before the recogniser; unit length."""
        x = crop.float().div(255).sub(0.5).div(0.5)
        return F.normalize(self.model(x.flip(1)).float(), dim=-1)

    def __call__(self, image):
        """FA:173-180. Cosine between the photo's face embedding and the mean embedding of the other album faces."""
        return mean_token_cosine(self.embed(aligned_crop(image, self.matrix)).reshape(1, 1, -1), self.prototype)


def face_rows_from_cache(cache_json, device):
    """FA:139-153. Recorded alignments (paper-10): the matrix and the stored clean 336 x 336 PNG of each album photo."""
    rows = []
    for row in json.loads(Path(cache_json).read_text())["rows"]:
        with Image.open(row["clean_png"]) as im:
            rows.append((row["selected_matrix"], torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)[None].float().to(device)))
    return rows


def face_rows_detect(paths, device, detector_path):
    """FA:106-128. Alignments computed here (MMPB-15): SCRFD det_10g on the rounded clean image, largest face."""
    import cv2
    from insightface.model_zoo import get_model
    from insightface.utils.face_align import estimate_norm
    detector = get_model(str(detector_path), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)
    rows = []
    for path in paths:
        clean = read_image(path, device).round().clamp(0, 255)
        boxes, marks = detector.detect(cv2.cvtColor(clean[0].byte().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR), max_num=0)
        best = max(range(len(boxes)), key=lambda i: float((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))) if len(boxes) else None
        rows.append((estimate_norm(marks[best], image_size=112).tolist() if best is not None else None, clean))
    return rows


# ------------------------------------------------------------------ the pieces of the optimisation loop
def seed_everything(seed):
    """A:74-83."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    if torch.version.hip:
        torch.backends.cudnn.enabled = False


def jpeg_view(image, quality):
    """A:270-277. A real JPEG (4:4:4) of the rounded image in the forward pass; the gradient passes straight through."""
    rgb = image[0].detach().round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="JPEG", quality=quality, subsampling=0)
    buffer.seek(0)
    decoded = torch.from_numpy(np.array(Image.open(buffer).convert("RGB"))).permute(2, 0, 1)[None].to(image)
    return image + (decoded - image).detach()


def smooth_gradient(gradient):
    """A:280-286. 3 x 3 Gaussian with sigma 3 (exp(-(x^2 + y^2) / 18), normalised), per channel, zero padding."""
    axis = torch.arange(3, device=gradient.device, dtype=gradient.dtype) - 1
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / 18)
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    return F.conv2d(gradient, kernel, padding=1, groups=3)


def quantize_bounded(protected, clean, epsilon=EPS):
    """A:289-298. Whole-number pixels that stay within 4 of the clean image AND of the rounded clean image."""
    rounded = clean.round().clamp(0, 255)
    lower = torch.maximum((clean - epsilon).ceil(), (rounded - epsilon).ceil()).clamp(0, 255)
    upper = torch.minimum((clean + epsilon).floor(), (rounded + epsilon).floor()).clamp(0, 255)
    return torch.maximum(torch.minimum(protected.round(), upper), lower).byte()


def loss_terms(view, encoder, clean_features, prototype, bank, face):
    """A:301-327 and E:54-58. Returns (L, the four terms). The sum is formed in the pinned order."""
    features = encoder(view)
    l_self, l_proto, l_ces = self_term(features, clean_features), prototype_term(features, prototype), ces_term(features, bank)
    loss = W_SELF * l_self
    loss = loss + W_PROTO * l_proto + W_CES * l_ces
    l_face = face(view) if face is not None and face.active else None
    if l_face is not None:
        loss = loss + W_FACE * l_face
    return loss, dict(self_raw=l_self, prototype_raw=l_proto, evidence_mean_raw=l_ces, face_raw=l_face if l_face is not None else 0.0)


def protect(clean, encoder, prototype, bank, face, seed, steps=STEPS):
    """E:43-84. One photo. `seed` = run seed (0 or 202) + the photo's position in the album (G:189).
    Returns (the protected image as whole numbers [1, 3, 336, 336], one row of logged values per step)."""
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
    protected = (clean + delta.detach().clamp(-EPS, EPS)).clamp(0, 255)
    return quantize_bounded(protected, clean), trace


def main():
    """One photo from the command line (see README.md, "How to run one photo")."""
    p = argparse.ArgumentParser(description="Final method, reading-aid implementation: protect ONE photo.")
    p.add_argument("--album", nargs="+", type=Path, required=True, help="all photos of the person, in album order")
    p.add_argument("--index", type=int, required=True, help="position of the photo to protect in --album")
    p.add_argument("--bank", type=Path, required=True, help="the person's evidence bank (.pt)")
    p.add_argument("--face-weights", type=Path, required=True, help="buffalo_l_w600k_r50_torch.pt")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--face-cache", type=Path, help="recorded alignments (paper-10)")
    g.add_argument("--detector", type=Path, help="SCRFD det_10g.onnx; alignments are computed (MMPB-15)")
    p.add_argument("--seed", type=int, required=True, help="run seed: 0 or 202 (the image seed is seed + index)")
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", type=Path, required=True, help="a new folder")
    a = p.parse_args()
    a.output.mkdir(parents=True)
    torch.set_num_threads(2)                                                  # G:140
    seed_everything(a.seed)                                                   # G:160
    encoder = ClipEncoder(a.device)
    album = [read_image(path, a.device) for path in a.album]
    prototype, _ = leave_one_out_prototype(encoder, album, a.index)
    bank = load_ces_bank(a.bank, a.index, a.device)
    rows = face_rows_from_cache(a.face_cache, a.device) if a.face_cache else face_rows_detect(a.album, a.device, a.detector)
    face = FaceTerm(a.face_weights, rows, a.index, a.device)
    png, trace = protect(album[a.index], encoder, prototype, bank, face, a.seed + a.index, a.steps)
    Image.fromarray(png[0].permute(1, 2, 0).cpu().numpy()).save(a.output / (a.album[a.index].stem + ".png"))
    with (a.output / (a.album[a.index].stem + "_loss.csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    print(json.dumps(dict(photo=str(a.album[a.index]), steps=a.steps, face_active=face.active, last=trace[-1])))


if __name__ == "__main__":
    main()
