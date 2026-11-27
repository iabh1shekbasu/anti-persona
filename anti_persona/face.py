"""L_face: the face-prototype term with insightface buffalo_l (w600k_r50), the recogniser of MyVLM's face gate.

The ONNX recogniser is converted once into a node list + weights (``scripts/convert_buffalo_l.py``) and replayed
here op by op as a differentiable, frozen torch module. Faces are found with SCRFD det_10g (largest face), aligned
to 112 x 112 with the 5-point similarity transform, normalised to [-1, 1] and fed in B,G,R channel order (the order
insightface uses). The face prototype is the mean embedding of the OTHER album photos (leave-one-out). The term is
absent for a photo with no detected face, or when no other album photo has one.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .encoder import read_image
from .losses import mean_token_cosine

ONNX_OPS = {"Conv", "PRelu", "BatchNormalization", "Add", "Flatten", "Gemm"}


class BuffaloL(torch.nn.Module):
    """Differentiable replay of the buffalo_l w600k_r50 ONNX graph (BN-folded iResNet-50; six op types)."""

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


def load_buffalo_l(weights_path, device):
    saved = torch.load(weights_path, map_location="cpu", weights_only=True)
    return BuffaloL(saved).eval().requires_grad_(False).to(device)


def aligned_crop(image, matrix):
    """The 112 x 112 aligned face. Forward value = OpenCV's warp of the rounded image (what a real face pipeline
    sees); gradient = that of the bilinear sampling of the same transform (straight-through)."""
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
    """Leave-one-out face prototype term. ``rows`` = per album photo (alignment matrix or None, clean 336 x 336
    image). ``active`` is False when the query photo or every other photo has no face (the term is then absent)."""

    def __init__(self, weights_path, rows, index, device, model=None):
        self.model = model if model is not None else load_buffalo_l(weights_path, device)
        self.matrix = rows[index][0]
        with torch.no_grad():
            others = [self.embed(aligned_crop(clean, m)) for i, (m, clean) in enumerate(rows) if i != index and m is not None]
        self.active = self.matrix is not None and bool(others)
        self.prototype = torch.stack([e[0] for e in others]).mean(0).reshape(1, 1, -1) if others else None

    def embed(self, crop):
        """Pixels to [-1, 1]; channels reversed to B,G,R before the recogniser; unit length."""
        x = crop.float().div(255).sub(0.5).div(0.5)
        return F.normalize(self.model(x.flip(1)).float(), dim=-1)

    def __call__(self, image):
        """Cosine between the photo's face embedding and the mean embedding of the other album faces."""
        return mean_token_cosine(self.embed(aligned_crop(image, self.matrix)).reshape(1, 1, -1), self.prototype)


def face_rows_from_cache(cache_json, device):
    """Recorded alignments: a JSON with ``rows`` = [{selected_matrix, clean_png}] in album order."""
    rows = []
    for row in json.loads(Path(cache_json).read_text())["rows"]:
        with Image.open(row["clean_png"]) as im:
            rows.append((row["selected_matrix"], torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1)[None].float().to(device)))
    return rows


def load_detector(detector_path):
    """SCRFD det_10g of the insightface buffalo_l pack (CPU, 640 x 640, threshold 0.5)."""
    from insightface.model_zoo import get_model
    detector = get_model(str(detector_path), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=0.5)
    return detector


def face_rows_detect(paths, device, detector_path=None, detector=None):
    """Alignments computed here: SCRFD det_10g on the rounded clean 336 x 336 image, largest face, 5-point
    similarity transform to the 112 x 112 ArcFace template."""
    import cv2
    from insightface.utils.face_align import estimate_norm
    detector = detector if detector is not None else load_detector(detector_path)
    rows = []
    for path in paths:
        clean = read_image(path, device).round().clamp(0, 255)
        boxes, marks = detector.detect(cv2.cvtColor(clean[0].byte().permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR), max_num=0)
        best = max(range(len(boxes)), key=lambda i: float((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]))) if len(boxes) else None
        rows.append((estimate_norm(marks[best], image_size=112).tolist() if best is not None else None, clean))
    return rows
