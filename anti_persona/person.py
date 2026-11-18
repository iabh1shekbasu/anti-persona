"""DeepLabV3 person masks -> 576 person-patch flags per album photo (used ONLY to choose the CES anchors).

DeepLabV3-ResNet101 with torchvision's COCO-with-VOC-labels weights (``deeplabv3_resnet101_coco-586e9e4e.pth``,
class 15 = person). Every person in the photo is marked; the target person is not separated from other people.
Per photo: person probability > 0.5 at full resolution, dilated by 2% of the long side (so hair and edges are not
clipped), squashed to 336 x 336 exactly as ``read_image`` squashes the photo; a 14 x 14 patch of the 24 x 24 CLIP grid
counts as person when at least 25% of it is person.
The segmenter is never applied to the photo being protected inside the loop and does not enter any loss.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import DILATE_FRACTION, IMAGE_SIZE, PERSON_CLASS, PERSON_PROB, TOKEN_FRACTION

DEEPLAB_URL = "https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth"
DEEPLAB_FILE = "deeplabv3_resnet101_coco-586e9e4e.pth"


def load_deeplab(weights, device):
    """``weights``: path to deeplabv3_resnet101_coco-586e9e4e.pth (scripts/download_weights.py fetches it)."""
    from torchvision.models.segmentation import deeplabv3_resnet101
    model = deeplabv3_resnet101(weights=None, weights_backbone=None, num_classes=21, aux_loss=True)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.to(device).eval()


@torch.no_grad()
def person_probability(model, rgb, device):
    """torchvision preset: shorter side to 520, ImageNet normalisation; person probability at full size."""
    h, w = rgb.shape[:2]
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255)[None].to(device)
    scale = 520 / min(h, w)
    x = F.interpolate(x, (round(h * scale), round(w * scale)), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    logits = model((x - mean) / std)["out"]
    prob = logits.softmax(1)[:, PERSON_CLASS:PERSON_CLASS + 1]
    return F.interpolate(prob, (h, w), mode="bilinear", align_corners=False)[0, 0].cpu()


def dilate(mask, radius, device):
    """Square dilation as two 1-D max-pools (identical to one k x k max-pool)."""
    k = 2 * radius + 1
    x = mask[None, None].float().to(device)
    x = F.max_pool2d(x, (k, 1), stride=1, padding=(radius, 0))
    x = F.max_pool2d(x, (1, k), stride=1, padding=(0, radius))
    return (x[0, 0] > 0.5).cpu()


@torch.no_grad()
def person_tokens(model, path, device):
    """576 booleans (24 x 24 grid, row-major) for one photo, plus the 336 x 336 mask for inspection."""
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    h, w = rgb.shape[:2]
    prob = person_probability(model, rgb, device)
    original = prob > PERSON_PROB
    original_dil = dilate(original, max(1, round(DILATE_FRACTION * max(h, w))), device)
    mask336 = F.interpolate(original_dil[None, None].float(), (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                            align_corners=False)[0, 0] > 0.5
    fraction = F.avg_pool2d(mask336[None, None].float(), 14)[0, 0].flatten()
    return fraction >= TOKEN_FRACTION, mask336


def save_tokens(path, tokens, mask336):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, tokens=tokens.numpy(), mask336=mask336.numpy())
