"""Image loading and the frozen CLIP ViT-L/14-336 vision tower (the encoder shared by Yo'LLaVA and MyVLM)."""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .config import CLIP_MEAN, CLIP_NAME, CLIP_STD, IMAGE_SIZE


def read_image(path, device):
    """RGB, squashed to 336 x 336 (bicubic, no aspect-ratio preservation), float values in [0, 255]."""
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))
    x = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div(255).mul(255)
    return F.interpolate(x[None].to(device), (IMAGE_SIZE, IMAGE_SIZE), mode="bicubic", align_corners=False).clamp(0, 255)


class ClipEncoder(torch.nn.Module):
    """CLIP ViT-L/14-336 vision tower. ``forward(image)`` takes [1, 3, 336, 336] in [0, 255] and returns 25 entries,
    each [577, 1, 1024] (token 0 = summary token): entry 0 = the tokens after the pre-transformer layer norm,
    entries 1..24 = the output of each transformer block. Weights are frozen."""

    def __init__(self, device, name=CLIP_NAME):
        super().__init__()
        from transformers import CLIPModel  # lazy: keeps `--help` and the CPU tests free of the download
        self.vision = CLIPModel.from_pretrained(name).to(device).float().eval().vision_model.requires_grad_(False)
        self.register_buffer("mean", torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1) * 255)

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
