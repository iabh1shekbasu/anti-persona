"""A small stand-in for the CLIP encoder, for the CPU test of the LOOP only (test code; not part of the method).

Why: a long CPU run of the loop with the real CLIP ViT-L/14-336 needs far more time and memory than a quick test
should. The loop itself (JPEG draw, smoothing every third step, sign step, clamp, rounding)
does not care which encoder it is given, so it is tested with this cheap one; the real encoder is tested separately on
the loss values and on the step-1 gradient.

Same interface as the real one: image [1, 3, 336, 336] in [0, 255] -> 25 entries, each [577, 1, 1024] (token 0 = summary
token). Fixed pseudo-random weights from a seeded CPU generator, so two separate processes build the very same network.
"""
import torch
import torch.nn.functional as F


class StandIn(torch.nn.Module):
    def __init__(self, seed=20261115):
        super().__init__()
        g = torch.Generator(device='cpu')
        g.manual_seed(seed)
        self.register_buffer('patch', torch.randn(1024, 3, 14, 14, generator=g) * 0.05)
        self.register_buffer('cls', torch.randn(1, 1, 1024, generator=g) * 0.5)
        self.register_buffer('pos', torch.randn(1, 577, 1024, generator=g) * 0.2)
        self.register_buffer('u', torch.randn(24, 1024, 16, generator=g) / 32)
        self.register_buffer('v', torch.randn(24, 16, 1024, generator=g) / 4)

    def forward(self, image):
        x = F.conv2d((image - 127.5) / 127.5, self.patch, stride=14)           # [1, 1024, 24, 24]
        x = x.reshape(1, 1024, 576).permute(0, 2, 1)
        x = torch.cat((self.cls, x), dim=1) + self.pos
        features = [x.permute(1, 0, 2).contiguous()]
        for layer in range(24):
            mixed = x + 0.5 * x.mean(1, keepdim=True)                           # every token sees the whole image
            x = x + torch.tanh(mixed @ self.u[layer]) @ self.v[layer]
            features.append(x.permute(1, 0, 2).contiguous())
        return features
