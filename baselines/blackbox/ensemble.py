"""Black-box encoder: our self term on a leave-one-out surrogate ensemble, with dynamic scaling.

Call draw() ONCE per optimisation step; it draws ratio ~ U[scale_min, scale_max) (default U[0.5, 1)) from this object's
own CPU generator (seeded by reseed(image_seed)) and sets size = max(16, int(ratio * 336)). Every call until the next
draw() -- the adversarial image AND the clean reference -- is bicubic-resized (align_corners=False, no clamp) to that
size before the surrogates see it, so adversarial and reference features are always computed at the same scale.
Surrogates with a fixed input size (SigLIP 384, ConvNeXtV2 224, supervised ViT 224) then re-resize to it; the
CLIP-family ViTs (A, B, D) run at the scaled size with bicubically interpolated position embeddings.

    encoder = EnsembleEncoder.for_combo('F6_noA', device='cuda')  # dynamic scaling on (default)
    encoder.reseed(image_seed)                                    # once per photo, before the loop
    for step in range(steps):
        encoder.draw()                                            # one scale per step
        features = encoder(view)                                  # adversarial view, 336 x 336, 0-255
        with torch.no_grad():
            clean = encoder(clean_image)                          # same scale
        loss = encoder.self_term(features, clean)
The self term per surrogate is our CLIP self term (per entry the mean token cosine between adversarial and clean
features, CLS included where the model has one, averaged over the surrogate's entries); self_term then takes the
equal-weight mean over the surrogates.
"""
import torch
import torch.nn.functional as F

from . import surrogates as surrogate_module

WORK = 336


class EnsembleFeatures(list):
    """Flat list of feature entries (all surrogates, in member order) + groups [(key, start, stop)] + size."""

    def __init__(self, entries, groups, size):
        super().__init__(entries)
        self.groups, self.size = groups, size

    def split(self):
        return [self[a:b] for _, a, b in self.groups]


def entry_self_term(features, clean_features):
    """Our CLIP self term for ONE encoder: per entry mean token cosine, then mean over entries."""
    if not features or len(features) != len(clean_features):
        raise ValueError('Every feature entry needs its clean reference.')
    terms = []
    for feature, clean in zip(features, clean_features):
        dimensions = feature.shape[-1]
        query = F.normalize(feature.reshape(-1, dimensions), dim=-1)
        reference = F.normalize(clean.reshape(-1, dimensions), dim=-1).detach()
        terms.append(F.cosine_similarity(query, reference).mean())
    return torch.stack(terms).mean()


def ensemble_mean(values):
    """Equal-weight mean over surrogates; one member is returned unchanged ."""
    return values[0] if len(values) == 1 else torch.stack(values).mean()


class EnsembleEncoder(torch.nn.Module):
    def __init__(self, members, dynamic=True, scale_min=0.5, scale_max=1.0):
        super().__init__()
        keys = [m.key for m in members]
        if not members or len(set(keys)) != len(keys):
            raise ValueError(f'need distinct members, got {keys}')
        if not 0 < scale_min <= scale_max <= 1:
            raise ValueError('scale range must lie in (0, 1]')
        self.members = torch.nn.ModuleList(members)
        self.keys, self.dynamic, self.scale_min, self.scale_max = keys, dynamic, scale_min, scale_max
        self.generator = torch.Generator(device='cpu')
        self.size, self.ratio, self.drawn, self.draws = WORK, 1.0, False, 0
        self.log = []           # (draw index, size, batch) of every forward since reseed (tests check adv/refs agree)

    @classmethod
    def for_combo(cls, combo, device='cuda', **kwargs):
        return cls(surrogate_module.load(surrogate_module.COMBOS[combo], device), **kwargs)

    def reseed(self, seed):
        self.generator.manual_seed(int(seed))
        self.size, self.ratio, self.drawn, self.draws = WORK, 1.0, False, 0
        self.log = []

    def draw(self):
        """Draw this step's scale (no-op at 336 when dynamic scaling is off). Returns the size."""
        if self.dynamic:
            self.ratio = torch.rand(1, generator=self.generator).item() * (self.scale_max - self.scale_min) + self.scale_min
            self.size = max(16, int(self.ratio * WORK))
        else:
            self.ratio, self.size = 1.0, WORK
        self.drawn = True
        self.draws += 1
        return self.size

    def scaled(self, image):
        if image.shape[-2:] != (WORK, WORK):
            raise ValueError(f'EnsembleEncoder takes the 336 x 336 working image, got {tuple(image.shape)}')
        if self.dynamic and not self.drawn:
            raise RuntimeError('call reseed(seed) and then draw() once per step before encoding')
        if not self.dynamic:
            return image
        return F.interpolate(image, (self.size, self.size), mode='bicubic', align_corners=False)

    def member_features(self, image):
        """[[entries of member 0], [entries of member 1], ...] at the current scale."""
        x = self.scaled(image)
        self.log.append((self.draws, self.size, x.shape[0]))
        out = []
        for m in self.members:
            features = m(x)
            layout = m.layout(self.size)
            if len(features) != len(layout):
                raise ValueError(f'{m.key}: {len(features)} entries, layout says {len(layout)}')
            out.append(features)
        return out

    def forward(self, image):
        per = self.member_features(image)
        entries, groups = [], []
        for m, features in zip(self.members, per):
            groups.append((m.key, len(entries), len(entries) + len(features)))
            entries += features
        return EnsembleFeatures(entries, groups, self.size)

    def self_term(self, features, clean_features):
        if features.groups != clean_features.groups or features.size != clean_features.size:
            raise ValueError('adversarial and clean features come from different members or scales')
        return ensemble_mean([entry_self_term(f, c) for f, c in zip(features.split(), clean_features.split())])

    def layouts(self):
        return [m.layout(self.size) for m in self.members]

    def identity(self):
        return dict(members=[m.identity() for m in self.members], dynamic=self.dynamic, scale_min=self.scale_min,
                    scale_max=self.scale_max, size_rule='max(16, int(ratio*336)), ratio ~ U[min, max) from a CPU '
                    'generator seeded with the image seed', resize='bicubic, align_corners=False, before the surrogate')
