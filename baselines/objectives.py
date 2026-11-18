"""Loss backends of the baselines.

Every baseline optimises the INSTANCE part of our loss (the CLIP self term: mean token cosine between the protected
photo's CLIP ViT-L/14-336 features and its own clean features, averaged over the 25 feature entries) with its own
update rule. No prototype, no evidence term, no JPEG in the loop, no gradient smoothing: those are our contribution.
In the final tables every baseline also gets an INSTANCE face term:

    L = L_self + w * cos( buffalo_l(aligned_crop(x, M)), buffalo_l(aligned_crop(x_clean, M)) ),   w = 1/28

x_clean = the SAME photo's rounded clean start, M = the SCRFD det_10g 5-point alignment of x_clean (largest face),
buffalo_l w600k_r50 fed B,G,R (as MyVLM's face gate). No album and no prototype; no detected face -> no face term.
w = 1/28 = 0.125 / 3.5: our face weight divided by the sum of our CLIP-term weights (1 + 0.5 + 2), i.e. the share
the face term has in the final method. DiffAttack / WDP multiply our loss by their own adversarial weight a
(10 / 0.003); the face coefficient is scaled with it (a * w), so the face term has the same share for every baseline.

Interface shared by all backends (called by rules.protect):
  begin(clean)      once per run, before the first step
  step(iteration)   once per optimisation step, before the loss calls of that step
  __call__(image)   scalar loss to MINIMISE for one [B, 3, 336, 336] view in pixel units [0, 255]
"""
import math
from pathlib import Path

import torch

from anti_persona.encoder import ClipEncoder, read_image
from anti_persona.face import aligned_crop, face_rows_detect, load_buffalo_l
from anti_persona.losses import mean_token_cosine, self_term

FACE_WEIGHT = 1 / 28
assert FACE_WEIGHT == 0.125 / 3.5


class SelfObjective:
    """White-box: our CLIP self term on CLIP ViT-L/14-336 (Yo'LLaVA's and MyVLM's vision tower)."""

    def __init__(self, encoder):
        self.encoder, self.clean_features = encoder, None

    @classmethod
    def load(cls, device):
        return cls(ClipEncoder(device))

    def begin(self, clean):
        with torch.no_grad():
            self.clean_features = self.encoder(clean)

    def step(self, iteration):
        return None

    def __call__(self, image):
        return self_term(self.encoder(image), self.clean_features)

    def batched(self, views):
        """Self term of a [B, 3, 336, 336] batch against the clean features repeated B times = mean of the B per-view
        self terms (used by GLEAM-v's white-box loop)."""
        clean = [c.expand(c.shape[0], views.shape[0], c.shape[2]) for c in self.clean_features]
        return self_term(self.encoder(views), clean)


class EnsembleSelfObjective:
    """Black-box: our self term on a leave-one-out surrogate ensemble with dynamic scaling (blackbox/ensemble.py).
    Set .seed to the image seed before each run (begin() reseeds the scale generator with it). Each step draws one
    scale; the clean reference is re-encoded at that scale (no gradient)."""

    def __init__(self, encoder):
        self.encoder, self.seed, self.clean, self.clean_features, self.calls = encoder, None, None, None, 0

    def begin(self, clean):
        assert self.seed is not None, 'set the image seed first'
        self.encoder.reseed(self.seed)
        self.clean, self.clean_features, self.calls = clean, None, 0

    def step(self, iteration):
        self.encoder.draw()
        with torch.no_grad():
            self.clean_features = self.encoder(self.clean)

    def __call__(self, image):
        self.calls += 1
        return self.encoder.self_term(self.encoder(image), self.clean_features)


class InstanceFace:
    """cos(buffalo_l(crop(x)), buffalo_l(crop(x_clean))) for ONE photo; target = that photo's own clean embedding."""

    def __init__(self, source, face_weights, detector, device, matrix='detect'):
        self.source = str(source)
        self.model = load_buffalo_l(face_weights, device)
        self.clean = read_image(Path(source), device).round().clamp(0, 255)
        if matrix == 'detect':
            matrix = face_rows_detect([Path(source)], device, detector_path=detector)[0][0]
        self.matrix = matrix
        self.active = matrix is not None
        self.target = None
        if self.active:
            with torch.no_grad():
                self.target = self.embed(self.clean)[0].detach()

    def embed(self, image):
        """Aligned 112 x 112 crop -> [-1, 1] -> B,G,R -> buffalo_l -> unit length."""
        crop = aligned_crop(image, self.matrix)
        x = crop.float().div(255).sub(0.5).div(0.5)
        return torch.nn.functional.normalize(self.model(x.flip(1)).float(), dim=-1)

    def loss(self, image):
        if not self.active:
            return image.sum() * 0
        return mean_token_cosine(self.embed(image).reshape(1, 1, -1), self.target.reshape(1, 1, -1))

    def self_check(self):
        """loss(clean) == 1: the target is this photo's own clean embedding."""
        if not self.active:
            return None
        with torch.no_grad():
            value = float(self.loss(self.clean))
        assert abs(value - 1.0) < 1e-4, (self.source, value)
        return value


class FaceSelf:
    """A baseline's loss backend + c * InstanceFace.loss. `outer` = the multiplier the baseline itself applies to this
    backend's output (1 for the noise baselines; DiffAttack 10, WDP 0.003); c = w / outer, so the optimised objective is
    outer * L_self + w * L_face. For DI-FGSM / SIA / GLEAM-v the face term sees the UNTRANSFORMED image of the step
    (note_plain), because the alignment matrix is valid only in the photo's own geometry."""

    def __init__(self, base, face, weight=FACE_WEIGHT, outer=1.0):
        self.base, self.face = base, face
        self.face_weight, self.outer = float(weight), float(outer)
        self.weight = self.face_weight / self.outer
        self.calls, self.plain, self.iteration = [], None, None

    def begin(self, clean):
        self.calls, self.plain, self.iteration = [], None, None
        return self.base.begin(clean)

    def step(self, iteration):
        self.iteration = iteration
        return self.base.step(iteration)

    def note_plain(self, image):
        self.plain = image

    def __call__(self, image):
        base = self.base(image)
        plain, self.plain = self.plain, None
        view = image if plain is None else plain
        face = self.face.loss(view) if self.face.active else None
        total = base if face is None else base + self.weight * face
        if torch.is_grad_enabled():
            self.calls.append(dict(step=None if self.iteration is None else self.iteration + 1, total=float(total.detach()),
                                   self_raw=float(base.detach()), face_raw=float(face.detach()) if face is not None else 0.0,
                                   face_view='plain' if plain is not None else 'same'))
        return total


def check_calls(calls, weight, active):
    """Every call: total == self + c * face (relative 1e-5), finite; exactly the baseline loss without a face."""
    assert calls, 'no optimisation call recorded'
    for c in calls:
        assert all(math.isfinite(c[k]) for k in ('total', 'self_raw', 'face_raw')), c
        if not active:
            assert c['face_raw'] == 0.0 and c['total'] == c['self_raw'], c
        assert abs(c['total'] - (c['self_raw'] + weight * c['face_raw'])) <= 1e-5 * max(1.0, abs(c['total'])), c
