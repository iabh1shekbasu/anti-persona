# REFERENCE COPY (anti-persona release): the generator that produced the paper's baseline face-term images, kept for its exact
# procedure and settings. Only absolute paths were replaced by environment variables; the logic is unchanged.
# It runs inside the research harness it was written for (a queue PLAN.json, the legacy baseline module and the
# stock wrapper); see baselines/README.md, section "DiffAttack and WDP".
"""Baselines + INSTANCE-LEVEL face term. The prototype is part of our method, so a baseline gets only the face term,
with the photo's own face as target (no prototype of faces).

For a baseline the optimised objective becomes

    L_baseline(x)  +  w * cos( buffalo_l(aligned_crop(x, M)),  buffalo_l(aligned_crop(x_clean, M)) )

  x        the protected image of ONE training photo (pixel units, 336 x 336), as the baseline's own loop builds it
  x_clean  THE SAME PHOTO's clean start point, round(attack.read_image(source)) (= FACE.prepare_alignments' row['clean'])
  M        the SCRFD det_10g 5-point landmark matrix of x_clean (FACE.prepare_alignments, largest face; the same function,
           detector file and input Ours' proactive generator uses), aligned 112 x 112 crop by FACE.aligned_crop
  buffalo_l  face_bl.load_buffalo_l of LEGACY_FACE_BL_DIR (w600k_r50, torch executor), behind the colour-fix wrapper
           channel_order.ChannelOrder(order='bgr') of LEGACY_CHANNEL_ORDER_DIR = the order MyVLM's face gate feeds the
           network
  cos      attack.prototype_cosine([q], [t]) with ONE reference = the plain cosine of the two unit embeddings (the same
           function Ours' face term calls; here the reference is the photo's own clean embedding, NOT a prototype)
  w        0.125 (the number Ours uses) or 1/28 = 0.125 / 3.5 (the share the face term has in the final method)

Minimised, so the face embedding is pushed away from the photo's own clean face (the sign of Ours' face term).
No album, no other photo, no prototype, no leave-one-out mean. A photo without a detected face gets no face term
(face.active False: the baseline's own loss tensor is returned unchanged, as Ours does).

Where the face term sees the image (the alignment matrix is valid only in the photo's own geometry):
  plain loops (I-FGSM, PGD, R+FGSM, TI-MI-FGSM)  the image the loop hands the objective (= the photo)
  DI-FGSM   the UNTRANSFORMED adversarial image of the same step (hook on rules.di_transform), CLIP self term on the DI view
  SIA       the UNTRANSFORMED image of the same step (hook on sia.blocktransform: x * 255 with x = image / 255), each of
            the 20 copies adds w * face / 20 (rules.protect divides every copy's loss by 20) = w * face per step
  GLEAM     one extra backward per step of w * face(clean + delta) next to the 40 view copies (their mean self term)
  DiffAttack / WDP  inside the replaced 'our loss' call, on the 336 photo that call scores (DiffAttack: projected decode
            view; WDP: projected pasted-back photo); the stock multiplier (10 / 0.003) then scales self + w * face alike
"""
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent                      # this code folder
BLOCK = HERE.parent
import os  # noqa: E402
FIX1 = Path(os.environ['LEGACY_FACE_BL_DIR'])              # face_bl.py + buffalo_l_w600k_r50_torch.pt (research harness)
COMMON = Path(os.environ['LEGACY_CHANNEL_ORDER_DIR'])      # channel_order.py (colour-fix wrapper; research harness)
DETECTOR = Path.home() / '.insightface/models/buffalo_l/det_10g.onnx'
WEIGHTS = {'w125': 0.125, 'w28': 0.03571428571428571}       # 0.125 = Ours' number; 1/28 = 0.125 / 3.5 (final-method share)
assert WEIGHTS['w28'] == 0.125 / 3.5 == 1 / 28
ORDER = 'bgr'
for p in (str(FIX1), str(COMMON)):
    if p not in sys.path:
        sys.path.insert(0, p)
import channel_order as CO      # noqa: E402


def check_weight(tag, weight):
    """Every generation task asserts its face weight against its tag."""
    assert tag in WEIGHTS and weight == WEIGHTS[tag], (tag, weight)
    assert (weight == 0.125) if tag == 'w125' else (weight == 1 / 28 == 0.125 / 3.5), (tag, weight)
    return weight


def load_recogniser(device):
    """buffalo_l w600k_r50 behind ChannelOrder('bgr'), built through face_bl.load_buffalo_l (colour fix installed)."""
    import face_bl as FB
    assert Path(FB.__file__).resolve() == FIX1 / 'face_bl.py', FB.__file__
    CO.self_test()
    CO.install(FB, ORDER)
    built = CO.CALLS['built']
    model = FB.load_buffalo_l(device)
    assert isinstance(model, CO.ChannelOrder) and model.order == ORDER and CO.CALLS['built'] == built + 1
    return model, FB


def detect_matrix(source, device):
    """SCRFD det_10g matrix of round(attack.read_image(source)) via FACE.prepare_alignments (Ours' function), or None."""
    from anti_persona import face as FACE
    rows = FACE.prepare_alignments([Path(source)], device, DETECTOR)
    assert len(rows) == 1
    return rows[0]['selected_matrix']


class InstanceFace:
    """cos(buffalo_l(crop(x)), buffalo_l(crop(x_clean))) for ONE photo; target = that photo's own clean embedding."""

    def __init__(self, source, matrix, model, device):
        from anti_persona import attack
        from anti_persona import face as FACE
        self.FACE, self.attack = FACE, attack
        self.source, self.model, self.matrix = str(source), model, matrix
        self.clean = attack.read_image(Path(source), device).round().clamp(0, 255)   # == prepare_alignments row['clean']
        self.active = matrix is not None
        self.target = None
        if self.active:
            with torch.no_grad():
                self.target = FACE.encode(model, 'buffalo_l', FACE.aligned_crop(self.clean, matrix))[0].detach()
            assert self.target.shape == (512,) and abs(float(self.target.norm()) - 1) < 1e-4

    def embed(self, image):
        return self.FACE.encode(self.model, 'buffalo_l', self.FACE.aligned_crop(image, self.matrix))

    def loss(self, image):
        if not self.active:
            return image.sum() * 0
        q = self.embed(image)
        return self.attack.prototype_cosine([q.reshape(1, 1, -1)], [self.target.reshape(1, 1, -1)])

    def self_check(self):
        """The target is the clean embedding of THIS photo: loss(clean) == 1 (cosine of an embedding with itself)."""
        if not self.active:
            return None
        with torch.no_grad():
            value = float(self.loss(self.clean))
            direct = float((self.embed(self.clean)[0] * self.target).sum())
        assert abs(value - 1.0) < 1e-4 and abs(direct - value) < 1e-5, (self.source, value, direct)
        return value

    def record(self):
        return dict(term='instance: cos(buffalo_l(aligned_crop(x, M)), buffalo_l(aligned_crop(x_clean, M))), minimised',
                    target='the clean embedding of the SAME photo (round(attack.read_image(source))); no album, no '
                           'prototype, no leave-one-out mean', source=self.source, matrix=self.matrix,
                    active=bool(self.active), detector=str(DETECTOR), recogniser='fix1_gate_face/face_bl.load_buffalo_l '
                    '(w600k_r50) behind channel_order.ChannelOrder', face_channels=getattr(self.model, 'order', None))


class FaceSelf:
    """The baseline's own loss backend + c * InstanceFace.loss (begin / step / __call__ / record, as rules.protect,
    gleam and the DiffAttack / WDP loss swaps call it). With an inactive face the baseline tensor itself is returned.

    face_weight = w (0.125 or 1/28) = the coefficient of the face term in the baseline's TOTAL objective.
    outer = the multiplier the baseline itself applies to this backend's output (1 for the noise baselines; DiffAttack
    attack_loss_weight 10; WDP adv_optim_weight 0.003). The coefficient used inside is c = w / outer, so the optimised
    objective is  outer * self + w * face + (the baseline's other terms)  = L_baseline + w * face."""

    def __init__(self, base, face, weight, tag, outer=1.0):
        self.base, self.face, self.tag = base, face, tag
        self.face_weight = check_weight(tag, float(weight))
        self.outer = float(outer)
        assert self.outer > 0
        self.weight = self.face_weight / self.outer                    # internal coefficient c
        assert abs(self.weight * self.outer - self.face_weight) < 1e-12
        self.calls, self.plain, self.iteration = [], None, None

    def begin(self, clean):
        self.calls, self.plain, self.iteration = [], None, None
        return self.base.begin(clean)

    def step(self, iteration):
        self.iteration = iteration
        return self.base.step(iteration)

    def note_plain(self, image):
        self.plain = image

    def face_term(self, view):
        return self.face.loss(view) if self.face.active else None

    def __call__(self, image):
        self_term = self.base(image)
        plain, self.plain = self.plain, None
        view = image if plain is None else plain
        face_term = self.face_term(view)
        total = self_term if face_term is None else self_term + self.weight * face_term
        if torch.is_grad_enabled():
            grad = 0.0
            if face_term is not None:
                (g,) = torch.autograd.grad(face_term, view, retain_graph=True)
                grad = float(g.double().norm())
            self.calls.append(dict(step=None if self.iteration is None else self.iteration + 1, total=float(total.detach()),
                                   self_raw=float(self_term.detach()),
                                   face_raw=float(face_term.detach()) if face_term is not None else 0.0,
                                   face_view='plain' if plain is not None else 'same', face_grad_l2=grad))
        return total

    def record(self):
        base = self.base.record()
        return dict(preset='baseline+instance_face', weights=dict(baseline=1.0, face=self.face_weight), face_weight=self.face_weight,
                    outer_multiplier=self.outer, internal_face_coefficient=self.weight,
                    objective='outer * (self + internal * face) = outer * self + face_weight * face',
                    weight_tag=self.tag, face=self.face.record(), baseline_loss=base)


def check_calls(calls, weight, active, expect_steps=None):
    """Per optimisation call: total == self + c * face (relative 1e-6, c = the internal coefficient), every value finite;
    exactly the baseline's loss when the face is inactive. When active, the face term must be connected to the image:
    its gradient > 0 on at least one call (a zero cosine / zero gradient on ONE call is valid, e.g. at the
    clean start where the cosine is at its maximum, so it is telemetry, not a failure). Returns the worst residual."""
    worst = 0.0
    assert calls, 'no optimisation call recorded'
    if expect_steps is not None:
        assert len(calls) == expect_steps, (len(calls), expect_steps)
    for c in calls:
        assert all(math.isfinite(c[k]) for k in ('total', 'self_raw', 'face_raw', 'face_grad_l2')), c
        if not active:
            assert c['face_raw'] == 0.0 and c['face_grad_l2'] == 0.0 and c['total'] == c['self_raw'], c
        worst = max(worst, abs(c['total'] - (c['self_raw'] + weight * c['face_raw'])) / max(1.0, abs(c['total'])))
    if active:
        assert any(c['face_grad_l2'] > 0.0 for c in calls), 'face term never had a gradient'
    assert worst < 1e-5, ('total != self + face coefficient * face', worst)
    return worst


def install_di_hook(rules, objective):
    original = rules.di_transform
    assert original.__name__ == 'di_transform', original

    def di_transform(image):
        objective.note_plain(image)
        return original(image)
    rules.di_transform = di_transform
    return original


def install_sia_hook(sia, objective):
    """rules.protect calls objective(sia.blocktransform(image / 255) * 255): note x * 255 (= the untransformed image)."""
    original = sia.blocktransform

    def blocktransform(x):
        objective.note_plain(x * 255)
        return original(x)
    sia.blocktransform = blocktransform
    return original


def face_rows_file():
    return BLOCK / 'gen' / 'FACE_ROWS.json'


def cached_matrix(plan, album, photo, source, source_sha256):
    """DiffAttack / WDP (their envs have no insightface): the SCRFD matrix the noise-baseline tasks detected in process
    for this photo, collected by the orchestrator into FACE_ROWS.json (pinned by sha256 in the queue PLAN)."""
    path = Path(plan['face_rows'])
    import hashlib
    assert hashlib.sha256(path.read_bytes()).hexdigest() == plan['face_rows_sha256'], 'FACE_ROWS.json changed'
    rows = json.loads(path.read_text())['rows']
    row = rows[f'{album}/{photo}']
    assert row['source'] == str(source) and row['source_sha256'] == source_sha256, (row, source)
    return row['matrix']
