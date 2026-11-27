"""The four loss terms of Anti-Persona and their weighted sum.

    L = 1 * L_self + 0.5 * L_proto + 2 * L_CES + 0.125 * L_face          (minimised over the perturbation)

L_self, L_proto and L_CES are plain means over the 25 CLIP feature entries; no entry has its own weight.
L_self / L_proto are position-matched (token t vs token t, summary token + all 576 patches).
L_CES is not position-matched: every anchor is compared with all 576 patches and a soft maximum is taken.
"""
import math

import torch
import torch.nn.functional as F

from .config import MARGIN, TAU, TOP_SHARE, W_CES, W_FACE, W_PROTO, W_SELF


def mean_token_cosine(feature, reference):
    """Cosine between token t of the photo and token t of the reference, averaged over all tokens of one entry.
    The reference carries no gradient."""
    dim = feature.shape[-1]
    query = F.normalize(feature.reshape(-1, dim), dim=-1)
    target = F.normalize(reference.reshape(-1, dim), dim=-1).detach()
    return F.cosine_similarity(query, target).mean()


def self_term(features, clean_features):
    """L_self: reference = the clean photo itself."""
    return torch.stack([mean_token_cosine(f, c) for f, c in zip(features, clean_features)]).mean()


def prototype_term(features, prototype):
    """L_proto: reference = the leave-one-out album mean at the same grid position (all 577 tokens, no mask)."""
    return torch.stack([mean_token_cosine(f, p) for f, p in zip(features, prototype)]).mean()


def patch_response(anchors, patches, temperature=TAU):
    """r_k = tau * (logsumexp_j(cos(a_k, f_j) / tau) - log P): a soft maximum over the photo's P patches."""
    similarity = F.normalize(anchors, dim=-1) @ F.normalize(patches, dim=-1).T
    return temperature * (torch.logsumexp(similarity / temperature, dim=-1) - math.log(len(patches)))


def ces_term(features, bank):
    """L_CES. Per entry: penalty_k = tau * softplus((r_k - threshold_k + m) / tau), mean over the strongest 25% of
    the anchors; then a plain mean over the 25 entries. ``bank[l] = (anchors [N, 1024], thresholds [N])``."""
    terms = []
    for feature, (anchors, thresholds) in zip(features, bank):
        response = patch_response(anchors, feature[1:, 0, :])                 # patches only, no summary token
        penalty = TAU * F.softplus((response - thresholds + MARGIN) / TAU)    # smooth max(0, r - threshold + m)
        terms.append(penalty.topk(max(1, math.ceil(len(penalty) * TOP_SHARE))).values.mean())
    return torch.stack(terms).mean()


def loss_terms(view, encoder, clean_features, prototype, bank, face):
    """Returns (L, dict of the four raw terms). The sum is formed in the order of the code that made the paper's
    images (self, then prototype + CES, then face), so the floating-point result is identical."""
    features = encoder(view)
    l_self, l_proto, l_ces = self_term(features, clean_features), prototype_term(features, prototype), ces_term(features, bank)
    loss = W_SELF * l_self
    loss = loss + W_PROTO * l_proto + W_CES * l_ces
    l_face = face(view) if face is not None and face.active else None
    if l_face is not None:
        loss = loss + W_FACE * l_face
    return loss, dict(self_raw=l_self, prototype_raw=l_proto, evidence_mean_raw=l_ces,
                      face_raw=l_face if l_face is not None else 0.0)
