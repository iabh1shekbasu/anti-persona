"""Update rules of the eight epsilon-bounded baselines (final settings of the paper) and their optimisation loop.

Shared loop conventions (identical to our method's loop): pixel units [0, 255]; the state delta is clamped to +-4 after
every step and the encoder sees (clean + delta.clamp(-4, 4)).clamp(0, 255); descent on the loss
(delta <- delta - step * direction); the saved PNG is quantize_bounded (whole numbers within 4 of the float and of the
rounded clean image). eps = 4/255 for every rule. The anchors of the self term are always the clean photo's features.

  rule       iterations x step   start                          update / view
  rfgsm      1 x 2/255           2/255 * sign(N(0, I))          one signed step (Tramer et al. 2018)
  ifgsm      100 x 0.1/255       zero                           sign (Kurakin et al. 2017)
  difgsm     100 x 0.1/255       zero                           sign; input diversity p 0.5, resize up to 1.15x +
                                                                random zero pad, back to 336 (Xie et al. 2019)
  ti_mifgsm  100 x 0.1/255       zero                           TIM 15x15 Gaussian (nsig 3) on the gradient, then
                                                                momentum mu 1 on g / mean|g|, sign (Dong et al. 2019)
  pgd        100 x 0.1/255       U(-4, 4)                       sign (Madry et al. 2018)
  sia        10 x 0.4/255        zero                           20 block-transformed copies (3x3 blocks, the authors'
                                                                7 ops), mean loss, momentum mu 1, sign (Wang et al. 2023)
  gleam      10 x 3/255          zero                           GLEAM-v: 20 GDE + 20 LFE copies, TI 5x5 (sigma 3),
                                                                momentum mu 1, sign (Liu et al. 2025, vision only)

Step counts: I-FGSM, DI-FGSM, TI-MI-FGSM and PGD use a common 100 x 0.1/255; SIA (T = 10, alpha = eps / T) and GLEAM-v
(T = 10, alpha = 0.75 eps as in the official eval_gleam.py) use their papers' settings; R+FGSM is its single step.
Random draws: CPU generator seeded with the image seed for the PGD / R+FGSM start; the device RNG after
seed_everything(image seed) for the DI / SIA / GLEAM draws. Image seed = run seed (0 or 202) + the photo's index in its
album (the same rule as our method).
"""
import torch
import torch.nn.functional as F

from anti_persona.optim import quantize_bounded, seed_everything

from . import transfer_ops as T

EPSILON = 4
DECAY = 1.0
RULES = {
    'rfgsm':     dict(iterations=1, step=2.0, start='rsign', alpha=2.0, momentum=False, kernel=None, transform=None),
    'ifgsm':     dict(iterations=100, step=0.1, start='zero', momentum=False, kernel=None, transform=None),
    'difgsm':    dict(iterations=100, step=0.1, start='zero', momentum=False, kernel=None, transform='di'),
    'ti_mifgsm': dict(iterations=100, step=0.1, start='zero', momentum=True, kernel='tim15', transform=None),
    'pgd':       dict(iterations=100, step=0.1, start='uniform', momentum=False, kernel=None, transform=None),
    'sia':       dict(iterations=10, step=0.4, start='zero', momentum=True, kernel=None, transform='sia'),
    'gleam':     dict(iterations=10, step=3.0, start='zero', momentum=True, kernel='gleam_ti5', transform='gleam'),
}
LABELS = {'rfgsm': 'R+FGSM', 'ifgsm': 'I-FGSM', 'difgsm': 'DI-FGSM', 'ti_mifgsm': 'TI-MI-FGSM', 'pgd': 'PGD',
          'sia': 'SIA', 'gleam': 'GLEAM-v'}
DETERMINISTIC = ('ifgsm', 'ti_mifgsm')      # white-box: one run, counted for both image seeds
DI = dict(prob=0.5, resize_rate=1.15)
GLEAM_CHUNK = 5                             # white-box GLEAM-v: copies encoded per batch (memory only)


def image_seed(run_seed, album_index):
    return run_seed + album_index


def di_transform(image, prob=DI['prob'], resize_rate=DI['resize_rate']):
    """DI-FGSM input diversity on the adversarial image (device RNG): with probability `prob` bicubic resize to a random
    (h, w) in [H, round(1.15 H)] x [W, round(1.15 W)], zero pad at a random offset to round(1.15 H) x round(1.15 W),
    bicubic back to H x W; otherwise unchanged."""
    if torch.rand(1, device=image.device).item() > prob:
        return image
    _, _, H, W = image.shape
    max_h = max(H + 1, int(round(H * resize_rate)))
    max_w = max(W + 1, int(round(W * resize_rate)))
    rnd_h = torch.randint(low=H, high=max_h + 1, size=(1,), device=image.device).item()
    rnd_w = torch.randint(low=W, high=max_w + 1, size=(1,), device=image.device).item()
    x = F.interpolate(image, size=(rnd_h, rnd_w), mode='bicubic', align_corners=False)
    pad_h, pad_w = max_h - rnd_h, max_w - rnd_w
    pad_top = torch.randint(low=0, high=pad_h + 1, size=(1,), device=image.device).item()
    pad_left = torch.randint(low=0, high=pad_w + 1, size=(1,), device=image.device).item()
    x = F.pad(x, (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top), mode='constant', value=0.0)
    return F.interpolate(x, size=(H, W), mode='bicubic', align_corners=False)


def _note(objective, image):
    if hasattr(objective, 'note_plain'):
        objective.note_plain(image)


def protect(clean, objective, rule, seed, sia=None):
    """One run of `rule` (not gleam) on `clean` ([1, 3, 336, 336] float pixels). Returns (uint8 PNG tensor, trace)."""
    spec = RULES[rule]
    assert rule != 'gleam', 'use gleam_protect'
    if spec['transform'] == 'sia':
        assert isinstance(sia, T.SIATransform)
    kernel = T.tim_kernel(clean.device).to(clean.dtype) if spec['kernel'] == 'tim15' else None
    seed_everything(seed)
    objective.begin(clean)
    if spec['start'] == 'uniform':
        generator = torch.Generator(device='cpu').manual_seed(seed)
        start = torch.empty(clean.shape, dtype=clean.dtype).uniform_(-EPSILON, EPSILON, generator=generator)
        delta = start.to(clean.device).requires_grad_(True)
    elif spec['start'] == 'rsign':
        generator = torch.Generator(device='cpu').manual_seed(seed)
        start = spec['alpha'] * torch.randn(clean.shape, dtype=clean.dtype, generator=generator).sign()
        delta = start.to(clean.device).requires_grad_(True)
    else:
        delta = torch.zeros_like(clean, requires_grad=True)
    momentum = torch.zeros_like(clean)
    trace = []
    for iteration in range(spec['iterations']):
        objective.step(iteration)
        if delta.grad is not None:
            delta.grad.zero_()
        if spec['transform'] == 'sia':
            loss_value, copies = 0., sia.num_scale
            for _ in range(copies):
                image = (clean + delta.clamp(-EPSILON, EPSILON)).clamp(0, 255)
                x = image / 255
                _note(objective, x * 255)          # the face term (if any) sees the untransformed image
                loss = objective(sia.blocktransform(x) * 255) / copies
                loss.backward()
                loss_value += float(loss.detach())
        else:
            copies = 1
            image = (clean + delta.clamp(-EPSILON, EPSILON)).clamp(0, 255)
            if spec['transform'] == 'di':
                _note(objective, image)
                image = di_transform(image)
            loss = objective(image)
            loss.backward()
            loss_value = float(loss.detach())
        if not torch.isfinite(delta.grad).all():
            raise FloatingPointError('Non-finite attack gradient.')
        with torch.no_grad():
            gradient = delta.grad
            if kernel is not None:
                gradient = F.conv2d(gradient, kernel, stride=1, padding='same', groups=3)
            if spec['momentum']:
                momentum = DECAY * momentum + gradient / (gradient.abs().mean() + 1e-12)
                direction = momentum.sign()
            else:
                direction = gradient.sign()
            delta.copy_((delta - spec['step'] * direction).clamp(-EPSILON, EPSILON))
        trace.append(dict(step=iteration + 1, loss=loss_value, loss_calls=copies,
                          mean_abs_delta=float(delta.detach().abs().mean()), max_abs_delta=float(delta.detach().abs().max())))
    protected = (clean + delta.detach().clamp(-EPSILON, EPSILON)).clamp(0, 255)
    return quantize_bounded(protected, clean, EPSILON), trace


def gleam_protect(clean, objective, seed, transforms, blackbox=False):
    """GLEAM-v: per step 20 GDE copies then 20 LFE copies of (clean + delta), the gradient of the mean loss over the 40
    copies, TI 5x5 on it, momentum mu 1, sign step of 3/255.

    White-box (objective = SelfObjective or FaceSelf around it): the copies are encoded in batches of 5 against the
    clean features; with a face term, w * face(clean + delta) is back-propagated once per step into the same gradient.
    Black-box (objective = EnsembleSelfObjective or FaceSelf around it): one ensemble call per copy, each divided by 40;
    with a face term every copy adds w * face of the untransformed image (same gradient)."""
    spec = RULES['gleam']
    num = T.GLEAM['num_warping']
    copies = 2 * num
    kernel = transforms.kernel(clean.device, clean.dtype)
    seed_everything(seed)
    objective.begin(clean)
    face_self = hasattr(objective, 'face')
    base = objective.base if face_self else objective
    delta = torch.zeros_like(clean, requires_grad=True)
    momentum = torch.zeros_like(clean)
    trace = []
    for iteration in range(spec['iterations']):
        objective.step(iteration)
        if delta.grad is not None:
            delta.grad.zero_()
        branch_loss = {}
        for branch, view in (('gde', transforms.gde), ('lfe', transforms.lfe)):
            total = 0.
            if blackbox:
                for _ in range(num):
                    image = (clean + delta.clamp(-EPSILON, EPSILON)).clamp(0, 255)
                    _note(objective, image)
                    loss = objective(view(image)) / copies
                    loss.backward()
                    total += float(loss.detach()) * copies / num
            else:
                for _ in range(num // GLEAM_CHUNK):
                    image = (clean + delta.clamp(-EPSILON, EPSILON)).clamp(0, 255)
                    views = torch.cat([view(image) for _ in range(GLEAM_CHUNK)])
                    loss = base.batched(views) * (GLEAM_CHUNK / copies)
                    loss.backward()
                    total += float(loss.detach()) * copies / num
            branch_loss[branch] = total
        face_raw = 0.0
        if face_self and not blackbox and objective.face.active:
            image = (clean + delta.clamp(-EPSILON, EPSILON)).clamp(0, 255)
            face = objective.face.loss(image)
            (objective.weight * face).backward()
            face_raw = float(face.detach())
        if not torch.isfinite(delta.grad).all():
            raise FloatingPointError('Non-finite attack gradient.')
        with torch.no_grad():
            gradient = F.conv2d(delta.grad, kernel, stride=(1, 1), groups=3, padding=(T.GLEAM['ti_kernel'] - 1) // 2)
            momentum = DECAY * momentum + gradient / (gradient.abs().mean() + 1e-12)
            direction = momentum.sign()
            delta.copy_((delta - spec['step'] * direction).clamp(-EPSILON, EPSILON))
        trace.append(dict(step=iteration + 1, loss=0.5 * (branch_loss['gde'] + branch_loss['lfe']), loss_face=face_raw,
                          loss_calls=copies, mean_abs_delta=float(delta.detach().abs().mean()),
                          max_abs_delta=float(delta.detach().abs().max())))
    protected = (clean + delta.detach().clamp(-EPSILON, EPSILON)).clamp(0, 255)
    return quantize_bounded(protected, clean, EPSILON), trace
