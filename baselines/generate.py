"""Protect ONE photo with ONE epsilon-bounded baseline (white-box or black-box), as in the paper's final tables.

  python -m baselines.generate --baseline pgd --photo <photo> --album-index 3 --seed 0 --out <folder> \
      --face-weights <buffalo_l_w600k_r50_torch.pt> --detector <buffalo_l/det_10g.onnx> [--blackbox F6_noA] [--no-face]

  --baseline     rfgsm | ifgsm | difgsm | ti_mifgsm | pgd | sia | gleam   (settings: baselines/rules.py, README.md)
  --album-index  the photo's position in the person's album; image seed = --seed + album index (our method's rule)
  --seed         run seed, 0 or 202 (I-FGSM / TI-MI-FGSM are deterministic in the white-box setting: one run)
  --blackbox     attack a leave-one-out surrogate ensemble with dynamic scaling instead of the victims' encoder
  --no-face      leave out the instance face term (paper: every baseline row carries it, weight 1/28)

Writes <out>/<photo stem>.png (336 x 336, integer L-inf <= 4 from the rounded clean photo), <stem>_loss.csv and
manifest.json. DiffAttack and WDP have their own generators (baselines/diffattack, baselines/wdp).
"""
import argparse
import csv
import json
import math
from pathlib import Path

BASELINES = ('rfgsm', 'ifgsm', 'difgsm', 'ti_mifgsm', 'pgd', 'sia', 'gleam')


def psnr(a, b):
    mse = float(((a.double() - b.double()) ** 2).mean())
    return None if mse == 0 else 10 * math.log10(255 ** 2 / mse)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--baseline', required=True, choices=BASELINES)
    p.add_argument('--photo', type=Path, required=True)
    p.add_argument('--album-index', type=int, required=True)
    p.add_argument('--seed', type=int, required=True, choices=(0, 202))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--face-weights', type=Path, help='buffalo_l w600k_r50 torch weights (scripts/ converts them)')
    p.add_argument('--detector', type=Path, help='insightface buffalo_l det_10g.onnx')
    p.add_argument('--no-face', action='store_true')
    p.add_argument('--blackbox', choices=[f'F6_no{k}' for k in 'ABDSXW'])
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if not a.no_face and (a.face_weights is None or a.detector is None):
        p.error('the face term needs --face-weights and --detector (or pass --no-face)')

    import torch
    from PIL import Image
    from anti_persona.encoder import read_image
    from anti_persona.optim import seed_everything
    from baselines import objectives as O
    from baselines import rules as R
    from baselines import transfer_ops as T

    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    seed = R.image_seed(a.seed, a.album_index)
    seed_everything(seed)                                   # before any model is built (as the paper's generators)
    if a.blackbox:
        from baselines.blackbox.ensemble import EnsembleEncoder
        base = O.EnsembleSelfObjective(EnsembleEncoder.for_combo(a.blackbox, device=a.device))
        base.seed = seed
    else:
        base = O.SelfObjective.load(a.device)
    face = None
    if not a.no_face:
        face = O.InstanceFace(a.photo, a.face_weights, a.detector, a.device)
        face.self_check()
        objective = O.FaceSelf(base, face)
    else:
        objective = base
    clean = read_image(a.photo, a.device)
    if a.baseline == 'gleam':
        png, trace = R.gleam_protect(clean, objective, seed, T.GleamTransforms(), blackbox=bool(a.blackbox))
    else:
        sia = T.SIATransform() if a.baseline == 'sia' else None
        png, trace = R.protect(clean, objective, a.baseline, seed, sia=sia)
    if face is not None and objective.calls:          # white-box GLEAM-v adds its face term outside __call__
        O.check_calls(objective.calls, objective.weight, face.active)
    out = a.out / f'{a.photo.stem}.png'
    Image.fromarray(png[0].permute(1, 2, 0).cpu().numpy()).save(out)
    rounded = clean.round().clamp(0, 255)
    linf = int((png.float() - rounded).abs().max())
    assert linf <= R.EPSILON, linf
    with (a.out / f'{a.photo.stem}_loss.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    (a.out / 'manifest.json').write_text(json.dumps(dict(
        baseline=a.baseline, label=R.LABELS[a.baseline], rule=R.RULES[a.baseline], epsilon=R.EPSILON,
        photo=str(a.photo), album_index=a.album_index, run_seed=a.seed, image_seed=seed,
        setting=f'black-box {a.blackbox}' if a.blackbox else 'white-box (CLIP ViT-L/14-336)',
        loss='self' + ('' if face is None else ' + 1/28 * instance face'),
        face_active=None if face is None else bool(face.active), linf_int_vs_rounded_clean=linf,
        psnr_vs_rounded_clean=psnr(png.float(), rounded), steps=len(trace)), indent=1))
    print(json.dumps(dict(out=str(out), linf=linf, loss_first=trace[0]['loss'], loss_last=trace[-1]['loss'])))


if __name__ == '__main__':
    main()
