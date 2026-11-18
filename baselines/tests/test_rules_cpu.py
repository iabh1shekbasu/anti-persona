"""CPU smoke test of baselines/rules.py with a tiny stand-in encoder (no model download, no GPU).

Checks for every rule whose sources are available: the run finishes, the saved image is whole numbers within 4 of the
rounded clean image, its self term is below the clean image's (1), and a fixed seed reproduces the image exactly. SIA / TI-MI-FGSM /
GLEAM-v need the TransferAttack / GLEAM clones (baselines/third_party) and are skipped without them.

    python baselines/tests/test_rules_cpu.py
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from baselines import objectives as O  # noqa: E402
from baselines import rules as R  # noqa: E402
from baselines import transfer_ops as T  # noqa: E402


class TinyEncoder(torch.nn.Module):
    """25 entries of [577, B, 64] from a fixed random patch embedding (stand-in for CLIP ViT-L/14-336)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.register_buffer('patch', torch.randn(64, 3, 14, 14, generator=g) * 0.05)
        self.register_buffer('mix', torch.randn(24, 64, 64, generator=g) / 8)

    def forward(self, image):
        x = torch.nn.functional.conv2d((image - 127.5) / 127.5, self.patch, stride=14).flatten(2).permute(2, 0, 1)
        x = torch.cat([x.mean(0, keepdim=True), x])                     # [577, B, 64], token 0 = summary
        out = [x]
        for layer in range(24):
            x = x + torch.tanh(x @ self.mix[layer])
            out.append(x)
        return out


def run(rule, seed=3):
    torch.manual_seed(0)
    clean = (torch.rand(1, 3, 336, 336) * 255)
    objective = O.SelfObjective(TinyEncoder())
    if rule == 'gleam':
        return clean, R.gleam_protect(clean, objective, seed, T.GleamTransforms())
    sia = T.SIATransform() if rule == 'sia' else None
    return clean, R.protect(clean, objective, rule, seed, sia=sia)


def available(rule):
    need = {'sia': T.SIA_FILE, 'ti_mifgsm': T.TIM_FILE, 'gleam': T.NURBS_FILE}.get(rule)
    return need is None or need.is_file()


def test_rules():
    checked = []
    for rule in R.RULES:
        if not available(rule):
            print(f'skip {rule} (third-party source not found)')
            continue
        clean, (png, trace) = run(rule)
        assert png.dtype == torch.uint8 and png.shape == clean.shape
        assert int((png.float() - clean.round()).abs().max()) <= R.EPSILON, rule
        assert len(trace) == R.RULES[rule]['iterations'], rule
        objective = O.SelfObjective(TinyEncoder())
        objective.begin(clean)
        with torch.no_grad():
            protected = float(objective(png.float()))
        assert protected < 1.0 - 1e-4, (rule, protected)          # the self term of the clean image is 1
        _, (again, _) = run(rule)
        assert torch.equal(png, again), f'{rule}: same seed, different image'
        checked.append(rule)
        print(f'ok {rule}: self term of the saved image {protected:.4f} (clean 1.0), {len(trace)} steps')
    assert {'rfgsm', 'ifgsm', 'difgsm', 'pgd'} <= set(checked)


if __name__ == '__main__':
    test_rules()
    print('baselines CPU smoke test OK')
