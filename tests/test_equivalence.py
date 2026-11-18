"""CPU equivalence test: the package `anti_persona` vs the single-file reference implementation.

`tests/reference/final_method_reference.py` is the one-file restatement of the final method that was tested equal
(largest difference 0) to the code that produced the paper's images (sha256 9ac4fb4a...cbd9). This test checks that
the package computes exactly the same numbers, on a tiny synthetic input, with no downloads and no GPU:

  * encoder   the stand-in encoder of `tests/standin_encoder.py` (same interface as CLIP ViT-L/14-336: 25 entries of
              577 x 1024; fixed pseudo-random weights). The real encoder class is the same code on both sides.
  * face      a small synthetic network in the buffalo_l node-list format (Conv / BatchNormalization / PRelu / Add /
              Flatten / Gemm), so the op-by-op replay, the aligned crop and the leave-one-out face prototype are
              exercised without the insightface weights. One album photo has "no face" to test that path.
  * album     3 synthetic photos of different sizes; 4 synthetic "public" calibration photos; random person patches.

Checked with tolerance 0 (bit-for-bit): read_image, the leave-one-out prototype, the evidence bank (anchors,
thresholds, reference index), every loss term, the JPEG / smoothing / rounding helpers, every logged value of every
optimisation step, and the final PNG written by the album command-line pipeline (`protect_album`).

    python -m unittest tests.test_equivalence -v
"""
import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from standin_encoder import StandIn  # noqa: E402

from anti_persona import bank as B  # noqa: E402
from anti_persona import encoder as E  # noqa: E402
from anti_persona import face as FA  # noqa: E402
from anti_persona import losses as L  # noqa: E402
from anti_persona import optim as O  # noqa: E402
from anti_persona.protect import protect_album  # noqa: E402

STEPS = 7          # covers JPEG draws, no-JPEG steps and two smoothed steps (3 and 6)
SEED = 202


def load_reference():
    spec = importlib.util.spec_from_file_location("final_method_reference", HERE / "reference" / "final_method_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_photo(rng, height, width):
    """Smooth colour gradients plus texture, so that JPEG and the encoder see structure."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    base = np.stack([128 + 90 * np.sin(xx / (17 + 5 * c) + c) * np.cos(yy / (23 + 3 * c)) for c in range(3)], -1)
    return np.clip(base + rng.normal(0, 18, (height, width, 3)), 0, 255).astype(np.uint8)


def synthetic_face_net(seed=7):
    """A tiny network in the node-list format that scripts/convert_buffalo_l.py writes for buffalo_l."""
    g = torch.Generator().manual_seed(seed)

    def r(*shape, scale=0.1):
        return torch.randn(*shape, generator=g) * scale
    tensors = {"w0": r(8, 3, 3, 3), "b0": r(8), "bn_s": 1 + r(8), "bn_b": r(8), "bn_m": r(8), "bn_v": 1 + r(8).abs(),
               "slope": r(8).abs(), "w1": r(8, 8, 3, 3), "b1": r(8), "w2": r(4, 8, 4, 4), "w3": r(16, 784, scale=0.02),
               "b3": r(16)}
    conv = dict(strides=[2, 2], pads=[1, 1, 1, 1], dilations=[1, 1], group=1)
    nodes = [
        dict(op="Conv", inputs=["data", "w0", "b0"], outputs=["c1"], attrs=conv),
        dict(op="BatchNormalization", inputs=["c1", "bn_s", "bn_b", "bn_m", "bn_v"], outputs=["n1"], attrs=dict(epsilon=1e-5)),
        dict(op="PRelu", inputs=["n1", "slope"], outputs=["p1"], attrs={}),
        dict(op="Conv", inputs=["p1", "w1", "b1"], outputs=["c2"], attrs=dict(conv, strides=[1, 1])),
        dict(op="Add", inputs=["p1", "c2"], outputs=["a1"], attrs={}),
        dict(op="Conv", inputs=["a1", "w2"], outputs=["c3"], attrs=dict(strides=[4, 4], pads=[0, 0, 0, 0], dilations=[1, 1], group=1)),
        dict(op="Flatten", inputs=["c3"], outputs=["f"], attrs=dict(axis=1)),
        dict(op="Gemm", inputs=["f", "w3", "b3"], outputs=["embedding"], attrs=dict(transB=1)),
    ]
    return dict(nodes=nodes, tensors=tensors, input="data", output="embedding")


def assert_same(test, a, b, what):
    if torch.is_tensor(a):
        test.assertEqual(a.shape, b.shape, what)
        test.assertEqual(a.dtype, b.dtype, what)
        test.assertTrue(torch.equal(a, b), f"{what}: largest difference {float((a.double() - b.double()).abs().max())}")
    else:
        test.assertEqual(a, b, what)


class Equivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.ref = load_reference()
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        rng = np.random.RandomState(20261117)
        (root / "album").mkdir()
        (root / "public").mkdir()
        cls.paths = []
        for i, (h, w) in enumerate([(300, 240), (336, 336), (260, 410)]):
            p = root / "album" / f"photo_{i}.png"
            Image.fromarray(synthetic_photo(rng, h, w)).save(p)
            cls.paths.append(p)
        cls.public_paths = []
        for k in range(4):
            p = root / "public" / f"public_{k}.jpg"
            Image.fromarray(synthetic_photo(rng, 280 + 10 * k, 320)).save(p, quality=92)
            cls.public_paths.append(p)
        g = torch.Generator().manual_seed(11)
        cls.tokens = [torch.rand(576, generator=g) < q for q in (0.45, 0.04, 0.3)]   # photo 1: fewer than 32 patches
        torch.save(synthetic_face_net(), root / "face_net.pt")
        cls.face_weights = root / "face_net.pt"
        cls.encoder = StandIn()
        cls.album = [E.read_image(p, "cpu") for p in cls.paths]
        matrix = [[0.5, 0.0, -20.0], [0.0, 0.5, -30.0]]
        cls.rows = [(matrix, a.round().clamp(0, 255)) for a in cls.album]
        cls.rows[2] = (None, cls.rows[2][1])                                         # photo 2: no face detected
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_read_image(self):
        for p in self.paths:
            assert_same(self, E.read_image(p, "cpu"), self.ref.read_image(p, "cpu"), f"read_image {p.name}")

    def test_prototype_and_bank(self):
        ref = self.ref
        for index in range(3):
            proto_p, own_p = B.leave_one_out_prototype(self.encoder, self.album, index)
            proto_r, own_r = ref.leave_one_out_prototype(self.encoder, self.album, index)
            for l in range(25):
                assert_same(self, proto_p[l], proto_r[l], f"prototype photo {index} entry {l}")
                assert_same(self, own_p[l], own_r[l], f"clean features photo {index} entry {l}")
        public = [E.read_image(p, "cpu") for p in self.public_paths]
        bank_p = B.build_ces_bank(self.encoder, self.album, self.tokens, self.public_paths)
        bank_r = ref.build_ces_bank(self.encoder, self.album, self.tokens, public)
        for l in range(25):
            for key in ("anchors", "thresholds", "reference_index"):
                assert_same(self, bank_p[l][key], bank_r[l][key], f"bank entry {l} {key}")
        path = self.root / "bank.pt"
        B.save_ces_bank(bank_p, path)
        for index in range(3):
            loaded_p, loaded_r = B.load_ces_bank(path, index, "cpu"), ref.load_ces_bank(path, index, "cpu")
            memory_p = B.bank_without(bank_p, index)
            for l in range(25):
                for k in range(2):
                    assert_same(self, loaded_p[l][k], loaded_r[l][k], f"loaded bank photo {index} entry {l}")
                    assert_same(self, memory_p[l][k], loaded_r[l][k], f"in-memory bank photo {index} entry {l}")
        self.assertEqual(int((bank_p[0]["reference_index"] == 1).sum()), int(self.tokens[1].sum()))   # < 32 anchors kept

    def test_helpers(self):
        ref = self.ref
        g = torch.Generator().manual_seed(3)
        image = torch.rand(1, 3, 336, 336, generator=g) * 255
        for q in (60, 77, 95):
            assert_same(self, O.jpeg_view(image, q), ref.jpeg_view(image, q), f"jpeg_view q{q}")
        grad = torch.randn(1, 3, 336, 336, generator=g)
        assert_same(self, O.smooth_gradient(grad), ref.smooth_gradient(grad), "smooth_gradient")
        noisy = image + torch.randn(1, 3, 336, 336, generator=g) * 6
        assert_same(self, O.quantize_bounded(noisy, image), ref.quantize_bounded(noisy, image), "quantize_bounded")
        anchors, patches = torch.randn(40, 1024, generator=g), torch.randn(576, 1024, generator=g)
        assert_same(self, L.patch_response(anchors, patches), ref.patch_response(anchors, patches), "patch_response")

    def test_loss_terms(self):
        ref = self.ref
        index = 0
        proto, clean_features = ref.leave_one_out_prototype(self.encoder, self.album, index)
        public = [E.read_image(p, "cpu") for p in self.public_paths]
        bank = B.bank_without(ref.build_ces_bank(self.encoder, self.album, self.tokens, public), index)
        face_p = FA.FaceTerm(self.face_weights, self.rows, index, "cpu")
        face_r = ref.FaceTerm(self.face_weights, self.rows, index, "cpu")
        self.assertTrue(face_p.active and face_r.active)
        assert_same(self, face_p.prototype, face_r.prototype, "face prototype")
        view = O.jpeg_view(self.album[index] + 3.0, 81)
        loss_p, terms_p = L.loss_terms(view, self.encoder, clean_features, proto, bank, face_p)
        loss_r, terms_r = ref.loss_terms(view, self.encoder, clean_features, proto, bank, face_r)
        assert_same(self, loss_p, loss_r, "total loss")
        for key in terms_r:
            assert_same(self, terms_p[key], terms_r[key], key)
        self.assertFalse(FA.FaceTerm(self.face_weights, self.rows, 2, "cpu").active)            # photo without a face

    def test_album_pipeline_equals_reference_loop(self):
        """protect_album (the CLI pipeline) vs the reference loop, photo by photo, bit for bit."""
        ref = self.ref
        out = self.root / "out"
        face_model = FA.load_buffalo_l(self.face_weights, "cpu")
        records = protect_album(self.paths, out, SEED, self.encoder, self.tokens, self.public_paths, "cpu",
                                face_model=face_model, face_rows=self.rows, steps=STEPS, log=lambda *_: None)
        self.assertEqual([r["face_active"] for r in records], [True, True, False])
        public = [E.read_image(p, "cpu") for p in self.public_paths]
        bank_r = ref.build_ces_bank(self.encoder, self.album, self.tokens, public)
        torch.save(bank_r, self.root / "bank_ref.pt")
        for index, path in enumerate(self.paths):
            proto, _ = ref.leave_one_out_prototype(self.encoder, self.album, index)
            bank = ref.load_ces_bank(self.root / "bank_ref.pt", index, "cpu")
            face = ref.FaceTerm(self.face_weights, self.rows, index, "cpu")
            png_r, trace_r = ref.protect(self.album[index], self.encoder, proto, bank, face, SEED + index, STEPS)
            png_p = torch.from_numpy(np.array(Image.open(out / f"{path.stem}.png"))).permute(2, 0, 1)[None]
            assert_same(self, png_p, png_r, f"final PNG photo {index}")
            self.assertGreater(int((png_p.float() != self.album[index].round()).sum()), 0)
            self.assertLessEqual(float((png_p.float() - self.album[index].round()).abs().max()), 4.0)
            with (out / f"{path.stem}_loss.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), STEPS)
            jpeg_steps = 0
            for row, expected in zip(rows, trace_r):
                for key, value in expected.items():
                    got = row[key]
                    if value is None:
                        self.assertEqual(got, "", f"photo {index} step {expected['step']} {key}")
                    else:
                        self.assertEqual(float(got), float(value), f"photo {index} step {expected['step']} {key}")
                jpeg_steps += expected["jpeg_quality"] is not None
            self.assertTrue(0 < jpeg_steps < STEPS, "the test should see both JPEG and plain steps")


class CommandLine(unittest.TestCase):
    def test_help(self):
        result = subprocess.run([sys.executable, "-m", "anti_persona.protect", "--help"], cwd=REPO,
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--album", result.stdout)
        self.assertIn("--seed", result.stdout)


if __name__ == "__main__":
    unittest.main()
