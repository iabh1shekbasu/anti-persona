"""Convert insightface buffalo_l w600k_r50.onnx (the recogniser of MyVLM's face gate) into the torch node list +
weights that ``anti_persona.face.BuffaloL`` replays op by op, and verify the replay against onnxruntime.

    python scripts/convert_buffalo_l.py --onnx weights/buffalo_l/w600k_r50.onnx --out weights/buffalo_l_w600k_r50_torch.pt

The graph is a BN-folded iResNet-50 with six op types (Conv, BatchNormalization, PRelu, Add, Flatten, Gemm); the
conversion refuses any other op. Verification: random 112 x 112 inputs in [-1, 1] through onnxruntime and through
the torch replay must agree (cosine >= 0.9999 between the embeddings). For the paper the replay was additionally
checked against insightface's own pipeline on 15 real aligned faces (min cosine 0.99999994).
Needs: onnx, onnxruntime (pip install -r requirements.txt).
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anti_persona.face import ONNX_OPS, BuffaloL  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert(onnx_path):
    import onnx
    from onnx import numpy_helper
    graph = onnx.load(str(onnx_path)).graph
    tensors = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in graph.initializer}
    nodes = []
    for n in graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        attrs = {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in attrs.items()}
        nodes.append(dict(op=n.op_type, inputs=list(n.input), outputs=list(n.output), attrs=attrs))
    ops = {n["op"] for n in nodes}
    assert ops <= ONNX_OPS, f"unexpected ops {ops - ONNX_OPS}"
    for n in nodes:                                    # the replay assumes these attribute values
        a = n["attrs"]
        if n["op"] == "Conv":
            assert a["pads"][0] == a["pads"][2] and a["pads"][1] == a["pads"][3]
        if n["op"] == "Flatten":
            assert a["axis"] == 1
        if n["op"] == "Gemm":
            assert a.get("transB") == 1 and a.get("alpha", 1.0) == 1.0 and a.get("beta", 1.0) == 1.0 and not a.get("transA")
    return dict(nodes=nodes, tensors=tensors, input=graph.input[0].name, output=graph.output[0].name,
                onnx_sha256=sha256(onnx_path))


def verify(onnx_path, saved, samples=4):
    import numpy as np
    import onnxruntime
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = BuffaloL(saved).eval()
    g = torch.Generator().manual_seed(0)
    worst = 1.0
    for _ in range(samples):
        x = torch.rand(1, 3, 112, 112, generator=g) * 2 - 1
        ref = session.run(None, {session.get_inputs()[0].name: x.numpy().astype(np.float32)})[0]
        with torch.no_grad():
            got = model(x).numpy()
        cos = float((ref * got).sum() / (np.linalg.norm(ref) * np.linalg.norm(got)))
        worst = min(worst, cos)
    return worst


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        sys.exit(f"{args.out} exists; remove it to convert again")
    saved = convert(args.onnx)
    worst = verify(args.onnx, saved)
    print(f"torch replay vs onnxruntime: min cosine {worst:.7f}")
    if worst < 0.9999:
        sys.exit("verification FAILED; nothing written")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, args.out)
    print(f"wrote {args.out} (sha256 {sha256(args.out)})")


if __name__ == "__main__":
    main()
