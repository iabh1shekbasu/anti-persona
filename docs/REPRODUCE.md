# Reproducing the paper

Step by step: **weights → data → banks → protect → victims → evaluate**. Expected numbers are at the end and in
[RESULTS.md](RESULTS.md). The paper's runs used AMD MI210 GPUs (ROCm 6.2, torch 2.1.2, transformers 4.37.2); any
CUDA GPU with ≥ 8 GB works for the protection step.

> **Bit-exactness.** 501 signed-gradient steps amplify tiny differences in GPU arithmetic, so two runs of the same
> photo and seed on different hardware do not give the same bytes. Compare runs by their scores (the protection
> rates below), not by file identity. On the CPU the package reproduces the reference implementation bit for bit
> (`python -m unittest tests.test_equivalence`).

## 0. Environment

```bash
conda create -n anti-persona python=3.10 -y && conda activate anti-persona
# install torch / torchvision for your CUDA or ROCm (https://pytorch.org), then
pip install -r requirements.txt
python -m unittest tests.test_equivalence -v        # CPU sanity check, ~20 s
```

## 1. Weights and calibration photos

```bash
python scripts/download_weights.py      # -> weights/
python scripts/download_coco100.py      # -> data/coco100/  (or --from-local /path/to/coco/val2017)
```

| File | Source | Used for |
|---|---|---|
| `weights/clip-vit-large-patch14-336/` | Hugging Face `openai/clip-vit-large-patch14-336`, revision `ce19dc91…` | the encoder (25 hidden states) |
| `weights/deeplabv3_resnet101_coco-586e9e4e.pth` | torchvision `COCO_WITH_VOC_LABELS_V1` | person patches → CES anchors |
| `weights/buffalo_l/det_10g.onnx` | insightface buffalo_l (v0.7) | face detection + 5-point alignment |
| `weights/buffalo_l/w600k_r50.onnx` → `weights/buffalo_l_w600k_r50_torch.pt` | insightface buffalo_l, converted by `scripts/convert_buffalo_l.py` | the face term (MyVLM's gate recogniser) |
| `data/coco100/*.jpg` | COCO val2017, ids in `configs/coco100.json` | CES threshold calibration |

All files are checked against `scripts/checksums/`. The torch conversion verifies itself against onnxruntime
(cosine ≥ 0.9999); it reproduced the tensors used for the paper exactly (the `.pt` bytes depend on the torch
version, the tensors do not).

## 2. Data

The benchmark (Paper-10 + MMPB-15, 25 identities) is coming soon.

## 3. Banks and protection

The album CLI builds everything per person (token prototype sums, DeepLabV3 person patches, the CES evidence bank
calibrated on COCO-100, face alignments) and protects every photo leave-one-out:

```bash
python -m anti_persona.protect --album <folder of one person> --out <dir> --seed 0 [--order order.txt]
```

To reproduce the paper, protect each album in the paper's album order (`--order`, released with the benchmark; it
fixes each photo's leave-one-out position and its image seed = run seed + position), for run seeds 0 and 202.

- Reactive: the test photos are protected (references = the person's other test photos).
- Proactive: the training photos are protected (references = the person's other training photos).
- Budget: 165 + 170 photos × 2 seeds × ≈ 53 s on one MI210 ≈ 10 GPU-hours.
- The evidence bank of each album is cached in `<out>/_cache/bank.pt` with the album order in `bank.json`.

Details that matter for exactness, all handled by the CLI: photos are squashed (not cropped) to 336 × 336 with
bicubic interpolation; anchors come from the photo at full resolution through DeepLabV3 (dilated 2%, patch ≥ 25%
person); the face detector runs on the rounded 336 × 336 clean image and the largest face is used; outputs are
rounded to whole numbers within 4 of both the clean and the rounded clean image. For Paper-10 the paper used face
alignments recorded earlier with the same detector settings (`--face-cache` accepts such a file); the CLI recomputes
them.

## 4. Victims

| Victim | Code | Checkpoints |
|---|---|---|
| Yo'LLaVA (LLaVA-1.5-13B + learned concept tokens), reactive | `victims/yollava/` | Paper-10: the authors' released concept checkpoints (yuheng: our seed-42 victim); MMPB-15: trained by `victims/yollava/personalize.py`, victim seed 42, epoch 14 |
| Yo'LLaVA, proactive | `victims/yollava/` | all 25 identities: our concepts, one per (training album, victim model seed 42–46), trained by `eval/proactive_yollava.py train` (same recipe), epoch 14; checkpoints coming soon |
| MyVLM (LLaVA + concept head + concept embedding) | `victims/myvlm/` | trained per person, victim seed 42, concept name `sks`, buffalo_l face gate with threshold 0.675 |

How to train and query them, and which external checkpoints are needed: [eval/README.md](../eval/README.md).

## 5. Evaluation

The protection rate is the share of **all** test photos on which the personalized model does not recognise the
person; per seed the photos of all identities are pooled, and tables report mean ± |s0 − s202| / 2 over the two
image seeds. Yo'LLaVA answers are read with the semantic parser (a leading yes/no, negation phrases). Scripts and
exact prompts: [eval/README.md](../eval/README.md).

### Proactive Yo'LLaVA (victim model seeds 42–46)

Each person's training album is protected (§3, image seeds 0 and 202), a Yo'LLaVA concept is trained on every album at
each victim model seed, and the concepts are asked about the person's clean test photos. The Clean row trains on the
unprotected album, squashed to 336 × 336. Per person and model seed:

```bash
P=thao; S=42                                   # every identity, S = 42 43 44 45 46
python -m eval.proactive_yollava clean-album --train-dir <clean training photos of $P> --out albums/clean/$P  # once
for M in clean ours_s0 ours_s202; do           # + each baseline: <method>_s0, <method>_s202 (deterministic: one album)
  python -m eval.proactive_yollava train --identity $P --model-path <llava-v1.5-13b> --json-path <training JSONs> \
      --protected-album albums/$M/$P --method $M --model-seed $S --work runs/proactive
done
python -m eval.proactive_yollava score --identity $P --model-path <llava-v1.5-13b> --model-seed $S \
    --concept clean=runs/proactive/checkpoints/$P/clean-c$S \
    --concept ours_s0=runs/proactive/checkpoints/$P/ours_s0-c$S \
    --concept ours_s202=runs/proactive/checkpoints/$P/ours_s202-c$S \
    --test-dir <clean test photos of $P> --negative-dir <other people's photos> --out scores/paper10/$P-c$S
```

Then `python -m eval.aggregate --input paper10=scores/paper10 --input mmpb15=scores/mmpb15 --per-model-seed`: per model
seed the usual mean over image seeds 0 and 202, and **mean ± std (population) over the five model seeds**. Training is
`victims/yollava/personalize.py` with the recipe in [eval/README.md](../eval/README.md#proactive-yollava) (15 epochs,
epoch 14 scored); only the album and `--seed` differ between concepts. Our trained concepts (25 identities × 5 model
seeds × every method): checkpoints coming soon; they load with `score --concept NAME=<folder>`.

## 6. Expected numbers

| Setting | Ours | Where |
|---|---:|---|
| Reactive Yo'LLaVA, both sets | 92.1 ± 1.8 | RESULTS §1 |
| Reactive MyVLM, both sets | 94.8 ± 0.9 | RESULTS §1 |
| Proactive MyVLM | 84.5 ± 0.3 | RESULTS §2 |
| Proactive Yo'LLaVA, mean ± std over model seeds 42–46 (Clean 29.1 ± 7.8) | 91.2 ± 4.4 | RESULTS §2 |
| Black-box (6 held-out surrogates) | 78.4 ± 0.1 | RESULTS §3 |
| Reactive Yo'LLaVA after JPEG Q85 | 78.5 ± 2.1 | RESULTS §4 |
| PSNR / SSIM / LPIPS-Alex (330 images) | 38.81 dB / 0.9453 / 0.0555 | RESULTS §6 |

Re-runs on other GPUs should land within the seed-to-seed spread of each cell; a single re-scoring of the same
images moved MyVLM by up to ±0.7 points in our runs (sampling inside the victim).

## Baselines

[baselines/README.md](../baselines/README.md) lists the nine baselines, their settings and the instance face term
(weight 1/28) they all carry in the paper's tables.
