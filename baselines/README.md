# Baselines

Nine baselines, all optimising **our loss** with **their own update rule / procedure**, run once at their own
settings (no tuning). In the final tables of the paper every baseline also carries the **instance face term** (below).

## Objective

Every baseline optimises the instance part of our loss, the CLIP self term

    L_self(x) = mean over the 25 hidden states of the mean token cosine between CLIP ViT-L/14-336 features of x and of
                the clean photo (summary token + 576 patches)

(minimised), on the encoder our method attacks (white-box) or on a surrogate ensemble (black-box). The prototype,
CES, JPEG-in-the-loop and gradient smoothing are our contribution and are never used by a baseline. Final tables:

    L = L_self + w * cos( buffalo_l(aligned_crop(x, M)), buffalo_l(aligned_crop(x_clean, M)) ),   w = 1/28

the photo's OWN clean face embedding as target (no album, no prototype), M = SCRFD det_10g alignment of the clean
photo (largest face), buffalo_l w600k_r50 fed B,G,R. w = 0.125 / 3.5 = our face weight over the sum of our CLIP-term
weights (1 + 0.5 + 2). No detected face: no face term. For DI-FGSM, SIA and GLEAM-v the face term sees the
untransformed image of the step (the alignment is only valid in the photo's own geometry). DiffAttack and WDP multiply
our loss by their own adversarial weight a (10 and 0.003); the face coefficient is scaled with it (a * w), so the face
term has the same share for every baseline.

## Settings (eps = 4/255 for every bounded rule; pixel units)

| Baseline | Paper | Iterations x step | Start | Update / transformation |
|---|---|---|---|---|
| R+FGSM | Tramer et al. 2018 | 1 x 2/255 | 2/255 · sign(N(0, I)) | one signed step |
| I-FGSM | Kurakin et al. 2017 | 100 x 0.1/255 | zero | sign |
| DI-FGSM | Xie et al. 2019 | 100 x 0.1/255 | zero | sign; input diversity p = 0.5: bicubic resize to [336, 386], random zero pad to 386, bicubic back to 336 (adversarial image only) |
| TI-MI-FGSM | Dong et al. 2019 | 100 x 0.1/255 | zero | TIM 15 x 15 Gaussian (nsig 3, TransferAttack) on the gradient, momentum μ = 1 on g / mean\|g\|, sign |
| PGD | Madry et al. 2018 | 100 x 0.1/255 | U(−4, 4)/255 | sign |
| SIA | Wang et al. 2023 | 10 x 0.4/255 (T = 10, α = ε/T) | zero | 20 block-transformed copies per step (3 x 3 blocks, the authors' 7 ops), mean loss, momentum μ = 1, sign |
| GLEAM-v | Liu et al. 2025 | 10 x 3/255 (T = 10, α = 0.75 ε as eval_gleam.py) | zero | 20 GDE (scale U(1.1, 1.8) + pad) + 20 LFE (30 x 30 NURBS warp, ±10 px) copies, TI 5 x 5 (σ 3), momentum μ = 1, sign; vision-only (CMFA needs attacker text) |
| DiffAttack | Chen et al. 2024 | authors' defaults: DDIM 20 steps, attack from step 15, 30 latent iterations (AdamW 1e-2), guidance 2.5 | — | SD-2-base latent attack; the classifier loss replaced by ours (x10, the authors' attack_loss_weight); in-loop L2 projection to **PSNR ≥ 30 dB** |
| WDP | Salar et al. 2025 | authors' defaults, 35 protection steps (paper) | — | weakened-diffusion-purification latent optimisation of the aligned face; the FR loss replaced by ours on the pasted-back photo (x0.003, the authors' weight); in-loop projection to **PSNR ≥ 30 dB** |

Common loop (identical to our method's): the state δ is clamped to ±4 after every step, the encoder sees
clip(x + δ, 0, 255), descent on the loss, and the saved PNG has integer L-inf ≤ 4 from the rounded clean photo.
Image seed = run seed (0 or 202) + the photo's index in its album. In the white-box setting I-FGSM and TI-MI-FGSM are
deterministic (one run, counted for both seeds); in the black-box setting every rule runs both seeds (dynamic scaling is
random). DiffAttack / WDP: the authors' two hard-coded seeds (process seed, VAE posterior sample) are set to the image
seed.

## Black-box setting

Six leave-one-out ensembles of five surrogates (`blackbox/surrogates.py`): A OpenAI CLIP ViT-B/16, B OpenAI CLIP
ViT-L/14-224, D CLIPA ViT-L/14-336, S SigLIP so400m-384, X ConvNeXt-V2-L, W supervised ViT-L/16 (`F6_noA` … `F6_noW`).
Dynamic scaling: one scale U[0.5, 1) of 336 per step, the clean reference re-encoded at the same scale; self term =
equal-weight mean over the five surrogates (`blackbox/ensemble.py`). The victims' CLIP ViT-L/14-336 is never loaded.

## Run

```bash
bash baselines/third_party/fetch.sh     # TransferAttack (SIA, TIM) and GLEAM sources; DiffAttack / WDP repos
export TRANSFERATTACK_DIR=$PWD/baselines/third_party/TransferAttack GLEAM_DIR=$PWD/baselines/third_party/GLEAM

# white-box, PGD + face, image seed 0 + album index 3
python -m baselines.generate --baseline pgd --photo <photo> --album-index 3 --seed 0 --out <new folder> \
    --face-weights <buffalo_l_w600k_r50_torch.pt> --detector <buffalo_l/det_10g.onnx>
# black-box (ensemble without CLIP ViT-B/16)
python -m baselines.generate --baseline difgsm ... --blackbox F6_noA
```

`--face-weights` is the torch conversion of insightface's `w600k_r50.onnx` made by the weight scripts in `scripts/`.

| File | Contents |
|---|---|
| `rules.py` | the seven bounded update rules, the shared loop (`protect`) and GLEAM-v's loop (`gleam_protect`) |
| `objectives.py` | loss backends: white-box self term, black-box ensemble self term, the instance face term, their sum |
| `transfer_ops.py` | the authors' SIA / TIM / GLEAM functions compiled from their repositories (AST, never imported) |
| `blackbox/` | the six surrogates and the leave-one-out ensemble with dynamic scaling |
| `generate.py` | command line: one photo, one baseline, one run seed |
| `reference/` | DiffAttack and WDP generators (and the face-term module) exactly as they produced the paper's images, paths parameterised |
| `third_party/` | where the authors' repositories go (`fetch.sh`, pinned commits; not vendored) |

## DiffAttack and WDP

`reference/diffattack/generate.py` and `reference/wdp/generate35.py` are the generators that produced the paper's
DiffAttack + face and WDP + face images, with only absolute paths replaced by environment variables. They run the
authors' code unedited from its clone (`DIFFATTACK_DIR`, `WDP_DIR`) through our stock wrappers
(`reference/*/gen_*.py`: input staging, labels, alignment / paste-back, output checks) and swap one loss statement in
memory (AST), so the published procedure is otherwise untouched. They still expect the research harness they were
written for: a queue folder with a `PLAN.json` (photo list with source sha256 and album index, the encoder record, the
run seed, the face weight and the recorded SCRFD matrices) and the legacy baseline module (`LEGACY_BASELINES_DIR`,
`LEGACY_FACE_BL_DIR`, `LEGACY_CHANNEL_ORDER_DIR`). **They are provided for their exact procedure and settings, not as
turnkey scripts**; porting them onto `baselines/objectives.py` (the same self + face loss) is straightforward but was not
done for this release, and they were not re-run here.

## Tests

- `tests/test_rules_cpu.py` (CPU, tiny stand-in encoder, no downloads): every rule finishes, the saved image stays
  within 4 of the rounded clean image, its self term drops below the clean image's, and a fixed seed reproduces it.
  SIA / TI-MI-FGSM / GLEAM-v are skipped unless the third-party clones are present.
- Equivalence with the code that produced the paper's images (checked once on the CPU before release, same stand-in
  encoder on both sides, one real photo, same seed): `rules.protect` / `rules.gleam_protect` with
  `objectives.SelfObjective` give bit-identical PNGs and per-step losses for all seven bounded rules; with the instance
  face term (`objectives.FaceSelf` + `InstanceFace`, real buffalo_l and SCRFD) PGD, DI-FGSM and SIA are bit-identical
  as well. GLEAM-v with the face term was not compared (the original side exceeded the 4 GB CPU limit).
- Not run here: the black-box ensemble (a copy of the pinned module with only the cache path parameterised), and
  DiffAttack / WDP (reference copies).
