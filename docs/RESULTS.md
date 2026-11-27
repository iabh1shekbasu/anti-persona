# Results

All numbers are copied from the final result tables of the paper. They are **protection rates**: the percentage of
**all** test photos on which the personalized model does **not** recognise the person (higher is better). Unless
stated otherwise:

- **Benchmark:** the Yo'LLaVA dataset (10 identities, 90 test photos) + the MMPB dataset (15 curated identities,
  75 test photos); 25 identities, 165 test photos. The benchmark is coming soon.
- **Seeds:** perturbation (image) seeds 0 and 202; victim model seed 42. `mean ± std` is over the two image seeds,
  with std = |s0 − s202| / 2. I-FGSM and TI-MI-FGSM are deterministic; one image set is counted for both seeds.
  Exception: proactive Yo'LLaVA is run at victim model seeds 42–45 and reported as mean ± std (population) over the
  model seeds of the per-seed numbers.
- **Yo'LLaVA:** semantic answer parser (leading yes/no, negation phrases). The Yo'LLaVA dataset uses the authors' released
  concept checkpoints (our seed-42, epoch-14 victim for yuheng); the MMPB dataset uses our seed-42, epoch-14 victims.
- **MyVLM:** neutral concept name `sks`, face-gate threshold 0.675, victim seed 42.
- **Baselines** use our loss with their own update rule and their published settings (ε = 4/255; DiffAttack and WDP
  tuned to PSNR ≈ 30 dB), **plus an instance-level face term** (buffalo_l cosine to the photo's own clean face) with
  weight 1/28 (DiffAttack / WDP: a · 1/28 with a = 10 / 0.003).
- **Ours:** ε = 4/255, 501 steps; every number below comes from the same 330 protected images
  (165 photos × 2 seeds) unless the setting needs other images (proactive: the training albums are protected).

## Headline

| Setting | Ours | Strongest baseline (+ face term) |
|---|---:|---|
| Reactive, Yo'LLaVA | **92.1 ± 1.8** | I-FGSM 80.6 ± 0.0 |
| Reactive, MyVLM | **94.8 ± 0.9** | GLEAM-v 84.8 ± 1.2, PGD 84.2 ± 1.8 |
| Proactive, MyVLM | **84.5 ± 0.3** | GLEAM-v 71.2 ± 2.7 |
| Black-box transfer (Yo'LLaVA, 6 held-out surrogates) | **74.0 ± 0.4** | GLEAM-v 51.1 ± 0.2 |
| After JPEG Q85 (reactive Yo'LLaVA) | **78.5 ± 2.1** | GLEAM-v 49.1 ± 0.6 |
| Image quality (330 images) | PSNR 38.81 dB, SSIM 0.9453, LPIPS-Alex 0.0555 | — |

## 1. Reactive protection (the victim is personalized on clean photos; the test photo is protected)

### Yo'LLaVA

| Method | Yo'LLaVA dataset | MMPB dataset | Overall (165 photos) |
|---|---:|---:|---:|
| Clean | 0.0 ± 0.0 | 16.0 ± 0.0 | 7.3 ± 0.0 |
| R+FGSM + face | 0.0 ± 0.0 | 19.3 ± 2.0 | 8.8 ± 0.9 |
| I-FGSM + face | 74.4 ± 0.0 | 88.0 ± 0.0 | 80.6 ± 0.0 |
| DI-FGSM + face | 58.3 ± 2.8 | 82.0 ± 0.7 | 69.1 ± 1.2 |
| TI-MI-FGSM + face | 18.9 ± 0.0 | 58.7 ± 0.0 | 37.0 ± 0.0 |
| PGD + face | 69.4 ± 0.6 | 85.3 ± 1.3 | 76.7 ± 0.9 |
| SIA + face | 7.2 ± 1.7 | 52.0 ± 0.0 | 27.6 ± 0.9 |
| GLEAM-v + face | 26.7 ± 1.1 | 73.3 ± 1.3 | 47.9 ± 1.2 |
| DiffAttack + face | 13.3 ± 0.0 | 61.3 ± 0.0 | 35.2 ± 0.0 |
| WDP + face | 6.7 ± 0.0 | 28.7 ± 0.7 | 16.7 ± 0.3 |
| **Ours** | **88.9 ± 1.1** | **96.0 ± 2.7** | **92.1 ± 1.8** |

### MyVLM

| Method | Yo'LLaVA dataset | MMPB dataset | Overall (165 photos) |
|---|---:|---:|---:|
| Clean | 26.7 ± 0.0 | 37.3 ± 0.0 | 31.5 ± 0.0 |
| R+FGSM + face | 24.4 ± 0.0 | 36.7 ± 3.3 | 30.0 ± 1.5 |
| I-FGSM + face | 84.4 ± 0.0 | 77.3 ± 0.0 | 81.2 ± 0.0 |
| DI-FGSM + face | 80.6 ± 3.9 | 84.0 ± 1.3 | 82.1 ± 1.5 |
| TI-MI-FGSM + face | 43.3 ± 0.0 | 54.7 ± 0.0 | 48.5 ± 0.0 |
| PGD + face | 81.1 ± 2.2 | 88.0 ± 1.3 | 84.2 ± 1.8 |
| SIA + face | 44.4 ± 1.1 | 50.0 ± 0.7 | 47.0 ± 0.3 |
| GLEAM-v + face | 86.7 ± 3.3 | 82.7 ± 1.3 | 84.8 ± 1.2 |
| DiffAttack + face | 62.8 ± 0.6 | 85.3 ± 2.7 | 73.0 ± 0.9 |
| WDP + face | 47.8 ± 0.0 | 48.7 ± 0.7 | 48.2 ± 0.3 |
| **Ours** | **95.0 ± 0.6** | **94.7 ± 1.3** | **94.8 ± 0.9** |

The per-identity cells are in the paper's supplementary material.

## 2. Proactive protection (the victim is personalized on the protected album; the test photos are clean)

### MyVLM

| Method | Yo'LLaVA dataset | MMPB dataset | Overall |
|---|---:|---:|---:|
| Clean | 12.2 ± 0.0 | 22.7 ± 0.0 | 17.0 ± 0.0 |
| R+FGSM + face | 30.6 ± 5.0 | 23.3 ± 8.7 | 27.3 ± 1.2 |
| I-FGSM + face | 67.8 ± 0.0 | 56.0 ± 0.0 | 62.4 ± 0.0 |
| DI-FGSM + face | 60.0 ± 1.1 | 50.0 ± 0.7 | 55.5 ± 0.3 |
| TI-MI-FGSM + face | 24.4 ± 0.0 | 48.0 ± 0.0 | 35.2 ± 0.0 |
| PGD + face | 53.9 ± 1.7 | 58.7 ± 4.0 | 56.1 ± 0.9 |
| SIA + face | 21.7 ± 1.7 | 32.7 ± 6.0 | 26.7 ± 3.6 |
| GLEAM-v + face | 72.8 ± 3.9 | 69.3 ± 1.3 | 71.2 ± 2.7 |
| DiffAttack + face | 52.8 ± 2.8 | 78.0 ± 2.0 | 64.2 ± 2.4 |
| WDP + face | 22.2 ± 4.4 | 44.7 ± 3.3 | 32.4 ± 0.9 |
| **Ours** | **81.7 ± 0.6** | **88.0 ± 1.3** | **84.5 ± 0.3** |

### Yo'LLaVA

Mean ± std (population) over victim model seeds 42, 43, 44, 45; per model seed, the mean over image seeds 0 and
202. One concept per training album and model seed, all 25 identities with our concepts, epoch 14
([eval/README.md](../eval/README.md#proactive-yollava)).

| Method | Yo'LLaVA dataset | MMPB dataset | Overall (165 photos) |
|---|---:|---:|---:|
| Clean | 22.5 ± 5.9 | 32.0 ± 8.7 | 26.8 ± 7.2 |
| **Ours** | **92.4 ± 1.1** | **94.7 ± 2.0** | **93.4 ± 0.6** |

Baseline rows: coming soon.

## 3. Black-box transfer (Yo'LLaVA, overall 165 photos)

Each column holds one surrogate out: the perturbation is optimised on an ensemble of the other surrogates and
evaluated on Yo'LLaVA (whose CLIP-L/14-336 encoder is never used). Dynamic loss scaling across the ensemble.

| Held-out surrogate | Clean | DI-FGSM | GLEAM-v | I-FGSM | PGD | SIA | TI-MI-FGSM | R+FGSM | **Ours** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenAI CLIP ViT-B/16 | 20.6 | 48.8 | 51.5 | 45.5 | 41.2 | 38.2 | 30.3 | 20.0 | **75.8** |
| OpenAI CLIP ViT-L/14-224 | 20.6 | 42.4 | 43.3 | 34.8 | 35.5 | 35.8 | 25.2 | 19.4 | **55.8** |
| CLIPA ViT-L/14-336 | 20.6 | 50.0 | 52.1 | 47.9 | 45.5 | 37.9 | 28.5 | 20.6 | **77.0** |
| SigLIP so400m-384 | 20.6 | 49.4 | 50.3 | 45.2 | 44.8 | 38.2 | 28.5 | 19.4 | **76.7** |
| ConvNeXt-V2-L | 20.6 | 52.7 | 52.1 | 47.9 | 44.8 | 37.9 | 27.0 | 20.3 | **77.0** |
| ViT-L/16 (supervised) | 20.6 | 53.0 | 57.0 | 47.6 | 47.6 | 38.8 | 29.1 | 20.3 | **82.1** |
| **Average** | 20.6 | 49.4 ± 0.2 | 51.1 ± 0.2 | 44.8 ± 1.5 | 43.2 ± 1.4 | 37.8 ± 0.6 | 28.1 ± 0.5 | 20.0 ± 0.5 | **74.0 ± 0.4** |

All baselines carry the face term (1/28). The Clean column here uses our seed-42 epoch-14 victims for all 25 people.

## 4. Robustness to JPEG re-encoding (reactive Yo'LLaVA, overall)

| Method | PNG | Q95 | Q90 | Q85 | Q80 | Q75 | Q70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 7.3 | 7.3 | 6.1 | 6.1 | 6.1 | 6.7 | 6.1 |
| I-FGSM + face | 80.6 | 63.0 | 44.8 | 29.1 | 23.0 | 20.0 | 17.6 |
| DI-FGSM + face | 69.1 | 60.0 | 50.0 | 43.6 | 38.5 | 33.3 | 28.8 |
| PGD + face | 76.7 | 63.9 | 40.3 | 26.4 | 19.4 | 16.4 | 14.2 |
| GLEAM-v + face | 47.9 | 50.0 | 47.6 | 49.1 | 46.7 | 43.9 | 41.8 |
| DiffAttack + face | 35.2 | 35.8 | 33.6 | 33.9 | 34.8 | 33.0 | 33.3 |
| **Ours** | **92.1** | **89.1** | **84.8** | **78.5** | **72.4** | **64.2** | **56.1** |

Standard deviations and the remaining baselines: see the supplementary material.

## 5. VQA-style identity questions (reactive, both datasets)

Share of photo × question answers that do not claim the identity (Yo'LLaVA: presence, who-choice, who, name
questions; MyVLM: presence and who-choice).

| Method | Yo'LLaVA | MyVLM |
|---|---:|---:|
| Clean | 25.6 | 18.5 |
| I-FGSM + face | 74.1 | 69.4 |
| DI-FGSM + face | 65.3 | 73.0 |
| PGD + face | 70.9 | 68.2 |
| GLEAM-v + face | 52.7 | 76.2 |
| DiffAttack + face | 47.5 | 68.0 |
| **Ours** | **75.9 ± 0.6** | **82.1 ± 0.3** |

## 6. Image quality (330 protected images vs the clean 336 × 336 photo)

| Method | LPIPS-Alex ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---:|---:|---:|
| TI-MI-FGSM + face | 0.0591 | 37.95 | 0.9504 |
| GLEAM-v + face | 0.0750 | 37.56 | 0.9285 |
| DiffAttack + face | 0.0693 | 30.07 | 0.9042 |
| WDP + face | 0.0209 | 34.96* | 0.9678 |
| **Ours** | 0.0555 | **38.81** | 0.9453 |

\* WDP leaves 60 of 330 images unchanged (infinite PSNR); its PSNR is the mean over the changed images.
The ε-bounded gradient baselines (I-FGSM, DI-FGSM, PGD, R+FGSM, SIA) reach 39.3–42.6 dB with the same 4/255 budget.

## 7. Component ablation (reactive, both datasets, same scoring jobs)

| Variant | Yo'LLaVA | MyVLM | Yo'LLaVA after JPEG Q85 | MyVLM face gate fires ↓ | PSNR | LPIPS |
|---|---:|---:|---:|---:|---:|---:|
| Full (Ours) | 92.1 | 95.5 | 78.5 | 7.6 | 38.81 | 0.0555 |
| w/o prototype term | 91.8 | 93.9 | 79.4 | 7.6 | 38.75 | 0.0564 |
| w/o self term | 87.6 | 96.4 | 75.5 | 6.4 | 38.95 | 0.0512 |
| w/o CES (evidence) term | 75.8 | 97.3 | 57.9 | 3.0 | 38.89 | 0.0542 |
| w/o face term | 94.2 | 88.2 | 83.9 | 96.7 | 38.85 | 0.0566 |
| w/o JPEG in the loop | 77.0 | 84.5 | 9.1 | 5.5 | 38.65 | 0.0576 |
| w/o gradient smoothing | 90.9 | 93.9 | 74.5 | 13.6 | 38.75 | 0.0576 |
| w/o JPEG and smoothing | 77.6 | 86.7 | 10.3 | 12.1 | 38.54 | 0.0601 |
| Clean (no protection) | 7.3 | 31.5 | 6.1 | 97.6 | – | – |

All rows of this table were answered in the same scoring jobs; the full method's MyVLM cell reads 95.5 here and 94.8
in the main table (two scoring runs of the same images; MyVLM generation is sampled per job).

## 8. Compute cost (one AMD MI210, per 336 × 336 image)

| | PGD (501 steps) | Ours (501 steps) |
|---|---:|---:|
| Time per image | 28.9 s | 53.1 s |
| Time per step | 58 ms | 106 ms |
| Peak GPU memory (allocated) | 2.49 GiB | 3.69 GiB |
| Prototype + bank build per identity | – | 6.6 s (prototype) + COCO-100 calibration |
