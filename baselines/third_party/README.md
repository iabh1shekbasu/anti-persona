# Third-party repositories used by the baselines (not vendored)

The baselines use the authors' own transformations / procedures, taken from their repositories at the commits below.
None of the repositories is modified on disk: the needed functions are compiled from the source files at run time
(SIA, TIM, GLEAM-v), or the authors' scripts are run with one loss statement swapped in memory (DiffAttack, WDP).

| Baseline | Repository | Commit | Used for |
|---|---|---|---|
| SIA, TI-MI-FGSM | https://github.com/Trustworthy-AI-Group/TransferAttack | `ae960f546a907a193931a2df12afd0f3fef2e0a5` | `transferattack/input_transformation/sia.py` (block transforms, defaults), `tim.py` (`generate_kernel`) |
| GLEAM-v | https://github.com/LuckAlex/GLEAM | `2ec57f387c0ae53d74e69d2cbaecc820589e933a` | `NURBSAttacker.py` (`bspline_basis`, `init_control_points`, `nurbs_ffd_torch`, `ImageAttacker.dim`), `utils.py` (`gkern`, `get_kernel`) |
| DiffAttack | https://github.com/WindVChen/DiffAttack | `e7094d04c35b2cea57b419f9e763db2605b644b7` | `main.py`, `diff_latent_attack.py` and helpers (run as published; the classifier loss is replaced by our loss) |
| WDP | https://github.com/parham1998/Facial-Privacy-Protection | `bd0a8e4c874b0e8bc385cbc95161033a30b8cbb7` | the authors' protection script (run as published; the adversarial loss is replaced by our loss) |

Fetch them into this folder (or anywhere, then set the environment variables):

```bash
bash baselines/third_party/fetch.sh            # clones the four repositories at the pinned commits
export TRANSFERATTACK_DIR=$PWD/baselines/third_party/TransferAttack
export GLEAM_DIR=$PWD/baselines/third_party/GLEAM
export DIFFATTACK_DIR=$PWD/baselines/third_party/DiffAttack
export WDP_DIR=$PWD/baselines/third_party/WDP
```

Each repository has its own licence; read it before use. DiffAttack and WDP also need their own model downloads
(Stable Diffusion 2 base, Inception-v3, the AMT-GAN face-recognition assets and dlib's 68-landmark model); follow the
authors' READMEs.
