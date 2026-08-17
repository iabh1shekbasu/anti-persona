# Reproducing Tables 1 & 2

This guide reproduces the two main tables of the paper for the **Ours** method.
The protection attack (`anti_persona/`) and the victim code (`victims/`) are all
in this repository.

- **Table 1 — reactive**: victim personalized on *clean* references; the *test*
  image is protected at inference (Yo'LLaVA + MyVLM).
- **Table 2 — proactive**: *reference* images protected before personalization;
  victim trained on protected data, evaluated on *clean* test images (Yo'LLaVA).

**Metric — Protection Rate**: % of identity queries the personalized LVLM fails
to recognize (higher = stronger). In `evaluate.py` / `run_paper_eval.py` this is
the per-category `x.xxx (c/t)` line (GT = `No`, so `Pred = No` = protected).

## 0. Environments & assets

- **Attack** (`anti_persona/`): `bash setup_env.sh` → conda env `anti-persona`.
- **Yo'LLaVA victim** (`victims/yollava/`, needs the `llava` deps + LLaVA-1.5-13B
  weights): run its scripts from inside `victims/yollava/` so `llava/` imports.
- **MyVLM victim** (`victims/myvlm/`, needs the `myvlm` deps): the FR head
  downloads [CVLface](https://github.com/mk-minchul/CVLface) AdaFace models from
  Hugging Face on first use.
- **Data**: [Yo'LLaVA dataset](https://huggingface.co/datasets/thaoshibe/YoLLaVA).
  10 identities: `ciin, denisdang, khanhvy, oong, phuc-map, thao, thuytien,
  viruss, willinvietnam, yuheng` (`ciin`/`yuheng` have 5 test images, rest 10).

### Checkpoints (clean reactive victims)

Reactive Table 1 uses the **authors' pretrained Yo'LLaVA checkpoints** for 9
identities (`pretrained_concepts/checkpoints/<id>/best-token.pt`, loaded with
`--epoch best --exp_name '' --prefix_token 16`), plus a **retrained seed-7
`yuheng`** (the authors' `yuheng` checkpoint recognizes 0/5 of its own clean
images).

> **Checkpoints (Google Drive):** _<add link here>_

## 1. Generate protected images  (`anti_persona/`)

Paper "Ours" config (`configs/ours.yaml`):

```bash
bash scripts/protect.sh <input_dir> <output_dir>
```

- **Reactive (Table 1):** protect the identity's **test** images.
- **Proactive (Table 2):** protect the identity's **train / reference** images.

Verified: mean **PSNR 39.4 dB**, L∞ = 4/255 on an RTX A6000.

## 2. Table 1 — reactive

Lay the protected test images out as `<data_root>/<method>/*.png` (the eval
enumerates the immediate sub-directories of `--data_root` as categories).

**Yo'LLaVA column** — from `victims/yollava/`:

```bash
cd victims/yollava
CUDA_VISIBLE_DEVICES=<gpu> python evaluate.py \
    --model_path /path/to/llava-v1.5-13b \
    --checkpoint_path /path/to/pretrained_concepts/checkpoints \
    --sks_name <id> --epoch best --exp_name "" --prefix_token 16 \
    --data_root <data_root> --save_txt
```

The `relaxed_smooth_eps4: x.xxx (c/t)` line is the protection rate. **Verified**
for `thao` with the authors' `best` checkpoint: Ours = `1.000 (10/10)` = 100%,
matching the paper.

**MyVLM column** — from `victims/myvlm/` (concept head trained on clean refs;
uses the CVLface AdaFace FR head, **not** InsightFace `buffalo_l`):

```bash
cd victims/myvlm
python run_paper_eval.py --concepts <id> --variants clean ours
```

## 3. Table 2 — proactive  (Yo'LLaVA)

Protect the **reference** images, personalize on them, evaluate on the **clean**
test set — from `victims/yollava/`:

```bash
# 1) protect the reference images (attack env)
bash ../../scripts/protect.sh <data>/<id>/train protected_train/<id>

# 2) personalize on the PROTECTED references
cd victims/yollava
python personalize.py --model_path /path/to/llava-v1.5-13b \
    --data_root protected_train --sks_name <id> \
    --checkpoint_path <ckpt> --exp_name ours --prefix_token 16 --seed 42

# 3) evaluate recognition on the CLEAN test images
python evaluate.py --model_path /path/to/llava-v1.5-13b \
    --checkpoint_path <ckpt> --sks_name <id> --exp_name ours \
    --epoch best --prefix_token 16 --data_root <clean_test_root> --save_txt
```

## Notes

- The PGD attack is not bit-reproducible across hardware (no
  `--strict_determinism`), so a fresh run can differ by a couple of borderline
  images per 90 (pooled Ours ≈ 91–93%); the evaluation is deterministic.
- Reference hardware: a single NVIDIA RTX A6000 (48 GB) or RTX 5000 (32 GB).
- Licensing: `victims/myvlm/` is Snap non-commercial-academic — see
  [`licenses/THIRD_PARTY_NOTICES.md`](../licenses/THIRD_PARTY_NOTICES.md).
