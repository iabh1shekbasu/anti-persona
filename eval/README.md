# Evaluation: proactive and reactive, on Yo'LLaVA and MyVLM

Everything here runs the two personalized LVLMs exactly as in the paper and turns their answers into the table numbers.
Run all commands from the repository root (`python -m eval.<script> --help` lists every flag).

| File | What it does |
|---|---|
| `semantic_parser.py` | The answer parsers. Yo'LLaVA: `semantic_answer`. MyVLM: `myvlm_recognised` (MyVLM's own probe parser, `victims/myvlm/probe_scoring.py`, called unchanged). Pure Python. |
| `yollava_victim.py` | Yo'LLaVA victim: LLaVA-1.5-13B + one concept (17 token rows), question, greedy decoding, image preparation, JPEG delivery. |
| `reactive_yollava.py` | Reactive Yo'LLaVA for one identity: clean + every protected arm + target-absent photos, PNG and JPEG qualities. |
| `proactive_yollava.py` | Proactive Yo'LLaVA: `clean-album` (the Clean training album), `train` one concept per album and victim model seed, `score` the concepts on the clean test photos. |
| `myvlm_victim.py` | MyVLM victim: album building (face rule, name to `sks`), training config, training, inference, buffalo_l gate rows. |
| `reactive_myvlm.py` | `train` the clean victim of one identity; `score` clean / protected / target-absent photos. |
| `proactive_myvlm.py` | `train` one concept per protected album; `score` the concepts on the clean test photos. |
| `aggregate.py` | Answer files to table cells: % of all test photos not recognised, mean ± std over image seeds 0 and 202 (proactive Yo'LLaVA: over victim model seeds 42–46). |
| `tests/test_semantic_parser.py` | CPU test of the Yo'LLaVA and MyVLM parsers on 58 answers recorded in the paper's scoring jobs + 31 edge cases. |
| `tests/test_proactive_yollava.py` | CPU test of the model-seed aggregation (reproduces the paper's proactive Yo'LLaVA cells from counts of recorded answers), the proactive command line, its training command and the checkpoint checks. |

## Metric

- **Cell = % of ALL test photos the victim does NOT recognise** (higher = better protection). No conditioning on the
  clean answer; the Clean row is the same quantity for unprotected photos.
- Yo'LLaVA: not recognised = the semantic parser returns `no`. Rules, in order: lowercase, drop `<...>` concept tokens,
  strip leading punctuation; an answer starting with the word "yes" / "no" is yes / no; else any negation phrase
  ("is not", "not in", "not visible", "cannot see", "do not see", "not recognize", ...) is no; else any affirmative phrase
  ("is in", "is visible", "can see", "appears", ...) is yes; else invalid. **An invalid answer counts as recognised.**
- MyVLM: not recognised = `myvlm_recognised(answer, "sks")` is False.
- Image seeds: every method is generated with image seeds 0 and 202 (the per-photo seed is run seed + the photo's index
  in its album). For a group of photos (one dataset, or both datasets together) the number is computed per seed by
  pooling the photos, and reported as **mean ± |seed 0 − seed 202| / 2**. A deterministic method (Clean, I-FGSM,
  TI-MI-FGSM) is one run counted for both seeds (± 0.0). Victim model seed: 42 everywhere, except proactive Yo'LLaVA,
  which is run at victim model seeds 42–46 and reported as **mean ± std over the model seeds** (see
  [Proactive Yo'LLaVA](#proactive-yollava)).
- Arm / concept names `<method>_s0` and `<method>_s202` are paired automatically by `aggregate.py`.

`aggregate.py` was checked against the recorded answers of the paper's scoring jobs: it reproduces the published cells
exactly, e.g. Ours (final) reactive Yo'LLaVA 92.1 ± 1.8 and reactive MyVLM 94.8 ± 0.9 (paper-10 / MMPB-15 / both:
88.9 ± 1.1 / 96.0 ± 2.7 / 92.1 ± 1.8 and 95.0 ± 0.6 / 94.7 ± 1.3 / 94.8 ± 0.9), PGD + face Yo'LLaVA 76.7 ± 0.9
(JPEG Q85 26.4 ± 1.5), PGD + face MyVLM 84.2 ± 1.8, GLEAM-v + face MyVLM 84.8 ± 1.2, I-FGSM + face
Yo'LLaVA 80.6 ± 0.0, Clean 7.3 / 31.5.

## Yo'LLaVA protocol

- Base model: LLaVA-1.5-13B (`liuhaotian/llava-v1.5-13b`), vendored LLaVA code in `victims/yollava/llava`.
- Concept: `<sks>` + 16 prefix tokens, trained with `victims/yollava/personalize.py` (the Yo'LLaVA trainer; flags
  `--prefix_token 16 --epoch 15 --user_prompt --recog_only --random_image --text_only --seed S`, S = 42; proactive
  Yo'LLaVA: S = 42–46; training JSONs via `--json_path`). **The epoch-14 checkpoint is scored** (`14-token.pt`,
  `14-lmhead.pt`).
- Question: `"<sks> is <token0>…<token15> Can you see <sks> in this photo? Answer with a single word or phrase."`,
  greedy decoding, 32 new tokens.
- Images: clean photos are read as RGB, squashed to 336 × 336 (bicubic), rounded and saved as PNG (the same start point
  as the protection code); protected images are 336 × 336 PNGs fed as they are; JPEG delivery = PIL JPEG at quality Q
  with 4:4:4 sampling.
- Target-absent control: other people's photos (`--negative-dir`, 9 per paper-10 identity, 14 per MMPB-15 identity).

## MyVLM protocol

- MyVLM with the LLaVA backbone, VQA personalisation, PERSON concept, face concept head **insightface buffalo_l**
  (`victims/myvlm/concept_heads/face_recognition/head_buffalo.py`, the upstream MyVLM head), gate threshold **0.675**,
  concept identifier **`sks`** (the person's name in the captions and Q&A is replaced by `sks`).
- Training: `victims/myvlm/train_vqa_fixed.py` (MyVLM's trainer + one data-loader fix, see its docstring); 200 steps,
  lr 1.0, reg_lambda 10, batch min(2, kept photos), float32, victim seed 42. A training photo without a buffalo_l face is
  dropped (MyVLM's own rule); a concept left with no photo is UNTRAINABLE and recognises nothing.
- Inference: `inference/run_myvlm_inference.py` at iteration 199, prompt
  `"Can you see if sks is in this photo? Answer with a single word or phrase."`.

## Reactive vs proactive

- **Reactive**: the victim's concept is trained on CLEAN photos; protected TEST photos are asked about.
- **Proactive** (MyVLM): the TRAINING album is protected, the concept is trained on it, and CLEAN test photos are asked
  about.
- **Proactive** (Yo'LLaVA): the same, with one Yo'LLaVA concept per album and victim model seed (next section).

## Proactive Yo'LLaVA

The person's training album is protected, a Yo'LLaVA concept is trained on it, and the concept is asked about the
person's clean test photos. The Clean row is a concept trained on the unprotected album, squashed to 336 × 336 like the
protection code reads it. Every method uses the same training recipe; only the training album and the model seed change.

```bash
# 0) the Clean album of one person (training photos -> 336 x 336 PNGs, same file stems)
python -m eval.proactive_yollava clean-album --train-dir <clean training photos> --out albums/clean/thao

# 1) one concept per (album, model seed), one GPU each: Clean, Ours at image seeds 0 and 202, each baseline
python -m eval.proactive_yollava train --identity thao --model-path <llava-v1.5-13b> \
    --json-path <training JSONs> --protected-album albums/ours_s0/thao \
    --method ours_s0 --model-seed 42 --work runs/proactive
#    -> runs/proactive/checkpoints/thao/ours_s0-c42/14-token.pt and 14-lmhead.pt   (--dry-run: checks + command only)

# 2) score every concept of one person at one model seed (one model load)
python -m eval.proactive_yollava score --identity thao --model-path <llava-v1.5-13b> --model-seed 42 \
    --concept clean=runs/proactive/checkpoints/thao/clean-c42 \
    --concept ours_s0=runs/proactive/checkpoints/thao/ours_s0-c42 \
    --concept ours_s202=runs/proactive/checkpoints/thao/ours_s202-c42 \
    --test-dir <clean test photos> --negative-dir <other people's photos> --out scores/paper10/thao-c42

# 3) the table over people and model seeds 42-46
python -m eval.aggregate --input paper10=scores/paper10 --input mmpb15=scores/mmpb15 [--per-model-seed]
```

- **Training** (`train`): `victims/yollava/personalize.py --json_path <training JSONs> --data_root <album> --sks_name
  <identity> --prefix_token 16 --epoch 15 --seed S --user_prompt --recog_only --random_image --text_only`. The trainer's
  own settings: AdamW, lr 1e-3, batch 1; only the 17 concept rows of the input embedding and the LM head change. 15
  epochs are trained and **the epoch-14 checkpoint is scored**.
- **Training data** per person, read from `<training JSONs>/<identity>/`: `positive_recognition.json` (the album
  photos), `negative_recognition-laion.json` (100 photos without the person), `negative_recognition-random-imgs.json`
  (99 random images), `text-only-conversation.json`. The trainer takes the positive photos from the album by file name,
  so the album must hold exactly the file names of `positive_recognition.json`; every other image is opened at the path
  written in the JSON (a relative path is read from the folder you run the command in). `train` checks both before it
  starts.
- **Model seeds** 42, 43, 44, 45, 46 (`--model-seed`): the trainer's seed (Python, NumPy, torch and the order of the
  training samples). Scoring always uses seed 42 and greedy decoding, so the model seed changes only the concept.
- **Image seeds**: Ours and every baseline have protected albums for image seeds 0 and 202 (concept names `<method>_s0`,
  `<method>_s202`); I-FGSM, TI-MI-FGSM and Clean have one album.
- **Scoring** (`score`): question, decoding and image preparation as in the Yo'LLaVA protocol above. Rows with
  `set = test` (the person's clean test photos) make the table; `--negative-dir` (target-absent) and
  `--train-album NAME=ALBUM` (the concept asked about its own training photos) are controls only.
- **Number**: per model seed exactly as in every other table (photos pooled per image seed, mean of seeds 0 and 202);
  then **mean ± std (population) over model seeds 42–46**. `aggregate.py` does this whenever the rows carry more than
  one model seed; `--per-model-seed` adds each model seed's own table. A model seed with missing photos is left out of
  a cell and the cell says so.
- **Expected** (25 identities, 165 test photos, model seeds 42–46): **Ours 91.2 ± 4.4**, Clean 29.1 ± 7.8
  (paper-10 / MMPB-15: Ours 90.1 ± 4.6 / 92.5 ± 4.6, Clean 26.0 ± 8.8 / 32.8 ± 7.9).
  `tests/test_proactive_yollava.py` reproduces these cells from counts of the recorded answers.
- All 25 identities use our concepts here (the Yo'LLaVA authors' checkpoints are used in reactive only).
- **Our concept checkpoints**: coming soon. A concept folder holds `14-token.pt` and `14-lmhead.pt`, each [17, 5120]:
  the rows of `<identity>`, `<token0>` … `<token15>` for the input embedding and the LM head of LLaVA-1.5-13B. Score it
  with `score --identity <identity> --concept NAME=<folder>`; `read_concept` in `yollava_victim.py` checks the files
  before the model is loaded and `load_concept` reads the rows back after writing them.

## External checkpoints and data (not in this repository)

| Item | Used by | Where to get it |
|---|---|---|
| LLaVA-1.5-13B | Yo'LLaVA | `liuhaotian/llava-v1.5-13b` (Hugging Face) |
| Yo'LLaVA concept checkpoints, paper-10 (except yuheng) | reactive Yo'LLaVA, paper-10 | the Yo'LLaVA authors' release (github.com/WisconsinAIVision/YoLLaVA) |
| Our Yo'LLaVA concepts (MMPB-15 and yuheng, seed 42, epoch 14) | reactive Yo'LLaVA | **placeholder: to be released**; or train them with `victims/yollava/personalize.py` (recipe above) on the clean albums |
| Our proactive Yo'LLaVA concepts (25 identities × model seeds 42–46 × Clean / Ours / 9 baselines, epoch 14) | proactive Yo'LLaVA | **checkpoints coming soon**; or train them with `proactive_yollava.py train` |
| Yo'LLaVA training JSONs | Yo'LLaVA training | paper-10: Yo'LLaVA release (`training-data`); MMPB-15: released with the benchmark (coming soon) |
| MyVLM LLaVA backbone and its dependencies | MyVLM | see `victims/myvlm` and the MyVLM repository (github.com/snap-research/MyVLM) |
| insightface `buffalo_l` model pack | MyVLM face gate, face term | insightface model zoo (`~/.insightface/models/buffalo_l`) |
| MyVLM per-photo captions + 9 Q&A pairs | MyVLM training | **coming soon, with the benchmark** |
| Our MyVLM `sks` victims (seed 42) | reactive MyVLM | **placeholder: to be released**; or train with `reactive_myvlm.py train` |
| Benchmark photos | everything | coming soon |

## What is faithful and what is simplified

- Faithful (same code path or a verbatim port): parsers, Yo'LLaVA model / concept loading, question, decoding, image
  preparation and JPEG delivery, the Yo'LLaVA trainer and recipe, MyVLM trainer (with the fix), MyVLM inference config,
  gate rows, the neutral-name replacement, the metric and the seed aggregation (checked on recorded answers, above).
- One Yo'LLaVA trainer: the paper's paper-10 concepts were trained with Yo'LLaVA's original training script, the
  MMPB-15 concepts with `personalize.py`. Both run the same training loop; `personalize.py` only adds `--json_path`,
  sets `CUBLAS_WORKSPACE_CONFIG`, prints layer shapes and saves `best-*.pt` only on a strict improvement (never scored), so `personalize.py --json_path` stands for both. The flags that
  `proactive_yollava.py train` builds equal those of every recorded proactive training command of the paper (output
  paths aside), and `clean-album` gives the paper's Clean album pixel for pixel on the photos compared.
- Simplified: the paper's runs went through a job queue that pinned the sha256 of every input, checked that every task
  finished, and refused stale outputs; those checks are not re-implemented here. The MyVLM album builder keeps photos in
  stem order and does not record whether a face was found only on a bordered canvas (the paper's MyVLM albums never used
  the bordered-canvas fallback). The paper's proactive Yo'LLaVA jobs scored the Clean concept and one method's concept
  per job; `score` takes all concepts of one person and model seed in one job (each concept overwrites the same 17
  rows and decoding is greedy). None of this changes an answer.
- Not run here: the GPU scripts were not executed in this repository (no GPU in the release check); tested on the CPU:
  the parsers and `aggregate.py` on recorded answers, and the proactive Yo'LLaVA command line, training command and
  checkpoint checks.
