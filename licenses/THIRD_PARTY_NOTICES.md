# Third-Party Notices

This repository is **mixed-license**. Our own code is MIT; the vendored victim
code retains its original license. Please observe each component's terms.

| Path | Component | Copyright | License |
|---|---|---|---|
| `anti_persona/`, `scripts/`, `configs/`, `docs/`, `tests/`, `eval/`, `baselines/` (except `baselines/third_party/`) | Anti-Persona (this work) | © 2026 Abhishek Basu, Fahad Shamshad, Karthik Nandakumar | [MIT](../LICENSE) |
| `victims/yollava/llava/` | [LLaVA](https://github.com/haotian-liu/LLaVA) | © Haotian Liu et al. | [Apache-2.0](Apache-2.0.txt) |
| `victims/yollava/{personalize,evaluate}.py` | [Yo'LLaVA](https://github.com/WisconsinAIVision/YoLLaVA) (NeurIPS 2024) | © Yo'LLaVA authors | [Apache-2.0](Apache-2.0.txt) |
| `victims/myvlm/` | [MyVLM](https://github.com/snap-research/MyVLM) | © 2024 Snap Inc. | [Snap non-commercial academic](MyVLM-Snap-LICENSE.txt) |

## ⚠️ Important: the MyVLM component is non-commercial

`victims/myvlm/` is redistributed from [MyVLM](https://github.com/snap-research/MyVLM)
under Snap Inc.'s license, which permits use **for non-commercial, academic
purposes only** and requires that Snap's copyright notice be retained (kept at
`victims/myvlm/LICENSE`). Portions of MyVLM derived from LLaVA / MiniGPT-4 are
under Apache-2.0 / BSD as noted in that file.

If you need a permissively (MIT/Apache) licensed subset, use only `anti_persona/`
and `victims/yollava/`, and obtain MyVLM separately from its upstream repository.

## Models and data that are downloaded, not redistributed

| What | Obtained by | Licence |
|---|---|---|
| CLIP ViT-L/14-336 (`openai/clip-vit-large-patch14-336`) | `scripts/download_weights.py` | MIT (OpenAI CLIP) |
| DeepLabV3-ResNet101 (torchvision) | `scripts/download_weights.py` | BSD-3-Clause (torchvision) |
| insightface buffalo_l (SCRFD det_10g, ArcFace w600k_r50) | `scripts/download_weights.py` | non-commercial research use only (insightface model zoo) |
| COCO val2017, 100 photos of `configs/coco100.json` | `scripts/download_coco100.py` | CC BY 4.0 annotations; images under their Flickr licences |
| Yo'LLaVA dataset (Paper-10) | official Yo'LLaVA release | see the Yo'LLaVA release |
| MMPB (MMPB-15) | gated Hugging Face dataset `snu-aidas/MMPB` | CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike); accept the dataset's conditions on Hugging Face |

The MyVLM face-recognition head downloads
[CVLface](https://github.com/mk-minchul/CVLface) AdaFace models at runtime from
Hugging Face; those weights are governed by their own licenses and are **not**
redistributed here.
