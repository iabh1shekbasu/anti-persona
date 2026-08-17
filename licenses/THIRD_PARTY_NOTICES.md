# Third-Party Notices

This repository is **mixed-license**. Our own code is MIT; the vendored victim
code retains its original license. Please observe each component's terms.

| Path | Component | Copyright | License |
|---|---|---|---|
| `anti_persona/`, `scripts/`, `configs/`, `docs/` | Anti-Persona (this work) | © 2026 Abhishek Basu, Fahad Shamshad, Karthik Nandakumar | [MIT](../LICENSE) |
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

The MyVLM face-recognition head downloads
[CVLface](https://github.com/mk-minchul/CVLface) AdaFace models at runtime from
Hugging Face; those weights are governed by their own licenses and are **not**
redistributed here.
