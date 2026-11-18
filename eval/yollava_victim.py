"""The Yo'LLaVA victim as used for every Yo'LLaVA number in the paper (used by reactive_yollava.py and
proactive_yollava.py).

* Base model: LLaVA-1.5-13B (``--model-path``, e.g. a local copy of liuhaotian/llava-v1.5-13b), loaded with the vendored
  LLaVA code in ``victims/yollava/llava`` (``load_pretrained_model``).
* Concept: 17 new tokens ``<sks>`` + ``<token0>..<token15>``; the checkpoint's ``<epoch>-token.pt`` and
  ``<epoch>-lmhead.pt`` rows are written into the input embedding and the LM head (epoch 14 in the paper).
* Question: ``"<sks> is <token0>...<token15> Can you see <sks> in this photo? Answer with a single word or phrase."``
  (``sks`` = the concept name the checkpoint was trained with), greedy decoding, 32 new tokens, seed 42.
* Images: clean photos are prepared exactly like the protection code reads them (RGB -> 336 x 336 bicubic squash ->
  rounded -> PNG); protected images are already 336 x 336 PNGs and are fed as they are. JPEG delivery = PIL JPEG of the
  PNG at quality Q with 4:4:4 sampling.
"""
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
YOLLAVA_DIR = REPO / 'victims' / 'yollava'
N_PREFIX = 16
N_TOKENS = 1 + N_PREFIX          # <sks> + <token0>..<token15>
HIDDEN = 5120                    # LLaVA-1.5-13B embedding width


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def concept_files(checkpoint_dir, epoch='14'):
    """{'token': <dir>/<epoch>-token.pt, 'lmhead': <dir>/<epoch>-lmhead.pt}; raises if either file is missing."""
    files = {k: Path(checkpoint_dir) / f'{epoch}-{k}.pt' for k in ('token', 'lmhead')}
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'concept checkpoint incomplete, missing: {missing}')
    return files


def read_concept(checkpoint_dir, epoch='14', hidden=HIDDEN):
    """Load one concept checkpoint (trained here or released) on the CPU and check it before any model is loaded.

    A concept is two tensors of shape [17, hidden]: the rows of the 17 added tokens in the order <sks>, <token0> ...
    <token15>; ``token.pt`` goes into the input embedding, ``lmhead.pt`` into the LM head. Returns (token, lmhead).
    """
    import torch
    files = concept_files(checkpoint_dir, epoch)
    token = torch.load(files['token'], map_location='cpu')
    lmhead = torch.load(files['lmhead'], map_location='cpu')
    for name, t in (('token', token), ('lmhead', lmhead)):
        if not isinstance(t, torch.Tensor) or tuple(t.shape) != (N_TOKENS, hidden):
            shape = tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__
            raise ValueError(f'{files[name]}: shape {shape} != ({N_TOKENS}, {hidden})')
    return token, lmhead


def prepare_clean(path, out_png):
    """RGB -> float [0, 255] -> 336 x 336 bicubic (align_corners False) -> clamp -> round -> uint8 PNG."""
    import torch
    import torch.nn.functional as F
    with Image.open(path) as im:
        rgb = np.array(im.convert('RGB'))
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).mul(255)
    x = F.interpolate(x[None], (336, 336), mode='bicubic', align_corners=False)[0].clamp(0, 255)
    Image.fromarray(x.round().byte().permute(1, 2, 0).numpy()).save(out_png)
    return Path(out_png)


def jpeg_copy(png, out_jpg, quality):
    with Image.open(png) as image:
        image.convert('RGB').save(out_jpg, quality=quality, subsampling=0)
    return Path(out_jpg)


class YoLLaVA:
    """LLaVA-1.5-13B + one Yo'LLaVA concept at a time."""

    def __init__(self, model_path, sks):
        import torch
        if str(YOLLAVA_DIR) not in sys.path:
            sys.path.insert(0, str(YOLLAVA_DIR))
        from llava.eval.my_llava import eval_model, get_image_tensor, get_query
        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        torch.set_num_threads(2)
        torch.manual_seed(42)
        np.random.seed(42)
        self.get_image_tensor, self.eval_model = get_image_tensor, eval_model
        self.model_path, self.sks = str(model_path), sks
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            model_path=self.model_path, model_base=None, model_name=get_model_name_from_path(self.model_path))
        tokens = [f'<{sks}>'] + [f'<token{i}>' for i in range(N_PREFIX)]
        added = self.tokenizer.add_tokens(tokens)
        self.ids = self.tokenizer.convert_tokens_to_ids(tokens)
        if added != len(tokens) or len(set(self.ids)) != len(tokens):
            raise RuntimeError(f'the {len(tokens)} concept tokens were not all new (added {added})')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()
        options = SimpleNamespace(model_path=self.model_path, conv_mode=None, temperature=0, top_p=None, num_beams=1,
                                  max_new_tokens=32)
        prompt = tokens[0] + ' is ' + ''.join(tokens[1:])
        self.question = prompt + f' Can you see <{sks}> in this photo? Answer with a single word or phrase.'
        self.options = get_query(options, self.question, model=self.model)

    def load_concept(self, checkpoint_dir, epoch='14'):
        """Write the concept's 17 token rows and 17 LM-head rows, read them back; returns the checkpoint sha256s."""
        import torch
        files = concept_files(checkpoint_dir, epoch)
        token, lmhead = read_concept(checkpoint_dir, epoch, hidden=self.model.get_input_embeddings().weight.shape[1])
        with torch.no_grad():
            self.model.get_input_embeddings().weight[self.ids] = token.to(self.model.device, dtype=self.model.dtype)
            self.model.lm_head.weight[self.ids] = lmhead.to(self.model.lm_head.weight.device, dtype=self.model.dtype)
        self.model.eval()
        # read-back: the rows the model will use are exactly this checkpoint's (NaN counts as equal to NaN)
        for got, want in ((self.model.get_input_embeddings().weight[self.ids], token),
                          (self.model.lm_head.weight[self.ids], lmhead)):
            got, want = got.detach().cpu(), want.to(dtype=self.model.dtype)
            if not torch.isclose(got, want, rtol=0, atol=0, equal_nan=True).all():
                raise RuntimeError(f'{checkpoint_dir}: rows read back differ from the checkpoint')
        return {k: sha256(p) for k, p in files.items()}

    def ask(self, image_path):
        tensor, sizes = self.get_image_tensor(self.options, [str(image_path)], self.model, self.processor)
        return self.eval_model(self.options, self.model, tensor, sizes, self.processor, self.tokenizer)


def write_jsonl(path, rows):
    with Path(path).open('x') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
