"""The six surrogate encoders of the black-box setting, as feature-entry extractors for our loss.

The black-box rows swap ONLY the encoder: instead of the victims' CLIP ViT-L/14-336, five of these six surrogates (one
held out per ensemble, combos F6_noA ... F6_noW) provide the features. Each surrogate returns a list of
[tokens, batch, dim] feature entries, token-first like anti_persona.encoder.ClipEncoder, and says how its tokens map
onto the image grid:
  A  OpenAI CLIP ViT-B/16 (clip package, ~/.cache/clip/ViT-B-16.pt)   variable size, pos-embed bicubic interpolation
  B  OpenAI CLIP ViT-L/14 224 (clip package, ViT-L-14.pt)             variable size, pos-embed bicubic interpolation
  D  CLIPA ViT-L/14-336 datacomp1B (open_clip hf-hub UCSC-VLAA)       variable size, pos-embed bicubic interpolation
  S  SigLIP so400m patch14-384 (HF)                                   re-resized to 384
  X  ConvNeXtV2-L 22k-224 (HF)                                        re-resized to 224; 5 entries = stem + 4 stages
  W  supervised ViT-L/16 224 (HF google/vit-large-patch16-224)        re-resized to 224
Entries: ViTs = pre-block tokens (after the model's own pre-norm, if any) + every block output; ConvNeXtV2 = the stem
output + every stage output, each [C, H, W] map flattened row-major to H*W tokens. CLS: A, B, D, W carry a class token
at index 0; S and X have none.
Normalisation: each model's own preprocessing mean/std (0-255 scale), applied BEFORE the fixed-size resize; CLIPA (D)
uses its own preprocess_cfg mean/std (ImageNet constants).
Weights are read from the local caches (HF_HUB_CACHE for D/S/X/W snapshots, ~/.cache/clip for A/B); download them once
with the libraries' own loaders (see baselines/README.md).
"""
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

# Hugging Face hub cache holding the surrogate snapshots (loaded by path from refs/main, never resolved online);
# override with HF_HUB_CACHE. The two OpenAI CLIP archives come from the clip package cache (~/.cache/clip).
HF_HUB = Path(os.environ.get('HF_HUB_CACHE', Path.home() / '.cache/huggingface/hub'))
CLIP_DIR = Path.home() / '.cache/clip'
POOL = ('A', 'B', 'D', 'S', 'X', 'W')
COMBOS = {'F6_no' + k: [m for m in POOL if m != k] for k in POOL}   # leave-one-out ensembles F6_noA .. F6_noW
NORMALISATION_NOTE = ('D (CLIPA) uses its own preprocess_cfg mean/std (ImageNet constants); all others use their '
                      'own preprocessing constants (OpenAI CLIP or their HF preprocessor config).')
OPENAI_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_STD = (0.26862954, 0.26130258, 0.27577711)
SPECS = {
    'A': dict(name='openai_clip_vitb16', family='openai_clip', model='ViT-B/16', file=CLIP_DIR / 'ViT-B-16.pt',
              fixed=None, patch=16, cls=True),
    'B': dict(name='openai_clip_vitl14_224', family='openai_clip', model='ViT-L/14', file=CLIP_DIR / 'ViT-L-14.pt',
              fixed=None, patch=14, cls=True),
    'D': dict(name='clipa_vitl14_336', family='open_clip_hfhub', model='hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B',
              repo='models--UCSC-VLAA--ViT-L-14-CLIPA-336-datacomp1B', fixed=None, patch=14, cls=True),
    'S': dict(name='siglip_so400m_384', family='hf_siglip', model='google/siglip-so400m-patch14-384',
              repo='models--google--siglip-so400m-patch14-384', fixed=384, patch=14, cls=False),
    'X': dict(name='convnextv2_large', family='hf_convnextv2', model='facebook/convnextv2-large-22k-224',
              repo='models--facebook--convnextv2-large-22k-224', fixed=224, strides=(4, 4, 8, 16, 32), cls=False),
    'W': dict(name='vit_large_supervised', family='hf_vit', model='google/vit-large-patch16-224',
              repo='models--google--vit-large-patch16-224', fixed=224, patch=16, cls=True),
}
WEIGHT_FILES = {   # the files that carry each surrogate's weights (hashed at build, stat-bound at every load)
    'A': [SPECS['A']['file']], 'B': [SPECS['B']['file']],
    'D': ['open_clip_pytorch_model.bin', 'open_clip_config.json'],
    'S': ['model.safetensors', 'config.json', 'preprocessor_config.json'],
    'X': ['model.safetensors', 'config.json', 'preprocessor_config.json'],
    'W': ['pytorch_model.bin', 'config.json', 'preprocessor_config.json'],
}


def snapshot(key):
    spec = SPECS[key]
    repo = HF_HUB / spec['repo']
    return repo / 'snapshots' / (repo / 'refs/main').read_text().strip()


def weight_files(key):
    if key in ('A', 'B'):
        return [Path(p) for p in WEIGHT_FILES[key]]
    return [snapshot(key) / name for name in WEIGHT_FILES[key]]


def bound_files(key):
    """Every file a loader of this surrogate could read: the clip-package archive (A, B: the loader opens exactly that
    file), or EVERY file of the snapshot directory (D, S, X, W: the loaders choose among the directory's files, e.g.
    a safetensors next to a .bin), so whatever file is picked is hashed at build and stat-checked at every task."""
    if key in ('A', 'B'):
        return weight_files(key)
    return sorted(snapshot(key) / name for name in os.listdir(snapshot(key)))


FILE_PREFIX = {'A': 'visual.', 'B': 'visual.', 'D': 'visual.', 'S': 'vision_model.', 'X': 'convnextv2.', 'W': 'vit.'}
CHECKPOINT = {'A': 0, 'B': 0, 'D': 0, 'S': 0, 'X': 0, 'W': 0}     # index into weight_files(key) of the weight file


def fingerprint(module):
    """sha256 over every state_dict entry of the loaded network (name, shape, dtype and an evenly strided sample of at
    most ~4096 values per tensor): cheap enough for every task, and any other checkpoint changes it."""
    import hashlib
    h = hashlib.sha256()
    for name, t in sorted(module.state_dict().items()):
        t = t.detach().to('cpu').contiguous()
        h.update(name.encode() + str(tuple(t.shape)).encode() + str(t.dtype).encode())
        flat = t.reshape(-1)
        h.update(flat[::max(1, flat.numel() // 4096)].numpy().tobytes())
    return h.hexdigest()


def file_tensors(key):
    """{name: tensor} read DIRECTLY from the recorded weight file (no loader file selection involved)."""
    path = weight_files(key)[CHECKPOINT[key]]
    if key in ('A', 'B'):
        return torch.jit.load(str(path), map_location='cpu').state_dict()
    if str(path).endswith('.safetensors'):
        from safetensors.torch import load_file
        return load_file(str(path), device='cpu')
    return torch.load(str(path), map_location='cpu', weights_only=True)


def verify_against_file(surrogate):
    """Build-time proof that the loader used the recorded file: every state_dict entry of the loaded network equals
    the recorded file's tensor of the same name (after the checkpoint's model prefix), as float32."""
    stored = file_tensors(surrogate.key)
    prefix = FILE_PREFIX[surrogate.key]
    loaded = surrogate.net.state_dict()
    missing = [n for n in loaded if prefix + n not in stored]
    assert not missing, (surrogate.key, 'loaded entries absent from the recorded file', missing[:5])
    differ = [n for n, t in loaded.items() if not torch.equal(t.detach().cpu(), stored[prefix + n].to(t.dtype))]
    assert not differ, (surrogate.key, 'loaded weights differ from the recorded file', differ[:5])
    return len(loaded)


def interpolate_patch_pos(pos, gh, gw):
    """[1 + g*g, D] positional table -> [1 + gh*gw, D]; bicubic on the patch grid, CLS row unchanged."""
    grid = round((pos.shape[0] - 1) ** 0.5)
    if (gh, gw) == (grid, grid):
        return pos
    patch = F.interpolate(pos[1:].T.reshape(1, -1, grid, grid), size=(gh, gw), mode='bicubic',
                          align_corners=False).reshape(-1, gh * gw).T
    return torch.cat([pos[:1], patch], dim=0)


class Surrogate(torch.nn.Module):
    """forward(image in pixel units [0,255], any size) -> list of [tokens, batch, dim] feature entries."""

    def __init__(self, key, device):
        super().__init__()
        self.key, self.spec, self.device = key, SPECS[key], torch.device(device)
        family = self.spec['family']
        loader = getattr(self, '_load_' + family)
        with torch.random.fork_rng(devices=[]):      # loading must not move any global RNG the attack uses
            mean, std = loader()                      # self.net on CPU, float32, eval, frozen
        self.fingerprint = fingerprint(self.net)      # of the weights actually loaded
        self.net = self.net.to(self.device)
        self.register_buffer('mean', torch.tensor(mean, device=self.device).view(1, 3, 1, 1) * 255)
        self.register_buffer('std', torch.tensor(std, device=self.device).view(1, 3, 1, 1) * 255)
        self.files = [str(p) for p in weight_files(key)]

    # ------------------------------------------------------------------ loaders
    def _load_openai_clip(self):
        import clip
        assert self.spec['file'].is_file(), self.spec['file']
        model, _ = clip.load(self.spec['model'], device='cpu', jit=False, download_root=str(CLIP_DIR))
        self.net = model.visual.float().eval().requires_grad_(False)
        native = round((self.net.positional_embedding.shape[0] - 1) ** 0.5) * self.net.conv1.kernel_size[0]
        assert self.net.conv1.kernel_size[0] == self.spec['patch'] and native == 224, (self.key, native)
        return OPENAI_MEAN, OPENAI_STD

    def _load_open_clip_hfhub(self):
        import open_clip
        # the cached snapshot of hf-hub:UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B, loaded from its directory (no hub lookup,
        # so an inherited HF cache variable cannot redirect it; the file hashes are bound at build)
        model, _ = open_clip.create_model_from_pretrained('local-dir:' + str(snapshot('D')), device='cpu')
        self.net = model.visual.float().eval().requires_grad_(False)
        assert self.net.conv1.kernel_size[0] == self.spec['patch'] and self.net.positional_embedding.shape[0] == 577
        assert self.net.transformer.batch_first and self.net.class_embedding is not None
        cfg = json.loads((snapshot('D') / 'open_clip_config.json').read_text())['preprocess_cfg']
        return tuple(cfg['mean']), tuple(cfg['std'])

    def _hf_mean_std(self):
        cfg = json.loads((snapshot(self.key) / 'preprocessor_config.json').read_text())
        return tuple(cfg['image_mean']), tuple(cfg['image_std'])

    def _load_hf_siglip(self):
        from transformers import SiglipVisionModel
        outer = SiglipVisionModel.from_pretrained(str(snapshot('S')), use_safetensors=True)
        self.net = outer.vision_model.float().eval().requires_grad_(False)
        assert self.net.config.image_size == 384 and self.net.config.patch_size == 14
        return self._hf_mean_std()

    def _load_hf_convnextv2(self):
        from transformers import ConvNextV2Model
        self.net = ConvNextV2Model.from_pretrained(str(snapshot('X')), use_safetensors=True).float().eval().requires_grad_(False)
        return self._hf_mean_std()

    def _load_hf_vit(self):
        from transformers import ViTModel
        # the checkpoint is ViTForImageClassification: no pooler weights, so do not build (and randomly init) one
        self.net = ViTModel.from_pretrained(str(snapshot('W')), add_pooling_layer=False, use_safetensors=False).float() \
            .eval().requires_grad_(False)
        assert self.net.config.image_size == 224 and self.net.config.patch_size == 16
        return self._hf_mean_std()

    # ------------------------------------------------------------------ geometry
    def input_size(self, size):
        """Pixel size the network actually sees for a (square) working image of `size` pixels."""
        return self.spec['fixed'] or size

    def layout(self, size):
        """[(has_cls, grid_h, grid_w, pixel_stride)] per feature entry for a working image of `size` pixels."""
        n = self.input_size(size)
        if self.key == 'X':
            return [(False, n // s, n // s, s) for s in self.spec['strides']]
        g, p = n // self.spec['patch'], self.spec['patch']
        entries = {'A': 13, 'B': 25, 'D': 25, 'S': 28, 'W': 25}[self.key]
        return [(self.spec['cls'], g, g, p)] * entries

    # ------------------------------------------------------------------ features
    def forward(self, image):
        x = (image - self.mean) / self.std
        if self.spec['fixed'] is not None:
            x = F.interpolate(x, (self.spec['fixed'], self.spec['fixed']), mode='bicubic', align_corners=False)
        features = getattr(self, '_features_' + self.spec['family'])(x)
        return features

    def _features_openai_clip(self, x):
        v = self.net
        x = v.conv1(x)
        b, d, gh, gw = x.shape
        x = x.reshape(b, d, gh * gw).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(b, 1, d, dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        x = x + interpolate_patch_pos(v.positional_embedding, gh, gw).to(x.dtype)
        x = v.ln_pre(x).permute(1, 0, 2)          # LND, the clip package's transformer layout
        features = [x.contiguous()]
        for block in v.transformer.resblocks:
            x = block(x)
            features.append(x.contiguous())
        return features

    def _features_open_clip_hfhub(self, x):
        v = self.net
        x = v.conv1(x)
        b, d, gh, gw = x.shape
        x = x.reshape(b, d, gh * gw).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).reshape(1, 1, -1).expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + interpolate_patch_pos(v.positional_embedding, gh, gw).to(x.dtype)
        x = v.ln_pre(v.patch_dropout(x))          # both Identity for CLIPA in eval (no_ln_pre, no dropout)
        features = [x.permute(1, 0, 2).contiguous()]
        for block in v.transformer.resblocks:     # batch-first blocks (asserted at load)
            x = block(x)
            features.append(x.permute(1, 0, 2).contiguous())
        return features

    def _features_hf_siglip(self, x):
        h = self.net.embeddings(x)
        features = [h.permute(1, 0, 2).contiguous()]
        for layer in self.net.encoder.layers:
            out = layer(h, attention_mask=None, output_attentions=False)
            h = out[0] if isinstance(out, tuple) else out
            features.append(h.permute(1, 0, 2).contiguous())
        return features

    def _features_hf_vit(self, x):
        h = self.net.embeddings(x)
        features = [h.permute(1, 0, 2).contiguous()]
        for layer in self.net.encoder.layer:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
            features.append(h.permute(1, 0, 2).contiguous())
        return features

    def _features_hf_convnextv2(self, x):
        h = self.net.embeddings(x)
        states = [h]
        for stage in self.net.encoder.stages:
            h = stage(h)
            states.append(h)
        return [s.reshape(s.shape[0], s.shape[1], -1).permute(2, 0, 1).contiguous() for s in states]

    def identity(self):
        return dict(key=self.key, name=self.spec['name'], family=self.spec['family'], model=self.spec['model'],
                    files=self.files, fingerprint=self.fingerprint, input='fixed %d' % self.spec['fixed'] if self.spec['fixed'] else
                    'working size, pos-embed interpolated', cls=self.spec['cls'],
                    mean=[round(float(v), 8) for v in self.mean.flatten() / 255],
                    std=[round(float(v), 8) for v in self.std.flatten() / 255])


def load(keys, device):
    assert keys and set(keys) <= set(POOL) and len(set(keys)) == len(keys), keys
    return [Surrogate(k, device) for k in keys]
