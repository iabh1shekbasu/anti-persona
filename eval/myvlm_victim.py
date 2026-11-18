"""The MyVLM victim as used for every MyVLM number in the paper (shared by reactive_myvlm.py and proactive_myvlm.py).

* MyVLM with the LLaVA backbone, VQA personalisation, PERSON concept, face concept head = insightface buffalo_l
  (``victims/myvlm/concept_heads/face_recognition/head_buffalo.py``, the upstream MyVLM head), gate threshold 0.675
  (cosine distance of a face to the concept keys), concept identifier ``sks`` (a neutral name; the person's real name
  never reaches the model: every occurrence in the captions and Q&A is replaced by ``sks``).
* Training (one concept per album, victim seed 42): MyVLM's ``concept_embedding_training/train.py`` run through
  ``victims/myvlm/train_vqa_fixed.py`` (one bug fix in the data loader: the caption instruction is appended to a COPY
  of a photo's Q&A list, so the 9 VQA : 1 caption mix stays fixed). Config: 200 optimisation steps, lr 1.0,
  reg_lambda 10, batch 2 (min with the number of kept photos), save / val interval 25, float32. A training photo in
  which buffalo_l finds no face is dropped (MyVLM's own face rule); an album with no kept photo is UNTRAINABLE.
* Inference: MyVLM's ``inference/run_myvlm_inference.py`` at iteration 199 with the prompt
  ``"Can you see if sks is in this photo? Answer with a single word or phrase."``; decision = MyVLM's probe parser
  (eval/semantic_parser.myvlm_recognised). The gate row of every photo (did buffalo_l fire on it) is recorded.
"""
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MYVLM_DIR = REPO / 'victims' / 'myvlm'
NEUTRAL = 'sks'
SEED = 42
ITERATION = 199
THRESHOLD = 0.675
PROMPT = 'Can you see if {concept} is in this photo? Answer with a single word or phrase.'
RECIPE = dict(optimization_steps=200, learning_rate=1.0, batch_size=2, reg_lambda=10.0, save_interval=25,
              val_interval=25)


def neutralize(text, identity):
    """Every whole-word, case-insensitive occurrence of the identity's name -> 'sks'."""
    return re.sub(r'(?i)\b' + re.escape(identity) + r'\b', NEUTRAL, text)


def _head():
    if str(MYVLM_DIR) not in sys.path:
        sys.path.insert(0, str(MYVLM_DIR))
    from concept_heads.face_recognition.head_buffalo import FaceConceptHead
    return FaceConceptHead()


def build_album(photos, captions, qa, identity, data_root):
    """Copy the training photos into data_root/<identity>/, drop copies without a buffalo_l face, write captions.json
    and additional_llava_vqa_data.json (Q&A keyed by the COPY's absolute path) with the name replaced by 'sks'.
    photos: {name: path}; captions: {name: [caption, ...]}; qa: {name: [[question, answer] x 9]}."""
    album = Path(data_root) / identity
    album.mkdir(parents=True, exist_ok=False)
    head = _head()
    kept, dropped = {}, []
    for name in sorted(photos, key=lambda n: (len(Path(n).stem), Path(n).stem)):
        dst = album / name
        shutil.copyfile(photos[name], dst)
        if head.extract_signal([dst]).get(dst) is None:
            dst.unlink()
            dropped.append(name)
            continue
        kept[name] = [neutralize(c, identity) for c in captions[name]]
    (album / 'captions.json').write_text(json.dumps(kept, indent=4))
    vqa = {str(album / n): [[neutralize(q, identity), neutralize(a, identity)] for q, a in qa[n]] for n in kept}
    (album / 'additional_llava_vqa_data.json').write_text(json.dumps(vqa, indent=4, sort_keys=True))
    return kept, dropped


def training_config(identity, work, batch_size, seed=SEED):
    rows = dict(concept_name=identity, concept_identifier=NEUTRAL, concept_type='PERSON', vlm_type='LLAVA',
                personalization_task='VQA', output_root=str(Path(work) / 'out'), data_root=str(Path(work) / 'data'),
                concept_head_path='models', threshold=THRESHOLD, optimization_steps=RECIPE['optimization_steps'],
                learning_rate=RECIPE['learning_rate'], batch_size=batch_size, reg_lambda=RECIPE['reg_lambda'],
                save_interval=RECIPE['save_interval'], val_interval=RECIPE['val_interval'], seed=seed, device='cuda',
                torch_dtype='float32', face_head='buffalo_l')
    return ''.join(f'{k}: {v}\n' for k, v in rows.items())


def train(identity, photos, captions, qa, work, python=sys.executable, seed=SEED):
    """Train one concept; returns its checkpoint folder (None when UNTRAINABLE)."""
    work = Path(work)
    kept, dropped = build_album(photos, captions, qa, identity, work / 'data')
    record = dict(identity=identity, kept=list(kept), dropped=dropped)
    if not kept:
        (work / 'CONCEPT.json').write_text(json.dumps(dict(record, status='UNTRAINABLE'), indent=1))
        return None
    (work / 'config.yaml').write_text(training_config(identity, work, min(RECIPE['batch_size'], len(kept)), seed))
    with (work / 'train.log').open('w') as log:
        subprocess.run([python, '-u', str(MYVLM_DIR / 'train_vqa_fixed.py'), '--config_path', str(work / 'config.yaml')],
                       cwd=str(MYVLM_DIR), env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    ckpt = work / 'out' / identity / f'seed_{seed}'
    assert (ckpt / 'concept_embeddings_llava_vqa.pt').is_file(), ckpt
    (work / 'CONCEPT.json').write_text(json.dumps(dict(record, status='TRAINED', checkpoint_dir=str(ckpt)), indent=1))
    return ckpt


def infer(identity, checkpoint_dir, image_paths, out, python=sys.executable):
    """MyVLM inference of one concept on a list of photos -> {path: raw answer}."""
    import yaml
    out = Path(out)
    cfg = dict(concept_name=identity, concept_identifier=NEUTRAL, concept_type='PERSON', vlm_type='LLAVA',
               personalization_task='VQA', image_paths=[str(p) for p in image_paths], checkpoint_path=str(checkpoint_dir),
               output_path=str(out / 'myvlm'), concept_head_path='models', seed=SEED, iterations=[ITERATION],
               face_head='buffalo_l', threshold=THRESHOLD, torch_dtype='float32', prompts=[PROMPT])
    (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    with (out / 'inference.log').open('w') as log:
        subprocess.run([python, '-B', str(MYVLM_DIR / 'inference' / 'run_myvlm_inference.py'), '--config_path',
                        str(out / 'config.yaml')], cwd=str(MYVLM_DIR), env=os.environ.copy(), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    result = out / 'myvlm' / identity / 'inference_outputs' / f'seed_{SEED}' / 'inference_outputs_llava_vqa.json'
    block = json.loads(result.read_text())[f'iteration_{ITERATION}']
    prompt = PROMPT.format(concept=NEUTRAL)
    assert set(block) == {str(p) for p in image_paths}, 'inference output does not cover exactly the photos'
    return {p: block[p][prompt] for p in block}


def gate_rows(checkpoint_dir, image_paths):
    """For each photo: min cosine distance of its buffalo_l faces to the concept keys, and whether the gate fires."""
    import torch
    import torch.nn.functional as F
    head = _head()
    signals = head.extract_signal([Path(p) for p in image_paths])
    ck = Path(checkpoint_dir) / 'concept_embeddings_llava_vqa.pt'
    keys = F.normalize(torch.load(ck, map_location='cpu', weights_only=False)[ITERATION]['keys'].float(), dim=-1)
    rows = {}
    for p in image_paths:
        feat = signals[Path(p)]
        d = float((1 - F.normalize(feat.float(), dim=-1) @ keys.T).min()) if feat is not None else None
        rows[str(p)] = dict(distance=d, gate=d is not None and d <= THRESHOLD, no_face=feat is None)
    return rows


def score(identity, checkpoint_dir, photos_by_role, out, method=None):
    """photos_by_role: [(role, path)], role = 'clean' / 'absent' / <arm>. Writes out/records.jsonl.
    method: proactive only, the method whose protected album trained the scored concept."""
    from eval.semantic_parser import myvlm_recognised
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    paths = [str(p) for _, p in photos_by_role]
    assert len(set(paths)) == len(paths), 'duplicate photo paths'
    answers = infer(identity, checkpoint_dir, paths, out)
    gates = gate_rows(checkpoint_dir, paths)
    rows = []
    with (out / 'records.jsonl').open('x') as f:
        for role, p in photos_by_role:
            raw = answers[str(p)]
            row = dict(identity=identity, method=method, role=role, image=str(p), raw=raw,
                       recognised=myvlm_recognised(raw, NEUTRAL), **gates[str(p)])
            f.write(json.dumps(row) + '\n')
            rows.append(row)
    return rows


def load_album_text(captions_json, qa_json):
    """captions.json {name: [caption]} and additional_llava_vqa_data.json {<any path>/<name>: [[q, a] x 9]}."""
    caps = json.loads(Path(captions_json).read_text())
    qa = {Path(k).name: v for k, v in json.loads(Path(qa_json).read_text()).items()}
    missing = sorted(set(caps) - set(qa))
    assert not missing, f'captioned photos without Q&A: {missing}'
    return caps, {n: qa[n] for n in caps}
