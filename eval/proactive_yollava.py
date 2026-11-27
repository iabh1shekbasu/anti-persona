"""Proactive Yo'LLaVA evaluation: a Yo'LLaVA concept is trained on the person's PROTECTED training album (one concept
per method and victim model seed), then asked about the person's CLEAN test photos and about other people's photos
(target-absent control). The Clean row is a concept trained on the unprotected album, squashed to 336 x 336 like the
protection code reads it.

  # 0) the Clean album (training photos -> 336 x 336 bicubic squash -> rounded -> <stem>.png)
  python -m eval.proactive_yollava clean-album --train-dir <clean training photos> --out <new folder>

  # 1) train one concept on one album at one victim model seed (one GPU; 15 epochs, the epoch-14 files are scored)
  python -m eval.proactive_yollava train --identity thao --model-path <llava-v1.5-13b> \
      --json-path <training JSON root> --protected-album <album of 336 x 336 PNGs> \
      --method ours_s0 --model-seed 42 --work <folder>
      -> <work>/checkpoints/thao/ours_s0-c42/14-token.pt, 14-lmhead.pt

  # 2) score every concept of one identity and model seed (one model load; the paper scored clean first)
  python -m eval.proactive_yollava score --identity thao --model-path <llava-v1.5-13b> --model-seed 42 \
      --concept clean=<dir> --concept ours_s0=<dir> --concept ours_s202=<dir> \
      --test-dir <clean test photos> --negative-dir <other people's photos> \
      [--train-album ours_s0=<the album it was trained on>] --out <new folder>

  # 3) the table: % of all test photos not recognised, mean ± std over model seeds
  python -m eval.aggregate --input paper10=<score outputs> --input mmpb15=<score outputs> [--per-model-seed]

Training = victims/yollava/personalize.py (the Yo'LLaVA trainer) with the paper's recipe (RECIPE_FLAGS, --seed = the
victim model seed). It reads four conversation files per person from <json-path>/<identity>/ (positive_recognition,
negative_recognition-laion, negative_recognition-random-imgs, text-only-conversation). The images of
positive_recognition are taken from the album by file name (the trainer's redirect), so the album must hold exactly
those file names; every other image is opened at the path written in the JSON (relative paths: from the folder the
command runs in). Both are checked before training.

A checkpoint folder holds <epoch>-token.pt and <epoch>-lmhead.pt, each [17, 5120]: the rows of <identity>,
<token0>..<token15>. Released concept checkpoints have the same layout and are scored the same way.

score writes <out>/predictions.jsonl, one row per (concept, photo): identity, concept, model_seed, set (test / negative /
train), label (yes = the person is in the photo), name, image, raw answer, semantic answer, error; and <out>/summary.json.
Only set == test rows enter the table. The negative and training-album rows are controls (a concept that says no to its
own training photos did not learn the person).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from eval.reactive_yollava import photos

MODEL_SEEDS = (42, 43, 44, 45, 46)      # victim model seeds of the paper's proactive Yo'LLaVA table
EPOCH = '14'                            # the checkpoint that is scored (15 epochs are trained: 0..14)
RECIPE_FLAGS = ('--prefix_token', '16', '--epoch', '15', '--user_prompt', '--recog_only', '--random_image',
                '--text_only')
JSON_TYPES = ('positive_recognition', 'negative_recognition-laion', 'negative_recognition-random-imgs',
              'text-only-conversation')
NAME_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.+-]*$')


def _name(text):
    if not NAME_RE.match(text):
        raise argparse.ArgumentTypeError(f'{text!r}: use letters, digits and _ . + - only')
    return text


def name_dir(text):
    name, _, folder = text.partition('=')
    if not name or not folder:
        raise argparse.ArgumentTypeError('takes NAME=FOLDER')
    return _name(name), Path(folder)


def concept_dir(work, identity, method, model_seed):
    """Where personalize.py writes the concept: <checkpoint_path>/<sks_name>/<exp_name>."""
    return Path(work) / 'checkpoints' / identity / f'{method}-c{model_seed}'


def train_command(identity, model_path, album, json_path, work, method, model_seed, python=sys.executable):
    """The paper's training command for one concept (only --data_root, --exp_name, --seed and the paths vary)."""
    from eval.yollava_victim import YOLLAVA_DIR
    return [python, '-u', str(YOLLAVA_DIR / 'personalize.py'), '--json_path', str(json_path),
            '--data_root', str(album), '--model_path', str(model_path), '--sks_name', identity,
            '--checkpoint_path', str(Path(work) / 'checkpoints'), '--exp_name', f'{method}-c{model_seed}',
            '--tensorboard_path', str(Path(work) / 'runs'), *RECIPE_FLAGS[:4], '--seed', str(model_seed),
            *RECIPE_FLAGS[4:]]


def check_training_inputs(identity, album, json_path):
    """Raise unless the trainer will find every file it opens; returns counts (photos, conversations per file)."""
    folder = Path(json_path) / identity
    problems, counts = [], {}
    data = {}
    for kind in JSON_TYPES:
        f = folder / f'{kind}.json'
        if not f.is_file():
            problems.append(f'missing {f}')
            continue
        data[kind] = json.loads(f.read_text())
        counts[kind] = sum(len(v) for v in data[kind].values())
    if 'positive_recognition' in data:
        wanted = sorted({os.path.basename(k) for k in data['positive_recognition']})
        have = sorted(p.name for p in photos(album)) if Path(album).is_dir() else []
        if wanted != have:
            problems.append(f'album {album} must hold exactly the photos named in positive_recognition.json: '
                            f'missing {sorted(set(wanted) - set(have))}, extra {sorted(set(have) - set(wanted))}')
        counts['album_photos'] = len(have)
    for kind in JSON_TYPES[1:]:
        lost = [k for k in data.get(kind, {}) if not Path(k).is_file()]
        if lost:
            problems.append(f'{kind}.json: {len(lost)} image(s) not found, e.g. {lost[0]}')
    if problems:
        raise FileNotFoundError('training inputs:\n  ' + '\n  '.join(problems))
    return counts


def clean_album(train_dir, out):
    from eval.yollava_victim import prepare_clean
    src = photos(train_dir)
    stems = [p.stem for p in src]
    if not src or len(set(stems)) != len(stems):
        raise ValueError(f'{train_dir}: no photos, or two photos share a file stem')
    Path(out).mkdir(parents=True, exist_ok=False)
    return [prepare_clean(p, Path(out) / f'{p.stem}.png') for p in src]


def train(a):
    def full(p):
        return Path(os.path.abspath(p)) if Path(p).exists() else p  # keep symlink names: the model folder name picks the chat template
    album, json_path, work, model = full(a.protected_album), full(a.json_path), Path(a.work).resolve(), full(a.model_path)
    out = concept_dir(work, a.identity, a.method, a.model_seed)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'output folder is not empty (never reuse one): {out}')
    counts = check_training_inputs(a.identity, album, json_path)
    cmd = train_command(a.identity, model, album, json_path, work, a.method, a.model_seed)
    print(json.dumps(dict(command=cmd, checkpoint_dir=str(out), inputs=counts)), flush=True)
    if a.dry_run:
        return
    from eval.yollava_victim import YOLLAVA_DIR, concept_files
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join([str(YOLLAVA_DIR)] + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else []))
    subprocess.run(cmd, env=env, check=True)
    files = concept_files(out, EPOCH)
    print(json.dumps(dict(checkpoint_dir=str(out), epoch=EPOCH, files=[str(f) for f in files.values()])))


def score(a):
    from eval.semantic_parser import semantic_answer
    from eval.yollava_victim import YoLLaVA, prepare_clean, read_concept, write_jsonl
    names = [n for n, _ in a.concept]
    if len(set(names)) != len(names):
        raise ValueError(f'concept names must be unique: {names}')
    albums = dict(a.train_album)
    if set(albums) - set(names):
        raise ValueError(f'--train-album for unknown concept(s): {sorted(set(albums) - set(names))}')
    for _, ckpt in a.concept:                          # every checkpoint is checked before the model is loaded
        read_concept(ckpt, a.epoch)
    a.out.mkdir(parents=True, exist_ok=False)
    delivered = {}
    for kind, folder in (('test', a.test_dir), ('negative', a.negative_dir)):
        (a.out / 'prepared' / kind).mkdir(parents=True)
        delivered[kind] = [(q.stem, prepare_clean(q, a.out / 'prepared' / kind / f'{q.stem}.png')) for q in photos(folder)]

    victim = YoLLaVA(a.model_path, a.identity)
    rows, concepts = [], {}
    for name, ckpt in a.concept:
        concepts[name] = dict(checkpoint=str(ckpt), checkpoint_sha256=victim.load_concept(ckpt, a.epoch),
                              train_album=str(albums[name]) if name in albums else None)
        work = [('test', 'yes', stem, image) for stem, image in delivered['test']]
        work += [('negative', 'no', stem, image) for stem, image in delivered['negative']]
        work += [('train', 'yes', p.name, p) for p in (photos(albums[name]) if name in albums else [])]
        for kind, label, photo, image in work:
            row = dict(identity=a.identity, concept=name, model_seed=a.model_seed, set=kind, label=label, name=photo,
                       image=str(image), raw=None, semantic=None, error=None)
            try:
                row['raw'] = victim.ask(image)
                row['semantic'] = semantic_answer(row['raw'])
            except Exception as error:      # kept like the original: an error is never evidence of protection
                row['error'] = repr(error)
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_jsonl(a.out / 'predictions.jsonl', rows)
    test = {n: [r for r in rows if r['concept'] == n and r['set'] == 'test'] for n in names}
    (a.out / 'summary.json').write_text(json.dumps(dict(
        identity=a.identity, model_seed=a.model_seed, epoch=a.epoch, concepts=concepts, question=victim.question,
        decoder='greedy, temperature 0, 32 new tokens', seed=42, test_photos=len(delivered['test']),
        negative_photos=len(delivered['negative']),
        not_recognised={n: sum(r['semantic'] == 'no' for r in t) for n, t in test.items()},
        errors=sum(r['error'] is not None for r in rows)), indent=1))
    for n, t in test.items():
        print(n, sum(r['semantic'] == 'no' for r in t), 'of', len(t), 'clean test photos not recognised')
    if any(r['error'] for r in rows):
        raise RuntimeError('evaluation had runtime errors; see predictions.jsonl')


def parser():
    p = argparse.ArgumentParser(prog='python -m eval.proactive_yollava', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('clean-album', help='the Clean album: training photos squashed to 336 x 336 PNGs')
    c.add_argument('--train-dir', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    t = sub.add_parser('train', help='train one concept on one (protected) album at one victim model seed')
    t.add_argument('--identity', type=_name, required=True, help='concept name (<identity>), = the JSON folder name')
    t.add_argument('--model-path', required=True, help='LLaVA-1.5-13B folder')
    t.add_argument('--json-path', type=Path, required=True, help='folder holding <identity>/<conversation files>.json')
    t.add_argument('--protected-album', type=Path, required=True, help='the training album (protected, or clean-album)')
    t.add_argument('--method', type=_name, required=True, help='concept name in the tables, e.g. clean, ours_s0')
    t.add_argument('--model-seed', type=int, choices=MODEL_SEEDS, default=42)
    t.add_argument('--work', type=Path, required=True, help='checkpoints go to <work>/checkpoints/<identity>/')
    t.add_argument('--dry-run', action='store_true', help='check the inputs and print the command only')
    s = sub.add_parser('score', help='score concepts of one identity and model seed on the clean test photos')
    s.add_argument('--identity', type=_name, required=True, help='concept name the checkpoints were trained with')
    s.add_argument('--model-path', required=True, help='LLaVA-1.5-13B folder')
    s.add_argument('--concept', type=name_dir, action='append', required=True,
                   help='NAME=CHECKPOINT_DIR (holds <epoch>-token.pt and <epoch>-lmhead.pt); repeat')
    s.add_argument('--model-seed', type=int, choices=MODEL_SEEDS, required=True,
                   help='the model seed the concepts were trained with (recorded in every row)')
    s.add_argument('--epoch', default=EPOCH)
    s.add_argument('--test-dir', type=Path, required=True)
    s.add_argument('--negative-dir', type=Path, required=True)
    s.add_argument('--train-album', type=name_dir, action='append', default=[],
                   help='NAME=ALBUM: also ask concept NAME about the album it was trained on (control rows)')
    s.add_argument('--out', type=Path, required=True)
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.cmd == 'clean-album':
        print(f'{len(clean_album(a.train_dir, a.out))} photos written to {a.out}')
    elif a.cmd == 'train':
        train(a)
    else:
        score(a)


if __name__ == '__main__':
    main()
