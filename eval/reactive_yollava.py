"""Reactive Yo'LLaVA evaluation for ONE identity: the victim's concept is trained on CLEAN photos; it is then asked
about the person's test photos, clean and protected by each method ("arm"), and about other people's photos
(target-absent control).

Port of the scoring script behind the paper's reactive Yo'LLaVA tables (same model loading, concept loading, prompt,
greedy decoding, image preparation, JPEG delivery and record format). Every raw answer is stored; decisions are made
afterwards by eval/aggregate.py with the semantic parser.

  python -m eval.reactive_yollava --identity thao --model-path <llava-v1.5-13b> \
      --checkpoint-dir <concept dir holding 14-token.pt / 14-lmhead.pt> --epoch 14 \
      --test-dir <clean test photos of the person> --negative-dir <other people's photos> \
      --arm ours_s0=<dir of protected PNGs> --arm ours_s202=<dir> ... --qualities 0 85 --out <new folder>

A protected arm folder holds one ``<stem>.png`` (336 x 336) per clean test photo ``<stem>.*``. Quality 0 = the PNG;
85 = JPEG quality 85 of it. Writes <out>/predictions.jsonl and <out>/summary.json.

Paper victims: paper-10 = the Yo'LLaVA authors' released concept checkpoints (yuheng: ours, seed 42, epoch 14),
MMPB-15 = our concepts (seed 42, epoch 14). See eval/README.md.
"""
import argparse
import json
from pathlib import Path

EXTS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')


def photos(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS)


def parse_arm(text):
    name, _, folder = text.partition('=')
    if not name or not folder:
        raise argparse.ArgumentTypeError('--arm takes NAME=FOLDER')
    return name, Path(folder)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--identity', required=True, help='concept name the checkpoint was trained with (<identity>)')
    p.add_argument('--model-path', required=True, help='LLaVA-1.5-13B folder')
    p.add_argument('--checkpoint-dir', type=Path, required=True)
    p.add_argument('--epoch', default='14')
    p.add_argument('--test-dir', type=Path, required=True)
    p.add_argument('--negative-dir', type=Path, required=True)
    p.add_argument('--arm', type=parse_arm, action='append', default=[], help='NAME=FOLDER of protected PNGs')
    p.add_argument('--qualities', type=int, nargs='+', default=[0, 85], help='0 = PNG, else JPEG quality')
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if any(q != 0 and not 1 <= q <= 100 for q in a.qualities):
        p.error('JPEG quality must be 1..100 or 0 for PNG')

    from eval.yollava_victim import YoLLaVA, jpeg_copy, prepare_clean, write_jsonl
    from eval.semantic_parser import semantic_answer
    a.out.mkdir(parents=True, exist_ok=False)
    prepared = a.out / 'prepared'
    prepared.mkdir()
    examples = []
    for path in photos(a.test_dir):
        examples.append(('clean', path.stem, 'yes', prepare_clean(path, prepared / f'clean_{path.stem}.png')))
        for name, folder in a.arm:
            protected = folder / f'{path.stem}.png'
            if not protected.is_file():
                raise FileNotFoundError(protected)
            examples.append((name, path.stem, 'yes', protected))
    for path in photos(a.negative_dir):
        examples.append(('target_absent', path.stem, 'no', prepare_clean(path, prepared / f'target_absent_{path.stem}.png')))

    victim = YoLLaVA(a.model_path, a.identity)
    hashes = victim.load_concept(a.checkpoint_dir, a.epoch)
    rows = []
    for arm, query, label, path in examples:
        for quality in a.qualities:
            row = dict(identity=a.identity, arm=arm, query=query, label=label, quality=quality, image=str(path),
                       raw=None, semantic=None, error=None)
            try:
                delivered = path if not quality else jpeg_copy(path, a.out / f'{arm}_{query}_q{quality}.jpg', quality)
                row['raw'] = victim.ask(delivered)
                row['semantic'] = semantic_answer(row['raw'])
            except Exception as error:      # kept like the original: an error is never evidence of protection
                row['error'] = repr(error)
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_jsonl(a.out / 'predictions.jsonl', rows)
    (a.out / 'summary.json').write_text(json.dumps(dict(
        identity=a.identity, checkpoint=str(a.checkpoint_dir), epoch=a.epoch, checkpoint_sha256=hashes,
        question=victim.question, decoder='greedy, temperature 0, 32 new tokens', seed=42,
        arms=[n for n, _ in a.arm], qualities=a.qualities, errors=sum(r['error'] is not None for r in rows)), indent=1))
    if any(r['error'] for r in rows):
        raise RuntimeError('evaluation had runtime errors; see predictions.jsonl')


if __name__ == '__main__':
    main()
