"""Reactive MyVLM evaluation for ONE identity (concept identifier "sks", victim seed 42, buffalo_l gate 0.675).

The victim is trained once on the person's CLEAN training photos; it is then asked about the clean test photos, the
protected test photos of every method ("arm") and other people's photos (target-absent control).

  # 1) train the clean victim (skip if you have one)
  python -m eval.reactive_myvlm train --identity thao --train-dir <clean training photos> \
      --captions <captions.json> --qa <additional_llava_vqa_data.json> --work <new folder>

  # 2) score
  python -m eval.reactive_myvlm score --identity thao --checkpoint-dir <work>/out/thao/seed_42 \
      --test-dir <clean test photos> --negative-dir <other people's photos> \
      --arm ours_s0=<dir of protected PNGs> [--arm ...] --out <new folder>

The captions / Q&A files are the per-photo MyVLM training texts (one caption list and nine question-answer pairs per
training photo; the person's name is replaced by "sks" automatically). Clean test photos are scored as 336 x 336 PNGs
prepared like the protection code reads them, so clean and protected photos have the same geometry.
records.jsonl rows: role (clean / absent / <arm>), raw answer, recognised (MyVLM probe parser), gate, distance.
"""
import argparse
from pathlib import Path

from eval.reactive_yollava import parse_arm, photos


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train', help='train the clean victim of one identity')
    t.add_argument('--identity', required=True)
    t.add_argument('--train-dir', type=Path, required=True)
    t.add_argument('--captions', type=Path, required=True)
    t.add_argument('--qa', type=Path, required=True)
    t.add_argument('--work', type=Path, required=True)
    s = sub.add_parser('score', help='score clean / protected / target-absent photos')
    s.add_argument('--identity', required=True)
    s.add_argument('--checkpoint-dir', type=Path, required=True)
    s.add_argument('--test-dir', type=Path, required=True)
    s.add_argument('--negative-dir', type=Path, required=True)
    s.add_argument('--arm', type=parse_arm, action='append', default=[])
    s.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    from eval import myvlm_victim as M
    if a.cmd == 'train':
        caps, qa = M.load_album_text(a.captions, a.qa)
        album = {q.name: q for q in photos(a.train_dir) if q.name in caps}
        print(M.train(a.identity, album, caps, qa, a.work))
        return
    from eval.yollava_victim import prepare_clean
    prep = a.out.parent / (a.out.name + '_prepared')
    prep.mkdir(parents=True, exist_ok=False)
    roles = []
    for q in photos(a.test_dir):
        roles.append(('clean', prepare_clean(q, prep / f'clean_{q.stem}.png')))
        roles += [(name, folder / f'{q.stem}.png') for name, folder in a.arm]
    roles += [('absent', prepare_clean(q, prep / f'absent_{q.stem}.png')) for q in photos(a.negative_dir)]
    rows = M.score(a.identity, a.checkpoint_dir, roles, a.out)
    print(f'{len(rows)} answers written to {a.out / "records.jsonl"}')


if __name__ == '__main__':
    main()
