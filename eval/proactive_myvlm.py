"""Proactive MyVLM evaluation: one MyVLM concept ("sks", victim seed 42) is trained on the person's PROTECTED training
album per method, then asked about the person's CLEAN test photos and about other people's photos.

  python -m eval.proactive_myvlm train --identity thao --protected-album <folder of protected training PNGs> \
      --captions <captions.json> --qa <additional_llava_vqa_data.json> --work <new folder>

  python -m eval.proactive_myvlm score --identity thao --concept ours_s0=<work>/out/thao/seed_42 [--concept ...] \
      --test-dir <clean test photos> --negative-dir <other people's photos> --out <new folder>

The protected album holds one PNG per training photo, named like the captioned training photo it replaces (the
captions and Q&A of the clean photo are reused for its protected copy, as in the paper). Training photos in which
MyVLM's buffalo_l head finds no face are dropped by MyVLM's own rule; an album left with none is UNTRAINABLE and every
test photo then counts as not recognised for that concept (no concept can be injected). Recorded per concept in
<out>/<concept>/records.jsonl.
"""
import argparse
import json
from pathlib import Path

from eval.reactive_yollava import photos


def parse_concept(text):
    name, _, folder = text.partition('=')
    if not name or not folder:
        raise argparse.ArgumentTypeError('--concept takes NAME=CHECKPOINT_DIR')
    return name, Path(folder)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train', help='train one concept on a protected album')
    t.add_argument('--identity', required=True)
    t.add_argument('--protected-album', type=Path, required=True)
    t.add_argument('--captions', type=Path, required=True)
    t.add_argument('--qa', type=Path, required=True)
    t.add_argument('--work', type=Path, required=True)
    s = sub.add_parser('score', help='score trained concepts on clean test photos')
    s.add_argument('--identity', required=True)
    s.add_argument('--concept', type=parse_concept, action='append', required=True)
    s.add_argument('--test-dir', type=Path, required=True)
    s.add_argument('--negative-dir', type=Path, required=True)
    s.add_argument('--out', type=Path, required=True)
    a = p.parse_args()

    from eval import myvlm_victim as M
    if a.cmd == 'train':
        caps, qa = M.load_album_text(a.captions, a.qa)
        album = {}
        for name in caps:
            protected = a.protected_album / (Path(name).stem + '.png')
            if not protected.is_file():
                raise FileNotFoundError(protected)
            album[name] = protected
        print(json.dumps(dict(checkpoint=str(M.train(a.identity, album, caps, qa, a.work)))))
        return
    from eval.yollava_victim import prepare_clean
    a.out.mkdir(parents=True, exist_ok=False)
    prep = a.out / 'prepared'
    prep.mkdir()
    roles = [('clean', prepare_clean(q, prep / f'clean_{q.stem}.png')) for q in photos(a.test_dir)]
    roles += [('absent', prepare_clean(q, prep / f'absent_{q.stem}.png')) for q in photos(a.negative_dir)]
    for name, ckpt in a.concept:
        rows = M.score(a.identity, ckpt, roles, a.out / name, method=name)
        print(name, sum(not r['recognised'] for r in rows if r['role'] == 'clean'), 'of',
              sum(r['role'] == 'clean' for r in rows), 'clean test photos not recognised')


if __name__ == '__main__':
    main()
