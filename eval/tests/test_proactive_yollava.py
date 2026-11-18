"""CPU test of proactive Yo'LLaVA: the model-seed aggregation in eval/aggregate.py, the CLI of eval/proactive_yollava.py
(arguments, training command, input checks) and the concept-checkpoint checks of eval/yollava_victim.py.

    python -m pytest eval/tests/test_proactive_yollava.py      or      python eval/tests/test_proactive_yollava.py

fixtures/proactive_yollava_modelseeds.json holds, per (model seed, dataset, person, concept), how many of the person's
test photos were not recognised, counted from the recorded answers of the paper's scoring jobs (people anonymised), and
a few real answer rows. The aggregation must reproduce the paper's cells from it.
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from eval import aggregate as A  # noqa: E402
from eval import proactive_yollava as PY  # noqa: E402
from eval.semantic_parser import semantic_answer  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / 'fixtures' / 'proactive_yollava_modelseeds.json'


def _write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def _row(identity, concept, seed, kind, raw, name):
    return dict(identity=identity, concept=concept, model_seed=seed, set=kind, label='no' if kind == 'negative' else 'yes',
                name=name, image=None, raw=raw, semantic=semantic_answer(raw), error=None)


def _rows_from_cells(cells):
    """Per-photo answer rows that reproduce the recorded counts, plus 'No' controls that must NOT be counted."""
    out = {}
    for seed, dataset, identity, concept, no, n in cells:
        rows = out.setdefault((dataset, seed, identity), [])
        rows += [_row(identity, concept, seed, 'test', 'No' if k < no else 'Yes', f't{k}') for k in range(n)]
        rows += [_row(identity, concept, seed, 'negative', 'No', 'n0'), _row(identity, concept, seed, 'train', 'No', 'a0')]
    return out


def _counts(rows_by_key, tmp):
    for (dataset, seed, identity), rows in rows_by_key.items():
        _write(Path(tmp) / dataset / f'c{seed}' / identity / 'predictions.jsonl', rows)
    datasets = sorted({k[0] for k in rows_by_key})
    return A.collect([(d, Path(tmp) / d) for d in datasets], 0)


def test_paper_cells_from_recorded_counts():
    data = json.loads(FIXTURE.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        counts = _counts(_rows_from_cells(data['cells']), tmp)
    text, result = A.table(counts, per_model_seed=True)
    assert len(data['expected']) == 6
    for key, (mean, std) in data['expected'].items():
        method, group = key.split(' | ')
        got = result[(method, group)]
        assert (round(got[0], 1), round(got[1], 1)) == (mean, std), (key, got)
    assert '| Method | mmpb15 | paper10 | both |' in text, text
    assert '| ours | 92.5 ± 4.6 | 90.1 ± 4.6 | 91.2 ± 4.4 |' in text, text
    assert 'Model seed 46' in text
    # the headline of the docs, recomputed here: Ours 91.2 ± 4.4, Clean 29.1 ± 7.8 over model seeds 42-46
    assert (round(result[('ours', 'both')][0], 1), round(result[('clean', 'both')][0], 1)) == (91.2, 29.1)


def test_recorded_rows():
    rows = json.loads(FIXTURE.read_text())['rows']
    assert {r['set'] for r in rows} == {'test', 'negative', 'train'}
    for r in rows:
        assert semantic_answer(r['raw']) == r['semantic'], r
    with tempfile.TemporaryDirectory() as tmp:
        got = list(A.decisions(_write(Path(tmp) / 'predictions.jsonl', rows)))
    want = [(r['identity'], r['concept'], r['semantic'] == 'no', r['model_seed']) for r in rows if r['set'] == 'test']
    assert got == want, got


def test_model_seed_math():
    # seed 42: ours_s0 8/10, ours_s202 9/10 -> 85.0; seed 43: 10/10, 9/10 -> 95.0; mean 90.0, population std 5.0
    cells = [(42, 'd', 'x', 'ours_s0', 8, 10), (42, 'd', 'x', 'ours_s202', 9, 10),
             (43, 'd', 'x', 'ours_s0', 10, 10), (43, 'd', 'x', 'ours_s202', 9, 10),
             (42, 'd', 'x', 'clean', 2, 10), (43, 'd', 'x', 'clean', 4, 10)]
    with tempfile.TemporaryDirectory() as tmp:
        counts = _counts(_rows_from_cells(cells), tmp)
    r = A.across_model_seeds(counts, 'ours', ['d'])
    assert (r['mean'], r['std'], r['seeds'], r['per_model_seed']) == (90.0, 5.0, [42, 43], {42: 85.0, 43: 95.0}), r
    assert A.across_model_seeds(counts, 'clean', ['d'])['mean'] == 30.0
    # one model seed: the plain image-seed table (mean ± |s0 - s202| / 2)
    one = {k: v for k, v in counts.items() if k[4] == 42}
    text, result = A.table(one)
    assert result[('ours', 'd')] == (85.0, 5.0) and 'model seeds' not in text, text


def test_incomplete_model_seed_is_left_out():
    cells = [(s, 'd', i, c, 5, 10) for s in (42, 43, 44) for i in ('x', 'y') for c in ('ours_s0', 'ours_s202')]
    cells = [c for c in cells if not (c[0] == 43 and c[2] == 'y')] + [(43, 'd', 'x', 'clean', 1, 10)]
    with tempfile.TemporaryDirectory() as tmp:
        counts = _counts(_rows_from_cells(cells), tmp)
    r = A.across_model_seeds(counts, 'ours', ['d'])
    assert r['seeds'] == [42, 44], r
    assert '(2 of 3 model seeds)' in A.table(counts)[0]


def test_mixed_files_refused():
    counts = {('d', 'x', 'ours', 0, None): [1, 2], ('d', 'x', 'ours', 0, 42): [1, 2]}
    try:
        A.model_seeds(counts)
    except ValueError:
        return
    raise AssertionError('mixing answer files with and without a model seed must fail')


# reactive Yo'LLaVA + MyVLM rows; EXPECTED_REACTIVE is the output of eval/aggregate.py before model seeds were added
def _reactive_inputs(tmp):
    answers = ('No', 'Yes', '<sks> is not in this photo.', 'I can see <sks> on the left.', 'Not sure.')
    for d, ids in (('paper10', ('bo', 'thao')), ('mmpb15', ('m1', 'm2', 'm3'))):
        rows, k = [], 0
        for i in ids:
            for arm in ('clean', 'ours_s0', 'ours_s202', 'ifgsm', 'pgd_s0', 'pgd_s202', 'target_absent'):
                for q in range(4):
                    for quality in (0, 85):
                        k += 1
                        rows.append(dict(identity=i, arm=arm, query=f'q{q}', label='no' if arm == 'target_absent' else 'yes',
                                         quality=quality, image=None, raw=answers[(k * 7 + len(i)) % 5], semantic=None,
                                         error=None))
        _write(Path(tmp) / d / 'predictions.jsonl', rows)
    return [(d, Path(tmp) / d) for d in ('paper10', 'mmpb15')]


EXPECTED_REACTIVE = {
    0: '| Method | mmpb15 | paper10 | both |\n|---|---:|---:|---:|\n'
       '| clean | 33.3 ± 0.0 | 37.5 ± 0.0 | 35.0 ± 0.0 |\n'
       '| ifgsm | 33.3 ± 0.0 | 37.5 ± 0.0 | 35.0 ± 0.0 |\n'
       '| ours | 41.7 ± 0.0 | 37.5 ± 0.0 | 40.0 ± 0.0 |\n'
       '| pgd | 41.7 ± 8.3 | 43.8 ± 6.2 | 42.5 ± 7.5 |\n\n'
       '| Identity | clean | ifgsm | ours | pgd |\n|---|---:|---:|---:|---:|\n'
       '| mmpb15/m1 | 25.0 | 50.0 | 37.5 | 37.5 |\n'
       '| mmpb15/m2 | 25.0 | 25.0 | 50.0 | 37.5 |\n'
       '| mmpb15/m3 | 50.0 | 25.0 | 37.5 | 50.0 |\n'
       '| paper10/bo | 25.0 | 50.0 | 37.5 | 37.5 |\n'
       '| paper10/thao | 50.0 | 25.0 | 37.5 | 50.0 |',
    85: '| Method | mmpb15 | paper10 | both |\n|---|---:|---:|---:|\n'
        '| clean | 41.7 ± 0.0 | 37.5 ± 0.0 | 40.0 ± 0.0 |\n'
        '| ifgsm | 33.3 ± 0.0 | 37.5 ± 0.0 | 35.0 ± 0.0 |\n'
        '| ours | 41.7 ± 8.3 | 43.8 ± 6.2 | 42.5 ± 7.5 |\n'
        '| pgd | 41.7 ± 0.0 | 37.5 ± 0.0 | 40.0 ± 0.0 |\n\n'
        '| Identity | clean | ifgsm | ours | pgd |\n|---|---:|---:|---:|---:|\n'
        '| mmpb15/m1 | 25.0 | 25.0 | 50.0 | 37.5 |\n'
        '| mmpb15/m2 | 50.0 | 25.0 | 37.5 | 50.0 |\n'
        '| mmpb15/m3 | 50.0 | 50.0 | 37.5 | 37.5 |\n'
        '| paper10/bo | 25.0 | 25.0 | 50.0 | 37.5 |\n'
        '| paper10/thao | 50.0 | 50.0 | 37.5 | 37.5 |',
}


def test_reactive_output_unchanged():
    with tempfile.TemporaryDirectory() as tmp:
        inputs = _reactive_inputs(tmp)
        for quality, want in EXPECTED_REACTIVE.items():
            counts = A.collect(inputs, quality)
            assert {k[4] for k in counts} == {None}
            assert A.table(counts, per_person=True, per_model_seed=True)[0] == want, quality


# --- CLI -------------------------------------------------------------------------------------------------------------

def test_train_command_is_the_paper_recipe():
    cmd = PY.train_command('thao', '/m/llava-v1.5-13b', '/a/album', '/j', '/w', 'ours_s0', 44, python='py')
    assert cmd == ['py', '-u', str(REPO / 'victims' / 'yollava' / 'personalize.py'), '--json_path', '/j',
                   '--data_root', '/a/album', '--model_path', '/m/llava-v1.5-13b', '--sks_name', 'thao',
                   '--checkpoint_path', '/w/checkpoints', '--exp_name', 'ours_s0-c44', '--tensorboard_path', '/w/runs',
                   '--prefix_token', '16', '--epoch', '15', '--seed', '44', '--user_prompt', '--recog_only',
                   '--random_image', '--text_only'], cmd
    assert PY.concept_dir('/w', 'thao', 'ours_s0', 44) == Path('/w/checkpoints/thao/ours_s0-c44')


def _fails(argv):
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            PY.parser().parse_args(argv)
        except SystemExit:
            return True
    return False


def test_arguments():
    p = PY.parser()
    a = p.parse_args(['score', '--identity', 'thao', '--model-path', 'm', '--model-seed', '45', '--concept', 'clean=/c',
                      '--concept', 'ours_s0=/o', '--test-dir', 't', '--negative-dir', 'n', '--train-album', 'ours_s0=/al',
                      '--out', 'o'])
    assert a.concept == [('clean', Path('/c')), ('ours_s0', Path('/o'))] and a.epoch == '14' and a.model_seed == 45
    assert a.train_album == [('ours_s0', Path('/al'))]
    t = p.parse_args(['train', '--identity', 'thao', '--model-path', 'm', '--json-path', 'j', '--protected-album', 'a',
                      '--method', 'clean', '--work', 'w'])
    assert t.model_seed == 42 and not t.dry_run
    base = ['score', '--identity', 'thao', '--model-path', 'm', '--test-dir', 't', '--negative-dir', 'n', '--out', 'o']
    assert _fails(base + ['--concept', 'clean=/c'])                              # --model-seed is required
    assert _fails(base + ['--concept', 'clean=/c', '--model-seed', '41'])        # only the paper's model seeds
    assert _fails(base + ['--concept', 'clean', '--model-seed', '42'])           # NAME=DIR
    assert _fails(base[:2] + ['th<ao>'] + base[3:] + ['--concept', 'c=/c', '--model-seed', '42'])


def _training_inputs(tmp, album_names=('0.png', '1.png'), drop_negative=False):
    tmp = Path(tmp)
    album, js = tmp / 'album', tmp / 'json' / 'thao'
    album.mkdir()
    js.mkdir(parents=True)
    for n in album_names:
        (album / n).write_bytes(b'png')
    neg = tmp / 'neg.png'
    if not drop_negative:
        neg.write_bytes(b'png')
    conv = [{'Human': 'Can you see <thao> in this photo?', 'AI': 'Yes'}]
    files = {'positive_recognition': {'/elsewhere/thao/0.png': conv, '/elsewhere/thao/1.png': conv * 2},
             'negative_recognition-laion': {str(neg): conv},
             'negative_recognition-random-imgs': {str(neg): conv},
             'text-only-conversation': {str(album / '0.png'): conv}}
    for k, v in files.items():
        (js / f'{k}.json').write_text(json.dumps(v))
    return album, tmp / 'json'


def _raises(fn, *args):
    try:
        fn(*args)
    except (FileNotFoundError, ValueError):
        return True
    return False


def test_training_input_checks():
    with tempfile.TemporaryDirectory() as tmp:
        album, js = _training_inputs(tmp)
        counts = PY.check_training_inputs('thao', album, js)
        assert counts['album_photos'] == 2 and counts['positive_recognition'] == 3, counts
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            PY.main(['train', '--identity', 'thao', '--model-path', 'm', '--json-path', str(js), '--protected-album',
                     str(album), '--method', 'ours_s0', '--model-seed', '43', '--work', str(Path(tmp) / 'w'), '--dry-run'])
        plan = json.loads(out.getvalue())
        assert plan['command'][plan['command'].index('--seed') + 1] == '43'
        assert plan['checkpoint_dir'].endswith('checkpoints/thao/ours_s0-c43')
    with tempfile.TemporaryDirectory() as tmp:                     # album photo names differ from the JSON
        album, js = _training_inputs(tmp, album_names=('0.png', '2.png'))
        assert _raises(PY.check_training_inputs, 'thao', album, js)
    with tempfile.TemporaryDirectory() as tmp:                     # a negative image the trainer would not find
        album, js = _training_inputs(tmp, drop_negative=True)
        assert _raises(PY.check_training_inputs, 'thao', album, js)


def test_concept_checkpoint_checks():
    try:
        import torch
    except ImportError:
        print('torch not installed: concept checkpoint checks skipped')
        return
    from eval.yollava_victim import concept_files, read_concept
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        assert _raises(concept_files, d, '14')
        torch.save(torch.zeros(17, 8), d / '14-token.pt')
        assert _raises(concept_files, d, '14')                    # lmhead missing
        torch.save(torch.ones(17, 8), d / '14-lmhead.pt')
        token, lmhead = read_concept(d, '14', hidden=8)
        assert token.shape == lmhead.shape == (17, 8)
        assert _raises(read_concept, d, '14')                     # 13B width is 5120
        torch.save(torch.zeros(16, 8), d / '14-token.pt')
        assert _raises(read_concept, d, '14', 8)                  # 16 rows: would broadcast silently


TESTS = [test_paper_cells_from_recorded_counts, test_recorded_rows, test_model_seed_math,
         test_incomplete_model_seed_is_left_out, test_mixed_files_refused, test_reactive_output_unchanged,
         test_train_command_is_the_paper_recipe, test_arguments, test_training_input_checks,
         test_concept_checkpoint_checks]

if __name__ == '__main__':
    for t in TESTS:
        t()
    print(f"proactive Yo'LLaVA OK: {len(TESTS)} tests")
