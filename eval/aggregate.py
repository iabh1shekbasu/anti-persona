"""Turn the answer files of eval/*.py into the paper's numbers.

Metric (every table): % of ALL test photos the victim does NOT recognise (higher = better protection). No
conditioning on the clean answer; the Clean row is the same number for unprotected photos.
  Yo'LLaVA: not recognised = semantic parser says 'no'. An invalid answer counts as recognised (not protected).
  MyVLM:    not recognised = MyVLM's probe parser does not detect a recognition.
Seeds: every method is run with image seeds 0 and 202. A method's number for a group of photos (one dataset, or both
datasets together) is computed per seed by POOLING the photos of that group, then reported as
    mean over the two seeds  ±  |seed 0 - seed 202| / 2.
A deterministic method (one run, e.g. I-FGSM, TI-MI-FGSM, Clean) counts the same run for both seeds (± 0.0). Per-person
cells pool the photos of both seeds.
Victim model seeds (proactive Yo'LLaVA, eval/proactive_yollava.py; every row carries its model_seed): the number above
is computed per model seed, and a cell reports
    mean over the model seeds  ±  population std over the model seeds
of those per-model-seed numbers. A model seed enters a cell only if it has both image seeds and as many photos as the
fullest model seed (otherwise the cell says how many seeds it used). --per-model-seed adds one table per model seed.
Per-person cells then pool the photos of every image and model seed. Answer files without a model seed (reactive
Yo'LLaVA, MyVLM) are aggregated exactly as before.

Method and seed are read from the arm / concept name: "<method>_s<seed>" (seed 0 or 202), anything else is
deterministic.

  python -m eval.aggregate --input paper10=<folder or file> --input mmpb15=<folder or file>
      [--quality 0] [--per-person] [--per-model-seed] [--json out.json]

Each input is a predictions.jsonl / records.jsonl or a folder searched recursively for them.
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

SEEDS = (0, 202)
SEED_RE = re.compile(r'^(?P<method>.+?)_s(?P<seed>0|202)(?P<tail>_it100)?$')


def method_seed(name):
    m = SEED_RE.match(name)
    if not m:
        return name, None
    return m.group('method') + (m.group('tail') or ''), int(m.group('seed'))


def decisions(path, quality=0):
    """Yield (identity, arm_or_concept, protected, model_seed) for every TEST photo answer of one answer file.

    model_seed is None for answer files without one (reactive Yo'LLaVA, MyVLM)."""
    from eval.semantic_parser import myvlm_recognised, semantic_answer
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get('error'):
            raise ValueError(f'{path}: an answer has a runtime error; re-run that job')
        if 'role' in r:                                          # MyVLM records (decision re-parsed from raw)
            if r['role'] == 'absent':
                continue
            name = r.get('method') or r['role']           # proactive: the method; reactive: the arm / clean
            identity = r.get('identity') or Path(path).parent.name.removeprefix('re_')
            yield identity, name, not myvlm_recognised(r['raw'], 'sks'), None
        elif 'set' in r:                                         # proactive Yo'LLaVA (clean test photos only)
            if r['set'] == 'test':
                yield r['identity'], r['concept'], semantic_answer(r['raw']) == 'no', int(r['model_seed'])
        else:                                                    # reactive Yo'LLaVA
            if r['label'] == 'yes' and r['quality'] == quality:
                yield r['identity'], r['arm'], semantic_answer(r['raw']) == 'no', None


def files(target):
    target = Path(target)
    if target.is_file():
        return [target]
    return sorted(list(target.rglob('predictions.jsonl')) + list(target.rglob('records.jsonl')))


def collect(inputs, quality):
    """counts[(dataset, identity, method, image seed, model seed)] = [not recognised, photos]"""
    counts = defaultdict(lambda: [0, 0])
    for dataset, target in inputs:
        for f in files(target):
            for identity, name, protected, model_seed in decisions(f, quality):
                method, seed = method_seed(name)
                c = counts[(dataset, identity, method, seed, model_seed)]
                c[0] += int(protected)
                c[1] += 1
    return counts


def model_seeds(counts):
    """The victim model seeds present: [None] for answer files without one, else the sorted seeds."""
    seeds = {k[4] for k in counts}
    if None in seeds and len(seeds) > 1:
        raise ValueError('answer files with and without a victim model seed cannot be aggregated together')
    return [None] if None in seeds else sorted(seeds)


def pooled(counts, method, datasets, identities=None, model_seed=None):
    """{image seed: (not recognised, photos)} over the photos of the group at one model seed."""
    out = {}
    for seed in SEEDS:
        hit = tot = 0
        for (d, i, m, s, ms), (p, n) in counts.items():
            if m != method or d not in datasets or ms != model_seed or (identities and i not in identities):
                continue
            if s == seed or s is None:                          # deterministic: the one run stands for each seed
                hit, tot = hit + p, tot + n
        out[seed] = (hit, tot)
    return out


def per_seed(counts, method, datasets, identities=None, model_seed=None):
    return {s: 100 * h / t if t else None for s, (h, t) in pooled(counts, method, datasets, identities,
                                                                   model_seed).items()}


def summary(values):
    a, b = values[SEEDS[0]], values[SEEDS[1]]
    if a is None or b is None:
        return None, None
    return (a + b) / 2, abs(a - b) / 2


def across_model_seeds(counts, method, datasets, identities=None):
    """dict(mean, std, seeds, per_model_seed): mean ± population std over the model seeds of the per-model-seed number
    (mean over image seeds 0 and 202 of the pooled %). Only complete model seeds count (see the module docstring)."""
    found = {}
    for ms in model_seeds(counts):
        pool = pooled(counts, method, datasets, identities, ms)
        if all(t for _, t in pool.values()):
            found[ms] = (sum(t for _, t in pool.values()), summary(per_seed(counts, method, datasets, identities, ms))[0])
    full = max((n for n, _ in found.values()), default=0)
    per = {ms: v for ms, (n, v) in found.items() if n == full}
    if not per:
        return dict(mean=None, std=None, seeds=[], per_model_seed={})
    xs = list(per.values())
    return dict(mean=statistics.mean(xs), std=statistics.pstdev(xs), seeds=list(per), per_model_seed=per)


def seed_table(counts, methods, groups, model_seed=None):
    """Rows of the image-seed table (mean ± |s0 - s202| / 2) at one model seed; returns (lines, result)."""
    lines = ['| Method | ' + ' | '.join(g for g, _ in groups) + ' |', '|---|' + '---:|' * len(groups)]
    result = {}
    for m in methods:
        cells = []
        for g, ds in groups:
            mean, std = summary(per_seed(counts, m, ds, model_seed=model_seed))
            result[(m, g)] = (mean, std)
            cells.append('n/a' if mean is None else f'{mean:.1f} ± {std:.1f}')
        lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
    return lines, result


def table(counts, per_person=False, per_model_seed=False):
    seeds = model_seeds(counts)
    datasets = sorted({k[0] for k in counts})
    methods = sorted({k[2] for k in counts}, key=lambda m: (m != 'clean', m))
    groups = [(d, [d]) for d in datasets] + ([('both', datasets)] if len(datasets) > 1 else [])
    if len(seeds) == 1:
        lines, result = seed_table(counts, methods, groups, seeds[0])
    else:
        lines = ['| Method | ' + ' | '.join(g for g, _ in groups) + ' |', '|---|' + '---:|' * len(groups)]
        result = {}
        for m in methods:
            cells = []
            for g, ds in groups:
                r = across_model_seeds(counts, m, ds)
                result[(m, g)] = (r['mean'], r['std'])
                if r['mean'] is None:
                    cells.append('n/a')
                    continue
                part = '' if len(r['seeds']) == len(seeds) else f' ({len(r["seeds"])} of {len(seeds)} model seeds)'
                cells.append(f'{r["mean"]:.1f} ± {r["std"]:.1f}{part}')
            lines.append(f'| {m} | ' + ' | '.join(cells) + ' |')
        lines += ['', f'Mean ± std (population) over victim model seeds {", ".join(map(str, seeds))}; per model seed: '
                      'mean over image seeds 0 and 202 of the pooled %.']
        if per_model_seed:
            for ms in seeds:
                lines += ['', f'Model seed {ms} (mean ± |seed 0 − seed 202| / 2):', '']
                lines += seed_table(counts, methods, groups, ms)[0]
    if per_person:
        ids = sorted({(k[0], k[1]) for k in counts}, key=lambda x: (x[0], x[1].lower()))
        lines += ['', '| Identity | ' + ' | '.join(methods) + ' |', '|---|' + '---:|' * len(methods)]
        for d, i in ids:
            cells = []
            for m in methods:
                hit = sum(p for (dd, ii, mm, s, _), (p, n) in counts.items() if (dd, ii, mm) == (d, i, m)
                          for _ in ((1, 2) if s is None else (1,)))
                tot = sum(n for (dd, ii, mm, s, _), (p, n) in counts.items() if (dd, ii, mm) == (d, i, m)
                          for _ in ((1, 2) if s is None else (1,)))
                cells.append(f'{100 * hit / tot:.1f}' if tot else 'n/a')
            lines.append(f'| {d}/{i} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines), result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', action='append', required=True, help='DATASET=PATH (file or folder)')
    p.add_argument('--quality', type=int, default=0, help="Yo'LLaVA reactive: 0 = PNG, else the JPEG quality scored")
    p.add_argument('--per-person', action='store_true')
    p.add_argument('--per-model-seed', action='store_true', help='proactive Yo\'LLaVA: also one table per model seed')
    p.add_argument('--json', type=Path)
    a = p.parse_args(argv)
    inputs = []
    for item in a.input:
        d, _, path = item.partition('=')
        if not d or not path:
            p.error('--input takes DATASET=PATH')
        inputs.append((d, path))
    counts = collect(inputs, a.quality)
    text, result = table(counts, a.per_person, a.per_model_seed)
    print(text)
    if a.json:
        out = {f'{m} | {g}': dict(mean=v[0], std=v[1]) for (m, g), v in result.items()}
        if len(model_seeds(counts)) > 1:
            datasets = sorted({k[0] for k in counts})
            groups = {d: [d] for d in datasets} | ({'both': datasets} if len(datasets) > 1 else {})
            for (m, g) in result:
                r = across_model_seeds(counts, m, groups[g])
                out[f'{m} | {g}'].update(model_seeds=r['seeds'], per_model_seed=r['per_model_seed'])
        a.json.write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
