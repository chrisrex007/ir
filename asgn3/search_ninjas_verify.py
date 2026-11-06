"""Independently verify exported test runs and model fitting provenance."""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent


def verify(root):
    split = json.loads((root / 'search_ninjas_split.json').read_text())
    ids = {part: {q['qid'] for q in queries} for part, queries in split.items()}
    assert not (ids['train'] & ids['validation'] or ids['train'] & ids['test'] or ids['validation'] & ids['test'])
    relevant = defaultdict(set)
    for line in (BASE / 'cranqrel').read_text().splitlines():
        q, d, code = line.split()
        if 1 <= int(code) <= 4:
            relevant[q].add(d)
    rows = list(csv.DictReader((root / 'search_ninjas_metrics.csv').open()))
    label_of = {q['qid']: q['original_id'] for q in split['test']}
    for method in ('zone', 'svm', 'zone_bm25_tiebreak'):
        run = defaultdict(list)
        lines = (root / f'search_ninjas_{method}_test_seqid.txt').read_text().splitlines()
        label_lines = (root / f'search_ninjas_{method}_test_labels.txt').read_text().splitlines()
        assert len(lines) == len(label_lines)
        for line, label_line in zip(lines, label_lines):
            q, placeholder, doc, rank, score, tag = line.split()
            assert placeholder == 'Q0' and tag == 'search_ninjas'
            assert 1 <= int(doc) <= 1400 and math.isfinite(float(score))
            assert int(rank) == len(run[q]) + 1
            assert label_line.split() == [label_of[q]] + line.split()[1:]
            run[q].append((doc, float(score)))
        assert set(run) == ids['test']
        for ranked in run.values():
            assert len(ranked) == 100 and len({d for d, _ in ranked}) == 100
            assert all(a[1] >= b[1] for a, b in zip(ranked, ranked[1:]))
        for k in (5, 10, 20, 50, 100):
            measured = []
            for q, ranked in run.items():
                hits = len({d for d, _ in ranked[:k]} & relevant[q])
                p, r = hits / k, hits / len(relevant[q]) if relevant[q] else 0
                measured.append((p, r, 2*p*r/(p+r) if p+r else 0))
            expected = next(row for row in rows if row['method'] == method and row['split'] == 'test' and int(row['k']) == k)
            for column, name in enumerate(('precision', 'recall', 'f1')):
                assert abs(sum(row[column] for row in measured)/len(measured)-float(expected[name])) < 1e-12
        print(f'{method}: independent metrics agree at all 5 cutoffs for {len(run)} test queries')
    for name, expected, role in [('model', ids['train'], 'evaluation'),
                                  ('final_model', ids['train'] | ids['validation'], 'deployment')]:
        bundle = joblib.load(root / f'search_ninjas_{name}.joblib')
        meta_path = 'search_ninjas_metadata.json' if role == 'evaluation' else 'search_ninjas_final_metadata.json'
        meta = json.loads((root / meta_path).read_text())
        assert bundle['metadata'] == meta
        assert meta['role'] == role and set(meta['fit_qids']) == expected
        assert set(meta['fit_qids']).isdisjoint(ids['test'])
        assert int(bundle['svm'].steps[0][1].n_samples_seen_) == len(expected)*1400
        print(f'{role}: scaler fitted to {len(expected)*1400} pairs; test queries excluded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=BASE / 'search_ninjas_artifacts')
    verify(parser.parse_args().artifacts)
