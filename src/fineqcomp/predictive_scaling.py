"""Retrospective extrapolation of compression curves across distinct data sizes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def predict(n, y, target_n, family):
    """Fit with training rows only; return predictions in the original units."""
    x = np.log2(np.asarray(n, dtype=float) / 2000)
    z = np.log2(np.asarray(target_n, dtype=float) / 2000)
    y = np.asarray(y, dtype=float)
    if family == 'constant':
        return np.full(len(z), y.mean())
    if family == 'largest_scale':
        return np.full(len(z), y[x == x.max()].mean())
    if family not in ('log_data', 'power'):
        raise ValueError(f'unknown family: {family}')
    if family == 'power' and np.any(y <= 0):
        raise ValueError('power fit needs positive observations')
    coefficients = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]),
        np.log(y) if family == 'power' else y, rcond=None)[0]
    result = np.column_stack([np.ones(len(z)), z]) @ coefficients
    return np.exp(result) if family == 'power' else result


def analyze(source: Path):
    budgets = list(csv.DictReader((source / 'r_star_by_arm.csv').open()))
    points = list(csv.DictReader((source / 'per_seed_points.csv').open()))
    predictions = []
    # Successive prefixes test extrapolation, never a random split of curve rows.
    for cutoff in (8000, 16000):
        for target in ('r_star_bits_axis', 'r_star_accuracy_axis'):
            train = [r for r in budgets if int(r['unique_rows']) <= cutoff]
            test = [r for r in budgets if int(r['unique_rows']) > cutoff]
            for family in ('constant', 'largest_scale', 'log_data', 'power'):
                values = predict([r['unique_rows'] for r in train],
                    [r[target] for r in train], [r['unique_rows'] for r in test], family)
                for row, value in zip(test, values, strict=True):
                    predictions.append(dict(cutoff=cutoff, target=target, family=family,
                        seed=int(row['seed']), n=int(row['unique_rows']),
                        observed=float(row[target]), predicted=float(value)))
        # Each codec is a fixed measured point on the curve. Do not interpolate
        # between codecs or call a predicted crossing a measured file.
        for codec in sorted({r['codec'] for r in points}):
            subset = [r for r in points if r['codec'] == codec]
            train = [r for r in subset if int(r['unique_rows']) <= cutoff]
            test = [r for r in subset if int(r['unique_rows']) > cutoff]
            if not train or not test:
                continue
            for family in ('constant', 'largest_scale', 'log_data'):
                values = predict([r['unique_rows'] for r in train],
                    [r['retention_bits'] for r in train],
                    [r['unique_rows'] for r in test], family)
                for row, value in zip(test, values, strict=True):
                    predictions.append(dict(cutoff=cutoff, target='retention_curve',
                        family=family, codec=codec, seed=int(row['seed']),
                        n=int(row['unique_rows']), observed=float(row['retention_bits']),
                        predicted=float(value)))
    scores = []
    for cutoff, target, family in sorted({(r['cutoff'], r['target'], r['family']) for r in predictions}):
        group = [r for r in predictions if (r['cutoff'], r['target'], r['family']) == (cutoff, target, family)]
        errors = np.array([r['predicted'] - r['observed'] for r in group])
        scores.append(dict(cutoff=cutoff, target=target, family=family,
            rmse=float(np.sqrt(np.mean(errors**2))), bias=float(errors.mean()), rows=len(group)))
    return dict(status='retrospective_development',
        boundary='One receiver, corpus, fixed compute and codec family. Seeds and codec rows are not independent datasets. Budget labels use prior interpolated crossings.',
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (source / 'r_star_by_arm.csv', source / 'per_seed_points.csv')},
        scores=scores, predictions=predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('results/1_rate_behaviour_frontier/adapter_bits_track_unique_data'))
    parser.add_argument('--out', type=Path, default=Path('results/6_predictive_scaling/summary.json'))
    args = parser.parse_args()
    result = analyze(args.source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['scores'], indent=2))


if __name__ == '__main__':
    main()
