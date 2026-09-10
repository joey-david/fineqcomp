"""Compare receiver-relative measurements without fitting on the test panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from fineqcomp.artifacts import read_json, write_json


MEASURES = ('context', 'online_excess', 'transfer', 'shared_gain', 'gain_integral')
LAWS = ('power', 'affine', 'log_affine')
FORECASTS = ('early_identity', 'inverse_sqrt', 'inverse_time')


def achieved(grid, base, reference, rho=.9):
    if base <= reference:
        return None
    passing = [r for r in grid if base - r['selection'] >= rho * (base - reference)]
    return min((r['file_bits'] for r in passing), default=None)


def curve_forecast(features, family):
    """Predict a future crossing using early curves and a public time horizon."""
    predictions = []
    for replica in (0, 1):
        trace = sorted((t for t in features['traces'] if t['replica'] == replica), key=lambda t: t['step'])
        base = features['base_rate']['selection']
        if family == 'early_identity':
            estimate = achieved(trace[-1]['grid'], base, trace[-1]['reference']['selection'])
        else:
            exponent = .5 if family == 'inverse_sqrt' else 1.
            times = np.array([t['step'] for t in trace], dtype=float)
            design = np.column_stack([np.ones(len(times)), times ** -exponent])
            future = np.array([1., features['full_horizon'] ** -exponent])
            def forecast(losses):
                return float(future @ np.linalg.lstsq(design, losses, rcond=None)[0])
            reference = forecast([t['reference']['selection'] for t in trace])
            grid = []
            for cell in trace[-1]['grid']:
                losses = [next(r['selection'] for r in t['grid'] if r['key'] == cell['key']) for t in trace]
                loss = forecast(losses)
                if loss >= 0:
                    grid.append(dict(file_bits=cell['file_bits'], selection=loss))
            estimate = achieved(grid, base, reference) if reference >= 0 else None
        predictions.append(estimate)
    return float(np.mean(predictions)) if all(p is not None for p in predictions) else None


def extract(features):
    """Code contrasts are in bits for the original number of training examples.

    Shared gain is a conservative pointwise overlap of two learners' gains,
    not mutual information. Online excess remains an early learner's code cost.
    """
    base = np.array(features['base']['per_example_bits'])
    n = features['training_rows']
    endpoints = [max((t for t in features['traces'] if t['replica'] == r), key=lambda t: t['step'])
                 for r in (0, 1)]
    gains = [base - t['feature']['per_example_bits'] for t in endpoints]
    context = next(r for r in features['frozen_context'] if r['requested'] == 4)
    transfer = [
        (features['base_halves'][1-r]['bits'] - endpoints[r]['other_half']['bits']) /
        endpoints[r]['other_half']['examples'] for r in (0, 1)]
    areas = []
    for replica in (0, 1):
        trace = sorted((t for t in features['traces'] if t['replica'] == replica), key=lambda t: t['step'])
        times = np.array([0] + [t['step'] for t in trace], float)
        values = np.array([0.] + [(features['base']['bits'] - t['feature']['bits']) / len(base) for t in trace])
        areas.append(float(np.sum(np.diff(times) * (values[:-1] + values[1:]) / 2) / times[-1]))
    return dict(context=None if context['gain_bits'] is None else n * context['gain_bits'] / len(base),
        online_excess=features['online_excess_bits'], transfer=n * float(np.mean(transfer)),
        shared_gain=n * float(np.minimum(*gains).mean()), gain_integral=n * float(np.mean(areas)))


def candidates():
    return ([dict(measure=m, law=law, container=container) for m in MEASURES for law in LAWS
             for container in (False, True)] + [dict(forecast=f) for f in FORECASTS] +
            [dict(baseline=b) for b in ('global_mean', 'container', 'corpus_mean')])


def fit(rows, method):
    if 'forecast' in method:
        return dict(method=method)
    y = np.array([r['target_bits'] for r in rows], float)
    if 'baseline' in method:
        mode = method['baseline']
        return dict(method=method, mean=float(y.mean()),
            ratio=float(np.mean(y / [r['container_bits'] for r in rows])),
            corpora={task: float(np.mean([r['target_bits'] for r in rows if r['task'] == task]))
                     for task in {r['task'] for r in rows}})
    x = [r['measures'][method['measure']] for r in rows]
    if any(v is None or not np.isfinite(v) for v in x):
        return None
    x = np.array(x)
    if method['law'] != 'affine' and np.any(x <= 0):
        return None
    if method['container']:
        y = y / [r['container_bits'] for r in rows]
    z = x if method['law'] == 'affine' else np.log(x)
    # Center and scale on training rows only for a stable least-squares solve.
    center, scale = float(z.mean()), max(float(z.std()), 1e-12)
    design = np.column_stack([np.ones(len(z)), (z - center) / scale])
    coefficients = np.linalg.lstsq(design, np.log(y) if method['law'] == 'power' else y, rcond=None)[0]
    return dict(method=method, center=center, scale=scale, coefficients=coefficients.tolist())


def predict(fitted, row):
    if fitted is None:
        return None
    method = fitted['method']
    if 'forecast' in method:
        return row['forecasts'][method['forecast']]
    if 'baseline' in method:
        mode = method['baseline']
        return (fitted['ratio'] * row['container_bits'] if mode == 'container' else
                fitted['corpora'].get(row['task'], fitted['mean']) if mode == 'corpus_mean' else fitted['mean'])
    x = row['measures'][method['measure']]
    if x is None or not np.isfinite(x) or (method['law'] != 'affine' and x <= 0):
        return None
    z = x if method['law'] == 'affine' else np.log(x)
    value = float(np.dot([1., (z - fitted['center']) / fitted['scale']], fitted['coefficients']))
    if method['law'] == 'power':
        if value > 700:
            return None
        value = float(np.exp(value))
    if method['container']:
        value *= row['container_bits']
    return value if np.isfinite(value) and value > 0 else None


def metrics(records):
    valid = [r for r in records if r['predicted'] is not None and r['predicted'] > 0]
    if not valid:
        return dict(n=len(records), covered=0, mean_absolute_log2_error=None)
    y, p = np.array([[r['observed'], r['predicted']] for r in valid]).T
    log_errors = np.log2(p / y)
    slope = float(np.polyfit(np.log2(p), np.log2(y), 1)[0]) if np.std(np.log2(p)) > 1e-9 else None
    return dict(n=len(records), covered=len(valid), rmse_bytes=float(np.sqrt(np.mean(((p-y)/8)**2))),
        bias_bytes=float(np.mean((p-y)/8)), mean_absolute_log2_error=float(np.abs(log_errors).mean()),
        median_factor_error=float(2 ** np.median(np.abs(log_errors))),
        factor_two_fraction=float(np.mean(np.abs(log_errors) <= 1)), calibration_slope=slope)


def cross_validate(rows, method):
    records = []
    for axis in ('task', 'family'):
        for heldout in sorted({r[axis] for r in rows}):
            train = [r for r in rows if r[axis] != heldout]
            if not train:
                continue
            fitted = fit(train, method)
            for row in rows:
                if row[axis] == heldout:
                    records.append(dict(axis=axis, heldout=heldout, run_id=row['run_id'],
                        observed=row['target_bits'], predicted=predict(fitted, row)))
    return records


def choose(rows):
    trials = []
    for method in candidates():
        records = cross_validate(rows, method)
        scored = metrics(records)
        # An abstaining method cannot win by omitting its difficult cases.
        eligible = scored['covered'] == scored['n'] and scored['n'] > 0
        trials.append(dict(method=method, metrics=scored, eligible=eligible))
    eligible = [t for t in trials if t['eligible']]
    best = min(eligible, key=lambda t: (t['metrics']['mean_absolute_log2_error'], json.dumps(t['method'], sort_keys=True)))
    return best['method'], trials


def develop(rows):
    """Nested grouped tests evaluate selection of a combination, not just its fit."""
    nested = []
    for axis in ('task', 'family'):
        for heldout in sorted({r[axis] for r in rows}):
            train = [r for r in rows if r[axis] != heldout]
            method, _ = choose(train)
            fitted = fit(train, method)
            for row in rows:
                if row[axis] == heldout:
                    nested.append(dict(axis=axis, heldout=heldout, method=method, run_id=row['run_id'],
                        observed=row['target_bits'], predicted=predict(fitted, row)))
    method, trials = choose(rows)
    return dict(fitted=fit(rows, method), candidates=trials, nested_predictions=nested, nested_metrics=metrics(nested))


def load_rows(config, source, stage):
    rows, exclusions, provenance = [], [], {}
    for cell in config['cells']:
        if cell['stage'] != stage:
            continue
        run = cell['run']; root = source / 'cells' / run['run_id']
        if not (root / 'complete.json').exists():
            raise ValueError(f'incomplete {stage} cell: {run["run_id"]}')
        features, targets = read_json(root / 'features.json'), read_json(root / 'targets.json')
        provenance[run['run_id']] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                    for name in ('features.json', 'targets.json', 'data_contract.json')}
        target = next(r for r in targets['budgets'] if r['retention'] == .9 and not r['scaled'])
        if target['status'] != 'measured':
            exclusions.append(dict(run_id=run['run_id'], reason=target['status'])); continue
        model = run['model']['key']
        family = 'qwen' if model.startswith('qwen') else model.split('_')[0].rstrip('0123456789')
        container = next(r['file_bits'] for r in features['traces'][0]['grid']
                         if r['rank'] == 1 and r['precision'] == 16)
        rows.append(dict(run_id=run['run_id'], model=model, family=family, task=run['dataset_key'],
            stage=stage, measures=extract(features), forecasts={f: curve_forecast(features, f) for f in FORECASTS},
            container_bits=container, target_bits=target['file_bits'], verification_retention=target['verification_retention'],
            below_smallest_tested=target['below_smallest_tested']))
    return rows, exclusions, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['discover', 'development', 'outer_audit'], required=True)
    parser.add_argument('--lock', type=Path)
    args = parser.parse_args(); config = read_json(args.config)
    stage = 'discovery' if args.phase == 'discover' else args.phase
    rows, exclusions, provenance = load_rows(config, args.source, stage)
    if args.phase == 'discover':
        result = develop(rows)
        result['config_sha256'] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    else:
        if args.lock is None:
            raise ValueError('freeze discovery selection before opening later outcomes')
        lock = read_json(args.lock)
        if lock['config_sha256'] != hashlib.sha256(args.config.read_bytes()).hexdigest():
            raise ValueError('panel changed after selection')
        records = [dict(run_id=r['run_id'], observed=r['target_bits'], predicted=predict(lock['fitted'], r)) for r in rows]
        result = dict(locked_sha256=hashlib.sha256(args.lock.read_bytes()).hexdigest(), predictions=records, metrics=metrics(records))
    result.update(stage=stage, rows=rows, exclusions=exclusions, inputs=provenance,
        verification=dict(measured=len(rows), passed=sum(r['verification_retention'] is not None and r['verification_retention'] >= .9 for r in rows)),
        boundary='Selection-half achieved bytes, with separate verification. Old model panel; no prospective or universal scaling claim.')
    if args.out.exists():
        raise ValueError('refusing to replace a frozen analysis')
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k in ('stage', 'metrics', 'nested_metrics', 'verification', 'exclusions')}, indent=2))


if __name__ == '__main__':
    main()
