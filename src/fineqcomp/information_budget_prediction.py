"""Compare receiver-relative measurements without fitting on the test panel."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from fineqcomp.artifacts import read_json, write_json
from fineqcomp.codec import MAGIC, container_header_bits, read_container


MEASURES = ('context', 'online_excess', 'transfer', 'shared_gain', 'gain_integral', 'spectral')
LAWS = ('power', 'affine', 'log_affine')
FORECASTS = ('early_identity', 'inverse_sqrt', 'inverse_time')


def public_container_bits(path):
    """FP16 capacity before entropy compression; depends only on public shapes.

    This is a control for addressing cost, not an achieved information rate.
    The achieved adapter targets continue to use complete serialized files.
    """
    header, payload = read_container(path, MAGIC)
    if header['bits'] != 16:
        raise ValueError('public capacity control requires the FP16 container')
    expected = sum(2 * tensor['count'] for tensor in header['tensors'])
    if len(payload) != expected:
        raise ValueError('unexpected FP16 payload layout')
    return container_header_bits(MAGIC, header) + 8 * expected


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
    # Spectral bits are already a size, so unlike the likelihood measures they
    # are not scaled to the corpus. Artifacts written before this was recorded
    # leave it absent, which makes its candidates ineligible rather than wrong.
    spectral = [t.get('spectral') for t in endpoints]
    return dict(context=None if context['gain_bits'] is None else n * context['gain_bits'] / len(base),
        online_excess=features['online_excess_bits'], transfer=n * float(np.mean(transfer)),
        shared_gain=n * float(np.minimum(*gains).mean()), gain_integral=n * float(np.mean(areas)),
        spectral=None if any(v is None for v in spectral) else float(np.mean(spectral)))


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
    x = [r['measures'].get(method['measure']) for r in rows]
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
    x = row['measures'].get(method['measure'])
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


def choose(rows, pool=None):
    trials = []
    for method in pool if pool is not None else candidates():
        records = cross_validate(rows, method)
        scored = metrics(records)
        # An abstaining method cannot win by omitting its difficult cases.
        eligible = scored['covered'] == scored['n'] and scored['n'] > 0
        trials.append(dict(method=method, metrics=scored, eligible=eligible))
    eligible = [t for t in trials if t['eligible']]
    if not eligible:
        return None, trials
    best = min(eligible, key=lambda t: (t['metrics']['mean_absolute_log2_error'], json.dumps(t['method'], sort_keys=True)))
    return best['method'], trials


def develop(rows, pool=None):
    """Nested grouped tests evaluate selection of a combination, not just its fit."""
    nested = []
    for axis in ('task', 'family'):
        for heldout in sorted({r[axis] for r in rows}):
            train = [r for r in rows if r[axis] != heldout]
            method, _ = choose(train, pool)
            # A fold with nothing eligible is uncovered, not absent.
            fitted = fit(train, method) if method is not None else None
            for row in rows:
                if row[axis] == heldout:
                    nested.append(dict(axis=axis, heldout=heldout, method=method, run_id=row['run_id'],
                        observed=row['target_bits'], predicted=predict(fitted, row)))
    method, trials = choose(rows, pool)
    return dict(fitted=fit(rows, method) if method is not None else None, candidates=trials,
                nested_predictions=nested, nested_metrics=metrics(nested))


def spearman(left, right):
    pairs = [(a, b) for a, b in zip(left, right)
             if a is not None and b is not None and np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3:
        return None
    ranked = [np.argsort(np.argsort(np.array(column, float))) for column in zip(*pairs)]
    if min(np.std(column) for column in ranked) < 1e-12:
        return None
    return float(np.corrcoef(*ranked)[0, 1])


def measure_redundancy(rows):
    """Whether the five measures are five theories or one scalar seen five ways.

    Each measure is a likelihood change read off the same receiver over the same
    rows, so they can agree almost exactly while being written up as different
    accounts of what information a fine-tune carries. Rank correlation says how
    much any two of them share. Restricting the search to one measure at a time
    and re-running the nested selection says whether any of them predicts what
    the others cannot; measures whose honest errors sit on top of each other are
    not independent evidence, whatever they are called.
    """
    columns = {m: [r['measures'].get(m) for r in rows] for m in MEASURES}
    pairs = {f'{a}|{b}': spearman(columns[a], columns[b])
             for index, a in enumerate(MEASURES) for b in MEASURES[index + 1:]}
    alone = {m: metrics(develop(rows, [c for c in candidates() if c.get('measure') == m])['nested_predictions'])
             for m in MEASURES}
    errors = [v['mean_absolute_log2_error'] for v in alone.values() if v['mean_absolute_log2_error'] is not None]
    strong = [name for name, value in pairs.items() if value is not None and abs(value) >= .95]
    return dict(pairwise_spearman=pairs, nested_alone=alone,
        indistinguishable_pairs=strong,
        spread_of_nested_error=None if len(errors) < 2 else float(max(errors) - min(errors)),
        boundary='Redundancy among the measures on this panel. It does not show that any of '
                 'them is right, only whether they are separate.')


def selection_null(rows, repeats=200, seed=0):
    """How good does the best of many rules look when the targets carry no signal?

    Twelve discovery cells cannot support a search over thirty-six candidates,
    and the winner's own cross-validated error is the minimum of that search, so
    it is biased downwards by exactly the amount this measures. Permuting the
    targets across cells breaks every real relationship while leaving the
    measures, the panel structure, the eligibility rule and the selection rule
    untouched, so the error still reachable is the part of the winner's showing
    that selection alone explains.
    """
    def best_of_search(sample):
        scores = [t['metrics']['mean_absolute_log2_error'] for t in choose(sample)[1] if t['eligible']]
        return min(scores) if scores else None
    observed = best_of_search(rows)
    generator = np.random.default_rng(seed)
    targets = [r['target_bits'] for r in rows]
    null = []
    for _ in range(repeats):
        permuted = generator.permutation(targets)
        drawn = best_of_search([dict(r, target_bits=float(t)) for r, t in zip(rows, permuted)])
        if drawn is not None:
            null.append(drawn)
    if observed is None or not null:
        return dict(observed=observed, repeats=len(null), p_value=None)
    return dict(observed=observed, repeats=len(null),
        null_median=float(np.median(null)), null_best=float(min(null)),
        p_value=float((1 + sum(score <= observed for score in null)) / (1 + len(null))),
        boundary='A permutation null for the search, not a test of any one rule. '
                 'It bounds selection advantage; it does not validate the winner.')


def winner_stability(rows):
    """How often the same rule wins when one cell is removed.

    A search that returns a different combination each time a single receiver
    drops out has not identified a law, whatever its error looks like.
    """
    winners = [json.dumps(choose(rows[:index] + rows[index + 1:])[0], sort_keys=True)
               for index in range(len(rows))]
    counts = Counter(winners)
    modal, share = counts.most_common(1)[0]
    return dict(leave_one_out=len(winners), distinct_winners=len(counts),
        modal_winner=json.loads(modal), modal_share=share / len(winners),
        full_sample_winner=choose(rows)[0])


def screening_report(rows, developed=None, repeats=200, seed=0):
    """Report the search as screening: optimistic score, honest score, baselines.

    Three numbers have to be kept apart. The winner's cross-validated error is
    the minimum of a search and is optimistic. The nested error re-runs the whole
    selection inside every fold and is the honest one. A fixed baseline needs no
    selection at all, so the margin against it is what the search bought.
    """
    selected, trials = choose(rows)
    eligible = [t for t in trials if t['eligible']]
    if not eligible:
        return dict(stage='screening', selected=None, candidates=len(candidates()), cells=len(rows),
                    boundary='No candidate covered every cell, so nothing was selected.')
    optimistic = min(t['metrics']['mean_absolute_log2_error'] for t in eligible)
    baselines = {t['method']['baseline']: metrics(cross_validate(rows, t['method']))
                 for t in eligible if 'baseline' in t['method']}
    honest = metrics((developed if developed is not None else develop(rows))['nested_predictions'])
    best_baseline = min((v['mean_absolute_log2_error'] for v in baselines.values()
                         if v['mean_absolute_log2_error'] is not None), default=None)
    return dict(stage='screening', selected=selected, candidates=len(candidates()), cells=len(rows),
        optimistic_mean_absolute_log2_error=optimistic, nested=honest, baselines=baselines,
        margin_over_best_baseline=None if best_baseline is None or honest['mean_absolute_log2_error'] is None
            else best_baseline - honest['mean_absolute_log2_error'],
        null=selection_null(rows, repeats, seed), stability=winner_stability(rows),
        boundary='Screening on the discovery panel. The selected rule is frozen here and '
                 'is not evidence until development and outer audit read it unchanged.')


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
        exposure = read_json(root / 'training_exposure.json')
        target = next(r for r in targets['budgets'] if r['retention'] == .9 and not r['scaled'])
        if target['status'] != 'measured':
            exclusions.append(dict(run_id=run['run_id'], reason=target['status'], exposure=exposure)); continue
        model = run['model']['key']
        family = 'qwen' if model.startswith('qwen') else model.split('_')[0].rstrip('0123456789')
        early = min(features['traces'], key=lambda t: (t['step'], t['replica']))
        code = next(r for r in early['grid'] if r['rank'] == 1 and r['precision'] == 16)
        container_path = root / f"replica{early['replica']}/step{early['step']}/codecs/{code['key']}.fqcb"
        container = public_container_bits(container_path)
        provenance[run['run_id']]['capacity_container'] = hashlib.sha256(container_path.read_bytes()).hexdigest()
        rows.append(dict(run_id=run['run_id'], model=model, family=family, task=run['dataset_key'],
            stage=stage, measures=extract(features), forecasts={f: curve_forecast(features, f) for f in FORECASTS},
            container_bits=container, target_bits=target['file_bits'], verification_retention=target['verification_retention'],
            below_smallest_tested=target['below_smallest_tested'], exposure=exposure))
    return rows, exclusions, provenance


def exposure_summary(records):
    """What each receiver was taught, beside the span its scores are read over.

    A cell can miss its retention target because the receiver learned nothing,
    or because the training limit cut the responses it is later scored on in
    full. Those are different failures, so excluded cells keep their exposure
    here rather than leaving the analysis with only the cells that succeeded.
    """
    cells = []
    for record in records:
        e = record.get('exposure')
        if e is None:
            continue
        cells.append(dict(run_id=record['run_id'], train_limit=e['train_limit'], score_limit=e['score_limit'],
            dropped_rows=e['original_rows'] - e['usable_rows'],
            truncated_fraction=e['rows_truncated'] / max(e['usable_rows'], 1),
            unsupervised_fraction=e['unsupervised_response_tokens'] / max(e['response_tokens'], 1)))
    return dict(cells=cells, missing=sum(r.get('exposure') is None for r in records),
        limits_matched=all(c['train_limit'] >= c['score_limit'] for c in cells) if cells else None,
        worst_unsupervised_fraction=max((c['unsupervised_fraction'] for c in cells), default=None),
        boundary='Exposure counts the training corpus. The rate panels are scored rows, '
                 'so a matched limit is necessary for comparability, not sufficient.')


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
        result['screening'] = screening_report(rows, developed=result)
        result['redundancy'] = measure_redundancy(rows)
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
        exposure=exposure_summary(rows + exclusions),
        verification=dict(measured=len(rows), passed=sum(r['verification_retention'] is not None and r['verification_retention'] >= .9 for r in rows)),
        boundary='Selection-half achieved bytes, with separate verification. Old model panel; no prospective or universal scaling claim.')
    if args.out.exists():
        raise ValueError('refusing to replace a frozen analysis')
    write_json(args.out, result)
    print(json.dumps({k: v for k, v in result.items()
                      if k in ('stage', 'metrics', 'nested_metrics', 'verification', 'exclusions')} |
                     ({'screening': {k: v for k, v in result['screening'].items() if k != 'baselines'},
                       'redundancy': {k: v for k, v in result['redundancy'].items()
                                      if k != 'nested_alone'}} if 'screening' in result else {}) |
                     {'exposure': {k: v for k, v in result['exposure'].items() if k != 'cells'}}, indent=2))


if __name__ == '__main__':
    main()
