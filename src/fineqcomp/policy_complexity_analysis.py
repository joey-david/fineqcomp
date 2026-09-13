"""Report finite-range learning difficulty separately from adapter file budgets."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from fineqcomp.artifacts import read_json


def learning_area(ns, losses, base, reference, previous_reference, plateau_fraction=.05):
    if len(ns) != len(losses) or len(set(ns)) != len(ns) or sorted(ns) != list(ns):
        raise ValueError('area needs sorted distinct data sizes and one loss each')
    gain = base - reference
    if gain <= 0:
        return dict(status='no_positive_reference_gain', area=None)
    residual = (np.array(losses, float) - reference) / gain
    x, y = np.array([0, *ns]), np.array([1., *residual])
    area = float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))
    logx = np.log2(1 + x)
    plateau = abs(previous_reference - reference) <= plateau_fraction * gain
    return dict(status='plateau_screen_passed' if plateau else 'optimization_unresolved',
                area=area, area_over_range=area / max(ns),
                log_area=float(np.sum(np.diff(logx) * (y[:-1] + y[1:]) / 2)),
                residual=residual.tolist(), reference_gain=gain, maximum_n=max(ns),
                interpretation='Finite-range learner-dependent difficulty, not information bits; no clipping.')


def budget(record, threshold, domain='recall', metric='nll'):
    selection, verification = f'selection_{domain}', f'verification_{domain}'
    candidates = [dict(key='base', file_bits=0, scores=record['base']), *record['grid']]
    passing = [r for r in candidates if r['scores'][selection][metric] <= threshold]
    if not passing:
        return dict(status='target_unattained', threshold=threshold, file_bits=None)
    chosen = min(passing, key=lambda r: (r['file_bits'], r['key']))
    observed = chosen['scores'][verification][metric]
    return dict(status='verified' if observed <= threshold else 'verification_failed',
                threshold=threshold, key=chosen['key'], file_bits=chosen['file_bits'],
                bits_per_base_parameter=chosen['file_bits'] / record['base_parameters'],
                selection_nll=chosen['scores'][selection][metric], verification_nll=observed)


def reduce(root):
    root = Path(root)
    lock = read_json(root / 'lock.json')
    config = lock['config']
    records, missing = [], []
    for i, cell in enumerate(lock['cells']):
        record = read_json(root / 'cells' / f'{i:04d}' / 'complete.json')
        if record is None:
            missing.append(i)
            continue
        if record['cell'] != cell or [c['step'] for c in record['checkpoints']] != config['checkpoints']:
            raise ValueError(f'cell contract mismatch: {i}')
        records.append(record)
    groups = defaultdict(list)
    for r in records:
        c = r['cell']
        groups[(c['model']['key'], c['seed'], c['policy'], c['rank'], c['repeat'])].append(r)
    areas, budgets = [], []
    for key, group in groups.items():
        group.sort(key=lambda r: r['cell']['n'])
        ns = [r['cell']['n'] for r in group]
        if ns != config['n_values']:
            continue
        reference = group[-1]
        for domain in ('recall', 'unseen'):
            for metric in ('nll', 'action_nll'):
                split = f'verification_{domain}'
                for checkpoint_index, step in enumerate(config['checkpoints']):
                    losses = [r['checkpoints'][checkpoint_index]['scores'][split][metric] for r in group]
                    final = reference['checkpoints'][-1]['scores'][split][metric]
                    previous = reference['checkpoints'][-2]['scores'][split][metric]
                    value = learning_area(ns, losses, reference['base'][split][metric], final, previous,
                                          config['plateau_fraction'])
                    areas.append(dict(group=list(key), domain=domain, metric=metric, step=step, **value))
                for target in config['common_nll_targets']:
                    budgets.append(dict(group=list(key), domain=domain, metric=metric,
                                        **budget(reference, target, domain, metric)))
    return dict(complete=not missing, expected_cells=len(lock['cells']), completed_cells=len(records),
                missing_cells=missing, learning_areas=areas, budgets=budgets,
                boundary='Pilot description only. No fit or exponent selected from these results. '
                         'Unseen random exceptions are not learnable; recall includes covered and uncovered keys.')
