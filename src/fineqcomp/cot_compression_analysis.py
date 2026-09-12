"""Read CoT compression predictions and keep task accuracy separate from contrasts."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from fineqcomp.artifacts import read_json, write_json
from fineqcomp.correction_conditioning import strict_answer
from fineqcomp.evaluation import _normalize_number


def predictions(path, expected=None):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError(f'duplicate IDs: {path}')
    if expected is not None and len(rows) != expected:
        raise ValueError(f'expected {expected} responses: {path}')
    for row in rows:
        actual = _normalize_number(row['response']) == row['expected']
        if actual != row['correct']:
            raise ValueError(f'stored score disagrees with parser: {path}, {row["example_id"]}')
    return rows


def accuracy(rows):
    return sum(r['correct'] for r in rows) / len(rows)


def answer_only(text):
    """Check the requested direct format, apart from surrounding whitespace."""
    return re.fullmatch(r'\s*####\s*[-+]?\d[\d,]*(?:\.\d+)?\s*', text) is not None


def paired_strict_difference(left, right):
    """Compare matched answers and resample whole templates."""
    right_by_id = {row['example_id']: row for row in right}
    if set(right_by_id) != {row['example_id'] for row in left}:
        raise ValueError('paired answer IDs differ')
    deltas = defaultdict(list)
    both = left_only = right_only = 0
    for a in left:
        b = right_by_id[a['example_id']]
        ac, bc = bool(a['strict_correct']), bool(b['strict_correct'])
        deltas[a['template']].append(int(ac) - int(bc))
        both += ac and bc
        left_only += ac and not bc
        right_only += bc and not ac
    values = np.array([np.mean(group) for group in deltas.values()])
    rng = np.random.default_rng(260910)
    interval = np.quantile(
        values[rng.integers(0, len(values), (5000, len(values)))].mean(axis=1),
        [.025, .975],
    )
    return dict(difference=float(values.mean()), ci95=interval.tolist(),
                both_correct=both, left_only=left_only, right_only=right_only)


def expected_states(config, rank):
    """Name every adapter state that the configured runner must write."""
    states = {'base', 'raw', 'mask_half'}
    states.update(f'uniform_b{bits}' for bits in config.get('uniform_bits', ()))
    states.update(f'scale_{scale:g}' for scale in config.get('scales', (.25, .5, .75)))
    states.update(f'svd_r{value}' for value in config.get('svd_ranks', (1, 2, 4, 8))
                  if value <= rank)
    if config.get('svd_binary', True):
        states.add('svd_r2_binary')
    states.update(f'random_r2_{draw}' for draw in range(config.get('random_draws', 3)))
    return states


def historical(source):
    study = source / 'results/3_chain_of_thought_under_compression'
    pairs = list(csv.DictReader((study / 'conditional_trace_rate/pairs.csv').open()))
    expected = read_json(study / 'conditional_trace_rate_lock.json')['frozen_grid']['test_rows']
    cells, provenance = [], {}
    for pair in pairs:
        for arm in ('aligned', 'permuted'):
            run = source / 'runs' / pair[f'{arm}_run_id']
            for point in ('raw', 'uniform2', 'blend1_50', 'binary'):
                path = run / 'predictions' / ('raw_task.jsonl' if point == 'raw' else f'task_{point}.jsonl')
                rows = predictions(path, expected)
                provenance[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
                score = accuracy(rows)
                if abs(score - float(pair[f'{arm}_{point}'])) > 1e-9:
                    raise ValueError(f'published pair disagrees: {path}')
                codec = read_json(run / 'codec_metrics' / f'{point}.json') if point != 'raw' else None
                anchor = read_json(run / 'codec_metrics/binary.json')['retained_gain']['baseline_score']
                cells.append(dict(model=pair['model_key'], seed=int(pair['seed']), arm=arm, point=point,
                    n=len(rows), accuracy=score, base_accuracy=anchor,
                    strict_accuracy=sum(strict_answer(r['response']) == r['expected'] for r in rows) / len(rows),
                    file_bytes=None if codec is None else codec['storage']['file_bits'] / 8,
                    effective_bits_per_value=None if codec is None else codec['storage']['effective_bits_per_value']))
    groups = []
    for model in sorted({c['model'] for c in cells}):
        for point in ('raw', 'uniform2', 'blend1_50', 'binary'):
            arms = {a: [c for c in cells if c['model'] == model and c['point'] == point and c['arm'] == a]
                    for a in ('aligned', 'permuted')}
            aligned, permuted = [float(np.mean([c['accuracy'] for c in arms[a]])) for a in arms]
            groups.append(dict(model=model, point=point, seeds=len(arms['aligned']),
                aligned_accuracy=aligned, permuted_accuracy=permuted, gap=aligned-permuted,
                base_accuracy=float(np.mean([c['base_accuracy'] for c in arms['aligned']])),
                aligned_file_bytes=None if point == 'raw' else float(np.mean([c['file_bytes'] for c in arms['aligned']]))))
    return dict(cells=cells, groups=groups, source_sha256=provenance,
        files_checked=len(provenance), predictions_checked=sum(c['n'] for c in cells),
        boundary='Saved GSM8K tests on three models and three seeds. Raw BF16 is a reference, '
                 'not a measured container size. Compression targets only the trained adapter. '
                 'The aligned-minus-permuted gap can shrink because either arm changes. '
                 'Final answers do not establish correctness of every written step.')


def pilot(root):
    lock = read_json(root / 'lock.json')
    cells, missing, arm_rows = [], [], {}
    for spec in lock['cells']:
        run = root / 'cells' / spec['run_id']
        if not (run / 'complete.json').exists():
            missing.append(spec['run_id'])
        paths = sorted(run.glob('*/predictions.jsonl'))
        observed = {path.parent.name for path in paths
                    if (path.parent / 'direct/predictions.jsonl').exists()}
        expected = expected_states(lock['config'], spec['adapter']['rank'])
        if observed != expected:
            missing.append(f"{spec['run_id']}: states {sorted(expected - observed)}")
        for path in paths:
            direct_path = path.parent / 'direct/predictions.jsonl'
            if not direct_path.exists():
                continue
            cot = predictions(path, len(lock['examples']))
            direct = {r['example_id']: r for r in predictions(direct_path, len(lock['examples']))}
            if set(direct) != {r['example_id'] for r in cot}:
                raise ValueError(f'unpaired modes: {path}')
            storage = read_json(run / 'codecs' / f'{path.parent.name}.json')
            if path.parent.name != 'base' and storage is None:
                raise ValueError(f'missing serialized size: {path.parent}')
            for split in ('development', 'test'):
                left = [r for r in cot if r['split'] == split]
                if not left:
                    continue
                right = [direct[r['example_id']] for r in left]
                paired = paired_strict_difference(left, right)
                arm = 'permuted' if spec['dataset_key'].endswith('_permuted') else 'aligned'
                arm_rows[(spec['model']['key'], spec['seed'], path.parent.name, split, arm)] = {
                    'cot': left, 'direct': right,
                }
                cells.append(dict(model=spec['model']['key'], seed=spec['seed'], dataset=spec['dataset_key'],
                    codec=path.parent.name, split=split, n=len(left),
                    file_bytes=0 if storage is None else storage['file_bits'] / 8,
                    cot_accuracy=accuracy(left), direct_accuracy=accuracy(right),
                    cot_strict_accuracy=float(np.mean([r['strict_correct'] for r in left])),
                    direct_strict_accuracy=float(np.mean([r['strict_correct'] for r in right])),
                    cot_strict_parse_rate=float(np.mean([r['strict_prediction'] is not None for r in left])),
                    direct_strict_parse_rate=float(np.mean([r['strict_prediction'] is not None for r in right])),
                    paired_cot_minus_direct=paired['difference'], paired_ci95=paired['ci95'],
                    cot_cap_rate=float(np.mean([r['hit_generation_limit'] for r in left])),
                    direct_cap_rate=float(np.mean([r['hit_generation_limit'] for r in right])),
                    cot_mean_tokens=float(np.mean([r['completion_tokens'] for r in left])),
                    direct_mean_tokens=float(np.mean([r['completion_tokens'] for r in right])),
                    cot_repeated_8gram_fraction=float(np.mean([r['repeated_8gram_fraction'] for r in left])),
                    direct_repeated_8gram_fraction=float(np.mean([r['repeated_8gram_fraction'] for r in right])),
                    direct_stop_rate=float(np.mean([r.get('stopped_by_string', False) for r in right])),
                    direct_answer_only_rate=float(np.mean([answer_only(r['response']) for r in right])),
                    direct_multiline_rate=float(np.mean([
                        len([line for line in r['response'].splitlines() if line.strip()]) > 1
                        for r in right])))
                )
    arm_comparisons = []
    keys = sorted({key[:-1] for key in arm_rows})
    for key in keys:
        arms = {arm: arm_rows.get((*key, arm)) for arm in ('aligned', 'permuted')}
        if any(rows is None for rows in arms.values()):
            missing.append(f'{key}: unpaired training arms')
            continue
        for mode in ('cot', 'direct'):
            paired = paired_strict_difference(arms['aligned'][mode], arms['permuted'][mode])
            arm_comparisons.append(dict(
                model=key[0], seed=key[1], codec=key[2], split=key[3], mode=mode,
                n=len(arms['aligned'][mode]), paired_aligned_minus_permuted=paired['difference'],
                paired_ci95=paired['ci95'], both_correct=paired['both_correct'],
                aligned_only=paired['left_only'], permuted_only=paired['right_only'],
            ))
    return dict(complete=not missing, missing=sorted(set(missing)), cells=cells,
                arm_comparisons=arm_comparisons, config=lock['config'])


def plot(report, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2), sharey=True)
    points = ('raw', 'uniform2', 'blend1_50', 'binary')
    for ax, model in zip(axes, sorted({r['model'] for r in report['groups']}), strict=True):
        rows = {r['point']: r for r in report['groups'] if r['model'] == model}
        for arm, label, color in [('aligned', 'Matched CoT training', '#176b87'),
                                  ('permuted', 'Mismatched CoT control', '#b85232')]:
            ax.plot(range(4), [100*rows[p][f'{arm}_accuracy'] for p in points], 'o-', label=label, color=color)
            seeds = [[100*c['accuracy'] for c in report['cells']
                      if c['model'] == model and c['point'] == p and c['arm'] == arm] for p in points]
            ax.fill_between(range(4), [min(v) for v in seeds], [max(v) for v in seeds], color=color, alpha=.12)
        ax.axhline(100*rows['raw']['base_accuracy'], color='#777777', linestyle=':', label='Frozen base')
        ax.set_title(model.replace('_base', '').replace('_', ' '))
        ax.set_xticks(range(4), ['Raw', '2 bit', '1.5 bit', '1 bit'])
        ax.set_xlabel('Increasing adapter compression →')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('GSM8K final-answer accuracy (%)')
    fig.legend(*axes[-1].get_legend_handles_labels(), fontsize=9, loc='lower center', ncol=3)
    fig.suptitle('Compression can preserve useful CoT tuning while removing harmful tuning')
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(out / 'cot_compression.png', dpi=180)
    fig.savefig(out / 'cot_compression.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--pilot', type=Path)
    args = parser.parse_args()
    result = historical(args.source)
    if args.pilot:
        result['pilot'] = pilot(args.pilot)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(args.out / 'audit.json', result)
    plot(result, args.out)
    print(json.dumps({k: result[k] for k in ('files_checked', 'predictions_checked', 'groups')}))


if __name__ == '__main__':
    main()
