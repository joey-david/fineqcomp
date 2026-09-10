"""Receiver-relative codes and early functional frontiers on one locked panel."""

from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
import math
from pathlib import Path

import torch

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.codec import encode_tensor_map, decode_adapter_tensor_map, truncate_lora_rank, pad_lora_rank
from fineqcomp.config import RunSpec
from fineqcomp.data import read_jsonl
from fineqcomp.evaluation import completion_nll, write_predictions
from fineqcomp.modeling import CausalExampleDataset, ModelSession
from fineqcomp.training import train_adapter


def group_key(row):
    return str(row.metadata.get('source_problem') or row.prompt)


def order_groups(rows, seed):
    return sorted(rows, key=lambda r: (
        hashlib.sha256(f'{seed}:{r.example_id}'.encode()).hexdigest(), r.example_id))


def partition(rows, seed):
    """Keep repeated prompts and source-problem augmentations in one half."""
    result = [[], []]
    for row in order_groups(rows, seed):
        side = int(hashlib.sha256(f'split:{seed}:{group_key(row)}'.encode()).hexdigest()[:8], 16) % 2
        result[side].append(row)
    return result


def measured_budget(grid, base, reference, retention, split='selection', scaled=False):
    """An achieved file size; never interpolate it or discard censoring."""
    gain = base[split] - reference[split]
    if gain <= 0:
        return {'status': 'no_positive_reference_gain', 'file_bits': None}
    allowed = [r for r in grid if scaled or r['scale'] == 1.]
    passing = [r for r in allowed if base[split] - r[split] >= retention * gain]
    if not passing:
        return {'status': 'above_grid', 'file_bits': None,
                'largest_tested_bits': max(r['file_bits'] for r in allowed)}
    best = min(passing, key=lambda r: (r['file_bits'], r['key']))
    cheaper = [r for r in allowed if r['file_bits'] < best['file_bits']]
    return {'status': 'measured', 'file_bits': best['file_bits'], 'key': best['key'],
            'below_smallest_tested': not cheaper,
            'verification_retention': ((base['verification'] - best['verification']) /
                (base['verification'] - reference['verification'])
                if base['verification'] > reference['verification'] else None)}


def prepare(config, out):
    owners = ('information_budget_search.py', 'training.py', 'modeling.py', 'adapters.py', 'codec.py', 'evaluation.py', 'data.py', 'config.py')
    lock = dict(config=config, source_sha256={name: hashlib.sha256(
        Path(__file__).with_name(name).read_bytes()).hexdigest() for name in owners})
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        old = read_json(out / 'lock.json')
        if old is not None and old != lock:
            raise ValueError('inputs changed; choose a fresh output directory')
        if old is None:
            write_json(out / 'lock.json', lock)
    return config['cells']


def score(session, run, rows, path, config):
    old = read_json(path)
    if old is not None:
        return old
    with torch.inference_mode():
        result = completion_nll(session.model, session.tokenizer, rows, run.model,
                                config['max_length'], config['batch_size'])
    records = [dict(example_id=r.example_id, **m) for r, m in zip(rows, result, strict=True)]
    if any(r['tokens'] <= 0 or not math.isfinite(r['sum_nll']) for r in records):
        raise ValueError('empty or invalid scored response')
    write_predictions(path.with_suffix('.jsonl'), records)
    summary = {'bits': sum(r['sum_nll'] for r in records) / math.log(2),
        'tokens': sum(r['tokens'] for r in records), 'examples': len(records),
        'per_example_bits': [r['sum_nll'] / math.log(2) for r in records],
        'response_tokens': [r['tokens'] for r in records]}
    write_json(path, summary)
    return summary


def sweep(session, run, tensors, data, root, config, scales):
    grid = []
    reference = {}
    apply_adapter_tensors(session.model, tensors)
    for split, rows in data.items():
        reference[split] = score(session, run, rows, root / f'raw_{split}.json', config)['bits']
    for rank in config['ranks']:
        reduced = truncate_lora_rank(tensors, rank)
        for bits in config['bits']:
            for scale in scales:
                key = f'r{rank}_b{bits}_s{scale:g}'
                path = root / 'codecs' / f'{key}.fqcb'
                path.parent.mkdir(parents=True, exist_ok=True)
                scaled = {n: t * scale if '.lora_B.' in n else t for n, t in reduced.items()}
                storage = encode_tensor_map(scaled, path, bits)
                if storage['file_bits'] != path.stat().st_size * 8:
                    raise ValueError('serialized length mismatch')
                _, decoded = decode_adapter_tensor_map(path)
                apply_adapter_tensors(session.model, pad_lora_rank(decoded, run.adapter.rank))
                row = dict(key=key, rank=rank, precision=bits, scale=scale, file_bits=storage['file_bits'])
                for split, rows in data.items():
                    row[split] = score(session, run, rows, root / 'scores' / f'{key}_{split}.json', config)['bits']
                grid.append(row)
    write_json(root / 'frontier.json', dict(reference=reference, grid=grid))
    apply_adapter_tensors(session.model, tensors)
    return reference, grid


def run_cell(config, source, out, index, smoke=False):
    cell = prepare(config, out)[index]
    run = RunSpec.from_dict(cell['run'])
    root = out / ('smoke' if smoke else 'cells') / run.run_id
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError('cell already claimed')
        if (root / 'complete.json').exists():
            return
        cfg = dict(config)
        if smoke:
            cfg.update(replica_rows=16, steps=[1, 2], block_ends=[4, 8],
                       feature_rows=4, rate_rows=8, block_updates=1, ranks=[1, 16], bits=[2, 16])
        data_root = source / 'prepared/natural' / str(run.dataset_key) / f'seed{run.seed}'
        original_train = read_jsonl(data_root / 'train.jsonl')
        calibration = read_jsonl(data_root / 'calibration.jsonl')
        test = read_jsonl(data_root / 'test.jsonl')
        bounded = lambda rows: [r for r in rows if len((r.prompt + r.response).encode()) <= cfg['max_row_bytes']]
        calibration_groups = {group_key(r) for r in calibration}
        eval_groups = calibration_groups | {group_key(r) for r in test}
        train = bounded([r for r in original_train if group_key(r) not in eval_groups])
        sides = partition(train, cfg['seed'])
        if min(map(len, sides)) < cfg['replica_rows']:
            raise ValueError('too few distinct-group rows for independent replicas')
        replicas = [s[:cfg['replica_rows']] for s in sides]
        selected = order_groups(bounded(calibration), cfg['seed'])
        verified = order_groups(bounded([r for r in test if group_key(r) not in calibration_groups]), cfg['seed'])
        selection = selected[:cfg['rate_rows'] // 2]
        verification = verified[:cfg['rate_rows'] // 2]
        if min(len(selection), len(verification)) < cfg['rate_rows'] // 2:
            raise ValueError('rate panels are undersized')
        feature = selection[:cfg['feature_rows']]
        rate_data = {'selection': selection, 'verification': verification}
        target_path = source / 'runs' / run.run_id / 'raw_channel.pt'
        contract = dict(stage=cell['stage'],
            training_original=len(original_train), training_eligible=len(train),
            overlap_groups=len({group_key(r) for r in original_train} & eval_groups),
            replica_ids=[[r.example_id for r in s] for s in replicas],
            replica_groups=[len({group_key(r) for r in s}) for s in replicas],
            selection_ids=[r.example_id for r in selection], verification_ids=[r.example_id for r in verification],
            verification_groups=[group_key(r) for r in verification],
            files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data_root.glob('*.jsonl')},
            target_sha256=hashlib.sha256(target_path.read_bytes()).hexdigest(),
            byte_filter=cfg['max_row_bytes'], target='select on calibration; verify on test; entire bounded response')
        prior = read_json(root / 'data_contract.json')
        if prior is not None and prior != contract:
            raise ValueError('data or finished adapter changed; refusing cached scores')
        write_json(root / 'data_contract.json', contract)
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            base_scores = {k: score(session, run, v, root / f'base_{k}.json', cfg) for k, v in rate_data.items()}
            base = {k: v['bits'] for k, v in base_scores.items()}
            base_feature = score(session, run, feature, root / 'base_feature.json', cfg)
            base_halves = [score(session, run, s[:cfg['feature_rows']], root / f'base_half{i}.json', cfg)
                           for i, s in enumerate(replicas)]
            frozen = []
            for count in (1, 4):
                demos = []
                for row in replicas[0]:
                    if len(('\n\n'.join(demos + [row.prompt + row.response])).encode()) <= cfg['context_bytes']:
                        demos.append(row.prompt + row.response)
                    if len(demos) == count:
                        break
                if not demos:
                    frozen.append(dict(requested=count, actual=0, gain_bits=None))
                    continue
                prefix = '\n\n'.join(demos) + '\n\n'
                rows = [replace(r, prompt=prefix + r.prompt) for r in feature]
                measured = score(session, run, rows, root / f'context_{count}.json', cfg)
                if measured['response_tokens'] != base_feature['response_tokens']:
                    raise ValueError('context changed the evaluated response span')
                frozen.append(dict(requested=count, actual=len(demos), bytes=len(prefix.encode()),
                                   gain_bits=base_feature['bits'] - measured['bits']))
            # These are prefixes of a declared training schedule, not short
            # runs whose cosine schedule reaches zero at the probe cutoff.
            full_data = CausalExampleDataset(session.tokenizer, original_train, run.model,
                run.training.max_length, run.training.label_span)
            full_horizon = math.ceil(len(full_data) / run.training.effective_batch_size) * run.training.epochs
            if run.training.max_updates is not None:
                full_horizon = min(full_horizon, run.training.max_updates)
            write_json(root / 'training_exposure.json', dict(original_rows=len(original_train),
                usable_rows=len(full_data), full_horizon=full_horizon,
                rows_at_length_limit=sum(len(r['input_ids']) == run.training.max_length for r in full_data),
                supervised_tokens=sum(int((r['labels'][1:] != -100).sum()) for r in full_data),
                boundary='Rows at the length limit may be truncated; this count does not prove truncation.'))
            del full_data
            traces = []
            for replica, taught in enumerate(replicas):
                if replica:
                    session.unload(); session.attach(run.adapter, run.seed + replica)
                # One nonempty epoch always supplies at least one update.
                # A large epoch ceiling plus an exact cap survives row drops.
                training = replace(run.training, epochs=max(cfg['steps']),
                    effective_batch_size=16, micro_batch_size=4, max_updates=max(cfg['steps']),
                    eval_every_updates=None, restore_best=False)
                def checkpoint(step, optimizer):
                    if step not in cfg['steps']:
                        return
                    tensors = adapter_tensors(session.model, run.adapter.method)
                    path = root / f'replica{replica}/step{step}'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(tensors, path.with_suffix('.pt'))
                    measured = score(session, run, feature, path / 'feature.json', cfg)
                    other = score(session, run, replicas[1-replica][:cfg['feature_rows']], path / 'other_half.json', cfg)
                    reference, grid = sweep(session, run, tensors, rate_data, path, cfg, [1.])
                    traces.append(dict(replica=replica, step=step, feature=measured, other_half=other,
                                       reference=reference, grid=grid))
                    session.model.train()
                measured = train_adapter(session.model, session.tokenizer, taught, selection, run.model,
                    training, run.seed + replica, root / f'replica{replica}/training.jsonl',
                    on_update=checkpoint, schedule_updates=max(full_horizon, max(cfg['steps'])))
                if measured['optimizer_updates'] != max(cfg['steps']) or {
                    t['step'] for t in traces if t['replica'] == replica} != set(cfg['steps']):
                    raise ValueError('probe stopped before a requested checkpoint')
                write_json(root / f'replica{replica}/training.json', measured)
            # A real block-prequential code: score each unseen block before
            # training on it. Optimizer resets are part of this public learner.
            session.unload(); session.attach(run.adapter, run.seed)
            stream = replicas[0]
            previous = 0
            online = []
            for end in cfg['block_ends']:
                block = stream[previous:end]
                before = score(session, run, block, root / f'online/block{end}_before.json', cfg)
                training = replace(run.training, epochs=cfg['block_updates'],
                    effective_batch_size=16, micro_batch_size=4, max_updates=cfg['block_updates'],
                    restore_best=False, eval_every_updates=None, warmup_ratio=0.)
                trained = train_adapter(session.model, session.tokenizer, block, selection, run.model, training,
                                        run.seed, root / f'online/block{end}_training.jsonl')
                if trained['optimizer_updates'] != cfg['block_updates']:
                    raise ValueError('online learner stopped before its requested update count')
                after = score(session, run, feature, root / f'online/block{end}_heldout.json', cfg)
                online.append(dict(start=previous, end=end, before=before, heldout=after))
                previous = end
            features = dict(frozen_context=frozen, base=base_feature, base_halves=base_halves, traces=traces, online=online,
                base_rate=base,
                full_horizon=full_horizon, training_rows=len(original_train),
                online_excess_bits=sum(r['before']['bits'] for r in online) - previous * online[-1]['heldout']['bits'] / len(feature),
                boundary='Early learner-relative codes. Online excess uses independent heldout loss at the probe endpoint, not the finished target. No abstract source entropy claim.')
            write_json(root / 'features.json', features)
            # Endpoint information is loaded only after all early features are
            # saved, and never enters their computation.
            session.unload(); session.attach(run.adapter, run.seed)
            if hashlib.sha256(target_path.read_bytes()).hexdigest() != contract['target_sha256']:
                raise ValueError('finished adapter changed during the probe')
            final = torch.load(target_path, map_location='cpu', weights_only=True)
            reference, grid = sweep(session, run, final, rate_data, root / 'target', cfg, cfg['scales'])
            budgets = [dict(retention=rho, scaled=scaled,
                **measured_budget(grid, base, reference, rho, scaled=scaled))
                for rho in cfg['retentions'] for scaled in (False, True)]
            write_json(root / 'targets.json', dict(base=base, reference=reference, budgets=budgets))
            write_json(root / 'complete.json', dict(complete=True, smoke=smoke, stage=cell['stage']))
        finally:
            session.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'run'], required=True)
    parser.add_argument('--cell', type=int, default=0)
    args = parser.parse_args(); config = read_json(args.config)
    if args.phase == 'prepare':
        print(json.dumps({'cells': len(prepare(config, args.out))}))
    else:
        run_cell(config, args.source_root, args.out, args.cell, args.phase == 'smoke')


if __name__ == '__main__':
    main()
