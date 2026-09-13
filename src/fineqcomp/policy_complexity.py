"""Collect data-learning curves and actual adapter rate curves independently."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path

import torch

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.codec import (decode_adapter_tensor_map, encode_tensor_map,
                            pad_lora_rank, truncate_lora_rank)
from fineqcomp.config import AdapterSpec, ModelSpec, TrainingSpec
from fineqcomp.evaluation import completion_nll
from fineqcomp.modeling import CausalExampleDataset, ModelSession
from fineqcomp.policy_data import make_policy_data
from fineqcomp.training import train_adapter

SPLITS = ('selection_recall', 'verification_recall', 'selection_unseen', 'verification_unseen')


def load_config(path):
    path = Path(path)
    config = read_json(path)
    if config is None:
        raise ValueError(f'missing configuration: {path}')
    if 'extends' in config:
        parent = load_config(path.parent / config['extends'])
        parent.update({k: v for k, v in config.items() if k != 'extends'})
        config = parent
    return config


def contract(config):
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in ('policy_complexity.py', 'policy_data.py', 'training.py',
                            'modeling.py', 'evaluation.py', 'codec.py', 'adapters.py')}
    cells = []
    for model, seed, policy, n, rank, repeat in itertools.product(
            config['models'], config['seeds'], config['conditions'], config['n_values'],
            config['training_ranks'], config.get('repeat_factors', [1])):
        cells.append(dict(model=model, seed=seed, policy=policy, n=n, rank=rank, repeat=repeat))
    return dict(config=config, sources=sources, cells=cells)


def prepare(config, out):
    locked = contract(config)
    previous = read_json(out / 'lock.json')
    if previous is not None and previous != locked:
        raise ValueError('config or source differs from the output lock; use a new output root')
    write_json(out / 'lock.json', locked)
    return locked


def cell_data(cell, config):
    return make_policy_data(cell['policy'], cell['n'], seed=cell['seed'],
                            repeat_factor=cell['repeat'], **config['data'])


def validate_tokens(session, rows, maximum):
    data = CausalExampleDataset(session.tokenizer, rows, session.spec, maximum)
    if len(data) != len(rows) or any(p + r > maximum for p, r in data.token_lengths):
        raise ValueError('task contract forbids dropped or truncated responses')
    return data


def score(session, rows, config):
    validate_tokens(session, rows, config['training']['max_length'])
    records = completion_nll(session.model, session.tokenizer, rows, session.spec,
                             config['training']['max_length'], config['batch_size'])
    if len(records) != len(rows) or any(r['tokens'] <= 0 or not math.isfinite(r['sum_nll']) for r in records):
        raise ValueError('invalid complete-response score')
    result = dict(nll=sum(r['sum_nll'] for r in records) / sum(r['tokens'] for r in records),
                  rows=[dict(example_id=e.example_id, **e.metadata, **r)
                        for e, r in zip(rows, records, strict=True)])
    # Diagnostic excludes common response wording, so format learning cannot
    # masquerade as learning the policy. The primary score remains full text.
    content = [replace(e, prompt=e.prompt + ' The correct action is ',
                       response=e.response.split(' The correct action is ', 1)[1]) for e in rows]
    validate_tokens(session, content, config['training']['max_length'])
    action = completion_nll(session.model, session.tokenizer, content, session.spec,
                           config['training']['max_length'], config['batch_size'])
    result['action_nll'] = sum(r['sum_nll'] for r in action) / sum(r['tokens'] for r in action)
    for row, a in zip(result['rows'], action, strict=True):
        row['action_sum_nll'], row['action_tokens'] = a['sum_nll'], a['tokens']
    return result


def evaluate(session, data, config):
    return {split: score(session, data[split], config) for split in SPLITS}


def run_base(config, out, model_index):
    model = config['models'][model_index]
    root = out / 'base' / model['key']
    with claim_run(root) as claimed:
        if not claimed or (root / 'complete.json').exists():
            return
        session = ModelSession.load(ModelSpec(**model))
        results = []
        for seed, policy in itertools.product(config['seeds'], config['conditions']):
            cell = dict(seed=seed, policy=policy, n=max(config['n_values']), repeat=1)
            data = cell_data(cell, config)
            results.append(dict(seed=seed, policy=policy, **score(session, data['matching'], config)))
        spreads = [max(r['nll'] for r in results if r['seed'] == s) -
                   min(r['nll'] for r in results if r['seed'] == s) for s in config['seeds']]
        write_json(root / 'complete.json', dict(results=results, nll_spreads=spreads,
                   matched=max(spreads) <= config['matching_tolerance_nats']))


def run_cell(config, out, cell, index):
    root = out / 'cells' / f'{index:04d}'
    with claim_run(root) as claimed:
        if not claimed or (root / 'complete.json').exists():
            return
        diagnostic = read_json(out / 'base' / cell['model']['key'] / 'complete.json')
        if not diagnostic or not diagnostic['matched']:
            raise ValueError('base mismatch gate not passed; inspect matching-only diagnostic')
        data = cell_data(cell, config)
        write_json(root / 'data.json', {k: [r.to_dict() for r in v] for k, v in data.items()})
        session = ModelSession.load(ModelSpec(**cell['model']))
        baseline = evaluate(session, data, config)
        write_json(root / 'base.json', baseline)
        adapter = AdapterSpec(key=f'r{cell["rank"]}', method='full_lora', rank=cell['rank'],
                              alpha=2 * cell['rank'], dropout=0., last_n_layers=None,
                              target_modules=tuple(config['target_modules']))
        session.attach(adapter, cell['seed'])
        trainable, total = session.model.get_nb_trainable_parameters()
        base_parameters = total - trainable
        tokenized = validate_tokens(session, data['train'], config['training']['max_length'])
        settings = config['training']
        if len(tokenized) % settings['effective_batch_size']:
            raise ValueError('training rows must fill whole effective batches')
        horizon = max(config['checkpoints'])
        per_epoch = len(tokenized) // settings['effective_batch_size']
        spec = TrainingSpec(**settings, epochs=math.ceil(horizon / per_epoch), max_updates=horizon,
                            restore_best=False, eval_every_updates=None)
        history = []
        def checkpoint(step, optimizer):
            if step not in config['checkpoints']:
                return
            scores = evaluate(session, data, config)
            history.append(dict(step=step, scores=scores))
            write_json(root / f'checkpoint_{step}.json', history[-1])
            session.model.train()
        trained = train_adapter(session.model, session.tokenizer, data['train'], data['selection_recall'],
                                session.spec, spec, cell['seed'], root / 'training.jsonl',
                                on_update=checkpoint)
        if trained['optimizer_updates'] != horizon or [r['step'] for r in history] != config['checkpoints']:
            raise ValueError('missing requested training updates')
        final = adapter_tensors(session.model, 'full_lora')
        from safetensors.torch import save_file
        save_file({k: v.contiguous() for k, v in final.items()}, str(root / 'adapter.safetensors'))
        grid = []
        if cell['n'] == max(config['n_values']) or config.get('compress_all_n', False):
            for rank in config['codec']['ranks']:
                if rank > cell['rank']:
                    continue
                reduced = truncate_lora_rank(final, rank)
                for bits in config['codec']['bits']:
                    path = root / 'codecs' / f'r{rank}_b{bits}.fqcb'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    storage = encode_tensor_map(reduced, path, bits)
                    if storage['file_bits'] != 8 * path.stat().st_size:
                        raise ValueError('codec length mismatch')
                    _, decoded = decode_adapter_tensor_map(path)
                    apply_adapter_tensors(session.model, pad_lora_rank(decoded, cell['rank']))
                    item = dict(key=path.stem, rank=rank, bits=bits, file_bits=storage['file_bits'],
                                bits_per_base_parameter=storage['file_bits'] / base_parameters,
                                scores=evaluate(session, data, config))
                    write_json(path.with_suffix('.json'), item)
                    grid.append(item)
        write_json(root / 'complete.json', dict(cell=cell, training=trained, base_parameters=base_parameters,
            checkpoints=history, grid=grid, base=baseline,
            contract='Full response plus EOS NLL in nats; prompt masked; action suffix separate.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=('prepare', 'base', 'run', 'reduce'), required=True)
    parser.add_argument('--cell', type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    locked = prepare(config, args.out)
    if args.phase == 'base':
        run_base(config, args.out, args.cell)
    elif args.phase == 'run':
        run_cell(config, args.out, locked['cells'][args.cell], args.cell)
    elif args.phase == 'reduce':
        from fineqcomp.policy_complexity_analysis import reduce
        write_json(args.out / 'summary.json', reduce(args.out))
    else:
        print(json.dumps(dict(cells=len(locked['cells']), models=len(config['models']))))


if __name__ == '__main__':
    main()
