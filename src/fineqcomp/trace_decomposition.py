"""Split a paired aligned/permuted LoRA update into a mean and a half-difference.

The completed CoT compression pilot showed that a small enough code keeps a
model's final-answer accuracy while losing the information specific to matching
each trace to its problem: on Qwen the rank-two binary state scored 58 aligned
against 60 permuted. That leaves two candidate contents inside one adapter and
no way to tell them apart from the aligned file alone.

This module separates them arithmetically rather than by training anything new.
For each adapted projection, the aligned and permuted dense updates give

    mean       M = (D_aligned + D_permuted) / 2
    difference E = (D_aligned - D_permuted) / 2

with M + E = D_aligned and M - E = D_permuted exactly. Both parts are formed as
LoRA factors of rank 2r by concatenation, never as dense matrices, so the
existing balanced-SVD truncation applies unchanged. Compressing M and E to a
rank separately and evaluating each part alone, plus both recombinations, asks
which part survives which rank. It is a decomposition of two trained files, not
a claim that the parts are independently learned or that either is a mechanism.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path

import torch
import yaml

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.codec import (
    _lora_pairs,
    decode_adapter_tensor_map,
    encode_tensor_map,
    pad_lora_rank,
    truncate_lora_rank,
)
from fineqcomp.config import RunSpec
from fineqcomp.correction_conditioning import (
    examples,
    generation_rows,
    score_rows,
    summary,
)
from fineqcomp.evaluation import generate_response_records, write_predictions
from fineqcomp.modeling import ModelSession


ALIGNED_KEY = 'cot_math'
PERMUTED_KEY = 'cot_math_permuted'


def _paired_names(aligned, permuted):
    if set(aligned) != set(permuted):
        only = sorted(set(aligned) ^ set(permuted))
        raise ValueError(f'aligned and permuted adapters differ in tensors: {only[:3]}')
    for name, tensor in aligned.items():
        if tuple(tensor.shape) != tuple(permuted[name].shape):
            raise ValueError(f'shape mismatch for {name}')
    return [pair[:2] for pair in _lora_pairs(aligned)]


def decompose(aligned, permuted):
    """Exact rank-2r LoRA factors for the mean and half-difference updates.

    A LoRA pair contributes B @ A to its projection, so stacking the two
    adapters' A rows and their B columns writes any weighted sum of the two
    updates as one wider pair. The halving sits on B, which leaves A shared by
    both parts; this is the factorization, not a compression step, and it is
    exact up to floating point.
    """
    names = _paired_names(aligned, permuted)
    mean, difference = {}, {}
    for a_name, b_name in names:
        stacked_a = torch.cat([aligned[a_name].float(), permuted[a_name].float()], dim=0)
        left, right = aligned[b_name].float(), permuted[b_name].float()
        mean[a_name] = stacked_a
        difference[a_name] = stacked_a.clone()
        mean[b_name] = torch.cat([left, right], dim=1) / 2
        difference[b_name] = torch.cat([left, -right], dim=1) / 2
    return mean, difference


def recombine(mean, difference, sign):
    """Add or subtract two LoRA-factored updates, again by concatenation."""
    if sign not in (1, -1):
        raise ValueError('sign must be +1 or -1')
    names = _paired_names(mean, difference)
    combined = {}
    for a_name, b_name in names:
        combined[a_name] = torch.cat([mean[a_name].float(), difference[a_name].float()], dim=0)
        combined[b_name] = torch.cat(
            [mean[b_name].float(), sign * difference[b_name].float()], dim=1
        )
    return combined


def lora_rank(tensors):
    ranks = {int(tensors[a_name].shape[0]) for a_name, _ in _paired_names(tensors, tensors)}
    if len(ranks) != 1:
        raise ValueError(f'inconsistent LoRA ranks: {sorted(ranks)}')
    return ranks.pop()


def part_ranks(config, container_rank):
    """Ranks a single part may take: it must fit the trained container."""
    ranks = sorted({int(r) for r in config['ranks']})
    if ranks[0] < 1:
        raise ValueError('ranks must be at least one')
    over = [r for r in ranks if r > container_rank]
    if over:
        raise ValueError(f'part ranks exceed the rank-{container_rank} container: {over}')
    return ranks


def recombination_ranks(config, container_rank):
    """Ranks both recombinations may take: two parts share one container.

    A recombination of two rank-k parts has rank 2k, so the trained rank-16
    container admits recombinations only up to k = 8. Ranks above that are
    dropped here rather than silently truncated, because truncating the sum
    would stop it reconstructing the aligned update.
    """
    return [r for r in part_ranks(config, container_rank) if 2 * r <= container_rank]


def states(mean, difference, config, container_rank):
    """Name every evaluated adapter state and its LoRA factors, before coding."""
    built = {}
    parts = {}
    for rank in part_ranks(config, container_rank):
        parts[rank] = (truncate_lora_rank(mean, rank), truncate_lora_rank(difference, rank))
        built[f'mean_r{rank}'] = parts[rank][0]
        built[f'difference_r{rank}'] = parts[rank][1]
    for rank in recombination_ranks(config, container_rank):
        mean_part, difference_part = parts[rank]
        built[f'aligned_recombined_r{rank}'] = recombine(mean_part, difference_part, 1)
        built[f'permuted_recombined_r{rank}'] = recombine(mean_part, difference_part, -1)
    return built


def expected_states(config, container_rank):
    names = {'base'}
    for rank in part_ranks(config, container_rank):
        names |= {f'mean_r{rank}', f'difference_r{rank}'}
    for rank in recombination_ranks(config, container_rank):
        names |= {f'aligned_recombined_r{rank}', f'permuted_recombined_r{rank}'}
    return names


def pairs(manifest, config):
    """One decomposition cell per model and training seed, aligned with permuted."""
    wanted = [r for r in manifest
              if r['model']['key'] in config['models'] and r['seed'] in config['seeds']]
    grouped = defaultdict(dict)
    for row in wanted:
        if row['dataset_key'] in (ALIGNED_KEY, PERMUTED_KEY):
            grouped[(row['model']['key'], row['seed'])][row['dataset_key']] = row
    cells = []
    for key in sorted(grouped):
        cell = grouped[key]
        if set(cell) != {ALIGNED_KEY, PERMUTED_KEY}:
            raise ValueError(f'{key} has no matched aligned/permuted pair')
        cells.append({'model': key[0], 'seed': key[1],
                      'aligned': cell[ALIGNED_KEY], 'permuted': cell[PERMUTED_KEY]})
    if len(cells) != len(config['models']) * len(config['seeds']):
        raise ValueError('every requested model and seed needs a matched pair')
    return cells


def prepare(config, source, out):
    manifest = [json.loads(line) for line in
                (source / '.cache/prepared/f03-manifest.jsonl').read_text().splitlines()]
    cells = pairs(manifest, config)
    rows = examples(config, source)
    lock = {'config': config, 'cells': cells, 'examples': [r.to_dict() for r in rows],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'symbolic_sha256': hashlib.sha256(
                (source / '.cache/prepared/gsm-symbolic-source.jsonl').read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        previous = read_json(out / 'lock.json')
        if previous is not None and previous != lock:
            raise ValueError('locked inputs changed; use a fresh output folder')
        if previous is None:
            write_json(out / 'lock.json', lock)
    return lock


def load_channel(source, run_id):
    return torch.load(source / 'runs' / run_id / 'raw_channel.pt',
                      map_location='cpu', weights_only=True)


def run(config, source, out, index, smoke=False):
    lock = prepare(config, source, out)
    cell = lock['cells'][index]
    aligned_run = RunSpec.from_dict(cell['aligned'])
    permuted_run = RunSpec.from_dict(cell['permuted'])
    if aligned_run.adapter != permuted_run.adapter:
        raise ValueError('the pair must share one adapter placement and rank')
    root = out / ('smoke' if smoke else 'cells') / f"{cell['model']}__s{cell['seed']}"
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError('cell already running')
        rows = examples(config, source)
        if smoke:
            rows = rows[:4]
        mean, difference = decompose(load_channel(source, aligned_run.run_id),
                                     load_channel(source, permuted_run.run_id))
        container = aligned_run.adapter.rank
        if lora_rank(mean) != 2 * container:
            raise ValueError('the decomposition did not keep both adapters in full')
        built = states(mean, difference, config, container)
        if smoke:
            built = {key: built[key] for key in config.get('smoke_keys', [])}
        session = ModelSession.load(aligned_run.model)
        try:
            session.attach(aligned_run.adapter, aligned_run.seed)
            transformations = {} if smoke else {'base': adapter_tensors(
                session.model, aligned_run.adapter.method)}
            for key, tensors in built.items():
                # Serialize and read back, so what is evaluated is a file with a
                # measured size rather than an in-memory tensor.
                path = root / 'codecs' / f'{key}.fqcb'
                path.parent.mkdir(parents=True, exist_ok=True)
                storage = encode_tensor_map(tensors, path, config.get('bits', 16), blend=0.0)
                if storage['file_bits'] != path.stat().st_size * 8:
                    raise ValueError('serialized length mismatch')
                _, decoded = decode_adapter_tensor_map(path)
                transformations[key] = pad_lora_rank(decoded, container)
                write_json(path.with_suffix('.json'), storage)
            for key, tensors in transformations.items():
                modes = config.get('generation_modes', ['cot'])
                targets = {mode: (root / key if mode == 'cot' else root / key / mode)
                           for mode in modes}
                if all((p / 'summary.json').exists() for p in targets.values()):
                    continue
                apply_adapter_tensors(session.model, tensors)
                for mode, mode_target in targets.items():
                    if (mode_target / 'summary.json').exists():
                        continue
                    records = generate_response_records(
                        session.model, session.tokenizer, generation_rows(rows, mode),
                        aligned_run.model, config['batch_size'], config['max_new_tokens'],
                        aligned_run.seed,
                        **({'stop_strings': ['\n']} if mode == 'direct' else {}))
                    if mode == 'direct':
                        records = [dict(r, response='#### ' + r['response']) for r in records]
                    scored = score_rows(rows, records)
                    write_predictions(mode_target / 'predictions.jsonl', scored)
                    write_json(mode_target / 'summary.json', summary(scored))
            write_json(root / 'complete.json',
                       {'complete': True, 'rows': len(rows), 'smoke': smoke,
                        'states': sorted(transformations), 'container_rank': container})
        finally:
            session.unload()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare', 'smoke', 'run', 'reduce'], required=True)
    parser.add_argument('--cell', type=int, default=0)
    args = parser.parse_args()
    config = (json.loads(args.config.read_text()) if args.config.suffix == '.json'
              else yaml.safe_load(args.config.read_text()))
    if args.phase == 'prepare':
        print(json.dumps({'cells': len(prepare(config, args.source_root, args.out)['cells'])}))
    elif args.phase in ('smoke', 'run'):
        run(config, args.source_root, args.out, args.cell, args.phase == 'smoke')
    else:
        rows = [{'path': str(p.relative_to(args.out)), 'summary': read_json(p)}
                for p in sorted((args.out / 'cells').rglob('summary.json'))]
        write_json(args.out / 'summary.json', rows)


if __name__ == '__main__':
    main()
