"""Paired input interventions and prefix/suffix adapter crossovers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import fcntl
import json
from pathlib import Path
import re

import torch
import yaml

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.behavioral_trajectory import intervention
from fineqcomp.codec import decode_adapter_tensor_map, encode_tensor_map, pad_lora_rank, truncate_lora_rank
from fineqcomp.config import RunSpec
from fineqcomp.data import _convert_gsm8k
from fineqcomp.evaluation import completion_nll, generate_response_records, _normalize_number, write_predictions
from fineqcomp.modeling import ModelSession


def examples(config, source):
    rows = []
    for line in (source / 'prepared/gsm-symbolic-source.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['instance'] not in config['instances']:
            continue
        example = _convert_gsm8k([row], 'symbolic')[0]
        template = str(row['original_id'])
        # The split depends only on template identity, not either answer.
        fold = int(hashlib.sha256(template.encode()).hexdigest()[:8], 16) % 5
        rows.append(replace(example,
            example_id=f"symbolic-{template}-{row['instance']}",
            metadata={'template': template, 'instance': row['instance'],
                      'split': 'development' if fold == 0 else 'test'}))
    if len(rows) != 100 * len(config['instances']):
        raise ValueError('incomplete symbolic template panel')
    return sorted(rows, key=lambda x: (x.metadata['template'], x.metadata['instance']))


def candidates(raw, rank, seed, config=None):
    """Actual files plus a public zero-update receiver; no nominal rate target."""
    yield 'raw', raw, 16, 0.0
    config = config or {}
    for scale in config.get('scales', (.25, .5, .75)):
        yield f'scale_{scale:g}', {n: t * scale if '.lora_B.' in n else t for n, t in raw.items()}, 16, 0.0
    yield 'mask_half', raw, 0, .5
    for target in config.get('svd_ranks', (1, 2, 4, 8)):
        if target <= rank:
            yield f'svd_r{target}', truncate_lora_rank(raw, target), 16, 0.0
    if config.get('svd_binary', True):
        yield 'svd_r2_binary', truncate_lora_rank(raw, 2), 1, 0.0
    for draw in range(config.get('random_draws', 3)):
        reduced = intervention(raw, 'random', 2, seed + 1009 * draw)
        yield f'random_r2_{draw}', truncate_lora_rank(reduced, 2), 16, 0.0


def strict_answer(text):
    matches = re.findall(r'\\boxed\{([-+\d.,]+)\}|(?:[Aa]nswer\s*(?:is)?\s*[:=]?|####)\s*([-+]?\d[\d,]*(?:\.\d+)?)', text)
    return _normalize_number(next(v for v in matches[-1] if v)) if matches else None


def score_rows(rows, records, prefix=None):
    result = []
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        text = (prefix[index]['response'] if prefix else '') + record['response']
        gold = _normalize_number(row.response)
        strict = strict_answer(text)
        result.append({'example_id': row.example_id, **row.metadata, **record,
                       'response': text, 'expected': gold,
                       'correct': _normalize_number(text) == gold,
                       'strict_prediction': strict, 'strict_correct': strict == gold,
                       'prefix_terminated': prefix[index]['terminated_with_eos'] if prefix else False})
    return result


def summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['split']].append(row)
    result = {}
    for split, group in groups.items():
        prefixes = Counter(tuple(re.findall(r'\w+', r['response'].lower())[:24]) for r in group)
        pairs = defaultdict(list)
        for row in group:
            pairs[row['template']].append(row)
        result[split] = {'n': len(group), 'accuracy': sum(r['correct'] for r in group) / len(group),
            'strict_accuracy': sum(r['strict_correct'] for r in group) / len(group),
            'strict_parse_rate': sum(r['strict_prediction'] is not None for r in group) / len(group),
            'cap_rate': sum(r['hit_generation_limit'] for r in group) / len(group),
            'largest_prefix_share': max(prefixes.values()) / len(group),
            'all_variants_correct': sum(all(r['strict_correct'] for r in p) for p in pairs.values()) / len(pairs)}
    return result


def prepare(config, source, out):
    manifest = [json.loads(line) for line in (source / 'prepared/f03-manifest.jsonl').read_text().splitlines()]
    cells = [r for r in manifest if r['model']['key'] in config['models'] and r['seed'] in config['seeds']]
    cells.sort(key=lambda r: (r['model']['key'], r['seed'], r['dataset_key']))
    if len(cells) != len(config['models']) * len(config['seeds']) * 2:
        raise ValueError('both aligned and permuted adapters are required')
    rows = examples(config, source)
    lock = {'config': config, 'cells': cells, 'examples': [r.to_dict() for r in rows],
            'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'symbolic_sha256': hashlib.sha256((source / 'prepared/gsm-symbolic-source.jsonl').read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        previous = read_json(out / 'lock.json')
        if previous is not None and previous != lock:
            raise ValueError('locked inputs changed; use a fresh output folder')
        if previous is None:
            write_json(out / 'lock.json', lock)
    return lock


def run(config, source, out, index, smoke=False):
    lock = prepare(config, source, out)
    run = RunSpec.from_dict(lock['cells'][index])
    root = out / ('smoke' if smoke else 'cells') / run.run_id
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError('cell already running')
        rows = examples(config, source)
        if smoke:
            rows = rows[:4]
        session = ModelSession.load(run.model)
        try:
            session.attach(run.adapter, run.seed)
            zero = adapter_tensors(session.model, run.adapter.method)
            raw = torch.load(source / 'runs' / run.run_id / 'raw_channel.pt', map_location='cpu', weights_only=True)
            options = list(candidates(raw, run.adapter.rank, run.seed, config))
            if smoke:
                options = [options[0], next(x for x in options if x[0] == 'mask_half')]
            transformations = {'base': zero}
            for key, tensors, bits, blend in options:
                path = root / 'codecs' / f'{key}.fqcb'
                path.parent.mkdir(parents=True, exist_ok=True)
                storage = encode_tensor_map(tensors, path, bits, blend=blend)
                if storage['file_bits'] != path.stat().st_size * 8:
                    raise ValueError('serialized length mismatch')
                _, decoded = decode_adapter_tensor_map(path)
                transformations[key] = pad_lora_rank(decoded, run.adapter.rank)
                write_json(path.with_suffix('.json'), storage)
            for key, tensors in transformations.items():
                target = root / key
                if (target / 'summary.json').exists():
                    continue
                apply_adapter_tensors(session.model, tensors)
                records = generate_response_records(session.model, session.tokenizer, rows, run.model,
                    config['batch_size'], config['max_new_tokens'], run.seed)
                scored = score_rows(rows, records)
                write_predictions(target / 'predictions.jsonl', scored)
                # Same answer completion under both paired questions. This is a
                # conditional sensitivity measurement, not a scalar MI estimate.
                by_template = defaultdict(list)
                for row in rows:
                    by_template[row.metadata['template']].append(row)
                answer_rows, answer_keys = [], []
                for row in rows:
                    group = by_template[row.metadata['template']]
                    partner = group[(group.index(row) + 1) % len(group)]
                    for kind, answer_target in [('own', row), ('paired', partner)]:
                        answer_rows.append(replace(row, response=' The answer is: ' + _normalize_number(answer_target.response)))
                        answer_keys.append({'example_id': row.example_id, 'answer_source_id': answer_target.example_id,
                            'kind': kind, 'different_answers': _normalize_number(row.response) != _normalize_number(partner.response)})
                with torch.inference_mode():
                    nll = completion_nll(session.model, session.tokenizer, answer_rows, run.model, 1536, config['batch_size'])
                write_predictions(target / 'answer_nll.jsonl', [dict(k, **s) for k, s in zip(answer_keys, nll, strict=True)])
                write_json(target / 'summary.json', summary(scored))
            # F3: use the identical generated prefix with both suffix adapters.
            # Replaying text clears the cache in every arm; raw/raw and mask/mask
            # control for that restart rather than confounding it with a switch.
            pairs = [(length, kind) for length in config.get('prefix_lengths', [config['prefix_tokens']])
                     for kind in config.get('crossover_kinds', ['raw', 'mask_half'])]
            if smoke:
                pairs = [(config.get('prefix_lengths', [config['prefix_tokens']])[0], kind) for kind in ('raw', 'mask_half')]
            for prefix_length, prefix_kind in pairs:
                apply_adapter_tensors(session.model, transformations[prefix_kind])
                crossover_root = root / 'crossovers' / f't{prefix_length}'
                prefix_path = crossover_root / f'{prefix_kind}_prefix.json'
                prefixes = read_json(prefix_path)
                if prefixes is None:
                    prefixes = generate_response_records(session.model, session.tokenizer, rows, run.model,
                        config['batch_size'], prefix_length, run.seed)
                    write_json(prefix_path, prefixes)
                for suffix_kind in (('raw', 'mask_half') if smoke else config.get('crossover_kinds', ['raw', 'mask_half'])):
                    target = crossover_root / f'{prefix_kind}_to_{suffix_kind}'
                    if (target / 'summary.json').exists():
                        continue
                    apply_adapter_tensors(session.model, transformations[suffix_kind])
                    live = [i for i, p in enumerate(prefixes) if not p['terminated_with_eos']]
                    continuation_rows = [replace(rows[i], prompt=rows[i].prompt + prefixes[i]['response']) for i in live]
                    suffixes = generate_response_records(session.model, session.tokenizer, continuation_rows, run.model,
                        config['batch_size'], config['max_new_tokens'] - prefix_length, run.seed) if live else []
                    records = [dict(response='', hit_generation_limit=False, completion_tokens=0) for _ in rows]
                    for i, record in zip(live, suffixes, strict=True):
                        records[i] = record
                    scored = score_rows(rows, records, prefixes)
                    write_predictions(target / 'predictions.jsonl', scored)
                    write_json(target / 'summary.json', summary(scored))
            write_json(root / 'complete.json', {'complete': True, 'rows': len(rows), 'smoke': smoke})
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
    config = yaml.safe_load(args.config.read_text())
    if args.phase == 'prepare':
        print(json.dumps({'cells': len(prepare(config, args.source_root, args.out)['cells'])}))
    elif args.phase in ('smoke', 'run'):
        run(config, args.source_root, args.out, args.cell, args.phase == 'smoke')
    else:
        rows = [{'path': str(p.relative_to(args.out)), 'summary': read_json(p)} for p in sorted((args.out / 'cells').rglob('summary.json'))]
        write_json(args.out / 'summary.json', rows)


if __name__ == '__main__':
    main()
