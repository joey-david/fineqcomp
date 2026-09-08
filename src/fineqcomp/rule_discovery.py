"""Separate learning a sharing rule from continued use of its input cue."""

from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import hashlib
import json
from pathlib import Path

import torch
import yaml

from fineqcomp.adapters import adapter_tensors, apply_adapter_tensors
from fineqcomp.artifacts import claim_run, read_json, write_json
from fineqcomp.campaign import _adapter, _model
from fineqcomp.codec import decode_adapter_tensor_map, encode_tensor_map, pad_lora_rank, truncate_lora_rank
from fineqcomp.config import TrainingSpec, load_campaign
from fineqcomp.evaluation import evaluate_constrained_labels, write_predictions
from fineqcomp.information_scaling import Condition, build_dataset, epochs_for, _prompt
from fineqcomp.modeling import ModelSession
from fineqcomp.training import train_adapter


def cue_rows(info, prototypes, seed, mode):
    data = build_dataset(info, Condition(name='shared', prototype_count=prototypes), seed)
    rows = []
    for row in data.examples:
        family, item = row.metadata['family'], row.metadata['item']
        actual = family % prototypes
        shown = None if mode == 'hidden' else actual if mode == 'correct' else (actual + 1) % prototypes
        cued = data.label_indices[(shown if shown is not None else actual) * info['items_per_family'] + item]
        rows.append(replace(row, prompt=_prompt(family, item, shown, info['prompt_layout']),
            metadata=dict(row.metadata, cued_label=cued, cue_disagrees=cued != row.metadata['label_index'])))
    return rows


def prepare(config, out):
    cells = []
    for p in config['prototypes']:
        for seed in config['seeds']:
            for arm in ('cached_hidden', 'cached_revealed', 'hidden_hidden', 'cue_cue', 'cue_hidden', 'hidden_cue'):
                cells.append({'prototypes': p, 'seed': seed, 'arm': arm, 'slug': f'p{p}/seed{seed}/{arm}'})
    lock = {'config': config, 'cells': cells, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        old = read_json(out / 'lock.json')
        if old is not None and old != lock:
            raise ValueError('locked design changed')
        if old is None:
            write_json(out / 'lock.json', lock)
    return lock


def measure(session, info, rows, path):
    old = read_json(path)
    if old is not None:
        return old
    metrics, predictions = evaluate_constrained_labels(session.model, session.tokenizer, rows,
        session.spec, info['labels'], info['evaluation_batch_size'])
    discordant = []
    for row, pred in zip(rows, predictions, strict=True):
        pred['cue_disagrees'] = row.metadata['cue_disagrees']
        pred['cued_label'] = row.metadata['cued_label']
        pred['follows_cue'] = pred['prediction'] == pred['cued_label']
        if pred['cue_disagrees']:
            discordant.append(pred['follows_cue'])
    metrics.update(discordant_n=len(discordant), follows_wrong_cue=sum(discordant) / len(discordant) if discordant else None)
    write_predictions(path.with_suffix('.jsonl'), predictions)
    write_json(path, metrics)
    return metrics


def run(config, source, out, index, smoke):
    cell = prepare(config, out)['cells'][index]
    root = out / ('smoke' if smoke else 'cells') / cell['slug']
    info = yaml.safe_load((source / 'configs/design_2026_09_07/f10_storage_versus_discovery.yaml').read_text())
    info['codebook_dir'] = str(source / 'codebooks')
    campaign = load_campaign(source / 'configs/compressibility.yaml')
    spec, adapter = _model(info['model'], campaign), _adapter('all_linear_r16', campaign)
    data = {mode: cue_rows(info, cell['prototypes'], cell['seed'], mode) for mode in ('hidden', 'correct', 'wrong')}
    # Split unseen mappings by item parity within every unseen family, so both
    # halves contain all prototypes. The partition is fixed before inference.
    partitions = {'taught': list(range(512)), 'development': [i for i in range(512, 768) if i % 2 == 0],
                  'test': [i for i in range(512, 768) if i % 2 == 1]}
    if smoke:
        partitions = {k: v[:4] for k, v in partitions.items()}
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError('cell already running')
        write_json(root / 'data_contract.json', {'partitions': partitions, 'prototypes': cell['prototypes'],
            'source_bits': 4 * 16 * cell['prototypes'], 'labels': info['labels'],
            'source_config': info, 'score_scope': 'next-token distribution conditional on the 16 labels'})
        session = ModelSession.load(spec)
        try:
            session.attach(adapter, cell['seed'])
            if cell['arm'].startswith('cached_'):
                revealed = cell['arm'] == 'cached_revealed'
                study = 'revealed' if revealed else 'hidden'
                condition = f"p{cell['prototypes']}" + ('_cue' if revealed else '')
                raw_path = source / f"runs_information_scaling/f10/full/{study}/{condition}/all_linear_r16/seed{cell['seed']}/n512/raw_channel.pt"
                apply_adapter_tensors(session.model, torch.load(raw_path, weights_only=True, map_location='cpu'))
                phases = []
            else:
                phases = cell['arm'].split('_')
            for phase, mode in enumerate(phases, 1):
                checkpoint = root / f'phase{phase}.pt'
                if checkpoint.exists():
                    apply_adapter_tensors(session.model, torch.load(checkpoint, weights_only=True, map_location='cpu'))
                else:
                    taught = data['correct' if mode == 'cue' else 'hidden'][:512]
                    updates = 2 if smoke else config['phase_updates']
                    if smoke:
                        taught = taught[:32]
                    training = replace(TrainingSpec(**info['training']), epochs=epochs_for(updates, len(taught), 16),
                        max_updates=updates, eval_every_updates=None, restore_best=False)
                    # Every arm resets the optimizer at the same phase boundary.
                    metrics = train_adapter(session.model, session.tokenizer, taught, taught, spec, training,
                        cell['seed'] * 1000 + phase, root / f'phase{phase}_training.jsonl')
                    if metrics['optimizer_updates'] != updates:
                        raise ValueError('phase update budget mismatch')
                    torch.save(adapter_tensors(session.model, adapter.method), checkpoint)
                    write_json(root / f'phase{phase}_training.json', metrics)
                for cue, examples in data.items():
                    for split, indices in partitions.items():
                        measure(session, info, [examples[i] for i in indices], root / f'phase{phase}/{cue}_{split}.json')
            tensors = adapter_tensors(session.model, adapter.method)
            for cue, examples in data.items():
                for split, indices in partitions.items():
                    measure(session, info, [examples[i] for i in indices], root / f'raw/{cue}_{split}.json')
            if not phases:
                write_json(root / 'complete.json', {'complete': True, 'smoke': smoke})
                return
            candidates = []
            grid = [(r, b) for r in (1, 2, 4, 8, 16) for b in (1, 2, 4, 16)]
            if smoke:
                grid = [(1, 2), (16, 16)]
            for rank, bits in grid:
                key = f'r{rank}_b{bits}'
                path = root / f'codecs/{key}.fqcb'; path.parent.mkdir(exist_ok=True)
                storage = encode_tensor_map(truncate_lora_rank(tensors, rank), path, bits)
                assert storage['file_bits'] == path.stat().st_size * 8
                _, decoded = decode_adapter_tensor_map(path)
                apply_adapter_tensors(session.model, pad_lora_rank(decoded, 16))
                for cue in ('hidden', 'correct'):
                    metrics = measure(session, info, [data[cue][i] for i in partitions['development']], root / f'codecs/{key}_{cue}_development.json')
                    candidates.append({'key': key, 'cue': cue, 'file_bits': storage['file_bits'], **metrics})
            selections = {}
            for cue in ('hidden', 'correct'):
                passing = [r for r in candidates if r['cue'] == cue and r['accuracy'] >= config['accuracy_target']]
                selections[cue] = min(passing, key=lambda r: r['file_bits']) if passing else None
            write_json(root / 'candidates.json', candidates)
            write_json(root / 'selection.json', selections)
            for cue, chosen in selections.items():
                if chosen is None:
                    continue
                _, decoded = decode_adapter_tensor_map(root / f"codecs/{chosen['key']}.fqcb")
                apply_adapter_tensors(session.model, pad_lora_rank(decoded, 16))
                for evaluation_cue in ('hidden', 'correct', 'wrong'):
                    measure(session, info, [data[evaluation_cue][i] for i in partitions['test']], root / f"selected_{cue}/{evaluation_cue}_test.json")
            write_json(root / 'complete.json', {'complete': True, 'smoke': smoke})
        finally:
            session.unload()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--phase', choices=['prepare', 'smoke', 'run', 'reduce'], required=True)
    p.add_argument('--cell', type=int, default=0)
    a = p.parse_args(); config = yaml.safe_load(a.config.read_text())
    if a.phase == 'prepare':
        print(json.dumps({'cells': len(prepare(config, a.out)['cells'])}))
    elif a.phase in ('smoke', 'run'):
        run(config, a.source_root, a.out, a.cell, a.phase == 'smoke')
    else:
        write_json(a.out / 'summary.json', [{'path': str(p.relative_to(a.out)), 'selection': read_json(p)} for p in sorted((a.out / 'cells').rglob('selection.json'))])


if __name__ == '__main__':
    main()
