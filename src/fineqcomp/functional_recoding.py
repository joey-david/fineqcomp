"""Compare task training, teacher imitation, and actual functional code sizes."""

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
from fineqcomp.codec import decode_adapter_tensor_map, encode_tensor_map, pad_lora_rank, truncate_lora_rank
from fineqcomp.config import RunSpec
from fineqcomp.data import read_jsonl
from fineqcomp.evaluation import evaluate_natural, _normalize_answer_text, write_predictions
from fineqcomp.modeling import ModelSession
from fineqcomp.training import train_adapter


def gauge(tensors, scale):
    """Exact power-of-two LoRA gauge; the represented BA update is unchanged."""
    return {n: t * scale if '.lora_A.' in n else t / scale for n, t in tensors.items()}


def prepare(config, source, out):
    manifest = [json.loads(s) for s in (source / 'prepared/f14-manifest.jsonl').read_text().splitlines()]
    teachers = {(r['model']['key'], r['adapter']['rank']): r for r in manifest if r['seed'] == 11}
    cells = []
    for model in config['models']:
        for seed in config['seeds']:
            for objective, teacher_rank, rank in [('oracle', 16, 1), ('oracle', 16, 4), ('oracle', 16, 16), ('teacher', 16, 1), ('teacher', 64, 1)]:
                cells.append({'teacher': teachers[model, teacher_rank], 'seed': seed, 'objective': objective,
                              'rank': rank, 'teacher_rank': teacher_rank,
                              'slug': f'{model}/seed{seed}/{objective}_t{teacher_rank}_r{rank}'})
    lock = {'config': config, 'cells': cells, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.prepare.lock').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        previous = read_json(out / 'lock.json')
        if previous is not None and previous != lock:
            raise ValueError('design changed after preparation')
        if previous is None:
            write_json(out / 'lock.json', lock)
    return lock


def score(session, spec, rows, path, batch, teacher=None):
    previous = read_json(path)
    if previous is not None:
        predictions = [json.loads(line) for line in path.with_suffix('.jsonl').read_text().splitlines()]
        return previous, predictions
    metrics, predictions = evaluate_natural(session.model, session.tokenizer, rows, spec.model,
        'xbrl_tags', batch_size=batch, max_new_tokens=64, generation_seed=11)
    if teacher is not None:
        if [r['example_id'] for r in predictions] != [r['example_id'] for r in teacher]:
            raise ValueError('teacher and student predictions are not aligned')
        for a, b in zip(predictions, teacher, strict=True):
            a['teacher_agreement'] = _normalize_answer_text(a['response']) == _normalize_answer_text(b['response'])
        metrics['teacher_agreement'] = sum(r['teacher_agreement'] for r in predictions) / len(predictions)
    write_predictions(path.with_suffix('.jsonl'), predictions)
    write_json(path, metrics)
    return metrics, predictions


def run(config, source, out, index, smoke=False):
    cell = prepare(config, source, out)['cells'][index]
    spec = RunSpec.from_dict(cell['teacher'])
    root = out / ('smoke' if smoke else 'cells') / cell['slug']
    with claim_run(root) as acquired:
        if not acquired:
            raise RuntimeError('cell already running')
        data_root = source / 'prepared/natural/xbrl_tags/seed11'
        train = read_jsonl(data_root / 'train.jsonl')[:config['training_rows']]
        development = read_jsonl(data_root / 'calibration.jsonl')[:config['development_rows']]
        test = read_jsonl(data_root / 'test.jsonl')[:config['test_rows']]
        if {r.prompt for r in train} & {r.prompt for r in development + test}:
            raise ValueError('training and evaluation prompts overlap')
        if smoke:
            train, development, test = train[:16], development[:4], test[:4]
        session = ModelSession.load(spec.model)
        try:
            session.attach(spec.adapter, 11)
            raw = torch.load(source / 'runs' / spec.run_id / 'raw_channel.pt', map_location='cpu', weights_only=True)
            apply_adapter_tensors(session.model, raw)
            teacher = {}
            for split, rows in [('development', development), ('test', test)]:
                teacher[split] = score(session, spec, rows, root / f'teacher_{split}.json', config['batch_size'])[1]
            if cell['objective'] == 'teacher':
                _, targets = score(session, spec, train, root / 'teacher_training.json', config['batch_size'])
                # Hard behavioral distillation: train only on teacher completions
                # from training inputs. Evaluation questions never supply targets.
                train = [replace(r, response=' ' + _normalize_answer_text(p['response'])) for r, p in zip(train, targets, strict=True)]
                if any(not r.response.strip() for r in train):
                    raise ValueError('teacher emitted empty training targets')
            session.unload()
            adapter = replace(spec.adapter, key=f'matched_r{cell["rank"]}', rank=cell['rank'], alpha=2 * cell['rank'])
            session.attach(adapter, cell['seed'])
            checkpoint = root / 'student.pt'
            if checkpoint.exists():
                apply_adapter_tensors(session.model, torch.load(checkpoint, map_location='cpu', weights_only=True))
            else:
                updates = 2 if smoke else config['updates']
                training = replace(spec.training, epochs=(updates * 16 + len(train) - 1) // len(train),
                    max_updates=updates, effective_batch_size=16, micro_batch_size=4,
                    restore_best=False, eval_every_updates=None, warmup_ratio=.03)
                measured = train_adapter(session.model, session.tokenizer, train, development, spec.model,
                    training, cell['seed'], root / 'training.jsonl')
                if measured['optimizer_updates'] != updates:
                    raise ValueError('training did not reach the fixed update count')
                torch.save(adapter_tensors(session.model, 'full_lora'), checkpoint)
                write_json(root / 'training.json', measured)
            tensors = adapter_tensors(session.model, 'full_lora')
            raw_predictions = {}
            for split, rows in [('development', development), ('test', test)]:
                raw_predictions[split] = score(session, spec, rows, root / f'raw_{split}.json', config['batch_size'], teacher[split])[1]
            # Gauge invariance is an executable control, not an assumed property
            # of aggregate task accuracy or of nominal quantization precision.
            for scale in (.25, 4.):
                apply_adapter_tensors(session.model, gauge(tensors, scale))
                score(session, spec, development, root / f'gauge_{scale:g}.json', config['batch_size'], raw_predictions['development'])
            grid = [(r, b) for r in (1, 2, 4, 8, 16) if r <= cell['rank'] for b in (1, 2, 4, 8, 16)]
            if smoke:
                grid = [(1, 2), (cell['rank'], 16)]
            candidates = []
            for rank, bits in grid:
                key = f'r{rank}_b{bits}'
                path = root / 'codecs' / f'{key}.fqcb'; path.parent.mkdir(exist_ok=True)
                storage = encode_tensor_map(truncate_lora_rank(tensors, rank), path, bits)
                assert storage['file_bits'] == path.stat().st_size * 8
                _, decoded = decode_adapter_tensor_map(path)
                apply_adapter_tensors(session.model, pad_lora_rank(decoded, cell['rank']))
                metrics, _ = score(session, spec, development, root / 'codecs' / f'{key}_development.json', config['batch_size'], teacher['development'])
                candidates.append({'key': key, 'file_bits': storage['file_bits'], **metrics})
            # Different code sets answer different questions. Both thresholds
            # are absolute and fixed, rather than relative to each student's gain.
            selections = {}
            for metric, target in [('exact_match', config['task_target']), ('teacher_agreement', config['fidelity_target'])]:
                passing = [r for r in candidates if r[metric] >= target]
                selections[metric] = min(passing, key=lambda r: r['file_bits']) if passing else None
            write_json(root / 'selection.json', selections)
            write_json(root / 'candidates.json', candidates)
            for key in sorted({r['key'] for r in selections.values() if r is not None}):
                _, decoded = decode_adapter_tensor_map(root / 'codecs' / f'{key}.fqcb')
                apply_adapter_tensors(session.model, pad_lora_rank(decoded, cell['rank']))
                score(session, spec, test, root / 'codecs' / f'{key}_test.json', config['batch_size'], teacher['test'])
            write_json(root / 'complete.json', {'complete': True, 'smoke': smoke, 'objective': cell['objective']})
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
        print(json.dumps({'cells': len(prepare(config, a.source_root, a.out)['cells'])}))
    elif a.phase in ('smoke', 'run'):
        run(config, a.source_root, a.out, a.cell, a.phase == 'smoke')
    else:
        records = [{'path': str(p.relative_to(a.out)), 'selection': read_json(p)} for p in sorted((a.out / 'cells').rglob('selection.json'))]
        write_json(a.out / 'summary.json', records)


if __name__ == '__main__':
    main()
